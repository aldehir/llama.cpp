#include "sampling.h"
#include "unicode.h"

#include "llama.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// apply the sampler to a fresh candidate array and return which tokens survived
static std::vector<bool> apply_mask(struct llama_sampler * smpl, size_t n_vocab) {
    std::vector<llama_token_data> cur;
    cur.reserve(n_vocab);
    for (size_t i = 0; i < n_vocab; i++) {
        cur.emplace_back(llama_token_data{(llama_token) i, 0.0f, 0.0f});
    }
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };

    llama_sampler_apply(smpl, &cur_p);

    std::vector<bool> valid(n_vocab);
    for (size_t i = 0; i < n_vocab; i++) {
        valid[cur[i].id] = std::isfinite(cur[i].logit);
    }
    return valid;
}

static void test_masking() {
    const std::vector<std::string> pieces = {
        /*  0 */ "hello",             // ascii
        /*  1 */ "",                  // control token
        /*  2 */ "\xe4",              // lead byte of a 3-byte sequence
        /*  3 */ "\xb8",              // continuation byte
        /*  4 */ "\xad",              // continuation byte
        /*  5 */ "\xe4\xb8\xad",      // complete 3-byte sequence
        /*  6 */ "\xff",              // invalid byte
        /*  7 */ "a\xc3",             // ascii + lead byte of a 2-byte sequence
        /*  8 */ "\xa9",              // continuation byte
        /*  9 */ "\xb8\xad",          // two continuation bytes
        /* 10 */ "\xb8m",             // continuation byte + ascii
        /* 11 */ "\xe4\xb8",          // incomplete 3-byte sequence
        /* 12 */ "\xf0\x9f\x98",      // incomplete 4-byte sequence
        /* 13 */ "\x80z",             // stray continuation byte + ascii
    };

    auto * smpl = common_sampler_init_utf8_validate(pieces);

    // state 0: tokens starting with a continuation byte or containing invalid bytes are masked
    {
        const auto valid = apply_mask(smpl, pieces.size());
        assert( valid[0]);
        assert( valid[1]);
        assert( valid[2]);
        assert(!valid[3]);
        assert(!valid[4]);
        assert( valid[5]);
        assert(!valid[6]);
        assert( valid[7]);
        assert(!valid[8]);
        assert(!valid[9]);
        assert(!valid[10]);
        assert( valid[11]);
        assert( valid[12]);
        assert(!valid[13]);
    }

    // mid-sequence: after a 3-byte lead, only continuation bytes and empty pieces are valid
    llama_sampler_accept(smpl, 2); // "\xe4", expecting 2 continuation bytes
    {
        const auto valid = apply_mask(smpl, pieces.size());
        assert(!valid[0]);
        assert( valid[1]); // control tokens stay valid, EOG remains reachable
        assert(!valid[2]);
        assert( valid[3]);
        assert( valid[4]);
        assert(!valid[5]);
        assert(!valid[6]);
        assert(!valid[7]);
        assert( valid[8]);
        assert( valid[9]);  // completes the sequence
        assert(!valid[10]); // ascii after only one of two continuation bytes
        assert(!valid[11]);
        assert(!valid[12]);
        assert(!valid[13]);
    }

    // completing the sequence returns to state 0
    llama_sampler_accept(smpl, 9); // "\xb8\xad"
    {
        const auto valid = apply_mask(smpl, pieces.size());
        assert(valid[0] && !valid[3]);
    }

    // token ending in a 2-byte lead leaves 1 continuation byte expected
    llama_sampler_accept(smpl, 7); // "a\xc3"
    {
        const auto valid = apply_mask(smpl, pieces.size());
        assert(!valid[0]);
        assert( valid[4]);
        assert(!valid[9]); // second continuation byte would be stray
        assert( valid[10]);
    }

    // accept recovers from tokens that do not fit the state (e.g. prompt tokens)
    llama_sampler_accept(smpl, 6); // "\xff" is invalid from any state
    {
        const auto valid = apply_mask(smpl, pieces.size());
        assert(valid[0] && !valid[3]);
    }

    // reset returns to state 0
    llama_sampler_accept(smpl, 2);
    llama_sampler_reset(smpl);
    {
        const auto valid = apply_mask(smpl, pieces.size());
        assert(valid[0] && !valid[3]);
    }

    // clone preserves state
    llama_sampler_accept(smpl, 2);
    auto * clone = llama_sampler_clone(smpl);
    {
        const auto valid = apply_mask(clone, pieces.size());
        assert(!valid[0] && valid[3]);
    }

    llama_sampler_free(clone);
    llama_sampler_free(smpl);

    printf("test_masking: OK\n");
}

// reference check: a piece is valid from state 0 if it parses as a sequence of
// complete codepoints, optionally ending in an incomplete one
static bool piece_valid_ref(const std::string & piece) {
    size_t pos = 0;
    while (pos < piece.size()) {
        const auto result = common_parse_utf8_codepoint(piece, pos);
        if (result.status == utf8_parse_result::INVALID) {
            return false;
        }
        if (result.status == utf8_parse_result::INCOMPLETE) {
            // the parser defers continuation byte checks until the sequence is complete;
            // any non-continuation byte here is guaranteed to become INVALID, so check eagerly
            for (size_t i = pos + 1; i < piece.size(); i++) {
                if (((unsigned char) piece[i] & 0xc0) != 0x80) {
                    return false;
                }
            }
            return true;
        }
        pos += result.bytes_consumed;
    }
    return true;
}

// exhaustively compare the sampler's state-0 masking against common_parse_utf8_codepoint,
// which is what the PEG parser uses to validate model output
static void test_parser_equivalence() {
    std::vector<std::string> pieces;

    for (int b0 = 0; b0 < 256; b0++) {
        pieces.push_back(std::string(1, (char) b0));
    }
    for (int b0 = 0; b0 < 256; b0++) {
        for (int b1 = 0; b1 < 256; b1++) {
            std::string piece(1, (char) b0);
            piece += (char) b1;
            pieces.push_back(piece);
        }
    }
    // 3- and 4-byte sequences around structural boundaries
    const int interesting[] = { 0x00, 0x41, 0x7f, 0x80, 0x9f, 0xbf, 0xc0, 0xc2, 0xdf, 0xe0, 0xed, 0xef, 0xf0, 0xf4, 0xf7, 0xf8, 0xff };
    for (const int b0 : interesting) {
        for (const int b1 : interesting) {
            for (const int b2 : interesting) {
                std::string piece = { (char) b0, (char) b1, (char) b2 };
                pieces.push_back(piece);
                for (const int b3 : interesting) {
                    pieces.push_back(piece + (char) b3);
                }
            }
        }
    }

    auto * smpl = common_sampler_init_utf8_validate(pieces);
    const auto valid = apply_mask(smpl, pieces.size());

    for (size_t i = 0; i < pieces.size(); i++) {
        const bool expected = piece_valid_ref(pieces[i]);
        if (valid[i] != expected) {
            printf("mismatch for piece:");
            for (const char c : pieces[i]) {
                printf(" %02x", (unsigned char) c);
            }
            printf(" (sampler: %d, parser: %d)\n", (int) valid[i], (int) expected);
            assert(false);
        }
    }

    llama_sampler_free(smpl);

    printf("test_parser_equivalence: OK (%zu pieces)\n", pieces.size());
}

int main() {
    test_masking();
    test_parser_equivalence();

    printf("all tests passed\n");
    return 0;
}
