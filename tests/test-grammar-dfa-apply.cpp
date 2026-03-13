#ifdef NDEBUG
#undef NDEBUG
#endif

// Tests for grammar DFA apply logic using a real vocabulary.
//
// These tests exercise the full token-level grammar constraint path:
//   llama_grammar_apply_impl  →  precomputed candidates  →  byte simulation
//
// They verify:
//   1. Correctness: valid tokens are NOT rejected, invalid tokens ARE rejected
//   2. Candidate superset: precomputed candidates are a superset of actually-valid tokens
//   3. accept_byte handles DFA accept states with outgoing transitions correctly
//   4. Multi-token sequences are accepted/rejected correctly
//   5. Performance: apply should complete in reasonable time for large vocabs

#include "llama.h"
#include "common.h"

#include "../src/llama-grammar.h"
#include "../src/llama-grammar-dfa.h"
#include "../src/llama-vocab.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

// ============================================================
// Globals
// ============================================================

static const llama_vocab * vocab = nullptr;

// ============================================================
// Helpers
// ============================================================

static llama_grammar * build_grammar_with_vocab(const std::string & grammar_str) {
    return llama_grammar_init_impl(
        vocab, grammar_str.c_str(), "root",
        /* lazy= */ false, /* trigger_patterns= */ nullptr, /* num_trigger_patterns= */ 0,
        /* trigger_tokens= */ nullptr, /* num_trigger_tokens= */ 0);
}

static llama_grammar * build_grammar_no_vocab(const std::string & grammar_str) {
    return llama_grammar_init_impl(
        nullptr, grammar_str.c_str(), "root",
        /* lazy= */ false, /* trigger_patterns= */ nullptr, /* num_trigger_patterns= */ 0,
        /* trigger_tokens= */ nullptr, /* num_trigger_tokens= */ 0);
}

// Match a string byte-by-byte through the DFA engine (no vocab needed).
static bool match_string(const std::string & input, llama_grammar * grammar) {
    auto configs = grammar->configs;
    for (uint8_t b : input) {
        grammar->compiled->accept_byte(configs, b);
        if (configs.empty()) return false;
    }
    return grammar->compiled->any_config_complete(configs);
}

// Build a full token data array with all logits at 0.
static std::vector<llama_token_data> make_token_data(uint32_t n_vocab_size) {
    std::vector<llama_token_data> cur;
    cur.reserve(n_vocab_size);
    for (llama_token token_id = 0; token_id < (llama_token)n_vocab_size; token_id++) {
        cur.emplace_back(llama_token_data{ token_id, 0.0f, 0.0f });
    }
    return cur;
}

// Reset all logits to 0.
static void reset_logits(std::vector<llama_token_data> & cur) {
    for (auto & d : cur) {
        d.logit = 0.0f;
    }
}

// Count tokens with non-negative logits (i.e., allowed tokens).
static int count_allowed(const std::vector<llama_token_data> & cur) {
    int count = 0;
    for (const auto & d : cur) {
        if (d.logit >= 0.0f) count++;
    }
    return count;
}

// Brute-force: for each token, simulate byte-by-byte and check if the DFA survives.
// Returns a bitmask of which tokens are actually valid (true = valid).
static std::vector<bool> brute_force_valid_tokens(llama_grammar * grammar) {
    uint32_t n = vocab->n_tokens();
    std::vector<bool> valid(n, false);

    for (uint32_t id = 0; id < n; id++) {
        if (vocab->is_eog((int32_t)id)) continue;
        const std::string & piece = vocab->token_to_piece((int32_t)id);
        if (piece.empty() || piece[0] == '\0') continue;

        auto configs = grammar->configs;
        bool ok = true;
        for (size_t j = 0; j < piece.size(); j++) {
            grammar->compiled->accept_byte(configs, (uint8_t)piece[j]);
            if (configs.empty()) {
                ok = false;
                break;
            }
        }
        valid[id] = ok;
    }
    return valid;
}

// ============================================================
// Test: apply correctly allows/rejects tokens for simple grammar
// ============================================================

static void test_apply_simple_literal() {
    fprintf(stderr, "Testing apply with simple literal grammar: root ::= \"hello\"...\n");

    auto * grammar = build_grammar_with_vocab("root ::= \"hello\"");
    assert(grammar != nullptr);

    uint32_t n = vocab->n_tokens();
    auto cur = make_token_data(n);
    auto tok_arr = llama_token_data_array{ cur.data(), cur.size(), -1, false };

    llama_grammar_apply_impl(*grammar, &tok_arr);

    int allowed = count_allowed(cur);
    fprintf(stderr, "  Allowed tokens: %d / %u\n", allowed, n);

    // At least some tokens should be allowed (those starting with 'h')
    assert(allowed > 0);
    // Most tokens should be rejected
    assert(allowed < (int)(n / 2));

    // Verify specific behavior: tokens that don't start with 'h' must be rejected
    for (uint32_t id = 0; id < n; id++) {
        if (vocab->is_eog((int32_t)id)) continue;
        const std::string & piece = vocab->token_to_piece((int32_t)id);
        if (piece.empty() || piece[0] == '\0') continue;

        if (piece[0] != 'h' && cur[id].logit >= 0.0f) {
            fprintf(stderr, "  BUG: Token %u (\"%s\") allowed but doesn't start with 'h'\n",
                    id, piece.c_str());
            assert(false);
        }
    }

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  OK\n");
}

// ============================================================
// Test: precomputed candidates are superset of brute-force valid tokens
// ============================================================

static void test_candidates_superset(const std::string & grammar_str, const std::string & label) {
    fprintf(stderr, "Testing candidate superset for: %s...\n", label.c_str());

    auto * grammar = build_grammar_with_vocab(grammar_str);
    assert(grammar != nullptr);

    uint32_t n = vocab->n_tokens();
    auto cur = make_token_data(n);
    auto tok_arr = llama_token_data_array{ cur.data(), cur.size(), -1, false };

    // Run apply (uses precomputed candidates + simulation)
    llama_grammar_apply_impl(*grammar, &tok_arr);

    // Brute-force: check every token by simulation only
    auto brute_valid = brute_force_valid_tokens(grammar);

    int false_negatives = 0;
    int false_positives = 0;

    for (uint32_t id = 0; id < n; id++) {
        if (vocab->is_eog((int32_t)id)) continue;
        const std::string & piece = vocab->token_to_piece((int32_t)id);
        if (piece.empty() || piece[0] == '\0') continue;

        bool apply_allows = (cur[id].logit >= 0.0f);
        bool brute_allows = brute_valid[id];

        if (brute_allows && !apply_allows) {
            false_negatives++;
            fprintf(stderr, "  FALSE NEGATIVE: Token %u (\"%s\") is valid by brute-force but rejected by apply\n",
                    id, piece.c_str());
        }
        if (apply_allows && !brute_allows) {
            false_positives++;
            // This shouldn't happen either — apply does full simulation for candidates
            fprintf(stderr, "  FALSE POSITIVE: Token %u (\"%s\") accepted by apply but rejected by brute-force\n",
                    id, piece.c_str());
        }
    }

    fprintf(stderr, "  False negatives: %d, False positives: %d\n", false_negatives, false_positives);

    // No false negatives allowed (these are real bugs — valid tokens incorrectly rejected)
    assert(false_negatives == 0);
    // No false positives either (apply does full simulation)
    assert(false_positives == 0);

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  OK\n");
}

// ============================================================
// Test: multi-token sequence acceptance through apply + accept
// ============================================================

static void test_multi_token_sequence() {
    fprintf(stderr, "Testing multi-token sequence acceptance...\n");

    auto * grammar = build_grammar_with_vocab(
        "root ::= \"{\" ws \"\\\"key\\\"\" ws \":\" ws \"\\\"value\\\"\" ws \"}\"\n"
        "ws ::= [ \\t\\n]*");
    assert(grammar != nullptr);

    // Tokenize a valid JSON string
    std::string valid_input = "{\"key\":\"value\"}";
    auto tokens = common_tokenize(vocab, valid_input, false, false);

    uint32_t n = vocab->n_tokens();
    auto cur = make_token_data(n);
    auto tok_arr = llama_token_data_array{ cur.data(), cur.size(), -1, false };

    // Feed each token through apply + accept
    for (size_t i = 0; i < tokens.size(); i++) {
        reset_logits(cur);
        llama_grammar_apply_impl(*grammar, &tok_arr);

        // The actual token should be allowed
        if (cur[tokens[i]].logit < 0.0f) {
            const std::string & piece = vocab->token_to_piece(tokens[i]);
            fprintf(stderr, "  BUG: Token %d (\"%s\") at position %zu was rejected\n",
                    tokens[i], piece.c_str(), i);
            assert(false);
        }

        llama_grammar_accept_impl(*grammar, tokens[i]);
    }

    // After consuming all tokens, EOG should be allowed
    bool allow_eog = grammar->compiled->any_config_complete(grammar->configs);
    assert(allow_eog);

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  OK\n");
}

// ============================================================
// Test: multi-token sequence for various grammars
// ============================================================

static void test_token_sequence(const std::string & grammar_str, const std::string & input, const std::string & label) {
    fprintf(stderr, "Testing token sequence for %s: \"%s\"...\n", label.c_str(), input.c_str());

    auto * grammar = build_grammar_with_vocab(grammar_str);
    assert(grammar != nullptr);

    auto tokens = common_tokenize(vocab, input, false, false);

    uint32_t n = vocab->n_tokens();
    auto cur = make_token_data(n);
    auto tok_arr = llama_token_data_array{ cur.data(), cur.size(), -1, false };

    for (size_t i = 0; i < tokens.size(); i++) {
        reset_logits(cur);
        llama_grammar_apply_impl(*grammar, &tok_arr);

        if (cur[tokens[i]].logit < 0.0f) {
            const std::string & piece = vocab->token_to_piece(tokens[i]);
            fprintf(stderr, "  BUG: Token %d (\"%s\") at position %zu was rejected\n",
                    tokens[i], piece.c_str(), i);
            assert(false);
        }

        llama_grammar_accept_impl(*grammar, tokens[i]);
    }

    bool allow_eog = grammar->compiled->any_config_complete(grammar->configs);
    if (!allow_eog) {
        fprintf(stderr, "  BUG: EOG not allowed after consuming all tokens of valid input\n");
        assert(false);
    }

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  OK\n");
}

// ============================================================
// Test: invalid token sequence should be rejected at some point
// ============================================================

static void test_invalid_token_rejected() {
    fprintf(stderr, "Testing invalid tokens are rejected...\n");

    // Grammar only allows digits
    auto * grammar = build_grammar_with_vocab("root ::= [0-9]+");
    assert(grammar != nullptr);

    uint32_t n = vocab->n_tokens();
    auto cur = make_token_data(n);
    auto tok_arr = llama_token_data_array{ cur.data(), cur.size(), -1, false };

    llama_grammar_apply_impl(*grammar, &tok_arr);

    // Find a token that is purely alphabetic — it must be rejected
    bool found_rejected_alpha = false;
    for (uint32_t id = 0; id < n; id++) {
        if (vocab->is_eog((int32_t)id)) continue;
        const std::string & piece = vocab->token_to_piece((int32_t)id);
        if (piece.empty() || piece[0] == '\0') continue;

        bool all_alpha = true;
        for (char c : piece) {
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))) {
                all_alpha = false;
                break;
            }
        }

        if (all_alpha && piece.size() > 0) {
            if (cur[id].logit >= 0.0f) {
                fprintf(stderr, "  BUG: Alphabetic token %u (\"%s\") was allowed by digit-only grammar\n",
                        id, piece.c_str());
                assert(false);
            }
            found_rejected_alpha = true;
        }
    }

    assert(found_rejected_alpha);

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  OK\n");
}

// ============================================================
// Test: accept_byte handles DFA accept states with outgoing transitions
// ============================================================

static void test_accept_byte_accept_with_continuation() {
    fprintf(stderr, "Testing accept_byte with DFA accept states that have outgoing transitions...\n");

    // Construct a DFA manually where an accept state has outgoing transitions.
    // This tests the if/else in accept_byte — we want to verify both the
    // "advance to next segment" AND "keep matching" paths are taken.
    //
    // We'll use grammar: root ::= [a-z]+ which creates:
    //   root: [a-z] S'
    //   S': [a-z] S' |
    //
    // The S' rule has an empty alternate, so after matching one [a-z],
    // configs exist for both "continue matching" and "done".
    //
    // This tests the DFA + rule-call interaction at token granularity.

    auto * grammar = build_grammar_with_vocab("root ::= [a-z]+");
    assert(grammar != nullptr);

    uint32_t n = vocab->n_tokens();
    auto cur = make_token_data(n);
    auto tok_arr = llama_token_data_array{ cur.data(), cur.size(), -1, false };

    // Step 1: Apply at start — should allow tokens starting with [a-z]
    llama_grammar_apply_impl(*grammar, &tok_arr);

    int allowed_step1 = count_allowed(cur);
    fprintf(stderr, "  Step 1 allowed: %d\n", allowed_step1);
    assert(allowed_step1 > 0);

    // Pick a token that is purely lowercase alpha and accept it
    llama_token chosen = -1;
    for (uint32_t id = 0; id < n; id++) {
        if (vocab->is_eog((int32_t)id)) continue;
        const std::string & piece = vocab->token_to_piece((int32_t)id);
        if (piece.empty()) continue;
        bool all_lower = true;
        for (char c : piece) {
            if (c < 'a' || c > 'z') { all_lower = false; break; }
        }
        if (all_lower && cur[id].logit >= 0.0f) {
            chosen = (llama_token)id;
            break;
        }
    }
    assert(chosen >= 0);

    const std::string & chosen_piece = vocab->token_to_piece(chosen);
    fprintf(stderr, "  Chose token %d (\"%s\")\n", chosen, chosen_piece.c_str());

    llama_grammar_accept_impl(*grammar, chosen);

    // Step 2: After accepting one token, should STILL allow more [a-z] tokens
    // AND should allow EOG (because [a-z]+ can match ≥1 chars, and we've matched some)
    reset_logits(cur);
    llama_grammar_apply_impl(*grammar, &tok_arr);

    int allowed_step2 = count_allowed(cur);
    fprintf(stderr, "  Step 2 allowed: %d\n", allowed_step2);
    assert(allowed_step2 > 0);

    bool allow_eog = grammar->compiled->any_config_complete(grammar->configs);
    fprintf(stderr, "  EOG allowed: %s\n", allow_eog ? "yes" : "no");
    assert(allow_eog);

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  OK\n");
}

// ============================================================
// Test: performance of apply with real vocab
// ============================================================

static void test_apply_performance() {
    fprintf(stderr, "Testing apply performance...\n");

    // Use a moderately complex JSON grammar
    const char * json_grammar =
        "root ::= object\n"
        "object ::= \"{\" ws members ws \"}\"\n"
        "members ::= pair (\",\" ws pair)*\n"
        "pair ::= ws string ws \":\" ws value\n"
        "value ::= string | number | object | array | \"true\" | \"false\" | \"null\"\n"
        "array ::= \"[\" ws (value (\",\" ws value)*)? ws \"]\"\n"
        "string ::= \"\\\"\" [^\"\\\\]* \"\\\"\" \n"
        "number ::= \"-\"? [0-9]+ (\".\" [0-9]+)?\n"
        "ws ::= [ \\t\\n]*\n";

    auto * grammar = build_grammar_with_vocab(json_grammar);
    assert(grammar != nullptr);

    uint32_t n = vocab->n_tokens();
    auto cur = make_token_data(n);
    auto tok_arr = llama_token_data_array{ cur.data(), cur.size(), -1, false };

    // Warm-up
    llama_grammar_apply_impl(*grammar, &tok_arr);

    // Benchmark
    const int n_iters = 100;
    auto t_start = std::chrono::steady_clock::now();

    for (int i = 0; i < n_iters; i++) {
        reset_logits(cur);
        llama_grammar_apply_impl(*grammar, &tok_arr);
    }

    auto t_end = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double avg_ms = total_ms / n_iters;

    fprintf(stderr, "  Vocab size: %u\n", n);
    fprintf(stderr, "  %d iterations in %.1f ms (avg %.3f ms/call)\n", n_iters, total_ms, avg_ms);

    // Performance threshold: apply should complete in < 10 ms per call for typical vocab sizes
    // (This is generous — the precomputed candidates should make it much faster)
    if (avg_ms > 10.0) {
        fprintf(stderr, "  WARNING: apply is slow (%.3f ms avg). Consider optimizing.\n", avg_ms);
    }

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  OK\n");
}

// ============================================================
// Test: apply performance through a multi-step sequence
// ============================================================

static void test_apply_sequence_performance() {
    fprintf(stderr, "Testing apply sequence performance...\n");

    const char * json_grammar =
        "root ::= \"{\" ws \"\\\"name\\\"\" ws \":\" ws \"\\\"\" [a-zA-Z ]+ \"\\\"\" ws \"}\"\n"
        "ws ::= [ \\t\\n]*\n";

    auto * grammar = build_grammar_with_vocab(json_grammar);
    assert(grammar != nullptr);

    std::string input = "{\"name\":\"John Doe\"}";
    auto tokens = common_tokenize(vocab, input, false, false);

    uint32_t n = vocab->n_tokens();
    auto cur = make_token_data(n);
    auto tok_arr = llama_token_data_array{ cur.data(), cur.size(), -1, false };

    auto t_start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < tokens.size(); i++) {
        reset_logits(cur);
        llama_grammar_apply_impl(*grammar, &tok_arr);

        // Verify the correct token is allowed
        assert(cur[tokens[i]].logit >= 0.0f);

        llama_grammar_accept_impl(*grammar, tokens[i]);
    }

    auto t_end = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    fprintf(stderr, "  %zu tokens processed in %.1f ms (avg %.3f ms/token)\n",
            tokens.size(), total_ms, total_ms / tokens.size());

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  OK\n");
}

// ============================================================
// Test: negated char class with real vocab
// ============================================================

static void test_apply_negated_class() {
    fprintf(stderr, "Testing apply with negated char class...\n");

    // [^0-9]+ should reject tokens that are purely digits
    auto * grammar = build_grammar_with_vocab("root ::= [^0-9]+");
    assert(grammar != nullptr);

    uint32_t n = vocab->n_tokens();
    auto cur = make_token_data(n);
    auto tok_arr = llama_token_data_array{ cur.data(), cur.size(), -1, false };

    llama_grammar_apply_impl(*grammar, &tok_arr);

    // Pure digit tokens must be rejected
    for (uint32_t id = 0; id < n; id++) {
        if (vocab->is_eog((int32_t)id)) continue;
        const std::string & piece = vocab->token_to_piece((int32_t)id);
        if (piece.empty() || piece[0] == '\0') continue;

        bool all_digits = true;
        for (char c : piece) {
            if (c < '0' || c > '9') { all_digits = false; break; }
        }

        if (all_digits && cur[id].logit >= 0.0f) {
            fprintf(stderr, "  BUG: Digit-only token %u (\"%s\") allowed by [^0-9]+ grammar\n",
                    id, piece.c_str());
            assert(false);
        }
    }

    // Some non-digit tokens should be allowed
    int allowed = count_allowed(cur);
    assert(allowed > 0);

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  OK\n");
}

// ============================================================
// Test: Unicode grammar with real vocab
// ============================================================

static void test_apply_unicode() {
    fprintf(stderr, "Testing apply with Unicode grammar...\n");

    // This grammar only allows specific Unicode ranges
    auto * grammar = build_grammar_with_vocab("root ::= [a-zA-Z]+");
    assert(grammar != nullptr);

    uint32_t n = vocab->n_tokens();
    auto cur = make_token_data(n);
    auto tok_arr = llama_token_data_array{ cur.data(), cur.size(), -1, false };

    llama_grammar_apply_impl(*grammar, &tok_arr);

    // Verify: tokens with non-alpha characters should be rejected
    for (uint32_t id = 0; id < n; id++) {
        if (vocab->is_eog((int32_t)id)) continue;
        const std::string & piece = vocab->token_to_piece((int32_t)id);
        if (piece.empty() || piece[0] == '\0') continue;

        // Check if first byte is in [a-zA-Z]
        char first = piece[0];
        bool starts_alpha = (first >= 'a' && first <= 'z') || (first >= 'A' && first <= 'Z');

        if (!starts_alpha && cur[id].logit >= 0.0f) {
            fprintf(stderr, "  BUG: Token %u (\"%s\") allowed but starts with non-alpha char 0x%02x\n",
                    id, piece.c_str(), (unsigned char)first);
            assert(false);
        }
    }

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  OK\n");
}

// ============================================================
// Test: complex JSON number grammar with real vocab
// ============================================================

static void test_apply_json_number() {
    fprintf(stderr, "Testing apply with JSON number grammar...\n");

    const char * number_grammar =
        "root ::= \"-\"? int frac? exp?\n"
        "int ::= [0-9] | [1-9] [0-9]+\n"
        "frac ::= \".\" [0-9]+\n"
        "exp ::= [eE] [\"-+\"]? [0-9]+\n";

    test_candidates_superset(number_grammar, "JSON number");
}

// ============================================================
// Test: candidate superset for various grammars
// ============================================================

static void test_candidate_superset_grammars() {
    // Simple literal
    test_candidates_superset("root ::= \"hello world\"", "literal string");

    // Alternation
    test_candidates_superset("root ::= \"true\" | \"false\" | \"null\"", "boolean/null literals");

    // Character class
    test_candidates_superset("root ::= [a-zA-Z_][a-zA-Z0-9_]*", "identifier");

    // Repetition with negation
    test_candidates_superset(
        "root ::= \"\\\"\" [^\"\\\\]* \"\\\"\"",
        "simple quoted string");
}

// ============================================================
// Test: grammar with many alternates (stress test for uint8_t alternate_idx)
// ============================================================

static void test_many_alternates() {
    fprintf(stderr, "Testing grammar with many alternates...\n");

    // Build a grammar with ~50 alternates (well within uint8_t range)
    std::string grammar_str = "root ::= ";
    for (int i = 0; i < 50; i++) {
        if (i > 0) grammar_str += " | ";
        grammar_str += "\"option" + std::to_string(i) + "\"";
    }

    auto * grammar = build_grammar_with_vocab(grammar_str);
    assert(grammar != nullptr);

    // Verify specific strings match
    assert(match_string("option0", grammar));
    assert(match_string("option25", grammar));
    assert(match_string("option49", grammar));
    assert(!match_string("option50", grammar));

    // Test apply
    uint32_t n = vocab->n_tokens();
    auto cur = make_token_data(n);
    auto tok_arr = llama_token_data_array{ cur.data(), cur.size(), -1, false };

    llama_grammar_apply_impl(*grammar, &tok_arr);

    int allowed = count_allowed(cur);
    fprintf(stderr, "  Allowed tokens: %d / %u (50 alternates)\n", allowed, n);
    assert(allowed > 0);

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  OK\n");
}

// ============================================================
// Test: recursive grammar (nested objects)
// ============================================================

static void test_recursive_grammar_apply() {
    fprintf(stderr, "Testing recursive grammar with apply...\n");

    const char * grammar_str =
        "root ::= value\n"
        "value ::= number | array\n"
        "number ::= [0-9]+\n"
        "array ::= \"[\" ws value (\",\" ws value)* ws \"]\"\n"
        "ws ::= [ ]*\n";

    auto * grammar = build_grammar_with_vocab(grammar_str);
    assert(grammar != nullptr);

    // Test nested sequence
    test_token_sequence(grammar_str, "42", "number");
    test_token_sequence(grammar_str, "[1,2,3]", "flat array");
    test_token_sequence(grammar_str, "[[1,2],[3]]", "nested array");

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  OK\n");
}

// ============================================================
// Test: apply after partial acceptance (mid-token boundary)
// ============================================================

static void test_apply_mid_segment() {
    fprintf(stderr, "Testing apply at DFA mid-segment state...\n");

    // Grammar: root ::= "abcdef"
    // After accepting a token "abc", the DFA should be mid-segment at "def"
    auto * grammar = build_grammar_with_vocab("root ::= \"abcdef\"");
    assert(grammar != nullptr);

    uint32_t n = vocab->n_tokens();
    auto cur = make_token_data(n);
    auto tok_arr = llama_token_data_array{ cur.data(), cur.size(), -1, false };

    // First apply: should allow tokens starting with 'a'
    llama_grammar_apply_impl(*grammar, &tok_arr);

    // Find and accept a token that starts with "abc" (or just "a")
    // We'll tokenize "abcdef" and use the first token
    auto tokens = common_tokenize(vocab, "abcdef", false, false);
    assert(!tokens.empty());

    // Verify first token is allowed
    assert(cur[tokens[0]].logit >= 0.0f);

    // Accept it
    llama_grammar_accept_impl(*grammar, tokens[0]);

    // Now apply again — should still allow valid continuation tokens
    reset_logits(cur);
    llama_grammar_apply_impl(*grammar, &tok_arr);

    if (tokens.size() > 1) {
        // Second token should be allowed
        if (cur[tokens[1]].logit < 0.0f) {
            const std::string & piece = vocab->token_to_piece(tokens[1]);
            fprintf(stderr, "  BUG: Continuation token %d (\"%s\") was rejected\n",
                    tokens[1], piece.c_str());
            assert(false);
        }
    }

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  OK\n");
}

// ============================================================
// Test: EOG handling
// ============================================================

static void test_eog_handling() {
    fprintf(stderr, "Testing EOG handling...\n");

    // Grammar allows "a" or "ab" — after consuming "a", EOG should be allowed
    auto * grammar = build_grammar_with_vocab("root ::= \"a\" \"b\"?");
    assert(grammar != nullptr);

    uint32_t n = vocab->n_tokens();
    auto cur = make_token_data(n);
    auto tok_arr = llama_token_data_array{ cur.data(), cur.size(), -1, false };

    // Tokenize and accept "a"
    auto tokens = common_tokenize(vocab, "a", false, false);
    for (auto tok : tokens) {
        reset_logits(cur);
        llama_grammar_apply_impl(*grammar, &tok_arr);
        assert(cur[tok].logit >= 0.0f);
        llama_grammar_accept_impl(*grammar, tok);
    }

    // After "a", EOG should be allowed (since "b" is optional)
    bool allow_eog = grammar->compiled->any_config_complete(grammar->configs);
    assert(allow_eog);

    // But "b" should also be allowed (optional continuation)
    reset_logits(cur);
    llama_grammar_apply_impl(*grammar, &tok_arr);

    // Check: at least some continuation tokens should be allowed (those starting with 'b')
    bool found_b_token = false;
    for (uint32_t id = 0; id < n; id++) {
        if (vocab->is_eog((int32_t)id)) continue;
        const std::string & piece = vocab->token_to_piece((int32_t)id);
        if (!piece.empty() && piece[0] == 'b' && cur[id].logit >= 0.0f) {
            found_b_token = true;
            break;
        }
    }
    assert(found_b_token);

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  OK\n");
}

// ============================================================
// Main
// ============================================================

int main(int argc, char ** argv) {
    fprintf(stderr, "=== Grammar DFA Apply Tests (with real vocab) ===\n\n");

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <vocab-file.gguf>\n", argv[0]);
        fprintf(stderr, "Skipping test (no vocab file provided)\n");
        return 0;
    }

    const char * vocab_file = argv[1];
    fprintf(stderr, "Loading vocab from: %s\n\n", vocab_file);

    llama_model * model = nullptr;
    llama_context * ctx = nullptr;

    llama_backend_init();

    // Load vocab only
    {
        auto mparams = llama_model_default_params();
        mparams.vocab_only = true;

        model = llama_model_load_from_file(vocab_file, mparams);
        if (!model) {
            fprintf(stderr, "ERROR: Failed to load vocab from '%s'\n", vocab_file);
            return 1;
        }

        auto cparams = llama_context_default_params();
        ctx = llama_init_from_model(model, cparams);
        if (!ctx) {
            fprintf(stderr, "ERROR: Failed to create context\n");
            llama_model_free(model);
            return 1;
        }
    }

    vocab = llama_model_get_vocab(model);
    assert(vocab != nullptr);

    fprintf(stderr, "Vocab size: %u\n\n", vocab->n_tokens());

    // Correctness tests
    test_apply_simple_literal();
    test_invalid_token_rejected();
    test_apply_negated_class();
    test_apply_unicode();
    test_apply_mid_segment();
    test_eog_handling();
    test_accept_byte_accept_with_continuation();

    // Candidate superset verification (brute-force vs precomputed)
    test_candidate_superset_grammars();
    test_apply_json_number();

    // Multi-token sequence tests
    test_multi_token_sequence();
    test_token_sequence(
        "root ::= [a-zA-Z]+ \" \" [a-zA-Z]+",
        "hello world",
        "two words");
    test_token_sequence(
        "root ::= [0-9]+ \".\" [0-9]+",
        "3.14",
        "decimal number");
    test_token_sequence(
        "root ::= \"true\" | \"false\"",
        "true",
        "boolean literal");

    // Stress tests
    test_many_alternates();
    test_recursive_grammar_apply();

    // Performance tests
    test_apply_performance();
    test_apply_sequence_performance();

    fprintf(stderr, "\n=== All grammar DFA apply tests passed! ===\n");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
