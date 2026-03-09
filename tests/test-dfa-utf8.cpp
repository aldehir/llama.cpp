#include <cstdlib>
#include <string>
#include <iostream>
#include <vector>

#include "peg-parser/testing.h"
#include "common-dfa-utf8.h"

// ----------------------------------------------------------------------------
// Minimal byte-level DFA stepper (mirrors the sampler's internal logic so
// we can validate the automaton without needing a model / vocab).
// ----------------------------------------------------------------------------

struct test_dfa {
    uint32_t n_states;
    uint32_t start;
    uint32_t dead;
    std::vector<uint32_t> accept;

    // edges[state] = sorted vector of { lo, hi, dst }
    struct edge { uint8_t lo; uint8_t hi; uint32_t dst; };
    std::vector<std::vector<edge>> edges;
};

static test_dfa build_test_dfa(const common_dfa_data & d) {
    test_dfa td;
    td.n_states = d.n_states;
    td.start    = d.start_state;
    td.dead     = d.dead_state;
    td.accept   = d.accept_states;
    td.edges.resize(d.n_states);
    for (const auto & t : d.transitions) {
        td.edges[t.src].push_back({ t.byte_lo, t.byte_hi, t.dst });
    }
    for (auto & es : td.edges) {
        std::sort(es.begin(), es.end(),
                  [](const test_dfa::edge & a, const test_dfa::edge & b) {
                      return a.lo < b.lo;
                  });
    }
    return td;
}

static uint32_t step(const test_dfa & dfa, uint32_t state, uint8_t byte) {
    if (state == dfa.dead) return dfa.dead;
    const auto & es = dfa.edges[state];
    int lo = 0, hi = (int)es.size() - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (byte < es[mid].lo)       hi = mid - 1;
        else if (byte > es[mid].hi)  lo = mid + 1;
        else                         return es[mid].dst;
    }
    return dfa.dead;
}

static uint32_t run(const test_dfa & dfa, const uint8_t * bytes, size_t n) {
    uint32_t s = dfa.start;
    for (size_t i = 0; i < n && s != dfa.dead; ++i) {
        s = step(dfa, s, bytes[i]);
    }
    return s;
}

static uint32_t run(const test_dfa & dfa, const std::string & s) {
    return run(dfa, reinterpret_cast<const uint8_t *>(s.data()), s.size());
}

static bool is_accept(const test_dfa & dfa, uint32_t state) {
    for (uint32_t a : dfa.accept) {
        if (state == a) return true;
    }
    return false;
}

static bool accepts(const test_dfa & dfa, const std::string & s) {
    return is_accept(dfa, run(dfa, s));
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

static void test_ascii(testing & t) {
    auto d = common_dfa_utf8();
    auto dfa = build_test_dfa(d);

    t.test("single ascii chars", [&](testing & t) {
        t.assert_true("NUL",       accepts(dfa, std::string(1, '\0')));
        t.assert_true("space",     accepts(dfa, " "));
        t.assert_true("A",         accepts(dfa, "A"));
        t.assert_true("tilde",     accepts(dfa, "~"));
        t.assert_true("DEL (7F)",  accepts(dfa, "\x7F"));
    });

    t.test("ascii string", [&](testing & t) {
        t.assert_true(accepts(dfa, "Hello, world!"));
    });

    t.test("empty string stays at start (accept)", [&](testing & t) {
        t.assert_true(is_accept(dfa, dfa.start));
    });
}

static void test_multibyte(testing & t) {
    auto d = common_dfa_utf8();
    auto dfa = build_test_dfa(d);

    t.test("2-byte codepoints", [&](testing & t) {
        // U+00A9 © = C2 A9
        t.assert_true("copyright",  accepts(dfa, "\xC2\xA9"));
        // U+00FF ÿ = C3 BF
        t.assert_true("y-umlaut",   accepts(dfa, "\xC3\xBF"));
        // U+07FF = DF BF (max 2-byte)
        t.assert_true("max 2-byte", accepts(dfa, "\xDF\xBF"));
    });

    t.test("3-byte codepoints", [&](testing & t) {
        // U+0800 = E0 A0 80 (min 3-byte)
        t.assert_true("min 3-byte", accepts(dfa, "\xE0\xA0\x80"));
        // U+20AC € = E2 82 AC
        t.assert_true("euro sign",  accepts(dfa, "\xE2\x82\xAC"));
        // U+FFFD = EF BF BD
        t.assert_true("replacement", accepts(dfa, "\xEF\xBF\xBD"));
    });

    t.test("4-byte codepoints", [&](testing & t) {
        // U+10000 = F0 90 80 80 (min 4-byte)
        t.assert_true("min 4-byte",  accepts(dfa, "\xF0\x90\x80\x80"));
        // U+1F600 😀 = F0 9F 98 80
        t.assert_true("grinning",    accepts(dfa, "\xF0\x9F\x98\x80"));
        // U+10FFFF = F4 8F BF BF (max valid)
        t.assert_true("max valid",   accepts(dfa, "\xF4\x8F\xBF\xBF"));
    });

    t.test("mixed multibyte string", [&](testing & t) {
        // "Héllo 🌍" = H é l l o   🌍
        t.assert_true(accepts(dfa, "H\xC3\xA9llo \xF0\x9F\x8C\x8D"));
    });
}

static void test_invalid(testing & t) {
    auto d = common_dfa_utf8();
    auto dfa = build_test_dfa(d);

    t.test("bare continuation byte", [&](testing & t) {
        t.assert_true(!accepts(dfa, "\x80"));
        t.assert_true(!accepts(dfa, "\xBF"));
    });

    t.test("overlong 2-byte (C0, C1)", [&](testing & t) {
        t.assert_true("C0 80", !accepts(dfa, "\xC0\x80"));
        t.assert_true("C1 BF", !accepts(dfa, "\xC1\xBF"));
    });

    t.test("overlong 3-byte (E0 80-9F)", [&](testing & t) {
        t.assert_true("E0 80 80", !accepts(dfa, "\xE0\x80\x80"));
        t.assert_true("E0 9F BF", !accepts(dfa, "\xE0\x9F\xBF"));
    });

    t.test("surrogates (ED A0-BF)", [&](testing & t) {
        // U+D800 = ED A0 80
        t.assert_true("ED A0 80", !accepts(dfa, "\xED\xA0\x80"));
        // U+DFFF = ED BF BF
        t.assert_true("ED BF BF", !accepts(dfa, "\xED\xBF\xBF"));
    });

    t.test("overlong 4-byte (F0 80-8F)", [&](testing & t) {
        t.assert_true("F0 80 80 80", !accepts(dfa, "\xF0\x80\x80\x80"));
        t.assert_true("F0 8F BF BF", !accepts(dfa, "\xF0\x8F\xBF\xBF"));
    });

    t.test("above U+10FFFF", [&](testing & t) {
        // F4 90 80 80 = U+110000
        t.assert_true("F4 90 80 80", !accepts(dfa, "\xF4\x90\x80\x80"));
        // F5+ lead bytes
        t.assert_true("F5 80 80 80", !accepts(dfa, "\xF5\x80\x80\x80"));
        t.assert_true("FF",          !accepts(dfa, "\xFF"));
    });

    t.test("truncated sequences", [&](testing & t) {
        // 2-byte missing continuation
        t.assert_true("C2 alone", !accepts(dfa, "\xC2"));
        // 3-byte missing last continuation
        t.assert_true("E2 82 alone", !accepts(dfa, "\xE2\x82"));
        // 4-byte missing last two
        t.assert_true("F0 9F alone", !accepts(dfa, "\xF0\x9F"));
        // 4-byte missing last one
        t.assert_true("F0 9F 98 alone", !accepts(dfa, "\xF0\x9F\x98"));
    });

    t.test("valid prefix then invalid", [&](testing & t) {
        // "A" then bare continuation
        t.assert_true(!accepts(dfa, "A\x80"));
        // valid 2-byte then truncated 2-byte
        t.assert_true(!accepts(dfa, "\xC2\xA9\xC2"));
    });
}

static void test_intermediate_states(testing & t) {
    auto d = common_dfa_utf8();
    auto dfa = build_test_dfa(d);

    t.test("mid-sequence states are not accept", [&](testing & t) {
        // After first byte of 2-byte sequence
        uint32_t s = run(dfa, "\xC2");
        t.assert_true("after C2: not dead",   s != dfa.dead);
        t.assert_true("after C2: not accept", !is_accept(dfa, s));

        // After first byte of 4-byte sequence
        s = run(dfa, "\xF0");
        t.assert_true("after F0: not dead",   s != dfa.dead);
        t.assert_true("after F0: not accept", !is_accept(dfa, s));

        // After two bytes of 4-byte sequence
        s = run(dfa, "\xF0\x9F");
        t.assert_true("after F0 9F: not dead",   s != dfa.dead);
        t.assert_true("after F0 9F: not accept", !is_accept(dfa, s));
    });
}

static void test_params(testing & t) {
    auto d = common_dfa_utf8();
    auto p = d.params();

    t.test("params basic fields", [&](testing & t) {
        t.assert_equal("n_states", (uint32_t)9, p.n_states);
        t.assert_equal("start",    (uint32_t)0, p.start_state);
        t.assert_equal("dead",     (uint32_t)1, p.dead_state);
        t.assert_equal("n_accept", (size_t)1,   p.n_accept_states);
        t.assert_equal("accept[0]", (uint32_t)0, p.accept_states[0]);
    });

    t.test("no token limit by default", [&](testing & t) {
        t.assert_true("null limit states", p.token_limit_states == nullptr);
        t.assert_equal("n_limit_states", (size_t)0, p.n_token_limit_states);
    });

    t.test("token limit round-trip", [&](testing & t) {
        common_dfa_data d2 = common_dfa_utf8();
        d2.token_limit_states = {0};
        d2.token_limit = 42;
        auto p2 = d2.params();
        t.assert_true("non-null limit states", p2.token_limit_states != nullptr);
        t.assert_equal("n_limit_states", (size_t)1, p2.n_token_limit_states);
        t.assert_equal("limit value", (int32_t)42, p2.token_limit);
    });
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

int main(int argc, char * argv[]) {
    testing t(std::cout);
    if (argc >= 2) {
        t.set_filter(argv[1]);
    }

    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    if (verbose) {
        t.verbose = std::string(verbose) == "1";
    }

    t.test("ascii",               test_ascii);
    t.test("multibyte",           test_multibyte);
    t.test("invalid",             test_invalid);
    t.test("intermediate_states", test_intermediate_states);
    t.test("params",              test_params);

    return t.summary();
}
