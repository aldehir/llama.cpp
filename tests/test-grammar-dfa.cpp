#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../src/llama-grammar.h"
#include "../src/llama-grammar-dfa.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

// ============================================================
// Test helpers
// ============================================================

static llama_grammar * build_grammar(const std::string & grammar_str) {
    return llama_grammar_init_impl(nullptr, grammar_str.c_str(), "root", false, nullptr, 0, nullptr, 0);
}

// Test that a string is accepted by the DFA engine
static bool dfa_match_string(const std::string & input, llama_grammar * grammar) {
    if (!grammar->compiled) {
        fprintf(stderr, "  ⚠ No compiled grammar available\n");
        return false;
    }

    auto configs = grammar->configs;  // copy to avoid modifying grammar state
    for (size_t i = 0; i < input.size(); i++) {
        grammar->compiled->accept_byte(configs, (uint8_t)input[i]);
        if (configs.empty()) {
            return false;
        }
    }

    return grammar->compiled->any_config_complete(configs);
}


// ============================================================
// UTF-8 encoding tests
// ============================================================

static void test_utf8_encoding() {
    fprintf(stderr, "Testing UTF-8 encoding...\n");

    // ASCII
    auto bytes = encode_utf8_bytes('A');
    assert(bytes.size() == 1 && bytes[0] == 0x41);

    // 2-byte: α = U+03B1
    bytes = encode_utf8_bytes(0x03B1);
    assert(bytes.size() == 2 && bytes[0] == 0xCE && bytes[1] == 0xB1);

    // 3-byte: € = U+20AC
    bytes = encode_utf8_bytes(0x20AC);
    assert(bytes.size() == 3 && bytes[0] == 0xE2 && bytes[1] == 0x82 && bytes[2] == 0xAC);

    // 4-byte: 𝄞 = U+1D11E
    bytes = encode_utf8_bytes(0x1D11E);
    assert(bytes.size() == 4 && bytes[0] == 0xF0 && bytes[1] == 0x9D && bytes[2] == 0x84 && bytes[3] == 0x9E);

    fprintf(stderr, "  ✅ UTF-8 encoding tests passed\n");
}

// ============================================================
// NFA construction tests
// ============================================================

static void test_nfa_construction() {
    fprintf(stderr, "Testing NFA construction...\n");

    // Test byte match
    auto nfa = grammar_nfa::byte_match(0x61, 0x7A);  // [a-z]
    assert(nfa.states.size() == 2);
    assert(nfa.states[nfa.start].byte_transitions.size() == 1);
    assert(nfa.states[nfa.start].byte_transitions[0].lo == 0x61);
    assert(nfa.states[nfa.start].byte_transitions[0].hi == 0x7A);

    // Test concat
    auto a = grammar_nfa::byte_match('a', 'a');
    auto b = grammar_nfa::byte_match('b', 'b');
    auto ab = grammar_nfa::concat(std::move(a), std::move(b));
    // Should have 4 states: a_start, a_accept, b_start, b_accept
    // with epsilon from a_accept to b_start
    assert(ab.states.size() == 4);

    // Test alternation
    auto x = grammar_nfa::byte_match('x', 'x');
    auto y = grammar_nfa::byte_match('y', 'y');
    auto xy = grammar_nfa::alternation({std::move(x), std::move(y)});
    // Should have 6 states: new_start, new_accept, x_start, x_accept, y_start, y_accept
    assert(xy.states.size() == 6);

    // Test epsilon
    auto eps = grammar_nfa::epsilon_nfa();
    assert(eps.states.size() == 2);
    assert(eps.states[eps.start].epsilon_transitions.size() == 1);
    assert(eps.states[eps.start].epsilon_transitions[0] == eps.accept);

    fprintf(stderr, "  ✅ NFA construction tests passed\n");
}

// ============================================================
// DFA construction and matching tests
// ============================================================

static bool dfa_accepts(const byte_dfa & dfa, const std::string & input) {
    uint16_t state = dfa.start_state;
    for (uint8_t b : input) {
        state = dfa.transitions[state][b];
        if (state == 0) return false;
    }
    return dfa.accept[state];
}

static bool dfa_accepts_bytes(const byte_dfa & dfa, const std::vector<uint8_t> & input) {
    uint16_t state = dfa.start_state;
    for (uint8_t b : input) {
        state = dfa.transitions[state][b];
        if (state == 0) return false;
    }
    return dfa.accept[state];
}

static void test_dfa_ascii_range() {
    fprintf(stderr, "Testing DFA for ASCII range [a-z]...\n");

    auto nfa = codepoint_range_to_byte_nfa('a', 'z');
    auto dfa = byte_dfa::from_nfa(nfa);
    dfa.minimize();

    assert(dfa_accepts(dfa, "a"));
    assert(dfa_accepts(dfa, "m"));
    assert(dfa_accepts(dfa, "z"));
    assert(!dfa_accepts(dfa, "A"));
    assert(!dfa_accepts(dfa, "0"));
    assert(!dfa_accepts(dfa, ""));
    assert(!dfa_accepts(dfa, "ab"));

    fprintf(stderr, "  ✅ ASCII range DFA tests passed\n");
}

static void test_dfa_multibyte_range() {
    fprintf(stderr, "Testing DFA for multi-byte range [α-ω]...\n");

    // α = U+03B1, ω = U+03C9
    auto nfa = codepoint_range_to_byte_nfa(0x03B1, 0x03C9);
    auto dfa = byte_dfa::from_nfa(nfa);
    dfa.minimize();

    // α = CE B1
    assert(dfa_accepts_bytes(dfa, {0xCE, 0xB1}));
    // β = CE B2
    assert(dfa_accepts_bytes(dfa, {0xCE, 0xB2}));
    // ω = CF 89
    assert(dfa_accepts_bytes(dfa, {0xCF, 0x89}));
    // γ = CE B3
    assert(dfa_accepts_bytes(dfa, {0xCE, 0xB3}));
    // Should not match ASCII 'a'
    assert(!dfa_accepts_bytes(dfa, {0x61}));
    // Should not match beyond ω: U+03CA = CF 8A
    assert(!dfa_accepts_bytes(dfa, {0xCF, 0x8A}));
    // Should not match below α: U+03B0 = CE B0
    assert(!dfa_accepts_bytes(dfa, {0xCE, 0xB0}));

    fprintf(stderr, "  ✅ Multi-byte range DFA tests passed\n");
}

static void test_dfa_wildcard() {
    fprintf(stderr, "Testing DFA for wildcard (valid UTF-8 char)...\n");

    auto dfa = build_valid_utf8_dfa();

    // ASCII
    assert(dfa_accepts_bytes(dfa, {0x41}));  // 'A'
    assert(dfa_accepts_bytes(dfa, {0x00}));  // NULL
    assert(dfa_accepts_bytes(dfa, {0x7F}));  // DEL

    // 2-byte: α = CE B1
    assert(dfa_accepts_bytes(dfa, {0xCE, 0xB1}));
    // 3-byte: € = E2 82 AC
    assert(dfa_accepts_bytes(dfa, {0xE2, 0x82, 0xAC}));
    // 4-byte: 𝄞 = F0 9D 84 9E
    assert(dfa_accepts_bytes(dfa, {0xF0, 0x9D, 0x84, 0x9E}));

    // Invalid: overlong 2-byte for ASCII
    assert(!dfa_accepts_bytes(dfa, {0xC0, 0x80}));
    assert(!dfa_accepts_bytes(dfa, {0xC1, 0xBF}));
    // Invalid: bare continuation byte
    assert(!dfa_accepts_bytes(dfa, {0x80}));
    // Invalid: too short
    assert(!dfa_accepts_bytes(dfa, {0xCE}));
    // Invalid: surrogate (U+D800 = ED A0 80)
    assert(!dfa_accepts_bytes(dfa, {0xED, 0xA0, 0x80}));

    fprintf(stderr, "  ✅ Wildcard (valid UTF-8) DFA tests passed\n");
}

static void test_dfa_complement() {
    fprintf(stderr, "Testing DFA complement for negated class [^a]...\n");

    // Build DFA for [a], then complement and intersect with valid UTF-8
    auto nfa = codepoint_range_to_byte_nfa('a', 'a');
    auto positive_dfa = byte_dfa::from_nfa(nfa);
    positive_dfa.minimize();
    positive_dfa.complement();
    auto utf8_dfa = build_valid_utf8_dfa();
    auto negated = byte_dfa::intersect(positive_dfa, utf8_dfa);
    negated.minimize();

    // 'a' should NOT be accepted
    assert(!dfa_accepts(negated, "a"));
    // Other ASCII chars should be accepted
    assert(dfa_accepts(negated, "b"));
    assert(dfa_accepts(negated, "z"));
    assert(dfa_accepts(negated, "0"));
    assert(dfa_accepts(negated, "A"));
    // Multi-byte chars should be accepted
    assert(dfa_accepts_bytes(negated, {0xCE, 0xB1}));  // α
    // Empty string should NOT be accepted
    assert(!dfa_accepts(negated, ""));
    // Multi-char strings should NOT be accepted (only matches single char)
    assert(!dfa_accepts(negated, "bc"));

    fprintf(stderr, "  ✅ DFA complement tests passed\n");
}

static void test_dfa_mixed_class() {
    fprintf(stderr, "Testing DFA for mixed class [a-zα-ω]...\n");

    // Build alternation of ASCII and multi-byte ranges
    auto ascii_nfa = codepoint_range_to_byte_nfa('a', 'z');
    auto greek_nfa = codepoint_range_to_byte_nfa(0x03B1, 0x03C9);
    auto combined = grammar_nfa::alternation({std::move(ascii_nfa), std::move(greek_nfa)});
    auto dfa = byte_dfa::from_nfa(combined);
    dfa.minimize();

    // ASCII
    assert(dfa_accepts(dfa, "a"));
    assert(dfa_accepts(dfa, "z"));
    // Greek
    assert(dfa_accepts_bytes(dfa, {0xCE, 0xB1}));  // α
    assert(dfa_accepts_bytes(dfa, {0xCF, 0x89}));  // ω
    // Not in range
    assert(!dfa_accepts(dfa, "A"));
    assert(!dfa_accepts(dfa, "0"));
    assert(!dfa_accepts_bytes(dfa, {0xCF, 0x8A}));  // beyond ω

    fprintf(stderr, "  ✅ Mixed class DFA tests passed\n");
}

// ============================================================
// Segment DFA tests
// ============================================================

static void test_segment_dfa_simple_string() {
    fprintf(stderr, "Testing segment DFA for simple string \"hello\"...\n");

    // Build elements for "hello"
    llama_grammar_element elements[] = {
        {LLAMA_GRETYPE_CHAR, 'h'},
        {LLAMA_GRETYPE_CHAR, 'e'},
        {LLAMA_GRETYPE_CHAR, 'l'},
        {LLAMA_GRETYPE_CHAR, 'l'},
        {LLAMA_GRETYPE_CHAR, 'o'},
        {LLAMA_GRETYPE_END, 0},
    };

    auto dfa = build_segment_dfa(elements, elements + 5);

    assert(dfa_accepts(dfa, "hello"));
    assert(!dfa_accepts(dfa, "hell"));
    assert(!dfa_accepts(dfa, "helloo"));
    assert(!dfa_accepts(dfa, "world"));
    assert(!dfa_accepts(dfa, ""));

    fprintf(stderr, "  ✅ Simple string segment DFA tests passed\n");
}

static void test_segment_dfa_char_range() {
    fprintf(stderr, "Testing segment DFA for char range [a-z]...\n");

    llama_grammar_element elements[] = {
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, 'z'},
        {LLAMA_GRETYPE_END, 0},
    };

    auto dfa = build_segment_dfa(elements, elements + 2);

    assert(dfa_accepts(dfa, "a"));
    assert(dfa_accepts(dfa, "m"));
    assert(dfa_accepts(dfa, "z"));
    assert(!dfa_accepts(dfa, "A"));
    assert(!dfa_accepts(dfa, "0"));
    assert(!dfa_accepts(dfa, ""));
    assert(!dfa_accepts(dfa, "ab"));

    fprintf(stderr, "  ✅ Char range segment DFA tests passed\n");
}

static void test_segment_dfa_negated() {
    fprintf(stderr, "Testing segment DFA for negated class [^a]...\n");

    llama_grammar_element elements[] = {
        {LLAMA_GRETYPE_CHAR_NOT, 'a'},
        {LLAMA_GRETYPE_END, 0},
    };

    auto dfa = build_segment_dfa(elements, elements + 1);

    assert(!dfa_accepts(dfa, "a"));
    assert(dfa_accepts(dfa, "b"));
    assert(dfa_accepts(dfa, "z"));
    assert(dfa_accepts(dfa, "0"));
    assert(!dfa_accepts(dfa, ""));

    fprintf(stderr, "  ✅ Negated class segment DFA tests passed\n");
}

// ============================================================
// Grammar compilation and runtime tests
// ============================================================

static void test_grammar_simple() {
    fprintf(stderr, "Testing full grammar: root ::= \"hello\"...\n");

    auto * grammar = build_grammar("root ::= \"hello\"");
    assert(grammar != nullptr);

    assert(dfa_match_string("hello", grammar));
    assert(!dfa_match_string("hell", grammar));
    assert(!dfa_match_string("helloo", grammar));
    assert(!dfa_match_string("world", grammar));
    assert(!dfa_match_string("", grammar));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ Simple grammar DFA tests passed\n");
}

static void test_grammar_alternation() {
    fprintf(stderr, "Testing full grammar: root ::= \"hello\" | \"hi\"...\n");

    auto * grammar = build_grammar("root ::= \"hello\" | \"hi\"");
    assert(grammar != nullptr);

    assert(dfa_match_string("hello", grammar));
    assert(dfa_match_string("hi", grammar));
    assert(!dfa_match_string("hel", grammar));
    assert(!dfa_match_string("h", grammar));
    assert(!dfa_match_string("hey", grammar));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ Alternation grammar DFA tests passed\n");
}

static void test_grammar_char_class() {
    fprintf(stderr, "Testing full grammar: root ::= [a-z]...\n");

    auto * grammar = build_grammar("root ::= [a-z]");
    assert(grammar != nullptr);

    assert(dfa_match_string("a", grammar));
    assert(dfa_match_string("m", grammar));
    assert(dfa_match_string("z", grammar));
    assert(!dfa_match_string("A", grammar));
    assert(!dfa_match_string("0", grammar));
    assert(!dfa_match_string("ab", grammar));
    assert(!dfa_match_string("", grammar));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ Char class grammar DFA tests passed\n");
}

static void test_grammar_repetition() {
    fprintf(stderr, "Testing full grammar: root ::= [a-z]+...\n");

    auto * grammar = build_grammar("root ::= [a-z]+");
    assert(grammar != nullptr);

    assert(dfa_match_string("a", grammar));
    assert(dfa_match_string("abc", grammar));
    assert(dfa_match_string("hello", grammar));
    assert(!dfa_match_string("", grammar));
    assert(!dfa_match_string("Hello", grammar));
    assert(!dfa_match_string("abc123", grammar));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ Repetition grammar DFA tests passed\n");
}

static void test_grammar_optional() {
    fprintf(stderr, "Testing full grammar: root ::= \"a\" \"b\"?...\n");

    auto * grammar = build_grammar("root ::= \"a\" \"b\"?");
    assert(grammar != nullptr);

    assert(dfa_match_string("a", grammar));
    assert(dfa_match_string("ab", grammar));
    assert(!dfa_match_string("b", grammar));
    assert(!dfa_match_string("abc", grammar));
    assert(!dfa_match_string("", grammar));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ Optional grammar DFA tests passed\n");
}

static void test_grammar_rule_ref() {
    fprintf(stderr, "Testing full grammar with rule references...\n");

    auto * grammar = build_grammar(
        "root ::= greeting \" \" name\n"
        "greeting ::= \"hello\" | \"hi\"\n"
        "name ::= [A-Z] [a-z]+");
    assert(grammar != nullptr);

    assert(dfa_match_string("hello World", grammar));
    assert(dfa_match_string("hi Bob", grammar));
    assert(!dfa_match_string("hello", grammar));
    assert(!dfa_match_string("hello world", grammar));  // lowercase w
    assert(!dfa_match_string("hey World", grammar));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ Rule reference grammar DFA tests passed\n");
}

static void test_grammar_unicode() {
    fprintf(stderr, "Testing full grammar with Unicode...\n");

    auto * grammar = build_grammar("root ::= [α-ω]+");
    assert(grammar != nullptr);

    assert(dfa_match_string("α", grammar));
    assert(dfa_match_string("αβγ", grammar));
    assert(dfa_match_string("ω", grammar));
    assert(!dfa_match_string("a", grammar));
    assert(!dfa_match_string("", grammar));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ Unicode grammar DFA tests passed\n");
}

static void test_grammar_negation() {
    fprintf(stderr, "Testing full grammar with negation [^\"]+...\n");

    auto * grammar = build_grammar("root ::= [^\"]+ ");
    assert(grammar != nullptr);

    assert(dfa_match_string("hello ", grammar));
    assert(dfa_match_string("abc ", grammar));
    assert(!dfa_match_string("\"abc\" ", grammar));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ Negation grammar DFA tests passed\n");
}

static void test_grammar_json_string() {
    fprintf(stderr, "Testing JSON string grammar...\n");

    auto * grammar = build_grammar(
        "root ::= \"\\\"\" chars \"\\\"\" \n"
        "chars ::= char*\n"
        "char ::= [^\"\\\\] | \"\\\\\" [\"\\\\/bfnrt]");
    assert(grammar != nullptr);

    assert(dfa_match_string("\"hello\"", grammar));
    assert(dfa_match_string("\"\"", grammar));
    assert(dfa_match_string("\"hello world\"", grammar));
    assert(dfa_match_string("\"line\\n\"", grammar));
    assert(dfa_match_string("\"tab\\t\"", grammar));
    assert(!dfa_match_string("hello", grammar));
    assert(!dfa_match_string("\"unterminated", grammar));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ JSON string grammar DFA tests passed\n");
}

static void test_grammar_quantifiers() {
    fprintf(stderr, "Testing grammar with quantifiers...\n");

    auto * grammar = build_grammar("root ::= \"a\"{2,4}");
    assert(grammar != nullptr);

    assert(!dfa_match_string("a", grammar));
    assert(dfa_match_string("aa", grammar));
    assert(dfa_match_string("aaa", grammar));
    assert(dfa_match_string("aaaa", grammar));
    assert(!dfa_match_string("aaaaa", grammar));
    assert(!dfa_match_string("", grammar));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ Quantifier grammar DFA tests passed\n");
}

static void test_grammar_wildcard() {
    fprintf(stderr, "Testing grammar with wildcard...\n");

    auto * grammar = build_grammar("root ::= .+");
    assert(grammar != nullptr);

    assert(dfa_match_string("a", grammar));
    assert(dfa_match_string("hello", grammar));
    assert(dfa_match_string("αβγ", grammar));
    assert(dfa_match_string("hello world 123", grammar));
    assert(!dfa_match_string("", grammar));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ Wildcard grammar DFA tests passed\n");
}

static void test_valid_bytes() {
    fprintf(stderr, "Testing get_valid_bytes...\n");

    auto * grammar = build_grammar("root ::= [a-c]");
    assert(grammar != nullptr);
    assert(grammar->compiled != nullptr);

    auto valid = grammar->compiled->get_valid_bytes(grammar->configs);

    assert(valid.test('a'));
    assert(valid.test('b'));
    assert(valid.test('c'));
    assert(!valid.test('d'));
    assert(!valid.test('z'));
    assert(!valid.test('0'));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ get_valid_bytes tests passed\n");
}

static void test_grammar_clone() {
    fprintf(stderr, "Testing grammar clone with DFA...\n");

    auto * grammar = build_grammar("root ::= \"hello\"");
    assert(grammar != nullptr);

    // Accept first two bytes
    grammar->compiled->accept_byte(grammar->configs, 'h');
    grammar->compiled->accept_byte(grammar->configs, 'e');

    // Clone
    auto * cloned = llama_grammar_clone_impl(*grammar);
    assert(cloned != nullptr);
    assert(cloned->compiled != nullptr);

    // Continue original
    grammar->compiled->accept_byte(grammar->configs, 'l');
    grammar->compiled->accept_byte(grammar->configs, 'l');
    grammar->compiled->accept_byte(grammar->configs, 'o');
    assert(grammar->compiled->any_config_complete(grammar->configs));

    // Continue clone
    cloned->compiled->accept_byte(cloned->configs, 'l');
    cloned->compiled->accept_byte(cloned->configs, 'l');
    cloned->compiled->accept_byte(cloned->configs, 'o');
    assert(cloned->compiled->any_config_complete(cloned->configs));

    llama_grammar_free_impl(grammar);
    llama_grammar_free_impl(cloned);
    fprintf(stderr, "  ✅ Grammar clone DFA tests passed\n");
}

// ============================================================
// DFA intersection tests
// ============================================================

static void test_dfa_intersection() {
    fprintf(stderr, "Testing DFA intersection...\n");

    // Intersect [a-z] with [m-z] → should give [m-z]
    auto nfa1 = codepoint_range_to_byte_nfa('a', 'z');
    auto dfa1 = byte_dfa::from_nfa(nfa1);
    dfa1.minimize();

    auto nfa2 = codepoint_range_to_byte_nfa('m', 'z');
    auto dfa2 = byte_dfa::from_nfa(nfa2);
    dfa2.minimize();

    auto inter = byte_dfa::intersect(dfa1, dfa2);
    inter.minimize();

    assert(dfa_accepts(inter, "m"));
    assert(dfa_accepts(inter, "z"));
    assert(!dfa_accepts(inter, "a"));
    assert(!dfa_accepts(inter, "l"));
    assert(!dfa_accepts(inter, "A"));
    assert(!dfa_accepts(inter, ""));

    fprintf(stderr, "  ✅ DFA intersection tests passed\n");
}

static void test_dfa_intersection_disjoint() {
    fprintf(stderr, "Testing DFA intersection of disjoint sets...\n");

    // Intersect [a-c] with [x-z] → should accept nothing
    auto nfa1 = codepoint_range_to_byte_nfa('a', 'c');
    auto dfa1 = byte_dfa::from_nfa(nfa1);
    dfa1.minimize();

    auto nfa2 = codepoint_range_to_byte_nfa('x', 'z');
    auto dfa2 = byte_dfa::from_nfa(nfa2);
    dfa2.minimize();

    auto inter = byte_dfa::intersect(dfa1, dfa2);
    inter.minimize();

    assert(!dfa_accepts(inter, "a"));
    assert(!dfa_accepts(inter, "x"));
    assert(!dfa_accepts(inter, "z"));
    assert(!dfa_accepts(inter, ""));

    fprintf(stderr, "  ✅ DFA intersection disjoint tests passed\n");
}

static void test_dfa_intersection_multibyte() {
    fprintf(stderr, "Testing DFA intersection with multi-byte ranges...\n");

    // Intersect [α-ω] (U+03B1-U+03C9) with [β-ψ] (U+03B2-U+03C8)
    auto nfa1 = codepoint_range_to_byte_nfa(0x03B1, 0x03C9);
    auto dfa1 = byte_dfa::from_nfa(nfa1);
    dfa1.minimize();

    auto nfa2 = codepoint_range_to_byte_nfa(0x03B2, 0x03C8);
    auto dfa2 = byte_dfa::from_nfa(nfa2);
    dfa2.minimize();

    auto inter = byte_dfa::intersect(dfa1, dfa2);
    inter.minimize();

    // β (U+03B2 = CE B2) should be in intersection
    assert(dfa_accepts_bytes(inter, {0xCE, 0xB2}));
    // ψ (U+03C8 = CF 88) should be in intersection
    assert(dfa_accepts_bytes(inter, {0xCF, 0x88}));
    // α (U+03B1 = CE B1) should NOT be in intersection (not in second range)
    assert(!dfa_accepts_bytes(inter, {0xCE, 0xB1}));
    // ω (U+03C9 = CF 89) should NOT be in intersection (not in second range)
    assert(!dfa_accepts_bytes(inter, {0xCF, 0x89}));

    fprintf(stderr, "  ✅ DFA intersection multi-byte tests passed\n");
}

// ============================================================
// DFA minimization verification tests
// ============================================================

static void test_dfa_minimization_reduces_states() {
    fprintf(stderr, "Testing DFA minimization reduces state count...\n");

    // Build a DFA from alternation of individual chars a|b|c — NFA→DFA may
    // create redundant states; minimization should reduce them
    auto a = codepoint_range_to_byte_nfa('a', 'a');
    auto b = codepoint_range_to_byte_nfa('b', 'b');
    auto c = codepoint_range_to_byte_nfa('c', 'c');
    auto combined = grammar_nfa::alternation({std::move(a), std::move(b), std::move(c)});
    auto dfa = byte_dfa::from_nfa(combined);

    size_t before = dfa.transitions.size();
    dfa.minimize();
    size_t after = dfa.transitions.size();

    // Minimized DFA for [a-c] should need only: dead(0), start(1), accept(2) = 3 states
    // The unminimized version from alternation of 3 individual chars will have more
    assert(after <= before);
    assert(after <= 3);  // dead + start + accept

    // Verify it still works
    assert(dfa_accepts(dfa, "a"));
    assert(dfa_accepts(dfa, "b"));
    assert(dfa_accepts(dfa, "c"));
    assert(!dfa_accepts(dfa, "d"));
    assert(!dfa_accepts(dfa, ""));

    fprintf(stderr, "  ✅ DFA minimization state reduction tests passed (before=%zu, after=%zu)\n",
            before, after);
}

static void test_dfa_minimization_preserves_behavior() {
    fprintf(stderr, "Testing DFA minimization preserves matching behavior...\n");

    // Build a more complex DFA and verify identical behavior before/after minimization
    // Use "ab" | "ac" — creates shared prefix that minimization should handle
    auto ab_nfa = grammar_nfa::concat(
        grammar_nfa::byte_match('a', 'a'),
        grammar_nfa::byte_match('b', 'b'));
    auto ac_nfa = grammar_nfa::concat(
        grammar_nfa::byte_match('a', 'a'),
        grammar_nfa::byte_match('c', 'c'));
    auto combined = grammar_nfa::alternation({std::move(ab_nfa), std::move(ac_nfa)});
    auto dfa = byte_dfa::from_nfa(combined);

    // Test before minimization
    assert(dfa_accepts(dfa, "ab"));
    assert(dfa_accepts(dfa, "ac"));
    assert(!dfa_accepts(dfa, "a"));
    assert(!dfa_accepts(dfa, "ad"));
    assert(!dfa_accepts(dfa, "b"));
    assert(!dfa_accepts(dfa, "abc"));

    dfa.minimize();

    // Test after minimization — must be identical
    assert(dfa_accepts(dfa, "ab"));
    assert(dfa_accepts(dfa, "ac"));
    assert(!dfa_accepts(dfa, "a"));
    assert(!dfa_accepts(dfa, "ad"));
    assert(!dfa_accepts(dfa, "b"));
    assert(!dfa_accepts(dfa, "abc"));

    fprintf(stderr, "  ✅ DFA minimization behavior preservation tests passed\n");
}

// ============================================================
// Cross-boundary UTF-8 range tests
// ============================================================

static void test_dfa_3byte_range() {
    fprintf(stderr, "Testing DFA for 3-byte UTF-8 range [€-₿] (U+20AC-U+20BF)...\n");

    auto nfa = codepoint_range_to_byte_nfa(0x20AC, 0x20BF);
    auto dfa = byte_dfa::from_nfa(nfa);
    dfa.minimize();

    // € = U+20AC = E2 82 AC
    assert(dfa_accepts_bytes(dfa, {0xE2, 0x82, 0xAC}));
    // ₿ = U+20BF = E2 82 BF
    assert(dfa_accepts_bytes(dfa, {0xE2, 0x82, 0xBF}));
    // U+20B0 = E2 82 B0 (in range)
    assert(dfa_accepts_bytes(dfa, {0xE2, 0x82, 0xB0}));
    // U+20AB = E2 82 AB (just below range)
    assert(!dfa_accepts_bytes(dfa, {0xE2, 0x82, 0xAB}));
    // U+20C0 = E2 83 80 (just above range)
    assert(!dfa_accepts_bytes(dfa, {0xE2, 0x83, 0x80}));
    // ASCII 'a'
    assert(!dfa_accepts_bytes(dfa, {0x61}));

    fprintf(stderr, "  ✅ 3-byte UTF-8 range DFA tests passed\n");
}

static void test_dfa_4byte_range() {
    fprintf(stderr, "Testing DFA for 4-byte UTF-8 range [𝄞-𝄡] (U+1D11E-U+1D121)...\n");

    // Musical symbols: 𝄞 (treble clef) through 𝄡
    auto nfa = codepoint_range_to_byte_nfa(0x1D11E, 0x1D121);
    auto dfa = byte_dfa::from_nfa(nfa);
    dfa.minimize();

    // 𝄞 = U+1D11E = F0 9D 84 9E
    assert(dfa_accepts_bytes(dfa, {0xF0, 0x9D, 0x84, 0x9E}));
    // U+1D11F = F0 9D 84 9F
    assert(dfa_accepts_bytes(dfa, {0xF0, 0x9D, 0x84, 0x9F}));
    // U+1D121 = F0 9D 84 A1
    assert(dfa_accepts_bytes(dfa, {0xF0, 0x9D, 0x84, 0xA1}));
    // U+1D11D = F0 9D 84 9D (just below)
    assert(!dfa_accepts_bytes(dfa, {0xF0, 0x9D, 0x84, 0x9D}));
    // U+1D122 = F0 9D 84 A2 (just above)
    assert(!dfa_accepts_bytes(dfa, {0xF0, 0x9D, 0x84, 0xA2}));

    fprintf(stderr, "  ✅ 4-byte UTF-8 range DFA tests passed\n");
}

static void test_dfa_cross_utf8_length_boundary() {
    fprintf(stderr, "Testing DFA for cross-boundary range [\\x7E-\\xA0] (U+007E-U+00A0)...\n");

    // Range that crosses 1-byte → 2-byte UTF-8 boundary
    // U+007E = 0x7E (1-byte), U+00A0 = C2 A0 (2-byte)
    auto nfa = codepoint_range_to_byte_nfa(0x7E, 0xA0);
    auto dfa = byte_dfa::from_nfa(nfa);
    dfa.minimize();

    // 1-byte: ~ (U+007E)
    assert(dfa_accepts_bytes(dfa, {0x7E}));
    // 1-byte: DEL (U+007F)
    assert(dfa_accepts_bytes(dfa, {0x7F}));
    // 2-byte: U+0080 = C2 80
    assert(dfa_accepts_bytes(dfa, {0xC2, 0x80}));
    // 2-byte: U+00A0 = C2 A0
    assert(dfa_accepts_bytes(dfa, {0xC2, 0xA0}));
    // Below range: U+007D
    assert(!dfa_accepts_bytes(dfa, {0x7D}));
    // Above range: U+00A1 = C2 A1
    assert(!dfa_accepts_bytes(dfa, {0xC2, 0xA1}));

    fprintf(stderr, "  ✅ Cross UTF-8 length boundary range tests passed\n");
}

static void test_dfa_cross_2byte_3byte_boundary() {
    fprintf(stderr, "Testing DFA for range crossing 2-byte→3-byte boundary (U+07FE-U+0802)...\n");

    // U+07FE: 2-byte (DF BE), U+0800: 3-byte (E0 A0 80)
    auto nfa = codepoint_range_to_byte_nfa(0x07FE, 0x0802);
    auto dfa = byte_dfa::from_nfa(nfa);
    dfa.minimize();

    // U+07FE = DF BE (2-byte)
    assert(dfa_accepts_bytes(dfa, {0xDF, 0xBE}));
    // U+07FF = DF BF (2-byte, last 2-byte char)
    assert(dfa_accepts_bytes(dfa, {0xDF, 0xBF}));
    // U+0800 = E0 A0 80 (3-byte, first 3-byte char)
    assert(dfa_accepts_bytes(dfa, {0xE0, 0xA0, 0x80}));
    // U+0801 = E0 A0 81
    assert(dfa_accepts_bytes(dfa, {0xE0, 0xA0, 0x81}));
    // U+0802 = E0 A0 82
    assert(dfa_accepts_bytes(dfa, {0xE0, 0xA0, 0x82}));
    // U+07FD = DF BD (below range)
    assert(!dfa_accepts_bytes(dfa, {0xDF, 0xBD}));
    // U+0803 = E0 A0 83 (above range)
    assert(!dfa_accepts_bytes(dfa, {0xE0, 0xA0, 0x83}));

    fprintf(stderr, "  ✅ Cross 2-byte→3-byte boundary range tests passed\n");
}

// ============================================================
// Segment DFA tests: CHAR_ALT, CHAR_ANY, multi-element
// ============================================================

static void test_segment_dfa_char_alt() {
    fprintf(stderr, "Testing segment DFA for char class with alternatives [abc]...\n");

    // GBNF: [abc] is parsed as CHAR 'a', CHAR_ALT 'b', CHAR_ALT 'c'
    llama_grammar_element elements[] = {
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_CHAR_ALT, 'b'},
        {LLAMA_GRETYPE_CHAR_ALT, 'c'},
        {LLAMA_GRETYPE_END, 0},
    };

    auto dfa = build_segment_dfa(elements, elements + 3);

    assert(dfa_accepts(dfa, "a"));
    assert(dfa_accepts(dfa, "b"));
    assert(dfa_accepts(dfa, "c"));
    assert(!dfa_accepts(dfa, "d"));
    assert(!dfa_accepts(dfa, "ab"));
    assert(!dfa_accepts(dfa, ""));

    fprintf(stderr, "  ✅ Char alt segment DFA tests passed\n");
}

static void test_segment_dfa_char_any() {
    fprintf(stderr, "Testing segment DFA for wildcard (CHAR_ANY)...\n");

    llama_grammar_element elements[] = {
        {LLAMA_GRETYPE_CHAR_ANY, 0},
        {LLAMA_GRETYPE_END, 0},
    };

    auto dfa = build_segment_dfa(elements, elements + 1);

    // Single ASCII
    assert(dfa_accepts(dfa, "a"));
    assert(dfa_accepts(dfa, "Z"));
    assert(dfa_accepts(dfa, "0"));
    // 2-byte UTF-8
    assert(dfa_accepts_bytes(dfa, {0xCE, 0xB1}));  // α
    // 3-byte UTF-8
    assert(dfa_accepts_bytes(dfa, {0xE2, 0x82, 0xAC}));  // €
    // Should not accept empty
    assert(!dfa_accepts(dfa, ""));
    // Should not accept two chars
    assert(!dfa_accepts(dfa, "ab"));

    fprintf(stderr, "  ✅ Wildcard segment DFA tests passed\n");
}

static void test_segment_dfa_multi_element() {
    fprintf(stderr, "Testing segment DFA for multi-element sequence [a-z][0-9]...\n");

    llama_grammar_element elements[] = {
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, 'z'},
        {LLAMA_GRETYPE_CHAR, '0'},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, '9'},
        {LLAMA_GRETYPE_END, 0},
    };

    auto dfa = build_segment_dfa(elements, elements + 4);

    assert(dfa_accepts(dfa, "a0"));
    assert(dfa_accepts(dfa, "z9"));
    assert(dfa_accepts(dfa, "m5"));
    assert(!dfa_accepts(dfa, "a"));
    assert(!dfa_accepts(dfa, "0a"));
    assert(!dfa_accepts(dfa, "A0"));
    assert(!dfa_accepts(dfa, "a0b"));
    assert(!dfa_accepts(dfa, ""));

    fprintf(stderr, "  ✅ Multi-element segment DFA tests passed\n");
}

static void test_segment_dfa_negated_range() {
    fprintf(stderr, "Testing segment DFA for negated range [^a-z]...\n");

    llama_grammar_element elements[] = {
        {LLAMA_GRETYPE_CHAR_NOT, 'a'},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, 'z'},
        {LLAMA_GRETYPE_END, 0},
    };

    auto dfa = build_segment_dfa(elements, elements + 2);

    assert(!dfa_accepts(dfa, "a"));
    assert(!dfa_accepts(dfa, "m"));
    assert(!dfa_accepts(dfa, "z"));
    assert(dfa_accepts(dfa, "A"));
    assert(dfa_accepts(dfa, "0"));
    assert(dfa_accepts(dfa, "!"));
    // Multi-byte should be accepted (not in [a-z])
    assert(dfa_accepts_bytes(dfa, {0xCE, 0xB1}));  // α
    assert(!dfa_accepts(dfa, ""));

    fprintf(stderr, "  ✅ Negated range segment DFA tests passed\n");
}

static void test_segment_dfa_negated_with_alt() {
    fprintf(stderr, "Testing segment DFA for [^ab]...\n");

    // GBNF: [^ab] → CHAR_NOT 'a', CHAR_ALT 'b'
    llama_grammar_element elements[] = {
        {LLAMA_GRETYPE_CHAR_NOT, 'a'},
        {LLAMA_GRETYPE_CHAR_ALT, 'b'},
        {LLAMA_GRETYPE_END, 0},
    };

    auto dfa = build_segment_dfa(elements, elements + 2);

    assert(!dfa_accepts(dfa, "a"));
    assert(!dfa_accepts(dfa, "b"));
    assert(dfa_accepts(dfa, "c"));
    assert(dfa_accepts(dfa, "Z"));
    assert(dfa_accepts(dfa, "0"));
    assert(!dfa_accepts(dfa, ""));

    fprintf(stderr, "  ✅ Negated with alt segment DFA tests passed\n");
}

// ============================================================
// Grammar recursion and nested rule tests
// ============================================================

static void test_grammar_recursive() {
    fprintf(stderr, "Testing recursive grammar (nested parens)...\n");

    auto * grammar = build_grammar(
        "root ::= \"(\" inner \")\"\n"
        "inner ::= [a-z]+ | root");
    assert(grammar != nullptr);

    assert(dfa_match_string("(abc)", grammar));
    assert(dfa_match_string("((abc))", grammar));
    assert(dfa_match_string("(((x)))", grammar));
    assert(!dfa_match_string("()", grammar));
    assert(!dfa_match_string("(abc", grammar));
    assert(!dfa_match_string("abc)", grammar));
    assert(!dfa_match_string("", grammar));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ Recursive grammar tests passed\n");
}

static void test_grammar_deep_nesting() {
    fprintf(stderr, "Testing deep rule nesting...\n");

    auto * grammar = build_grammar(
        "root ::= a\n"
        "a ::= b\n"
        "b ::= c\n"
        "c ::= \"hello\"");
    assert(grammar != nullptr);

    assert(dfa_match_string("hello", grammar));
    assert(!dfa_match_string("hell", grammar));
    assert(!dfa_match_string("", grammar));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ Deep nesting grammar tests passed\n");
}

static void test_grammar_star() {
    fprintf(stderr, "Testing grammar with * (zero or more)...\n");

    auto * grammar = build_grammar("root ::= [a-z]*");
    assert(grammar != nullptr);

    assert(dfa_match_string("", grammar));
    assert(dfa_match_string("a", grammar));
    assert(dfa_match_string("abc", grammar));
    assert(dfa_match_string("hello", grammar));
    assert(!dfa_match_string("Hello", grammar));
    assert(!dfa_match_string("abc123", grammar));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ Star grammar tests passed\n");
}

static void test_grammar_complex_json_number() {
    fprintf(stderr, "Testing JSON number grammar...\n");

    auto * grammar = build_grammar(
        "root ::= integer fraction?\n"
        "integer ::= \"-\"? (\"0\" | [1-9] [0-9]*)\n"
        "fraction ::= \".\" [0-9]+");
    assert(grammar != nullptr);

    assert(dfa_match_string("0", grammar));
    assert(dfa_match_string("42", grammar));
    assert(dfa_match_string("-1", grammar));
    assert(dfa_match_string("3.14", grammar));
    assert(dfa_match_string("-0.5", grammar));
    assert(dfa_match_string("100", grammar));
    assert(!dfa_match_string("01", grammar));     // leading zero
    assert(!dfa_match_string(".", grammar));
    assert(!dfa_match_string("3.", grammar));      // trailing dot
    assert(!dfa_match_string("", grammar));
    assert(!dfa_match_string("abc", grammar));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ JSON number grammar tests passed\n");
}

static void test_grammar_json_array() {
    fprintf(stderr, "Testing JSON array grammar...\n");

    auto * grammar = build_grammar(
        "root ::= \"[\" ws items? ws \"]\"\n"
        "items ::= item (ws \",\" ws item)*\n"
        "item ::= [0-9]+\n"
        "ws ::= \" \"*");
    assert(grammar != nullptr);

    assert(dfa_match_string("[]", grammar));
    assert(dfa_match_string("[1]", grammar));
    assert(dfa_match_string("[1,2,3]", grammar));
    assert(dfa_match_string("[1 , 2 , 3]", grammar));
    assert(dfa_match_string("[ 42 ]", grammar));
    assert(!dfa_match_string("[", grammar));
    assert(!dfa_match_string("[1,]", grammar));
    assert(!dfa_match_string("", grammar));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ JSON array grammar tests passed\n");
}

static void test_grammar_json_object() {
    fprintf(stderr, "Testing JSON key-value grammar...\n");

    auto * grammar = build_grammar(
        "root ::= \"{\" ws pair (ws \",\" ws pair)* ws \"}\"\n"
        "pair ::= key ws \":\" ws value\n"
        "key ::= \"\\\"\" [a-z]+ \"\\\"\"\n"
        "value ::= \"\\\"\" [a-z]+ \"\\\"\" | [0-9]+\n"
        "ws ::= \" \"*");
    assert(grammar != nullptr);

    assert(dfa_match_string("{\"a\":\"b\"}", grammar));
    assert(dfa_match_string("{\"key\":42}", grammar));
    assert(dfa_match_string("{\"x\":1 , \"y\":2}", grammar));
    assert(!dfa_match_string("{}", grammar));       // needs at least one pair
    assert(!dfa_match_string("{\"a\"}", grammar));  // missing value
    assert(!dfa_match_string("", grammar));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ JSON key-value grammar tests passed\n");
}

static void test_grammar_multiple_alternates_in_rule() {
    fprintf(stderr, "Testing grammar with many alternates...\n");

    auto * grammar = build_grammar(
        "root ::= \"red\" | \"green\" | \"blue\" | \"yellow\" | \"purple\"");
    assert(grammar != nullptr);

    assert(dfa_match_string("red", grammar));
    assert(dfa_match_string("green", grammar));
    assert(dfa_match_string("blue", grammar));
    assert(dfa_match_string("yellow", grammar));
    assert(dfa_match_string("purple", grammar));
    assert(!dfa_match_string("orange", grammar));
    assert(!dfa_match_string("re", grammar));
    assert(!dfa_match_string("", grammar));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ Multiple alternates grammar tests passed\n");
}

// ============================================================
// accept_byte and config lifecycle tests
// ============================================================

static void test_accept_byte_configs_die() {
    fprintf(stderr, "Testing accept_byte: configs die on invalid input...\n");

    auto * grammar = build_grammar("root ::= \"abc\"");
    assert(grammar != nullptr);

    auto configs = grammar->configs;
    assert(!configs.empty());

    // Feed 'a' — should survive
    grammar->compiled->accept_byte(configs, 'a');
    assert(!configs.empty());

    // Feed 'x' — should die (expected 'b')
    grammar->compiled->accept_byte(configs, 'x');
    assert(configs.empty());

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ accept_byte config death tests passed\n");
}

static void test_accept_byte_complete_detection() {
    fprintf(stderr, "Testing accept_byte: complete detection...\n");

    auto * grammar = build_grammar("root ::= \"ab\"");
    assert(grammar != nullptr);

    auto configs = grammar->configs;

    // After 'a' — not complete
    grammar->compiled->accept_byte(configs, 'a');
    assert(!configs.empty());
    assert(!grammar->compiled->any_config_complete(configs));

    // After 'b' — complete
    grammar->compiled->accept_byte(configs, 'b');
    assert(!configs.empty());
    assert(grammar->compiled->any_config_complete(configs));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ accept_byte complete detection tests passed\n");
}

static void test_accept_byte_alternation_configs() {
    fprintf(stderr, "Testing accept_byte: alternation generates multiple configs...\n");

    // Grammar: "ab" | "ac" — after 'a', both alternates should still be alive
    auto * grammar = build_grammar("root ::= \"ab\" | \"ac\"");
    assert(grammar != nullptr);

    auto configs = grammar->configs;

    grammar->compiled->accept_byte(configs, 'a');
    assert(!configs.empty());
    // Should have configs for both alternates (at least 2, possibly deduplicated)
    // After 'a', we need 'b' or 'c' to continue

    // Feed 'b' — only "ab" path survives
    auto configs_b = configs;
    grammar->compiled->accept_byte(configs_b, 'b');
    assert(!configs_b.empty());
    assert(grammar->compiled->any_config_complete(configs_b));

    // Feed 'c' — only "ac" path survives
    auto configs_c = configs;
    grammar->compiled->accept_byte(configs_c, 'c');
    assert(!configs_c.empty());
    assert(grammar->compiled->any_config_complete(configs_c));

    // Feed 'd' — nothing survives
    auto configs_d = configs;
    grammar->compiled->accept_byte(configs_d, 'd');
    assert(configs_d.empty());

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ accept_byte alternation config tests passed\n");
}

static void test_accept_byte_utf8_multibyte() {
    fprintf(stderr, "Testing accept_byte with multi-byte UTF-8...\n");

    auto * grammar = build_grammar("root ::= [α-ω]");
    assert(grammar != nullptr);

    // α = CE B1
    auto configs = grammar->configs;
    grammar->compiled->accept_byte(configs, 0xCE);
    assert(!configs.empty());
    // Should not be complete yet (mid-character)
    assert(!grammar->compiled->any_config_complete(configs));

    grammar->compiled->accept_byte(configs, 0xB1);
    assert(!configs.empty());
    assert(grammar->compiled->any_config_complete(configs));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ accept_byte multi-byte UTF-8 tests passed\n");
}

// ============================================================
// get_valid_bytes extended tests
// ============================================================

static void test_valid_bytes_multibyte() {
    fprintf(stderr, "Testing get_valid_bytes with multi-byte UTF-8...\n");

    auto * grammar = build_grammar("root ::= [α-ω]");
    assert(grammar != nullptr);

    auto valid = grammar->compiled->get_valid_bytes(grammar->configs);

    // Should accept lead bytes for 2-byte Greek letters
    assert(valid.test(0xCE));  // α-ξ lead byte
    assert(valid.test(0xCF));  // ο-ω lead byte
    // Should NOT accept ASCII
    assert(!valid.test('a'));
    assert(!valid.test('z'));
    assert(!valid.test('0'));

    // After feeding the lead byte CE, valid bytes should be continuations
    auto configs = grammar->configs;
    grammar->compiled->accept_byte(configs, 0xCE);
    auto valid2 = grammar->compiled->get_valid_bytes(configs);
    // Should accept continuation bytes in range
    assert(valid2.test(0xB1));  // α continuation
    assert(valid2.test(0xBF));  // ξ continuation
    // Should NOT accept ASCII or lead bytes
    assert(!valid2.test('a'));
    assert(!valid2.test(0xCE));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ get_valid_bytes multi-byte tests passed\n");
}

static void test_valid_bytes_alternation() {
    fprintf(stderr, "Testing get_valid_bytes with alternation...\n");

    auto * grammar = build_grammar("root ::= \"abc\" | \"xyz\"");
    assert(grammar != nullptr);

    auto valid = grammar->compiled->get_valid_bytes(grammar->configs);
    // Should accept 'a' and 'x'
    assert(valid.test('a'));
    assert(valid.test('x'));
    assert(!valid.test('b'));
    assert(!valid.test('z'));

    // After 'a', should only accept 'b'
    auto configs = grammar->configs;
    grammar->compiled->accept_byte(configs, 'a');
    auto valid2 = grammar->compiled->get_valid_bytes(configs);
    assert(valid2.test('b'));
    assert(!valid2.test('x'));
    assert(!valid2.test('a'));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ get_valid_bytes alternation tests passed\n");
}

static void test_valid_bytes_complete_grammar() {
    fprintf(stderr, "Testing get_valid_bytes when grammar is complete...\n");

    auto * grammar = build_grammar("root ::= \"a\"");
    assert(grammar != nullptr);

    auto configs = grammar->configs;
    grammar->compiled->accept_byte(configs, 'a');
    assert(grammar->compiled->any_config_complete(configs));

    // When complete, no more bytes should be valid
    auto valid = grammar->compiled->get_valid_bytes(configs);
    assert(valid.none());

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ get_valid_bytes complete grammar tests passed\n");
}

// ============================================================
// Edge case tests
// ============================================================

static void test_dfa_single_char() {
    fprintf(stderr, "Testing DFA for single character 'x'...\n");

    auto nfa = codepoint_range_to_byte_nfa('x', 'x');
    auto dfa = byte_dfa::from_nfa(nfa);
    dfa.minimize();

    assert(dfa_accepts(dfa, "x"));
    assert(!dfa_accepts(dfa, "y"));
    assert(!dfa_accepts(dfa, ""));
    assert(!dfa_accepts(dfa, "xx"));

    // Should have exactly 3 states: dead, start, accept
    assert(dfa.transitions.size() == 3);

    fprintf(stderr, "  ✅ Single char DFA tests passed\n");
}

static void test_dfa_full_ascii_range() {
    fprintf(stderr, "Testing DFA for full ASCII range [\\x00-\\x7F]...\n");

    auto nfa = codepoint_range_to_byte_nfa(0x00, 0x7F);
    auto dfa = byte_dfa::from_nfa(nfa);
    dfa.minimize();

    assert(dfa_accepts_bytes(dfa, {0x00}));
    assert(dfa_accepts_bytes(dfa, {0x41}));  // 'A'
    assert(dfa_accepts_bytes(dfa, {0x7F}));
    assert(!dfa_accepts_bytes(dfa, {0x80}));  // continuation byte
    assert(!dfa_accepts(dfa, ""));
    assert(!dfa_accepts(dfa, "ab"));

    fprintf(stderr, "  ✅ Full ASCII range DFA tests passed\n");
}

static void test_grammar_long_string() {
    fprintf(stderr, "Testing grammar with longer string matching...\n");

    auto * grammar = build_grammar("root ::= [a-z]+");
    assert(grammar != nullptr);

    // 100 character string
    std::string long_str(100, 'a');
    assert(dfa_match_string(long_str, grammar));

    // Mixed valid
    std::string mixed(50, 'x');
    mixed += std::string(50, 'y');
    assert(dfa_match_string(mixed, grammar));

    // Fails at the end
    std::string almost = std::string(99, 'a') + "1";
    assert(!dfa_match_string(almost, grammar));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ Long string grammar tests passed\n");
}

static void test_grammar_exact_quantifiers() {
    fprintf(stderr, "Testing grammar with exact quantifiers...\n");

    // Exact count
    auto * grammar = build_grammar("root ::= \"a\"{3}");
    assert(grammar != nullptr);

    assert(!dfa_match_string("aa", grammar));
    assert(dfa_match_string("aaa", grammar));
    assert(!dfa_match_string("aaaa", grammar));

    llama_grammar_free_impl(grammar);

    // Min only: {2,}
    grammar = build_grammar("root ::= \"a\"{2,}");
    assert(grammar != nullptr);

    assert(!dfa_match_string("a", grammar));
    assert(dfa_match_string("aa", grammar));
    assert(dfa_match_string("aaa", grammar));
    assert(dfa_match_string("aaaaaaa", grammar));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ Exact quantifier grammar tests passed\n");
}

static void test_grammar_mixed_literals_and_classes() {
    fprintf(stderr, "Testing grammar mixing literals and character classes...\n");

    auto * grammar = build_grammar(
        "root ::= \"id:\" [0-9]+ \",name:\" [a-zA-Z]+");
    assert(grammar != nullptr);

    assert(dfa_match_string("id:42,name:Alice", grammar));
    assert(dfa_match_string("id:0,name:x", grammar));
    assert(!dfa_match_string("id:,name:Alice", grammar));  // no digits
    assert(!dfa_match_string("id:42,name:", grammar));      // no letters
    assert(!dfa_match_string("id:42name:Alice", grammar));  // missing comma
    assert(!dfa_match_string("", grammar));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ Mixed literals and classes grammar tests passed\n");
}

static void test_grammar_complement_negated_range() {
    fprintf(stderr, "Testing grammar with negated ranges [^0-9]+...\n");

    auto * grammar = build_grammar("root ::= [^0-9]+");
    assert(grammar != nullptr);

    assert(dfa_match_string("hello", grammar));
    assert(dfa_match_string("abc", grammar));
    assert(dfa_match_string("!@#", grammar));
    assert(!dfa_match_string("123", grammar));
    assert(!dfa_match_string("abc123", grammar));
    assert(!dfa_match_string("", grammar));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ Negated range grammar tests passed\n");
}

static void test_grammar_empty_rule() {
    fprintf(stderr, "Testing grammar with optional entire match...\n");

    auto * grammar = build_grammar("root ::= \"a\" | \"\"");
    assert(grammar != nullptr);

    assert(dfa_match_string("a", grammar));
    assert(dfa_match_string("", grammar));
    assert(!dfa_match_string("b", grammar));
    assert(!dfa_match_string("aa", grammar));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ Empty rule grammar tests passed\n");
}

static void test_grammar_compiled_structure() {
    fprintf(stderr, "Testing compiled grammar internal structure...\n");

    auto * grammar = build_grammar(
        "root ::= greeting name\n"
        "greeting ::= \"hi\"\n"
        "name ::= \"Bob\"");
    assert(grammar != nullptr);
    assert(grammar->compiled != nullptr);

    // Should have compiled rules
    assert(grammar->compiled->rules.size() >= 3);

    // Should have DFAs for terminal segments
    assert(!grammar->compiled->dfas.empty());

    // Each DFA should have a valid start state
    for (const auto & dfa : grammar->compiled->dfas) {
        assert(dfa.start_state > 0);  // not dead state
        assert(dfa.start_state < dfa.transitions.size());
    }

    // Verify it works end-to-end
    assert(dfa_match_string("hiBob", grammar));
    assert(!dfa_match_string("hi", grammar));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ Compiled grammar structure tests passed\n");
}

static void test_grammar_init_configs() {
    fprintf(stderr, "Testing init_configs produces valid configurations...\n");

    auto * grammar = build_grammar("root ::= \"hello\" | \"hi\"");
    assert(grammar != nullptr);

    // configs should have been initialized
    assert(!grammar->configs.empty());

    // Each config should have valid state
    for (const auto & cfg : grammar->configs) {
        assert(cfg.current.rule_id == grammar->compiled->start_rule);
        assert(cfg.current.dfa_state > 0);  // should be at DFA start state, not dead
    }

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ init_configs tests passed\n");
}

static void test_nfa_merge() {
    fprintf(stderr, "Testing NFA merge adjusts state indices...\n");

    auto a = grammar_nfa::byte_match('a', 'a');
    auto b = grammar_nfa::byte_match('b', 'b');

    size_t a_size = a.states.size();
    uint32_t offset = a.merge(b);

    // Offset should equal original size of a
    assert(offset == a_size);
    // Total states should be sum
    assert(a.states.size() == a_size + b.states.size());

    // Check that merged transitions reference adjusted indices
    for (size_t i = a_size; i < a.states.size(); i++) {
        for (const auto & t : a.states[i].byte_transitions) {
            assert(t.target >= offset || t.target < a_size);
        }
    }

    fprintf(stderr, "  ✅ NFA merge tests passed\n");
}

static void test_dfa_complement_double() {
    fprintf(stderr, "Testing double complement returns original behavior...\n");

    auto nfa = codepoint_range_to_byte_nfa('a', 'z');
    auto dfa = byte_dfa::from_nfa(nfa);
    dfa.minimize();

    // Record original behavior
    bool orig_a = dfa_accepts(dfa, "a");
    bool orig_z = dfa_accepts(dfa, "z");
    bool orig_A = dfa_accepts(dfa, "A");
    bool orig_empty = dfa_accepts(dfa, "");

    // Double complement
    dfa.complement();
    dfa.complement();
    dfa.minimize();

    assert(dfa_accepts(dfa, "a") == orig_a);
    assert(dfa_accepts(dfa, "z") == orig_z);
    assert(dfa_accepts(dfa, "A") == orig_A);
    assert(dfa_accepts(dfa, "") == orig_empty);

    fprintf(stderr, "  ✅ Double complement tests passed\n");
}

static void test_grammar_unicode_mixed() {
    fprintf(stderr, "Testing grammar mixing ASCII and Unicode...\n");

    auto * grammar = build_grammar("root ::= [a-zα-ω]+");
    assert(grammar != nullptr);

    assert(dfa_match_string("hello", grammar));
    assert(dfa_match_string("αβγ", grammar));
    assert(dfa_match_string("aαbβ", grammar));
    assert(!dfa_match_string("ABC", grammar));
    assert(!dfa_match_string("123", grammar));
    assert(!dfa_match_string("", grammar));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ Unicode mixed grammar tests passed\n");
}

static void test_grammar_whitespace_handling() {
    fprintf(stderr, "Testing grammar with whitespace patterns...\n");

    auto * grammar = build_grammar(
        "root ::= word (\" \" word)*\n"
        "word ::= [a-z]+");
    assert(grammar != nullptr);

    assert(dfa_match_string("hello", grammar));
    assert(dfa_match_string("hello world", grammar));
    assert(dfa_match_string("the quick brown fox", grammar));
    assert(!dfa_match_string("", grammar));
    assert(!dfa_match_string(" hello", grammar));   // leading space
    assert(!dfa_match_string("hello ", grammar));    // trailing space
    assert(!dfa_match_string("hello  world", grammar));  // double space

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ Whitespace handling grammar tests passed\n");
}

static void test_grammar_escape_sequences() {
    fprintf(stderr, "Testing grammar with escape sequences...\n");

    auto * grammar = build_grammar("root ::= \"\\n\" | \"\\t\" | \"\\\\\"");
    assert(grammar != nullptr);

    assert(dfa_match_string("\n", grammar));
    assert(dfa_match_string("\t", grammar));
    assert(dfa_match_string("\\", grammar));
    assert(!dfa_match_string("n", grammar));
    assert(!dfa_match_string("t", grammar));
    assert(!dfa_match_string("", grammar));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  ✅ Escape sequence grammar tests passed\n");
}

// ============================================================
// Main
// ============================================================

int main() {
    fprintf(stderr, "=== DFA Grammar Engine Unit Tests ===\n\n");

    // Low-level infrastructure tests
    test_utf8_encoding();
    test_nfa_construction();
    test_nfa_merge();
    test_dfa_ascii_range();
    test_dfa_multibyte_range();
    test_dfa_wildcard();
    test_dfa_complement();
    test_dfa_complement_double();
    test_dfa_mixed_class();
    test_dfa_single_char();
    test_dfa_full_ascii_range();

    // DFA intersection tests
    test_dfa_intersection();
    test_dfa_intersection_disjoint();
    test_dfa_intersection_multibyte();

    // DFA minimization tests
    test_dfa_minimization_reduces_states();
    test_dfa_minimization_preserves_behavior();

    // Cross-boundary UTF-8 range tests
    test_dfa_3byte_range();
    test_dfa_4byte_range();
    test_dfa_cross_utf8_length_boundary();
    test_dfa_cross_2byte_3byte_boundary();

    // Segment DFA tests
    test_segment_dfa_simple_string();
    test_segment_dfa_char_range();
    test_segment_dfa_negated();
    test_segment_dfa_char_alt();
    test_segment_dfa_char_any();
    test_segment_dfa_multi_element();
    test_segment_dfa_negated_range();
    test_segment_dfa_negated_with_alt();

    // Full grammar tests
    test_grammar_simple();
    test_grammar_alternation();
    test_grammar_char_class();
    test_grammar_repetition();
    test_grammar_optional();
    test_grammar_rule_ref();
    test_grammar_unicode();
    test_grammar_negation();
    test_grammar_json_string();
    test_grammar_quantifiers();
    test_grammar_wildcard();
    test_grammar_star();
    test_grammar_exact_quantifiers();
    test_grammar_recursive();
    test_grammar_deep_nesting();
    test_grammar_complex_json_number();
    test_grammar_json_array();
    test_grammar_json_object();
    test_grammar_multiple_alternates_in_rule();
    test_grammar_mixed_literals_and_classes();
    test_grammar_complement_negated_range();
    test_grammar_empty_rule();
    test_grammar_unicode_mixed();
    test_grammar_whitespace_handling();
    test_grammar_escape_sequences();
    test_grammar_long_string();

    // accept_byte and config lifecycle tests
    test_accept_byte_configs_die();
    test_accept_byte_complete_detection();
    test_accept_byte_alternation_configs();
    test_accept_byte_utf8_multibyte();

    // get_valid_bytes extended tests
    test_valid_bytes();
    test_valid_bytes_multibyte();
    test_valid_bytes_alternation();
    test_valid_bytes_complete_grammar();

    // Structure and clone tests
    test_grammar_compiled_structure();
    test_grammar_init_configs();
    test_grammar_clone();

    fprintf(stderr, "\n=== All DFA grammar tests passed! ===\n");
    return 0;
}
