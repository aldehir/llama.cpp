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
// Main
// ============================================================

int main() {
    fprintf(stderr, "=== DFA Grammar Engine Unit Tests ===\n\n");

    // Low-level infrastructure tests
    test_utf8_encoding();
    test_nfa_construction();
    test_dfa_ascii_range();
    test_dfa_multibyte_range();
    test_dfa_wildcard();
    test_dfa_complement();
    test_dfa_mixed_class();

    // Segment DFA tests
    test_segment_dfa_simple_string();
    test_segment_dfa_char_range();
    test_segment_dfa_negated();

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

    // Runtime feature tests
    test_valid_bytes();
    test_grammar_clone();

    fprintf(stderr, "\n=== All DFA grammar tests passed! ===\n");
    return 0;
}
