#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama.h"

#include "../src/llama-grammar.h"

#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>

static llama_grammar * build_grammar(const std::string & grammar_str) {
    return llama_grammar_init_impl(nullptr, grammar_str.c_str(), "root", false, nullptr, 0, nullptr, 0);
}

static bool match_string(const std::string & input, llama_grammar * grammar) {
    auto configs = grammar->configs;  // copy
    for (uint8_t b : input) {
        grammar->compiled->accept_byte(configs, b);
        if (configs.empty()) return false;
    }
    return grammar->compiled->any_config_complete(configs);
}

static void test_grammar_initialization() {
    fprintf(stderr, "Testing grammar initialization...\n");

    // Test that grammar initialization succeeds for a valid grammar
    auto * grammar = build_grammar(R"(root ::= "hello")");
    assert(grammar != nullptr);
    assert(grammar->compiled != nullptr);
    assert(!grammar->configs.empty());
    llama_grammar_free_impl(grammar);

    fprintf(stderr, "  Grammar initialization: OK\n");
}

static void test_simple_accept_reject() {
    fprintf(stderr, "Testing simple accept/reject...\n");

    auto * grammar = build_grammar(R"(root ::= "abc")");
    assert(grammar != nullptr);

    assert(match_string("abc", grammar));
    assert(!match_string("abd", grammar));
    assert(!match_string("ab", grammar));
    assert(!match_string("abcd", grammar));
    assert(!match_string("", grammar));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  Simple accept/reject: OK\n");
}

static void test_valid_bytes() {
    fprintf(stderr, "Testing get_valid_bytes...\n");

    auto * grammar = build_grammar(R"(root ::= [a-z]+)");
    assert(grammar != nullptr);

    // At the start, all lowercase letters should be valid
    auto valid = grammar->compiled->get_valid_bytes(grammar->configs);
    for (uint8_t b = 'a'; b <= 'z'; b++) {
        assert(valid[b]);
    }
    // Digits should not be valid
    for (uint8_t b = '0'; b <= '9'; b++) {
        assert(!valid[b]);
    }

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  get_valid_bytes: OK\n");
}

static void test_char_ranges() {
    fprintf(stderr, "Testing character ranges...\n");

    auto * grammar = build_grammar(R"(root ::= [a-zA-Z0-9_]+)");
    assert(grammar != nullptr);

    assert(match_string("hello", grammar));
    assert(match_string("HELLO", grammar));
    assert(match_string("hello123", grammar));
    assert(match_string("_test", grammar));
    assert(!match_string("hello world", grammar));
    assert(!match_string("", grammar));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  Character ranges: OK\n");
}

static void test_alternates() {
    fprintf(stderr, "Testing alternates...\n");

    auto * grammar = build_grammar(R"(root ::= "foo" | "bar" | "baz")");
    assert(grammar != nullptr);

    assert(match_string("foo", grammar));
    assert(match_string("bar", grammar));
    assert(match_string("baz", grammar));
    assert(!match_string("qux", grammar));
    assert(!match_string("foobar", grammar));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  Alternates: OK\n");
}

static void test_rule_refs() {
    fprintf(stderr, "Testing rule references...\n");

    auto * grammar = build_grammar(R"(
        root ::= expr
        expr ::= term ("+" term)*
        term ::= [0-9]+
    )");
    assert(grammar != nullptr);

    assert(match_string("42", grammar));
    assert(match_string("1+2", grammar));
    assert(match_string("1+2+3", grammar));
    assert(!match_string("+", grammar));
    assert(!match_string("1+", grammar));
    assert(!match_string("abc", grammar));

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  Rule references: OK\n");
}

static void test_element_level_init() {
    fprintf(stderr, "Testing element-level init...\n");

    // Build using the element-level API (same grammar as the original test)
    llama_grammar_parser parsed_grammar;

    std::vector<std::pair<std::string, uint32_t>> expected = {
        {"expr", 2},
        {"expr_6", 6},
        {"expr_7", 7},
        {"ident", 8},
        {"ident_10", 10},
        {"num", 9},
        {"num_11", 11},
        {"root", 0},
        {"root_1", 1},
        {"root_5", 5},
        {"term", 4},
        {"ws", 3},
        {"ws_12", 12},
    };

    std::vector<std::vector<llama_grammar_element>> expected_rules = {
        {{LLAMA_GRETYPE_RULE_REF, 5}, {LLAMA_GRETYPE_END, 0}},
        {
            {LLAMA_GRETYPE_RULE_REF, 2},
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_RULE_REF, 4},
            {LLAMA_GRETYPE_CHAR, 10},
            {LLAMA_GRETYPE_END, 0},
        },
        {{LLAMA_GRETYPE_RULE_REF, 4}, {LLAMA_GRETYPE_RULE_REF, 7}, {LLAMA_GRETYPE_END, 0}},
        {{LLAMA_GRETYPE_RULE_REF, 12}, {LLAMA_GRETYPE_END, 0}},
        {
            {LLAMA_GRETYPE_RULE_REF, 8},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_RULE_REF, 9},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_CHAR, 40},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_RULE_REF, 2},
            {LLAMA_GRETYPE_CHAR, 41},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_END, 0},
        },
        {{LLAMA_GRETYPE_RULE_REF, 1}, {LLAMA_GRETYPE_RULE_REF, 5}, {LLAMA_GRETYPE_ALT, 0}, {LLAMA_GRETYPE_RULE_REF, 1}, {LLAMA_GRETYPE_END, 0}},
        {
            {LLAMA_GRETYPE_CHAR, 45},
            {LLAMA_GRETYPE_CHAR_ALT, 43},
            {LLAMA_GRETYPE_CHAR_ALT, 42},
            {LLAMA_GRETYPE_CHAR_ALT, 47},
            {LLAMA_GRETYPE_RULE_REF, 4},
            {LLAMA_GRETYPE_END, 0},
        },
        {{LLAMA_GRETYPE_RULE_REF, 6}, {LLAMA_GRETYPE_RULE_REF, 7}, {LLAMA_GRETYPE_ALT, 0}, {LLAMA_GRETYPE_END, 0}},
        {
            {LLAMA_GRETYPE_CHAR, 97},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 122},
            {LLAMA_GRETYPE_RULE_REF, 10},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_END, 0},
        },
        {{LLAMA_GRETYPE_RULE_REF, 11}, {LLAMA_GRETYPE_RULE_REF, 3}, {LLAMA_GRETYPE_END, 0}},
        {
            {LLAMA_GRETYPE_CHAR, 97},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 122},
            {LLAMA_GRETYPE_CHAR_ALT, 48},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 57},
            {LLAMA_GRETYPE_CHAR_ALT, 95},
            {LLAMA_GRETYPE_RULE_REF, 10},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
        },
        {
            {LLAMA_GRETYPE_CHAR, 48},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 57},
            {LLAMA_GRETYPE_RULE_REF, 11},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_CHAR, 48},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 57},
            {LLAMA_GRETYPE_END, 0},
        },
        {
            {LLAMA_GRETYPE_CHAR, 32},
            {LLAMA_GRETYPE_CHAR_ALT, 9},
            {LLAMA_GRETYPE_CHAR_ALT, 10},
            {LLAMA_GRETYPE_RULE_REF, 12},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
        },
    };

    for (auto pair : expected) {
        parsed_grammar.symbol_ids[pair.first] = pair.second;
    }

    for (auto rule : expected_rules) {
        parsed_grammar.rules.emplace_back();
        for (auto element : rule) {
            parsed_grammar.rules.back().push_back(element);
        }
    }

    std::vector<const llama_grammar_element *> grammar_rules(parsed_grammar.c_rules());

    llama_grammar * grammar = llama_grammar_init_impl(nullptr, grammar_rules.data(), grammar_rules.size(), parsed_grammar.symbol_ids.at("root"));
    if (grammar == nullptr) {
        throw std::runtime_error("Failed to initialize llama_grammar");
    }

    // Verify that the grammar was compiled and configs are initialized
    assert(grammar->compiled != nullptr);
    assert(!grammar->configs.empty());

    // Verify valid bytes at initial state using DFA
    auto valid = grammar->compiled->get_valid_bytes(grammar->configs);

    // At start, the grammar expects an identifier (a-z) or a number (0-9) or '(' — these are
    // the initial bytes for the terms of this calculator grammar.
    // Letters a-z should be valid (start of identifier)
    assert(valid['a']);
    assert(valid['z']);
    // Digits 0-9 should be valid (start of number)
    assert(valid['0']);
    assert(valid['9']);
    // '(' should be valid (start of parenthesized expression)
    assert(valid['(']);
    // Other characters should not be valid
    assert(!valid['!']);
    assert(!valid['+']);
    assert(!valid[')']);

    llama_grammar_free_impl(grammar);
    fprintf(stderr, "  Element-level init: OK\n");
}

int main() {
    fprintf(stderr, "Running llama-grammar tests (DFA engine)...\n");
    test_grammar_initialization();
    test_simple_accept_reject();
    test_valid_bytes();
    test_char_ranges();
    test_alternates();
    test_rule_refs();
    test_element_level_init();
    fprintf(stderr, "All tests passed.\n");
    return 0;
}
