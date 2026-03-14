#pragma once

#include "llama-grammar-dfa.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

// ============================================================
// AST node types for GBNF grammar
// ============================================================

struct grammar_ast_node;
using ast_node_ptr = std::unique_ptr<grammar_ast_node>;

struct ast_char_range {
    uint32_t lo;
    uint32_t hi;
};

struct grammar_ast_node {
    enum type_t {
        LITERAL,       // exact code points (from "..." strings)
        CHAR_CLASS,    // [abc], [a-z], [^...], .
        RULE_REF,      // reference to named rule
        SEQUENCE,      // concatenation of children
        ALTERNATION,   // choice among children
        REPETITION,    // child repeated {min, max} times
        EXCLUSION,     // !("str1" | "str2") - matches any string not containing excluded substrings
    };
    type_t type;

    // LITERAL
    std::vector<uint32_t> code_points;

    // CHAR_CLASS
    std::vector<ast_char_range> ranges;  // {lo, hi} pairs
    bool negated = false;
    bool any_char = false;

    // RULE_REF
    std::string rule_name;
    uint32_t rule_id = 0;

    // SEQUENCE, ALTERNATION
    std::vector<ast_node_ptr> children;

    // REPETITION
    ast_node_ptr child;
    uint64_t min_rep = 0;
    uint64_t max_rep = UINT64_MAX;  // unbounded

    // EXCLUSION
    std::vector<std::vector<uint32_t>> excluded_strings;  // code point sequences

    // Deep copy
    ast_node_ptr clone() const;

    // Check if this subtree is purely terminal (no RULE_REFs)
    bool is_purely_terminal() const;
};

// ============================================================
// Terminal classification for rules
// ============================================================

enum class terminal_class {
    UNKNOWN,
    PURELY_TERMINAL,
    TRANSITIVELY_TERMINAL,
    NON_TERMINAL,
};

// ============================================================
// AST-based grammar parser
// ============================================================

struct llama_grammar_ast_parser {
    std::map<std::string, uint32_t> symbol_ids;
    std::vector<ast_node_ptr> rules;  // indexed by rule_id

    // Terminal classification per rule
    std::vector<terminal_class> rule_classes;

    uint32_t get_symbol_id(const char * src, size_t len);

    // Parsing: returns an AST node via out parameter
    const char * parse_alternates(const char * src, const std::string & rule_name,
                                  bool is_nested, ast_node_ptr & out);
    const char * parse_sequence(const char * src, const std::string & rule_name,
                                bool is_nested, ast_node_ptr & out);
    const char * parse_rule(const char * src);

    bool parse(const char * src);

    // Optimization passes
    void optimize();

private:
    void classify_terminals();
    void inline_terminal_refs();
    void flatten_nested(ast_node_ptr & node);

    bool check_transitively_terminal(uint32_t rule_id, std::vector<bool> & in_progress);
    void inline_refs_in_node(ast_node_ptr & node);
};

// ============================================================
// AST to NFA conversion (for purely terminal subtrees)
// ============================================================

grammar_nfa ast_to_nfa(const grammar_ast_node & node);

// ============================================================
// AST to compiled_grammar compilation
// ============================================================

compiled_grammar compile_grammar_from_ast(
    const std::vector<ast_node_ptr> & rules,
    uint32_t start_rule_index);
