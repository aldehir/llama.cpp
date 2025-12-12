#pragma once

#include "pcre-ast.h"
#include <string>
#include <stdexcept>

namespace pcre_parser {

class ParseError : public std::runtime_error {
public:
    size_t position;
    std::string context;

    ParseError(const std::string& msg, size_t pos, const std::string& ctx = "")
        : std::runtime_error(msg), position(pos), context(ctx) {}
};

// Parse a PCRE regex pattern into an AST
// Supports a subset of PCRE syntax needed for LLM pretokenizers:
// - Literals and escape sequences
// - Character classes: [abc], [a-z], [^...]
// - Unicode properties: \p{L}, \p{N}, \p{Han}, etc.
// - Shorthand classes: \s, \S, \d, \D, \w, \W
// - Quantifiers: *, +, ?, {n}, {n,}, {n,m}
// - Alternation: a|b
// - Groups: (...), (?:...), (?i:...)
// - Lookahead: (?=...), (?!...)
// - Anchors: ^, $ (limited support)
// - Special chars: \r, \n, \t
pcre_ast::NodePtr parse(const std::string& pattern);

// Parse with detailed error information
struct ParseResult {
    pcre_ast::NodePtr ast;
    bool success;
    std::string error_message;
    size_t error_position;
};

ParseResult parse_with_result(const std::string& pattern);

}  // namespace pcre_parser
