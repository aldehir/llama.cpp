#pragma once

#include "lexer.h"
#include "runtime.h"
#include "utils.h"

#include <functional>
#include <stdexcept>
#include <string>

namespace jinja {
namespace recipe {

// Predicate deciding whether a bare word is a quant type literal (e.g. "Q4_K").
// Recognized words compile to string literals so the runtime can compare them
// against pre-seeded type names; unrecognized words compile to identifiers.
using type_predicate = std::function<bool(const std::string &)>;

// Tokenize recipe-dialect source into jinja tokens.
// may throw recipe_exception on error
lexer_result tokenize(const std::string & source);

// Parse recipe-dialect source into a jinja::program, reusing the jinja runtime.
// The host seeds the context, executes the program, then reads back `type`.
// may throw recipe_exception on error
program parse(const std::string & source, const type_predicate & is_type = nullptr);

struct recipe_exception : public std::runtime_error {
    recipe_exception(const std::string & msg, const std::string & source, size_t pos)
        : std::runtime_error(fmt_error_with_source("recipe", msg, source, pos)) {}
};

} // namespace recipe
} // namespace jinja
