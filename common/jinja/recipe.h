#pragma once

#include "runtime.h"
#include "utils.h"

#include <stdexcept>
#include <string>

namespace jinja {

// Parser for the quantization-recipe evaluation dialect.
//
// The dialect is a small, total-by-construction language (no for/macro/calls,
// floor division only) that maps directly onto the existing jinja AST nodes.
// A recipe is parsed with the common PEG parser, then the resulting parse tree
// is walked to emit a jinja::program that the jinja runtime executes unchanged.
//
//   default Q4_K
//   set more_bits = layer < n_layer // 8 or layer >= 7 * n_layer // 8
//   if category == "ATTENTION_WV"
//       if more_bits
//           type = Q6_K
//       endif
//   endif
//
// The host seeds the context (layer, n_layer, category, ...) and the ggml type
// names as strings, runs the program, then reads back `type`.
program parse_recipe(const std::string & src);

struct recipe_exception : public std::runtime_error {
    recipe_exception(const std::string & msg, const std::string & source, size_t pos)
        : std::runtime_error(fmt_error_with_source("recipe", msg, source, pos)) {}
};

} // namespace jinja
