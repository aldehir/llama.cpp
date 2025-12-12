#pragma once

#include "pcre-ast.h"
#include <string>

namespace pcre_codegen {

struct CodegenOptions {
    std::string function_name = "regex_split";
    std::string namespace_name = "";
    bool generate_main = false;
    bool generate_header = true;
    bool inline_unicode_helpers = true;  // Include unicode helpers in output
    std::string indent = "    ";
};

// Generate C++ code from a PCRE AST
// The generated code is a standalone function that splits strings according to the regex
std::string generate(const pcre_ast::NodePtr& ast, const CodegenOptions& options = {});

// Generate just the matching function body (for embedding)
std::string generate_matcher_body(const pcre_ast::NodePtr& ast, const CodegenOptions& options = {});

}  // namespace pcre_codegen
