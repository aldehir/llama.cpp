#include "pcre-parser.h"
#include "pcre-codegen.h"
#include "pcre-ast.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s [options] <regex-pattern>\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "Generate C++ code from a PCRE regex pattern for pretokenization.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -h, --help           Show this help message\n");
    fprintf(stderr, "  -o, --output FILE    Output to file instead of stdout\n");
    fprintf(stderr, "  -n, --name NAME      Function name (default: regex_split)\n");
    fprintf(stderr, "  -N, --namespace NS   Wrap in namespace\n");
    fprintf(stderr, "  -m, --main           Generate main() function for testing\n");
    fprintf(stderr, "  -a, --ast            Print AST instead of generating code\n");
    fprintf(stderr, "  -f, --file FILE      Read pattern from file\n");
    fprintf(stderr, "  --no-helpers         Don't inline unicode helpers\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  %s \"'s|'t|'re|'ve|'m|'ll|'d| ?\\\\p{L}+| ?\\\\p{N}+\"\n", prog);
    fprintf(stderr, "  %s -n gpt2_split -o gpt2_tokenizer.cpp \"'s|'t|...\"\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "Supported PCRE features:\n");
    fprintf(stderr, "  - Literals and escape sequences (\\n, \\r, \\t, etc.)\n");
    fprintf(stderr, "  - Character classes: [abc], [a-z], [^...]\n");
    fprintf(stderr, "  - Unicode properties: \\p{L}, \\p{N}, \\p{Han}, etc.\n");
    fprintf(stderr, "  - Shorthand classes: \\s, \\S, \\d, \\D, \\w, \\W\n");
    fprintf(stderr, "  - Quantifiers: *, +, ?, {n}, {n,}, {n,m}\n");
    fprintf(stderr, "  - Non-greedy quantifiers: *?, +?, ??\n");
    fprintf(stderr, "  - Alternation: a|b\n");
    fprintf(stderr, "  - Groups: (...), (?:...), (?i:...)\n");
    fprintf(stderr, "  - Lookahead: (?=...), (?!...)\n");
}

static std::string read_file(const char* path) {
    std::ifstream file(path);
    if (!file) {
        fprintf(stderr, "Error: Cannot open file '%s'\n", path);
        return "";
    }
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    // Remove trailing newline if present
    while (!content.empty() && (content.back() == '\n' || content.back() == '\r')) {
        content.pop_back();
    }
    return content;
}

int main(int argc, char** argv) {
    const char* output_file = nullptr;
    const char* function_name = "regex_split";
    const char* namespace_name = nullptr;
    const char* pattern_file = nullptr;
    bool generate_main = false;
    bool print_ast = false;
    bool inline_helpers = true;
    const char* pattern = nullptr;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: -o requires an argument\n");
                return 1;
            }
            output_file = argv[i];
        }
        else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--name") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: -n requires an argument\n");
                return 1;
            }
            function_name = argv[i];
        }
        else if (strcmp(argv[i], "-N") == 0 || strcmp(argv[i], "--namespace") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: -N requires an argument\n");
                return 1;
            }
            namespace_name = argv[i];
        }
        else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--main") == 0) {
            generate_main = true;
        }
        else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--ast") == 0) {
            print_ast = true;
        }
        else if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--file") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: -f requires an argument\n");
                return 1;
            }
            pattern_file = argv[i];
        }
        else if (strcmp(argv[i], "--no-helpers") == 0) {
            inline_helpers = false;
        }
        else if (argv[i][0] == '-') {
            fprintf(stderr, "Error: Unknown option '%s'\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
        else {
            if (pattern != nullptr) {
                fprintf(stderr, "Error: Multiple patterns specified\n");
                return 1;
            }
            pattern = argv[i];
        }
    }

    // Get pattern
    std::string pattern_str;
    if (pattern_file != nullptr) {
        pattern_str = read_file(pattern_file);
        if (pattern_str.empty()) {
            return 1;
        }
    } else if (pattern != nullptr) {
        pattern_str = pattern;
    } else {
        fprintf(stderr, "Error: No pattern specified\n");
        print_usage(argv[0]);
        return 1;
    }

    // Parse the pattern
    auto result = pcre_parser::parse_with_result(pattern_str);
    if (!result.success) {
        fprintf(stderr, "Parse error at position %zu: %s\n",
                result.error_position, result.error_message.c_str());
        if (result.error_position < pattern_str.size()) {
            fprintf(stderr, "  %s\n", pattern_str.c_str());
            fprintf(stderr, "  %s^\n", std::string(result.error_position, ' ').c_str());
        }
        return 1;
    }

    // Print AST if requested
    if (print_ast) {
        std::cout << "Pattern: " << pattern_str << "\n\n";
        std::cout << "AST:\n";
        std::cout << pcre_ast::node_to_string(result.ast);
        return 0;
    }

    // Generate code
    pcre_codegen::CodegenOptions opts;
    opts.function_name = function_name;
    if (namespace_name) {
        opts.namespace_name = namespace_name;
    }
    opts.generate_main = generate_main;
    opts.inline_unicode_helpers = inline_helpers;

    std::string code = pcre_codegen::generate(result.ast, opts);

    // Add closing namespace brace if needed
    if (namespace_name) {
        code += "\n}  // namespace " + std::string(namespace_name) + "\n";
    }

    // Output
    if (output_file) {
        std::ofstream out(output_file);
        if (!out) {
            fprintf(stderr, "Error: Cannot open output file '%s'\n", output_file);
            return 1;
        }
        out << "// Generated by pcre-to-cpp from pattern:\n";
        out << "// " << pattern_str << "\n\n";
        out << code;
        fprintf(stderr, "Generated code written to %s\n", output_file);
    } else {
        std::cout << "// Generated by pcre-to-cpp from pattern:\n";
        std::cout << "// " << pattern_str << "\n\n";
        std::cout << code;
    }

    return 0;
}
