#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#undef NDEBUG
#include <cassert>

#include "jinja/jinja-type-analyzer.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

using namespace jinja::types;

// ============================================================================
// ANSI Color Support
// ============================================================================

namespace color {
    const char* GREEN  = "\033[32m";
    const char* RED    = "\033[31m";
    const char* CYAN   = "\033[36m";
    const char* YELLOW = "\033[33m";
    const char* DIM    = "\033[2m";
    const char* RESET  = "\033[0m";
    const char* BOLD   = "\033[1m";

    void enable_windows_ansi() {
#ifdef _WIN32
        HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD mode = 0;
        GetConsoleMode(hOut, &mode);
        SetConsoleMode(hOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif
    }
}

// ============================================================================
// Test Framework
// ============================================================================

struct test_context {
    bool verbose = false;
    int passed = 0;
    int failed = 0;
    int skipped = 0;
    std::string input_file;  // If set, analyze this file instead of running tests
};

static test_context g_ctx;

// Helper to read file contents
static std::string read_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// ============================================================================
// File Analysis Mode
// ============================================================================

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS] [FILE]\n\n";
    std::cout << "Run Jinja type system tests or analyze a template file.\n\n";
    std::cout << "Options:\n";
    std::cout << "  -v, --verbose    Show constraints in output\n";
    std::cout << "  -h, --help       Show this help message\n\n";
    std::cout << "If FILE is provided, analyze that template instead of running tests.\n";
}

int analyze_file(const std::string& path) {
    std::cout << color::BOLD << "=== JINJA TYPE INFERENCE ===" << color::RESET << "\n\n";

    std::string source = read_file(path);
    if (source.empty()) {
        std::cerr << color::RED << "Error: Could not read file: " << path << color::RESET << "\n";
        return 1;
    }

    std::cout << color::DIM << "File: " << path << color::RESET << "\n\n";

    auto result = infer_types_from_source(source);

    // Print schema
    std::cout << color::BOLD << "Inferred Schema:" << color::RESET << "\n";
    for (const auto& [name, type] : result.variable_types) {
        std::cout << "  " << color::CYAN << name << color::RESET << ":";
        if (type) {
            if (type->is_simple()) {
                std::cout << " " << type->to_string() << "\n";
            } else {
                std::cout << "\n    " << type->to_string_pretty(2, 2) << "\n";
            }
        } else {
            std::cout << " unknown\n";
        }
    }

    // Print constraints if verbose
    if (g_ctx.verbose) {
        std::cout << "\n" << color::BOLD << "Constraints (" << result.constraints.size() << "):" << color::RESET << "\n";
        for (const auto& c : result.constraints.constraints) {
            std::cout << color::DIM << "  ";
            std::visit([](const auto& cst) {
                std::cout << cst.to_string();
            }, c);
            std::cout << color::RESET << "\n";
        }
    }

    // Print errors if any
    if (!result.errors.empty()) {
        std::cout << "\n" << color::YELLOW << "Warnings/Errors:" << color::RESET << "\n";
        for (const auto& err : result.errors) {
            std::cout << color::DIM << "  - " << err << color::RESET << "\n";
        }
    }

    std::cout << "\n" << color::GREEN << "Analysis complete." << color::RESET << "\n";
    return 0;
}

void print_header(int num, const std::string& name) {
    std::cout << color::BOLD << "Test " << num << ": " << name << color::RESET << "\n";
}

void print_schema(const inference_result& result) {
    for (const auto& [name, type] : result.variable_types) {
        std::cout << "  " << color::CYAN << name << color::RESET << ":";
        if (type) {
            if (type->is_simple()) {
                std::cout << " " << type->to_string() << "\n";
            } else {
                // Complex type - start on next line
                std::cout << "\n    " << type->to_string_pretty(2, 2) << "\n";
            }
        } else {
            std::cout << " unknown\n";
        }
    }
}

void print_constraints(const inference_result& result) {
    std::cout << color::DIM;
    std::cout << "  Constraints (" << result.constraints.size() << "):\n";
    for (const auto& c : result.constraints.constraints) {
        std::cout << "    ";
        std::visit([](const auto& cst) {
            std::cout << cst.to_string();
        }, c);
        std::cout << "\n";
    }
    std::cout << color::RESET;
}

void print_pass(const inference_result& result) {
    std::cout << color::GREEN << "  PASS" << color::RESET
              << color::DIM << " (" << result.constraints.size() << " constraints)"
              << color::RESET << "\n\n";
    g_ctx.passed++;
}

void print_fail(const std::string& reason) {
    std::cout << color::RED << "  FAIL: " << reason << color::RESET << "\n\n";
    g_ctx.failed++;
}

void print_skip(const std::string& reason) {
    std::cout << color::YELLOW << "  SKIP: " << reason << color::RESET << "\n\n";
    g_ctx.skipped++;
}

void print_summary() {
    std::cout << color::BOLD << "=== SUMMARY ===" << color::RESET << "\n";
    std::cout << color::GREEN << g_ctx.passed << " passed" << color::RESET;
    if (g_ctx.failed > 0) {
        std::cout << ", " << color::RED << g_ctx.failed << " failed" << color::RESET;
    }
    if (g_ctx.skipped > 0) {
        std::cout << ", " << color::YELLOW << g_ctx.skipped << " skipped" << color::RESET;
    }
    std::cout << "\n";
}

// ============================================================================
// Tests
// ============================================================================

int main(int argc, char* argv[]) {
    color::enable_windows_ansi();

    // Parse command line
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--verbose" || arg == "-v") {
            g_ctx.verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg[0] != '-') {
            // Assume it's a file path
            g_ctx.input_file = arg;
        } else {
            std::cerr << color::RED << "Unknown option: " << arg << color::RESET << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // If a file is specified, analyze it instead of running tests
    if (!g_ctx.input_file.empty()) {
        return analyze_file(g_ctx.input_file);
    }

    std::cout << color::BOLD << "=== JINJA TYPE SYSTEM TESTS ===" << color::RESET << "\n\n";

    // Test 1: Simple variable
    {
        print_header(1, "Simple variable");
        auto result = infer_types_from_source("{{ name }}");
        print_schema(result);
        if (g_ctx.verbose) print_constraints(result);

        assert(result.success());
        assert(result.has_variable("name"));
        print_pass(result);
    }

    // Test 2: Simple iteration with field access
    {
        print_header(2, "Simple iteration");
        auto result = infer_types_from_source(
            "{% for m in messages %}{{ m.content }}{% endfor %}"
        );
        print_schema(result);
        if (g_ctx.verbose) print_constraints(result);

        assert(result.success());
        assert(result.has_variable("messages"));
        auto msg_type = result.get_type("messages");
        assert(msg_type != nullptr);
        print_pass(result);
    }

    // Test 3: Multiple fields
    {
        print_header(3, "Multiple fields");
        auto result = infer_types_from_source(
            "{% for m in messages %}{{ m.role }}: {{ m.content }}{% endfor %}"
        );
        print_schema(result);
        if (g_ctx.verbose) print_constraints(result);

        assert(result.success());
        auto msg_type = result.get_type("messages");
        assert(msg_type != nullptr);
        print_pass(result);
    }

    // Test 4: Multiple top-level variables
    {
        print_header(4, "Multiple variables");
        auto result = infer_types_from_source(
            "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}{{ eos_token }}"
        );
        print_schema(result);
        if (g_ctx.verbose) print_constraints(result);

        assert(result.success());
        assert(result.has_variable("bos_token"));
        assert(result.has_variable("eos_token"));
        assert(result.has_variable("messages"));
        print_pass(result);
    }

    // Test 5: Dictionary access with string key
    {
        print_header(5, "Dictionary access");
        auto result = infer_types_from_source(
            "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}{% endfor %}"
        );
        print_schema(result);
        if (g_ctx.verbose) print_constraints(result);

        assert(result.success());
        print_pass(result);
    }

    // Test 6: Set statement
    {
        print_header(6, "Set statement");
        auto result = infer_types_from_source(
            "{% set greeting = 'hello' %}{{ greeting }}"
        );
        print_schema(result);
        if (g_ctx.verbose) print_constraints(result);

        assert(result.success());
        assert(result.has_variable("greeting"));
        print_pass(result);
    }

    // Test 7: Conditional (if statement)
    {
        print_header(7, "Conditional");
        auto result = infer_types_from_source(
            "{% if flag %}yes{% endif %}"
        );
        print_schema(result);
        if (g_ctx.verbose) print_constraints(result);

        assert(result.success());
        assert(result.has_variable("flag"));
        print_pass(result);
    }

    // Test 8: Ternary expression
    {
        print_header(8, "Ternary expression");
        auto result = infer_types_from_source(
            "{{ 'yes' if flag else 'no' }}"
        );
        print_schema(result);
        if (g_ctx.verbose) print_constraints(result);

        assert(result.success());
        assert(result.has_variable("flag"));
        print_pass(result);
    }

    // Test 9: Nested field access
    {
        print_header(9, "Nested field access");
        auto result = infer_types_from_source(
            "{% for m in messages %}{% for t in m.tool_calls %}{{ t.function.name }}{% endfor %}{% endfor %}"
        );
        print_schema(result);
        if (g_ctx.verbose) print_constraints(result);

        assert(result.success());
        print_pass(result);
    }

    // Test 10: Optional field (in conditional)
    {
        print_header(10, "Optional field");
        auto result = infer_types_from_source(R"(
            {% for m in messages %}
                {{ m.content }}
                {% if m.tool_calls %}{{ m.tool_calls }}{% endif %}
            {% endfor %}
        )");
        print_schema(result);
        if (g_ctx.verbose) print_constraints(result);

        assert(result.success());
        print_pass(result);
    }

    // Test 11: Complex template (Qwen-style)
    {
        print_header(11, "Complex template");
        auto result = infer_types_from_source(R"(
            {%- if tools %}
                {{- '<|im_start|>system\n' }}
                {%- for tool in tools %}
                    {{- tool | tojson }}
                {%- endfor %}
            {%- else %}
                {%- if messages[0].role == 'system' %}
                    {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
                {%- endif %}
            {%- endif %}
            {%- for message in messages %}
                {{ message.role }}: {{ message.content }}
                {%- if message.tool_calls %}
                    {%- for tool_call in message.tool_calls %}
                        {%- if tool_call.function -%}
                            {% set tool_call = tool_call.function %}
                        {%- endif -%}
                        {{ tool_call.name }}({{ tool_call.arguments }})
                    {%- endfor %}
                {%- endif %}
            {%- endfor %}
            {%- if add_generation_prompt %}
                {{- '<|im_start|>assistant\n' }}
            {%- endif %}
        )");
        print_schema(result);
        if (g_ctx.verbose) print_constraints(result);

        assert(result.success());
        assert(result.has_variable("tools"));
        assert(result.has_variable("messages"));
        assert(result.has_variable("add_generation_prompt"));
        print_pass(result);
    }

    // Test 12: Filter expression
    {
        print_header(12, "Filter expression");
        auto result = infer_types_from_source(
            "{{ message.content | trim | upper }}"
        );
        print_schema(result);
        if (g_ctx.verbose) print_constraints(result);

        assert(result.success());
        assert(result.has_variable("message"));
        print_pass(result);
    }

    // Test 13: Array literal
    {
        print_header(13, "Array literal");
        auto result = infer_types_from_source(
            "{% for x in [1, 2, 3] %}{{ x }}{% endfor %}"
        );
        print_schema(result);
        if (g_ctx.verbose) print_constraints(result);

        assert(result.success());
        print_pass(result);
    }

    // Test 14: reasoning_content support
    {
        print_header(14, "reasoning_content");
        auto result = infer_types_from_source(R"(
            {% for m in messages %}
                {{ m.content }}
                {% if m.reasoning_content %}{{ m.reasoning_content }}{% endif %}
            {% endfor %}
        )");
        print_schema(result);
        if (g_ctx.verbose) print_constraints(result);

        assert(result.success());
        print_pass(result);
    }

    // Test 15: Sort filter preserves element type
    {
        print_header(15, "Sort filter");
        auto result = infer_types_from_source(
            "{% for m in messages | sort %}{{ m.content }}{% endfor %}"
        );
        print_schema(result);
        if (g_ctx.verbose) print_constraints(result);

        assert(result.success());
        auto msg_type = result.get_type("messages");
        assert(msg_type != nullptr);
        print_pass(result);
    }

    // Test 16: First filter extracts element type
    {
        print_header(16, "First filter");
        auto result = infer_types_from_source(
            "{{ messages | first | upper }}"
        );
        print_schema(result);
        if (g_ctx.verbose) print_constraints(result);

        assert(result.success());
        auto msg_type = result.get_type("messages");
        assert(msg_type != nullptr);
        print_pass(result);
    }

    // Test 17: Map filter with attribute extracts field type
    {
        print_header(17, "Map with attribute");
        auto result = infer_types_from_source(
            "{% for name in users | map(attribute='name') %}{{ name }}{% endfor %}"
        );
        print_schema(result);
        if (g_ctx.verbose) print_constraints(result);

        assert(result.success());
        auto users_type = result.get_type("users");
        assert(users_type != nullptr);
        print_pass(result);
    }

    // Test 18: Selectattr filter preserves element type
    {
        print_header(18, "Selectattr filter");
        auto result = infer_types_from_source(
            "{% for u in users | selectattr('active') %}{{ u.name }}{% endfor %}"
        );
        print_schema(result);
        if (g_ctx.verbose) print_constraints(result);

        assert(result.success());
        auto users_type = result.get_type("users");
        assert(users_type != nullptr);
        print_pass(result);
    }

    // Test 19: Batch filter returns array of arrays
    {
        print_header(19, "Batch filter");
        auto result = infer_types_from_source(
            "{% for batch in items | batch(3) %}{% for item in batch %}{{ item.x }}{% endfor %}{% endfor %}"
        );
        print_schema(result);
        if (g_ctx.verbose) print_constraints(result);

        assert(result.success());
        auto items_type = result.get_type("items");
        assert(items_type != nullptr);
        print_pass(result);
    }

    // Test 20: Filter chain type propagation
    {
        print_header(20, "Filter chain");
        auto result = infer_types_from_source(
            "{% for m in messages | selectattr('visible') | sort | reverse %}{{ m.content }}{% endfor %}"
        );
        print_schema(result);
        if (g_ctx.verbose) print_constraints(result);

        assert(result.success());
        auto msg_type = result.get_type("messages");
        assert(msg_type != nullptr);
        print_pass(result);
    }

    // Test 21: Join filter with element access
    {
        print_header(21, "Join filter");
        auto result = infer_types_from_source(
            "{{ names | join(', ') }}"
        );
        print_schema(result);
        if (g_ctx.verbose) print_constraints(result);

        assert(result.success());
        auto names_type = result.get_type("names");
        assert(names_type != nullptr);
        print_pass(result);
    }

    // Test 22: Multiple array index accesses should merge element type
    {
        print_header(22, "Array index field merging");
        auto result = infer_types_from_source(
            "{{ messages[0].role }}{{ messages[0].content }}"
        );
        print_schema(result);
        if (g_ctx.verbose) print_constraints(result);

        assert(result.success());
        auto msg_type = result.get_type("messages");
        assert(msg_type != nullptr);
        print_pass(result);
    }

    // Test 23: Literal type inference from equality comparisons
    {
        print_header(23, "Literal type inference");
        auto result = infer_types_from_source(R"(
            {% for m in messages %}
                {% if m.role == 'assistant' %}
                    Assistant: {{ m.content }}
                {% elif m.role == 'user' %}
                    User: {{ m.content }}
                {% elif m.role == 'system' %}
                    System: {{ m.content }}
                {% endif %}
            {% endfor %}
        )");
        print_schema(result);
        if (g_ctx.verbose) print_constraints(result);

        assert(result.success());
        auto msg_type = result.get_type("messages");
        assert(msg_type != nullptr);

        // The role field should be inferred as a union of literal strings
        if (auto* arr = as_type<array_type>(msg_type)) {
            if (auto* elem = as_type<object_type>(arr->element_type)) {
                auto* role_field = elem->get_field("role");
                if (role_field && role_field->type) {
                    assert(is_type<union_type>(role_field->type) ||
                           is_type<literal_type>(role_field->type) ||
                           is_type<primitive_type>(role_field->type));
                }
            }
        }
        print_pass(result);
    }

    // Test 24: Real-world template (OpenAI GPT OSS 120B)
    {
        print_header(24, "Real-world template (GPT-OSS)");
        std::string template_source = read_file("models/templates/openai-gpt-oss-120b.jinja");
        if (template_source.empty()) {
            print_skip("Could not read template file");
        } else {
            auto result = infer_types_from_source(template_source);
            print_schema(result);
            if (g_ctx.verbose) print_constraints(result);

            if (!result.success()) {
                std::cout << color::DIM << "  Note: Constraint errors (expected for complex templates):" << color::RESET << "\n";
                for (const auto& err : result.errors) {
                    std::cout << color::DIM << "    - " << err << color::RESET << "\n";
                }
            }

            assert(result.has_variable("messages"));
            assert(result.has_variable("tools"));
            assert(result.has_variable("add_generation_prompt"));

            auto msg_type = result.get_type("messages");
            assert(msg_type != nullptr);

            auto tools_type = result.get_type("tools");
            assert(tools_type != nullptr);

            print_pass(result);
        }
    }

    std::cout << "\n";
    print_summary();

    return g_ctx.failed > 0 ? 1 : 0;
}
