#include <string>
#include <iostream>

#undef NDEBUG
#include <cassert>

#include "jinja/jinja-type-analyzer.h"

using namespace jinja::types;

// Test helper to print results
void print_result(const std::string& test_name, const InferenceResult& result) {
    std::cout << "=== " << test_name << " ===\n";
    std::cout << result.to_string();
    std::cout << "\nConstraints:\n" << result.constraints_to_string() << "\n";
}

int main(void) {
    std::cout << "=== JINJA TYPE SYSTEM TESTS ===\n\n";

    // Test 1: Simple variable
    {
        std::cout << "Test 1: Simple variable\n";
        auto result = infer_types_from_source("{{ name }}");
        print_result("Test 1", result);

        assert(result.success());
        assert(result.has_variable("name"));
        std::cout << "PASS\n\n";
    }

    // Test 2: Simple iteration with field access
    {
        std::cout << "Test 2: Simple iteration\n";
        auto result = infer_types_from_source(
            "{% for m in messages %}{{ m.content }}{% endfor %}"
        );
        print_result("Test 2", result);

        assert(result.success());
        assert(result.has_variable("messages"));

        auto msg_type = result.get_type("messages");
        assert(msg_type != nullptr);
        std::cout << "messages type: " << msg_type->to_string() << "\n";
        std::cout << "messages kind: " << msg_type->kind() << "\n";
        std::cout << "PASS\n\n";
    }

    // Test 3: Multiple fields
    {
        std::cout << "Test 3: Multiple fields\n";
        auto result = infer_types_from_source(
            "{% for m in messages %}{{ m.role }}: {{ m.content }}{% endfor %}"
        );
        print_result("Test 3", result);

        assert(result.success());
        auto msg_type = result.get_type("messages");
        assert(msg_type != nullptr);
        std::cout << "messages type: " << msg_type->to_string() << "\n";
        std::cout << "PASS\n\n";
    }

    // Test 4: Multiple top-level variables
    {
        std::cout << "Test 4: Multiple variables\n";
        auto result = infer_types_from_source(
            "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}{{ eos_token }}"
        );
        print_result("Test 4", result);

        assert(result.success());
        assert(result.has_variable("bos_token"));
        assert(result.has_variable("eos_token"));
        assert(result.has_variable("messages"));
        std::cout << "PASS\n\n";
    }

    // Test 5: Dictionary access with string key
    {
        std::cout << "Test 5: Dictionary access\n";
        auto result = infer_types_from_source(
            "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}{% endfor %}"
        );
        print_result("Test 5", result);

        assert(result.success());
        std::cout << "PASS\n\n";
    }

    // Test 6: Set statement
    {
        std::cout << "Test 6: Set statement\n";
        auto result = infer_types_from_source(
            "{% set greeting = 'hello' %}{{ greeting }}"
        );
        print_result("Test 6", result);

        assert(result.success());
        assert(result.has_variable("greeting"));
        std::cout << "PASS\n\n";
    }

    // Test 7: Conditional (if statement)
    {
        std::cout << "Test 7: Conditional\n";
        auto result = infer_types_from_source(
            "{% if flag %}yes{% endif %}"
        );
        print_result("Test 7", result);

        assert(result.success());
        assert(result.has_variable("flag"));
        std::cout << "PASS\n\n";
    }

    // Test 8: Ternary expression
    {
        std::cout << "Test 8: Ternary expression\n";
        auto result = infer_types_from_source(
            "{{ 'yes' if flag else 'no' }}"
        );
        print_result("Test 8", result);

        assert(result.success());
        assert(result.has_variable("flag"));
        std::cout << "PASS\n\n";
    }

    // Test 9: Nested field access
    {
        std::cout << "Test 9: Nested field access\n";
        auto result = infer_types_from_source(
            "{% for m in messages %}{% for t in m.tool_calls %}{{ t.function.name }}{% endfor %}{% endfor %}"
        );
        print_result("Test 9", result);

        assert(result.success());
        std::cout << "PASS\n\n";
    }

    // Test 10: Optional field (in conditional)
    {
        std::cout << "Test 10: Optional field\n";
        auto result = infer_types_from_source(R"(
            {% for m in messages %}
                {{ m.content }}
                {% if m.tool_calls %}{{ m.tool_calls }}{% endif %}
            {% endfor %}
        )");
        print_result("Test 10", result);

        assert(result.success());
        std::cout << "PASS\n\n";
    }

    // Test 11: Complex template (Qwen-style)
    {
        std::cout << "Test 11: Complex template\n";
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
                        {{ tool_call.function.name }}({{ tool_call.function.arguments }})
                    {%- endfor %}
                {%- endif %}
            {%- endfor %}
            {%- if add_generation_prompt %}
                {{- '<|im_start|>assistant\n' }}
            {%- endif %}
        )");
        print_result("Test 11", result);

        assert(result.success());
        assert(result.has_variable("tools"));
        assert(result.has_variable("messages"));
        assert(result.has_variable("add_generation_prompt"));
        std::cout << "PASS\n\n";
    }

    // Test 12: Filter expression
    {
        std::cout << "Test 12: Filter expression\n";
        auto result = infer_types_from_source(
            "{{ message.content | trim | upper }}"
        );
        print_result("Test 12", result);

        assert(result.success());
        assert(result.has_variable("message"));
        std::cout << "PASS\n\n";
    }

    // Test 13: Array literal
    {
        std::cout << "Test 13: Array literal\n";
        auto result = infer_types_from_source(
            "{% for x in [1, 2, 3] %}{{ x }}{% endfor %}"
        );
        print_result("Test 13", result);

        assert(result.success());
        std::cout << "PASS\n\n";
    }

    // Test 14: reasoning_content support
    {
        std::cout << "Test 14: reasoning_content\n";
        auto result = infer_types_from_source(R"(
            {% for m in messages %}
                {{ m.content }}
                {% if m.reasoning_content %}{{ m.reasoning_content }}{% endif %}
            {% endfor %}
        )");
        print_result("Test 14", result);

        assert(result.success());
        std::cout << "PASS\n\n";
    }

    std::cout << "\n=== ALL JINJA TYPE SYSTEM TESTS PASSED ===\n";
    return 0;
}
