#include <string>
#include <vector>
#include <sstream>
#include <regex>
#include <iostream>

#undef NDEBUG
#include <cassert>

#include "jinja/jinja-parser.h"
#include "jinja/jinja-lexer.h"
#include "jinja/jinja-peg-parser.h"

int main(void) {
    std::string contents = "{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\\n' + message['content'] | trim + '<end_of_turn>\\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\\n'}}{% endif %}";

    //std::string contents = "{% if messages[0]['role'] != 'system' %}nice {{ messages[0]['content'] }}{% endif %}";

    //std::string contents = "<some_tokens> {{ messages[0]['content'] }} <another_token>";

    std::cout << "=== INPUT ===\n" << contents << "\n\n";

    jinja::lexer lexer;
    jinja::preprocess_options options;
    options.trim_blocks = true;
    options.lstrip_blocks = false;
    auto tokens = lexer.tokenize(contents, options);
    for (const auto & tok : tokens) {
        std::cout << "token: type=" << static_cast<int>(tok.t) << " text='" << tok.value << "'\n";
    }

    std::cout << "\n=== AST ===\n";
    jinja::program ast = jinja::parse_from_tokens(tokens);
    for (const auto & stmt : ast.body) {
        std::cout << "stmt type: " << stmt->type() << "\n";
    }

    auto make_non_special_string = [](const std::string & s) {
        jinja::value_string str_val = jinja::mk_val<jinja::value_string>(s);
        str_val->mark_input();
        return str_val;
    };

    jinja::value messages = jinja::mk_val<jinja::value_array>();
    jinja::value msg1 = jinja::mk_val<jinja::value_object>();
    (*msg1->val_obj)["role"]    = make_non_special_string("user");
    (*msg1->val_obj)["content"] = make_non_special_string("Hello, how are you?");
    messages->val_arr->push_back(std::move(msg1));
    jinja::value msg2 = jinja::mk_val<jinja::value_object>();
    (*msg2->val_obj)["role"]    = make_non_special_string("assistant");
    (*msg2->val_obj)["content"] = make_non_special_string("I am fine, thank you!");
    messages->val_arr->push_back(std::move(msg2));

    std::cout << "\n=== RUN (Lexer/Parser) ===\n";
    {
        jinja::context ctx;
        ctx.var["messages"] = messages->clone();
        jinja::vm vm(ctx);
        auto results = vm.execute(ast);

        std::cout << "\n=== RESULTS (Lexer/Parser) ===\n";
        for (const auto & res : results) {
            if (res->is_null()) {
                continue;
            }
            std::cout << "result type: " << res->type() << " | value: " << res->as_repr();
        }
    }

    std::cout << "\n=== AST (PEG parser) ===\n";
    jinja::program ast_peg({});
    try {
        ast_peg = jinja::parse_with_peg(contents);
        std::cout << "Parsed " << ast_peg.body.size() << " statements\n";
        for (const auto & stmt : ast_peg.body) {
            std::cout << "stmt type: " << stmt->type() << "\n";
        }
    } catch (const std::exception& e) {
        std::cout << "Parse error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\n=== RUN (PEG parser) ===\n";
    {
        jinja::context ctx;
        ctx.var["messages"] = messages->clone();
        jinja::vm vm(ctx);
        auto results = vm.execute(ast_peg);

        std::cout << "\n=== RESULTS (PEG parser) ===\n";
        for (const auto & res : results) {
            if (res->is_null()) {
                continue;
            }
            std::cout << "result type: " << res->type() << " | value: " << res->as_repr();
        }
    }

    return 0;
}
