#include "chat-parsers.h"

#include "chat-peg-parser.h"
#include "json-schema-to-grammar.h"

static common_chat_params common_chat_params_init_deepseek_v3_2(const common_chat_template &      tmpl,
                                                                common_chat_render_inputs & inputs) {
    common_chat_params data;

    data.prompt             = common_chat_template_direct_apply(tmpl, inputs);
    data.generation_prompt  = common_chat_template_generation_prompt(tmpl, inputs);
    data.format             = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking  = true;
    data.thinking_start_tag = "<think>";
    data.thinking_end_tag   = "</think>";
    data.preserved_tokens   = {
        "｜DSML｜",
        "<think>",
        "</think>",
    };

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = !inputs.json_schema.is_null() && inputs.json_schema.is_object();
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);

    const std::string DSML         = "｜DSML｜";
    const std::string THINK_START  = "<think>";
    const std::string THINK_END    = "</think>";
    const std::string FC_START     = "<" + DSML + "function_calls>";
    const std::string FC_END       = "</" + DSML + "function_calls>";
    const std::string INVOKE_START = "<" + DSML + "invoke";
    const std::string INVOKE_END   = "</" + DSML + "invoke>";
    const std::string PARAM_START  = "<" + DSML + "parameter";
    const std::string PARAM_END    = "</" + DSML + "parameter>";
    const std::string GEN_PROMPT   = "<｜Assistant｜>";

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt = GEN_PROMPT + THINK_START + msg.reasoning_content;
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += THINK_END + msg.render_content();
        }

        data.prompt += data.generation_prompt;
    }

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto generation_prompt = p.literal(GEN_PROMPT);
        auto end = p.end();

        auto reasoning = p.eps();
        if (extract_reasoning && inputs.enable_thinking) {
            reasoning = p.optional(THINK_START + p.reasoning(p.until(THINK_END)) + THINK_END);
        } else if (extract_reasoning) {
            // Thinking disabled but reasoning extraction requested: the generation prompt
            // contains an empty <think></think> pair that must still be consumed.
            reasoning = p.optional(p.literal(THINK_START) + p.until(THINK_END) + p.literal(THINK_END));
        }

        if (has_response_format) {
            auto response_format = p.rule("response-format",
                p.literal("```json") + p.space() +
                p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema)) +
                p.space() + p.literal("```"));
            return generation_prompt + reasoning + response_format + end;
        }

        if (!has_tools || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            return generation_prompt + reasoning + p.content(p.rest()) + end;
        }

        auto tool_choice = p.choice();
        common_chat_foreach_function(inputs.tools, [&](const json & tool) {
            const auto & function = tool.at("function");
            std::string  name     = function.at("name");
            auto params   = function.contains("parameters") ? function.at("parameters") : json::object();
            const auto & props    = params.contains("properties") ? params.at("properties") : json::object();

            std::set<std::string> required;
            if (params.contains("required")) {
                params.at("required").get_to(required);
            }

            auto schema_info = common_schema_info();
            schema_info.resolve_refs(params);

            std::vector<common_peg_parser> required_parsers;
            std::vector<common_peg_parser> optional_parsers;
            for (const auto & [param_name, param_schema] : props.items()) {
                bool is_required = required.find(param_name) != required.end();
                bool is_string   = schema_info.resolves_to_string(param_schema);

                auto arg = p.tool_arg(
                    p.tool_arg_open(
                        p.literal(PARAM_START + " name=\"") +
                        p.tool_arg_name(p.literal(param_name)) +
                        p.literal("\" string=\"" + std::string(is_string ? "true" : "false") + "\">")) +
                    (is_string
                         ? p.tool_arg_string_value(p.until(PARAM_END))
                         : p.tool_arg_json_value(p.schema(p.json(),
                                                          "tool-" + name + "-arg-" + param_name + "-schema",
                                                          param_schema, false))) +
                    p.tool_arg_close(p.literal(PARAM_END)));

                auto named_arg = p.rule("tool-" + name + "-arg-" + param_name, arg);
                if (is_required) {
                    required_parsers.push_back(named_arg);
                } else {
                    optional_parsers.push_back(named_arg);
                }
            }

            common_peg_parser args_seq = p.eps();
            for (size_t i = 0; i < required_parsers.size(); i++) {
                if (i > 0) {
                    args_seq = args_seq + p.space();
                }
                args_seq = args_seq + required_parsers[i];
            }

            if (!optional_parsers.empty()) {
                common_peg_parser any_opt = p.choice();
                for (const auto & opt : optional_parsers) {
                    any_opt |= opt;
                }
                args_seq = args_seq + p.repeat(p.space() + any_opt, 0, -1);
            }

            common_peg_parser invoke_body = args_seq;
            auto func_parser = p.tool(
                p.tool_open(p.literal(INVOKE_START + " name=\"") +
                            p.tool_name(p.literal(name)) + p.literal("\">\n")) +
                invoke_body + p.space() +
                p.tool_close(p.literal(INVOKE_END)));

            tool_choice |= p.rule("tool-" + name, func_parser);
        });

        auto require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

        common_peg_parser tool_calls = p.eps();
        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call",
                p.literal(FC_START) + p.space() + tool_choice +
                p.zero_or_more(p.space() + tool_choice) + p.space() + p.literal(FC_END));
        } else {
            tool_calls = p.trigger_rule("tool-call",
                p.literal(FC_START) + p.space() + tool_choice + p.space() + p.literal(FC_END));
        }

        if (!require_tools) {
            tool_calls = p.optional(tool_calls);
        }

        auto content_before_tools = p.content(p.until(FC_START));
        return generation_prompt + reasoning + content_before_tools + tool_calls + end;
    });


    if (include_grammar) {
        data.grammar_lazy = !(has_response_format || (has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED));
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            common_chat_foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto         schema   = function.contains("parameters") ? function.at("parameters") : json::object();
                builder.resolve_refs(schema);
            });
            if (has_response_format) {
                auto schema = inputs.json_schema;
                builder.resolve_refs(schema);
            }
            parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, FC_START },
        };
    }

    data.parser = std::move(parser);

    return data;
}

// DeepSeek V3.2 format detection: template defines dsml_token and uses it for tool calls.
// The template source contains the token as a variable assignment, not as a literal in markup.
static bool detect(const std::string & src) {
    return src.find("dsml_token") != std::string::npos &&
        src.find("function_calls") != std::string::npos &&
        src.find("DSML") != std::string::npos;
}

const common_chat_parser_def common_chat_parser_deepseek_v3_2 = { "deepseek-v3.2", detect, common_chat_params_init_deepseek_v3_2 };
