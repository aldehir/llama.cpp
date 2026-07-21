#include "chat-parsers.h"

#include "log.h"

#include <set>

const std::vector<common_chat_parser_def> & common_chat_parsers() {
    // ordered by detection precedence
    static const std::vector<common_chat_parser_def> parsers = {
        common_chat_parser_ministral_3,
        common_chat_parser_gpt_oss,
        common_chat_parser_functionary_v3_2,
        common_chat_parser_kimi_k2,
        common_chat_parser_cohere2moe,
        common_chat_parser_lfm2,
        common_chat_parser_lfm2_5,
        common_chat_parser_gigachat_v3,
        common_chat_parser_deepseek_v3_2,
        common_chat_parser_gemma4,
        common_chat_parser_minicpm5,
    };
    return parsers;
}

void common_chat_foreach_function(const json & tools, const std::function<void(const json &)> & fn) {
    for (const auto & tool : tools) {
        if (!tool.contains("type") || tool.at("type") != "function" || !tool.contains("function")) {
            LOG_INF("Skipping tool without function: %s", tool.dump(2).c_str());
            continue;
        }
        fn(tool);
    }
}

void common_chat_foreach_parameter(const json &                                                         function,
                                   const std::function<void(const std::string &, const json &, bool)> & fn) {
    if (!function.contains("parameters") || !function.at("parameters").is_object()) {
        return;
    }
    const auto & params = function.at("parameters");
    if (!params.contains("properties") || !params.at("properties").is_object()) {
        return;
    }
    const auto &          props = params.at("properties");
    std::set<std::string> required;
    if (params.contains("required") && params.at("required").is_array()) {
        params.at("required").get_to(required);
    }
    for (const auto & [name, prop] : props.items()) {
        bool is_required = (required.find(name) != required.end());
        fn(name, prop, is_required);
    }
}
