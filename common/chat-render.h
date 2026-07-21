#pragma once

#include "chat.h"

#include "nlohmann/json.hpp"

#include <chrono>
#include <optional>
#include <string>

struct common_chat_render_inputs {
    nlohmann::ordered_json                messages;
    nlohmann::ordered_json                tools;
    common_chat_tool_choice               tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    nlohmann::ordered_json                json_schema;
    bool                                  parallel_tool_calls = true;
    common_reasoning_format               reasoning_format    = COMMON_REASONING_FORMAT_AUTO;
    std::string                           grammar;
    bool                                  add_generation_prompt  = false;
    common_chat_continuation              continue_final_message = COMMON_CHAT_CONTINUATION_NONE;
    common_chat_msg                       continue_msg;
    bool                                  enable_thinking        = true;
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    nlohmann::ordered_json                extra_context;
    bool                                  add_bos = false;
    bool                                  add_eos = false;

    bool has_continuation() const {
        return continue_final_message != COMMON_CHAT_CONTINUATION_NONE && !continue_msg.empty();
    }
};

std::string common_chat_template_direct_apply(
    const common_chat_template &                  tmpl,
    const common_chat_render_inputs &              inputs,
    const std::optional<nlohmann::ordered_json> &  messages_override  = std::nullopt,
    const std::optional<nlohmann::ordered_json> &  tools_override     = std::nullopt,
    const std::optional<nlohmann::ordered_json> &  additional_context = std::nullopt);

std::string common_chat_template_generation_prompt(
    const common_chat_template &                  tmpl,
    const common_chat_render_inputs &              inputs,
    const std::optional<nlohmann::ordered_json> &  messages_override  = std::nullopt,
    const std::optional<nlohmann::ordered_json> &  tools_override     = std::nullopt,
    const std::optional<nlohmann::ordered_json> &  additional_context = std::nullopt);

std::optional<common_chat_params> common_chat_try_specialized_template(
        const common_chat_template &      tmpl,
        const std::string &               src,
        common_chat_render_inputs &       params);
