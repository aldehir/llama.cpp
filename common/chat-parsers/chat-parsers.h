// Registry of model-specific chat parsers. Each supported model template
// lives in its own file under common/chat-parsers/ and exports a single
// common_chat_parser_def describing how to detect and handle it.

#pragma once

#include "chat.h"
#include "chat-render.h"

#include <functional>
#include <string>
#include <vector>

struct common_chat_parser_def {
    const char * name;
    // returns true if the template source belongs to this model
    bool (*detect)(const std::string & src);
    // builds prompt, grammar and parser for a request. May adjust inputs
    // (e.g. message compatibility workarounds) before rendering.
    common_chat_params (*init)(const common_chat_template & tmpl, common_chat_render_inputs & inputs);
};

extern const common_chat_parser_def common_chat_parser_ministral_3;
extern const common_chat_parser_def common_chat_parser_gpt_oss;
extern const common_chat_parser_def common_chat_parser_functionary_v3_2;
extern const common_chat_parser_def common_chat_parser_kimi_k2;
extern const common_chat_parser_def common_chat_parser_cohere2moe;
extern const common_chat_parser_def common_chat_parser_lfm2;
extern const common_chat_parser_def common_chat_parser_lfm2_5;
extern const common_chat_parser_def common_chat_parser_gigachat_v3;
extern const common_chat_parser_def common_chat_parser_deepseek_v3_2;
extern const common_chat_parser_def common_chat_parser_gemma4;
extern const common_chat_parser_def common_chat_parser_minicpm5;

// all known parsers, ordered by detection precedence
const std::vector<common_chat_parser_def> & common_chat_parsers();

// shared helpers for parser implementations

void common_chat_foreach_function(const json & tools, const std::function<void(const json &)> & fn);

void common_chat_foreach_parameter(const json &                                                         function,
                                   const std::function<void(const std::string &, const json &, bool)> & fn);
