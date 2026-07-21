// Per-request chat session: binds the applied chat template output (prompt,
// grammar, parser) to the request's parse options.

#pragma once

#include "chat.h"

#include <memory>
#include <string>

// parse options derived from the request rather than the template
struct common_chat_session_opts {
    common_reasoning_format reasoning_format     = COMMON_REASONING_FORMAT_NONE;
    bool                    reasoning_in_content = false;
    bool                    parse_tool_calls     = true;
    bool                    is_continuation      = false;
    bool                    echo                 = false;
    bool                    debug                = false;
};

// Instantiated once per request and shared by the tasks serving it.
class common_chat_session {
  public:
    common_chat_session() = default;
    common_chat_session(common_chat_params params, const common_chat_session_opts & opts);

    const common_chat_params        & params() const { return params_; }
    const common_chat_parser_params & parser_params() const { return parser_params_; }

    // parse the accumulated model output
    common_chat_msg parse(const std::string & output, bool is_partial) const;

  private:
    common_chat_params        params_;
    common_chat_parser_params parser_params_;
};

using common_chat_session_ptr = std::shared_ptr<const common_chat_session>;
