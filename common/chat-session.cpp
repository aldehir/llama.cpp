#include "chat-session.h"

common_chat_session::common_chat_session(common_chat_params params, const common_chat_session_opts & opts)
    : params_(std::move(params)), parser_params_(params_) {
    // the parser arena is only needed for parsing, drop the duplicate copy
    params_.parser = {};

    parser_params_.reasoning_format     = opts.reasoning_format;
    parser_params_.reasoning_in_content = opts.reasoning_in_content;
    parser_params_.parse_tool_calls     = opts.parse_tool_calls;
    parser_params_.is_continuation      = opts.is_continuation;
    parser_params_.echo                 = opts.echo;
    parser_params_.debug                = opts.debug;
}

common_chat_msg common_chat_session::parse(const std::string & output, bool is_partial) const {
    return common_chat_parse(output, is_partial, parser_params_);
}
