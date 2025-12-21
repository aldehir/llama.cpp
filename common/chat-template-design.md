# Chat Template System Refactoring Design

## Overview

This document proposes a refactored interface for the chat template system that:
1. Encapsulates template initialization, formatting, and parsing in cohesive classes
2. Makes `ChatTemplate` reusable across requests (stateless)
3. Handles format detection dynamically based on per-request inputs
4. Provides a clean streaming parser abstraction

## Current Architecture Problems

1. **Disconnected lifecycle**: `common_chat_templates_init()`, `common_chat_templates_apply()`, and `common_chat_parse()` are separate functions with no clear relationship
2. **Manual syntax construction**: Users must manually create `common_chat_syntax` by copying fields from `common_chat_params`
3. **Format re-detection**: Each call to `common_chat_templates_apply_jinja()` re-detects the format by scanning template source
4. **No streaming abstraction**: Server manually accumulates text and re-parses on each token

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ChatTemplate                              │
│  (Immutable, reusable across requests)                          │
│  - Holds minja::chat_template(s)                                │
│  - Provides template metadata                                    │
│  - Factory for ChatFormatter                                     │
└─────────────────────┬───────────────────────────────────────────┘
                      │ creates (per request)
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ChatFormatter                              │
│  (Per-request, holds inputs + detected format)                  │
│  - Detects format based on template + inputs                    │
│  - Renders prompt via Jinja                                     │
│  - Provides generation params (grammar, triggers, stops)        │
│  - Factory for ChatOutputParser                                 │
└─────────────────────┬───────────────────────────────────────────┘
                      │ creates
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ChatOutputParser                             │
│  (Stateful, accumulates streaming output)                       │
│  - Parses incremental model output                              │
│  - Computes diffs for streaming responses                       │
│  - Manages tool call ID generation                              │
└─────────────────────────────────────────────────────────────────┘
```

## Class Definitions

### ChatTemplate

Immutable container for chat template(s). Created once per model, reused across all requests.

```cpp
class ChatTemplate {
public:
    //=== Factory Methods ===

    // Create from model metadata (reads chat_template from GGUF)
    static std::unique_ptr<ChatTemplate> from_model(
        const llama_model* model,
        const std::string& template_override = "",
        const std::string& bos_token_override = "",
        const std::string& eos_token_override = "");

    // Create from template string directly (for testing/custom templates)
    static std::unique_ptr<ChatTemplate> from_source(
        const std::string& template_source,
        const std::string& bos_token = "",
        const std::string& eos_token = "");

    //=== Template Metadata ===

    // Get template source (for debugging/logging)
    // variant: nullptr for default, "tool_use" for tool-specific template
    const std::string& source(const char* variant = nullptr) const;

    // Was template explicitly provided (vs defaulting to chatml)?
    bool was_explicit() const;

    // Does template support enable_thinking parameter?
    bool supports_thinking() const;

    // Get BOS/EOS tokens
    const std::string& bos_token() const;
    const std::string& eos_token() const;

    // Should BOS/EOS be added by caller?
    bool add_bos() const;
    bool add_eos() const;

    //=== Formatter Factory ===

    // Create a formatter for a specific request
    ChatFormatter formatter(ChatInputs inputs) const;

private:
    std::unique_ptr<minja::chat_template> template_default_;
    std::unique_ptr<minja::chat_template> template_tool_use_;
    std::string bos_token_;
    std::string eos_token_;
    bool add_bos_ = false;
    bool add_eos_ = false;
    bool has_explicit_template_ = false;
};
```

### ChatInputs

Input parameters for a single formatting request. Replaces `common_chat_templates_inputs`.

```cpp
struct ChatInputs {
    // Required
    std::vector<common_chat_msg> messages;

    // Generation control
    bool add_generation_prompt = true;

    // Tool calling
    std::vector<common_chat_tool> tools;
    common_chat_tool_choice tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    bool parallel_tool_calls = false;

    // Reasoning/thinking
    common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_NONE;
    bool enable_thinking = true;

    // Structured output (mutually exclusive)
    std::string grammar;
    std::string json_schema;

    // Advanced
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    std::map<std::string, std::string> extra_context;  // Additional Jinja variables
};
```

### ChatFormatter

Per-request formatter. Handles format detection, prompt rendering, and parser creation.

```cpp
class ChatFormatter {
public:
    //=== Prompt Rendering ===

    // Render the full prompt string
    std::string render() const;

    // Render a single message (for incremental chat)
    std::string render_single(
        const std::vector<common_chat_msg>& past_messages,
        const common_chat_msg& new_message,
        bool add_generation_prompt = true) const;

    //=== Generation Parameters ===

    // Get grammar for constrained generation (may be empty)
    const std::string& grammar() const;

    // Should grammar be applied lazily (after trigger)?
    bool grammar_lazy() const;

    // Patterns/words that trigger grammar application
    const std::vector<common_grammar_trigger>& grammar_triggers() const;

    // Tokens that should not be split by tokenizer
    const std::vector<std::string>& preserved_tokens() const;

    // Additional stop sequences
    const std::vector<std::string>& additional_stops() const;

    // Is thinking tag forced open in prompt?
    bool thinking_forced_open() const;

    //=== Format Info ===

    // Detected format enum
    common_chat_format format() const;

    // Human-readable format name
    const char* format_name() const;

    // Reasoning format being used
    common_reasoning_format reasoning_format() const;

    //=== Parser Factory ===

    // Create streaming parser for this format
    std::unique_ptr<ChatOutputParser> create_parser() const;

    // Convenience: parse complete (non-streaming) output
    common_chat_msg parse(const std::string& output) const;

    //=== Access to Raw Params (for compatibility) ===

    // Get legacy params struct
    const common_chat_params& params() const;

    // Get syntax for manual parsing
    common_chat_syntax syntax() const;

private:
    friend class ChatTemplate;

    ChatFormatter(const ChatTemplate* tmpl, ChatInputs inputs);

    void detect_format();  // Called lazily on first access

    const ChatTemplate* template_;
    ChatInputs inputs_;
    const ChatFormatHandler* handler_ = nullptr;  // Detected format handler
    mutable std::optional<common_chat_params> cached_params_;
};
```

### ChatOutputParser

Stateful streaming output parser. Accumulates text and provides incremental parsing.

```cpp
class ChatOutputParser {
public:
    //=== Streaming Interface ===

    // Add new text from model, returns current parsed message
    // is_partial: true during streaming, false on final update
    const common_chat_msg& update(const std::string& text_added, bool is_partial = true);

    // Get accumulated raw text
    const std::string& text() const;

    // Get current parsed message
    const common_chat_msg& message() const;

    //=== Diff Computation (for SSE streaming) ===

    // Compute diffs from previous state (call after update())
    std::vector<common_chat_msg_diff> compute_diffs();

    //=== Finalization ===

    // Mark parsing complete, validate final state
    void finish();

    // Was parsing completed successfully?
    bool is_finished() const;

    //=== Tool Call ID Management ===

    // Set generator for tool call IDs (called when tool call has no ID)
    void set_tool_call_id_generator(std::function<std::string()> generator);

    // Get generated tool call IDs (for caching across retries)
    const std::vector<std::string>& tool_call_ids() const;

private:
    friend class ChatFormatter;

    ChatOutputParser(common_chat_syntax syntax);

    common_chat_syntax syntax_;
    std::string accumulated_text_;
    common_chat_msg current_msg_;
    common_chat_msg prev_msg_;  // For diff computation
    std::vector<std::string> tool_call_ids_;
    std::function<std::string()> gen_tool_call_id_;
    bool finished_ = false;
};
```

### ChatFormatHandler (Strategy Pattern)

Base class for format-specific logic. Each supported format implements this interface.

```cpp
class ChatFormatHandler {
public:
    virtual ~ChatFormatHandler() = default;

    //=== Format Detection ===

    // Does this handler match the given template + inputs?
    virtual bool matches(
        const minja::chat_template& tmpl,
        const ChatInputs& inputs) const = 0;

    // Priority for detection (higher = checked first)
    // Allows more specific formats to take precedence
    virtual int priority() const { return 0; }

    //=== Initialization ===

    // Create generation params for this format
    virtual common_chat_params init_params(
        const minja::chat_template& tmpl,
        const ChatInputs& inputs,
        bool add_bos,
        bool add_eos) const = 0;

    //=== Parsing ===

    // Parse model output according to this format
    virtual void parse(common_chat_msg_parser& parser) const = 0;

    //=== Metadata ===

    virtual common_chat_format format() const = 0;
    virtual const char* name() const = 0;
};

// Registry for format handlers
class ChatFormatRegistry {
public:
    static ChatFormatRegistry& instance();

    // Register a handler (call during static init)
    void register_handler(std::unique_ptr<ChatFormatHandler> handler);

    // Find matching handler for template + inputs
    const ChatFormatHandler* detect(
        const minja::chat_template& tmpl,
        const ChatInputs& inputs) const;

    // Get handler by format enum (for compatibility)
    const ChatFormatHandler* get(common_chat_format format) const;

private:
    std::vector<std::unique_ptr<ChatFormatHandler>> handlers_;  // Sorted by priority
    std::unordered_map<common_chat_format, ChatFormatHandler*> by_format_;
};
```

## Usage Examples

### Basic Usage

```cpp
// === Server Initialization ===
auto chat_template = ChatTemplate::from_model(model);

// === Per-Request Handling ===
ChatInputs inputs;
inputs.messages = parse_messages(request["messages"]);
inputs.tools = parse_tools(request["tools"]);
inputs.enable_thinking = request.value("enable_thinking", true);

// Create formatter for this request
auto formatter = chat_template->formatter(std::move(inputs));

// Get the prompt to send to model
std::string prompt = formatter.render();

// Get generation parameters
sampling_params.grammar = formatter.grammar();
sampling_params.grammar_lazy = formatter.grammar_lazy();
sampling_params.grammar_triggers = formatter.grammar_triggers();
sampling_params.preserved_tokens = formatter.preserved_tokens();
for (const auto& stop : formatter.additional_stops()) {
    sampling_params.stop_sequences.push_back(stop);
}

// Log format info
LOG_INF("Using chat format: %s\n", formatter.format_name());
```

### Streaming Output Parsing

```cpp
// Create parser from formatter
auto parser = formatter.create_parser();
parser->set_tool_call_id_generator(generate_random_id);

// During token generation
while (generating) {
    std::string new_token = /* get next token */;

    // Update parser with new text
    const auto& msg = parser->update(new_token, /*is_partial=*/true);

    // Compute diffs for SSE streaming
    auto diffs = parser->compute_diffs();
    for (const auto& diff : diffs) {
        send_sse_event(diff_to_json(diff));
    }
}

// Finalize
parser->update("", /*is_partial=*/false);
parser->finish();

// Get final message
common_chat_msg final_message = parser->message();
```

### Non-Streaming Parsing

```cpp
// Simple case: parse complete output
std::string model_output = /* complete generation */;
common_chat_msg message = formatter.parse(model_output);
```

### Incremental Chat (Single Message Formatting)

```cpp
// Format just the new message (for chat continuation)
std::vector<common_chat_msg> history = /* previous messages */;
common_chat_msg new_msg = /* user's new message */;

std::string formatted = formatter.render_single(history, new_msg, /*add_gen_prompt=*/true);
```

## Format Handler Examples

### Llama 3.x Handler

```cpp
class Llama3xFormatHandler : public ChatFormatHandler {
public:
    bool matches(const minja::chat_template& tmpl,
                 const ChatInputs& inputs) const override {
        const auto& src = tmpl.source();
        return src.find("<|start_header_id|>ipython<|end_header_id|>") != std::string::npos;
    }

    int priority() const override { return 50; }

    common_chat_params init_params(
        const minja::chat_template& tmpl,
        const ChatInputs& inputs,
        bool add_bos, bool add_eos) const override
    {
        bool allow_python = tmpl.source().find("<|python_tag|>") != std::string::npos;
        // ... (existing common_chat_params_init_llama_3_x logic)
    }

    void parse(common_chat_msg_parser& parser) const override {
        // ... (existing Llama 3.x parsing logic from common_chat_parse)
    }

    common_chat_format format() const override {
        return COMMON_CHAT_FORMAT_LLAMA_3_X;
    }

    const char* name() const override { return "Llama 3.x"; }
};
```

### Generic Fallback Handler

```cpp
class GenericFormatHandler : public ChatFormatHandler {
public:
    bool matches(const minja::chat_template& tmpl,
                 const ChatInputs& inputs) const override {
        // Always matches as fallback
        return true;
    }

    int priority() const override { return -100; }  // Lowest priority

    // ... implementation

    common_chat_format format() const override {
        return COMMON_CHAT_FORMAT_GENERIC;
    }

    const char* name() const override { return "Generic"; }
};
```

## Migration Strategy

### Phase 1: New Classes Wrapping Existing Functions

1. Implement `ChatTemplate` calling `common_chat_templates_init()` internally
2. Implement `ChatFormatter` calling `common_chat_templates_apply()` internally
3. Implement `ChatOutputParser` calling `common_chat_parse()` internally
4. Add deprecation warnings to old functions

### Phase 2: Refactor Format Handlers

1. Create `ChatFormatHandler` base class
2. Move each `common_chat_params_init_*` function into handler's `init_params()`
3. Move parsing switch cases into handler's `parse()` methods
4. Build `ChatFormatRegistry` with all handlers

### Phase 3: Update Consumers

1. Update server to use new classes
2. Update any other consumers
3. Remove deprecated functions

## Compatibility

### Legacy Function Wrappers

For backwards compatibility, existing functions become thin wrappers:

```cpp
// Deprecated: use ChatTemplate::from_model()
common_chat_templates_ptr common_chat_templates_init(...) {
    return ChatTemplate::from_model(...)->to_legacy();
}

// Deprecated: use ChatFormatter::render() and ChatFormatter::params()
common_chat_params common_chat_templates_apply(...) {
    return ChatTemplate(...).formatter(inputs).params();
}

// Deprecated: use ChatFormatter::parse() or ChatOutputParser
common_chat_msg common_chat_parse(...) {
    // ... wrapper implementation
}
```

### ChatFormatter::syntax() for Manual Parsing

If users need the raw `common_chat_syntax` for custom parsing:

```cpp
common_chat_syntax ChatFormatter::syntax() const {
    return {
        .format = handler_->format(),
        .reasoning_format = inputs_.reasoning_format,
        .reasoning_in_content = /* computed */,
        .thinking_forced_open = params().thinking_forced_open,
        .parse_tool_calls = !inputs_.tools.empty(),
        .parser = params().parser,
    };
}
```

## File Organization

```
common/
├── chat.h                      # Existing (kept for compatibility)
├── chat.cpp                    # Existing (kept for compatibility)
├── chat-template.h             # New: ChatTemplate, ChatFormatter, ChatOutputParser
├── chat-template.cpp           # New: Implementation
├── chat-format-handler.h       # New: ChatFormatHandler base, ChatFormatRegistry
├── chat-format-handler.cpp     # New: Registry implementation
├── chat-formats/               # New: Per-format handlers
│   ├── llama-3x.cpp
│   ├── deepseek-r1.cpp
│   ├── hermes-pro.cpp
│   ├── mistral-nemo.cpp
│   ├── functionary.cpp
│   ├── generic.cpp
│   └── ...
├── chat-parser.h               # Existing (internal use)
├── chat-parser.cpp             # Existing (internal use)
└── ...
```

## Open Questions

1. **Thread safety**: Should `ChatTemplate` be thread-safe for concurrent `formatter()` calls? (Probably yes, since it's stateless)

2. **Format caching**: Should `ChatFormatter` cache the rendered prompt? (Probably not needed since render() is fast)

3. **Error handling**: Should parsing errors throw exceptions or return error codes? (Current code throws)

4. **PEG parser integration**: How should the PEG parser (`common_peg_arena`) be integrated? (Keep as optional field in syntax)

5. **Reasoning format auto-detection**: Should `COMMON_REASONING_FORMAT_AUTO` be resolved in `ChatFormatter` or left to handler?
