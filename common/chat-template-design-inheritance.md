# Chat Template System Refactoring Design (Inheritance Approach)

## Overview

This document proposes an alternative design using **inheritance** instead of the strategy pattern. Format-specific logic lives in subclasses rather than separate handler objects.

## Comparison: Strategy vs Inheritance

| Aspect | Strategy Pattern | Inheritance |
|--------|------------------|-------------|
| Format logic location | Separate handler objects | Subclasses |
| Selection mechanism | Registry lookup at runtime | Factory method returns correct subclass |
| Adding new formats | Register new handler | Add new subclass + factory case |
| Code organization | Handler per file | Subclass per file |
| Runtime flexibility | Can swap handlers | Fixed at construction |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ChatTemplate                              │
│  (Immutable, reusable across requests)                          │
│  - Holds minja::chat_template(s)                                │
│  - Factory for ChatFormatter subclasses                         │
└─────────────────────┬───────────────────────────────────────────┘
                      │ creates (per request)
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ChatFormatter (Abstract Base)                   │
│  - Common interface for all formats                             │
│  - Subclasses implement format-specific logic                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │
│  │ Llama3x      │ │ DeepSeekR1   │ │ HermesPro    │ ...         │
│  │ Formatter    │ │ Formatter    │ │ Formatter    │             │
│  └──────────────┘ └──────────────┘ └──────────────┘             │
└─────────────────────┬───────────────────────────────────────────┘
                      │ creates
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ChatOutputParser                             │
│  (Stateful, accumulates streaming output)                       │
│  - Also uses inheritance for format-specific parsing            │
└─────────────────────────────────────────────────────────────────┘
```

## Class Definitions

### ChatTemplate

Immutable container for chat template(s). Factory for format-specific formatters.

```cpp
class ChatTemplate {
public:
    //=== Factory Methods ===

    static std::unique_ptr<ChatTemplate> from_model(
        const llama_model* model,
        const std::string& template_override = "",
        const std::string& bos_token_override = "",
        const std::string& eos_token_override = "");

    static std::unique_ptr<ChatTemplate> from_source(
        const std::string& template_source,
        const std::string& bos_token = "",
        const std::string& eos_token = "");

    //=== Template Metadata ===

    const std::string& source(const char* variant = nullptr) const;
    bool was_explicit() const;
    bool supports_thinking() const;
    const std::string& bos_token() const;
    const std::string& eos_token() const;
    bool add_bos() const;
    bool add_eos() const;

    //=== Formatter Factory ===

    // Creates the appropriate ChatFormatter subclass based on template + inputs
    std::unique_ptr<ChatFormatter> formatter(ChatInputs inputs) const;

private:
    // Format detection - returns which subclass to instantiate
    common_chat_format detect_format(const ChatInputs& inputs) const;

    std::unique_ptr<minja::chat_template> template_default_;
    std::unique_ptr<minja::chat_template> template_tool_use_;
    std::string bos_token_;
    std::string eos_token_;
    bool add_bos_ = false;
    bool add_eos_ = false;
    bool has_explicit_template_ = false;
};
```

### ChatFormatter (Abstract Base Class)

```cpp
class ChatFormatter {
public:
    virtual ~ChatFormatter() = default;

    //=== Prompt Rendering ===

    // Render the full prompt string
    std::string render() const;

    // Render a single message (for incremental chat)
    std::string render_single(
        const std::vector<common_chat_msg>& past_messages,
        const common_chat_msg& new_message,
        bool add_generation_prompt = true) const;

    //=== Generation Parameters ===

    const std::string& grammar() const { return params_.grammar; }
    bool grammar_lazy() const { return params_.grammar_lazy; }
    const std::vector<common_grammar_trigger>& grammar_triggers() const { return params_.grammar_triggers; }
    const std::vector<std::string>& preserved_tokens() const { return params_.preserved_tokens; }
    const std::vector<std::string>& additional_stops() const { return params_.additional_stops; }
    bool thinking_forced_open() const { return params_.thinking_forced_open; }

    //=== Format Info ===

    virtual common_chat_format format() const = 0;
    virtual const char* format_name() const = 0;

    common_reasoning_format reasoning_format() const { return inputs_.reasoning_format; }

    //=== Parser Factory ===

    // Create streaming parser - returns format-specific subclass
    virtual std::unique_ptr<ChatOutputParser> create_parser() const = 0;

    // Convenience: parse complete output
    common_chat_msg parse(const std::string& output) const;

    //=== Compatibility ===

    const common_chat_params& params() const { return params_; }
    common_chat_syntax syntax() const;

protected:
    // Constructor for subclasses
    ChatFormatter(const ChatTemplate* tmpl, ChatInputs inputs);

    // Subclasses must implement: build format-specific params
    virtual void init_params() = 0;

    // Helpers for subclasses
    const minja::chat_template& get_template() const;
    std::string apply_template(
        const std::optional<nlohmann::json>& messages_override = std::nullopt,
        const std::optional<nlohmann::json>& tools_override = std::nullopt,
        const std::optional<nlohmann::json>& extra_context = std::nullopt) const;

    const ChatTemplate* template_;
    ChatInputs inputs_;
    common_chat_params params_;
};
```

### Format-Specific Formatter Subclasses

```cpp
//=== Llama 3.x ===

class Llama3xFormatter : public ChatFormatter {
public:
    Llama3xFormatter(const ChatTemplate* tmpl, ChatInputs inputs);

    common_chat_format format() const override { return COMMON_CHAT_FORMAT_LLAMA_3_X; }
    const char* format_name() const override { return "Llama 3.x"; }

    std::unique_ptr<ChatOutputParser> create_parser() const override;

protected:
    void init_params() override;

private:
    bool allow_builtin_tools_ = false;
};

//=== DeepSeek R1 ===

class DeepSeekR1Formatter : public ChatFormatter {
public:
    DeepSeekR1Formatter(const ChatTemplate* tmpl, ChatInputs inputs);

    common_chat_format format() const override { return COMMON_CHAT_FORMAT_DEEPSEEK_R1; }
    const char* format_name() const override { return "DeepSeek R1"; }

    std::unique_ptr<ChatOutputParser> create_parser() const override;

protected:
    void init_params() override;
};

//=== Hermes 2 Pro ===

class HermesProFormatter : public ChatFormatter {
public:
    HermesProFormatter(const ChatTemplate* tmpl, ChatInputs inputs);

    common_chat_format format() const override { return COMMON_CHAT_FORMAT_HERMES_2_PRO; }
    const char* format_name() const override { return "Hermes 2 Pro"; }

    std::unique_ptr<ChatOutputParser> create_parser() const override;

protected:
    void init_params() override;
};

//=== Mistral Nemo ===

class MistralNemoFormatter : public ChatFormatter {
public:
    MistralNemoFormatter(const ChatTemplate* tmpl, ChatInputs inputs);

    common_chat_format format() const override { return COMMON_CHAT_FORMAT_MISTRAL_NEMO; }
    const char* format_name() const override { return "Mistral Nemo"; }

    std::unique_ptr<ChatOutputParser> create_parser() const override;

protected:
    void init_params() override;
};

//=== Generic (Fallback) ===

class GenericFormatter : public ChatFormatter {
public:
    GenericFormatter(const ChatTemplate* tmpl, ChatInputs inputs);

    common_chat_format format() const override { return COMMON_CHAT_FORMAT_GENERIC; }
    const char* format_name() const override { return "Generic"; }

    std::unique_ptr<ChatOutputParser> create_parser() const override;

protected:
    void init_params() override;
};

//=== Content Only (No Tools) ===

class ContentOnlyFormatter : public ChatFormatter {
public:
    ContentOnlyFormatter(const ChatTemplate* tmpl, ChatInputs inputs);

    common_chat_format format() const override { return COMMON_CHAT_FORMAT_CONTENT_ONLY; }
    const char* format_name() const override { return "Content Only"; }

    std::unique_ptr<ChatOutputParser> create_parser() const override;

protected:
    void init_params() override;
};
```

### ChatOutputParser (Abstract Base Class)

```cpp
class ChatOutputParser {
public:
    virtual ~ChatOutputParser() = default;

    //=== Streaming Interface ===

    const common_chat_msg& update(const std::string& text_added, bool is_partial = true);
    const std::string& text() const { return accumulated_text_; }
    const common_chat_msg& message() const { return current_msg_; }

    //=== Diff Computation ===

    std::vector<common_chat_msg_diff> compute_diffs();

    //=== Finalization ===

    void finish();
    bool is_finished() const { return finished_; }

    //=== Tool Call ID Management ===

    void set_tool_call_id_generator(std::function<std::string()> generator);
    const std::vector<std::string>& tool_call_ids() const { return tool_call_ids_; }

protected:
    ChatOutputParser(common_chat_syntax syntax);

    // Subclasses implement format-specific parsing
    virtual void do_parse(common_chat_msg_parser& parser) = 0;

    common_chat_syntax syntax_;
    std::string accumulated_text_;
    common_chat_msg current_msg_;
    common_chat_msg prev_msg_;
    std::vector<std::string> tool_call_ids_;
    std::function<std::string()> gen_tool_call_id_;
    bool finished_ = false;
};
```

### Format-Specific Parser Subclasses

```cpp
class Llama3xOutputParser : public ChatOutputParser {
public:
    Llama3xOutputParser(common_chat_syntax syntax);

protected:
    void do_parse(common_chat_msg_parser& parser) override;
};

class DeepSeekR1OutputParser : public ChatOutputParser {
public:
    DeepSeekR1OutputParser(common_chat_syntax syntax);

protected:
    void do_parse(common_chat_msg_parser& parser) override;
};

class HermesProOutputParser : public ChatOutputParser {
public:
    HermesProOutputParser(common_chat_syntax syntax);

protected:
    void do_parse(common_chat_msg_parser& parser) override;
};

// ... etc for each format
```

## Factory Implementation

### ChatTemplate::formatter()

```cpp
std::unique_ptr<ChatFormatter> ChatTemplate::formatter(ChatInputs inputs) const {
    common_chat_format fmt = detect_format(inputs);

    switch (fmt) {
        case COMMON_CHAT_FORMAT_LLAMA_3_X:
        case COMMON_CHAT_FORMAT_LLAMA_3_X_WITH_BUILTIN_TOOLS:
            return std::make_unique<Llama3xFormatter>(this, std::move(inputs));

        case COMMON_CHAT_FORMAT_DEEPSEEK_R1:
            return std::make_unique<DeepSeekR1Formatter>(this, std::move(inputs));

        case COMMON_CHAT_FORMAT_HERMES_2_PRO:
            return std::make_unique<HermesProFormatter>(this, std::move(inputs));

        case COMMON_CHAT_FORMAT_MISTRAL_NEMO:
            return std::make_unique<MistralNemoFormatter>(this, std::move(inputs));

        case COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2:
            return std::make_unique<FunctionaryV32Formatter>(this, std::move(inputs));

        case COMMON_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1:
            return std::make_unique<FunctionaryV31Formatter>(this, std::move(inputs));

        // ... all other formats ...

        case COMMON_CHAT_FORMAT_CONTENT_ONLY:
            return std::make_unique<ContentOnlyFormatter>(this, std::move(inputs));

        case COMMON_CHAT_FORMAT_GENERIC:
        default:
            return std::make_unique<GenericFormatter>(this, std::move(inputs));
    }
}
```

### ChatTemplate::detect_format()

```cpp
common_chat_format ChatTemplate::detect_format(const ChatInputs& inputs) const {
    const auto& tmpl = (inputs.tools.empty() || !template_tool_use_)
        ? *template_default_
        : *template_tool_use_;
    const auto& src = tmpl.source();

    // Check formats in priority order (most specific first)

    // DeepSeek R1
    if (src.find("<｜tool▁calls▁begin｜>") != std::string::npos
        && inputs.json_schema.empty()) {
        return COMMON_CHAT_FORMAT_DEEPSEEK_R1;
    }

    // Command R7B
    if (src.find("<|END_THINKING|><|START_ACTION|>") != std::string::npos
        && inputs.json_schema.empty()) {
        return COMMON_CHAT_FORMAT_COMMAND_R7B;
    }

    // Hermes 2 Pro / Qwen 2.5
    if (src.find("<tool_call>") != std::string::npos
        && inputs.json_schema.empty()) {
        return COMMON_CHAT_FORMAT_HERMES_2_PRO;
    }

    // Llama 3.x
    if (src.find("<|start_header_id|>ipython<|end_header_id|>") != std::string::npos) {
        return COMMON_CHAT_FORMAT_LLAMA_3_X;
    }

    // Mistral Nemo
    if (src.find("[TOOL_CALLS]") != std::string::npos && !inputs.tools.empty()) {
        return COMMON_CHAT_FORMAT_MISTRAL_NEMO;
    }

    // ... etc ...

    // No tools = content only
    if (inputs.tools.empty() || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
        return COMMON_CHAT_FORMAT_CONTENT_ONLY;
    }

    // Fallback
    return COMMON_CHAT_FORMAT_GENERIC;
}
```

## Subclass Implementation Example

### Llama3xFormatter

```cpp
Llama3xFormatter::Llama3xFormatter(const ChatTemplate* tmpl, ChatInputs inputs)
    : ChatFormatter(tmpl, std::move(inputs))
{
    allow_builtin_tools_ = get_template().source().find("<|python_tag|>") != std::string::npos;
    init_params();
}

void Llama3xFormatter::init_params() {
    // Move logic from common_chat_params_init_llama_3_x here

    if (!inputs_.tools.empty()) {
        params_.grammar_lazy = inputs_.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;

        params_.grammar = build_grammar([&](const common_grammar_builder& builder) {
            std::vector<std::string> tool_rules;

            for (const auto& tool : inputs_.tools) {
                // Build grammar rules for each tool...
            }

            params_.grammar_triggers.push_back({
                COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
                "(\\{\\s*(?:\"type\"\\s*:\\s*\"function\"\\s*,\\s*)?\"name\"\\s*:\\s*\")[\\s\\S]*",
            });

            if (allow_builtin_tools_) {
                params_.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<|python_tag|>"});
                params_.preserved_tokens.push_back("<|python_tag|>");
            }

            builder.add_rule("root", string_join(tool_rules, " | "));
            params_.additional_stops.push_back("<|eom_id|>");
        });
    }

    // Apply template with Llama 3.x specific context
    nlohmann::json extra_context = {
        {"date_string", format_time(inputs_.now, "%d %b %Y")},
        {"tools_in_user_message", false},
        {"builtin_tools", allow_builtin_tools_ ? get_builtin_tools() : nlohmann::json()},
    };

    params_.prompt = apply_template(std::nullopt, std::nullopt, extra_context);
}

std::unique_ptr<ChatOutputParser> Llama3xFormatter::create_parser() const {
    return std::make_unique<Llama3xOutputParser>(syntax());
}
```

### Llama3xOutputParser

```cpp
Llama3xOutputParser::Llama3xOutputParser(common_chat_syntax syntax)
    : ChatOutputParser(std::move(syntax))
{}

void Llama3xOutputParser::do_parse(common_chat_msg_parser& parser) {
    // Move Llama 3.x parsing logic from common_chat_parse here

    static const common_regex function_regex(
        R"(\{"name"\s*:\s*"([^"]+)"\s*,\s*"parameters"\s*:\s*)"
    );

    // Parse JSON tool calls
    if (auto match = parser.try_find_regex(function_regex)) {
        std::string name = parser.str(match->groups[1]);
        auto args = parser.consume_json_with_dumped_args({{}});
        parser.add_tool_call(name, "", args.value.dump());
        // ... continue parsing
    } else {
        parser.add_content(parser.consume_rest());
    }
}
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

// Create formatter (returns appropriate subclass)
auto formatter = chat_template->formatter(std::move(inputs));

// Use polymorphically
std::string prompt = formatter->render();
LOG_INF("Format: %s\n", formatter->format_name());

// Get generation params
auto& grammar = formatter->grammar();
auto& triggers = formatter->grammar_triggers();
```

### Streaming Parsing

```cpp
// Create parser (returns appropriate subclass)
auto parser = formatter->create_parser();
parser->set_tool_call_id_generator(generate_id);

while (generating) {
    std::string token = get_next_token();
    parser->update(token, true);

    for (const auto& diff : parser->compute_diffs()) {
        send_sse(diff);
    }
}

parser->finish();
auto message = parser->message();
```

## File Organization

```
common/
├── chat-template.h                 # ChatTemplate class
├── chat-template.cpp               # ChatTemplate implementation
├── chat-formatter.h                # ChatFormatter base + subclass declarations
├── chat-formatter.cpp              # ChatFormatter base implementation
├── chat-formatter/                 # Subclass implementations
│   ├── llama-3x.cpp
│   ├── deepseek-r1.cpp
│   ├── hermes-pro.cpp
│   ├── mistral-nemo.cpp
│   ├── functionary.cpp
│   ├── generic.cpp
│   ├── content-only.cpp
│   └── ...
├── chat-output-parser.h            # ChatOutputParser base + subclass declarations
├── chat-output-parser.cpp          # ChatOutputParser base implementation
├── chat-output-parser/             # Subclass implementations
│   ├── llama-3x.cpp
│   ├── deepseek-r1.cpp
│   └── ...
└── ...
```

## Trade-offs vs Strategy Pattern

### Advantages of Inheritance

1. **Simpler mental model**: "Llama3xFormatter IS-A ChatFormatter"
2. **No registry needed**: Factory switch statement is explicit
3. **Easier debugging**: Call stack shows exact type
4. **IDE support**: Better autocomplete for subclass-specific methods
5. **Compile-time checking**: Missing format = missing case in switch

### Disadvantages of Inheritance

1. **Factory maintenance**: Must update switch statement for new formats
2. **Less runtime flexibility**: Can't swap implementations dynamically
3. **Tighter coupling**: Subclasses depend on base class details
4. **More files**: Two subclasses per format (Formatter + Parser)

### When to Prefer Inheritance

- Formats are well-known and change infrequently
- You want explicit control over which formats exist
- Debugging and code navigation are priorities
- Team prefers classical OOP patterns

### When to Prefer Strategy

- Formats may be added by plugins/extensions
- You want to test with mock handlers
- Runtime format switching is needed
- Team prefers composition over inheritance

## Migration Path

1. Create base classes `ChatFormatter` and `ChatOutputParser`
2. Create subclasses for each format, moving logic from:
   - `common_chat_params_init_*` → `XxxFormatter::init_params()`
   - `common_chat_parse` switch cases → `XxxOutputParser::do_parse()`
3. Implement `ChatTemplate::formatter()` factory
4. Update server to use new classes
5. Deprecate old free functions
