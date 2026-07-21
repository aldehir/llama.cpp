# Sketch: object-oriented chat handler / session API

Status: design sketch, not implemented.

## Problem with the current shape

The chat facility in `common/` is a bag of free functions that the server calls
per request:

1. Startup: `common_chat_templates_init()` returns an opaque
   `common_chat_templates` holding up to two `common_chat_template` variants
   (`template_default`, `template_tool_use`).
2. Per request: the server flattens the OAI request into
   `common_chat_templates_inputs` and calls `common_chat_templates_apply()`.
3. `common_chat_templates_apply_jinja()` re-detects the dialect on every
   request by substring-sniffing the template source
   (`common_chat_try_specialized_template()`), then delegates to one of the
   `common_chat_params_init_<dialect>()` free functions, or falls back to the
   differential autoparser - which re-runs `analyze_template()` (multiple
   probe renders + diffing) on every request.
4. The resulting `common_chat_params` is flattened into the `llama_params`
   JSON blob (`chat_format` as an int, `chat_parser` as a serialized PEG
   arena, plus grammar triggers, preserved tokens, delimiters, ...), then
   re-parsed on the task side by `server-schema.cpp` into
   `common_chat_parser_params`, which `task_result_state` feeds to the
   stateless `common_chat_parse()` on every stream update.

Issues:

- Dialect detection and template analysis are startup-time facts recomputed
  per request.
- The prompt-side result and the parse-side params are the same logical thing
  but travel through a lossy JSON round trip and two different structs.
- Adding a dialect means touching the detection chain, the init function, the
  format enum, and the parser dispatch - nothing groups them.
- The `template_tool_use` variant is special-cased inside apply; there is no
  room for more variants.

## Proposed shape

Two core types, replacing the free-function relay:

- `common_chat_handler` - one per jinja template (per variant). Constructed
  once at startup. Detection and template analysis happen in its factory.
  Immutable and shareable afterwards.
- `common_chat_session` - created by the handler for one request
  (`handler.new_session(req)`). Owns everything derived from that request:
  the rendered prompt, sampling constraints, and the incremental parse state
  for the reply.

Plus a thin router:

- `common_chat_handler_set` - owns one handler per template shipped in the
  GGUF ("default", "tool_use", later others). Replaces
  `common_chat_templates_init()`. Selection is a small policy instead of a
  special case buried in apply.

### Header sketch

```cpp
// common/chat-handler.h

// ~= today's common_chat_templates_inputs, minus use_jinja (legacy path
// stays on the old entry point during migration)
struct common_chat_request {
    std::vector<common_chat_msg>          messages;
    std::vector<common_chat_tool>         tools;
    common_chat_tool_choice               tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    std::string                           json_schema;   // response_format
    std::string                           grammar;
    bool                                  parallel_tool_calls = false;
    common_reasoning_format               reasoning_format = COMMON_REASONING_FORMAT_NONE;
    bool                                  enable_thinking  = true;
    bool                                  add_generation_prompt = true;
    common_chat_continuation              continue_final_message = COMMON_CHAT_CONTINUATION_NONE;
    std::map<std::string, std::string>    chat_template_kwargs;
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    bool                                  force_pure_content = false;
};

// grammar/trigger bundle, split out of common_chat_params
struct common_chat_constraints {
    std::string                         grammar;
    bool                                grammar_lazy = false;
    std::vector<common_grammar_trigger> grammar_triggers;
    std::vector<std::string>            preserved_tokens;
    std::vector<std::string>            additional_stops;
};

class common_chat_session;
using common_chat_session_ptr = std::unique_ptr<common_chat_session>;

class common_chat_handler {
  public:
    struct options {
        bool add_bos = false;
        bool add_eos = false;
        // server-level defaults (reasoning_format, default kwargs, ...)
    };

    virtual ~common_chat_handler() = default;

    // Detection happens here, once. Walks a registry of {name, detect(src),
    // make(tmpl, opts)} entries (today's common_chat_try_specialized_template
    // chain); falls back to the autoparser-backed handler, whose constructor
    // runs autoparser::analyze_template() a single time.
    static std::unique_ptr<common_chat_handler> create(
        std::unique_ptr<common_chat_template> tmpl, const options & opts);

    virtual const char * name() const = 0;  // "gpt-oss", "ministral-3", "auto", ...

    const common_chat_template & tmpl() const;
    const chat_template_caps &   caps() const;

    // computed once in create(), no more per-call probe render
    virtual bool supports_thinking() const;

    // absorb today's stateless helpers
    std::string format_example(const std::map<std::string, std::string> & kwargs) const;
    std::string format_single(const std::vector<common_chat_msg> & past,
                              const common_chat_msg & msg, bool add_ass) const;
    common_chat_prompt_preset asr_prompt() const;

    // the heart of the API
    virtual common_chat_session_ptr new_session(const common_chat_request & req) const = 0;

  protected:
    std::unique_ptr<common_chat_template> tmpl_;
    options                               opts_;
};

// One request bound to one handler. Prompt side is computed in the
// constructor; reply side is incremental parse state owned here, so the
// format-enum + serialized-parser round trip through JSON disappears.
class common_chat_session {
  public:
    virtual ~common_chat_session() = default;

    // prompt side
    const std::string &                prompt() const;
    const std::string &                generation_prompt() const;
    const common_chat_constraints &    constraints() const;
    const common_chat_msg_delimiters & message_delimiters() const;

    bool                supports_thinking() const;
    const std::string & thinking_start_tag() const;
    const std::string & thinking_end_tag() const;

    // reply side; replaces common_chat_parse(text, partial, params).
    // `accumulated` is the full generated text so far, same contract as today.
    virtual common_chat_msg parse(const std::string & accumulated, bool is_partial) = 0;

    // streaming convenience built on parse(): keeps the previous msg and
    // returns diffs (today spread across task_result_state)
    std::vector<common_chat_msg_diff> update(const std::string & accumulated, bool is_partial);

    // escape hatch if a task ever needs to cross a serialization boundary;
    // common_peg_arena already supports save()/load()
    std::string save() const;

  protected:
    common_chat_msg msg_prev_;  // for update()
};

// Owns one handler per template variant in the GGUF. Replaces
// common_chat_templates_init(). Each variant is its own fully-detected
// handler rather than a special case inside apply.
class common_chat_handler_set {
  public:
    static std::unique_ptr<common_chat_handler_set> init(
        const struct llama_model * model,
        const std::string &        chat_template_override,
        const common_chat_handler::options & opts);

    // policy formerly buried in apply_jinja:
    //   !req.tools.empty() && has("tool_use") ? tool_use : default
    const common_chat_handler & select(const common_chat_request & req) const;

    const common_chat_handler * get(const std::string & variant) const;  // may be null
    bool was_explicit() const;
    std::map<std::string, bool> get_caps() const;  // for /props

  private:
    // "default" always present; "tool_use" optional; room for more
    std::map<std::string, std::unique_ptr<common_chat_handler>> handlers_;
};
```

### Dialect handlers

Each `common_chat_params_init_<dialect>()` becomes a subclass whose
`new_session()` contains the same body, minus anything request-independent,
which moves to the constructor:

```cpp
class chat_handler_gpt_oss : public common_chat_handler {
  public:
    const char * name() const override { return "gpt-oss"; }
    common_chat_session_ptr new_session(const common_chat_request & req) const override;
};

class chat_handler_auto : public common_chat_handler {
  public:
    chat_handler_auto(...) {
        analysis_.analyze_template(*tmpl_);  // once, at startup
    }
    common_chat_session_ptr new_session(const common_chat_request & req) const override {
        // per-request: only peg_generator::generate_parser(tmpl, params, analysis_)
    }
  private:
    autoparser::autoparser analysis_;
};
```

The shared request -> `autoparser::generation_params` translation and the
workarounds (developer-role mapping, system-role fallback, non-null content,
StepFun trim, ...) live in a protected helper on the base class; the
source-sniffing workarounds can become virtuals that specific handlers
override instead of `src.find(...)` at apply time.

Most sessions can be instances of one concrete `common_chat_session_impl`
(fields = today's `common_chat_params` + a loaded `common_peg_arena` +
`common_chat_parser_params`); dialects only subclass the session when they
need custom parse behavior beyond what the PEG parser expresses.

### Server integration

- `server-context` startup: build the `common_chat_handler_set`; detection
  and autoparser analysis run here, once, and failures surface at load time
  instead of on the first request. Log the chosen handler name per variant;
  report it in `/props`.
- `oaicompat_chat_params_parse` (server-common.cpp): build
  `common_chat_request`, then
  `session = handler_set.select(req).new_session(req)`. Attach the session
  to the task (`server_task.chat_session`, a `unique_ptr`/`shared_ptr`)
  instead of flattening `chat_format` / `chat_parser` / `grammar_triggers` /
  `preserved_tokens` / `generation_prompt` / `message_delimiters` into
  `llama_params` JSON.
- `server-schema.cpp`: drop the deserialization fields above; sampling init
  reads `session->constraints()` directly.
- `server-task.cpp` / `task_result_state`: replace
  `common_chat_parser_params` + `common_chat_parse()` with
  `session->parse(...)` / `session->update(...)`. The session lives exactly
  as long as the task, which is the lifetime the parse state already
  implicitly had.

### Multiple templates per GGUF

Rather than teaching one handler to juggle variants, each variant template
becomes its own handler at init time (`handlers_["default"]`,
`handlers_["tool_use"]`, ...). A GGUF that ships several dialect templates is
then N independently-detected handlers plus a selection policy, instead of
conditionals inside apply. New variant kinds only extend `select()`.

## Migration plan

Big-bang rewrite is not realistic; `chat.cpp` is ~3k lines with consumers in
server, tools/completion, diffusion-cli, and tests.

1. Introduce `chat-handler.{h,cpp}` with `handler_set` / `handler` / `session`
   implemented as wrappers over the existing functions (`new_session()` calls
   `common_chat_templates_apply()` internally; `session::parse()` wraps
   `common_chat_parse()`). Hoist autoparser analysis caching into
   `chat_handler_auto`. Port the server to the new API. Old entry points stay
   for other consumers and tests.
2. Split the `common_chat_params_init_*` bodies into handler subclasses and
   move detection into `common_chat_handler::create()`'s registry (likely one
   file per dialect under `common/chat-dialects/`).
   `common_chat_try_specialized_template()` and the apply-time sniffing go
   away.
3. Remove the JSON round trip from the server task schema; port remaining
   consumers (tools/completion, diffusion-cli, tests); shrink or retire the
   old free-function API. `common_chat_format` may reduce further or vanish
   once nothing dispatches on it.

## Notes / caveats

- Handlers must stay immutable after create() (shared across slots/threads);
  all mutable state belongs to sessions.
- Per-request cost strictly decreases: jinja render + parser generation were
  always per request; analysis and detection stop being so.
- `common_chat_templates_support_enable_thinking()` currently performs a full
  probe apply per call; it becomes a cached `handler.supports_thinking()`.
- If `chat_template_kwargs` can change template structure enough to affect
  analysis, `chat_handler_auto` may need a small keyed cache of analyses
  rather than exactly one; default kwargs cover the common case.
- The legacy (non-jinja) path can stay on the old entry points until it is
  either wrapped in a trivial `chat_handler_legacy` or retired.
