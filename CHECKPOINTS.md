# Message-boundary prompt checkpoints — session handoff

This document summarizes work done on branch
`claude/add-message-splits-field-jipBx` to move llama.cpp server prompt
checkpoints from arbitrary token intervals onto message boundaries in
rendered chat prompts. It is intended to brief a follow-up agent that
didn't see the design conversation.

## TL;DR

The server used to capture context checkpoints (partial-KV snapshots
for SWA / recurrent / hybrid architectures) every `--checkpoint-every-nt`
tokens, plus a late-prompt safety capture via a "checkpoint_offsets"
breakout near the end of the prompt. These capture points are
semantically blind — they can land mid-message, mid-tool-call, anywhere
inside a rendered chat prompt. When a follow-up chat request shared the
first N messages with a cached one, reuse depended on whether a
checkpoint happened to align with the end of message N, which it
usually didn't.

This branch re-times captures in chat mode so they land at message
boundaries — the byte offsets already produced by `split_prompt_by_role`
for gpt-oss in `common_chat_params::message_splits`. When a follow-up
arrives, the prefix match against the slot's existing tokens gives a
clean resume point at the end of a message, not a mid-message chunk.

**Scope — Phase 1 only.** We wired captures to boundaries. The
cross-request LRU still works on whole-prompt entries via
`server_prompt_cache::load()`; teaching it to match individual
checkpoints by prefix hash is a deferred Phase 2 task.

## Commits on this branch

1. `f91a5a2` — **server : capture prompt checkpoints at message
   boundaries**. The main PR. Adds the data-flow plumbing, decode-loop
   edits, helper, and unit test.
2. `f3052ec` — **server : drop near-end checkpoint arm in
   message-boundary mode**. Follow-up cleanup: reorders the
   capture-decision cascade so the legacy "near end of prompt"
   unconditional capture doesn't fire in boundary mode. Arm 2 (boundary)
   is now checked before arm 1 (near-end), and arm 1 only runs when
   `checkpoint_token_positions` is empty.

## Data flow

```
common_chat_params_init_gpt_oss  (common/chat.cpp:1017)
    │
    ▼  populates chat_params.message_splits via split_prompt_by_role
    │  with gpt-oss role delimiters (<|start|>assistant, <|start|>user,
    │  <|start|>developer, <|start|>system, <|start|>functions).
    │
oaicompat_chat_params_parse  (tools/server/server-common.cpp:1085+)
    │
    ▼  serializes splits as llama_params["message_splits"] =
    │  [{role, pos, len}, ...] when non-empty.
    │
handle_completions_impl  (tools/server/server-context.cpp:3147+)
    │
    ▼  parse_message_splits_json  (server-context.cpp:40) parses JSON
    │  back into std::vector<common_chat_message_split>.
    ▼  message_splits_to_token_positions  (common/chat.cpp:123) maps
    │  byte offsets → cumulative token counts via per-slice
    │  tokenization. Validation first pass avoids touching the vocab
    │  on malformed input.
    │
server_task.checkpoint_token_positions  (server-task.h:148-155)
    │  (only attached to task i==0 in the batch loop)
    │
prompt-processing loop  (server-context.cpp:2176+)
    │
    ▼  push-loop break at boundaries  (server-context.cpp:2587-2604)
    ▼  capture-decision arm 2         (server-context.cpp:2629-2652)
    │
slot.prompt.checkpoints.emplace_back  (server-context.cpp:2704)
    │  uses LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY — partial KV only,
    │  not generated tokens.
```

## Critical files (with current line numbers)

- `common/chat.h` — declarations for `common_chat_message_split`,
  `split_prompt_by_role`, and the new `message_splits_to_token_positions`
  (header comment explains invariants and early-return conditions).
- `common/chat.cpp:123` — `message_splits_to_token_positions` impl.
  Two-pass: (1) validate contiguity + bounds + preamble, (2) tokenize
  each slice with `add_special=true` only for the first slice and
  `parse_special=true` for all. Returns cumulative positions.
- `common/chat.cpp:1017` — gpt-oss delim list consumed by
  `split_prompt_by_role`. The *only* template populating
  `message_splits` today.
- `tools/server/server-common.cpp:1085-1101` — `oaicompat_chat_params_parse`
  serializes `chat_params.message_splits` into `llama_params` as a JSON
  array when non-empty. Gated so non-chat / unsupported templates don't
  add noise. Prefill-assistant path is fine because it appends text
  *after* `common_chat_templates_apply()`; existing splits still point
  to valid earlier offsets.
- `tools/server/server-task.h:148-155` — new
  `std::vector<int> checkpoint_token_positions` field on `server_task`.
  Comment explains semantics: cumulative token counts, empty ⇒ legacy
  fallback.
- `tools/server/server-context.cpp:40` — `parse_message_splits_json`
  static helper. Parses the JSON array back into
  `std::vector<common_chat_message_split>`. Malformed input ⇒ empty
  vector ⇒ falls back to legacy behavior.
- `tools/server/server-context.cpp:2576-2604` — push loop break logic.
  Three conditions can terminate the sub-batch early:
  1. Existing `checkpoint_offsets = {4 + n_ubatch, 4}` breakout — still
     active in both modes. Ensures the tail is split so arm 2 can capture
     the last boundary.
  2. NEW: boundary match — when `slot.prompt.n_tokens()` equals any
     `p in checkpoint_token_positions`, break. Only active when
     positions are non-empty.
- `tools/server/server-context.cpp:2628-2672` — capture-decision
  cascade (post-f3052ec ordering):
  1. **Arm 2 (boundary mode)** at 2629: only when
     `!checkpoint_token_positions.empty()`. Captures iff
     `n_decoded == p` for some `p in positions` (exact equality,
     guaranteed by the push-loop break). Does NOT fall back to a
     near-end capture.
  2. **Arm 1 (near end of prompt)** at 2657: legacy path, only runs
     when positions are empty. `do_checkpoint && true` is a no-op; it
     just means "yes, capture unconditionally" subject to the shared
     guards below.
  3. **Arm 3 (checkpoint_every_nt)** at 2662: ultimate fallback.
     Requires `checkpoint_every_nt > 0` and ≥ that many tokens since
     last checkpoint.
- `tools/server/server-context.cpp:2674-2684` — shared guards applied
  to all three arms:
  - `pos_min >= 0 && slot.prompt.n_tokens() >= 64` (skip tiny prompts)
  - `!has_mtmd` (no captures after image chunks)
  - spacing: new checkpoint must be > 64 tokens past the previous one.
- `tools/server/server-context.cpp:2686-2718` — capture write path,
  unchanged. Uses `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`.
- `tools/server/server-context.cpp:3147-3175` — call site in
  `handle_completions_impl`. Only activates for the chat path:
  `inputs.size() == 1 && prompt.is_string() && data.contains("message_splits")`.
  Computes positions once, moves them into `task.checkpoint_token_positions`
  for task i==0.
- `tests/test-chat.cpp:3502-3601` — `test_message_splits_to_token_positions`.
  Three no-vocab tests (empty / preamble / non-contiguous) followed by
  a vocab-backed test that loads `models/ggml-vocab-qwen2.gguf` and
  verifies the special-token invariant: `concat(per-slice) == one-shot`.

## Capture-decision cascade walkthrough

The outer `if (slot.prompt.n_tokens() == slot.task->n_tokens())` at
`server-context.cpp:2615` catches "prompt is fully pushed, this round
finishes processing." It bypasses *all* arms — no checkpoint is
captured on the round that finishes the prompt. This means the final
round never captures directly; late-prompt captures happen in the
round *before* the final one, which the `checkpoint_offsets = {4, 4+n_ubatch}`
breakout creates by breaking the push loop 4 (or 4+n_ubatch) tokens
early.

### Arm 2 — boundary mode (new, primary in chat)

Fires when `checkpoint_token_positions` is non-empty. The push-loop
break at 2587-2604 guarantees that if a previous round decoded to
boundary P, the next round's `n_decoded = slot.prompt.n_tokens() - n_tokens_cur`
equals P. Arm 2 checks equality (not range) against the positions
vector and captures on match.

Arm 2 *cannot* capture the last boundary directly — a round that ends
at `prompt.n_tokens() == task.n_tokens()` falls into the outer
"processing done" branch. Instead, the `checkpoint_offsets` breakout
creates an intermediate round: if tail ≥ 5 tokens past the last
boundary, the final push-loop breaks at `task - 4`, and the capture
block runs with `n_decoded == last_boundary`, capturing it via arm 2.

### Arm 1 — near end of prompt (legacy-only)

Only runs when `checkpoint_token_positions.empty()` (post-f3052ec).
Fires when `task.n_tokens() < prompt.n_tokens() + n_ubatch` — i.e.,
"we're inside the last ubatch's worth of prompt." The point is an
unconditional SWA-recovery capture at the end of the prompt,
regardless of semantic alignment.

### Arm 3 — checkpoint_every_nt interval (legacy fallback)

Only runs when the first two don't. Needs
`params_base.checkpoint_every_nt > 0` and at least that many tokens
since the last checkpoint.

## Scope of an in-flight checkpoint

**Checkpoints do NOT include generation.** Two reasons:

1. The capture block at `server-context.cpp:2704` only executes inside
   the `SLOT_STATE_PROCESSING_PROMPT` / `SLOT_STATE_STARTED` branch at
   `server-context.cpp:2177`. Once the slot transitions to
   `SLOT_STATE_GENERATING` at 2887, this code is never re-entered for
   that slot.
2. The capture uses `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`, which
   excludes state that can be reconstructed by re-decoding (regular
   attention KV) and includes only SWA / recurrent / hybrid state.

A *separate* path, `slot::prompt_save()` at `server-context.cpp:131`,
does save full sequence state (flag = 0) when the slot is recycled;
this includes generated tokens, and feeds the cross-request
`server_prompt_cache`. This PR does not touch that path.

## Invariant the helper depends on

**Every role delimiter passed to `split_prompt_by_role` is a special
token in the vocab.**

With `parse_special=true`, the tokenizer emits a special token
atomically — no BPE merge can span it. Therefore the concatenation of
per-slice tokenizations (slicing the prompt at `splits[i].pos`) equals
the one-shot tokenization of the full prompt, provided:

- slice 0 is tokenized with `add_special=true` (prepends BOS if the
  vocab requires, matching one-shot behavior), and
- slices 1..N use `add_special=false` (no extra BOS mid-prompt).

This mirrors the existing `tokenize_mixed` pattern at
`tools/server/server-common.cpp:632-635`.

The unit test pins this for gpt-oss-style delimiters against the qwen2
vocab: `concat(per-slice tokens) == full tokens`. If a future template
wires `split_prompt_by_role` with a non-special delim the invariant
would break, the helper's positions would drift from the one-shot
tokenization used everywhere else, and `slot.prompt.tokens` would no
longer match the captured checkpoint positions — silent corruption.
Mitigation today: only gpt-oss populates `message_splits`, and its
delims are all `<|start|>*` special tokens.

## Known edge case

If the chat template's trailing generation prompt (the bytes after
the last message's `<|end|>`) tokenizes to fewer than 5 tokens, the
last message boundary will not be captured:

- The boundary-break in the push loop needs the boundary to be
  strictly less than `task.n_tokens()` — if positions[last] ==
  task.n_tokens(), the `checkpoint_offsets` breakout at `task - 4`
  fires first, and the round after that lands in the "processing done"
  bypass.
- Even with a trailing tail, the `checkpoint_offsets` 4-token breakout
  needs `remaining == 4` immediately after a push — which in turn
  needs `remaining == 5` before the push. Tails of 1-4 tokens skip the
  breakout entirely and the last boundary is lost.

gpt-oss's `<|start|>assistant<|channel|>analysis<|message|>` tail is
longer than 4 tokens so we're fine there, but this is worth
remembering when wiring up other templates.

## Deferred follow-ups (explicitly NOT in this PR)

1. **Cross-request checkpoint matching.** Today
   `server_prompt_cache::load()` at `tools/server/server-task.cpp:2008`
   matches whole cached prompts by longest common prefix. Teach it to
   walk each cached `server_prompt`'s checkpoint list and restore the
   best internal boundary when an incoming prompt shares only a
   mid-conversation prefix. Needs a cross-prompt checkpoint identity,
   e.g. an FNV hash of the prefix tokens through `n_tokens` (see
   `fnv_hash` at `tools/server/server-common.cpp:683` for precedent).
2. **Populate `message_splits` for other templates.** Llama-3, Qwen
   chat, Mistral, ChatML — each needs a `split_prompt_by_role` call in
   its `common_chat_params_init_*` function with the appropriate
   delims. Verify the special-token invariant for each vocab.
3. **Server integration test.** Test plan was drafted but not
   implemented: start a server with gpt-oss template, `--ctx-checkpoints 8`,
   `--checkpoint-every-nt 0`, and verify via `timings.cache_n` that a
   follow-up request reuses up to the end of a message boundary, not a
   chunk offset. Suggested location:
   `tools/server/tests/unit/test_chat_completion.py`.
4. **Consider removing `checkpoint_offsets` breakout in boundary mode.**
   Philosophically it's a legacy-mode concept (force a small final
   sub-batch for the unconditional near-end capture). In boundary mode
   it still serves as the mechanism that lets arm 2 capture the last
   boundary, but that's incidental. Could replace with an explicit
   "break N tokens before task end when the last boundary is in the
   tail" clause in the push loop.

## Build & test

Cmake needs tools enabled for the server target; this branch had to
reconfigure once:

```bash
cmake -B build -DLLAMA_BUILD_TOOLS=ON
cmake --build build --target llama-server test-chat -j$(nproc)
```

Run the unit test (from the project source root — the CMake config
sets `WORKING_DIRECTORY=${PROJECT_SOURCE_DIR}` so the vocab path
`models/ggml-vocab-qwen2.gguf` resolves):

```bash
./build/bin/test-chat
```

Expected tail:
```
[chat] All tests passed!
```

The new test is `test_message_splits_to_token_positions` at
`tests/test-chat.cpp:3502`, invoked from `main` at
`tests/test-chat.cpp:3744`.

## Open design questions to raise with the maintainer

- Should `message_splits` be moved out of `llama_params` JSON and onto
  a non-JSON sidecar, since it's an internal concept the API doesn't
  expose? (Currently it rides JSON because that matches how
  `oaicompat_chat_params_parse` hands everything else to the task.)
- Should Phase 2 cross-request matching use the FNV hash approach, or
  something denser like storing a rolling hash per checkpoint?
- For templates that emit tool-call JSON *inside* a message, should
  `split_prompt_by_role` also emit per-tool-call sub-boundaries? That
  would make partial-tool-call recovery possible but complicates the
  "each split has exactly one role" contract.
