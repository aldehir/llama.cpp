#pragma once

#include "llama.h"

// Reasoning budget sampler
//
// A stateful sampler that enforces a token budget on reasoning (thinking) output.
//
// Behavior:
//   1. COUNTING phase: Accept all tokens, counting each one, until either:
//      - The end token is seen (transition to PASSTHROUGH), or
//      - The budget is exhausted (transition to FORCING)
//   2. FORCING phase: Only allow the end token by setting all other logits to -inf.
//      Once the end token is accepted, transition to PASSTHROUGH.
//   3. PASSTHROUGH phase: Do nothing — allow all tokens through unmodified.
//
// This is intended to be inserted early in a sampler chain (before temperature,
// top-k, etc.) so that it can mask logits before other samplers run.

struct llama_sampler * common_sampler_init_reasoning_budget(int32_t budget, llama_token end_token);
