#pragma once

#include "llama.h"

#include <cstdint>
#include <cstddef>

// Byte-level DFA transition: matches bytes in [byte_lo, byte_hi] inclusive.
//
// Because the DFA operates on raw bytes, UTF-8 multi-byte sequences are
// encoded directly in the automaton structure (start-byte ranges route to
// continuation-byte states, etc.). This means:
//   - Tokens that end mid-UTF-8 simply land in a continuation-byte state.
//   - The next token resumes from that state — no special partial-UTF-8
//     tracking is needed in the sampler.
struct common_dfa_transition {
    uint32_t src;      // source state
    uint32_t dst;      // destination state
    uint8_t  byte_lo;  // byte range low  (inclusive)
    uint8_t  byte_hi;  // byte range high (inclusive)
};

// Parameters to construct a byte-level DFA token masking sampler.
struct common_dfa_params {
    const common_dfa_transition * transitions;
    size_t                        n_transitions;

    const uint32_t * accept_states;
    size_t           n_accept_states;

    uint32_t n_states;     // total number of states (0..n_states-1)
    uint32_t start_state;
    uint32_t dead_state;   // explicit dead/sink state

    // Optional token-limited region. When non-null, defines a set of DFA
    // states with a maximum number of accept() calls (generated tokens).
    //
    // When the limit is reached, apply() only permits tokens whose byte-level
    // simulation ends in a state OUTSIDE the region (and not the dead state).
    // This naturally enforces UTF-8 completeness at the exit boundary — if
    // the DFA is in a mid-UTF-8 continuation state, that state is still
    // inside the region, so the token is rejected.
    //
    // The counter is cumulative and resets on sampler reset.
    const uint32_t * token_limit_states;   // states in the limited region (nullptr = no limit)
    size_t           n_token_limit_states;
    int32_t          token_limit;           // max accept() calls while in the region
};

// Create a byte-level DFA token masking sampler.
//
// On apply(), tokens whose byte sequences lead to the dead state have their
// logits set to -INFINITY. When the token-limited region is exhausted, only
// tokens that exit the region are allowed.
//
// Uses a lazy adaptive per-state cache (built on first access) so that
// large DFAs (many states) only pay for states actually reached.
//
// The model is needed to obtain token byte representations.
struct llama_sampler * common_sampler_init_dfa(
        const struct llama_model * model,
        const common_dfa_params  & params);
