#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

// DFA transition: matches codepoints in [cp_lo, cp_hi] inclusive
struct common_dfa_transition {
    uint32_t src;    // source state
    uint32_t dst;    // destination state
    uint32_t cp_lo;  // codepoint range low  (inclusive)
    uint32_t cp_hi;  // codepoint range high (inclusive)
};

// Parameters to construct a DFA-based token masking sampler
struct common_dfa_params {
    const common_dfa_transition * transitions;
    size_t                        n_transitions;

    const uint32_t * accept_states;
    size_t           n_accept_states;

    uint32_t n_states;     // total number of states (states are 0..n_states-1)
    uint32_t start_state;
    uint32_t dead_state;   // explicit dead/sink state (user-provided)
};

// Create a DFA-based token masking sampler.
//
// On apply(), tokens whose codepoint sequences are not accepted by the DFA
// (i.e., lead to the dead state) have their logits set to -INFINITY.
//
// Uses prefix-matching: a token is valid if the DFA can consume all of its
// codepoints without entering the dead state, even if no accept state is
// reached yet.
//
// An adaptive per-state cache is built eagerly at init time. For each DFA
// state, if >50% of vocab tokens are accepted, the cache stores the rejected
// tokens (outliers); otherwise it stores the accepted tokens. This minimizes
// memory usage.
//
// The model is needed to decode tokens to their UTF-8 codepoint sequences.
struct llama_sampler * common_sampler_init_dfa(
        const struct llama_model * model,
        const common_dfa_params  & params);
