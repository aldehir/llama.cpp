#pragma once

#include "common-sampler-dfa.h"

#include <cstdint>
#include <vector>

// Self-contained DFA data that owns its storage and can produce a
// common_dfa_params view for common_sampler_init_dfa().
//
// The returned params reference internal vectors, so the common_dfa_data
// instance must outlive the params (or the sampler must be constructed
// before the data goes out of scope — the sampler copies what it needs).
struct common_dfa_data {
    std::vector<common_dfa_transition> transitions;
    std::vector<uint32_t>              accept_states;
    uint32_t                           n_states    = 0;
    uint32_t                           start_state = 0;
    uint32_t                           dead_state  = 0;

    // optional token limit
    std::vector<uint32_t>              token_limit_states;
    int32_t                            token_limit = 0;

    common_dfa_params params() const {
        common_dfa_params p{};
        p.transitions        = transitions.data();
        p.n_transitions      = transitions.size();
        p.accept_states      = accept_states.data();
        p.n_accept_states    = accept_states.size();
        p.n_states           = n_states;
        p.start_state        = start_state;
        p.dead_state         = dead_state;
        p.token_limit_states = token_limit_states.empty() ? nullptr : token_limit_states.data();
        p.n_token_limit_states = token_limit_states.size();
        p.token_limit        = token_limit;
        return p;
    }
};

// Build a byte-level DFA that accepts one or more valid UTF-8 codepoints.
//
// The automaton encodes the full UTF-8 specification:
//   - 1-byte:  00-7F
//   - 2-byte:  C2-DF, 80-BF
//   - 3-byte:  E0 A0-BF 80-BF | E1-EC 80-BF 80-BF |
//              ED 80-9F 80-BF | EE-EF 80-BF 80-BF
//   - 4-byte:  F0 90-BF 80-BF 80-BF | F1-F3 80-BF 80-BF 80-BF |
//              F4 80-8F 80-BF 80-BF
//
// Overlong encodings (C0-C1), surrogate halves (ED A0-BF), and out-of-range
// lead bytes (F5-FF) all go to the dead state.
//
// State 0 is both the start and accept state (each complete codepoint
// returns to state 0). State 1 is the dead state.
common_dfa_data common_dfa_utf8();
