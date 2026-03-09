#include "common-dfa-utf8.h"

// UTF-8 byte-level DFA states:
//
//   0  accept  — start state and after each complete codepoint
//   1  dead    — sink / error
//   2  tail1   — need 1 continuation byte  [80-BF] → 0
//   3  tail2   — need 2 continuation bytes [80-BF] → tail1
//   4  tail3   — need 3 continuation bytes [80-BF] → tail2
//   5  e0_s2   — after E0 lead: [A0-BF] → tail1  (reject overlong 3-byte)
//   6  ed_s2   — after ED lead: [80-9F] → tail1  (reject surrogates)
//   7  f0_s3   — after F0 lead: [90-BF] → tail2  (reject overlong 4-byte)
//   8  f4_s3   — after F4 lead: [80-8F] → tail2  (reject > U+10FFFF)

static constexpr uint32_t S_ACCEPT = 0;
static constexpr uint32_t S_DEAD   = 1;
static constexpr uint32_t S_TAIL1  = 2;
static constexpr uint32_t S_TAIL2  = 3;
static constexpr uint32_t S_TAIL3  = 4;
static constexpr uint32_t S_E0     = 5;
static constexpr uint32_t S_ED     = 6;
static constexpr uint32_t S_F0     = 7;
static constexpr uint32_t S_F4     = 8;

static constexpr uint32_t N_STATES = 9;

static void add(std::vector<common_dfa_transition> & t,
                uint32_t src, uint32_t dst, uint8_t lo, uint8_t hi) {
    t.push_back({ src, dst, lo, hi });
}

common_dfa_data common_dfa_utf8() {
    common_dfa_data d;

    auto & t = d.transitions;

    // --- from accept (state 0) ---

    // 1-byte ASCII
    add(t, S_ACCEPT, S_ACCEPT, 0x00, 0x7F);

    // 2-byte lead
    add(t, S_ACCEPT, S_TAIL1,  0xC2, 0xDF);

    // 3-byte leads
    add(t, S_ACCEPT, S_E0,     0xE0, 0xE0);
    add(t, S_ACCEPT, S_TAIL2,  0xE1, 0xEC);
    add(t, S_ACCEPT, S_ED,     0xED, 0xED);
    add(t, S_ACCEPT, S_TAIL2,  0xEE, 0xEF);

    // 4-byte leads
    add(t, S_ACCEPT, S_F0,     0xF0, 0xF0);
    add(t, S_ACCEPT, S_TAIL3,  0xF1, 0xF3);
    add(t, S_ACCEPT, S_F4,     0xF4, 0xF4);

    // --- continuation chains ---

    // tail1: last continuation byte
    add(t, S_TAIL1, S_ACCEPT, 0x80, 0xBF);

    // tail2: 2 continuations left (generic)
    add(t, S_TAIL2, S_TAIL1,  0x80, 0xBF);

    // tail3: 3 continuations left (generic)
    add(t, S_TAIL3, S_TAIL2,  0x80, 0xBF);

    // --- restricted first-continuation states ---

    // E0: reject overlong (first cont must be A0-BF)
    add(t, S_E0, S_TAIL1, 0xA0, 0xBF);

    // ED: reject surrogates (first cont must be 80-9F)
    add(t, S_ED, S_TAIL1, 0x80, 0x9F);

    // F0: reject overlong (first cont must be 90-BF)
    add(t, S_F0, S_TAIL2, 0x90, 0xBF);

    // F4: reject > U+10FFFF (first cont must be 80-8F)
    add(t, S_F4, S_TAIL2, 0x80, 0x8F);

    // --- metadata ---

    d.n_states    = N_STATES;
    d.start_state = S_ACCEPT;
    d.dead_state  = S_DEAD;
    d.accept_states.push_back(S_ACCEPT);

    return d;
}
