#include "common-sampler-dfa.h"
#include "common.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ----------------------------------------------------------------------------
// UTF-8 decoding (mirrors llama-grammar.cpp logic)
// ----------------------------------------------------------------------------

struct dfa_partial_utf8 {
    uint32_t value;
    int      n_remain;
};

static std::pair<std::vector<uint32_t>, dfa_partial_utf8> dfa_decode_utf8(
        const std::string & src,
        dfa_partial_utf8    partial_start) {
    static const int lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4 };

    const char *          pos = src.c_str();
    std::vector<uint32_t> code_points;
    code_points.reserve(src.size() + 1);

    uint32_t value    = partial_start.value;
    int      n_remain = partial_start.n_remain;

    // continue previous incomplete sequence
    while (*pos != 0 && n_remain > 0) {
        uint8_t next_byte = static_cast<uint8_t>(*pos);
        if ((next_byte >> 6) != 2) {
            // invalid continuation – abort
            code_points.push_back(0);
            return { std::move(code_points), { 0, -1 } };
        }
        value = (value << 6) + (next_byte & 0x3F);
        ++pos;
        --n_remain;
    }
    if (partial_start.n_remain > 0 && n_remain == 0) {
        code_points.push_back(value);
    }

    // decode subsequent full/partial sequences
    while (*pos != 0) {
        uint8_t first_byte = static_cast<uint8_t>(*pos);
        uint8_t highbits   = first_byte >> 4;
        n_remain = lookup[highbits] - 1;

        if (n_remain < 0) {
            code_points.clear();
            code_points.push_back(0);
            return { std::move(code_points), { 0, n_remain } };
        }

        uint8_t mask = (1 << (7 - n_remain)) - 1;
        value = first_byte & mask;
        ++pos;

        while (*pos != 0 && n_remain > 0) {
            value = (value << 6) + (static_cast<uint8_t>(*pos) & 0x3F);
            ++pos;
            --n_remain;
        }
        if (n_remain == 0) {
            code_points.push_back(value);
        }
    }

    return { std::move(code_points), { value, n_remain } };
}

// ----------------------------------------------------------------------------
// Internal DFA representation
// ----------------------------------------------------------------------------

// Per-state: sorted list of (cp_lo, cp_hi, dst) for binary search
struct dfa_edge {
    uint32_t cp_lo;
    uint32_t cp_hi;
    uint32_t dst;
};

struct dfa_table {
    uint32_t n_states;
    uint32_t start_state;
    uint32_t dead_state;
    std::unordered_set<uint32_t>      accept_states;
    std::vector<std::vector<dfa_edge>> edges; // edges[state] sorted by cp_lo
};

// Step the DFA one codepoint. Returns next state (may be dead_state).
static uint32_t dfa_step(const dfa_table & dfa, uint32_t state, uint32_t cp) {
    if (state == dfa.dead_state) {
        return dfa.dead_state;
    }
    const auto & es = dfa.edges[state];
    // binary search for the edge whose range contains cp
    int lo = 0, hi = (int)es.size() - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (cp < es[mid].cp_lo) {
            hi = mid - 1;
        } else if (cp > es[mid].cp_hi) {
            lo = mid + 1;
        } else {
            return es[mid].dst;
        }
    }
    // no matching edge → dead state
    return dfa.dead_state;
}

// Simulate a full codepoint sequence through the DFA.
// Returns the resulting state after consuming all codepoints, or dead_state if
// a dead state is reached at any point.
static uint32_t dfa_simulate(const dfa_table & dfa, uint32_t state,
                             const uint32_t * cps, size_t n_cps) {
    for (size_t i = 0; i < n_cps; ++i) {
        state = dfa_step(dfa, state, cps[i]);
        if (state == dfa.dead_state) {
            return dfa.dead_state;
        }
    }
    return state;
}

// Build the dfa_table from user params
static dfa_table dfa_build_table(const common_dfa_params & params) {
    dfa_table dfa;
    dfa.n_states    = params.n_states;
    dfa.start_state = params.start_state;
    dfa.dead_state  = params.dead_state;

    for (size_t i = 0; i < params.n_accept_states; ++i) {
        dfa.accept_states.insert(params.accept_states[i]);
    }

    dfa.edges.resize(params.n_states);
    for (size_t i = 0; i < params.n_transitions; ++i) {
        const auto & t = params.transitions[i];
        assert(t.src < params.n_states);
        assert(t.dst < params.n_states);
        assert(t.cp_lo <= t.cp_hi);
        dfa.edges[t.src].push_back({ t.cp_lo, t.cp_hi, t.dst });
    }

    // sort edges by cp_lo for binary search
    for (uint32_t s = 0; s < params.n_states; ++s) {
        std::sort(dfa.edges[s].begin(), dfa.edges[s].end(),
                  [](const dfa_edge & a, const dfa_edge & b) {
                      return a.cp_lo < b.cp_lo;
                  });
    }

    return dfa;
}

// ----------------------------------------------------------------------------
// Per-token decoded codepoints (precomputed at init)
// ----------------------------------------------------------------------------

struct dfa_token_info {
    std::vector<uint32_t> codepoints; // decoded codepoints (empty for special tokens)
};

// ----------------------------------------------------------------------------
// Adaptive per-state cache
// ----------------------------------------------------------------------------

struct dfa_state_cache {
    bool is_accept_heavy; // true → outliers are REJECTED tokens
                          // false → outliers are ACCEPTED tokens (with next_state)

    // The minority set of token ids
    std::unordered_set<llama_token> outlier_tokens;

    // For reject-heavy states: maps accepted token → resulting DFA state
    // For accept-heavy states: empty (re-simulate on accept)
    std::unordered_map<llama_token, uint32_t> outlier_next_states;
};

// ----------------------------------------------------------------------------
// Sampler context
// ----------------------------------------------------------------------------

struct common_sampler_dfa {
    dfa_table                       dfa;
    uint32_t                        current_state;
    dfa_partial_utf8                partial_utf8; // partial UTF-8 across token boundaries

    std::vector<dfa_token_info>     token_info;   // per-token decoded codepoints
    std::vector<dfa_state_cache>    cache;         // per-state adaptive cache

    int32_t                         n_vocab;
};

// Build the adaptive cache for all states
static void dfa_build_cache(common_sampler_dfa * ctx) {
    const auto & dfa   = ctx->dfa;
    const int32_t n_vocab = ctx->n_vocab;

    ctx->cache.resize(dfa.n_states);

    for (uint32_t state = 0; state < dfa.n_states; ++state) {
        if (state == dfa.dead_state) {
            // dead state rejects everything – treat as reject-heavy with empty outliers
            ctx->cache[state].is_accept_heavy = false;
            continue;
        }

        // simulate every token from this state
        std::vector<std::pair<llama_token, uint32_t>> accepted; // (token, next_state)
        std::vector<llama_token>                       rejected;

        for (int32_t tok = 0; tok < n_vocab; ++tok) {
            const auto & info = ctx->token_info[tok];
            if (info.codepoints.empty()) {
                // special / control token with no text → reject
                rejected.push_back(tok);
                continue;
            }

            uint32_t next = dfa_simulate(dfa, state,
                                         info.codepoints.data(),
                                         info.codepoints.size());
            if (next == dfa.dead_state) {
                rejected.push_back(tok);
            } else {
                accepted.push_back({ tok, next });
            }
        }

        auto & sc = ctx->cache[state];
        const bool accept_heavy = accepted.size() > rejected.size();
        sc.is_accept_heavy = accept_heavy;

        if (accept_heavy) {
            // store rejected tokens as outliers
            for (const auto & tok : rejected) {
                sc.outlier_tokens.insert(tok);
            }
            // outlier_next_states stays empty – re-simulate on accept()
        } else {
            // store accepted tokens as outliers (with their next state)
            for (const auto & [tok, ns] : accepted) {
                sc.outlier_tokens.insert(tok);
                sc.outlier_next_states[tok] = ns;
            }
        }
    }
}

// Decode all vocab tokens to codepoints
static void dfa_build_token_info(common_sampler_dfa * ctx,
                                 const struct llama_vocab * vocab) {
    const int32_t n_vocab = ctx->n_vocab;
    ctx->token_info.resize(n_vocab);

    for (int32_t tok = 0; tok < n_vocab; ++tok) {
        std::string piece = common_token_to_piece(vocab, tok, false);
        if (piece.empty()) {
            continue;
        }

        auto [cps, partial] = dfa_decode_utf8(piece, { 0, 0 });

        // remove trailing 0 sentinel if present
        if (!cps.empty() && cps.back() == 0) {
            cps.pop_back();
        }

        // if there's a partial UTF-8 sequence at the end, we still use
        // whatever complete codepoints were decoded (partial tokens are rare)
        ctx->token_info[tok].codepoints = std::move(cps);
    }
}

// ----------------------------------------------------------------------------
// Sampler interface implementation
// ----------------------------------------------------------------------------

static const char * dfa_sampler_name(const struct llama_sampler * /*smpl*/) {
    return "dfa";
}

static void dfa_sampler_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = static_cast<common_sampler_dfa *>(smpl->ctx);

    if (ctx->current_state == ctx->dfa.dead_state) {
        return; // stuck in dead state
    }

    const auto & sc = ctx->cache[ctx->current_state];

    if (!sc.is_accept_heavy) {
        // reject-heavy: accepted tokens (outliers) have cached next_state
        auto it = sc.outlier_next_states.find(token);
        if (it != sc.outlier_next_states.end()) {
            ctx->current_state = it->second;
            return;
        }
    }

    // accept-heavy or cache miss: re-simulate
    const auto & info = ctx->token_info[token];
    if (info.codepoints.empty()) {
        // special token – don't advance DFA
        return;
    }

    ctx->current_state = dfa_simulate(ctx->dfa, ctx->current_state,
                                      info.codepoints.data(),
                                      info.codepoints.size());
}

static void dfa_sampler_apply(struct llama_sampler * smpl,
                              llama_token_data_array * cur_p) {
    auto * ctx = static_cast<common_sampler_dfa *>(smpl->ctx);
    const uint32_t state = ctx->current_state;

    if (state == ctx->dfa.dead_state) {
        // everything is rejected
        for (size_t i = 0; i < cur_p->size; ++i) {
            cur_p->data[i].logit = -INFINITY;
        }
        cur_p->sorted = false;
        return;
    }

    const auto & sc = ctx->cache[state];

    if (sc.is_accept_heavy) {
        // most tokens accepted → mask only the outliers (rejected tokens)
        for (size_t i = 0; i < cur_p->size; ++i) {
            if (sc.outlier_tokens.count(cur_p->data[i].id)) {
                cur_p->data[i].logit = -INFINITY;
            }
        }
    } else {
        // most tokens rejected → mask everything NOT in the outlier set (accepted tokens)
        for (size_t i = 0; i < cur_p->size; ++i) {
            if (!sc.outlier_tokens.count(cur_p->data[i].id)) {
                cur_p->data[i].logit = -INFINITY;
            }
        }
    }

    cur_p->sorted = false;
}

static void dfa_sampler_reset(struct llama_sampler * smpl) {
    auto * ctx = static_cast<common_sampler_dfa *>(smpl->ctx);
    ctx->current_state = ctx->dfa.start_state;
    ctx->partial_utf8  = { 0, 0 };
}

static struct llama_sampler * dfa_sampler_clone(const struct llama_sampler * smpl) {
    const auto * ctx = static_cast<const common_sampler_dfa *>(smpl->ctx);
    auto * clone = new common_sampler_dfa(*ctx);

    static const struct llama_sampler_i dfa_sampler_i = {
        /* .name   = */ dfa_sampler_name,
        /* .accept = */ dfa_sampler_accept,
        /* .apply  = */ dfa_sampler_apply,
        /* .reset  = */ dfa_sampler_reset,
        /* .clone  = */ dfa_sampler_clone,
        /* .free   = */ [](struct llama_sampler * smpl) {
            delete static_cast<common_sampler_dfa *>(smpl->ctx);
        },
    };

    return llama_sampler_init(&dfa_sampler_i, clone);
}

static void dfa_sampler_free(struct llama_sampler * smpl) {
    delete static_cast<common_sampler_dfa *>(smpl->ctx);
}

static const struct llama_sampler_i dfa_sampler_vtable = {
    /* .name   = */ dfa_sampler_name,
    /* .accept = */ dfa_sampler_accept,
    /* .apply  = */ dfa_sampler_apply,
    /* .reset  = */ dfa_sampler_reset,
    /* .clone  = */ dfa_sampler_clone,
    /* .free   = */ dfa_sampler_free,
};

// ----------------------------------------------------------------------------
// Public init
// ----------------------------------------------------------------------------

struct llama_sampler * common_sampler_init_dfa(
        const struct llama_model * model,
        const common_dfa_params  & params) {

    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    auto * ctx = new common_sampler_dfa();
    ctx->n_vocab      = n_vocab;
    ctx->current_state = params.start_state;
    ctx->partial_utf8  = { 0, 0 };

    // build internal DFA representation
    ctx->dfa = dfa_build_table(params);

    // decode all tokens to codepoints
    dfa_build_token_info(ctx, vocab);

    // build adaptive per-state cache
    dfa_build_cache(ctx);

    return llama_sampler_init(&dfa_sampler_vtable, ctx);
}
