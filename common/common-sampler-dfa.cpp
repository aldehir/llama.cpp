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
// Internal DFA representation (byte-level)
// ----------------------------------------------------------------------------

struct dfa_edge {
    uint8_t  byte_lo;
    uint8_t  byte_hi;
    uint32_t dst;
};

struct dfa_table {
    uint32_t n_states;
    uint32_t start_state;
    uint32_t dead_state;
    std::unordered_set<uint32_t>       accept_states;
    std::vector<std::vector<dfa_edge>> edges; // edges[state] sorted by byte_lo
};

static uint32_t dfa_step_byte(const dfa_table & dfa, uint32_t state, uint8_t byte) {
    if (state == dfa.dead_state) {
        return dfa.dead_state;
    }
    const auto & es = dfa.edges[state];
    int lo = 0, hi = (int)es.size() - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (byte < es[mid].byte_lo) {
            hi = mid - 1;
        } else if (byte > es[mid].byte_hi) {
            lo = mid + 1;
        } else {
            return es[mid].dst;
        }
    }
    return dfa.dead_state;
}

static uint32_t dfa_simulate_bytes(const dfa_table & dfa, uint32_t state,
                                   const uint8_t * bytes, size_t n_bytes) {
    for (size_t i = 0; i < n_bytes; ++i) {
        state = dfa_step_byte(dfa, state, bytes[i]);
        if (state == dfa.dead_state) {
            return dfa.dead_state;
        }
    }
    return state;
}

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
        assert(t.byte_lo <= t.byte_hi);
        dfa.edges[t.src].push_back({ t.byte_lo, t.byte_hi, t.dst });
    }

    for (uint32_t s = 0; s < params.n_states; ++s) {
        std::sort(dfa.edges[s].begin(), dfa.edges[s].end(),
                  [](const dfa_edge & a, const dfa_edge & b) {
                      return a.byte_lo < b.byte_lo;
                  });
    }

    return dfa;
}

// ----------------------------------------------------------------------------
// Per-token raw bytes (precomputed at init)
// ----------------------------------------------------------------------------

struct dfa_token_info {
    std::string piece; // raw byte representation (empty for special tokens)
};

// ----------------------------------------------------------------------------
// Token-limited region (single, optional)
// ----------------------------------------------------------------------------

struct dfa_token_limit {
    std::unordered_set<uint32_t> member_states;
    int32_t                      max_tokens;
    int32_t                      counter;
};

// ----------------------------------------------------------------------------
// Adaptive per-state caches (lazy)
// ----------------------------------------------------------------------------

struct dfa_state_cache {
    bool is_accept_heavy;
    std::unordered_set<llama_token>           outlier_tokens;
    std::unordered_map<llama_token, uint32_t> outlier_next_states; // reject-heavy only
};

struct dfa_exit_cache_entry {
    bool is_exit_heavy;
    std::unordered_set<llama_token> outlier_tokens;
};

// ----------------------------------------------------------------------------
// Sampler context
// ----------------------------------------------------------------------------

struct common_sampler_dfa {
    dfa_table                    dfa;
    uint32_t                     current_state;

    std::vector<dfa_token_info>  token_info;
    int32_t                      n_vocab;

    // optional token limit (has_limit == false means no limit)
    bool                         has_limit;
    dfa_token_limit              limit;

    // lazy caches
    std::unordered_map<uint32_t, dfa_state_cache>      cache;
    std::unordered_map<uint32_t, dfa_exit_cache_entry>  exit_cache; // keyed by state
};

// ----------------------------------------------------------------------------
// Lazy cache builders
// ----------------------------------------------------------------------------

static const dfa_state_cache & dfa_get_or_build_cache(
        common_sampler_dfa * ctx, uint32_t state) {
    auto it = ctx->cache.find(state);
    if (it != ctx->cache.end()) {
        return it->second;
    }

    const auto &  dfa     = ctx->dfa;
    const int32_t n_vocab = ctx->n_vocab;

    dfa_state_cache sc;

    if (state == dfa.dead_state) {
        sc.is_accept_heavy = false;
        auto [ins, _] = ctx->cache.emplace(state, std::move(sc));
        return ins->second;
    }

    std::vector<std::pair<llama_token, uint32_t>> accepted;
    std::vector<llama_token>                      rejected;

    for (int32_t tok = 0; tok < n_vocab; ++tok) {
        const auto & info = ctx->token_info[tok];
        if (info.piece.empty()) {
            rejected.push_back(tok);
            continue;
        }

        uint32_t next = dfa_simulate_bytes(dfa, state,
                                           reinterpret_cast<const uint8_t *>(info.piece.data()),
                                           info.piece.size());
        if (next == dfa.dead_state) {
            rejected.push_back(tok);
        } else {
            accepted.push_back({ tok, next });
        }
    }

    sc.is_accept_heavy = accepted.size() > rejected.size();

    if (sc.is_accept_heavy) {
        for (const auto & tok : rejected) {
            sc.outlier_tokens.insert(tok);
        }
    } else {
        for (const auto & [tok, ns] : accepted) {
            sc.outlier_tokens.insert(tok);
            sc.outlier_next_states[tok] = ns;
        }
    }

    auto [ins, _] = ctx->cache.emplace(state, std::move(sc));
    return ins->second;
}

static const dfa_exit_cache_entry & dfa_get_or_build_exit_cache(
        common_sampler_dfa * ctx, uint32_t state) {
    auto it = ctx->exit_cache.find(state);
    if (it != ctx->exit_cache.end()) {
        return it->second;
    }

    const auto &  dfa     = ctx->dfa;
    const int32_t n_vocab = ctx->n_vocab;
    const auto &  limit   = ctx->limit;

    std::vector<llama_token> exits;
    std::vector<llama_token> stays;

    for (int32_t tok = 0; tok < n_vocab; ++tok) {
        const auto & info = ctx->token_info[tok];
        if (info.piece.empty()) {
            stays.push_back(tok);
            continue;
        }

        uint32_t next = dfa_simulate_bytes(dfa, state,
                                           reinterpret_cast<const uint8_t *>(info.piece.data()),
                                           info.piece.size());
        if (next == dfa.dead_state || limit.member_states.count(next)) {
            stays.push_back(tok);
        } else {
            exits.push_back(tok);
        }
    }

    dfa_exit_cache_entry ec;
    ec.is_exit_heavy = exits.size() > stays.size();

    if (ec.is_exit_heavy) {
        for (const auto & tok : stays) {
            ec.outlier_tokens.insert(tok);
        }
    } else {
        for (const auto & tok : exits) {
            ec.outlier_tokens.insert(tok);
        }
    }

    auto [ins, _] = ctx->exit_cache.emplace(state, std::move(ec));
    return ins->second;
}

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

static void dfa_apply_state_cache(const dfa_state_cache & sc,
                                  llama_token_data_array * cur_p) {
    if (sc.is_accept_heavy) {
        for (size_t i = 0; i < cur_p->size; ++i) {
            if (sc.outlier_tokens.count(cur_p->data[i].id)) {
                cur_p->data[i].logit = -INFINITY;
            }
        }
    } else {
        for (size_t i = 0; i < cur_p->size; ++i) {
            if (!sc.outlier_tokens.count(cur_p->data[i].id)) {
                cur_p->data[i].logit = -INFINITY;
            }
        }
    }
}

static void dfa_apply_exit_cache(const dfa_exit_cache_entry & ec,
                                 llama_token_data_array * cur_p) {
    if (ec.is_exit_heavy) {
        for (size_t i = 0; i < cur_p->size; ++i) {
            if (ec.outlier_tokens.count(cur_p->data[i].id)) {
                cur_p->data[i].logit = -INFINITY;
            }
        }
    } else {
        for (size_t i = 0; i < cur_p->size; ++i) {
            if (!ec.outlier_tokens.count(cur_p->data[i].id)) {
                cur_p->data[i].logit = -INFINITY;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Token info
// ----------------------------------------------------------------------------

static void dfa_build_token_info(common_sampler_dfa * ctx,
                                 const struct llama_vocab * vocab) {
    const int32_t n_vocab = ctx->n_vocab;
    ctx->token_info.resize(n_vocab);

    for (int32_t tok = 0; tok < n_vocab; ++tok) {
        ctx->token_info[tok].piece = common_token_to_piece(vocab, tok, false);
    }
}

// ----------------------------------------------------------------------------
// Sampler interface
// ----------------------------------------------------------------------------

static const char * dfa_sampler_name(const struct llama_sampler * /*smpl*/) {
    return "dfa";
}

static void dfa_sampler_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = static_cast<common_sampler_dfa *>(smpl->ctx);

    if (ctx->current_state == ctx->dfa.dead_state) {
        return;
    }

    // increment token limit counter if current state is in the region
    if (ctx->has_limit && ctx->limit.member_states.count(ctx->current_state)) {
        ctx->limit.counter++;
    }

    // advance DFA state
    const auto & sc = dfa_get_or_build_cache(ctx, ctx->current_state);

    if (!sc.is_accept_heavy) {
        auto it = sc.outlier_next_states.find(token);
        if (it != sc.outlier_next_states.end()) {
            ctx->current_state = it->second;
            return;
        }
    }

    // accept-heavy or cache miss: re-simulate
    const auto & info = ctx->token_info[token];
    if (info.piece.empty()) {
        return;
    }

    ctx->current_state = dfa_simulate_bytes(ctx->dfa, ctx->current_state,
                                            reinterpret_cast<const uint8_t *>(info.piece.data()),
                                            info.piece.size());
}

static void dfa_sampler_apply(struct llama_sampler * smpl,
                              llama_token_data_array * cur_p) {
    auto * ctx = static_cast<common_sampler_dfa *>(smpl->ctx);
    const uint32_t state = ctx->current_state;

    if (state == ctx->dfa.dead_state) {
        for (size_t i = 0; i < cur_p->size; ++i) {
            cur_p->data[i].logit = -INFINITY;
        }
        cur_p->sorted = false;
        return;
    }

    // check if the token limit is exhausted
    const bool limit_hit = ctx->has_limit
        && ctx->limit.member_states.count(state)
        && ctx->limit.counter >= ctx->limit.max_tokens;

    // always apply the normal DFA mask
    const auto & sc = dfa_get_or_build_cache(ctx, state);
    dfa_apply_state_cache(sc, cur_p);

    // additionally apply the exit mask if the limit is hit
    if (limit_hit) {
        const auto & ec = dfa_get_or_build_exit_cache(ctx, state);
        dfa_apply_exit_cache(ec, cur_p);
    }

    cur_p->sorted = false;
}

static void dfa_sampler_reset(struct llama_sampler * smpl) {
    auto * ctx = static_cast<common_sampler_dfa *>(smpl->ctx);
    ctx->current_state = ctx->dfa.start_state;
    if (ctx->has_limit) {
        ctx->limit.counter = 0;
    }
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
    ctx->n_vocab       = n_vocab;
    ctx->current_state = params.start_state;

    ctx->dfa = dfa_build_table(params);

    dfa_build_token_info(ctx, vocab);

    // set up optional token limit
    ctx->has_limit = (params.token_limit_states != nullptr && params.n_token_limit_states > 0);
    if (ctx->has_limit) {
        ctx->limit.max_tokens = params.token_limit;
        ctx->limit.counter    = 0;
        for (size_t i = 0; i < params.n_token_limit_states; ++i) {
            ctx->limit.member_states.insert(params.token_limit_states[i]);
        }
    }

    return llama_sampler_init(&dfa_sampler_vtable, ctx);
}
