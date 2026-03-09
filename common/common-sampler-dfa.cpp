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

// Step the DFA one byte. Returns next state (may be dead_state).
static uint32_t dfa_step_byte(const dfa_table & dfa, uint32_t state, uint8_t byte) {
    if (state == dfa.dead_state) {
        return dfa.dead_state;
    }
    const auto & es = dfa.edges[state];
    // binary search for the edge whose range contains byte
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

// Simulate a full byte sequence through the DFA.
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
// Token-limited regions
// ----------------------------------------------------------------------------

struct dfa_region {
    std::unordered_set<uint32_t> member_states;
    int32_t                      max_tokens;
    int32_t                      counter; // current token count
};

// ----------------------------------------------------------------------------
// Adaptive per-state caches (lazy)
// ----------------------------------------------------------------------------

struct dfa_state_cache {
    bool is_accept_heavy;

    // Minority set of token ids
    std::unordered_set<llama_token> outlier_tokens;

    // For reject-heavy: maps accepted token → resulting DFA state
    // For accept-heavy: empty (re-simulate on accept)
    std::unordered_map<llama_token, uint32_t> outlier_next_states;
};

struct dfa_exit_cache_entry {
    bool is_exit_heavy; // true → most accepted tokens exit → outliers STAY

    // If is_exit_heavy: contains tokens that are accepted but STAY in region
    // If !is_exit_heavy: contains tokens that are accepted and EXIT region
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

    // regions & per-state region membership
    std::vector<dfa_region>                regions;
    std::vector<std::vector<uint32_t>>     state_regions; // state → region indices

    // lazy caches
    std::unordered_map<uint32_t, dfa_state_cache>  cache;

    // exit cache keyed by (state << 32 | region_idx)
    std::unordered_map<uint64_t, dfa_exit_cache_entry> exit_cache;
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
        common_sampler_dfa * ctx, uint32_t state, uint32_t region_idx) {
    uint64_t key = (uint64_t)state << 32 | region_idx;
    auto it = ctx->exit_cache.find(key);
    if (it != ctx->exit_cache.end()) {
        return it->second;
    }

    const auto &  dfa     = ctx->dfa;
    const int32_t n_vocab = ctx->n_vocab;
    const auto &  region  = ctx->regions[region_idx];

    // Tokens that are accepted by the DFA AND end outside this region
    std::vector<llama_token> exits;
    // Tokens that are accepted by the DFA but stay in the region (or die)
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
        if (next == dfa.dead_state || region.member_states.count(next)) {
            stays.push_back(tok);
        } else {
            exits.push_back(tok);
        }
    }

    dfa_exit_cache_entry ec;
    ec.is_exit_heavy = exits.size() > stays.size();

    if (ec.is_exit_heavy) {
        // most tokens exit → store the ones that stay as outliers
        for (const auto & tok : stays) {
            ec.outlier_tokens.insert(tok);
        }
    } else {
        // most tokens stay → store the ones that exit as outliers
        for (const auto & tok : exits) {
            ec.outlier_tokens.insert(tok);
        }
    }

    auto [ins, _] = ctx->exit_cache.emplace(key, std::move(ec));
    return ins->second;
}

// ----------------------------------------------------------------------------
// Token info
// ----------------------------------------------------------------------------

static void dfa_build_token_info(common_sampler_dfa * ctx,
                                 const struct llama_vocab * vocab) {
    const int32_t n_vocab = ctx->n_vocab;
    ctx->token_info.resize(n_vocab);

    for (int32_t tok = 0; tok < n_vocab; ++tok) {
        std::string piece = common_token_to_piece(vocab, tok, false);
        ctx->token_info[tok].piece = std::move(piece);
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

    // increment counters for all regions that contain the current state
    const auto & sr = ctx->state_regions[ctx->current_state];
    for (uint32_t ri : sr) {
        ctx->regions[ri].counter++;
    }

    // advance DFA state
    const auto & sc = dfa_get_or_build_cache(ctx, ctx->current_state);

    if (!sc.is_accept_heavy) {
        // reject-heavy: accepted tokens have cached next_state
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

    // check which regions have exhausted their token limits
    const auto & sr = ctx->state_regions[state];
    std::vector<uint32_t> exhausted_regions;
    for (uint32_t ri : sr) {
        if (ctx->regions[ri].counter >= ctx->regions[ri].max_tokens) {
            exhausted_regions.push_back(ri);
        }
    }

    if (!exhausted_regions.empty()) {
        // apply exit masks: only allow tokens that exit ALL exhausted regions
        // first pass: apply normal DFA mask (reject dead-state tokens)
        const auto & sc = dfa_get_or_build_cache(ctx, state);
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

        // second pass: for each exhausted region, mask tokens that don't exit
        for (uint32_t ri : exhausted_regions) {
            const auto & ec = dfa_get_or_build_exit_cache(ctx, state, ri);
            if (ec.is_exit_heavy) {
                // most tokens exit → mask outliers (the ones that stay)
                for (size_t i = 0; i < cur_p->size; ++i) {
                    if (ec.outlier_tokens.count(cur_p->data[i].id)) {
                        cur_p->data[i].logit = -INFINITY;
                    }
                }
            } else {
                // most tokens stay → only allow outliers (the ones that exit)
                for (size_t i = 0; i < cur_p->size; ++i) {
                    if (!ec.outlier_tokens.count(cur_p->data[i].id)) {
                        cur_p->data[i].logit = -INFINITY;
                    }
                }
            }
        }
    } else {
        // no limit hit: normal DFA masking
        const auto & sc = dfa_get_or_build_cache(ctx, state);
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

    cur_p->sorted = false;
}

static void dfa_sampler_reset(struct llama_sampler * smpl) {
    auto * ctx = static_cast<common_sampler_dfa *>(smpl->ctx);
    ctx->current_state = ctx->dfa.start_state;
    for (auto & r : ctx->regions) {
        r.counter = 0;
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

    // build byte-level DFA
    ctx->dfa = dfa_build_table(params);

    // decode all tokens to raw bytes
    dfa_build_token_info(ctx, vocab);

    // build regions and state→region index
    ctx->state_regions.resize(params.n_states);
    for (size_t r = 0; r < params.n_token_limits; ++r) {
        const auto & tl = params.token_limits[r];
        dfa_region region;
        region.max_tokens = tl.max_tokens;
        region.counter    = 0;
        for (size_t s = 0; s < tl.n_states; ++s) {
            region.member_states.insert(tl.states[s]);
            ctx->state_regions[tl.states[s]].push_back((uint32_t)r);
        }
        ctx->regions.push_back(std::move(region));
    }

    // caches are lazy

    return llama_sampler_init(&dfa_sampler_vtable, ctx);
}
