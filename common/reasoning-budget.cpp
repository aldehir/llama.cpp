#include "reasoning-budget.h"
#include "common.h"
#include "unicode.h"

#include "log.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

struct token_matcher {
    std::vector<llama_token> tokens;
    size_t pos = 0;

    bool advance(llama_token token) {
        if (tokens.empty()) {
            return false;
        }

        if (token == tokens[pos]) {
            pos++;
            if (pos >= tokens.size()) {
                pos = 0;
                return true;
            }
        } else {
            pos = 0;
            if (token == tokens[0]) {
                pos = 1;
            }
        }
        return false;
    }

    void reset() { pos = 0; }
};

struct common_reasoning_budget_ctx {
    const llama_vocab * vocab;

    token_matcher start_matcher;
    token_matcher end_matcher;
    std::vector<llama_token> forced_tokens;

    int32_t budget;           // maximum tokens in reasoning block
    int32_t remaining;        // tokens remaining in budget

    common_reasoning_budget_state state;

    // for forcing
    size_t force_pos;         // next position in forced_tokens to force

    // backend sampling inputs
    struct ggml_tensor * inp_idx;  // I32 [1], token to keep
    struct ggml_tensor * inp_gate; // F32 [1], mask value (0 when passthrough)
};

static const char * common_reasoning_budget_name(const struct llama_sampler * /*smpl*/) {
    return "reasoning-budget";
}

static void common_reasoning_budget_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    switch (ctx->state) {
        case REASONING_BUDGET_IDLE:
        {
            if (ctx->start_matcher.advance(token)) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->remaining = ctx->budget;
                COM_TRC("activated, budget=%d tokens\n", ctx->budget);

                if (ctx->remaining <= 0) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    COM_TRC("%s", "budget=0, forcing immediately\n");
                }
            }
            break;
        }
        case REASONING_BUDGET_COUNTING:
        case REASONING_BUDGET_WAITING_UTF8:
        {
            if (ctx->end_matcher.advance(token)) {
                ctx->state = REASONING_BUDGET_DONE;
                COM_TRC("%s", "deactivated (natural end)\n");
                break;
            }

            bool utf8_complete = true;
            if (ctx->vocab != nullptr) {
                const std::string piece = common_token_to_piece(ctx->vocab, token, false);
                utf8_complete = common_utf8_is_complete(piece);
            }

            if (ctx->state == REASONING_BUDGET_WAITING_UTF8) {
                if (utf8_complete) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    ctx->end_matcher.reset();
                    COM_TRC("%s", "UTF-8 complete, now forcing end sequence\n");
                }
            } else if (ctx->state == REASONING_BUDGET_COUNTING) {
                ctx->remaining--;
                if (ctx->remaining <= 0) {
                    if (utf8_complete) {
                        ctx->state = REASONING_BUDGET_FORCING;
                        ctx->force_pos = 0;
                        ctx->end_matcher.reset();
                        COM_TRC("%s", "budget exhausted, forcing end sequence\n");
                    } else {
                        ctx->state = REASONING_BUDGET_WAITING_UTF8;
                        ctx->end_matcher.reset();
                        COM_TRC("%s", "budget exhausted, waiting for UTF-8 completion\n");
                    }
                }
            }
            break;
        }
        case REASONING_BUDGET_FORCING:
            ctx->force_pos++;
            if (ctx->force_pos >= ctx->forced_tokens.size()) {
                ctx->state = REASONING_BUDGET_DONE;
                COM_TRC("%s", "forced sequence complete, done\n");
            }
            break;
        case REASONING_BUDGET_DONE:
            // Re-arm on a new start tag: some models emit multiple <think> blocks
            // per response, and each should get a fresh budget window.
            if (ctx->start_matcher.advance(token)) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->remaining = ctx->budget;
                ctx->end_matcher.reset();
                COM_TRC("re-activated on new start tag, budget=%d tokens\n", ctx->budget);

                if (ctx->remaining <= 0) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    COM_TRC("%s", "budget=0, forcing immediately\n");
                }
            }
            break;
    }
}

static void common_reasoning_budget_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    if (ctx->state != REASONING_BUDGET_FORCING) {
        // passthrough — don't modify logits
        return;
    }

    if (ctx->force_pos >= ctx->forced_tokens.size()) {
        return;
    }

    const llama_token forced = ctx->forced_tokens[ctx->force_pos];

    // set all logits to -inf except the forced token
    for (size_t i = 0; i < cur_p->size; i++) {
        if (cur_p->data[i].id != forced) {
            cur_p->data[i].logit = -INFINITY;
        }
    }
}

// The backend implementation adds an inverse mask to the logits: inp_gate
// everywhere except inp_idx, which gets 0. While passthrough the gate is 0
// and the logits are unchanged. When forcing, a large negative gate removes
// every candidate but the forced token, whose logit keeps its original
// small magnitude so the relative threshold math of downstream samplers
// (e.g. min-p) is not affected by float rounding. The state machine itself
// remains on the CPU, driven by accept(); only the per-step {token, gate}
// pair is uploaded to the backend.
static void common_reasoning_budget_backend_apply(
        struct llama_sampler      * smpl,
        struct ggml_context       * ctx,
        struct ggml_cgraph        * gf,
        struct llama_sampler_data * data) {
    GGML_UNUSED(gf);

    auto * sctx = (common_reasoning_budget_ctx *) smpl->ctx;

    sctx->inp_gate = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name (sctx->inp_gate, "rbudget_gate");
    ggml_set_input(sctx->inp_gate);

    sctx->inp_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name (sctx->inp_idx, "rbudget_idx");
    ggml_set_input(sctx->inp_idx);

    ggml_tensor * ones = ggml_scale_bias(ctx, data->logits, 0.0f, 1.0f);

    ggml_tensor * mask = ggml_mul(ctx, ones, sctx->inp_gate);

    ggml_tensor * zero = ggml_reshape_2d(ctx, ggml_scale(ctx, sctx->inp_gate, 0.0f), 1, 1);

    mask = ggml_reshape_2d(ctx, mask, 1, ggml_nelements(mask));
    mask = ggml_set_rows(ctx, mask, zero, sctx->inp_idx);
    mask = ggml_reshape_1d(ctx, mask, ggml_nelements(mask));

    data->logits = ggml_add(ctx, data->logits, mask);
}

static void common_reasoning_budget_backend_set_input(struct llama_sampler * smpl) {
    auto * sctx = (common_reasoning_budget_ctx *) smpl->ctx;

    GGML_ASSERT(sctx->inp_idx  != nullptr);
    GGML_ASSERT(sctx->inp_gate != nullptr);

    int32_t idx  = 0;
    float   gate = 0.0f;

    if (sctx->state == REASONING_BUDGET_FORCING && sctx->force_pos < sctx->forced_tokens.size()) {
        idx  = sctx->forced_tokens[sctx->force_pos];
        gate = -1e9f;
    }

    ggml_backend_tensor_set(sctx->inp_idx,  &idx,  0, sizeof(idx));
    ggml_backend_tensor_set(sctx->inp_gate, &gate, 0, sizeof(gate));
}

static bool common_reasoning_budget_backend_init(
        struct llama_sampler       * smpl,
        ggml_backend_buffer_type_t   buft) {
    auto * device = ggml_backend_buft_get_device(buft);
    if (!device) {
        // CPU backend always supported
        return true;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 128*ggml_tensor_overhead() + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    llama_sampler_data data = {
        /*.logits     =*/ ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024*1024),
        /*.probs      =*/ nullptr,
        /*.sampled    =*/ nullptr,
        /*.candidates =*/ nullptr,
    };

    ggml_cgraph * gf = ggml_new_graph(ctx);

    common_reasoning_budget_backend_apply(smpl, ctx, gf, &data);

    ggml_build_forward_expand(gf, data.logits);

    bool res = true;
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        struct ggml_tensor * op = ggml_graph_node(gf, i);

        if (!ggml_backend_dev_supports_op(device, op)) {
            LOG_WRN("%s: device '%s' does not have support for op %s needed for sampler 'reasoning-budget'\n",
                    __func__, ggml_backend_dev_name(device), ggml_op_name(op->op));

            res = false;
            break;
        }
    }

    ggml_free(ctx);

    return res;
}

static void common_reasoning_budget_reset(struct llama_sampler * smpl) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;
    ctx->state = REASONING_BUDGET_IDLE;
    ctx->remaining = ctx->budget;
    ctx->start_matcher.reset();
    ctx->end_matcher.reset();
    ctx->force_pos = 0;
}

static struct llama_sampler * common_reasoning_budget_init_state(
        const struct llama_vocab * vocab, const std::vector<llama_token> & start_tokens,
        const std::vector<llama_token> & end_tokens, const std::vector<llama_token> & forced_tokens,
        int32_t budget, common_reasoning_budget_state initial_state);

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl);

static void common_reasoning_budget_free(struct llama_sampler * smpl) {
    delete (common_reasoning_budget_ctx *) smpl->ctx;
}

static struct llama_sampler_i common_reasoning_budget_i = {
    /* .name              = */ common_reasoning_budget_name,
    /* .accept            = */ common_reasoning_budget_accept,
    /* .apply             = */ common_reasoning_budget_apply,
    /* .reset             = */ common_reasoning_budget_reset,
    /* .clone             = */ common_reasoning_budget_clone,
    /* .free              = */ common_reasoning_budget_free,
    /* .backend_init      = */ common_reasoning_budget_backend_init,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ common_reasoning_budget_backend_apply,
    /* .backend_set_input = */ common_reasoning_budget_backend_set_input,
};

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl) {
    const auto * ctx = (const common_reasoning_budget_ctx *) smpl->ctx;

    auto * ctx_clone = new common_reasoning_budget_ctx(*ctx);

    // input tensors belong to the source sampler's graph
    ctx_clone->inp_idx  = nullptr;
    ctx_clone->inp_gate = nullptr;

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ ctx_clone
    );
}

static struct llama_sampler * common_reasoning_budget_init_state(
        const struct llama_vocab             * vocab,
        const std::vector<llama_token>       & start_tokens,
        const std::vector<llama_token>       & end_tokens,
        const std::vector<llama_token>       & forced_tokens,
        int32_t                                budget,
        common_reasoning_budget_state          initial_state) {
    // promote COUNTING with budget <= 0 to FORCING
    if (initial_state == REASONING_BUDGET_COUNTING && budget <= 0) {
        initial_state = REASONING_BUDGET_FORCING;
    }

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ new common_reasoning_budget_ctx {
            /* .vocab         = */ vocab,
            /* .start_matcher = */ { start_tokens, 0 },
            /* .end_matcher   = */ { end_tokens, 0 },
            /* .forced_tokens = */ forced_tokens,
            /* .budget        = */ budget,
            /* .remaining     = */ budget,
            /* .state         = */ initial_state,
            /* .force_pos     = */ 0,
            /* .inp_idx       = */ nullptr,
            /* .inp_gate      = */ nullptr,
        }
    );
}

struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab       * vocab,
        const std::vector<llama_token> & start_tokens,
        const std::vector<llama_token> & end_tokens,
        const std::vector<llama_token> & forced_tokens,
        int32_t                          budget,
        common_reasoning_budget_state    initial_state) {
    return common_reasoning_budget_init_state(vocab, start_tokens, end_tokens, forced_tokens, budget, initial_state);
}

common_reasoning_budget_state common_reasoning_budget_get_state(const struct llama_sampler * smpl) {
    if (!smpl) {
        return REASONING_BUDGET_IDLE;
    }
    return ((const common_reasoning_budget_ctx *)smpl->ctx)->state;
}

bool common_reasoning_budget_force(struct llama_sampler * smpl) {
    if (!smpl) {
        return false;
    }

    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    // only a sampler that is actively counting down the budget may be forced;
    // any other state (idle, already forcing/waiting, or done) is left untouched
    if (ctx->state != REASONING_BUDGET_COUNTING) {
        return false;
    }

    ctx->state = REASONING_BUDGET_FORCING;
    ctx->force_pos = 0;
    ctx->end_matcher.reset();
    COM_TRC("%s", "forced into forcing state (manual transition)\n");

    return true;
}
