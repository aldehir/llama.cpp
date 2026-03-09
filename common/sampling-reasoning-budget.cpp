#include "sampling-reasoning-budget.h"

#include <cmath>

enum reasoning_budget_state {
    REASONING_BUDGET_COUNTING,
    REASONING_BUDGET_FORCING,
    REASONING_BUDGET_PASSTHROUGH,
};

struct llama_sampler_reasoning_budget {
    int32_t budget;
    llama_token end_token;

    reasoning_budget_state state;
    int32_t n_sampled;
};

static const char * llama_sampler_reasoning_budget_name(const struct llama_sampler * /*smpl*/) {
    return "reasoning-budget";
}

static void llama_sampler_reasoning_budget_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (llama_sampler_reasoning_budget *) smpl->ctx;

    switch (ctx->state) {
        case REASONING_BUDGET_COUNTING:
            if (token == ctx->end_token) {
                ctx->state = REASONING_BUDGET_PASSTHROUGH;
            } else {
                ctx->n_sampled++;
                if (ctx->n_sampled >= ctx->budget) {
                    ctx->state = REASONING_BUDGET_FORCING;
                }
            }
            break;
        case REASONING_BUDGET_FORCING:
            // The only token that should have been sampled is the end token.
            ctx->state = REASONING_BUDGET_PASSTHROUGH;
            break;
        case REASONING_BUDGET_PASSTHROUGH:
            break;
    }
}

static void llama_sampler_reasoning_budget_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_reasoning_budget *) smpl->ctx;

    switch (ctx->state) {
        case REASONING_BUDGET_COUNTING:
            // Allow everything through — no modification.
            break;
        case REASONING_BUDGET_FORCING:
            // Set all logits to -inf except the end token.
            for (size_t i = 0; i < cur_p->size; i++) {
                if (cur_p->data[i].id != ctx->end_token) {
                    cur_p->data[i].logit = -INFINITY;
                }
            }
            break;
        case REASONING_BUDGET_PASSTHROUGH:
            // Allow everything through — no modification.
            break;
    }
}

static void llama_sampler_reasoning_budget_reset(struct llama_sampler * smpl) {
    auto * ctx = (llama_sampler_reasoning_budget *) smpl->ctx;
    ctx->state = REASONING_BUDGET_COUNTING;
    ctx->n_sampled = 0;
}

static struct llama_sampler * llama_sampler_reasoning_budget_clone(const struct llama_sampler * smpl) {
    const auto * ctx = (const llama_sampler_reasoning_budget *) smpl->ctx;
    return common_sampler_init_reasoning_budget(ctx->budget, ctx->end_token);
}

static void llama_sampler_reasoning_budget_free(struct llama_sampler * smpl) {
    delete (llama_sampler_reasoning_budget *) smpl->ctx;
}

static struct llama_sampler_i llama_sampler_reasoning_budget_i = {
    /* .name   = */ llama_sampler_reasoning_budget_name,
    /* .accept = */ llama_sampler_reasoning_budget_accept,
    /* .apply  = */ llama_sampler_reasoning_budget_apply,
    /* .reset  = */ llama_sampler_reasoning_budget_reset,
    /* .clone  = */ llama_sampler_reasoning_budget_clone,
    /* .free   = */ llama_sampler_reasoning_budget_free,
};

struct llama_sampler * common_sampler_init_reasoning_budget(int32_t budget, llama_token end_token) {
    return llama_sampler_init(
        &llama_sampler_reasoning_budget_i,
        new llama_sampler_reasoning_budget {
            /* .budget    = */ budget,
            /* .end_token = */ end_token,
            /* .state     = */ REASONING_BUDGET_COUNTING,
            /* .n_sampled = */ 0,
        }
    );
}
