// Test for state restore with fragmented KV cache
// This tests the fix for: https://github.com/ggml-org/llama.cpp/issues/17527
// The issue was that state restore required contiguous KV cache slots,
// which fails when the cache is fragmented.
//
// The fix changes find_slot(ubatch, true) to find_slot(ubatch, false)
// in state_read_meta(), allowing non-contiguous slot allocation.

#include "testing.h"

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static bool decode_prompt(testing & t, llama_context * ctx, llama_batch & batch, size_t n_tokens) {
    if (!t.assert_true("decode the interleaved prompt on seq 0, 1, 2", llama_decode(ctx, batch) == 0)) {
        fprintf(stderr, "%s : failed to decode seq 0\n", __func__);
        return false;
    }

    fprintf(stderr, "%s : processed prompt on seq 0, 1, 2 (%zu tokens each)\n", __func__, n_tokens);
    return true;
}

static bool save_seq_state(testing & t, llama_context * ctx, std::vector<uint8_t> & seq_state) {
    const size_t ncopy = llama_state_seq_get_data(ctx, seq_state.data(), seq_state.size(), 1);
    if (!t.assert_equal("bytes copied out of the seq 1 state", seq_state.size(), ncopy)) {
        fprintf(stderr, "%s : failed to save seq 1 state\n", __func__);
        return false;
    }
    fprintf(stderr, "%s : saved seq 1 state, %zu bytes\n", __func__, ncopy);
    return true;
}

static bool restore_seq_state(testing & t, llama_context * ctx, const std::vector<uint8_t> & seq_state) {
    // clear seq 1 to create a "hole" in the KV cache (fragmentation)
    // 0.20.20.20.2....
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, 1, -1, -1);
    fprintf(stderr, "%s : cleared seq 1 to create fragmentation\n", __func__);

    // Now the cache has holes where seq 1 was
    // This creates fragmentation - there's no contiguous block large enough
    // for the seq 1 state if we only look for contiguous slots

    // Restore seq 1 state into seq 1 (should work with non-contiguous allocation)
    // We use seq 1 since it's a valid sequence ID (0 to n_parallel-1)
    // Before the fix, this would fail with "failed to find available cells in kv cache"
    const size_t nset = llama_state_seq_set_data(ctx, seq_state.data(), seq_state.size(), 1);
    if (!t.assert_equal("bytes restored into seq 1 of the fragmented cache", seq_state.size(), nset)) {
        fprintf(stderr, "%s : FAILED to restore seq state into fragmented cache (got %zu, expected %zu)\n",
                __func__, nset, seq_state.size());
        fprintf(stderr, "%s : This is the bug - state restore fails with fragmented KV cache\n", __func__);
        return false;
    }
    fprintf(stderr, "%s : restored state into seq 1, %zu bytes\n", __func__, nset);
    return true;
}

static void decode_restored_state(testing & t, llama_context * ctx, llama_batch & batch, size_t n_tokens, uint32_t seed) {
    // Verify we can decode with the restored state
    // Generate one token to verify the restored state is usable
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));

    auto next_token = llama_sampler_sample(smpl, ctx, -1);
    auto next_token_str = common_token_to_piece(ctx, next_token);

    common_batch_clear(batch);
    common_batch_add(batch, next_token, (int)n_tokens, {1}, true);

    const std::string msg = "decode token " + std::to_string(next_token) + " '" + next_token_str + "' on the restored seq 1";
    if (!t.assert_true(msg, llama_decode(ctx, batch) == 0)) {
        fprintf(stderr, "%s : failed to decode with restored state\n", __func__);
        llama_sampler_free(smpl);
        return;
    }

    fprintf(stderr, "%s : successfully decoded with restored state, generated: '%s'\n", __func__, next_token_str.c_str());
    fprintf(stderr, "%s : SUCCESS - state restore works with fragmented KV cache\n", __func__);

    llama_sampler_free(smpl);
}

// the steps build on each other, so the first failure stops the sequence
static void test_restore(testing & t, const common_params & params, llama_context * ctx) {
    // tokenize prompt
    std::vector<llama_token> tokens(70, 1);

    // interleave the 3 sequences:
    // 01201230123...
    llama_batch batch = llama_batch_init(params.n_parallel*tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        for (int s = 0; s < params.n_parallel; ++s) {
            common_batch_add(batch, tokens[i], i, {s}, false);
        }
    }
    batch.logits[batch.n_tokens - 1] = true;

    // Save state of seq 1
    std::vector<uint8_t> seq_state;

    bool ok = false;

    t.test("decode_prompt", [&](testing & t) {
        ok = decode_prompt(t, ctx, batch, tokens.size());
    });

    if (ok) {
        t.test("save_seq_state", [&](testing & t) {
            seq_state.resize(llama_state_seq_get_size(ctx, 1));
            ok = save_seq_state(t, ctx, seq_state);
        });
    }

    if (ok) {
        t.test("restore_into_fragmented_cache", [&](testing & t) {
            ok = restore_seq_state(t, ctx, seq_state);
        });
    }

    if (ok) {
        t.test("decode_restored_state", [&](testing & t) {
            decode_restored_state(t, ctx, batch, tokens.size(), params.sampling.seed);
        });
    }

    llama_batch_free(batch);
}

int main(int argc, char ** argv) {
    common_params params;

    params.sampling.seed = 1234;
    params.kv_unified = true;
    params.n_parallel = 3;
    params.n_ctx = 256;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    testing t;
    t.capture_output = true;
    t.apply_env();

    // init

    ggml_backend_load_all();

    common_init_result_ptr llama_init;

    llama_context * ctx = nullptr;

    t.test("init", [&](testing & t) {
        llama_init = common_init_from_params(params);

        llama_model * model = llama_init->model();
        ctx = llama_init->context();

        if (!t.assert_true("load model '" + params.model.path + "' and create a context", model != nullptr && ctx != nullptr)) {
            fprintf(stderr, "%s : failed to init\n", __func__);
            ctx = nullptr;
        }
    });

    if (ctx != nullptr) {
        t.test("restore", [&](testing & t) {
            test_restore(t, params, ctx);
        });
    }

    return t.summary();
}
