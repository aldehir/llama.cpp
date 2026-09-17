#include "testing.h"

#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"
#include "llama-cpp.h"

#include "../src/llama-io.h"
#include "../src/llama-memory.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <vector>

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t count) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t pos = 0; pos < count; ++pos) {
        common_batch_add(batch, tokens[pos], pos, { 0 }, pos + 1 == count);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, tok, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

struct cache_buffer_collector : llama_io_write_i {
    std::set<ggml_backend_buffer_t> buffers;
    size_t size = 0;

    void write(const void *, size_t n) override {
        size += n;
    }

    void write_tensor(ggml_tensor * tensor, size_t, size_t n) override {
        buffers.insert(tensor->buffer);
        size += n;
    }

    size_t n_bytes() override {
        return size;
    }
};

static llama_context * init_ctx(llama_model * model, llama_context_params cparams, uint8_t fill) {
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr || fill == 0) {
        return ctx;
    }

    // Use a full ubatch so buffer discovery preserves prefill allocation sizes.
    const uint32_t n_tokens = llama_n_ubatch(ctx);
    if (!decode_tokens(ctx, std::vector<llama_token>(n_tokens, 0), n_tokens)) {
        llama_free(ctx);
        return nullptr;
    }
    llama_synchronize(ctx);
    cache_buffer_collector collector;
    llama_get_memory(ctx)->state_write(collector);
    llama_memory_clear(llama_get_memory(ctx), true);
    if (collector.buffers.empty()) {
        fprintf(stderr, "%s : no cache buffers found\n", __func__);
        llama_free(ctx);
        return nullptr;
    }
    for (auto * buffer : collector.buffers) {
        ggml_backend_buffer_clear(buffer, fill);
    }
    return ctx;
}

static llama_context * make_ctx(const common_params & params, llama_model * model, uint8_t fill) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = 8;
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (cparams.n_rs_seq + 1));
    return init_ctx(model, cparams, fill);
}

static float logit_diff(float a, float b) {
    return std::isfinite(a) && std::isfinite(b) ? std::fabs(a - b) : std::numeric_limits<float>::infinity();
}

// the first token whose logits differ by more than eps, or -1 when they all match
static int first_logit_mismatch(const float * a, const float * b, int n_vocab, float eps) {
    for (int token = 0; token < n_vocab; ++token) {
        if (logit_diff(a[token], b[token]) > eps) {
            return token;
        }
    }
    return -1;
}

static bool assert_logits_match(testing & t, const std::string & what, const float * a, const float * b, int n_vocab, float eps) {
    const int token = first_logit_mismatch(a, b, n_vocab, eps);
    std::string msg = what + " logits match";
    if (token >= 0) {
        msg += string_format(" (token %d: %g != %g)", token, (double) a[token], (double) b[token]);
    }
    return t.assert_true(msg, token < 0);
}

// Roll back multiple sequences, then replay them in a single batch whose
// per-seq token count exceeds n_ubatch: each seq's replay spans several
// ubatches while its rollback restore is still pending. Compared against a
// reference context that never advanced past the rollback point and decodes
// the identical replay batch.
static void test_multi_seq_split_replay(testing & t, const common_params & params, llama_model * model, const int n_vocab, uint8_t fill) {
    constexpr uint32_t  n_seqs     = 2;
    constexpr uint32_t  n_ubatch   = 16;
    constexpr uint32_t  n_prompt   = 19;
    constexpr uint32_t  n_rollback = 3;
    constexpr uint32_t  n_replay   = 40; // > n_ubatch so each seq spans multiple ubatches
    constexpr llama_pos p0         = n_prompt - n_rollback;

    const auto make_ctx_multi = [&]() {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max  = n_seqs;
        cparams.n_rs_seq   = 8;
        cparams.n_ctx      = 256;
        cparams.n_batch    = 256;
        cparams.n_ubatch   = n_ubatch;
        cparams.kv_unified = false;
        return llama_context_ptr(init_ctx(model, cparams, fill));
    };

    llama_context_ptr ctx_roll = make_ctx_multi();
    llama_context_ptr ctx_ref  = make_ctx_multi();
    if (!t.assert_true("multi-seq contexts initialize", ctx_roll && ctx_ref)) {
        return;
    }

    if (llama_n_rs_seq(ctx_roll.get()) < n_rollback) {
        t.skip("n_rs_seq is too small");
        return;
    }

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((7*(uint32_t) pos + 31*seq + 1) % (uint32_t) n_vocab);
    };

    bool ok = true;

    // both contexts decode the identical [0, p0) prefill; only ctx_roll decodes
    // the tail, which is then rolled back so its restore is pending at replay
    for (uint32_t s = 0; s < n_seqs && ok; ++s) {
        const std::string seq = "seq " + std::to_string(s);

        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (llama_pos pos = 0; pos < (llama_pos) p0; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && t.assert_true(seq + " prefill decodes on ctx_roll", llama_decode(ctx_roll.get(), batch) == 0);
        ok = ok && t.assert_true(seq + " prefill decodes on ctx_ref",  llama_decode(ctx_ref.get(),  batch) == 0);

        common_batch_clear(batch);
        for (llama_pos pos = p0; pos < (llama_pos) n_prompt; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && t.assert_true(seq + " tail decodes on ctx_roll", llama_decode(ctx_roll.get(), batch) == 0);
        llama_batch_free(batch);

        ok = ok && t.assert_true(seq + " rolls back to pos " + std::to_string(p0),
                llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), (llama_seq_id) s, p0, -1));

        // a second partial removal while one is pending must be refused
        ok = ok && t.assert_true(seq + " refuses a second partial removal while one is pending",
                !llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), (llama_seq_id) s, p0 - 1, -1));
    }
    if (!ok) {
        return;
    }

    llama_batch batch = llama_batch_init(n_seqs*n_replay, 0, 1);
    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t i = 0; i < n_replay; ++i) {
            const llama_pos pos = p0 + (llama_pos) i;
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, true);
        }
    }
    ok = t.assert_true("multi-seq replay decodes on ctx_roll", llama_decode(ctx_roll.get(), batch) == 0);
    ok = ok && t.assert_true("multi-seq replay decodes on ctx_ref", llama_decode(ctx_ref.get(), batch) == 0);
    llama_batch_free(batch);
    if (!ok) {
        return;
    }

    // identical ubatch shapes from bit-exact states: a correct implementation
    // matches bitwise, so eps only allows backend scheduling noise
    constexpr float eps = 1e-7f;

    float    diff_max  = 0.0f;
    uint32_t seq_first = 0;
    int32_t  pos_first = -1;
    for (uint32_t i = 0; i < n_seqs*n_replay; ++i) {
        const float * l_roll = llama_get_logits_ith(ctx_roll.get(), i);
        const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  i);
        if (l_roll == nullptr || l_ref == nullptr) {
            t.assert_true("multi-seq logits present at index " + std::to_string(i), false);
            return;
        }
        for (int token = 0; token < n_vocab; ++token) {
            const float diff = logit_diff(l_roll[token], l_ref[token]);
            if (diff > eps && pos_first < 0) {
                seq_first = i/n_replay;
                pos_first = p0 + (int32_t) (i%n_replay);
            }
            diff_max = std::max(diff_max, diff);
        }
    }

    if (!t.assert_true(string_format("multi-seq split replay logits match (max diff %g, first at seq %u pos %d)",
                (double) diff_max, seq_first, pos_first), diff_max <= eps)) {
        return;
    }

    fprintf(stderr, "%s : multi-seq split replay matched (max diff %g)\n", __func__, (double) diff_max);

    // seq-1-only decodes must be independent of seq 0's content: diverge seq 0
    // in ctx_ref only, then compare identical seq-1-only continuations bitwise
    constexpr uint32_t n_tail = 4;

    {
        llama_batch batch_tail = llama_batch_init(n_tail, 0, 1);
        for (uint32_t i = 0; i < n_tail; ++i) {
            const llama_pos pos = p0 + (llama_pos) (n_replay + i);
            common_batch_add(batch_tail, tok(0, pos + 7), pos, { 0 }, false);
        }
        ok = llama_decode(ctx_ref.get(), batch_tail) == 0;
        llama_batch_free(batch_tail);
    }

    float diff_tail = 0.0f;
    for (uint32_t i = 0; i < n_tail && ok; ++i) {
        const llama_pos pos = p0 + (llama_pos) (n_replay + i);
        llama_batch batch_one = llama_batch_init(1, 0, 1);
        common_batch_add(batch_one, tok(1, pos), pos, { 1 }, true);
        ok = llama_decode(ctx_roll.get(), batch_one) == 0;
        ok = ok && llama_decode(ctx_ref.get(), batch_one) == 0;
        llama_batch_free(batch_one);
        if (!ok) {
            break;
        }

        const float * l_roll = llama_get_logits_ith(ctx_roll.get(), 0);
        const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  0);
        ok = l_roll != nullptr && l_ref != nullptr;
        for (int token = 0; ok && token < n_vocab; ++token) {
            diff_tail = std::max(diff_tail, logit_diff(l_roll[token], l_ref[token]));
        }
    }

    if (!t.assert_true(string_format("seq-1-only decode is independent of seq 0 (ok=%d, max diff %g)",
                ok ? 1 : 0, (double) diff_tail), ok && diff_tail <= eps)) {
        return;
    }

    fprintf(stderr, "%s : seq-1-only decode independent of seq 0 (max diff %g)\n", __func__, (double) diff_tail);
}

static void test_rollback(testing & t, const common_params & params, llama_model * model, uint8_t fill) {
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_ptr ctx_src(make_ctx(params, model, fill));
    llama_context_ptr ctx_dst(make_ctx(params, model, fill));
    if (!t.assert_true("source and destination contexts initialize", ctx_src && ctx_dst)) {
        return;
    }

    if (llama_n_rs_seq(ctx_src.get()) == 0) {
        t.skip("n_rs_seq is disabled");
        return;
    }

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    } else {
        tokens = common_tokenize(ctx_src.get(), "The quick brown fox jumps over the lazy dog", true);
    }
    const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src.get());
    constexpr uint32_t n_rollback = 3;
    if (n_rs_seq < n_rollback) {
        t.skip("n_rs_seq is too small");
        return;
    }
    if (!t.assert_true("prompt has tokens", !tokens.empty())) {
        return;
    }
    tokens.resize(n_rs_seq + 1, tokens.back());

    const uint32_t  n_tokens     = tokens.size();
    const llama_pos rollback_pos = (llama_pos) n_tokens - n_rollback;

    // Decode the full prompt on the source, then roll back three positions.
    // Replaying them crosses DSV4's ratio-4 compressor boundary.
    // Rollback leaves the recurrent memory in a snapshot state (rs_idx != 0).
    if (!t.assert_true("prompt decodes on the source context", decode_tokens(ctx_src.get(), tokens, n_tokens))) {
        return;
    }
    if (!t.assert_true("source rolls back to pos " + std::to_string(rollback_pos),
                llama_memory_seq_rm(llama_get_memory(ctx_src.get()), 0, rollback_pos, -1))) {
        return;
    }

    // Save the rolled-back state and restore it into a fresh context.
    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx_src.get(), 0, 0);
    ckpt.load_tgt(ctx_dst.get(), 0, 0);

    constexpr float eps = 1e-5f;
    std::vector<std::vector<float>> logits_src_replay(n_rollback);
    const auto replay_and_compare = [&](testing & t, const char * mode) {
        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            const std::string where = std::string(mode) + " replay at position " + std::to_string(pos);
            if (!t.assert_true(where + " decodes on both contexts",
                        decode_one(ctx_src.get(), tokens[pos], pos) && decode_one(ctx_dst.get(), tokens[pos], pos))) {
                return;
            }

            const float * logits_src = llama_get_logits_ith(ctx_src.get(), 0);
            const float * logits_dst = llama_get_logits_ith(ctx_dst.get(), 0);
            if (!t.assert_true(where + " has logits on both contexts", logits_src != nullptr && logits_dst != nullptr)) {
                return;
            }

            logits_src_replay[i].assign(logits_src, logits_src + n_vocab);
            if (!assert_logits_match(t, where, logits_src, logits_dst, n_vocab, eps)) {
                return;
            }
        }
    };

    t.test("full_replay", [&](testing & t) {
        replay_and_compare(t, "full");
    });
    if (t.failures) {
        return;
    }

    t.test("partial_replay", [&](testing & t) {
        if (!t.assert_true("both contexts roll back to pos " + std::to_string(rollback_pos),
                    llama_memory_seq_rm(llama_get_memory(ctx_src.get()), 0, rollback_pos, -1) &&
                    llama_memory_seq_rm(llama_get_memory(ctx_dst.get()), 0, rollback_pos, -1))) {
            return;
        }

        constexpr llama_state_seq_flags partial_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        common_prompt_checkpoint ckpt_partial;
        ckpt_partial.update_tgt(ctx_src.get(), 0, partial_flags);
        ckpt_partial.load_tgt(ctx_dst.get(), 0, partial_flags);

        replay_and_compare(t, "partial");
    });
    if (t.failures) {
        return;
    }

    // Repeat the load into a context that already has its own rollback state:
    // groups 1..n_rs_seq hold a different prompt's history, and rs_idx[0] is
    // non-zero at load time. The restore must wipe that state and still match.
    t.test("dirty_context", [&](testing & t) {
        llama_context_ptr ctx_dirty(make_ctx(params, model, fill));
        if (!t.assert_true("dirty context initializes", ctx_dirty != nullptr)) {
            return;
        }

        std::vector<llama_token> noise = tokens;
        for (auto & tok : noise) {
            tok = (tok + 1) % n_vocab;
            if (tok < 0) {
                tok = 0;
            }
        }
        if (!t.assert_true("noise prompt decodes on the dirty context", decode_tokens(ctx_dirty.get(), noise, n_tokens))) {
            return;
        }
        if (!t.assert_true("dirty context rolls back to pos " + std::to_string(rollback_pos),
                    llama_memory_seq_rm(llama_get_memory(ctx_dirty.get()), 0, rollback_pos, -1))) {
            return;
        }

        ckpt.load_tgt(ctx_dirty.get(), 0, 0);

        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            const std::string where = "dirty replay at position " + std::to_string(pos);
            if (!t.assert_true(where + " decodes", decode_one(ctx_dirty.get(), tokens[pos], pos))) {
                return;
            }

            const float * logits_dirty = llama_get_logits_ith(ctx_dirty.get(), 0);
            if (!t.assert_true(where + " has logits", logits_dirty != nullptr)) {
                return;
            }

            if (!assert_logits_match(t, where, logits_src_replay[i].data(), logits_dirty, n_vocab, eps)) {
                return;
            }
        }

        fprintf(stderr, "%s : recurrent rollback checkpoint restored successfully\n", __func__);
    });
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    // extract our own --filter RE option before handing the rest to the common arg parser
    std::string filter;
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
            filter = argv[++i];
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }
    filtered_argv.push_back(nullptr);
    const int fargc = (int) filtered_argv.size() - 1;

    if (!common_params_parse(fargc, filtered_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    testing t;
    t.capture_output = true;
    t.apply_env();

    // the model loads before the filter applies so a filtered run still has it
    common_init_result_ptr llama_init;
    llama_model * model = nullptr;
    bool recurrent = false;
    t.test("load_model", [&](testing & t) {
        llama_init = common_init_from_params(params);
        model = llama_init->model();
        if (!t.assert_true("model loads: " + params.model.path, model != nullptr)) {
            return;
        }
        recurrent = llama_model_is_recurrent(model) || llama_model_is_hybrid(model);
        if (!recurrent) {
            t.skip("non-recurrent model");
        }
    });
    if (model == nullptr || !recurrent) {
        return t.summary();
    }

    if (!filter.empty()) {
        t.set_filter(filter);
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    for (uint8_t fill : { 0, 0x3e }) {
        t.test(string_format("fill_0x%02x", fill), [&](testing & t) {
            t.test("checkpoint", [&](testing & t) {
                test_rollback(t, params, model, fill);
            });
            t.test("multi_seq_split_replay", [&](testing & t) {
                test_multi_seq_split_replay(t, params, model, n_vocab, fill);
            });
        });
    }

    return t.summary();
}
