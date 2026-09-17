#include "testing.h"

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama-cpp.h"

#include <algorithm>
#include <clocale>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

struct llama_batch_ptr {
    llama_batch batch;

    llama_batch_ptr(int32_t n_tokens, int32_t embd, int32_t n_seq_max)
        : batch{llama_batch_init(n_tokens, embd, n_seq_max)} {}

    ~llama_batch_ptr() { llama_batch_free(batch); }

    llama_batch_ptr(const llama_batch_ptr &) = delete;
    llama_batch_ptr & operator=(const llama_batch_ptr &) = delete;
    llama_batch_ptr(llama_batch_ptr &&) = default;
    llama_batch_ptr & operator=(llama_batch_ptr &&) = default;

    llama_batch & get() { return batch; }
    const llama_batch & get() const { return batch; }
};

static llama_tokens generate_tokens(llama_context * ctx, llama_sampler * smpl, int & n_past, int32_t n_predict, llama_seq_id seq_id) {
    llama_tokens result;
    llama_batch_ptr batch(1, 0, 1);

    for (int i = 0; i < n_predict; i++) {
        auto next_token = llama_sampler_sample(smpl, ctx, -1);

        LOG("%d ", next_token);
        result.push_back(next_token);

        common_batch_clear(batch.get());
        common_batch_add(batch.get(), next_token, n_past, {seq_id}, true);

        if (llama_decode(ctx, batch.get())) {
            LOG_ERR("\n%s: failed to evaluate\n", __func__);
            return {};
        }
        n_past++;
    }

    return result;
}

// Test 1: baseline
// - decode all but the last token
// - save state to disk
// - decode the last token
// - generate n_predict tokens
static llama_tokens test_baseline(testing & t, struct llama_model * model, const struct common_params & params, const llama_tokens & tokens) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    auto n_past = 0;
    if (!t.assert_true("prompt decodes and the state is saved to " + params.out_file,
                common_prompt_batch_decode(ctx.get(), tokens, (int)tokens.size(), n_past, params.n_batch, params.out_file, true))) {
        return {};
    }

    LOG("\n=== Test 1: baseline ===\n");

    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 0);
    t.assert_true(string_format("generated %d tokens", params.n_predict), !result.empty());

    LOG("\n");

    return result;
}


// Test 2: sequence removal isolation
// - decode the same prefix into two sequences
// - remove sequence 0
// - verify that sequence 1 remains unchanged
static void test_seq_rm_isolated(
        testing                    & t,
        struct llama_model         * model,
        const struct common_params & params,
        const llama_tokens         & tokens) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_ctx      = 256;
    params_ctx.n_seq_max  = 2;
    params_ctx.kv_unified = true;

    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
    if (!t.assert_true("context is created", ctx != nullptr)) {
        return;
    }

    LOG("\n=== Test 2: sequence removal isolation ===\n");

    const size_t n_tokens = tokens.size() < 128 ? tokens.size() : 128;
    for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
        llama_batch_ptr batch(n_tokens, 0, 1);
        for (size_t i = 0; i < n_tokens; ++i) {
            common_batch_add(batch.get(), tokens[i], i, { seq_id }, i == n_tokens - 1);
        }

        if (!t.assert_true(string_format("prompt decodes for sequence %d", seq_id), llama_decode(ctx.get(), batch.get()) == 0)) {
            return;
        }
    }

    const auto get_seq_state = [&](llama_seq_id seq_id, std::vector<uint8_t> & state) {
        const size_t state_size = llama_state_seq_get_size(ctx.get(), seq_id);
        if (!t.assert_true(string_format("sequence %d state is not empty", seq_id), state_size != 0)) {
            return false;
        }

        state.resize(state_size);
        const size_t ncopy = llama_state_seq_get_data(ctx.get(), state.data(), state.size(), seq_id);
        return t.assert_true(string_format("sequence %d state length %zu matches expected length %zu", seq_id, ncopy, state.size()),
                ncopy == state.size());
    };

    std::vector<uint8_t> state_before;
    if (!get_seq_state(1, state_before)) {
        return;
    }

    if (!t.assert_true("sequence 0 is removed", llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, -1, -1))) {
        return;
    }

    std::vector<uint8_t> state_after;
    if (!get_seq_state(1, state_after)) {
        return;
    }

    t.assert_true("removing sequence 0 leaves sequence 1 unchanged", state_before == state_after);
}


// Test 3: state load
// - create a new context
// - load state from file
// - replay the last prompt token
// - generate n_predict tokens and compare against expected result
static void test_state_load(testing & t, struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const llama_tokens & expected_result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOG("\n=== Test 3: state load ===\n");

    // Load state from file
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!t.assert_true("state loads from " + params.out_file,
                llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out))) {
        return;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out - 1;
    if (!t.assert_true("last prompt token replays", common_replay_last_token(ctx.get(), tokens.back(), n_past))) {
        return;
    }
    n_past++;

    // Generate tokens
    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 0);
    if (!t.assert_true(string_format("generated %d tokens", params.n_predict), !result.empty())) {
        return;
    }

    t.assert_true("generation matches the baseline", result == expected_result);
}


// Test 4: seq copy (host)
// - create a multi-seq context
// - load state from file
// - replay the last prompt token
// - migrate KV cache from seq 0 to seq 1 via the CPU path
// - generate n_predict tokens on seq 1 and compare against expected result
static void test_seq_cp_host(testing & t, struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const llama_tokens & expected_result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOG("\n=== Test 4: seq copy (host) ===\n");

    // Load state from file
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!t.assert_true("state loads from " + params.out_file,
                llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out))) {
        return;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out - 1;
    if (!t.assert_true("last prompt token replays", common_replay_last_token(ctx.get(), tokens.back(), n_past))) {
        return;
    }
    n_past++;

    // Migrate KV cache from seq 0 to seq 1 (CPU path)
    {
        std::vector<uint8_t> seq_store(llama_state_seq_get_size(ctx.get(), 0));
        const size_t ncopy = llama_state_seq_get_data(ctx.get(), seq_store.data(), seq_store.size(), 0);
        if (!t.assert_true(string_format("seq copy data length %zu matches expected length %zu", ncopy, seq_store.size()), ncopy == seq_store.size())) {
            return;
        }
        LOG_TRC("%s: seq 0 copied, %zd bytes\n", __func__, ncopy);

        llama_memory_clear(llama_get_memory(ctx.get()), true);
        LOG_TRC("%s: kv cache cleared\n", __func__);

        const size_t nset = llama_state_seq_set_data(ctx.get(), seq_store.data(), seq_store.size(), 1);
        if (!t.assert_true(string_format("seq set data length %zu matches expected length %zu", nset, seq_store.size()), nset == seq_store.size())) {
            return;
        }
        LOG_TRC("%s: seq 1 restored, %zd bytes\n", __func__, nset);
    }

    // Generate tokens on seq 1
    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 1);
    if (!t.assert_true(string_format("generated %d tokens on seq 1", params.n_predict), !result.empty())) {
        return;
    }

    t.assert_true("generation on seq 1 matches the baseline", result == expected_result);
}


// Test 5: seq copy (device)
// - create a multi-seq context
// - load state from file
// - replay the last prompt token
// - migrate KV cache from seq 0 to seq 1 via the on-device path
// - generate n_predict tokens on seq 1 and compare against expected result
static void test_seq_cp_device(testing & t, struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const llama_tokens & expected_result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOG("\n=== Test 5: seq copy (device) ===\n");

    // Load state from file
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!t.assert_true("state loads from " + params.out_file,
                llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out))) {
        return;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out - 1;
    if (!t.assert_true("last prompt token replays", common_replay_last_token(ctx.get(), tokens.back(), n_past))) {
        return;
    }
    n_past++;

    // Migrate KV cache from seq 0 to seq 1 (on-device path)
    {
        std::vector<uint8_t> seq_store(llama_state_seq_get_size_ext(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE));
        const size_t ncopy = llama_state_seq_get_data_ext(ctx.get(), seq_store.data(), seq_store.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        if (!t.assert_true(string_format("seq copy data length %zu matches expected length %zu", ncopy, seq_store.size()), ncopy == seq_store.size())) {
            return;
        }
        LOG_TRC("%s: seq 0 copied, %zd bytes\n", __func__, ncopy);

        llama_memory_clear(llama_get_memory(ctx.get()), true);
        LOG_TRC("%s: kv cache cleared\n", __func__);

        const size_t nset = llama_state_seq_set_data_ext(ctx.get(), seq_store.data(), seq_store.size(), 1, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        if (!t.assert_true(string_format("seq set data length %zu matches expected length %zu", nset, seq_store.size()), nset == seq_store.size())) {
            return;
        }
        LOG_TRC("%s: seq 1 restored, %zd bytes\n", __func__, nset);
    }

    // Generate tokens on seq 1
    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 1);
    if (!t.assert_true(string_format("generated %d tokens on seq 1", params.n_predict), !result.empty())) {
        return;
    }

    t.assert_true("generation on seq 1 matches the baseline", result == expected_result);
}


// Test 6/7: seq copy (scatter)
// - decode the same prefix on two sequences, interleaving seq 0 cells between the seq 1 cells
// - save the seq 1 state, free the interleaved seq 0 cells, and restore via the given io path
// - the restore destination is non-contiguous: scatter reads are batched per contiguous run
// - save again on the host and compare the two blobs byte for byte
static void test_seq_cp_scatter(testing & t, struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, int test_num, bool on_device) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_ctx      = 256;
    params_ctx.n_seq_max  = 2;
    params_ctx.kv_unified = true;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    const char * path = on_device ? "device" : "host";

    LOG("\n=== Test %d: seq copy (%s, scatter) ===\n", test_num, path);

    const uint32_t flags = on_device ? LLAMA_STATE_SEQ_FLAGS_ON_DEVICE : LLAMA_STATE_SEQ_FLAGS_NONE;

    auto decode_one = [&](llama_token tok, int pos, llama_seq_id seq) {
        llama_batch_ptr batch(1, 0, 1);
        common_batch_add(batch.get(), tok, pos, { seq }, true);
        return llama_decode(ctx.get(), batch.get()) == 0;
    };

    // seq 0 cells 0,1,4 interleave the seq 1 cells 2,3,5
    if (!t.assert_true("interleaved state is built",
                decode_one(tokens[0], 0, 0) &&
                decode_one(tokens[1], 1, 0) &&
                decode_one(tokens[0], 0, 1) &&
                decode_one(tokens[1], 1, 1) &&
                decode_one(tokens[2], 2, 0) &&
                decode_one(tokens[2], 2, 1))) {
        return;
    }

    const auto get_seq_state = [&](llama_seq_id seq_id, uint32_t fl, std::vector<uint8_t> & state) {
        const size_t state_size = llama_state_seq_get_size_ext(ctx.get(), seq_id, fl);
        if (!t.assert_true(string_format("sequence %d state is not empty", seq_id), state_size != 0)) {
            return false;
        }

        state.resize(state_size);
        const size_t ncopy = llama_state_seq_get_data_ext(ctx.get(), state.data(), state.size(), seq_id, fl);
        return t.assert_true(string_format("sequence %d state length %zu matches expected length %zu", seq_id, ncopy, state.size()),
                ncopy == state.size());
    };

    // host blob: contains the KV data, used for the byte-for-byte comparison
    std::vector<uint8_t> state_before;
    if (!get_seq_state(1, LLAMA_STATE_SEQ_FLAGS_NONE, state_before)) {
        return;
    }

    // save via the io path under test
    std::vector<uint8_t> state_save;
    if (!get_seq_state(1, flags, state_save)) {
        return;
    }
    LOG_TRC("%s: seq 1 saved via %s, %zu bytes\n", __func__, path, state_save.size());

    // free seq 0's cells so the ring is fragmented: the restore destination (seq 1's interleaved cells) stays non-contiguous
    if (!t.assert_true("sequence 0 is removed", llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, -1, -1))) {
        return;
    }

    // restore via the io path under test
    const size_t nset = llama_state_seq_set_data_ext(ctx.get(), state_save.data(), state_save.size(), 1, flags);
    if (!t.assert_true(string_format("seq set data length %zu matches expected length %zu", nset, state_save.size()), nset == state_save.size())) {
        return;
    }
    LOG_TRC("%s: seq 1 restored via %s, %zu bytes\n", __func__, path, nset);

    std::vector<uint8_t> state_after;
    if (!get_seq_state(1, LLAMA_STATE_SEQ_FLAGS_NONE, state_after)) {
        return;
    }

    // the blob is serialized in sequence cell order, so identical bytes iff the restore wrote the same KV
    t.assert_true(string_format("restored KV state via %s is byte-identical to the saved state", path),
            state_before.size() == state_after.size() && memcmp(state_before.data(), state_after.data(), state_before.size()) == 0);
}


// Test 8: state blob round-trip
// compares blobs rather than generated text: a partially restored cell still decodes to plausible tokens
static void test_state_roundtrip(testing & t, struct llama_model * model, const struct common_params & params, const llama_tokens & tokens) {
    auto params_ctx = common_context_params_to_llama(params);
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    LOG("\n=== Test 8: state blob round-trip ===\n");

    if (!t.assert_true("prompt decodes",
                llama_decode(ctx.get(), llama_batch_get_one(const_cast<llama_token *>(tokens.data()), (int32_t) tokens.size())) == 0)) {
        return;
    }

    std::vector<uint8_t> blob_a(llama_state_seq_get_size(ctx.get(), 0));
    const size_t n_a = llama_state_seq_get_data(ctx.get(), blob_a.data(), blob_a.size(), 0);
    if (!t.assert_true(string_format("saved %zu bytes, expected %zu", n_a, blob_a.size()), n_a == blob_a.size())) {
        return;
    }

    if (!t.assert_true("seq 0 is erased", llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, -1, -1))) {
        return;
    }

    if (!t.assert_true("seq 0 is restored", llama_state_seq_set_data(ctx.get(), blob_a.data(), blob_a.size(), 0) == blob_a.size())) {
        return;
    }

    std::vector<uint8_t> blob_b(llama_state_seq_get_size(ctx.get(), 0));
    const size_t n_b = llama_state_seq_get_data(ctx.get(), blob_b.data(), blob_b.size(), 0);
    if (!t.assert_true(string_format("re-saved %zu bytes, expected %zu", n_b, n_a), n_b == n_a)) {
        return;
    }

    size_t n_diff = 0;
    size_t i_diff = 0;
    for (size_t i = 0; i < n_a; i++) {
        if (blob_a[i] != blob_b[i]) {
            if (n_diff == 0) {
                i_diff = i;
            }
            n_diff++;
        }
    }

    t.assert_true(string_format("state is unchanged across a restore (%zu of %zu bytes differ, first at offset %zu)", n_diff, n_a, i_diff), n_diff == 0);
}


// Run the full save/load test suite (tests 1-8) for a single model.
static void run_save_load_tests_for_model(testing & t, const std::string & model_path, const struct common_params & base_params) {
    struct common_params params = base_params;
    params.model.path = model_path;

    auto llama_init = common_init_from_params(params, true);
    auto * model = llama_init->model();

    if (!t.assert_true("model loads: " + model_path, model != nullptr)) {
        return;
    }

    GGML_ASSERT(llama_init->context() == nullptr);

    // Tokenize prompt or generate random tokens
    llama_tokens tokens;
    if (params.prompt.empty()) {
        const int n_prompt = params.n_batch;

        // this path is useful for model files that do not have a tokenizer
        LOG_INF("%s: no prompt provided, generating %d (n_batch) random tokens\n", __func__, n_prompt);

        const auto * vocab = llama_model_get_vocab(model);
        const auto n_vocab = llama_vocab_n_tokens(vocab);

        std::mt19937 rng(params.sampling.seed);
        std::uniform_int_distribution<llama_token> dist(0, n_vocab - 1);
        for (int i = 0; i < n_prompt; i++) {
            tokens.push_back(dist(rng));
        }
    } else {
        LOG_INF("%s: tokenizing prompt '%s'\n", __func__, params.prompt.c_str());

        auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
        tokens = common_tokenize(ctx.get(), params.prompt, true);
    }

    LOG_INF("%s: the input prompt is %d tokens\n", __func__, (int)tokens.size());

    // tests 3-5 replay the baseline generation from the state file it saves
    llama_tokens result_baseline;
    t.test("baseline", [&](testing & t) {
        result_baseline = test_baseline(t, model, params, tokens);
    });

    t.test("seq_rm_isolated", [&](testing & t) {
        test_seq_rm_isolated(t, model, params, tokens);
    });

    const auto needs_baseline = [&](testing & t) {
        if (result_baseline.empty()) {
            t.skip("baseline did not generate tokens");
            return false;
        }
        return true;
    };

    t.test("state_load", [&](testing & t) {
        if (needs_baseline(t)) {
            test_state_load(t, model, params, tokens, result_baseline);
        }
    });

    t.test("seq_cp_host", [&](testing & t) {
        if (needs_baseline(t)) {
            test_seq_cp_host(t, model, params, tokens, result_baseline);
        }
    });

    t.test("seq_cp_device", [&](testing & t) {
        if (needs_baseline(t)) {
            test_seq_cp_device(t, model, params, tokens, result_baseline);
        }
    });

    t.test("seq_cp_host_scatter", [&](testing & t) {
        test_seq_cp_scatter(t, model, params, tokens, 6, false);
    });

    t.test("seq_cp_device_scatter", [&](testing & t) {
        test_seq_cp_scatter(t, model, params, tokens, 7, true);
    });

    t.test("state_roundtrip", [&](testing & t) {
        test_state_roundtrip(t, model, params, tokens);
    });
}


int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.prompt = "";
    params.n_batch = 100;
    params.out_file = "dump_state.bin";
    params.sampling.seed = 1234;

    common_init();

    // extract our own --models DIR and --filter RE options before handing the rest to the common arg parser
    std::string models_dir;
    std::string filter;
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--models") == 0) {
            if (i + 1 >= argc) {
                LOG_ERR("%s: --models requires a directory argument\n", __func__);
                return 1;
            }
            models_dir = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--filter") == 0) {
            if (i + 1 >= argc) {
                LOG_ERR("%s: --filter requires a regex argument\n", __func__);
                return 1;
            }
            filter = argv[i + 1];
            i++;
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }
    filtered_argv.push_back(nullptr);
    const int fargc = (int)filtered_argv.size() - 1;

    // in --models mode there is no single model; set a placeholder so the common parser's
    // "--model is required" check passes (each model is set individually inside the loop)
    if (!models_dir.empty()) {
        params.model.path = models_dir;
    }

    if (!common_params_parse(fargc, filtered_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    if (params.n_parallel == 1) {
        LOG_TRC("%s: n_parallel == 1, enabling unified kv cache\n", __func__);
        params.kv_unified = true;
    }

    if (params.n_predict < 0) {
        params.n_predict = 16;
    }

    ggml_backend_load_all();

    testing t;
    t.capture_output = true;
    t.apply_env();
    if (!filter.empty()) {
        t.set_filter(filter);
    }

    // one node per model, named after the file
    const auto run_model = [&](const std::string & model_path) {
        t.test(std::filesystem::path(model_path).stem().string(), [&](testing & t) {
            LOG_INF("model %s\n", model_path.c_str());
            run_save_load_tests_for_model(t, model_path, params);
        });
    };

    if (!models_dir.empty()) {
        // run the suite over every dummy model in the directory
        if (!std::filesystem::exists(models_dir) || !std::filesystem::is_directory(models_dir)) {
            LOG_ERR("%s: models directory '%s' does not exist\n", __func__, models_dir.c_str());
            return 1;
        }

        std::vector<std::string> models;
        for (const auto & entry : std::filesystem::directory_iterator(models_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
                models.push_back(entry.path().string());
            }
        }
        std::sort(models.begin(), models.end());

        if (models.empty()) {
            LOG_ERR("%s: no .gguf models found in '%s'\n", __func__, models_dir.c_str());
            return 1;
        }

        for (const auto & model_path : models) {
            run_model(model_path);
        }

        return t.summary();
    }

    // single-model mode
    run_model(params.model.path);

    return t.summary();
}
