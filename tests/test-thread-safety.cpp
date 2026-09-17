// thread safety test
// - Loads a copy of the same model on each GPU, plus a copy on the CPU
// - Creates n_parallel (--parallel) contexts per model
// - Runs inference in parallel on each context

#include "testing.h"

#include <array>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include "llama.h"
#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"

// what one context thread produced, checked after all threads are joined
struct context_result {
    std::string error;
    std::string text;
};

static std::string model_name(int m, int gpu_dev_count) {
    if (m < gpu_dev_count) {
        return "gpu_" + std::to_string(m);
    }
    return m == gpu_dev_count ? "cpu" : "layer_split";
}

int main(int argc, char ** argv) {
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    testing t;
    t.capture_output = true;
    t.apply_env();

    llama_backend_init();
    llama_numa_init(params.numa);

    //llama_log_set([](ggml_log_level level, const char * text, void * /*user_data*/) {
    //    if (level == GGML_LOG_LEVEL_ERROR) {
    //        common_log_add(common_log_main(), level, "%s", text);
    //    }
    //}, NULL);

    auto cparams = common_context_params_to_llama(params);

    // each context has a single sequence
    cparams.n_seq_max = 1;

    int dev_count = ggml_backend_dev_count();
    std::vector<std::array<ggml_backend_dev_t, 2>> gpus;
    for (int i = 0; i < dev_count; ++i) {
        auto * dev = ggml_backend_dev_get(i);
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpus.push_back({dev, nullptr});
        }
    }
    const int gpu_dev_count = (int)gpus.size();
    const int num_models = gpu_dev_count + 1 + 1; // GPUs + 1 CPU model + 1 layer split
    //const int num_models = std::max(1, gpu_dev_count);
    const int num_contexts = std::max(1, params.n_parallel);

    std::vector<llama_model_ptr> models;

    t.test("load", [&](testing & t) {
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());

        for (int m = 0; m < num_models; ++m) {
            t.test(model_name(m, gpu_dev_count), [&](testing & t) {
                auto mparams = common_model_params_to_llama(params);

                if (m < gpu_dev_count) {
                    mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
                    mparams.devices = gpus[m].data();
                } else if (m == gpu_dev_count) {
                    mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
                    mparams.main_gpu = -1; // CPU model
                } else {
                    mparams.split_mode = LLAMA_SPLIT_MODE_LAYER;
                }

                llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
                if (!t.assert_true("load model " + std::to_string(m + 1) + "/" + std::to_string(num_models) + " from '" + params.model.path + "'", model != NULL)) {
                    LOG_ERR("%s: failed to load model '%s'\n", __func__, params.model.path.c_str());
                    return;
                }

                models.emplace_back(model);
            });

            if ((int) models.size() != m + 1) {
                break;
            }
        }
    });

    if ((int) models.size() != num_models) {
        return t.summary();
    }

    t.test("generate", [&](testing & t) {
        std::vector<std::thread> threads;
        std::atomic<bool> failed = false;

        std::vector<context_result> results(num_models*num_contexts);

        for  (int m = 0; m < num_models; ++m) {
            auto * model = models[m].get();
            for (int c = 0; c < num_contexts; ++c) {
                threads.emplace_back([&, m, c, model]() {
                    context_result & res = results[m*num_contexts + c];

                    LOG_INF("Creating context %d/%d for model %d/%d\n", c + 1, num_contexts, m + 1, num_models);

                    llama_context_ptr ctx { llama_init_from_model(model, cparams) };
                    if (ctx == NULL) {
                        LOG_ERR("failed to create context\n");
                        res.error = "failed to create context";
                        failed.store(true);
                        return;
                    }

                    std::unique_ptr<common_sampler, decltype(&common_sampler_free)> sampler { common_sampler_init(model, params.sampling), common_sampler_free };
                    if (sampler == NULL) {
                        LOG_ERR("failed to create sampler\n");
                        res.error = "failed to create sampler";
                        failed.store(true);
                        return;
                    }

                    llama_batch batch = {};
                    {
                        auto prompt = common_tokenize(ctx.get(), params.prompt, true);
                        if (prompt.empty()) {
                            LOG_ERR("failed to tokenize prompt\n");
                            res.error = "failed to tokenize prompt";
                            failed.store(true);
                            return;
                        }
                        batch = llama_batch_get_one(prompt.data(), prompt.size());
                        if (llama_decode(ctx.get(), batch)) {
                            LOG_ERR("failed to decode prompt\n");
                            res.error = "failed to decode prompt";
                            failed.store(true);
                            return;
                        }
                    }

                    const auto * vocab = llama_model_get_vocab(model);
                    std::string result = params.prompt;

                    for (int i = 0; i < params.n_predict; i++) {
                        llama_token token;
                        if (batch.n_tokens > 0) {
                            token = common_sampler_sample(sampler.get(), ctx.get(), batch.n_tokens - 1);
                        } else {
                            token = llama_vocab_bos(vocab);
                        }

                        result += common_token_to_piece(ctx.get(), token);

                        if (llama_vocab_is_eog(vocab, token)) {
                            break;
                        }

                        batch = llama_batch_get_one(&token, 1);

                        int ret = llama_decode(ctx.get(), batch);
                        if (ret == 1 && i > 0) {
                            LOG_INF("Context full, stopping generation.\n");
                            break;
                        }

                        if (ret != 0) {
                            LOG_ERR("Model %d/%d, Context %d/%d: failed to decode\n", m + 1, num_models, c + 1, num_contexts);
                            res.error = "failed to decode token " + std::to_string(token) + " at step " + std::to_string(i) + " (ret " + std::to_string(ret) + ")";
                            failed.store(true);
                            return;
                        }
                    }

                    res.text = result;

                    llama_synchronize(ctx.get());
                });
            }
        }

        for (auto & thread : threads) {
            thread.join();
        }

        if (failed) {
            LOG_ERR("One or more threads failed.\n");
        } else {
            LOG_INF("All threads finished without errors.\n");
        }

        for (int m = 0; m < num_models; ++m) {
            t.test(model_name(m, gpu_dev_count), [&](testing & t) {
                for (int c = 0; c < num_contexts; ++c) {
                    t.test("context_" + std::to_string(c + 1), [&](testing & t) {
                        const context_result & res = results[m*num_contexts + c];

                        if (res.error.empty()) {
                            LOG_INF("Model %d/%d, Context %d/%d: %s\n\n", m + 1, num_models, c + 1, num_contexts, res.text.c_str());
                        }

                        t.assert_true(res.error.empty() ? "generation finished without errors" : res.error, res.error.empty());
                    });
                }
            });
        }
    });

    return t.summary();
}
