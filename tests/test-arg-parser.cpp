#include "arg.h"
#include "common.h"
#include "download.h"
#include "llama.h"
#include "speculative.h"
#include "testing.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <sstream>
#include <unordered_set>

static std::string join_args(const std::vector<std::string> & argv) {
    std::string res;
    for (const auto & arg : argv) {
        res += res.empty() ? arg : " " + arg;
    }
    return res;
}

static std::vector<char *> list_str_to_char(std::vector<std::string> & argv) {
    std::vector<char *> res;
    for (auto & arg : argv) {
        res.push_back(const_cast<char *>(arg.data()));
    }
    return res;
}

static bool parse(std::vector<std::string> argv, common_params & params, enum llama_example ex) {
    return common_params_parse(argv.size(), list_str_to_char(argv).data(), params, ex);
}

static void assert_rejected(testing & t, std::vector<std::string> argv, common_params & params, enum llama_example ex = LLAMA_EXAMPLE_COMMON) {
    t.assert_true(join_args(argv) + " is rejected", false == parse(std::move(argv), params, ex));
}

static bool assert_accepted(testing & t, std::vector<std::string> argv, common_params & params, enum llama_example ex = LLAMA_EXAMPLE_COMMON) {
    return t.assert_true(join_args(argv) + " is accepted", true == parse(std::move(argv), params, ex));
}

static void test(testing & t) {
    common_params params;

    t.test("speculative output limits", [](testing & t) {
        auto assert_output_limits = [&](int32_t n_batch, int32_t n_parallel, int32_t n_draft,
                                        int32_t total, int32_t per_seq) {
            const auto limits = common_speculative_get_output_limits(n_batch, n_parallel, n_draft);
            const auto what = "n_batch=" + std::to_string(n_batch) + " n_parallel=" + std::to_string(n_parallel) + " n_draft=" + std::to_string(n_draft);
            t.assert_equal(what + " total", total, limits.total);
            t.assert_equal(what + " per_seq", per_seq, limits.per_seq);
        };

        assert_output_limits(16, 2,  3, 8, 4);
        assert_output_limits(16, 2, -1, 2, 1);
        assert_output_limits( 6, 2,  3, 6, 4);
        assert_output_limits( 2, 1,  3, 2, 2);
        assert_output_limits(
                std::numeric_limits<int32_t>::max(),
                std::numeric_limits<int32_t>::max(),
                std::numeric_limits<int32_t>::max(),
                std::numeric_limits<int32_t>::max(),
                std::numeric_limits<int32_t>::max());
    });

    t.test("synth rates", [](testing & t) {
        common_params_speculative spec;
        spec.synth_len = 3.4;

        auto assert_invalid = [&](const std::string & what, const common_params_speculative & value, int32_t n_max) {
            bool threw = false;
            try {
                common_speculative_synth_rates_resolve(&value, n_max);
            } catch (const std::invalid_argument &) {
                threw = true;
            }
            t.assert_true(what + " throws invalid_argument", threw);
        };

        auto assert_rate = [&](const std::vector<double> & rates, size_t i, double expected) {
            t.assert_true("rates[" + std::to_string(i) + "] is " + std::to_string(expected) + ", got " + std::to_string(rates[i]),
                          std::abs(rates[i] - expected) < 1e-5);
        };

        const auto rates = common_speculative_synth_rates_resolve(&spec, 4);
        if (t.assert_equal("synth_len 3.4 rates count", (size_t) 4, rates.size())) {
            assert_rate(rates, 0, 0.80581);
            assert_rate(rates, 1, 0.64933);
            assert_rate(rates, 2, 0.52323);
            assert_rate(rates, 3, 0.42163);
            t.assert_true("synth_len 3.4 rates sum to the expected length", std::abs(1.0 + rates[0] + rates[1] + rates[2] + rates[3] - 3.4) < 1e-8);
        }

        spec.synth_len = 1.0;
        t.assert_true("synth_len 1.0 resolves to all zeros", common_speculative_synth_rates_resolve(&spec, 4) == std::vector<double>({0.0, 0.0, 0.0, 0.0}));

        spec.synth_len = 5.0;
        t.assert_true("synth_len 5.0 resolves to all ones", common_speculative_synth_rates_resolve(&spec, 4) == std::vector<double>({1.0, 1.0, 1.0, 1.0}));

        spec.synth_len = 5.1;
        assert_invalid("synth_len 5.1", spec, 4);

        spec.synth_len = std::numeric_limits<double>::quiet_NaN();
        assert_invalid("synth_len nan", spec, 4);

        spec.synth_len = 0.0;
        assert_invalid("synth_len 0.0", spec, 4);

        spec.synth_len = -1.0;
        spec.synth_rates = {0.8, 0.6, 0.4};
        assert_invalid("3 synth_rates for n_max 4", spec, 4);

        spec.synth_rates = {0.8, 0.6, 0.4, 0.2};
        t.assert_true("explicit synth_rates are returned as given", common_speculative_synth_rates_resolve(&spec, 4) == spec.synth_rates);

        spec.synth_rates = {0.8, 0.9, 0.4, 0.2};
        assert_invalid("increasing synth_rates", spec, 4);

        spec.synth_rates = {0.8, std::numeric_limits<double>::quiet_NaN(), 0.4, 0.2};
        assert_invalid("nan synth_rate", spec, 4);

        spec.synth_rates = {0.8, 0.6, 0.4, -0.2};
        assert_invalid("negative synth_rate", spec, 4);

        spec.synth_rates = {0.8, 0.6, 0.4, 0.2};
        spec.synth_len = 3.0;
        assert_invalid("synth_rates together with synth_len", spec, 4);
    });

    t.test("base params to speculative", [](testing & t) {
        common_params base;
        base.n_parallel = 4;
        base.n_outputs_max_per_seq = 8;

        const auto draft = common_base_params_to_speculative(base);
        t.assert_equal("n_outputs_max", 4, draft.n_outputs_max);
        t.assert_equal("n_outputs_max_per_seq", 1, draft.n_outputs_max_per_seq);
    });

    // make sure there is no duplicated arguments in any examples
    t.test("no duplicated arguments", [&](testing & t) {
        for (int ex = 0; ex < LLAMA_EXAMPLE_COUNT; ex++) {
            t.test("example " + std::to_string(ex), [&](testing & t) {
                auto ctx_arg = common_params_parser_init(params, (enum llama_example)ex);
                common_params_add_preset_options(ctx_arg.options);
                std::unordered_set<std::string> seen_args;
                std::unordered_set<std::string> seen_env_vars;
                for (const auto & opt : ctx_arg.options) {
                    // check for args duplications
                    for (const auto & arg : opt.get_args()) {
                        t.assert_true("found different handlers for the same argument: " + arg, seen_args.insert(arg).second);
                    }
                    // check for env var duplications
                    for (const auto & env : opt.get_env()) {
                        t.assert_true("found different handlers for the same env var: " + env, seen_env_vars.insert(env).second);
                    }

                    // exclude spec args from this check
                    // ref: https://github.com/ggml-org/llama.cpp/pull/22397
                    const bool skip = opt.is_spec;

                    // ensure shorter argument precedes longer argument
                    if (!skip && opt.args.size() > 1) {
                        const std::string first(opt.args.front());
                        const std::string last(opt.args.back());

                        t.assert_true("shorter argument should come before longer one: " + first + ", " + last, first.length() <= last.length());
                    }

                    // same check for negated arguments
                    if (opt.args_neg.size() > 1) {
                        const std::string first(opt.args_neg.front());
                        const std::string last(opt.args_neg.back());

                        t.assert_true("shorter negated argument should come before longer one: " + first + ", " + last, first.length() <= last.length());
                    }
                }
            });
        }
    });

    t.test("invalid usage", [&](testing & t) {
        // missing value
        assert_rejected(t, {"binary_name", "-m"}, params);

        // wrong value (int)
        assert_rejected(t, {"binary_name", "-ngl", "hello"}, params);

        // wrong value (enum)
        assert_rejected(t, {"binary_name", "-sm", "hello"}, params);

        {
            common_params penalty_params;
            t.assert_equal("default penalty_last_n", 64, penalty_params.sampling.penalty_last_n);
            t.assert_equal("default dry_penalty_last_n", 64, penalty_params.sampling.dry_penalty_last_n);

            assert_rejected(t, {"binary_name", "--repeat-last-n", "-1"}, penalty_params);
            assert_rejected(t, {"binary_name", "--dry-penalty-last-n", "-1"}, penalty_params);
            assert_rejected(t, {"binary_name", "--repeat-penalty", "0"}, penalty_params);
            assert_rejected(t, {"binary_name", "--repeat-penalty", "-1"}, penalty_params);
            assert_rejected(t, {"binary_name", "--repeat-penalty", "nan"}, penalty_params);
            assert_rejected(t, {"binary_name", "--repeat-penalty", "inf"}, penalty_params);
            assert_rejected(t, {"binary_name", "--repeat-penalty", "-inf"}, penalty_params);

            const char * penalty_options[] = {"--frequency-penalty", "--presence-penalty"};
            const char * nonfinite_values[] = {"nan", "inf", "-inf"};
            for (const char * option : penalty_options) {
                for (const char * value : nonfinite_values) {
                    assert_rejected(t, {"binary_name", option, value}, penalty_params);
                }
            }
        }

        // non-existence arg in specific example (--draft cannot be used outside llama-speculative)
        assert_rejected(t, {"binary_name", "--draft", "123"}, params, LLAMA_EXAMPLE_EMBEDDING);

        assert_rejected(t, {"binary_name", "-lm", "hello"}, params);
    });

    t.test("valid usage", [&](testing & t) {
        if (assert_accepted(t, {"binary_name", "-m", "model_file.gguf"}, params)) {
            t.assert_equal("model.path", "model_file.gguf", params.model.path);
        }

        if (assert_accepted(t, {"binary_name", "-t", "1234"}, params)) {
            t.assert_equal("n_threads", 1234, params.cpuparams.n_threads);
        }

        if (assert_accepted(t, {"binary_name", "--verbose"}, params)) {
            t.assert_true("verbosity > 1, got " + std::to_string(params.verbosity), params.verbosity > 1);
        }

        if (assert_accepted(t, {"binary_name", "-m", "abc.gguf", "--predict", "6789", "--batch-size", "9090"}, params)) {
            t.assert_equal("model.path", "abc.gguf", params.model.path);
            t.assert_equal("n_predict", 6789, params.n_predict);
            t.assert_equal("n_batch", 9090, params.n_batch);
        }

        // --draft cannot be used outside llama-speculative
        if (assert_accepted(t, {"binary_name", "--spec-draft-n-max", "123"}, params, LLAMA_EXAMPLE_SPECULATIVE)) {
            t.assert_equal("speculative.draft.n_max", 123, params.speculative.draft.n_max);
        }

        {
            common_params synth_params;
            if (assert_accepted(t, {"binary_name", "--spec-synth-len", "3.4"}, synth_params, LLAMA_EXAMPLE_SERVER)) {
                t.assert_equal("speculative.synth_len", 3.4, synth_params.speculative.synth_len);
            }
        }

        {
            common_params synth_params;
            if (assert_accepted(t, {"binary_name", "--spec-synth-rates", "0.8,0.6,0.2"}, synth_params, LLAMA_EXAMPLE_SERVER)) {
                t.assert_true("speculative.synth_rates are 0.8,0.6,0.2", synth_params.speculative.synth_rates == std::vector<double>({0.8, 0.6, 0.2}));
            }
        }

        {
            common_params synth_params;
            assert_rejected(t, {"binary_name", "--spec-synth-len", "3.4x"}, synth_params, LLAMA_EXAMPLE_SERVER);
        }

        if (assert_accepted(t, {"binary_name", "-lm", "none"}, params)) {
            t.assert_equal("load_mode none", LLAMA_LOAD_MODE_NONE, params.load_mode);
        }

        if (assert_accepted(t, {"binary_name", "-lm", "mmap"}, params)) {
            t.assert_equal("load_mode mmap", LLAMA_LOAD_MODE_MMAP, params.load_mode);
        }

        if (assert_accepted(t, {"binary_name", "-lm", "mlock"}, params)) {
            t.assert_equal("load_mode mlock", LLAMA_LOAD_MODE_MLOCK, params.load_mode);
        }

        if (assert_accepted(t, {"binary_name", "-lm", "mmap+mlock"}, params)) {
            t.assert_equal("load_mode mmap+mlock", LLAMA_LOAD_MODE_MMAP_MLOCK, params.load_mode);
        }

        if (assert_accepted(t, {"binary_name", "-lm", "dio"}, params)) {
            t.assert_equal("load_mode dio", LLAMA_LOAD_MODE_DIRECT_IO, params.load_mode);
        }

        // multi-value args (CSV)
        if (assert_accepted(t, {"binary_name", "--lora", "file1.gguf,\"file2,2.gguf\",\"file3\"\"3\"\".gguf\",file4\".gguf"}, params)) {
            if (t.assert_equal("lora_adapters.size()", (size_t) 4, params.lora_adapters.size())) {
                t.assert_equal("lora_adapters[0].path", "file1.gguf", params.lora_adapters[0].path);
                t.assert_equal("lora_adapters[1].path", "file2,2.gguf", params.lora_adapters[1].path);
                t.assert_equal("lora_adapters[2].path", "file3\"3\".gguf", params.lora_adapters[2].path);
                t.assert_equal("lora_adapters[3].path", "file4\".gguf", params.lora_adapters[3].path);
            }
        }
    });

    t.test("environment variables", [&](testing & t) {
// skip this part on windows, because setenv is not supported
#ifdef _WIN32
        t.skip("setenv is not supported on windows");
#else
        setenv("LLAMA_ARG_THREADS", "blah", true);
        assert_rejected(t, {"binary_name"}, params);

        setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
        setenv("LLAMA_ARG_THREADS", "1010", true);
        if (assert_accepted(t, {"binary_name"}, params)) {
            t.assert_equal("model.path from LLAMA_ARG_MODEL", "blah.gguf", params.model.path);
            t.assert_equal("n_threads from LLAMA_ARG_THREADS", 1010, params.cpuparams.n_threads);
        }

        setenv("LLAMA_ARG_LOAD_MODE", "blah", true);
        assert_rejected(t, {"binary_name"}, params);

        setenv("LLAMA_ARG_LOAD_MODE", "mmap", true);
        if (assert_accepted(t, {"binary_name"}, params)) {
            t.assert_equal("load_mode mmap", LLAMA_LOAD_MODE_MMAP, params.load_mode);
        }

        setenv("LLAMA_ARG_LOAD_MODE", "mlock", true);
        if (assert_accepted(t, {"binary_name"}, params)) {
            t.assert_equal("load_mode mlock", LLAMA_LOAD_MODE_MLOCK, params.load_mode);
        }

        setenv("LLAMA_ARG_LOAD_MODE", "mmap+mlock", true);
        if (assert_accepted(t, {"binary_name"}, params)) {
            t.assert_equal("load_mode mmap+mlock", LLAMA_LOAD_MODE_MMAP_MLOCK, params.load_mode);
        }

        setenv("LLAMA_ARG_LOAD_MODE", "dio", true);
        if (assert_accepted(t, {"binary_name"}, params)) {
            t.assert_equal("load_mode dio", LLAMA_LOAD_MODE_DIRECT_IO, params.load_mode);
        }

        t.test("negated", [&](testing & t) {
            setenv("LLAMA_ARG_LOAD_MODE", "none", true);
            setenv("LLAMA_ARG_NO_PERF", "1", true); // legacy format
            if (assert_accepted(t, {"binary_name"}, params)) {
                t.assert_equal("load_mode none", LLAMA_LOAD_MODE_NONE, params.load_mode);
                t.assert_equal("no_perf", true, params.no_perf);
            }
        });

        t.test("overwritten", [&](testing & t) {
            setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
            setenv("LLAMA_ARG_THREADS", "1010", true);
            if (assert_accepted(t, {"binary_name", "-m", "overwritten.gguf"}, params)) {
                t.assert_equal("model.path", "overwritten.gguf", params.model.path);
                t.assert_equal("n_threads", 1010, params.cpuparams.n_threads);
            }
        });
#endif // _WIN32
    });

    t.test("download functions", [](testing & t) {
        const char * GOOD_URL = "http://ggml.ai/";
        const char * BAD_URL  = "http://ggml.ai/404";

        t.test("good URL", [&](testing & t) {
            auto res = common_remote_get_content(GOOD_URL, {});
            t.assert_equal("status", 200L, res.first);
            t.assert_true("body is not empty", res.second.size() > 0);
            std::string str(res.second.data(), res.second.size());
            t.assert_true("body mentions llama.cpp", str.find("llama.cpp") != std::string::npos);
        });

        t.test("bad URL", [&](testing & t) {
            auto res = common_remote_get_content(BAD_URL, {});
            t.assert_equal("status", 404L, res.first);
        });

        t.test("max size error", [&](testing & t) {
            common_remote_params params;
            params.max_size = 1;
            bool threw = false;
            try {
                common_remote_get_content(GOOD_URL, params);
            } catch (std::exception & e) {
                threw = true;
                printf("  expected error: %s\n\n", e.what());
            }
            t.assert_true("it should throw an error", threw);
        });
    });
}

int main(int argc, char ** argv) {
    testing t;
    t.capture_output = true;
    t.apply_env();
    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    test(t);

    return t.summary();
}
