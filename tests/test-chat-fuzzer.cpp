//  Fuzz test for grammar-constrained decoding of chat templates.
//
//  Loads a model, creates a context, and generates random token sequences
//  constrained by common_sampler (with grammar). Each step overwrites the
//  context's logits buffer with random values, then calls
//  common_sampler_sample() — the exact same code path used in production.
//
//  The model's embedded chat template is used by default. Use
//  --chat-template to override with a Jinja file.
//
//  Usage:
//    ./test-chat-fuzzer --model <model.gguf> [options]
//
#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "chat.h"
#include "common.h"
#include "log.h"
#include "sampling.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <regex>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string read_file(const std::string & path) {
    std::ifstream fs(path, std::ios_base::binary);
    if (!fs.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    fs.seekg(0, std::ios_base::end);
    auto size = fs.tellg();
    fs.seekg(0);
    std::string out;
    out.resize(static_cast<size_t>(size));
    fs.read(out.data(), static_cast<std::streamsize>(size));
    return out;
}

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

static common_chat_tool special_function_tool{
    "special_function", "I'm special",
    R"({"type":"object","properties":{"arg1":{"type":"integer","description":"The arg."}},"required":["arg1"]})",
};

static common_chat_tool python_tool{
    "python", "an ipython interpreter",
    R"({"type":"object","properties":{"code":{"type":"string","description":"Python code to execute."}},"required":["code"]})",
};

static std::vector<common_chat_tool> fuzz_tools{ special_function_tool, python_tool };

static common_chat_msg message_user{
    "user", "Hey there!", {}, {}, "", "", "",
};

// ---------------------------------------------------------------------------
// Build a common_sampler from chat params
// ---------------------------------------------------------------------------

static common_sampler * create_sampler(
        const llama_model        * model,
        const common_chat_params & chat_params,
        uint32_t                   seed) {

    const llama_vocab * vocab = llama_model_get_vocab(model);

    common_params_sampling sparams;
    sparams.seed         = seed;
    sparams.grammar      = chat_params.grammar;
    sparams.grammar_lazy = chat_params.grammar_lazy;
    sparams.grammar_triggers = chat_params.grammar_triggers;

    for (const auto & t : chat_params.preserved_tokens) {
        auto ids = common_tokenize(vocab, t, false, true);
        if (ids.size() == 1) {
            sparams.preserved_tokens.insert(ids[0]);
        }
    }

    sparams.temp = 1.0f;

    return common_sampler_init(model, sparams);
}

// ---------------------------------------------------------------------------
// Prime the context so llama_get_logits_ith works.
// ---------------------------------------------------------------------------

static bool prime_context(llama_context * ctx, const llama_vocab * vocab) {
    llama_token bos = llama_vocab_bos(vocab);
    if (bos == LLAMA_TOKEN_NULL) {
        bos = 0;
    }

    llama_batch batch = llama_batch_get_one(&bos, 1);
    if (llama_decode(ctx, batch) != 0) {
        return false;
    }

    return llama_get_logits_ith(ctx, -1) != nullptr;
}

// ---------------------------------------------------------------------------
// Core: generate one random grammar-constrained sequence
// ---------------------------------------------------------------------------

struct fuzz_generation {
    std::vector<llama_token> tokens;
    std::string              text;
};

static fuzz_generation generate_fuzz_sequence(
        llama_context     * ctx,
        const llama_vocab * vocab,
        common_sampler    * smpl,
        std::mt19937      & rng,
        int                 max_tokens) {

    common_sampler_reset(smpl);

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const llama_token tok_eos = llama_vocab_eos(vocab);
    const llama_token tok_eot = llama_vocab_eot(vocab);

    std::vector<llama_token> tokens;
    std::uniform_real_distribution<float> logit_dist(-1.0f, 1.0f);

    for (int step = 0; step < max_tokens; step++) {
        // Overwrite the context's logits buffer with random values
        float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) {
            break;
        }
        for (int32_t i = 0; i < n_vocab; i++) {
            logits[i] = logit_dist(rng);
        }

        // Sample using the production code path
        llama_token tok = common_sampler_sample(smpl, ctx, -1);

        if (tok == LLAMA_TOKEN_NULL) {
            break;
        }

        common_sampler_accept(smpl, tok, true);
        tokens.push_back(tok);

        if (tok == tok_eos || tok == tok_eot) {
            break;
        }
    }

    // Decode to text, excluding EOS/EOT
    std::vector<llama_token> text_tokens;
    for (auto t : tokens) {
        if (t != tok_eos && t != tok_eot) {
            text_tokens.push_back(t);
        }
    }

    fuzz_generation gen;
    gen.tokens = std::move(tokens);
    gen.text   = common_detokenize(vocab, text_tokens, true);
    return gen;
}

// ---------------------------------------------------------------------------
// Input modes
// ---------------------------------------------------------------------------

struct fuzz_mode {
    const char *            name;
    common_reasoning_format reasoning;
    bool                    with_tools;
};

static const fuzz_mode fuzz_modes[] = {
    { "tools",          COMMON_REASONING_FORMAT_NONE, true  },
    { "tools+thinking", COMMON_REASONING_FORMAT_AUTO, true  },
    { "plain",          COMMON_REASONING_FORMAT_NONE, false },
};

// ---------------------------------------------------------------------------
// Main fuzz driver
// ---------------------------------------------------------------------------

static bool run_fuzz_tests(
        llama_model               * model,
        llama_context             * ctx,
        const common_chat_templates * tmpls,
        int                         iterations,
        int                         max_tokens,
        uint32_t                    seed,
        bool                        detailed) {

    const llama_vocab * vocab = llama_model_get_vocab(model);

    bool all_passed     = true;
    int  total_tests    = 0;
    int  total_failures = 0;
    int  total_skipped  = 0;

    for (const auto & mode : fuzz_modes) {
        common_chat_templates_inputs inputs;
        inputs.messages = { message_user };
        inputs.reasoning_format = mode.reasoning;
        if (mode.with_tools) {
            inputs.tools = fuzz_tools;
        }

        common_chat_params params;
        try {
            params = common_chat_templates_apply(tmpls, inputs);
        } catch (const std::exception & e) {
            if (detailed) {
                fprintf(stderr, "  SKIP [%s]: %s\n", mode.name, e.what());
            }
            total_skipped++;
            continue;
        }

        if (params.grammar.empty()) {
            if (detailed) {
                fprintf(stderr, "  SKIP [%s]: no grammar\n", mode.name);
            }
            total_skipped++;
            continue;
        }

        fprintf(stderr, "  FUZZ [%s] format=%s lazy=%d\n",
                mode.name,
                common_chat_format_name(params.format),
                (int)params.grammar_lazy);

        common_sampler * smpl = create_sampler(model, params, seed);
        if (!smpl) {
            fprintf(stderr, "    FAIL: could not create sampler\n");
            total_failures++;
            all_passed = false;
            continue;
        }

        // Load PEG parser
        common_peg_arena parser_arena;
        bool has_parser = false;
        if (!params.parser.empty()) {
            try {
                parser_arena.load(params.parser);
                has_parser = true;
            } catch (const std::exception & e) {
                fprintf(stderr, "    WARN: parser load failed: %s\n", e.what());
            }
        }

        std::mt19937 rng(seed);
        int mode_failures = 0;

        for (int iter = 0; iter < iterations; iter++) {
            total_tests++;

            auto gen = generate_fuzz_sequence(ctx, vocab, smpl, rng, max_tokens);

            if (gen.tokens.empty()) {
                continue;
            }

            if (detailed) {
                fprintf(stderr, "    iter %d: %zu tokens, %zu bytes\n",
                        iter, gen.tokens.size(), gen.text.size());
            }

            // For lazy grammars, check if the trigger actually fired
            bool grammar_active = !params.grammar_lazy;
            if (params.grammar_lazy) {
                for (const auto & trigger : params.grammar_triggers) {
                    if (trigger.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD &&
                            gen.text.find(trigger.value) != std::string::npos) {
                        grammar_active = true;
                        break;
                    }
                    if (trigger.type == COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN) {
                        std::regex re(trigger.value);
                        if (std::regex_search(gen.text, re)) {
                            grammar_active = true;
                            break;
                        }
                    }
                }
                if (!grammar_active && detailed) {
                    fprintf(stderr, "      lazy grammar not triggered, skipping parse\n");
                }
            }

            // Try PEG parse — only meaningful when grammar was active
            if (has_parser && grammar_active && !gen.text.empty()) {
                try {
                    common_chat_parser_params pp(params);
                    auto msg = common_chat_peg_parse(parser_arena, gen.text, false, pp);

                    if (detailed) {
                        fprintf(stderr, "      parse OK: content=%zu tool_calls=%zu reasoning=%zu\n",
                                msg.content.size(), msg.tool_calls.size(),
                                msg.reasoning_content.size());
                    }
                } catch (const std::exception & e) {
                    fprintf(stderr, "    FAIL iter %d: parse error: %s\n", iter, e.what());
                    fprintf(stderr, "      text (%zu bytes): %.200s%s\n",
                            gen.text.size(), gen.text.c_str(),
                            gen.text.size() > 200 ? "..." : "");
                    fprintf(stderr, "      tokens (%zu):", gen.tokens.size());
                    for (size_t i = 0; i < gen.tokens.size() && i < 32; i++) {
                        fprintf(stderr, " %d", gen.tokens[i]);
                    }
                    if (gen.tokens.size() > 32) {
                        fprintf(stderr, " ...");
                    }
                    fprintf(stderr, "\n");
                    mode_failures++;
                    all_passed = false;
                }
            }
        }

        if (mode_failures > 0) {
            fprintf(stderr, "    %d/%d failed\n", mode_failures, iterations);
            total_failures += mode_failures;
        } else {
            fprintf(stderr, "    OK (%d iterations)\n", iterations);
        }

        common_sampler_free(smpl);
    }

    fprintf(stderr, "\n=== Fuzz Summary ===\n");
    fprintf(stderr, "Tests: %d, Failures: %d, Skipped: %d\n",
            total_tests, total_failures, total_skipped);

    return all_passed;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    std::string model_path;
    std::string chat_template_override;
    uint32_t    seed       = 0;
    bool        seed_set   = false;
    int         iterations = 10;
    int         max_tokens = 512;
    bool        detailed   = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--chat-template" && i + 1 < argc) {
            chat_template_override = read_file(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed     = (uint32_t)std::stoul(argv[++i]);
            seed_set = true;
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_tokens = std::stoi(argv[++i]);
        } else if (arg == "--detailed") {
            detailed = true;
            common_log_set_verbosity_thold(999);
        } else if (arg == "--help" || arg == "-h") {
            fprintf(stderr,
                "Usage: %s --model <model.gguf> [options]\n\n"
                "  --model <path>             GGUF model file\n"
                "  --chat-template <file>     Override chat template with a Jinja file\n"
                "  --seed <N>                 RNG seed (default: random)\n"
                "  --iterations <N>           Iterations per mode (default: 10)\n"
                "  --max-tokens <N>           Max tokens per sequence (default: 512)\n"
                "  --detailed                 Verbose output\n",
                argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "Error: --model is required\n");
        return 1;
    }

    if (!seed_set) {
        seed = (uint32_t)std::random_device{}();
    }

    fprintf(stderr, "=== Chat Template Grammar Fuzz Test ===\n");
    fprintf(stderr, "Model: %s, Seed: %u, Iterations: %d, Max tokens: %d\n",
            model_path.c_str(), seed, iterations, max_tokens);

    llama_backend_init();

    // Load model
    auto mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "Error: failed to load model from '%s'\n", model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    fprintf(stderr, "Vocab size: %d\n", llama_vocab_n_tokens(vocab));

    // Load chat template — from model by default, or from override file
    auto tmpls = common_chat_templates_init(model, chat_template_override);
    if (!tmpls) {
        fprintf(stderr, "Error: no chat template found (model has none and no --chat-template given)\n");
        llama_model_free(model);
        return 1;
    }

    fprintf(stderr, "Template: %s\n",
            chat_template_override.empty() ? "(embedded in model)" : "(override)");

    // Create context
    auto cparams = llama_context_default_params();
    cparams.n_ctx   = 4;
    cparams.n_batch = 4;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Error: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    if (!prime_context(ctx, vocab)) {
        fprintf(stderr, "Error: failed to prime context\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    fprintf(stderr, "\n");

    bool passed = run_fuzz_tests(model, ctx, tmpls.get(), iterations, max_tokens, seed, detailed);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    if (!passed) {
        fprintf(stderr, "\nFAILED — re-run with --seed %u to reproduce\n", seed);
        return 1;
    }

    fprintf(stderr, "\nAll fuzz tests passed.\n");
    return 0;
}
