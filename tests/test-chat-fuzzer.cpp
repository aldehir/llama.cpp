//  Fuzz test for grammar-constrained decoding of chat templates.
//
//  Loads a vocab-only model from a GGUF file, generates random token sequences
//  constrained by the grammar sampler, and verifies the output is parseable.
//
//  Usage:
//    ./test-chat-fuzzer --model <vocab.gguf> [--seed N] [--iterations N] [--max-tokens N] [--template <filter>] [--detailed]
//
#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "chat.h"
#include "common.h"
#include "log.h"

#include <algorithm>
#include <cassert>
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
        fs = std::ifstream("../" + path, std::ios_base::binary);
        if (!fs.is_open()) {
            throw std::runtime_error("Failed to open file: " + path);
        }
    }
    fs.seekg(0, std::ios_base::end);
    auto size = fs.tellg();
    fs.seekg(0);
    std::string out;
    out.resize(static_cast<size_t>(size));
    fs.read(out.data(), static_cast<std::streamsize>(size));
    return out;
}

static common_chat_templates_ptr read_templates(const std::string & path) {
    return common_chat_templates_ptr(common_chat_templates_init(/* model= */ nullptr, read_file(path)));
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
// Templates to fuzz
// ---------------------------------------------------------------------------

static std::vector<std::string> get_template_paths() {
    return {
        "models/templates/mistralai-Ministral-3-14B-Reasoning-2512.jinja",
        "models/templates/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.jinja",
        "models/templates/CohereForAI-c4ai-command-r7b-12-2024-tool_use.jinja",
        "models/templates/Qwen-QwQ-32B.jinja",
        "models/templates/NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use.jinja",
        "models/templates/google-gemma-2-2b-it.jinja",
        "models/templates/ibm-granite-granite-3.3-2B-Instruct.jinja",
        "models/templates/ByteDance-Seed-OSS.jinja",
        "models/templates/Qwen3-Coder.jinja",
        "models/templates/deepseek-ai-DeepSeek-V3.1.jinja",
        "models/templates/GLM-4.6.jinja",
        "models/templates/GLM-4.7-Flash.jinja",
        "models/templates/Kimi-K2-Thinking.jinja",
        "models/templates/moonshotai-Kimi-K2.jinja",
        "models/templates/LFM2-8B-A1B.jinja",
        "models/templates/Apertus-8B-Instruct.jinja",
        "models/templates/MiniMax-M2.jinja",
        "models/templates/NVIDIA-Nemotron-Nano-v2.jinja",
        "models/templates/CohereForAI-c4ai-command-r-plus-tool_use.jinja",
        "models/templates/mistralai-Mistral-Nemo-Instruct-2407.jinja",
        "models/templates/meetkai-functionary-medium-v3.1.jinja",
        "models/templates/meetkai-functionary-medium-v3.2.jinja",
        "models/templates/fireworks-ai-llama-3-firefunction-v2.jinja",
        "models/templates/deepseek-ai-DeepSeek-R1-Distill-Llama-8B.jinja",
        "models/templates/llama-cpp-deepseek-r1.jinja",
        "models/templates/deepseek-ai-DeepSeek-R1-Distill-Qwen-32B.jinja",
        "models/templates/meta-llama-Llama-3.1-8B-Instruct.jinja",
        "models/templates/meta-llama-Llama-3.3-70B-Instruct.jinja",
        "models/templates/openai-gpt-oss-120b.jinja",
        "models/templates/StepFun3.5-Flash.jinja",
        "models/templates/GigaChat3-10B-A1.8B.jinja",
        "models/templates/GigaChat3.1-10B-A1.8B.jinja",
        "models/templates/Mistral-Small-3.2-24B-Instruct-2506.jinja",
        "models/templates/unsloth-mistral-Devstral-Small-2507.jinja",
        "models/templates/unsloth-Apriel-1.5.jinja",
        "models/templates/Apriel-1.6-15b-Thinker-fixed.jinja",
    };
}

// ---------------------------------------------------------------------------
// Grammar sampler creation (mirrors common/sampling.cpp)
// ---------------------------------------------------------------------------

static llama_sampler * create_grammar_sampler(
        const llama_vocab        * vocab,
        const common_chat_params & params) {
    if (params.grammar.empty()) {
        return nullptr;
    }

    if (params.grammar_lazy) {
        std::vector<std::string> trigger_patterns;
        std::vector<llama_token> trigger_tokens;

        for (const auto & trigger : params.grammar_triggers) {
            switch (trigger.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                    trigger_patterns.push_back(regex_escape(trigger.value));
                    break;
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                    trigger_patterns.push_back(trigger.value);
                    break;
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL: {
                    const auto & pattern = trigger.value;
                    std::string anchored = "^$";
                    if (!pattern.empty()) {
                        anchored = (pattern.front() != '^' ? "^" : "")
                            + pattern
                            + (pattern.back() != '$' ? "$" : "");
                    }
                    trigger_patterns.push_back(anchored);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                    trigger_tokens.push_back(trigger.token);
                    break;
                default:
                    break;
            }
        }

        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & p : trigger_patterns) {
            trigger_patterns_c.push_back(p.c_str());
        }

        return llama_sampler_init_grammar_lazy_patterns(
            vocab, params.grammar.c_str(), "root",
            trigger_patterns_c.data(), trigger_patterns_c.size(),
            trigger_tokens.data(), trigger_tokens.size());
    }

    return llama_sampler_init_grammar(vocab, params.grammar.c_str(), "root");
}

// ---------------------------------------------------------------------------
// Core: generate one random grammar-constrained sequence
// ---------------------------------------------------------------------------

struct fuzz_generation {
    std::vector<llama_token> tokens;
    std::string              text;
};

static fuzz_generation generate_fuzz_sequence(
        const llama_vocab * vocab,
        llama_sampler     * grammar_smpl,
        std::mt19937      & rng,
        int                 max_tokens) {

    llama_sampler_reset(grammar_smpl);

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const llama_token tok_eos = llama_vocab_eos(vocab);
    const llama_token tok_eot = llama_vocab_eot(vocab);

    std::vector<llama_token_data> candidates(n_vocab);
    std::vector<llama_token> tokens;

    // Random logit distribution — simulates random model output
    std::uniform_real_distribution<float> logit_dist(-1.0f, 1.0f);

    for (int step = 0; step < max_tokens; step++) {
        // Fill with random logits
        for (int32_t i = 0; i < n_vocab; i++) {
            candidates[i] = { i, logit_dist(rng), 0.0f };
        }

        // Grammar sampler masks disallowed tokens (sets logit to -inf)
        llama_token_data_array arr = { candidates.data(), (size_t)n_vocab, -1, false };
        llama_sampler_apply(grammar_smpl, &arr);

        // Collect allowed tokens and sample from them using the random logits as weights
        std::vector<llama_token> allowed;
        std::vector<float>       weights;

        for (int32_t i = 0; i < n_vocab; i++) {
            if (candidates[i].logit > -1e9f) {  // not masked
                allowed.push_back(candidates[i].id);
                // Use exp(logit) as weight so higher random logits are more likely
                weights.push_back(std::exp(candidates[i].logit));
            }
        }

        if (allowed.empty()) {
            break;  // grammar complete or no valid continuations
        }

        std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
        llama_token tok = allowed[dist(rng)];

        llama_sampler_accept(grammar_smpl, tok);
        tokens.push_back(tok);

        // Stop on EOS/EOT
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
// Main driver
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

static bool run_fuzz_tests(
        const llama_vocab * vocab,
        const std::string & template_filter,
        int                 iterations,
        int                 max_tokens,
        uint32_t            seed,
        bool                detailed) {

    bool all_passed     = true;
    int  total_tests    = 0;
    int  total_failures = 0;
    int  total_skipped  = 0;

    for (const auto & tmpl_path : get_template_paths()) {
        // Filter
        if (!template_filter.empty()) {
            std::string path_lower = tmpl_path;
            std::string filt_lower = template_filter;
            std::transform(path_lower.begin(), path_lower.end(), path_lower.begin(), ::tolower);
            std::transform(filt_lower.begin(), filt_lower.end(), filt_lower.begin(), ::tolower);
            if (path_lower.find(filt_lower) == std::string::npos) {
                continue;
            }
        }

        common_chat_templates_ptr tmpls;
        try {
            tmpls = read_templates(tmpl_path);
        } catch (const std::exception & e) {
            fprintf(stderr, "  SKIP %s: %s\n", tmpl_path.c_str(), e.what());
            total_skipped++;
            continue;
        }

        for (const auto & mode : fuzz_modes) {
            common_chat_templates_inputs inputs;
            inputs.messages = { message_user };
            inputs.reasoning_format = mode.reasoning;
            if (mode.with_tools) {
                inputs.tools = fuzz_tools;
            }

            common_chat_params params;
            try {
                params = common_chat_templates_apply(tmpls.get(), inputs);
            } catch (const std::exception &) {
                total_skipped++;
                continue;
            }

            if (params.grammar.empty()) {
                total_skipped++;
                continue;
            }

            fprintf(stderr, "  FUZZ %s [%s] format=%s lazy=%d\n",
                    tmpl_path.c_str(), mode.name,
                    common_chat_format_name(params.format),
                    (int)params.grammar_lazy);

            llama_sampler * grammar_smpl = create_grammar_sampler(vocab, params);
            if (!grammar_smpl) {
                fprintf(stderr, "    FAIL: could not create grammar sampler\n");
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

                auto gen = generate_fuzz_sequence(vocab, grammar_smpl, rng, max_tokens);

                if (gen.tokens.empty()) {
                    continue;
                }

                if (detailed) {
                    fprintf(stderr, "    iter %d: %zu tokens, %zu bytes\n",
                            iter, gen.tokens.size(), gen.text.size());
                }

                // For lazy grammars, check if the trigger actually fired.
                // If not, the output is unconstrained random text — nothing to validate.
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

            llama_sampler_free(grammar_smpl);
        }
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
    std::string template_filter;
    uint32_t    seed       = 0;
    bool        seed_set   = false;
    int         iterations = 10;
    int         max_tokens = 512;
    bool        detailed   = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            seed     = (uint32_t)std::stoul(argv[++i]);
            seed_set = true;
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_tokens = std::stoi(argv[++i]);
        } else if (arg == "--template" && i + 1 < argc) {
            template_filter = argv[++i];
        } else if (arg == "--detailed") {
            detailed = true;
            common_log_set_verbosity_thold(999);
        } else if (arg == "--help" || arg == "-h") {
            fprintf(stderr,
                "Usage: %s --model <vocab.gguf> [options]\n\n"
                "  --model <path>       GGUF file (loaded vocab-only)\n"
                "  --seed <N>           RNG seed (default: random)\n"
                "  --iterations <N>     Iterations per template config (default: 10)\n"
                "  --max-tokens <N>     Max tokens per sequence (default: 512)\n"
                "  --template <filter>  Only test matching templates\n"
                "  --detailed           Verbose output\n",
                argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "Error: --model is required\nUsage: %s --model <vocab.gguf> [options]\n", argv[0]);
        return 1;
    }

    if (!seed_set) {
        seed = (uint32_t)std::random_device{}();
    }

    fprintf(stderr, "=== Chat Template Grammar Fuzz Test ===\n");
    fprintf(stderr, "Model: %s, Seed: %u, Iterations: %d, Max tokens: %d\n\n",
            model_path.c_str(), seed, iterations, max_tokens);

    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "Error: failed to load model from '%s'\n", model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    fprintf(stderr, "Vocab size: %d\n\n", llama_vocab_n_tokens(vocab));

    bool passed = run_fuzz_tests(vocab, template_filter, iterations, max_tokens, seed, detailed);

    llama_model_free(model);
    llama_backend_free();

    if (!passed) {
        fprintf(stderr, "\nFAILED — re-run with --seed %u to reproduce\n", seed);
        return 1;
    }

    fprintf(stderr, "\nAll fuzz tests passed.\n");
    return 0;
}
