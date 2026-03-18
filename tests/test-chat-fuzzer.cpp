//  Fuzz test for grammar-constrained decoding of chat templates.
//
//  Loads a vocab-only model from a GGUF file, iterates over chat templates,
//  generates grammar-constrained random token sequences, and verifies that
//  the resulting text is parseable by the PEG parser.
//
//  Usage:
//    ./test-chat-fuzzer --model <vocab.gguf> [--seed N] [--iterations N] [--max-tokens N] [--template <filter>] [--detailed]
//
#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "../src/llama-grammar.h"
#include "../src/unicode.h"
#include "chat.h"
#include "common.h"
#include "log.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <regex>
#include <set>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers copied from test-chat.cpp (static there, can't be shared)
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

static std::unique_ptr<llama_grammar> build_grammar(const std::string & grammar_str) {
    return std::unique_ptr<llama_grammar>(
        llama_grammar_init_impl(nullptr, grammar_str.c_str(), "root", false, nullptr, 0, nullptr, 0));
}

// ---------------------------------------------------------------------------
// Detailed grammar match (copied from test-chat.cpp)
// ---------------------------------------------------------------------------

static std::string format_codepoint(uint32_t cp) {
    if (cp >= 32 && cp < 127) {
        return std::string("'") + static_cast<char>(cp) + "'";
    } else if (cp == '\n') {
        return "'\\n'";
    } else if (cp == '\r') {
        return "'\\r'";
    } else if (cp == '\t') {
        return "'\\t'";
    } else {
        return "U+" + std::to_string(cp);
    }
}

static std::string format_expected_element(const llama_grammar_rules & /* rules*/, const llama_grammar_element * elem) {
    if (!elem) {
        return "<end>";
    }
    switch (elem->type) {
        case LLAMA_GRETYPE_END:       return "<end of rule>";
        case LLAMA_GRETYPE_ALT:       return "<alternative>";
        case LLAMA_GRETYPE_RULE_REF:  return "<rule-" + std::to_string(elem->value) + ">";
        case LLAMA_GRETYPE_CHAR: {
            std::string result;
            const llama_grammar_element * pos = elem;
            bool first = true;
            do {
                if (!first) result += " | ";
                first = false;
                if (pos[1].type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
                    result += "[" + format_codepoint(pos->value) + "-" + format_codepoint(pos[1].value) + "]";
                    pos += 2;
                } else {
                    result += format_codepoint(pos->value);
                    pos += 1;
                }
            } while (pos->type == LLAMA_GRETYPE_CHAR_ALT);
            return result;
        }
        case LLAMA_GRETYPE_CHAR_NOT: {
            std::string result = "[^";
            const llama_grammar_element * pos = elem;
            bool first = true;
            do {
                if (!first) result += " ";
                first = false;
                if (pos[1].type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
                    result += format_codepoint(pos->value) + "-" + format_codepoint(pos[1].value);
                    pos += 2;
                } else {
                    result += format_codepoint(pos->value);
                    pos += 1;
                }
            } while (pos->type == LLAMA_GRETYPE_CHAR_ALT);
            return result + "]";
        }
        case LLAMA_GRETYPE_CHAR_ANY:  return "<any char>";
        case LLAMA_GRETYPE_TOKEN:     return "<token-" + std::to_string(elem->value) + ">";
        case LLAMA_GRETYPE_TOKEN_NOT: return "<not-token-" + std::to_string(elem->value) + ">";
        default:                      return "<unknown>";
    }
}

static std::string get_expected_description(const llama_grammar_rules & rules, const llama_grammar_stacks & stacks) {
    if (stacks.empty()) {
        return "<no valid continuations>";
    }
    std::string result;
    std::set<std::string> seen;
    for (const auto & stack : stacks) {
        if (stack.empty()) {
            if (seen.insert("<end>").second) {
                if (!result.empty()) result += " OR ";
                result += "<end>";
            }
            continue;
        }
        const llama_grammar_element * elem = stack.back();
        std::string desc = format_expected_element(rules, elem);
        if (seen.insert(desc).second) {
            if (!result.empty()) result += " OR ";
            result += desc;
        }
    }
    return result;
}

struct grammar_match_result {
    bool        success            = false;
    size_t      matched_bytes      = 0;
    size_t      matched_codepoints = 0;
    size_t      total_bytes        = 0;
    size_t      total_codepoints   = 0;
    std::string matched_prefix;
    std::string failing_char;
    std::string expected_description;
    bool        incomplete = false;
};

static grammar_match_result match_string_detailed(const std::string & input, llama_grammar * grammar) {
    grammar_match_result result;
    result.total_bytes = input.size();

    const auto cpts         = unicode_cpts_from_utf8(input);
    result.total_codepoints = cpts.size();

    auto &       stacks_cur = llama_grammar_get_stacks(grammar);
    const auto & rules      = llama_grammar_get_rules(grammar);

    size_t byte_pos = 0;

    for (size_t i = 0; i < cpts.size(); i++) {
        const auto & cpt = cpts[i];

        std::string expected_before = get_expected_description(rules, stacks_cur);

        llama_grammar_accept(grammar, cpt);

        size_t cpt_bytes = 0;
        if (cpt < 0x80)       cpt_bytes = 1;
        else if (cpt < 0x800)   cpt_bytes = 2;
        else if (cpt < 0x10000) cpt_bytes = 3;
        else                    cpt_bytes = 4;

        if (stacks_cur.empty()) {
            result.matched_bytes        = byte_pos;
            result.matched_codepoints   = i;
            result.matched_prefix       = input.substr(0, byte_pos);
            result.failing_char         = format_codepoint(cpt);
            result.expected_description = expected_before;
            return result;
        }

        byte_pos += cpt_bytes;
    }

    result.matched_bytes      = byte_pos;
    result.matched_codepoints = cpts.size();
    result.matched_prefix     = input;

    // Check if grammar accepts end (empty stack = accepting)
    bool accepts_end = false;
    for (const auto & stack : stacks_cur) {
        if (stack.empty()) {
            accepts_end = true;
            break;
        }
    }

    if (accepts_end) {
        result.success = true;
    } else {
        result.incomplete            = true;
        result.expected_description = get_expected_description(rules, stacks_cur);
    }

    return result;
}

// ---------------------------------------------------------------------------
// Tool definitions (subset from test-chat.cpp)
// ---------------------------------------------------------------------------

static common_chat_tool special_function_tool{
    /* .name = */ "special_function",
    /* .description = */ "I'm special",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "arg1": {
                "type": "integer",
                "description": "The arg."
            }
        },
        "required": ["arg1"]
    })",
};

static common_chat_tool python_tool{
    /* .name = */ "python",
    /* .description = */ "an ipython interpreter",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute."
            }
        },
        "required": ["code"]
    })",
};

static std::vector<common_chat_tool> fuzz_tools{ special_function_tool, python_tool };

static common_chat_msg message_user{
    "user",
    "Hey there!",
    /* .content_parts = */ {},
    /* .tool_calls = */ {},
    /* .reasoning_content = */ "",
    /* .tool_name = */ "",
    /* .tool_call_id = */ "",
};

// ---------------------------------------------------------------------------
// Template configurations for fuzzing
// ---------------------------------------------------------------------------

struct fuzz_template_config {
    std::string template_path;
};

static std::vector<fuzz_template_config> get_template_configs() {
    return {
        { "models/templates/mistralai-Ministral-3-14B-Reasoning-2512.jinja" },
        { "models/templates/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.jinja" },
        { "models/templates/CohereForAI-c4ai-command-r7b-12-2024-tool_use.jinja" },
        { "models/templates/Qwen-QwQ-32B.jinja" },
        { "models/templates/NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use.jinja" },
        { "models/templates/google-gemma-2-2b-it.jinja" },
        { "models/templates/ibm-granite-granite-3.3-2B-Instruct.jinja" },
        { "models/templates/ByteDance-Seed-OSS.jinja" },
        { "models/templates/Qwen3-Coder.jinja" },
        { "models/templates/deepseek-ai-DeepSeek-V3.1.jinja" },
        { "models/templates/GLM-4.6.jinja" },
        { "models/templates/GLM-4.7-Flash.jinja" },
        { "models/templates/Kimi-K2-Thinking.jinja" },
        { "models/templates/moonshotai-Kimi-K2.jinja" },
        { "models/templates/LFM2-8B-A1B.jinja" },
        { "models/templates/Apertus-8B-Instruct.jinja" },
        { "models/templates/MiniMax-M2.jinja" },
        { "models/templates/NVIDIA-Nemotron-Nano-v2.jinja" },
        { "models/templates/CohereForAI-c4ai-command-r-plus-tool_use.jinja" },
        { "models/templates/mistralai-Mistral-Nemo-Instruct-2407.jinja" },
        { "models/templates/meetkai-functionary-medium-v3.1.jinja" },
        { "models/templates/meetkai-functionary-medium-v3.2.jinja" },
        { "models/templates/fireworks-ai-llama-3-firefunction-v2.jinja" },
        { "models/templates/deepseek-ai-DeepSeek-R1-Distill-Llama-8B.jinja" },
        { "models/templates/llama-cpp-deepseek-r1.jinja" },
        { "models/templates/deepseek-ai-DeepSeek-R1-Distill-Qwen-32B.jinja" },
        { "models/templates/meta-llama-Llama-3.1-8B-Instruct.jinja" },
        { "models/templates/meta-llama-Llama-3.3-70B-Instruct.jinja" },
        { "models/templates/openai-gpt-oss-120b.jinja" },
        { "models/templates/StepFun3.5-Flash.jinja" },
        { "models/templates/GigaChat3-10B-A1.8B.jinja" },
        { "models/templates/GigaChat3.1-10B-A1.8B.jinja" },
        { "models/templates/Mistral-Small-3.2-24B-Instruct-2506.jinja" },
        { "models/templates/unsloth-mistral-Devstral-Small-2507.jinja" },
        { "models/templates/unsloth-Apriel-1.5.jinja" },
        { "models/templates/Apriel-1.6-15b-Thinker-fixed.jinja" },
    };
}

// ---------------------------------------------------------------------------
// Input configurations to test per template
// ---------------------------------------------------------------------------

enum class fuzz_input_mode {
    TOOLS_ONLY,
    TOOLS_WITH_THINKING,
    PLAIN,
};

static const char * fuzz_input_mode_name(fuzz_input_mode mode) {
    switch (mode) {
        case fuzz_input_mode::TOOLS_ONLY:          return "tools";
        case fuzz_input_mode::TOOLS_WITH_THINKING: return "tools+thinking";
        case fuzz_input_mode::PLAIN:               return "plain";
    }
    return "unknown";
}

static common_chat_templates_inputs make_fuzz_inputs(fuzz_input_mode mode) {
    common_chat_templates_inputs inputs;
    inputs.messages = { message_user };

    switch (mode) {
        case fuzz_input_mode::TOOLS_ONLY:
            inputs.tools            = fuzz_tools;
            inputs.reasoning_format = COMMON_REASONING_FORMAT_NONE;
            break;
        case fuzz_input_mode::TOOLS_WITH_THINKING:
            inputs.tools            = fuzz_tools;
            inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
            break;
        case fuzz_input_mode::PLAIN:
            inputs.reasoning_format = COMMON_REASONING_FORMAT_NONE;
            break;
    }

    return inputs;
}

// ---------------------------------------------------------------------------
// Grammar sampler creation (mirrors common/sampling.cpp logic)
// ---------------------------------------------------------------------------

static llama_sampler * create_grammar_sampler(
        const llama_vocab            * vocab,
        const common_chat_params     & params) {
    if (params.grammar.empty()) {
        return nullptr;
    }

    if (params.grammar_lazy) {
        // Build trigger patterns/tokens same way as common/sampling.cpp
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
// Weighted fuzz sampler
// ---------------------------------------------------------------------------

struct fuzz_sampler {
    const llama_vocab * vocab;
    int32_t             n_vocab;
    llama_token         tok_eos;
    llama_token         tok_eot;

    // Tokens that get boosted probability during sampling
    std::set<llama_token> structural_tokens;   // {, }, [, ], ", \n etc.
    std::set<llama_token> trigger_tokens;      // template-specific markers
    std::set<llama_token> eos_tokens;          // EOS/EOT

    std::mt19937 rng;

    fuzz_sampler(const llama_vocab * vocab_, uint32_t seed) : vocab(vocab_), rng(seed) {
        n_vocab = llama_vocab_n_tokens(vocab);

        tok_eos = llama_vocab_eos(vocab);
        tok_eot = llama_vocab_eot(vocab);
        if (tok_eot != LLAMA_TOKEN_NULL) {
            eos_tokens.insert(tok_eot);
        }
        if (tok_eos != LLAMA_TOKEN_NULL) {
            eos_tokens.insert(tok_eos);
        }

        // Pre-scan vocab for structural tokens
        const std::vector<std::string> structural_strs = {
            "{", "}", "[", "]", "\"", "\n", "<", ">", ",", ":", "/", "\\",
        };

        for (llama_token id = 0; id < n_vocab; id++) {
            std::string piece = common_token_to_piece(vocab, id, true);
            for (const auto & s : structural_strs) {
                if (piece.find(s) != std::string::npos) {
                    structural_tokens.insert(id);
                    break;
                }
            }
        }
    }

    void set_boost_strings(const std::vector<std::string> & boost_strs) {
        trigger_tokens.clear();
        for (llama_token id = 0; id < n_vocab; id++) {
            std::string piece = common_token_to_piece(vocab, id, true);
            for (const auto & s : boost_strs) {
                if (!s.empty() && piece.find(s) != std::string::npos) {
                    trigger_tokens.insert(id);
                    break;
                }
            }
        }
    }

    // Sample a token from the allowed set, with weighted boosting.
    // Returns LLAMA_TOKEN_NULL if no tokens are allowed.
    llama_token sample(const std::vector<llama_token_data> & candidates) {
        // Collect allowed tokens
        std::vector<llama_token> allowed;
        std::vector<float>       weights;

        for (const auto & c : candidates) {
            if (c.logit >= 0.0f) {
                allowed.push_back(c.id);

                float w = 1.0f;
                if (eos_tokens.count(c.id)) {
                    w = std::exp(2.0f);  // moderate EOS boost
                } else if (trigger_tokens.count(c.id)) {
                    w = std::exp(5.0f);  // strong trigger boost
                } else if (structural_tokens.count(c.id)) {
                    w = std::exp(3.0f);  // structural boost
                }
                weights.push_back(w);
            }
        }

        if (allowed.empty()) {
            return LLAMA_TOKEN_NULL;
        }

        std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
        return allowed[dist(rng)];
    }
};

// ---------------------------------------------------------------------------
// Core fuzz routine
// ---------------------------------------------------------------------------

struct fuzz_result {
    bool        success = true;
    std::string error;
    std::string generated_text;
    std::vector<llama_token> generated_tokens;
};

static fuzz_result fuzz_one_iteration(
        fuzz_sampler     & sampler,
        llama_sampler    * grammar_smpl,
        int                max_tokens) {
    fuzz_result result;

    llama_sampler_reset(grammar_smpl);

    std::vector<llama_token_data> candidates;
    candidates.resize(sampler.n_vocab);

    std::vector<llama_token> tokens;

    for (int step = 0; step < max_tokens; step++) {
        // Reset logits
        for (int32_t i = 0; i < sampler.n_vocab; i++) {
            candidates[i] = { i, 0.0f, 0.0f };
        }

        llama_token_data_array arr = { candidates.data(), (size_t)sampler.n_vocab, -1, false };
        llama_sampler_apply(grammar_smpl, &arr);

        llama_token tok = sampler.sample(candidates);

        if (tok == LLAMA_TOKEN_NULL) {
            // No tokens allowed — grammar complete or stuck
            break;
        }

        // Check if this is an EOS token
        if (sampler.eos_tokens.count(tok)) {
            llama_sampler_accept(grammar_smpl, tok);
            tokens.push_back(tok);
            break;
        }

        llama_sampler_accept(grammar_smpl, tok);
        tokens.push_back(tok);
    }

    // Decode to text (exclude EOS/EOT from the decoded text)
    std::vector<llama_token> text_tokens;
    for (auto t : tokens) {
        if (!sampler.eos_tokens.count(t)) {
            text_tokens.push_back(t);
        }
    }
    result.generated_text   = common_detokenize(sampler.vocab, text_tokens, true);
    result.generated_tokens = tokens;
    result.success          = true;

    return result;
}

// ---------------------------------------------------------------------------
// Main fuzz test driver
// ---------------------------------------------------------------------------

static bool run_fuzz_tests(
        const llama_vocab   * vocab,
        const std::string   & template_filter,
        int                   iterations,
        int                   max_tokens,
        uint32_t              seed,
        bool                  detailed_debug) {

    auto configs = get_template_configs();
    bool all_passed = true;
    int  total_tests = 0;
    int  total_failures = 0;
    int  total_skipped = 0;

    const std::vector<fuzz_input_mode> modes = {
        fuzz_input_mode::TOOLS_ONLY,
        fuzz_input_mode::TOOLS_WITH_THINKING,
        fuzz_input_mode::PLAIN,
    };

    for (const auto & config : configs) {
        // Apply template filter
        if (!template_filter.empty()) {
            std::string path_lower = config.template_path;
            std::string filter_lower = template_filter;
            std::transform(path_lower.begin(), path_lower.end(), path_lower.begin(), ::tolower);
            std::transform(filter_lower.begin(), filter_lower.end(), filter_lower.begin(), ::tolower);
            if (path_lower.find(filter_lower) == std::string::npos) {
                continue;
            }
        }

        common_chat_templates_ptr tmpls;
        try {
            tmpls = read_templates(config.template_path);
        } catch (const std::exception & e) {
            fprintf(stderr, "  SKIP %s: %s\n", config.template_path.c_str(), e.what());
            total_skipped++;
            continue;
        }

        for (auto mode : modes) {
            auto inputs = make_fuzz_inputs(mode);

            common_chat_params params;
            try {
                params = common_chat_templates_apply(tmpls.get(), inputs);
            } catch (const std::exception & e) {
                if (detailed_debug) {
                    fprintf(stderr, "  SKIP %s [%s]: template apply failed: %s\n",
                            config.template_path.c_str(), fuzz_input_mode_name(mode), e.what());
                }
                total_skipped++;
                continue;
            }

            if (params.grammar.empty()) {
                if (detailed_debug) {
                    fprintf(stderr, "  SKIP %s [%s]: no grammar\n",
                            config.template_path.c_str(), fuzz_input_mode_name(mode));
                }
                total_skipped++;
                continue;
            }

            fprintf(stderr, "  FUZZ %s [%s] format=%s grammar_lazy=%d\n",
                    config.template_path.c_str(),
                    fuzz_input_mode_name(mode),
                    common_chat_format_name(params.format),
                    (int)params.grammar_lazy);

            // Create grammar sampler
            llama_sampler * grammar_smpl = create_grammar_sampler(vocab, params);
            if (!grammar_smpl) {
                fprintf(stderr, "    FAIL: could not create grammar sampler\n");
                total_failures++;
                all_passed = false;
                continue;
            }

            // Build PEG parser for validation
            common_peg_arena parser_arena;
            bool has_parser = false;
            if (!params.parser.empty()) {
                try {
                    parser_arena.load(params.parser);
                    has_parser = true;
                } catch (const std::exception & e) {
                    fprintf(stderr, "    WARN: could not load PEG parser: %s\n", e.what());
                }
            }

            // Build boost strings from grammar triggers and thinking tags
            std::vector<std::string> boost_strs;
            for (const auto & trigger : params.grammar_triggers) {
                if (trigger.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
                    boost_strs.push_back(trigger.value);
                }
            }
            if (!params.thinking_end_tag.empty()) {
                boost_strs.push_back(params.thinking_end_tag);
            }
            if (!params.thinking_start_tag.empty()) {
                boost_strs.push_back(params.thinking_start_tag);
            }

            fuzz_sampler sampler(vocab, seed);
            sampler.set_boost_strings(boost_strs);

            int mode_failures = 0;

            for (int iter = 0; iter < iterations; iter++) {
                total_tests++;

                auto result = fuzz_one_iteration(sampler, grammar_smpl, max_tokens);

                if (result.generated_tokens.empty()) {
                    if (detailed_debug) {
                        fprintf(stderr, "    iter %d: empty (no tokens generated)\n", iter);
                    }
                    continue;
                }

                if (detailed_debug) {
                    fprintf(stderr, "    iter %d: %zu tokens, %zu bytes: %.80s%s\n",
                            iter,
                            result.generated_tokens.size(),
                            result.generated_text.size(),
                            result.generated_text.c_str(),
                            result.generated_text.size() > 80 ? "..." : "");
                }

                // Validate against grammar (character-level)
                auto grammar_check = build_grammar(params.grammar);
                if (grammar_check) {
                    auto match = match_string_detailed(result.generated_text, grammar_check.get());
                    // For lazy grammars, the text before the trigger is unconstrained,
                    // so we only validate grammar match for non-lazy grammars
                    if (!params.grammar_lazy && !match.success) {
                        fprintf(stderr, "    FAIL iter %d: grammar mismatch\n", iter);
                        fprintf(stderr, "      matched %zu/%zu bytes, %zu/%zu codepoints\n",
                                match.matched_bytes, match.total_bytes,
                                match.matched_codepoints, match.total_codepoints);
                        if (match.incomplete) {
                            fprintf(stderr, "      grammar expects more: %s\n",
                                    match.expected_description.c_str());
                        } else {
                            fprintf(stderr, "      failing char: %s, expected: %s\n",
                                    match.failing_char.c_str(),
                                    match.expected_description.c_str());
                        }
                        fprintf(stderr, "      text: %s\n", result.generated_text.c_str());
                        fprintf(stderr, "      tokens:");
                        for (auto t : result.generated_tokens) {
                            fprintf(stderr, " %d", t);
                        }
                        fprintf(stderr, "\n");
                        mode_failures++;
                        all_passed = false;
                    }
                }

                // Validate PEG parse (skip partial-only content)
                if (has_parser && !result.generated_text.empty()) {
                    try {
                        common_chat_parser_params parser_params(params);
                        auto msg = common_chat_peg_parse(parser_arena, result.generated_text, false, parser_params);
                        // Parse succeeded — good
                        if (detailed_debug) {
                            fprintf(stderr, "      PEG parse OK: role=%s content_len=%zu tool_calls=%zu\n",
                                    msg.role.c_str(), msg.content.size(), msg.tool_calls.size());
                        }
                    } catch (const std::exception & e) {
                        // PEG parse can legitimately fail for partial/truncated output
                        // or sequences that ended at an odd point. Only count as failure
                        // if the grammar said the sequence was complete (non-lazy case).
                        if (!params.grammar_lazy) {
                            // Check if grammar considers this complete
                            auto gc = build_grammar(params.grammar);
                            if (gc) {
                                auto m = match_string_detailed(result.generated_text, gc.get());
                                if (m.success) {
                                    // Grammar says it's complete but PEG can't parse — real failure
                                    fprintf(stderr, "    FAIL iter %d: PEG parse failed on grammar-valid text\n", iter);
                                    fprintf(stderr, "      error: %s\n", e.what());
                                    fprintf(stderr, "      text: %s\n", result.generated_text.c_str());
                                    fprintf(stderr, "      tokens:");
                                    for (auto t : result.generated_tokens) {
                                        fprintf(stderr, " %d", t);
                                    }
                                    fprintf(stderr, "\n");
                                    mode_failures++;
                                    all_passed = false;
                                }
                            }
                        }
                    }
                }
            }

            if (mode_failures > 0) {
                fprintf(stderr, "    %d/%d iterations failed\n", mode_failures, iterations);
                total_failures += mode_failures;
            } else {
                fprintf(stderr, "    OK (%d iterations)\n", iterations);
            }

            llama_sampler_free(grammar_smpl);
        }
    }

    fprintf(stderr, "\n=== Fuzz Summary ===\n");
    fprintf(stderr, "Total tests: %d, Failures: %d, Skipped: %d\n",
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
                "Usage: %s --model <vocab.gguf> [options]\n"
                "\n"
                "Options:\n"
                "  --model <path>       Path to a GGUF file (vocab-only loading)\n"
                "  --seed <N>           RNG seed for reproducibility (default: random)\n"
                "  --iterations <N>     Fuzz iterations per template config (default: 10)\n"
                "  --max-tokens <N>     Max tokens per generated sequence (default: 512)\n"
                "  --template <filter>  Only test templates matching this substring\n"
                "  --detailed           Verbose debug output\n",
                argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "Error: --model is required\n");
        fprintf(stderr, "Usage: %s --model <vocab.gguf> [options]\n", argv[0]);
        return 1;
    }

    if (!seed_set) {
        seed = (uint32_t)std::random_device{}();
    }

    fprintf(stderr, "=== Chat Template Grammar Fuzz Test ===\n");
    fprintf(stderr, "Model:      %s\n", model_path.c_str());
    fprintf(stderr, "Seed:       %u\n", seed);
    fprintf(stderr, "Iterations: %d\n", iterations);
    fprintf(stderr, "Max tokens: %d\n", max_tokens);
    if (!template_filter.empty()) {
        fprintf(stderr, "Filter:     %s\n", template_filter.c_str());
    }
    fprintf(stderr, "\n");

    llama_backend_init();

    // Load vocab-only model
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
        fprintf(stderr, "\nFuzz test FAILED! Re-run with --seed %u to reproduce.\n", seed);
        return 1;
    }

    fprintf(stderr, "\nAll fuzz tests PASSED!\n");
    return 0;
}
