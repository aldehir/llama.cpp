#include "common.h"
#include "llama.h"

// Internal headers for grammar access (same pattern as tests)
#include "../../src/llama-grammar.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

// ============================================================
// Helpers
// ============================================================

static std::string read_file(const char * path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fprintf(stderr, "error: could not open '%s': %s\n", path, strerror(errno));
        return {};
    }
    std::stringstream buf;
    buf << in.rdbuf();
    return buf.str();
}

static std::string escape_for_display(const std::string & s) {
    std::string out;
    for (unsigned char c : s) {
        if (c == '\n')      out += "\\n";
        else if (c == '\t') out += "\\t";
        else if (c == '\r') out += "\\r";
        else if (c == '\\') out += "\\\\";
        else if (c < 0x20 || c == 0x7f) {
            char tmp[8];
            snprintf(tmp, sizeof(tmp), "\\x%02x", c);
            out += tmp;
        } else {
            out += (char)c;
        }
    }
    return out;
}

static void llama_log_callback_null(ggml_log_level, const char *, void *) {}

// ============================================================
// Usage
// ============================================================

static void print_usage(const char * prog) {
    printf("Usage: %s -g GRAMMAR_FILE [-m MODEL] [options]\n\n", prog);
    printf("Test grammar correctness and profile logit-masking performance.\n");
    printf("The model is used only for its vocabulary (vocab_only load).\n\n");
    printf("Required:\n");
    printf("  -g, --grammar-file FILE    GBNF grammar file\n\n");
    printf("Required for all modes except --graph:\n");
    printf("  -m, --model MODEL          Model file (vocab only)\n\n");
    printf("Modes (pick one, default: --validate):\n");
    printf("  --validate                 Validate test strings against grammar\n");
    printf("  --bench                    Benchmark grammar pipeline stages\n");
    printf("  --candidates               Dump candidate set analysis\n");
    printf("  --masking-report           Per-token accept/reject categorization\n");
    printf("  --graph                    Output graphviz dot graph of grammar\n\n");
    printf("Options:\n");
    printf("  --test-file FILE           Test strings file (+pass / -fail per line)\n");
    printf("  --test-pass STR            String expected to be accepted (repeatable)\n");
    printf("  --test-fail STR            String expected to be rejected (repeatable)\n");
    printf("  --bench-iters N            Iterations for apply timing (default: 1000)\n");
    printf("  --state STR                Advance grammar by this prefix first\n");
    printf("  --grammar-root NAME        Start rule name (default: root)\n");
    printf("  --log-disable              Suppress model loading logs\n");
    printf("  -h, --help                 Print this help\n");
}

// ============================================================
// Test string collection
// ============================================================

struct test_string {
    std::string str;
    bool expect_accept;
};

static std::vector<test_string> load_test_file(const char * path) {
    std::vector<test_string> tests;
    std::ifstream in(path);
    if (!in) {
        fprintf(stderr, "error: could not open test file '%s'\n", path);
        return tests;
    }
    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
        lineno++;
        if (line.empty()) continue;
        if (line[0] == '#') continue;
        if (line[0] == '+') {
            tests.push_back({line.substr(1), true});
        } else if (line[0] == '-') {
            tests.push_back({line.substr(1), false});
        } else {
            fprintf(stderr, "warning: %s:%d: line must start with + or - (skipping)\n", path, lineno);
        }
    }
    return tests;
}

// ============================================================
// Mode: validate
// ============================================================

static bool match_string(const std::string & input, const compiled_grammar & cg,
                         const std::vector<parse_config> & init_configs) {
    auto configs = init_configs;
    for (uint8_t b : input) {
        cg.accept_byte(configs, b);
        if (configs.empty()) return false;
    }
    return cg.any_config_complete(configs);
}

static void trace_string(const std::string & input, const compiled_grammar & cg,
                         const std::vector<parse_config> & init_configs) {
    auto configs = init_configs;
    fprintf(stderr, "    Trace: %zu initial config(s)\n", configs.size());
    for (size_t i = 0; i < input.size(); i++) {
        uint8_t b = (uint8_t)input[i];
        char display[8];
        if (b >= 0x20 && b < 0x7f) {
            snprintf(display, sizeof(display), "'%c'", b);
        } else {
            snprintf(display, sizeof(display), "0x%02x", b);
        }
        cg.accept_byte(configs, b);
        fprintf(stderr, "    byte[%zu]=%s -> %zu config(s)", i, display, configs.size());
        if (!configs.empty()) {
            const auto & c = configs[0];
            fprintf(stderr, " [first: rule=%u,alt=%u,seg=%u,dfa_st=%u]",
                    c.current.rule_id, c.current.alternate_idx,
                    c.current.segment_idx, c.current.dfa_state);
        }
        fprintf(stderr, "\n");
        if (configs.empty()) {
            fprintf(stderr, "    => dead at byte %zu\n", i);
            return;
        }
    }
    bool complete = cg.any_config_complete(configs);
    fprintf(stderr, "    => end of string, complete=%s\n", complete ? "true" : "false");
}

static int mode_validate(const llama_grammar & grammar,
                         const std::vector<test_string> & tests) {
    if (tests.empty()) {
        fprintf(stderr, "error: no test strings provided (use --test-pass, --test-fail, or --test-file)\n");
        return 1;
    }

    printf("Grammar: %zu rules, %zu DFAs, %zu initial configs\n",
           grammar.compiled->rules.size(),
           grammar.compiled->dfas.size(),
           grammar.configs.size());

    int pass = 0, fail = 0;
    for (const auto & t : tests) {
        bool accepted = match_string(t.str, *grammar.compiled, grammar.configs);
        bool ok = (accepted == t.expect_accept);

        if (ok) {
            printf("  PASS  \"%s\"  %s (expected: %s)\n",
                   escape_for_display(t.str).c_str(),
                   accepted ? "accepted" : "rejected",
                   t.expect_accept ? "accept" : "reject");
            pass++;
        } else {
            printf("  FAIL  \"%s\"  %s (expected: %s)\n",
                   escape_for_display(t.str).c_str(),
                   accepted ? "accepted" : "rejected",
                   t.expect_accept ? "accept" : "reject");
            trace_string(t.str, *grammar.compiled, grammar.configs);
            fail++;
        }
    }

    printf("\nResults: %d passed, %d failed, %d total\n", pass, fail, pass + fail);
    return fail > 0 ? 1 : 0;
}

// ============================================================
// Mode: bench
// ============================================================

struct timing_stats {
    double mean_ms;
    double min_ms;
    double max_ms;
    double stddev_ms;
};

static timing_stats compute_stats(const std::vector<double> & samples) {
    timing_stats s{};
    if (samples.empty()) return s;
    s.min_ms = *std::min_element(samples.begin(), samples.end());
    s.max_ms = *std::max_element(samples.begin(), samples.end());
    s.mean_ms = std::accumulate(samples.begin(), samples.end(), 0.0) / (double)samples.size();
    double var = 0.0;
    for (double v : samples) {
        double d = v - s.mean_ms;
        var += d * d;
    }
    s.stddev_ms = std::sqrt(var / (double)samples.size());
    return s;
}

static int mode_bench(const llama_vocab * vocab, const std::string & grammar_str,
                      const char * grammar_root, const std::string & state_prefix,
                      int bench_iters) {
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    using clock = std::chrono::steady_clock;

    // 1. Parse
    auto t0 = clock::now();
    llama_grammar_parser parser;
    if (!parser.parse(grammar_str.c_str())) {
        fprintf(stderr, "error: grammar parse failed\n");
        return 1;
    }
    auto t1 = clock::now();
    double parse_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (parser.symbol_ids.find(grammar_root) == parser.symbol_ids.end()) {
        fprintf(stderr, "error: grammar does not contain '%s' rule\n", grammar_root);
        return 1;
    }

    // 2. Compile
    auto t2 = clock::now();
    auto cg = std::make_shared<compiled_grammar>(
        compiled_grammar::compile(parser.rules, (uint32_t)parser.symbol_ids.at(grammar_root)));
    auto t3 = clock::now();
    double compile_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    size_t total_states = 0;
    for (const auto & dfa : cg->dfas) {
        total_states += dfa.transitions.size();
    }

    // 3. Trie + precompute (done together inside init_impl)
    auto t5 = clock::now();

    // Build grammar via init_impl (does trie + precompute internally)
    llama_grammar * grammar = llama_grammar_init_impl(
        vocab, grammar_str.c_str(), grammar_root, false, nullptr, 0, nullptr, 0);
    auto t6 = clock::now();
    double init_ms = std::chrono::duration<double, std::milli>(t6 - t5).count();

    if (!grammar) {
        fprintf(stderr, "error: grammar init failed\n");
        return 1;
    }

    // Advance state if prefix given
    if (!state_prefix.empty()) {
        for (uint8_t b : state_prefix) {
            grammar->compiled->accept_byte(grammar->configs, b);
            if (grammar->configs.empty()) {
                fprintf(stderr, "error: grammar died at prefix byte 0x%02x\n", b);
                llama_grammar_free_impl(grammar);
                return 1;
            }
        }
    }

    // 4. Benchmark apply_impl
    // Build token data array
    std::vector<llama_token_data> token_data(n_vocab);
    for (int32_t i = 0; i < n_vocab; i++) {
        token_data[i] = {i, 0.0f, 0.0f};
    }

    std::vector<double> timings;
    timings.reserve(bench_iters);

    for (int iter = 0; iter < bench_iters; iter++) {
        // Reset logits each iteration
        for (int32_t i = 0; i < n_vocab; i++) {
            token_data[i].logit = 0.0f;
        }
        llama_token_data_array cur_p = {token_data.data(), (size_t)n_vocab, -1, false};

        auto ta = clock::now();
        llama_grammar_apply_impl(*grammar, &cur_p);
        auto tb = clock::now();
        timings.push_back(std::chrono::duration<double, std::milli>(tb - ta).count());
    }

    auto stats = compute_stats(timings);

    // Count token categories from last iteration's results
    int cnt_precomp_reject = 0, cnt_precomp_accept = 0;
    int cnt_sim_accept = 0, cnt_sim_reject = 0;
    int cnt_eog = 0, cnt_empty = 0;

    // Re-run once with fresh logits to count categories
    for (int32_t i = 0; i < n_vocab; i++) {
        token_data[i].logit = 0.0f;
    }
    llama_token_data_array cur_p = {token_data.data(), (size_t)n_vocab, -1, false};
    llama_grammar_apply_impl(*grammar, &cur_p);

    // Build rejected_by_all + is_context the same way apply_impl does
    std::vector<bool> rejected_by_all(n_vocab, true);
    std::vector<bool> is_context(n_vocab, false);
    {
        const bool have_candidates = !grammar->compiled->dfa_candidates.empty();
        bool any_config_matched = false;

        if (have_candidates) {
            for (const auto & cfg : grammar->configs) {
                const auto & rule = grammar->compiled->rules[cfg.current.rule_id];
                const auto & alt = rule.alternates[cfg.current.alternate_idx];
                if (cfg.current.segment_idx >= alt.size()) continue;
                const auto & seg = alt[cfg.current.segment_idx];
                if (seg.type != compiled_segment::DFA_MATCH) continue;
                if (seg.id >= grammar->compiled->dfa_candidates.size()) continue;
                const auto & state_cands = grammar->compiled->dfa_candidates[seg.id];
                if (cfg.current.dfa_state >= state_cands.states.size()) continue;
                const auto & ts = state_cands.states[cfg.current.dfa_state];
                any_config_matched = true;

                for (int32_t tok : ts.context_set) {
                    if (tok >= 0 && tok < n_vocab) {
                        is_context[tok] = true;
                    }
                }

                if (ts.accept_heavy) {
                    size_t ti = 0;
                    for (int32_t t = 0; t < n_vocab; t++) {
                        if (ti < ts.token_set.size() && ts.token_set[ti] == t) {
                            ti++;
                        } else {
                            rejected_by_all[t] = false;
                        }
                    }
                } else {
                    for (int32_t tok : ts.token_set) {
                        if (tok >= 0 && tok < n_vocab) {
                            rejected_by_all[tok] = false;
                        }
                    }
                    for (int32_t tok : ts.context_set) {
                        if (tok >= 0 && tok < n_vocab) {
                            rejected_by_all[tok] = false;
                        }
                    }
                }
            }
        }
        if (!have_candidates || !any_config_matched) {
            std::fill(rejected_by_all.begin(), rejected_by_all.end(), false);
            std::fill(is_context.begin(), is_context.end(), true);
        }
    }

    for (int32_t i = 0; i < n_vocab; i++) {
        if (llama_vocab_is_eog(vocab, i)) {
            cnt_eog++;
        } else {
            const std::string piece = common_token_to_piece(vocab, i, true);
            if (piece.empty() || piece[0] == 0) {
                cnt_empty++;
            } else if (rejected_by_all[i]) {
                cnt_precomp_reject++;
            } else if (!is_context[i]) {
                cnt_precomp_accept++;
            } else if (token_data[i].logit == -INFINITY) {
                cnt_sim_reject++;
            } else {
                cnt_sim_accept++;
            }
        }
    }

    // Print report
    printf("Grammar: %zu rules, %zu DFAs (%zu states total) | Vocab: %d tokens\n\n",
           grammar->compiled->rules.size(),
           grammar->compiled->dfas.size(),
           total_states, n_vocab);

    if (!state_prefix.empty()) {
        printf("  State prefix: \"%s\" (%zu configs after advance)\n\n",
               escape_for_display(state_prefix).c_str(), grammar->configs.size());
    }

    printf("  Parse:              %8.2f ms\n", parse_ms);
    printf("  DFA compile:        %8.2f ms  (%zu DFAs, %zu states)\n",
           compile_ms, cg->dfas.size(), total_states);
    printf("  Init (trie+precomp):%8.2f ms\n\n", init_ms);

    printf("  Logit masking (%d iters):\n", bench_iters);
    printf("    Mean:   %.3f ms/call\n", stats.mean_ms);
    printf("    Min:    %.3f ms   Max: %.3f ms   Stddev: %.3f ms\n",
           stats.min_ms, stats.max_ms, stats.stddev_ms);
    printf("    Breakdown:\n");
    printf("      Precomp-rejected: %6d tokens (%5.1f%%)\n",
           cnt_precomp_reject, 100.0 * cnt_precomp_reject / n_vocab);
    printf("      Precomp-accepted: %6d tokens (%5.1f%%)\n",
           cnt_precomp_accept, 100.0 * cnt_precomp_accept / n_vocab);
    int cnt_simulated = cnt_sim_accept + cnt_sim_reject;
    printf("      Simulated:        %6d tokens (%5.1f%%)\n",
           cnt_simulated, 100.0 * cnt_simulated / n_vocab);
    printf("        Sim-accepted:   %6d tokens\n", cnt_sim_accept);
    printf("        Sim-rejected:   %6d tokens\n", cnt_sim_reject);
    printf("      EOG:              %6d tokens\n", cnt_eog);
    printf("      Empty/null:       %6d tokens\n", cnt_empty);

    llama_grammar_free_impl(grammar);
    return 0;
}

// ============================================================
// Mode: candidates
// ============================================================

static int mode_candidates(const llama_grammar & grammar, const llama_vocab * vocab) {
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const auto & cg = *grammar.compiled;

    size_t total_states = 0;
    size_t total_accept_heavy = 0;
    size_t total_reject_heavy = 0;
    size_t total_candidates = 0;
    size_t total_memory = 0;

    for (size_t dfa_idx = 0; dfa_idx < cg.dfas.size(); dfa_idx++) {
        const auto & dfa = cg.dfas[dfa_idx];

        if (dfa_idx >= cg.dfa_candidates.size()) {
            printf("DFA %zu: no candidate data\n", dfa_idx);
            continue;
        }

        const auto & dc = cg.dfa_candidates[dfa_idx];
        printf("DFA %zu: %zu states\n", dfa_idx, dfa.transitions.size());

        for (size_t st = 0; st < dc.states.size(); st++) {
            const auto & ts = dc.states[st];
            bool is_accept = st < dfa.accept.size() && dfa.accept[st];

            int32_t n_guaranteed;
            if (ts.accept_heavy) {
                // token_set is reject set
                n_guaranteed = n_vocab - (int32_t)ts.token_set.size() - (int32_t)ts.context_set.size();
            } else {
                // token_set is accept set
                n_guaranteed = (int32_t)ts.token_set.size();
            }

            const char * repr = ts.accept_heavy ? "accept-heavy" : "reject-heavy";
            const char * set_type = ts.accept_heavy ? "rejects" : "accepts";

            printf("  State %3zu: %s, token_set=%zu (%s), guaranteed=%d, context=%zu%s\n",
                   st, repr, ts.token_set.size(), set_type, n_guaranteed,
                   ts.context_set.size(),
                   is_accept ? " [accept]" : "");

            // Show up to 5 sample tokens from the set
            if (!ts.token_set.empty()) {
                printf("           token samples: ");
                size_t show = std::min((size_t)5, ts.token_set.size());
                for (size_t j = 0; j < show; j++) {
                    int32_t tok = ts.token_set[j];
                    std::string piece = common_token_to_piece(vocab, tok, true);
                    printf("%d:\"%s\"", tok, escape_for_display(piece).c_str());
                    if (j + 1 < show) printf(", ");
                }
                if (ts.token_set.size() > show) {
                    printf(", ... (%zu more)", ts.token_set.size() - show);
                }
                printf("\n");
            }
            if (!ts.context_set.empty()) {
                printf("           context samples: ");
                size_t show = std::min((size_t)5, ts.context_set.size());
                for (size_t j = 0; j < show; j++) {
                    int32_t tok = ts.context_set[j];
                    std::string piece = common_token_to_piece(vocab, tok, true);
                    printf("%d:\"%s\"", tok, escape_for_display(piece).c_str());
                    if (j + 1 < show) printf(", ");
                }
                if (ts.context_set.size() > show) {
                    printf(", ... (%zu more)", ts.context_set.size() - show);
                }
                printf("\n");
            }

            total_states++;
            if (ts.accept_heavy) total_accept_heavy++;
            else total_reject_heavy++;
            total_candidates += n_guaranteed + (int32_t)ts.context_set.size();
            total_memory += (ts.token_set.size() + ts.context_set.size()) * sizeof(int32_t);
        }
        printf("\n");
    }

    printf("Totals:\n");
    printf("  States: %zu across %zu DFAs\n", total_states, cg.dfas.size());
    if (total_states > 0) {
        printf("  Avg candidates/state: %zu\n", total_candidates / total_states);
    }
    printf("  Memory: candidate sets ~%.1f KB\n", (double)total_memory / 1024.0);
    printf("  Accept-heavy states: %zu (%.1f%%)\n",
           total_accept_heavy,
           total_states > 0 ? 100.0 * total_accept_heavy / total_states : 0.0);
    printf("  Reject-heavy states: %zu (%.1f%%)\n",
           total_reject_heavy,
           total_states > 0 ? 100.0 * total_reject_heavy / total_states : 0.0);

    return 0;
}

// ============================================================
// Mode: masking-report
// ============================================================

static int mode_masking_report(const llama_grammar & grammar, const llama_vocab * vocab) {
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const auto & cg = *grammar.compiled;

    bool allow_eog = cg.any_config_complete(grammar.configs);

    // Build rejected_by_all + is_context bitmasks (replicate apply_impl logic)
    const bool have_candidates = !cg.dfa_candidates.empty();
    std::vector<bool> rejected_by_all(n_vocab, true);
    std::vector<bool> is_context(n_vocab, false);

    if (have_candidates) {
        bool any_config_matched = false;

        for (const auto & cfg : grammar.configs) {
            const auto & rule = cg.rules[cfg.current.rule_id];
            const auto & alt = rule.alternates[cfg.current.alternate_idx];
            if (cfg.current.segment_idx >= alt.size()) continue;
            const auto & seg = alt[cfg.current.segment_idx];
            if (seg.type != compiled_segment::DFA_MATCH) continue;
            if (seg.id >= cg.dfa_candidates.size()) continue;
            const auto & state_cands = cg.dfa_candidates[seg.id];
            if (cfg.current.dfa_state >= state_cands.states.size()) continue;
            const auto & ts = state_cands.states[cfg.current.dfa_state];
            any_config_matched = true;

            for (int32_t tok : ts.context_set) {
                if (tok >= 0 && tok < n_vocab) {
                    is_context[tok] = true;
                }
            }

            if (ts.accept_heavy) {
                size_t ti = 0;
                for (int32_t t = 0; t < n_vocab; t++) {
                    if (ti < ts.token_set.size() && ts.token_set[ti] == t) {
                        ti++;
                    } else {
                        rejected_by_all[t] = false;
                    }
                }
            } else {
                for (int32_t tok : ts.token_set) {
                    if (tok >= 0 && tok < n_vocab) {
                        rejected_by_all[tok] = false;
                    }
                }
                for (int32_t tok : ts.context_set) {
                    if (tok >= 0 && tok < n_vocab) {
                        rejected_by_all[tok] = false;
                    }
                }
            }
        }
        if (!any_config_matched) {
            std::fill(rejected_by_all.begin(), rejected_by_all.end(), false);
            std::fill(is_context.begin(), is_context.end(), true);
        }
    } else {
        std::fill(rejected_by_all.begin(), rejected_by_all.end(), false);
        std::fill(is_context.begin(), is_context.end(), true);
    }

    // Categorize each token
    int cnt_eog_accept = 0, cnt_eog_reject = 0;
    int cnt_empty = 0, cnt_precomp_reject = 0, cnt_precomp_accept = 0;
    int cnt_sim_accept = 0, cnt_sim_reject = 0;

    printf("TOKEN_ID\tCATEGORY\tPIECE\n");

    for (int32_t id = 0; id < n_vocab; id++) {
        const char * category;

        if (llama_vocab_is_eog(vocab, id)) {
            if (allow_eog) {
                category = "eog-accept";
                cnt_eog_accept++;
            } else {
                category = "eog-reject";
                cnt_eog_reject++;
            }
        } else {
            std::string piece = common_token_to_piece(vocab, id, true);
            if (piece.empty() || piece[0] == 0) {
                category = "empty-reject";
                cnt_empty++;
            } else if (rejected_by_all[id]) {
                category = "precomp-reject";
                cnt_precomp_reject++;
            } else if (!is_context[id]) {
                category = "precomp-accept";
                cnt_precomp_accept++;
            } else {
                // Context-dependent — simulate
                auto sim_configs = grammar.configs;
                bool valid = true;
                for (size_t j = 0; j < piece.size(); j++) {
                    cg.accept_byte(sim_configs, (uint8_t)piece[j]);
                    if (sim_configs.empty()) {
                        valid = false;
                        break;
                    }
                }
                if (valid) {
                    category = "sim-accept";
                    cnt_sim_accept++;
                } else {
                    category = "sim-reject";
                    cnt_sim_reject++;
                }
            }
        }

        std::string piece = common_token_to_piece(vocab, id, true);
        printf("%d\t%s\t%s\n", id, category, escape_for_display(piece).c_str());
    }

    printf("\n# Summary:\n");
    printf("#   eog-accept:     %6d\n", cnt_eog_accept);
    printf("#   eog-reject:     %6d\n", cnt_eog_reject);
    printf("#   empty-reject:   %6d\n", cnt_empty);
    printf("#   precomp-reject: %6d\n", cnt_precomp_reject);
    printf("#   precomp-accept: %6d\n", cnt_precomp_accept);
    printf("#   sim-accept:     %6d\n", cnt_sim_accept);
    printf("#   sim-reject:     %6d\n", cnt_sim_reject);
    printf("#   total:          %6d\n", n_vocab);

    return 0;
}

// ============================================================
// Mode: graph
// ============================================================

// Escape a string for use inside a graphviz HTML label
static std::string dot_html_escape(const std::string & s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '<': out += "&lt;";  break;
            case '>': out += "&gt;";  break;
            case '&': out += "&amp;"; break;
            case '"': out += "&quot;"; break;
            default:  out += c;       break;
        }
    }
    return out;
}

// Format a single byte for display in a graph label
static std::string dot_format_byte(uint8_t b) {
    if (b >= 0x21 && b <= 0x7e) {
        switch (b) {
            case '<': return "&lt;";
            case '>': return "&gt;";
            case '&': return "&amp;";
            case '"': return "&quot;";
            case '\\': return "\\\\";
            default: return std::string(1, (char)b);
        }
    }
    switch (b) {
        case ' ':  return "SP";
        case '\t': return "\\t";
        case '\n': return "\\n";
        case '\r': return "\\r";
        case 0:    return "NUL";
        default: {
            char buf[8];
            snprintf(buf, sizeof(buf), "x%02X", b);
            return buf;
        }
    }
}

struct byte_range {
    uint8_t lo, hi;
};

static std::vector<byte_range> compress_byte_ranges(const std::vector<uint8_t> & bytes) {
    std::vector<byte_range> ranges;
    if (bytes.empty()) return ranges;
    uint8_t lo = bytes[0], hi = bytes[0];
    for (size_t i = 1; i < bytes.size(); i++) {
        if (bytes[i] == hi + 1) {
            hi = bytes[i];
        } else {
            ranges.push_back({lo, hi});
            lo = hi = bytes[i];
        }
    }
    ranges.push_back({lo, hi});
    return ranges;
}

static std::string format_ranges_label(const std::vector<byte_range> & ranges, size_t total_bytes) {
    // If nearly all bytes match, show as wildcard
    if (total_bytes >= 250) {
        char buf[32];
        snprintf(buf, sizeof(buf), "* (%zu bytes)", total_bytes);
        return buf;
    }

    std::string s;
    size_t max_ranges = 6;
    for (size_t i = 0; i < ranges.size() && i < max_ranges; i++) {
        if (!s.empty()) s += ", ";
        if (ranges[i].lo == ranges[i].hi) {
            s += dot_format_byte(ranges[i].lo);
        } else {
            s += dot_format_byte(ranges[i].lo) + "-" + dot_format_byte(ranges[i].hi);
        }
    }
    if (ranges.size() > max_ranges) {
        char buf[32];
        snprintf(buf, sizeof(buf), " +%zu more", ranges.size() - max_ranges);
        s += buf;
    }
    return s;
}

static int mode_graph(const std::string & grammar_str, const char * grammar_root) {
    // Parse grammar to get rule names
    llama_grammar_parser parser;
    if (!parser.parse(grammar_str.c_str())) {
        fprintf(stderr, "error: grammar parse failed\n");
        return 1;
    }
    if (parser.symbol_ids.find(grammar_root) == parser.symbol_ids.end()) {
        fprintf(stderr, "error: grammar does not contain '%s' rule\n", grammar_root);
        return 1;
    }

    // Build id -> name map
    std::map<uint32_t, std::string> rule_names;
    for (const auto & kv : parser.symbol_ids) {
        rule_names[kv.second] = kv.first;
    }

    // Compile
    uint32_t start_idx = parser.symbol_ids.at(grammar_root);
    auto cg = compiled_grammar::compile(parser.rules, start_idx);

    // Output dot
    printf("digraph grammar {\n");
    printf("    rankdir=LR;\n");
    printf("    compound=true;\n");
    printf("    fontname=\"Helvetica\";\n");
    printf("    node [fontname=\"Helvetica\", fontsize=10];\n");
    printf("    edge [fontname=\"Helvetica\", fontsize=9];\n");
    printf("\n");

    // --- Rule call graph ---
    printf("    subgraph cluster_rules {\n");
    printf("        label=<<B>Grammar Rules</B>>;\n");
    printf("        style=\"rounded,filled\"; fillcolor=\"#f0f0f0\";\n");
    printf("        node [shape=box, style=\"rounded,filled\", fillcolor=white];\n");
    printf("\n");

    for (size_t r = 0; r < cg.rules.size(); r++) {
        const auto & rule = cg.rules[r];
        std::string name = rule_names.count((uint32_t)r) ? rule_names[(uint32_t)r]
                                                          : "rule_" + std::to_string(r);
        bool is_start = (r == start_idx);

        // Build label showing alternates
        std::string label = "<B>" + dot_html_escape(name) + "</B>";
        if (is_start) label += " (start)";

        for (size_t a = 0; a < rule.alternates.size(); a++) {
            label += "<BR/><FONT POINT-SIZE=\"8\" COLOR=\"#666666\">alt " + std::to_string(a) + ": ";
            const auto & alt = rule.alternates[a];
            for (size_t s = 0; s < alt.size(); s++) {
                if (s > 0) label += " &#8594; ";
                if (alt[s].type == compiled_segment::DFA_MATCH) {
                    label += "DFA[" + std::to_string(alt[s].id) + "]";
                } else {
                    std::string callee = rule_names.count(alt[s].id)
                                         ? dot_html_escape(rule_names[alt[s].id])
                                         : "rule_" + std::to_string(alt[s].id);
                    label += "<I>" + callee + "</I>";
                }
            }
            if (alt.empty()) label += "(empty)";
            label += "</FONT>";
        }

        const char * color = is_start ? " color=\"#2060c0\" penwidth=2" : "";
        printf("        rule_%zu [label=<%s>%s];\n", r, label.c_str(), color);
    }

    // Rule call edges
    printf("\n");
    for (size_t r = 0; r < cg.rules.size(); r++) {
        const auto & rule = cg.rules[r];
        for (size_t a = 0; a < rule.alternates.size(); a++) {
            for (const auto & seg : rule.alternates[a]) {
                if (seg.type == compiled_segment::RULE_CALL) {
                    printf("        rule_%zu -> rule_%u [color=\"#888888\"];\n",
                           r, seg.id);
                }
            }
        }
    }

    printf("    }\n\n");

    // --- DFA subgraphs ---
    for (size_t d = 0; d < cg.dfas.size(); d++) {
        const auto & dfa = cg.dfas[d];
        size_t n_states = dfa.transitions.size();

        printf("    subgraph cluster_dfa_%zu {\n", d);
        printf("        label=<<B>DFA %zu</B> (%zu states)>;\n", d, n_states);
        printf("        style=\"rounded,dashed\"; color=\"#999999\";\n");
        printf("        node [shape=circle, width=0.35, fixedsize=true, fontsize=8];\n");
        printf("\n");

        for (size_t s = 0; s < n_states; s++) {
            if (s == 0) {
                // Dead state - skip
                continue;
            }
            bool is_accept = s < dfa.accept.size() && dfa.accept[s];
            bool is_start = (s == (size_t)dfa.start_state);

            const char * shape = is_accept ? "doublecircle" : "circle";
            std::string style;
            if (is_start) style = " style=bold";

            printf("        dfa%zu_s%zu [label=\"%zu\", shape=%s%s];\n",
                   d, s, s, shape, style.c_str());
        }

        // Transitions - group by (source, target), compress byte ranges
        printf("\n");
        for (size_t s = 1; s < n_states; s++) {
            // Collect bytes per target
            std::map<uint16_t, std::vector<uint8_t>> target_bytes;
            for (int b = 0; b < 256; b++) {
                uint16_t target = dfa.transitions[s][b];
                if (target != 0) {
                    target_bytes[target].push_back((uint8_t)b);
                }
            }

            for (const auto & kv : target_bytes) {
                uint16_t target = kv.first;
                const auto & bytes = kv.second;
                auto ranges = compress_byte_ranges(bytes);
                std::string label = format_ranges_label(ranges, bytes.size());
                printf("        dfa%zu_s%zu -> dfa%zu_s%u [label=<%s>];\n",
                       d, s, d, target, label.c_str());
            }
        }

        printf("    }\n\n");
    }

    printf("}\n");
    return 0;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char ** argv) {
    // Args
    const char * model_path = nullptr;
    const char * grammar_path = nullptr;
    const char * test_file_path = nullptr;
    const char * grammar_root = "root";
    std::string state_prefix;
    int bench_iters = 1000;
    bool disable_logging = false;

    enum mode_t { MODE_VALIDATE, MODE_BENCH, MODE_CANDIDATES, MODE_MASKING, MODE_GRAPH };
    mode_t mode = MODE_VALIDATE;

    std::vector<test_string> tests;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) { fprintf(stderr, "error: %s requires argument\n", arg.c_str()); return 1; }
            model_path = argv[i];
        } else if (arg == "-g" || arg == "--grammar-file") {
            if (++i >= argc) { fprintf(stderr, "error: %s requires argument\n", arg.c_str()); return 1; }
            grammar_path = argv[i];
        } else if (arg == "--validate") {
            mode = MODE_VALIDATE;
        } else if (arg == "--bench") {
            mode = MODE_BENCH;
        } else if (arg == "--candidates") {
            mode = MODE_CANDIDATES;
        } else if (arg == "--masking-report") {
            mode = MODE_MASKING;
        } else if (arg == "--graph") {
            mode = MODE_GRAPH;
        } else if (arg == "--test-file") {
            if (++i >= argc) { fprintf(stderr, "error: %s requires argument\n", arg.c_str()); return 1; }
            test_file_path = argv[i];
        } else if (arg == "--test-pass") {
            if (++i >= argc) { fprintf(stderr, "error: %s requires argument\n", arg.c_str()); return 1; }
            tests.push_back({argv[i], true});
        } else if (arg == "--test-fail") {
            if (++i >= argc) { fprintf(stderr, "error: %s requires argument\n", arg.c_str()); return 1; }
            tests.push_back({argv[i], false});
        } else if (arg == "--bench-iters") {
            if (++i >= argc) { fprintf(stderr, "error: %s requires argument\n", arg.c_str()); return 1; }
            bench_iters = std::atoi(argv[i]);
            if (bench_iters <= 0) { fprintf(stderr, "error: bench-iters must be > 0\n"); return 1; }
        } else if (arg == "--state") {
            if (++i >= argc) { fprintf(stderr, "error: %s requires argument\n", arg.c_str()); return 1; }
            state_prefix = argv[i];
        } else if (arg == "--grammar-root") {
            if (++i >= argc) { fprintf(stderr, "error: %s requires argument\n", arg.c_str()); return 1; }
            grammar_root = argv[i];
        } else if (arg == "--log-disable") {
            disable_logging = true;
        } else {
            fprintf(stderr, "error: unknown option '%s'\n", argv[i]);
            return 1;
        }
    }

    if (!model_path && mode != MODE_GRAPH) {
        fprintf(stderr, "error: --model is required for this mode\n");
        return 1;
    }
    if (!grammar_path) {
        fprintf(stderr, "error: --grammar-file is required\n");
        return 1;
    }

    // Load test file
    if (test_file_path) {
        auto file_tests = load_test_file(test_file_path);
        tests.insert(tests.end(), file_tests.begin(), file_tests.end());
    }

    // Read grammar
    std::string grammar_str = read_file(grammar_path);
    if (grammar_str.empty()) {
        fprintf(stderr, "error: grammar file is empty or could not be read\n");
        return 1;
    }

    // Graph mode: no model needed, just parse + compile
    if (mode == MODE_GRAPH) {
        return mode_graph(grammar_str, grammar_root);
    }

    // For bench mode, we do our own step-by-step timing
    if (mode == MODE_BENCH) {
        if (disable_logging) {
            llama_log_set(llama_log_callback_null, nullptr);
        }
        llama_backend_init();

        llama_model_params mparams = llama_model_default_params();
        mparams.vocab_only = true;
        llama_model * model = llama_model_load_from_file(model_path, mparams);
        if (!model) {
            fprintf(stderr, "error: could not load model\n");
            return 1;
        }
        const llama_vocab * vocab = llama_model_get_vocab(model);

        int rc = mode_bench(vocab, grammar_str, grammar_root, state_prefix, bench_iters);

        llama_model_free(model);
        llama_backend_free();
        return rc;
    }

    // All other modes: init grammar via init_impl
    if (disable_logging) {
        llama_log_set(llama_log_callback_null, nullptr);
    }
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "error: could not load model\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_grammar * grammar = llama_grammar_init_impl(
        vocab, grammar_str.c_str(), grammar_root, false, nullptr, 0, nullptr, 0);
    if (!grammar) {
        fprintf(stderr, "error: grammar init failed\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Advance state if prefix given
    if (!state_prefix.empty()) {
        for (size_t i = 0; i < state_prefix.size(); i++) {
            uint8_t b = (uint8_t)state_prefix[i];
            grammar->compiled->accept_byte(grammar->configs, b);
            if (grammar->configs.empty()) {
                fprintf(stderr, "error: grammar died advancing --state prefix at byte %zu (0x%02x)\n", i, b);
                llama_grammar_free_impl(grammar);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
        }
        fprintf(stderr, "Advanced grammar by %zu-byte prefix, %zu configs active\n",
                state_prefix.size(), grammar->configs.size());
    }

    int rc = 0;
    switch (mode) {
        case MODE_VALIDATE:
            rc = mode_validate(*grammar, tests);
            break;
        case MODE_CANDIDATES:
            rc = mode_candidates(*grammar, vocab);
            break;
        case MODE_MASKING:
            rc = mode_masking_report(*grammar, vocab);
            break;
        default:
            break;
    }

    llama_grammar_free_impl(grammar);
    llama_model_free(model);
    llama_backend_free();
    return rc;
}
