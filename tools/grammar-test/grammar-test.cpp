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
    printf("Usage: %s -m MODEL -g GRAMMAR_FILE [options]\n\n", prog);
    printf("Test grammar correctness and profile logit-masking performance.\n");
    printf("The model is used only for its vocabulary (vocab_only load).\n\n");
    printf("Required:\n");
    printf("  -m, --model MODEL          Model file (vocab only)\n");
    printf("  -g, --grammar-file FILE    GBNF grammar file\n\n");
    printf("Modes (pick one, default: --validate):\n");
    printf("  --validate                 Validate test strings against grammar\n");
    printf("  --bench                    Benchmark grammar pipeline stages\n");
    printf("  --candidates               Dump candidate set analysis\n");
    printf("  --masking-report           Per-token accept/reject categorization\n\n");
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
    int cnt_precomp_reject = 0, cnt_sim_accept = 0, cnt_sim_reject = 0;
    int cnt_eog = 0, cnt_empty = 0;

    // Re-run once with fresh logits to count categories
    for (int32_t i = 0; i < n_vocab; i++) {
        token_data[i].logit = 0.0f;
    }
    llama_token_data_array cur_p = {token_data.data(), (size_t)n_vocab, -1, false};
    llama_grammar_apply_impl(*grammar, &cur_p);

    // Build needs_simulation the same way apply_impl does, to distinguish categories
    std::vector<bool> needs_sim(n_vocab, false);
    {
        const bool have_candidates = !grammar->compiled->dfa_candidates.empty();
        if (have_candidates) {
            std::vector<bool> rejected_by_all(n_vocab, true);
            bool any_config_matched = false;

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
                }
            }
            if (any_config_matched) {
                for (int32_t t = 0; t < n_vocab; t++) {
                    needs_sim[t] = !rejected_by_all[t];
                }
            }
        } else {
            std::fill(needs_sim.begin(), needs_sim.end(), true);
        }
    }

    for (int32_t i = 0; i < n_vocab; i++) {
        if (llama_vocab_is_eog(vocab, i)) {
            cnt_eog++;
        } else {
            const std::string piece = common_token_to_piece(vocab, i, true);
            if (piece.empty() || piece[0] == 0) {
                cnt_empty++;
            } else if (!needs_sim[i]) {
                cnt_precomp_reject++;
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

            int32_t n_candidates;
            if (ts.accept_heavy) {
                // token_set is reject set
                n_candidates = n_vocab - (int32_t)ts.token_set.size();
            } else {
                // token_set is accept set
                n_candidates = (int32_t)ts.token_set.size();
            }

            const char * repr = ts.accept_heavy ? "accept-heavy" : "reject-heavy";
            const char * set_type = ts.accept_heavy ? "rejects" : "accepts";

            printf("  State %3zu: %s, token_set=%zu (%s), candidates=%d%s\n",
                   st, repr, ts.token_set.size(), set_type, n_candidates,
                   is_accept ? " [accept]" : "");

            // Show up to 5 sample tokens from the set
            if (!ts.token_set.empty()) {
                printf("           samples: ");
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

            total_states++;
            if (ts.accept_heavy) total_accept_heavy++;
            else total_reject_heavy++;
            total_candidates += n_candidates;
            total_memory += ts.token_set.size() * sizeof(int32_t);
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

    // Build needs_simulation bitmask (replicate apply_impl logic)
    const bool have_candidates = !cg.dfa_candidates.empty();
    std::vector<bool> needs_sim(n_vocab, !have_candidates);

    if (have_candidates) {
        std::vector<bool> rejected_by_all(n_vocab, true);
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
            }
        }
        if (any_config_matched) {
            for (int32_t t = 0; t < n_vocab; t++) {
                needs_sim[t] = !rejected_by_all[t];
            }
        }
    }

    // Categorize each token
    int cnt_eog_accept = 0, cnt_eog_reject = 0;
    int cnt_empty = 0, cnt_precomp = 0, cnt_sim_accept = 0, cnt_sim_reject = 0;

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
            } else if (!needs_sim[id]) {
                category = "precomp-reject";
                cnt_precomp++;
            } else {
                // Simulate
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
    printf("#   precomp-reject: %6d\n", cnt_precomp);
    printf("#   sim-accept:     %6d\n", cnt_sim_accept);
    printf("#   sim-reject:     %6d\n", cnt_sim_reject);
    printf("#   total:          %6d\n", n_vocab);

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

    enum mode_t { MODE_VALIDATE, MODE_BENCH, MODE_CANDIDATES, MODE_MASKING };
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

    if (!model_path) {
        fprintf(stderr, "error: --model is required\n");
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
