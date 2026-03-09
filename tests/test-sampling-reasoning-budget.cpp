#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

#include "peg-parser/testing.h"
#include "sampling-reasoning-budget.h"

// Helper: build a token data array from a list of token ids, all with logit 0.0
static std::vector<llama_token_data> make_candidates(const std::vector<llama_token> & tokens) {
    std::vector<llama_token_data> data;
    data.reserve(tokens.size());
    for (auto id : tokens) {
        data.push_back({id, 0.0f, 0.0f});
    }
    return data;
}

static llama_token_data_array make_cur_p(std::vector<llama_token_data> & data) {
    return { data.data(), data.size(), -1, false };
}

// Check whether a token's logit was set to -inf
static bool is_blocked(const llama_token_data_array & cur_p, llama_token id) {
    for (size_t i = 0; i < cur_p.size; i++) {
        if (cur_p.data[i].id == id) {
            return cur_p.data[i].logit == -INFINITY;
        }
    }
    return false;
}

static bool is_allowed(const llama_token_data_array & cur_p, llama_token id) {
    return !is_blocked(cur_p, id);
}

static constexpr llama_token END_TOKEN = 99;
static constexpr llama_token TOKEN_A   = 1;
static constexpr llama_token TOKEN_B   = 2;
static constexpr llama_token TOKEN_C   = 3;

static const std::vector<llama_token> ALL_TOKENS = { TOKEN_A, TOKEN_B, TOKEN_C, END_TOKEN };

static void test_passthrough_before_budget(testing & t) {
    t.test("all tokens allowed before budget", [](testing & t) {
        auto * smpl = common_sampler_init_reasoning_budget(5, END_TOKEN);

        auto data  = make_candidates(ALL_TOKENS);
        auto cur_p = make_cur_p(data);

        llama_sampler_apply(smpl, &cur_p);

        t.assert_true("token A allowed", is_allowed(cur_p, TOKEN_A));
        t.assert_true("token B allowed", is_allowed(cur_p, TOKEN_B));
        t.assert_true("token C allowed", is_allowed(cur_p, TOKEN_C));
        t.assert_true("end token allowed", is_allowed(cur_p, END_TOKEN));

        llama_sampler_free(smpl);
    });

    t.test("tokens allowed while counting", [](testing & t) {
        auto * smpl = common_sampler_init_reasoning_budget(5, END_TOKEN);

        // Accept 3 tokens (under budget of 5)
        llama_sampler_accept(smpl, TOKEN_A);
        llama_sampler_accept(smpl, TOKEN_B);
        llama_sampler_accept(smpl, TOKEN_C);

        auto data  = make_candidates(ALL_TOKENS);
        auto cur_p = make_cur_p(data);

        llama_sampler_apply(smpl, &cur_p);

        t.assert_true("token A allowed", is_allowed(cur_p, TOKEN_A));
        t.assert_true("token B allowed", is_allowed(cur_p, TOKEN_B));
        t.assert_true("end token allowed", is_allowed(cur_p, END_TOKEN));

        llama_sampler_free(smpl);
    });
}

static void test_forcing_at_budget(testing & t) {
    t.test("only end token allowed when budget exhausted", [](testing & t) {
        auto * smpl = common_sampler_init_reasoning_budget(3, END_TOKEN);

        // Exhaust the budget
        llama_sampler_accept(smpl, TOKEN_A);
        llama_sampler_accept(smpl, TOKEN_B);
        llama_sampler_accept(smpl, TOKEN_C);

        auto data  = make_candidates(ALL_TOKENS);
        auto cur_p = make_cur_p(data);

        llama_sampler_apply(smpl, &cur_p);

        t.assert_true("token A blocked", is_blocked(cur_p, TOKEN_A));
        t.assert_true("token B blocked", is_blocked(cur_p, TOKEN_B));
        t.assert_true("token C blocked", is_blocked(cur_p, TOKEN_C));
        t.assert_true("end token allowed", is_allowed(cur_p, END_TOKEN));

        llama_sampler_free(smpl);
    });

    t.test("budget of 1", [](testing & t) {
        auto * smpl = common_sampler_init_reasoning_budget(1, END_TOKEN);

        llama_sampler_accept(smpl, TOKEN_A);

        auto data  = make_candidates(ALL_TOKENS);
        auto cur_p = make_cur_p(data);

        llama_sampler_apply(smpl, &cur_p);

        t.assert_true("token A blocked", is_blocked(cur_p, TOKEN_A));
        t.assert_true("end token allowed", is_allowed(cur_p, END_TOKEN));

        llama_sampler_free(smpl);
    });
}

static void test_passthrough_after_end(testing & t) {
    t.test("passthrough after end token during counting", [](testing & t) {
        auto * smpl = common_sampler_init_reasoning_budget(10, END_TOKEN);

        // Accept 2 tokens then the end token (under budget)
        llama_sampler_accept(smpl, TOKEN_A);
        llama_sampler_accept(smpl, TOKEN_B);
        llama_sampler_accept(smpl, END_TOKEN);

        // Now should be passthrough
        auto data  = make_candidates(ALL_TOKENS);
        auto cur_p = make_cur_p(data);

        llama_sampler_apply(smpl, &cur_p);

        t.assert_true("token A allowed", is_allowed(cur_p, TOKEN_A));
        t.assert_true("token B allowed", is_allowed(cur_p, TOKEN_B));
        t.assert_true("token C allowed", is_allowed(cur_p, TOKEN_C));
        t.assert_true("end token allowed", is_allowed(cur_p, END_TOKEN));

        llama_sampler_free(smpl);
    });

    t.test("passthrough after forced end token", [](testing & t) {
        auto * smpl = common_sampler_init_reasoning_budget(2, END_TOKEN);

        // Exhaust budget
        llama_sampler_accept(smpl, TOKEN_A);
        llama_sampler_accept(smpl, TOKEN_B);

        // Now in FORCING state, accept the end token
        llama_sampler_accept(smpl, END_TOKEN);

        // Should be passthrough now
        auto data  = make_candidates(ALL_TOKENS);
        auto cur_p = make_cur_p(data);

        llama_sampler_apply(smpl, &cur_p);

        t.assert_true("token A allowed", is_allowed(cur_p, TOKEN_A));
        t.assert_true("token B allowed", is_allowed(cur_p, TOKEN_B));
        t.assert_true("token C allowed", is_allowed(cur_p, TOKEN_C));
        t.assert_true("end token allowed", is_allowed(cur_p, END_TOKEN));

        llama_sampler_free(smpl);
    });

    t.test("passthrough persists across many tokens", [](testing & t) {
        auto * smpl = common_sampler_init_reasoning_budget(1, END_TOKEN);

        // Exhaust budget and force end
        llama_sampler_accept(smpl, TOKEN_A);
        llama_sampler_accept(smpl, END_TOKEN);

        // Accept many more tokens in passthrough
        for (int i = 0; i < 100; i++) {
            llama_sampler_accept(smpl, TOKEN_B);
        }

        auto data  = make_candidates(ALL_TOKENS);
        auto cur_p = make_cur_p(data);

        llama_sampler_apply(smpl, &cur_p);

        t.assert_true("token A allowed", is_allowed(cur_p, TOKEN_A));
        t.assert_true("token B allowed", is_allowed(cur_p, TOKEN_B));

        llama_sampler_free(smpl);
    });
}

static void test_reset(testing & t) {
    t.test("reset returns to counting state", [](testing & t) {
        auto * smpl = common_sampler_init_reasoning_budget(2, END_TOKEN);

        // Exhaust budget
        llama_sampler_accept(smpl, TOKEN_A);
        llama_sampler_accept(smpl, TOKEN_B);

        // Verify forcing
        {
            auto data  = make_candidates(ALL_TOKENS);
            auto cur_p = make_cur_p(data);
            llama_sampler_apply(smpl, &cur_p);
            t.assert_true("token A blocked before reset", is_blocked(cur_p, TOKEN_A));
        }

        // Reset
        llama_sampler_reset(smpl);

        // Should be back to counting, all tokens allowed
        {
            auto data  = make_candidates(ALL_TOKENS);
            auto cur_p = make_cur_p(data);
            llama_sampler_apply(smpl, &cur_p);
            t.assert_true("token A allowed after reset", is_allowed(cur_p, TOKEN_A));
            t.assert_true("token B allowed after reset", is_allowed(cur_p, TOKEN_B));
        }

        llama_sampler_free(smpl);
    });

    t.test("reset from passthrough returns to counting", [](testing & t) {
        auto * smpl = common_sampler_init_reasoning_budget(5, END_TOKEN);

        // Enter passthrough via early end token
        llama_sampler_accept(smpl, END_TOKEN);

        // Reset
        llama_sampler_reset(smpl);

        // Exhaust the budget again to prove we're counting
        for (int i = 0; i < 5; i++) {
            llama_sampler_accept(smpl, TOKEN_A);
        }

        auto data  = make_candidates(ALL_TOKENS);
        auto cur_p = make_cur_p(data);
        llama_sampler_apply(smpl, &cur_p);
        t.assert_true("forcing after reset and re-exhaust", is_blocked(cur_p, TOKEN_A));
        t.assert_true("end token allowed", is_allowed(cur_p, END_TOKEN));

        llama_sampler_free(smpl);
    });
}

static void test_clone(testing & t) {
    t.test("clone preserves config", [](testing & t) {
        auto * smpl = common_sampler_init_reasoning_budget(3, END_TOKEN);

        // Partially use the budget
        llama_sampler_accept(smpl, TOKEN_A);
        llama_sampler_accept(smpl, TOKEN_B);

        // Clone (clone starts fresh per the clone implementation)
        auto * cloned = llama_sampler_clone(smpl);

        // The clone should be in initial counting state with budget=3
        // Accept 3 tokens to exhaust its budget
        llama_sampler_accept(cloned, TOKEN_A);
        llama_sampler_accept(cloned, TOKEN_B);
        llama_sampler_accept(cloned, TOKEN_C);

        auto data  = make_candidates(ALL_TOKENS);
        auto cur_p = make_cur_p(data);
        llama_sampler_apply(cloned, &cur_p);

        t.assert_true("cloned: token A blocked", is_blocked(cur_p, TOKEN_A));
        t.assert_true("cloned: end token allowed", is_allowed(cur_p, END_TOKEN));

        llama_sampler_free(smpl);
        llama_sampler_free(cloned);
    });
}

static void test_name(testing & t) {
    t.test("sampler name", [](testing & t) {
        auto * smpl = common_sampler_init_reasoning_budget(10, END_TOKEN);

        t.assert_equal("name", std::string("reasoning-budget"), std::string(llama_sampler_name(smpl)));

        llama_sampler_free(smpl);
    });
}

static void test_zero_budget(testing & t) {
    t.test("zero budget forces immediately", [](testing & t) {
        auto * smpl = common_sampler_init_reasoning_budget(0, END_TOKEN);

        // Budget is 0 but no tokens have been accepted yet, so n_sampled (0) >= budget (0)
        // The state machine starts in COUNTING and the transition happens in accept().
        // Since we haven't accepted any token, apply should still show COUNTING.
        // But after accepting any token the budget check triggers.
        // Let's accept one token to trigger the transition.
        llama_sampler_accept(smpl, TOKEN_A);

        auto data  = make_candidates(ALL_TOKENS);
        auto cur_p = make_cur_p(data);
        llama_sampler_apply(smpl, &cur_p);

        t.assert_true("token A blocked", is_blocked(cur_p, TOKEN_A));
        t.assert_true("end token allowed", is_allowed(cur_p, END_TOKEN));

        llama_sampler_free(smpl);
    });
}

int main(int argc, char * argv[]) {
    testing t(std::cout);
    if (argc >= 2) {
        t.set_filter(argv[1]);
    }

    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    if (verbose) {
        t.verbose = std::string(verbose) == "1";
    }

    t.test("passthrough_before_budget", test_passthrough_before_budget);
    t.test("forcing_at_budget",         test_forcing_at_budget);
    t.test("passthrough_after_end",     test_passthrough_after_end);
    t.test("reset",                     test_reset);
    t.test("clone",                     test_clone);
    t.test("name",                      test_name);
    t.test("zero_budget",              test_zero_budget);

    return t.summary();
}
