#include "reasoning-budget.h"
#include "unicode.h"

#include "llama.h"
#include "ggml.h"

#include "testing.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

static std::string pos_str(size_t pos) {
    return pos == SIZE_MAX ? "never" : std::to_string(pos);
}

// Reasoning budget sampler test helper
// These tests use nullptr vocab which safely falls back to treating all tokens as complete
// (The UTF-8 boundary detection logic is tested separately in test_utf8_boundary_detection)
static void test_reasoning_budget(
    testing & t,
    const std::vector<llama_token> & sequence,
    const std::vector<llama_tokens> & start_seqs,
    const std::vector<llama_tokens> & end_seqs,
    const std::vector<llama_token> & forced_tokens,
    int32_t budget,
    common_reasoning_budget_state initial_state,
    size_t expected_force_start,   // token index where forcing should start (SIZE_MAX = never)
    size_t expected_force_end      // token index where forcing should end (after this, no more forcing)
) {
    // Find the maximum token ID to ensure our vocab covers all tokens
    llama_token max_token = 0;
    for (auto t : sequence) max_token = std::max(max_token, t);
    for (const auto & seq : start_seqs) {
        for (auto t : seq) max_token = std::max(max_token, t);
    }
    for (const auto & seq : end_seqs) {
        for (auto t : seq) max_token = std::max(max_token, t);
    }
    for (auto t : forced_tokens) max_token = std::max(max_token, t);

    // Create a minimal sampler with mock vocabulary
    // For this test, we use nullptr as vocab since we're testing state transitions
    // The UTF-8 boundary check will treat all tokens as complete (safe fallback)
    auto * sampler = common_reasoning_budget_init(
        nullptr,  // vocab - not used for basic state machine tests
        start_seqs,
        end_seqs,
        forced_tokens,
        budget,
        initial_state
    );

    // Create a test token data array for checking forcing behavior
    // Vocab size must be large enough to include all tokens (start, end, forced, sequence)
    std::vector<llama_token_data> cur;
    const size_t n_vocab = (size_t)max_token + 1;
    for (size_t i = 0; i < n_vocab; i++) {
        cur.emplace_back(llama_token_data{(llama_token)i, logf((float)(i+1)), 0.0f});
    }
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };

    size_t actual_force_start = SIZE_MAX;
    size_t actual_force_end = SIZE_MAX;

    // Feed the sequence and track when forcing occurs
    for (size_t i = 0; i < sequence.size(); i++) {
        // Check if we're in forcing state by applying and seeing if logits are modified
        cur_p.selected = -1;
        for (size_t j = 0; j < cur.size(); j++) {
            cur[j].logit = logf((float)(j+1));  // reset logits
        }

        llama_sampler_apply(sampler, &cur_p);

        // Check if forcing is active (all logits except one should be -INFINITY)
        size_t finite_count = 0;
        llama_token finite_token = -1;
        for (size_t j = 0; j < cur.size(); j++) {
            if (std::isfinite(cur[j].logit)) {
                finite_count++;
                finite_token = cur[j].id;
            }
        }

        llama_sampler_accept(sampler, sequence[i]);

        fprintf(stderr, "    i=%zu: token=%d, finite_count=%zu, finite_token=%d\n", i, (int)sequence[i], finite_count, (int)finite_token);

        if (finite_count == 1) {
            if (actual_force_start == SIZE_MAX) {
                actual_force_start = i;
            }
            actual_force_end = i;
        } else if (actual_force_start != SIZE_MAX && actual_force_end != SIZE_MAX) {
            // Forcing stopped
            break;
        }
    }

    llama_sampler_free(sampler);

    // Verify forcing occurred at expected positions
    t.assert_true("forcing starts at " + pos_str(expected_force_start) + ", actual " + pos_str(actual_force_start),
                  actual_force_start == expected_force_start);

    if (expected_force_end != SIZE_MAX) {
        t.assert_true("forcing ends at or after " + pos_str(expected_force_end) + ", actual " + pos_str(actual_force_end),
                      actual_force_end >= expected_force_end);
    }
}

static llama_token get_forced_token(testing & t, struct llama_sampler * sampler, llama_token max_token) {
    std::vector<llama_token_data> cur;
    const size_t n_vocab = (size_t) max_token + 1;
    for (size_t i = 0; i < n_vocab; i++) {
        cur.emplace_back(llama_token_data{(llama_token) i, logf((float) (i + 1)), 0.0f});
    }

    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(sampler, &cur_p);

    size_t finite_count = 0;
    llama_token finite_token = LLAMA_TOKEN_NULL;
    for (size_t i = 0; i < cur.size(); i++) {
        if (std::isfinite(cur[i].logit)) {
            finite_count++;
            finite_token = cur[i].id;
        }
    }

    t.assert_true("sampler forces exactly one token, " + std::to_string(finite_count) + " logits are finite", finite_count == 1);
    return finite_token;
}

static void test_sequences(testing & t) {
    // Test 1: Basic budget with start/end tokens - no forcing (natural end before budget exhausted)
    t.test("natural end before budget exhausted", [](testing & t) {
        const std::vector<llama_token> start = {100};  // start token
        const std::vector<llama_token> end = {101};    // end token
        const std::vector<llama_token> forced = {102}; // forced token (not used in this test)
        const std::vector<llama_token> sequence = {100, 50, 51, 101, 52}; // start, two tokens, end, one more

        test_reasoning_budget(t, sequence, {start}, {end}, forced,
            5,      // budget of 5 tokens
            REASONING_BUDGET_IDLE,
            SIZE_MAX, SIZE_MAX); // no forcing expected (natural end)
    });

    // Test 2: Budget exhausted, forcing should occur
    // Flow: i=0 apply()->passthrough, accept(100)->COUNTING; i=1 accept(50)->remaining=1
    // i=2 accept(51)->remaining=0->FORCING; i=3 apply() forces token[0]; i=4 apply() forces token[1]
    // At i=4, accept() advances force_pos to 2 which equals forced_tokens.size(), so state becomes DONE
    t.test("budget exhausted forcing", [](testing & t) {
        const std::vector<llama_token> start = {100};
        const std::vector<llama_token> end = {101};
        const std::vector<llama_token> forced = {102, 101}; // forced message + end
        const std::vector<llama_token> sequence = {100, 50, 51, 52, 53}; // start + 4 tokens (budget=2)

        test_reasoning_budget(t, sequence, {start}, {end}, forced,
            2,      // budget of 2 tokens
            REASONING_BUDGET_IDLE,
            3,      // forcing starts at i=3 (accept at i=2 depletes budget, apply at i=3 forces)
            4);     // forcing continues through i=4 (accept at i=4 transitions to DONE)
    });

    // Test 3: Activate immediately with budget=0, forcing should start right away
    // Flow: init promotes COUNTING+budget=0 to FORCING, so apply() sees FORCING at i=0
    t.test("activate immediately budget=0", [](testing & t) {
        const std::vector<llama_token> start = {100};
        const std::vector<llama_token> end = {101};
        const std::vector<llama_token> forced = {102, 101};
        const std::vector<llama_token> sequence = {100, 50, 51, 52}; // start token first, then 3 tokens

        test_reasoning_budget(t, sequence, {start}, {end}, forced,
            0,      // budget of 0 tokens
            REASONING_BUDGET_COUNTING, // starts counting, promoted to FORCING since budget=0
            0,      // forcing starts at i=0 (initialized in FORCING, apply forces immediately)
            1);     // forcing continues through i=1 (accept at i=1 transitions to DONE)
    });

    // Test 4: No start/end tokens configured - passthrough (no forcing)
    t.test("no start/end configured", [](testing & t) {
        const std::vector<llama_token> start = {};
        const std::vector<llama_token> end = {};
        const std::vector<llama_token> forced = {102};
        const std::vector<llama_token> sequence = {50, 51, 52, 53};

        test_reasoning_budget(t, sequence, {start}, {end}, forced,
            2,      // budget
            REASONING_BUDGET_IDLE,
            SIZE_MAX, SIZE_MAX); // no forcing (no start/end configured)
    });

    // Test 5: Activate immediately with budget > 0, count down then force
    // Flow: i=0 accept(50)->remaining=1, i=1 accept(51)->remaining=0->FORCING
    // Forcing starts at i=2 (apply sees FORCING after accept at i=1 transitioned)
    t.test("activate immediately with budget", [](testing & t) {
        const std::vector<llama_token> start = {100};
        const std::vector<llama_token> end = {101};
        const std::vector<llama_token> forced = {102, 101};
        const std::vector<llama_token> sequence = {50, 51, 52, 53};

        test_reasoning_budget(t, sequence, {start}, {end}, forced,
            2,      // budget of 2 tokens
            REASONING_BUDGET_COUNTING,
            2,      // forcing starts at i=2 (after 2 accepts deplete budget, apply at i=2 forces)
            3);     // forcing continues through i=3
    });

    // Test 6: Multi-block thinking. First block ends naturally at i=2, second
    // start tag at i=3 re-arms the budget, which then exhausts at i=5.
    // Regression: before this fix, DONE absorbed all subsequent tokens and a
    // second <think> block ran unbudgeted.
    // Flow: i=0 accept(100)->COUNTING rem=2; i=1 accept(50)->rem=1;
    //       i=2 accept(101)->end_matcher matches, DONE;
    //       i=3 accept(100)->re-arm, COUNTING rem=2;
    //       i=4 accept(60)->rem=1; i=5 accept(61)->rem=0->FORCING;
    //       i=6 apply()->forces token[0]=102, accept(62)->force_pos=1, stay FORCING;
    //       i=7 apply()->forces token[1]=101, accept(63)->force_pos=2->DONE.
    t.test("multi-block re-arms budget after DONE", [](testing & t) {
        const std::vector<llama_token> start = {100};
        const std::vector<llama_token> end = {101};
        const std::vector<llama_token> forced = {102, 101};
        const std::vector<llama_token> sequence = {100, 50, 101, 100, 60, 61, 62, 63};

        test_reasoning_budget(t, sequence, {start}, {end}, forced,
            2,      // budget of 2 tokens (per block)
            REASONING_BUDGET_IDLE,
            6,      // forcing starts at i=6 (after second block exhausts at i=5)
            7);     // forcing continues through i=7
    });

    // Test 7: Multiple start sequences - the second sequence activates counting
    // Flow: i=0 accept(110), i=1 accept(111)->COUNTING rem=2; i=2 accept(50)->rem=1;
    //       i=3 accept(51)->rem=0->FORCING; i=4..5 apply() forces the end sequence
    t.test("multiple start sequences", [](testing & t) {
        const std::vector<llama_tokens> start = {{100}, {110, 111}};
        const std::vector<llama_tokens> end = {{101}};
        const std::vector<llama_token> forced = {102, 101};
        const std::vector<llama_token> sequence = {110, 111, 50, 51, 52, 53};

        test_reasoning_budget(t, sequence, start, end, forced,
            2,      // budget of 2 tokens
            REASONING_BUDGET_IDLE,
            4,      // forcing starts at i=4 (accept at i=3 depletes budget)
            5);     // forcing continues through i=5
    });

    // Test 8: Multiple end sequences - natural end via the second sequence
    // Flow: i=0 accept(100)->COUNTING rem=5; i=1 accept(50)->rem=4;
    //       i=2 accept(103)->partial end, rem=3; i=3 accept(104)->end matched, DONE
    t.test("multiple end sequences", [](testing & t) {
        const std::vector<llama_tokens> start = {{100}};
        const std::vector<llama_tokens> end = {{101}, {103, 104}};
        const std::vector<llama_token> forced = {102, 101};
        const std::vector<llama_token> sequence = {100, 50, 103, 104, 52};

        test_reasoning_budget(t, sequence, start, end, forced,
            5,      // budget of 5 tokens
            REASONING_BUDGET_IDLE,
            SIZE_MAX, SIZE_MAX); // no forcing expected (natural end)
    });
}

static void test_clone(testing & t) {
    const std::vector<llama_token> start = {100};
    const std::vector<llama_token> end = {101};
    const std::vector<llama_token> forced = {102, 101};

    t.test("mid counting", [&](testing & t) {
        auto * sampler = common_reasoning_budget_init(nullptr, {start}, {end}, forced, 2, REASONING_BUDGET_IDLE);

        llama_sampler_accept(sampler, 100); // COUNTING, remaining=2
        llama_sampler_accept(sampler, 50);  // COUNTING, remaining=1

        auto * clone = llama_sampler_clone(sampler);
        llama_sampler_accept(clone, 51); // should exhaust the cloned remaining budget

        t.assert_equal("cloned counting state keeps the remaining budget", 102, get_forced_token(t, clone, 102));

        llama_sampler_free(clone);
        llama_sampler_free(sampler);
    });

    t.test("mid forcing", [&](testing & t) {
        auto * sampler = common_reasoning_budget_init(nullptr, {start}, {end}, forced, 0, REASONING_BUDGET_FORCING);

        t.assert_equal("first forced token", 102, get_forced_token(t, sampler, 102));
        llama_sampler_accept(sampler, 102); // advance to the second forced token

        auto * clone = llama_sampler_clone(sampler);

        t.assert_equal("cloned forcing state keeps the force position", 101, get_forced_token(t, clone, 102));

        llama_sampler_free(clone);
        llama_sampler_free(sampler);
    });
}

static void test_force_manual(testing & t) {
    const std::vector<llama_token> start  = {100};
    const std::vector<llama_token> end    = {101};
    const std::vector<llama_token> forced = {102, 101};

    // if COUNTING, force() succeeds and begins forcing the end sequence from the start
    t.test("from counting", [&](testing & t) {
        auto * sampler = common_reasoning_budget_init(nullptr, {start}, {end}, forced, 5, REASONING_BUDGET_IDLE);

        llama_sampler_accept(sampler, 100); // COUNTING, remaining=5
        llama_sampler_accept(sampler, 50);  // COUNTING, remaining=4
        t.assert_true("state is COUNTING", common_reasoning_budget_get_state(sampler) == REASONING_BUDGET_COUNTING);

        t.assert_true("force() succeeds from COUNTING", common_reasoning_budget_force(sampler));
        t.assert_true("state is FORCING", common_reasoning_budget_get_state(sampler) == REASONING_BUDGET_FORCING);

        // forces the configured sequence from force_pos=0, then transitions to DONE
        t.assert_equal("first forced token", 102, get_forced_token(t, sampler, 102));
        llama_sampler_accept(sampler, 102);
        t.assert_equal("second forced token", 101, get_forced_token(t, sampler, 102));
        llama_sampler_accept(sampler, 101);
        t.assert_true("state is DONE", common_reasoning_budget_get_state(sampler) == REASONING_BUDGET_DONE);

        llama_sampler_free(sampler);
    });

    // if IDLE, force() is a no-op
    t.test("from idle", [&](testing & t) {
        auto * sampler = common_reasoning_budget_init(nullptr, {start}, {end}, forced, 5, REASONING_BUDGET_IDLE);

        t.assert_true("force() must not transition from IDLE", !common_reasoning_budget_force(sampler));
        t.assert_true("state is IDLE", common_reasoning_budget_get_state(sampler) == REASONING_BUDGET_IDLE);

        llama_sampler_free(sampler);
    });

    // if DONE, force() is a no-op
    t.test("from done", [&](testing & t) {
        auto * sampler = common_reasoning_budget_init(nullptr, {start}, {end}, forced, 5, REASONING_BUDGET_IDLE);

        llama_sampler_accept(sampler, 100); // COUNTING
        llama_sampler_accept(sampler, 101); // natural end -> DONE
        t.assert_true("state is DONE", common_reasoning_budget_get_state(sampler) == REASONING_BUDGET_DONE);

        t.assert_true("force() must not transition from DONE", !common_reasoning_budget_force(sampler));
        t.assert_true("state is still DONE", common_reasoning_budget_get_state(sampler) == REASONING_BUDGET_DONE);

        llama_sampler_free(sampler);
    });

    // if FORCING, force() is a no-op and must not rewind the force position
    t.test("from forcing", [&](testing & t) {
        auto * sampler = common_reasoning_budget_init(nullptr, {start}, {end}, forced, 0, REASONING_BUDGET_FORCING);

        t.assert_equal("first forced token", 102, get_forced_token(t, sampler, 102));
        llama_sampler_accept(sampler, 102); // advance to the second forced token (force_pos=1)

        t.assert_true("force() must not transition from FORCING", !common_reasoning_budget_force(sampler));
        t.assert_true("state is FORCING", common_reasoning_budget_get_state(sampler) == REASONING_BUDGET_FORCING);
        t.assert_equal("force() must not rewind the force position", 101, get_forced_token(t, sampler, 102));

        llama_sampler_free(sampler);
    });

    // a null sampler is safely ignored
    t.test("null sampler", [](testing & t) {
        t.assert_true("force() on a null sampler returns false", !common_reasoning_budget_force(nullptr));
    });
}

static void test_end_match(testing & t) {
    const std::vector<llama_tokens> start = {{100}};
    const std::vector<llama_tokens> end   = {{101}, {103, 104}};

    // natural end records the sequence that matched; re-arming clears it
    t.test("natural end", [&](testing & t) {
        auto * sampler = common_reasoning_budget_init(nullptr, start, end, {102, 101}, 5, REASONING_BUDGET_IDLE);

        t.assert_true("no end match before any token", common_reasoning_budget_get_end_match(sampler) == nullptr);

        llama_sampler_accept(sampler, 100); // COUNTING
        llama_sampler_accept(sampler, 50);
        llama_sampler_accept(sampler, 103);
        llama_sampler_accept(sampler, 104); // end matched via {103, 104}, DONE

        const llama_tokens * matched = common_reasoning_budget_get_end_match(sampler);
        if (t.assert_true("end match is recorded", matched != nullptr)) {
            t.assert_true("end match is {103, 104}", *matched == llama_tokens({103, 104}));
        }

        llama_sampler_accept(sampler, 100); // re-arm, COUNTING
        t.assert_true("re-arming clears the end match", common_reasoning_budget_get_end_match(sampler) == nullptr);

        llama_sampler_free(sampler);
    });

    // overlapping end sequences: the longest one ending at the position wins
    t.test("overlapping end sequences", [&](testing & t) {
        const std::vector<llama_tokens> end_overlap = {{104}, {103, 104}};

        auto * sampler = common_reasoning_budget_init(nullptr, start, end_overlap, {102, 104}, 5, REASONING_BUDGET_IDLE);

        llama_sampler_accept(sampler, 100); // COUNTING
        llama_sampler_accept(sampler, 103);
        llama_sampler_accept(sampler, 104); // both {104} and {103, 104} end here

        const llama_tokens * matched = common_reasoning_budget_get_end_match(sampler);
        if (t.assert_true("end match is recorded", matched != nullptr)) {
            t.assert_true("the longest end sequence {103, 104} wins", *matched == llama_tokens({103, 104}));
        }

        llama_sampler_free(sampler);
    });

    // forcing records the end sequence terminating forced_tokens
    t.test("forced end", [&](testing & t) {
        auto * sampler = common_reasoning_budget_init(nullptr, start, end, {102, 103, 104}, 0, REASONING_BUDGET_FORCING);

        llama_sampler_accept(sampler, 102);
        llama_sampler_accept(sampler, 103);
        t.assert_true("no end match before the forced sequence completes", common_reasoning_budget_get_end_match(sampler) == nullptr);
        llama_sampler_accept(sampler, 104); // forced sequence complete, DONE

        const llama_tokens * matched = common_reasoning_budget_get_end_match(sampler);
        if (t.assert_true("end match is recorded", matched != nullptr)) {
            t.assert_true("end match is {103, 104}", *matched == llama_tokens({103, 104}));
        }

        llama_sampler_free(sampler);
    });

    // forced_tokens not ending with a known end sequence records nothing
    t.test("forced tokens without an end sequence", [&](testing & t) {
        auto * sampler = common_reasoning_budget_init(nullptr, start, end, {102}, 0, REASONING_BUDGET_FORCING);

        llama_sampler_accept(sampler, 102); // forced sequence complete, DONE
        t.assert_true("state is DONE", common_reasoning_budget_get_state(sampler) == REASONING_BUDGET_DONE);
        t.assert_true("no end match is recorded", common_reasoning_budget_get_end_match(sampler) == nullptr);

        llama_sampler_free(sampler);
    });

    // a null sampler is safely ignored
    t.test("null sampler", [](testing & t) {
        t.assert_true("get_end_match() on a null sampler returns null", common_reasoning_budget_get_end_match(nullptr) == nullptr);
    });
}

// UTF-8 boundary detection unit test
// Tests common_utf8_is_complete() from reasoning-budget.h
static void test_utf8_boundary_detection(testing & t) {
    t.test("complete sequences", [](testing & t) {
        t.assert_true("ascii", common_utf8_is_complete("hello"));
        t.assert_true("empty", common_utf8_is_complete(""));
        t.assert_true("complete 2-byte UTF-8 (U+00A0)", common_utf8_is_complete("\xC2\xA0"));
        t.assert_true("complete 3-byte UTF-8 (left double quote)", common_utf8_is_complete("\xE2\x80\x9C"));
        t.assert_true("complete 4-byte UTF-8 (emoji)", common_utf8_is_complete("\xF0\x9F\x98\x80"));
        t.assert_true("ASCII + complete 2-byte", common_utf8_is_complete("abc\xC3\xA9"));
    });

    t.test("incomplete sequences", [](testing & t) {
        t.assert_true("2-byte start, missing continuation", !common_utf8_is_complete(std::string("\xC2", 1)));
        t.assert_true("3-byte start + 1 cont, missing 1", !common_utf8_is_complete(std::string("\xE2\x80", 2)));
        t.assert_true("3-byte start, missing 2", !common_utf8_is_complete(std::string("\xE2", 1)));
        t.assert_true("4-byte start + 2 cont, missing 1", !common_utf8_is_complete(std::string("\xF0\x9F\x98", 3)));
        t.assert_true("4-byte start + 1 cont, missing 2", !common_utf8_is_complete(std::string("\xF0\x9F", 2)));
        t.assert_true("4-byte start, missing 3", !common_utf8_is_complete(std::string("\xF0", 1)));
        t.assert_true("orphan continuation byte", !common_utf8_is_complete(std::string("\x80", 1)));
    });

    // Mixed: ASCII followed by start of multi-byte
    t.test("mixed", [](testing & t) {
        t.assert_true("ASCII + incomplete 2-byte", !common_utf8_is_complete(std::string("hello\xC3", 6)));
        t.assert_true("ASCII + complete 2-byte", common_utf8_is_complete(std::string("hello\xC3\xA9", 7)));
    });
}

int main(int argc, char ** argv) {
    testing t;
    t.capture_output = true;
    t.apply_env();
    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("sequence", test_sequences);
    t.test("clone", test_clone);
    t.test("force", test_force_manual);
    t.test("end match", test_end_match);
    t.test("utf8", test_utf8_boundary_detection);

    return t.summary();
}
