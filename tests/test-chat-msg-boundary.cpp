#include "chat-msg-boundary.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

static void test_aho_corasick_single_pattern() {
    printf("  test_aho_corasick_single_pattern...\n");
    aho_corasick ac;
    ac.add_pattern("abc", 0);
    ac.build();

    auto matches = aho_corasick_search(ac, "xabcyabcz");
    assert(matches.size() == 2);
    assert(matches[0].pos == 1);
    assert(matches[0].pattern_idx == 0);
    assert(matches[1].pos == 5);
    assert(matches[1].pattern_idx == 0);
}

static void test_aho_corasick_multiple_patterns() {
    printf("  test_aho_corasick_multiple_patterns...\n");
    aho_corasick ac;
    ac.add_pattern("he", 0);
    ac.add_pattern("she", 1);
    ac.add_pattern("his", 2);
    ac.add_pattern("hers", 3);
    ac.build();

    auto matches = aho_corasick_search(ac, "ushers");
    // "she" at 1, "he" at 2, "hers" at 2
    assert(!matches.empty());

    // Verify patterns are found at expected positions
    auto has_match = [&](int pat_idx, size_t pos) -> bool {
        for (const auto & m : matches) {
            if (m.pattern_idx == pat_idx && m.pos == pos) return true;
        }
        return false;
    };
    (void) has_match;  // used in asserts below
    assert(has_match(1, 1));  // "she" at position 1
    assert(has_match(0, 2));  // "he" at position 2
    assert(has_match(3, 2));  // "hers" at position 2
}

static void test_aho_corasick_no_match() {
    printf("  test_aho_corasick_no_match...\n");
    aho_corasick ac;
    ac.add_pattern("xyz", 0);
    ac.build();

    auto matches = aho_corasick_search(ac, "abcdef");
    assert(matches.empty());
}

static void test_aho_corasick_empty_text() {
    printf("  test_aho_corasick_empty_text...\n");
    aho_corasick ac;
    ac.add_pattern("abc", 0);
    ac.build();

    auto matches = aho_corasick_search(ac, "");
    assert(matches.empty());
}

static void test_boundaries_gpt_oss() {
    printf("  test_boundaries_gpt_oss...\n");
    std::string prompt =
        "<|start|>system<|channel|>final<|message|>You are helpful.<|end|>"
        "<|start|>user<|channel|>final<|message|>Hello!<|end|>"
        "<|start|>assistant";

    auto boundaries = common_chat_find_msg_boundaries(prompt, {
        {"<|start|>system",    "system"},
        {"<|start|>user",      "user"},
        {"<|start|>assistant", "assistant"},
    });

    assert(boundaries.size() == 3);

    assert(boundaries[0].role == "system");
    assert(boundaries[0].start == 0);

    assert(boundaries[1].role == "user");
    assert(boundaries[1].start == prompt.find("<|start|>user"));
    assert(boundaries[0].end == boundaries[1].start);

    assert(boundaries[2].role == "assistant");
    assert(boundaries[2].start == prompt.find("<|start|>assistant"));
    assert(boundaries[1].end == boundaries[2].start);
    assert(boundaries[2].end == prompt.size());
}

static void test_boundaries_chatml() {
    printf("  test_boundaries_chatml...\n");
    std::string prompt =
        "<|im_start|>system\nYou are helpful.<|im_end|>\n"
        "<|im_start|>user\nHello!<|im_end|>\n"
        "<|im_start|>assistant\n";

    auto boundaries = common_chat_find_msg_boundaries(prompt, {
        {"<|im_start|>system",    "system"},
        {"<|im_start|>user",      "user"},
        {"<|im_start|>assistant", "assistant"},
    });

    assert(boundaries.size() == 3);
    assert(boundaries[0].role == "system");
    assert(boundaries[0].start == 0);
    assert(boundaries[1].role == "user");
    assert(boundaries[2].role == "assistant");

    // Verify contiguous coverage
    for (size_t i = 0; i + 1 < boundaries.size(); i++) {
        assert(boundaries[i].end == boundaries[i + 1].start);
    }
    assert(boundaries.back().end == prompt.size());
}

static void test_boundaries_llama3() {
    printf("  test_boundaries_llama3...\n");
    std::string prompt =
        "<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n";

    auto boundaries = common_chat_find_msg_boundaries(prompt, {
        {"<|start_header_id|>system",    "system"},
        {"<|start_header_id|>user",      "user"},
        {"<|start_header_id|>assistant", "assistant"},
    });

    assert(boundaries.size() == 3);
    assert(boundaries[0].role == "system");
    assert(boundaries[1].role == "user");
    assert(boundaries[2].role == "assistant");
}

static void test_boundaries_empty() {
    printf("  test_boundaries_empty...\n");
    auto boundaries = common_chat_find_msg_boundaries("", {{"<|start|>", "system"}});
    assert(boundaries.empty());

    boundaries = common_chat_find_msg_boundaries("hello", {});
    assert(boundaries.empty());
}

static void test_detect_chatml() {
    printf("  test_detect_chatml...\n");
    std::string prompt =
        "<|im_start|>system\nYou are helpful.<|im_end|>\n"
        "<|im_start|>user\nHello!<|im_end|>\n"
        "<|im_start|>assistant\n";

    auto boundaries = common_chat_detect_msg_boundaries(prompt);
    assert(boundaries.size() == 3);
    assert(boundaries[0].role == "system");
    assert(boundaries[1].role == "user");
    assert(boundaries[2].role == "assistant");
}

static void test_detect_llama3() {
    printf("  test_detect_llama3...\n");
    std::string prompt =
        "<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n";

    auto boundaries = common_chat_detect_msg_boundaries(prompt);
    assert(boundaries.size() == 3);
    assert(boundaries[0].role == "system");
    assert(boundaries[1].role == "user");
    assert(boundaries[2].role == "assistant");
}

static void test_detect_gemma() {
    printf("  test_detect_gemma...\n");
    std::string prompt =
        "<start_of_turn>user\nHello!<end_of_turn>\n"
        "<start_of_turn>model\n";

    auto boundaries = common_chat_detect_msg_boundaries(prompt);
    assert(boundaries.size() == 2);
    assert(boundaries[0].role == "user");
    assert(boundaries[1].role == "assistant");
}

static void test_detect_mistral() {
    printf("  test_detect_mistral...\n");
    std::string prompt =
        "[SYSTEM_PROMPT]You are helpful.[/SYSTEM_PROMPT]"
        "[INST]Hello![/INST]";

    auto boundaries = common_chat_detect_msg_boundaries(prompt);
    assert(boundaries.size() == 2);
    assert(boundaries[0].role == "system");
    assert(boundaries[1].role == "user");
}

static void test_detect_deepseek() {
    printf("  test_detect_deepseek...\n");
    std::string prompt =
        "<｜User｜>Hello!<｜end▁of▁sentence｜>"
        "<｜Assistant｜>";

    auto boundaries = common_chat_detect_msg_boundaries(prompt);
    assert(boundaries.size() == 2);
    assert(boundaries[0].role == "user");
    assert(boundaries[1].role == "assistant");
}

static void test_detect_unknown() {
    printf("  test_detect_unknown...\n");
    auto boundaries = common_chat_detect_msg_boundaries("just some random text");
    assert(boundaries.empty());
}

static void test_boundary_contiguity() {
    printf("  test_boundary_contiguity...\n");
    // Verify that boundaries form contiguous, non-overlapping coverage
    std::string prompt =
        "<|im_start|>system\nSys prompt.<|im_end|>\n"
        "<|im_start|>user\nMsg 1<|im_end|>\n"
        "<|im_start|>assistant\nReply 1<|im_end|>\n"
        "<|im_start|>user\nMsg 2<|im_end|>\n"
        "<|im_start|>assistant\n";

    auto boundaries = common_chat_detect_msg_boundaries(prompt);
    assert(boundaries.size() == 5);

    // First boundary starts at 0
    assert(boundaries[0].start == 0);

    // Contiguous: each end == next start
    for (size_t i = 0; i + 1 < boundaries.size(); i++) {
        assert(boundaries[i].end == boundaries[i + 1].start);
    }

    // Last boundary ends at prompt.size()
    assert(boundaries.back().end == prompt.size());

    // Roles alternate correctly
    assert(boundaries[0].role == "system");
    assert(boundaries[1].role == "user");
    assert(boundaries[2].role == "assistant");
    assert(boundaries[3].role == "user");
    assert(boundaries[4].role == "assistant");
}

int main() {
    printf("Testing Aho-Corasick automaton...\n");
    test_aho_corasick_single_pattern();
    test_aho_corasick_multiple_patterns();
    test_aho_corasick_no_match();
    test_aho_corasick_empty_text();

    printf("Testing boundary finding...\n");
    test_boundaries_gpt_oss();
    test_boundaries_chatml();
    test_boundaries_llama3();
    test_boundaries_empty();

    printf("Testing auto-detection...\n");
    test_detect_chatml();
    test_detect_llama3();
    test_detect_gemma();
    test_detect_mistral();
    test_detect_deepseek();
    test_detect_unknown();

    printf("Testing boundary properties...\n");
    test_boundary_contiguity();

    printf("All tests passed!\n");
    return 0;
}
