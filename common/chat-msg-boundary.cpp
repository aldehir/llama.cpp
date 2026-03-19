#include "chat-msg-boundary.h"
#include "common.h"

#include <algorithm>

std::vector<aho_corasick_match> aho_corasick_search(const aho_corasick & ac, const std::string & text) {
    std::vector<aho_corasick_match> matches;
    int cur = 0;

    for (size_t i = 0; i < text.size(); i++) {
        char ch = text[i];

        // Follow fail links until we find a transition or reach root
        while (cur != 0 && ac.nodes[cur].children.find(ch) == ac.nodes[cur].children.end()) {
            cur = ac.nodes[cur].fail;
        }

        auto it = ac.nodes[cur].children.find(ch);
        if (it != ac.nodes[cur].children.end()) {
            cur = it->second;
        }
        // else cur stays at 0 (root)

        // Check for matches at this node and via suffix links
        int tmp = cur;
        while (tmp != 0) {
            if (ac.nodes[tmp].output >= 0) {
                int    idx     = ac.nodes[tmp].output;
                size_t pat_len = ac.patterns[idx].size();
                matches.push_back({idx, i + 1 - pat_len});
            }
            tmp = ac.nodes[tmp].fail;
        }
    }

    // Sort by position, then by longest pattern first for same position
    std::sort(matches.begin(), matches.end(), [&](const aho_corasick_match & a, const aho_corasick_match & b) {
        if (a.pos != b.pos) {
            return a.pos < b.pos;
        }
        return ac.patterns[a.pattern_idx].size() > ac.patterns[b.pattern_idx].size();
    });

    return matches;
}

std::vector<common_chat_msg_boundary> common_chat_find_msg_boundaries(
        const std::string &                        prompt,
        const std::vector<common_chat_msg_marker> & markers) {
    if (markers.empty() || prompt.empty()) {
        return {};
    }

    // Build automaton
    aho_corasick ac;
    for (size_t i = 0; i < markers.size(); i++) {
        ac.add_pattern(markers[i].pattern, (int) i);
    }
    ac.build();

    // Search
    auto matches = aho_corasick_search(ac, prompt);
    if (matches.empty()) {
        return {};
    }

    // Convert matches to boundaries.
    // At each position, keep only the longest (first after sort) match.
    std::vector<common_chat_msg_boundary> boundaries;
    size_t prev_pos = (size_t) -1;

    for (const auto & m : matches) {
        if (m.pos == prev_pos) {
            continue;  // skip shorter matches at the same position
        }
        prev_pos = m.pos;

        // Close the previous boundary
        if (!boundaries.empty()) {
            boundaries.back().end = m.pos;
        }

        int idx = m.pattern_idx < (int) markers.size() ? m.pattern_idx : 0;
        boundaries.push_back({markers[idx].role, m.pos, prompt.size()});
    }

    return boundaries;
}

std::vector<common_chat_msg_boundary> common_chat_detect_msg_boundaries(const std::string & prompt) {
    if (prompt.empty()) {
        return {};
    }

    // ChatML family: <|im_start|>role
    if (prompt.find("<|im_start|>") != std::string::npos) {
        return common_chat_find_msg_boundaries(prompt, {
            {"<|im_start|>system",    "system"},
            {"<|im_start|>user",      "user"},
            {"<|im_start|>assistant", "assistant"},
            {"<|im_start|>tool",      "tool"},
        });
    }

    // Llama 3 family: <|start_header_id|>role
    if (prompt.find("<|start_header_id|>") != std::string::npos) {
        return common_chat_find_msg_boundaries(prompt, {
            {"<|start_header_id|>system",    "system"},
            {"<|start_header_id|>user",      "user"},
            {"<|start_header_id|>assistant", "assistant"},
            {"<|start_header_id|>tool",      "tool"},
            {"<|start_header_id|>ipython",   "ipython"},
        });
    }

    // Gemma family: <start_of_turn>role
    if (prompt.find("<start_of_turn>") != std::string::npos) {
        return common_chat_find_msg_boundaries(prompt, {
            {"<start_of_turn>user",  "user"},
            {"<start_of_turn>model", "assistant"},
        });
    }

    // Command R family: <|START_OF_TURN_TOKEN|><|ROLE_TOKEN|>
    if (prompt.find("<|START_OF_TURN_TOKEN|>") != std::string::npos) {
        return common_chat_find_msg_boundaries(prompt, {
            {"<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>",  "system"},
            {"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",    "user"},
            {"<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>", "assistant"},
        });
    }

    // Phi-3 family: <|system|>, <|user|>, <|assistant|>
    if (prompt.find("<|user|>") != std::string::npos && prompt.find("<|assistant|>") != std::string::npos
            && prompt.find("<|end|>") != std::string::npos) {
        return common_chat_find_msg_boundaries(prompt, {
            {"<|system|>",    "system"},
            {"<|user|>",      "user"},
            {"<|assistant|>", "assistant"},
        });
    }

    // Mistral family: [INST] and [SYSTEM_PROMPT]
    if (prompt.find("[INST]") != std::string::npos) {
        return common_chat_find_msg_boundaries(prompt, {
            {"[SYSTEM_PROMPT]", "system"},
            {"[INST]",          "user"},
        });
    }

    // DeepSeek v2/v3 family
    if (prompt.find("<｜User｜>") != std::string::npos || prompt.find("<｜Assistant｜>") != std::string::npos) {
        return common_chat_find_msg_boundaries(prompt, {
            {"<｜User｜>",      "user"},
            {"<｜Assistant｜>", "assistant"},
        });
    }

    return {};
}

std::vector<common_chat_msg_boundary_token> common_chat_msg_boundaries_to_tokens(
        const llama_vocab                           * vocab,
        const std::vector<llama_token>              & tokens,
        const std::vector<common_chat_msg_boundary> & boundaries,
        bool                                          special) {
    if (boundaries.empty() || tokens.empty()) {
        return {};
    }

    std::vector<common_chat_msg_boundary_token> result;
    result.reserve(boundaries.size());

    size_t byte_offset = 0;
    size_t b_idx       = 0;

    for (int32_t t = 0; t < (int32_t) tokens.size(); t++) {
        // Advance past any boundaries whose start falls at or before the current byte offset
        while (b_idx < boundaries.size() && boundaries[b_idx].start <= byte_offset) {
            if (b_idx > 0 && !result.empty()) {
                result.back().token_end = t;
            }
            result.push_back({boundaries[b_idx].role, t, (int32_t) tokens.size()});
            b_idx++;
        }

        std::string piece = common_token_to_piece(vocab, tokens[t], special);
        byte_offset += piece.size();
    }

    // Pick up any remaining boundaries that start at or after the last token's bytes
    while (b_idx < boundaries.size()) {
        if (!result.empty()) {
            result.back().token_end = (int32_t) tokens.size();
        }
        result.push_back({boundaries[b_idx].role, (int32_t) tokens.size(), (int32_t) tokens.size()});
        b_idx++;
    }

    return result;
}
