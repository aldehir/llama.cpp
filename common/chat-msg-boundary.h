#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <queue>
#include <string>
#include <vector>

struct llama_vocab;

using llama_token = int32_t;

// A pattern-to-role mapping for message boundary detection.
struct common_chat_msg_marker {
    std::string pattern;  // the start-of-message marker string to search for
    std::string role;     // the role this marker represents
};

// A message boundary in the rendered prompt string.
// [start, end) are byte offsets: start is inclusive, end is exclusive.
struct common_chat_msg_boundary {
    std::string role;   // "system", "user", "assistant", "tool", etc.
    size_t      start;  // byte offset of the message start marker
    size_t      end;    // byte offset of next message start, or end of prompt
};

// Aho-Corasick automaton for multi-pattern string matching.
struct aho_corasick {
    struct node {
        std::map<char, int> children;
        int                 fail   = 0;
        int                 output = -1;  // pattern index, or -1 if none
    };

    std::vector<node>        nodes;
    std::vector<std::string> patterns;

    aho_corasick() {
        nodes.emplace_back();  // root node
    }

    void add_pattern(const std::string & pattern, int index) {
        int cur = 0;
        for (char ch : pattern) {
            auto it = nodes[cur].children.find(ch);
            if (it == nodes[cur].children.end()) {
                nodes[cur].children[ch] = (int) nodes.size();
                cur = (int) nodes.size();
                nodes.emplace_back();
            } else {
                cur = it->second;
            }
        }
        nodes[cur].output = index;
        if (index >= (int) patterns.size()) {
            patterns.resize(index + 1);
        }
        patterns[index] = pattern;
    }

    void build() {
        std::queue<int> q;
        // Initialize fail links for depth-1 nodes
        for (auto & [ch, child] : nodes[0].children) {
            nodes[child].fail = 0;
            q.push(child);
        }
        // BFS to build fail links
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            for (auto & [ch, v] : nodes[u].children) {
                int f = nodes[u].fail;
                while (f != 0 && nodes[f].children.find(ch) == nodes[f].children.end()) {
                    f = nodes[f].fail;
                }
                auto it = nodes[f].children.find(ch);
                nodes[v].fail = (it != nodes[f].children.end() && it->second != v) ? it->second : 0;
                if (nodes[v].output == -1) {
                    nodes[v].output = nodes[nodes[v].fail].output;
                }
                q.push(v);
            }
        }
    }
};

struct aho_corasick_match {
    int    pattern_idx;
    size_t pos;  // byte offset where the pattern starts
};

// Search text using a built automaton. Returns matches sorted by position.
std::vector<aho_corasick_match> aho_corasick_search(const aho_corasick & ac, const std::string & text);

// Find message boundaries in the rendered prompt.
// Multiple markers can map to the same role.
// Returns boundaries sorted by position; end of each = start of next, or end of prompt.
std::vector<common_chat_msg_boundary> common_chat_find_msg_boundaries(
    const std::string &                        prompt,
    const std::vector<common_chat_msg_marker> & markers);

// Auto-detect template family from the rendered prompt and find message boundaries.
// Recognizes ChatML, Llama3, Gemma, Mistral, Command R, and Phi-3 families.
// Returns empty vector if the template family is not recognized.
std::vector<common_chat_msg_boundary> common_chat_detect_msg_boundaries(const std::string & prompt);

// A message boundary expressed as token positions.
// [token_start, token_end) are token indices: token_start is inclusive, token_end is exclusive.
struct common_chat_msg_boundary_token {
    std::string role;
    int32_t     token_start;  // first token index belonging to this message
    int32_t     token_end;    // first token index of the next message (or total token count)
};

// Convert byte-offset boundaries to token-position boundaries.
// Walks the token list, accumulates byte lengths via common_token_to_piece(),
// and maps each byte-offset boundary to the token index where it falls.
// `special` controls whether special tokens are decoded to their text form.
std::vector<common_chat_msg_boundary_token> common_chat_msg_boundaries_to_tokens(
    const llama_vocab                           * vocab,
    const std::vector<llama_token>              & tokens,
    const std::vector<common_chat_msg_boundary> & boundaries,
    bool                                          special);
