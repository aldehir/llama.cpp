#pragma once

#include <array>
#include <bitset>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <vector>

struct llama_grammar_element;
struct llama_vocab;

// ============================================================
// NFA (Thompson construction over bytes)
// ============================================================

struct grammar_nfa {
    struct transition {
        uint8_t  lo;      // inclusive lower byte bound
        uint8_t  hi;      // inclusive upper byte bound
        uint32_t target;  // target state
    };

    struct state {
        std::vector<transition> byte_transitions;
        std::vector<uint32_t>   epsilon_transitions;
    };

    std::vector<state> states;
    uint32_t start;
    uint32_t accept;

    uint32_t add_state();

    // Merge all states from 'other' into this NFA, adjusting state indices.
    // Returns the offset applied to other's state indices.
    uint32_t merge(const grammar_nfa & other);

    // Thompson construction primitives
    static grammar_nfa byte_match(uint8_t lo, uint8_t hi);
    static grammar_nfa concat(grammar_nfa a, grammar_nfa b);
    static grammar_nfa alternation(std::vector<grammar_nfa> parts);
    static grammar_nfa epsilon_nfa();
};

// ============================================================
// DFA (deterministic finite automaton over bytes)
// ============================================================

struct byte_dfa {
    // transitions[state][byte] -> next_state (0 = dead/reject state)
    std::vector<std::array<uint16_t, 256>> transitions;
    std::vector<bool> accept;
    uint16_t start_state;  // typically 1 (state 0 is dead)

    // Build from NFA via subset construction
    static byte_dfa from_nfa(const grammar_nfa & nfa);

    // Minimize using Hopcroft's algorithm, then canonicalize state numbering
    void minimize();

    // Complement: flip accept/reject states (except dead state 0)
    void complement();

    // Product construction: intersection of two DFAs
    static byte_dfa intersect(const byte_dfa & a, const byte_dfa & b);

    // Equality: two DFAs match if they have the same structure
    bool operator==(const byte_dfa & other) const;

    // Total heap memory used by this DFA (transitions + accept vectors)
    size_t size_bytes() const;
};

// ============================================================
// UTF-8 expansion: code point ranges -> byte-level NFAs
// ============================================================

// Encode a Unicode code point as UTF-8 bytes
std::vector<uint8_t> encode_utf8_bytes(uint32_t cp);

// Build NFA matching UTF-8 encoding of any code point in [lo, hi]
grammar_nfa codepoint_range_to_byte_nfa(uint32_t lo, uint32_t hi);

// Build NFA matching any single valid UTF-8 character (equivalent to '.')
grammar_nfa valid_utf8_char_nfa();

// Build (or return cached) DFA for valid single UTF-8 character
byte_dfa build_valid_utf8_dfa();

// ============================================================
// Grammar compilation structures
// ============================================================

struct compiled_segment {
    enum type_t : uint8_t { DFA_MATCH, RULE_CALL };
    type_t   type;
    uint32_t id;  // dfa index for DFA_MATCH, rule_id for RULE_CALL
};

struct compiled_rule {
    // Each alternate is a sequence of segments (DFA matches and rule calls)
    std::vector<std::vector<compiled_segment>> alternates;

    size_t size_bytes() const;
};

// ============================================================
// Runtime state for DFA-based grammar engine
// ============================================================

struct grammar_position {
    uint32_t rule_id;
    uint8_t  alternate_idx;
    uint8_t  segment_idx;
    uint16_t dfa_state;

    bool operator==(const grammar_position & other) const {
        return rule_id == other.rule_id &&
               alternate_idx == other.alternate_idx &&
               segment_idx == other.segment_idx &&
               dfa_state == other.dfa_state;
    }
};

struct grammar_call_frame {
    uint32_t rule_id;
    uint8_t  alternate_idx;
    uint8_t  segment_idx;    // segment to resume AFTER the call returns
    uint16_t dfa_state;      // DFA state to resume (for the next segment)

    bool operator==(const grammar_call_frame & other) const {
        return rule_id == other.rule_id &&
               alternate_idx == other.alternate_idx &&
               segment_idx == other.segment_idx &&
               dfa_state == other.dfa_state;
    }
};

struct parse_config {
    grammar_position                current;
    std::vector<grammar_call_frame> call_stack;

    bool operator==(const parse_config & other) const {
        return current == other.current && call_stack == other.call_stack;
    }
};

// ============================================================
// Vocab trie for efficient prefix-based token lookup
// ============================================================

struct vocab_trie_node {
    // Tokens that terminate exactly at this node
    std::vector<int32_t> tokens;  // llama_token = int32_t

    // Children indexed by next byte (dense for fast DFA walk)
    std::unique_ptr<vocab_trie_node> children[256];

    // Precomputed: all token IDs at or below this node (sorted).
    // Built once by cache_subtree_tokens() after trie construction.
    std::vector<int32_t> subtree_tokens;

    // Recursively collect all token IDs at or below this node
    void collect_tokens(std::vector<int32_t> & out) const;

    // Count all tokens at or below this node
    uint32_t count_tokens() const;

    // Precompute subtree_tokens for this node and all descendants.
    // Must be called after the trie is fully built.
    void cache_subtree_tokens();

    // Total memory used by this node and all descendants (recursive)
    size_t size_bytes() const;
};

struct vocab_byte_trie {
    vocab_trie_node root;
    uint32_t n_tokens_inserted = 0;

    void build(const llama_vocab & vocab);

    size_t size_bytes() const;
};

// ============================================================
// Per-DFA-state precomputed candidate token sets
// ============================================================

struct dfa_state_token_set {
    // Adaptive representation: stores whichever is smaller.
    // If accept_heavy == true:
    //   Most tokens are candidates → store the REJECT set (tokens to skip).
    //   Tokens NOT in the set need simulation.
    // If accept_heavy == false:
    //   Most tokens are rejected → store the ACCEPT set (tokens to simulate).
    //   Only tokens IN the set need simulation.
    bool accept_heavy = false;
    std::vector<int32_t> token_set;  // sorted

    size_t size_bytes() const;
};

struct dfa_state_candidates {
    // Per-state adaptive token sets. State 0 (dead) has empty set.
    std::vector<dfa_state_token_set> states;

    // Precompute candidates for all states of a DFA using the vocab trie.
    void precompute(const byte_dfa & dfa, const vocab_byte_trie & trie,
                    uint32_t total_vocab_tokens);

    size_t size_bytes() const;

private:
    // Walk trie with multiple DFA states simultaneously.
    // active_states maps dfa_state -> list of original starting states that
    // have reached this dfa_state at this trie depth.
    // candidate_bits[s] is the bitset of candidate tokens for starting state s.
    void walk_multi(const byte_dfa & dfa,
                    const vocab_trie_node & node,
                    const std::vector<std::pair<uint16_t, std::vector<uint16_t>>> & active_groups,
                    std::vector<std::vector<uint8_t>> & candidate_bits);
};

// ============================================================
// Memory usage breakdown for compiled grammar
// ============================================================

struct compiled_grammar_mem_info {
    size_t rules_bytes;       // compiled_rule structures
    size_t dfas_bytes;        // byte_dfa transition tables
    size_t candidates_bytes;  // dfa_state_candidates token sets
    size_t total_bytes;       // sum of above
};

// ============================================================
// Compiled grammar (immutable after construction)
// ============================================================

struct compiled_grammar {
    std::vector<byte_dfa>      dfas;
    std::vector<compiled_rule> rules;
    uint32_t                   start_rule;

    // Per-DFA-state precomputed candidate token sets
    std::vector<dfa_state_candidates> dfa_candidates;

    // Compile from parsed grammar rules
    static compiled_grammar compile(
        const std::vector<std::vector<llama_grammar_element>> & parsed_rules,
        uint32_t start_rule_index);

    // Precompute candidate token sets for all DFA states using the vocab trie.
    // Must be called after compile() and after the trie is built.
    void precompute_token_candidates(const vocab_byte_trie & trie, uint32_t total_vocab_tokens);

    // Initialize runtime configs for the start rule
    std::vector<parse_config> init_configs() const;

    // Advance configs through one byte
    void accept_byte(std::vector<parse_config> & configs, uint8_t byte) const;

    // Get set of valid next bytes across all configs
    std::bitset<256> get_valid_bytes(const std::vector<parse_config> & configs) const;

    // Check if any config is in a "complete" state (rule finished, stack empty)
    bool any_config_complete(const std::vector<parse_config> & configs) const;

    // Memory usage breakdown (rules, DFAs, candidate cache)
    compiled_grammar_mem_info mem_info() const;

private:
    // Expand a config to reach a DFA_MATCH segment (or mark as complete)
    // Returns all reachable configs after resolving RULE_CALLs and rule completions
    void advance_to_terminal(
        const parse_config & cfg,
        std::vector<parse_config> & out_configs) const;

    // Deduplicate configs
    static void dedup_configs(std::vector<parse_config> & configs);
};

// ============================================================
// DFA construction from grammar elements
// ============================================================

// Build a DFA for a terminal segment (sequence of CHAR/CHAR_NOT/CHAR_ANY/CHAR_RNG_UPPER/CHAR_ALT elements)
byte_dfa build_segment_dfa(const llama_grammar_element * begin, const llama_grammar_element * end);
