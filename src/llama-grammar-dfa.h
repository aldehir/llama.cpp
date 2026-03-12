#pragma once

#include <array>
#include <bitset>
#include <cstdint>
#include <map>
#include <set>
#include <vector>

struct llama_grammar_element;

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

    // Minimize using Hopcroft's algorithm
    void minimize();

    // Complement: flip accept/reject states (except dead state 0)
    void complement();

    // Product construction: intersection of two DFAs
    static byte_dfa intersect(const byte_dfa & a, const byte_dfa & b);
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
// Compiled grammar (immutable after construction)
// ============================================================

struct compiled_grammar {
    std::vector<byte_dfa>      dfas;
    std::vector<compiled_rule> rules;
    uint32_t                   start_rule;

    // Compile from parsed grammar rules
    static compiled_grammar compile(
        const std::vector<std::vector<llama_grammar_element>> & parsed_rules,
        uint32_t start_rule_index);

    // Initialize runtime configs for the start rule
    std::vector<parse_config> init_configs() const;

    // Advance configs through one byte
    void accept_byte(std::vector<parse_config> & configs, uint8_t byte) const;

    // Get set of valid next bytes across all configs
    std::bitset<256> get_valid_bytes(const std::vector<parse_config> & configs) const;

    // Check if any config is in a "complete" state (rule finished, stack empty)
    bool any_config_complete(const std::vector<parse_config> & configs) const;

    // Left-factor rules to reduce runtime config proliferation.
    // Merges alternates that share common prefixes (identical RULE_CALL or DFA_MATCH segments).
    void left_factor();

private:
    // Left-factor a single rule, potentially creating new suffix rules.
    // Returns true if the rule was modified.
    bool left_factor_rule(uint32_t rule_id);
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
