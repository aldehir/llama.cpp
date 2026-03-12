#include "llama-grammar-dfa.h"
#include "llama-grammar.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <map>
#include <queue>
#include <set>

// ============================================================
// NFA implementation
// ============================================================

uint32_t grammar_nfa::add_state() {
    uint32_t id = (uint32_t) states.size();
    states.emplace_back();
    return id;
}

uint32_t grammar_nfa::merge(const grammar_nfa & other) {
    uint32_t offset = (uint32_t) states.size();
    states.reserve(states.size() + other.states.size());
    for (const auto & s : other.states) {
        auto & ns = states.emplace_back();
        for (const auto & t : s.byte_transitions) {
            ns.byte_transitions.push_back({t.lo, t.hi, t.target + offset});
        }
        for (uint32_t e : s.epsilon_transitions) {
            ns.epsilon_transitions.push_back(e + offset);
        }
    }
    return offset;
}

grammar_nfa grammar_nfa::byte_match(uint8_t lo, uint8_t hi) {
    grammar_nfa nfa;
    nfa.start  = nfa.add_state();
    nfa.accept = nfa.add_state();
    nfa.states[nfa.start].byte_transitions.push_back({lo, hi, nfa.accept});
    return nfa;
}

grammar_nfa grammar_nfa::concat(grammar_nfa a, grammar_nfa b) {
    uint32_t offset = a.merge(b);
    // Connect a's accept to b's start via epsilon
    a.states[a.accept].epsilon_transitions.push_back(b.start + offset);
    a.accept = b.accept + offset;
    return a;
}

grammar_nfa grammar_nfa::alternation(std::vector<grammar_nfa> parts) {
    if (parts.empty()) {
        return epsilon_nfa();
    }
    if (parts.size() == 1) {
        return std::move(parts[0]);
    }

    grammar_nfa result;
    uint32_t new_start  = result.add_state();
    uint32_t new_accept = result.add_state();
    result.start  = new_start;
    result.accept = new_accept;

    for (auto & part : parts) {
        uint32_t offset = result.merge(part);
        result.states[new_start].epsilon_transitions.push_back(part.start + offset);
        result.states[part.accept + offset].epsilon_transitions.push_back(new_accept);
    }
    return result;
}

grammar_nfa grammar_nfa::epsilon_nfa() {
    grammar_nfa nfa;
    nfa.start  = nfa.add_state();
    nfa.accept = nfa.add_state();
    nfa.states[nfa.start].epsilon_transitions.push_back(nfa.accept);
    return nfa;
}

// ============================================================
// UTF-8 encoding
// ============================================================

std::vector<uint8_t> encode_utf8_bytes(uint32_t cp) {
    std::vector<uint8_t> result;
    if (cp <= 0x7F) {
        result.push_back((uint8_t) cp);
    } else if (cp <= 0x7FF) {
        result.push_back((uint8_t)(0xC0 | (cp >> 6)));
        result.push_back((uint8_t)(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        result.push_back((uint8_t)(0xE0 | (cp >> 12)));
        result.push_back((uint8_t)(0x80 | ((cp >> 6) & 0x3F)));
        result.push_back((uint8_t)(0x80 | (cp & 0x3F)));
    } else if (cp <= 0x10FFFF) {
        result.push_back((uint8_t)(0xF0 | (cp >> 18)));
        result.push_back((uint8_t)(0x80 | ((cp >> 12) & 0x3F)));
        result.push_back((uint8_t)(0x80 | ((cp >> 6) & 0x3F)));
        result.push_back((uint8_t)(0x80 | (cp & 0x3F)));
    }
    return result;
}

// Build NFA for a single code point (matches its UTF-8 byte sequence)
static grammar_nfa single_codepoint_nfa(uint32_t cp) {
    auto bytes = encode_utf8_bytes(cp);
    grammar_nfa result = grammar_nfa::byte_match(bytes[0], bytes[0]);
    for (size_t i = 1; i < bytes.size(); i++) {
        result = grammar_nfa::concat(std::move(result),
                                     grammar_nfa::byte_match(bytes[i], bytes[i]));
    }
    return result;
}

// Build NFA for a byte-level range within a single UTF-8 encoding length.
// lo_bytes and hi_bytes must have the same length.
static grammar_nfa byte_range_nfa_recursive(
        const std::vector<uint8_t> & lo_bytes,
        const std::vector<uint8_t> & hi_bytes,
        size_t pos) {
    assert(lo_bytes.size() == hi_bytes.size());
    size_t n = lo_bytes.size();

    if (pos == n - 1) {
        // Last byte position: simple byte range
        return grammar_nfa::byte_match(lo_bytes[pos], hi_bytes[pos]);
    }

    if (lo_bytes[pos] == hi_bytes[pos]) {
        // Same prefix byte — emit it, recurse on suffix
        return grammar_nfa::concat(
            grammar_nfa::byte_match(lo_bytes[pos], lo_bytes[pos]),
            byte_range_nfa_recursive(lo_bytes, hi_bytes, pos + 1));
    }

    // Different prefix bytes — split into up to three parts
    std::vector<grammar_nfa> parts;

    // Part 1: lo[pos] with restricted continuation (lo[pos+1:]..0xBF)
    {
        auto hi1 = lo_bytes;
        for (size_t i = pos + 1; i < n; i++) {
            hi1[i] = 0xBF;
        }
        parts.push_back(grammar_nfa::concat(
            grammar_nfa::byte_match(lo_bytes[pos], lo_bytes[pos]),
            byte_range_nfa_recursive(lo_bytes, hi1, pos + 1)));
    }

    // Part 2: middle prefix bytes with full continuation range [0x80..0xBF]
    if (lo_bytes[pos] + 1 < hi_bytes[pos]) {
        // Build full continuation NFA (all remaining bytes are [0x80..0xBF])
        grammar_nfa cont = grammar_nfa::byte_match(0x80, 0xBF);
        for (size_t i = pos + 2; i < n; i++) {
            cont = grammar_nfa::concat(std::move(cont),
                                       grammar_nfa::byte_match(0x80, 0xBF));
        }
        parts.push_back(grammar_nfa::concat(
            grammar_nfa::byte_match((uint8_t)(lo_bytes[pos] + 1),
                                    (uint8_t)(hi_bytes[pos] - 1)),
            std::move(cont)));
    }

    // Part 3: hi[pos] with restricted continuation (0x80..hi[pos+1:])
    {
        auto lo3 = hi_bytes;
        for (size_t i = pos + 1; i < n; i++) {
            lo3[i] = 0x80;
        }
        parts.push_back(grammar_nfa::concat(
            grammar_nfa::byte_match(hi_bytes[pos], hi_bytes[pos]),
            byte_range_nfa_recursive(lo3, hi_bytes, pos + 1)));
    }

    return grammar_nfa::alternation(std::move(parts));
}

// Build NFA for a code point range within a single UTF-8 length class
static grammar_nfa single_length_range_nfa(uint32_t lo, uint32_t hi) {
    auto lo_bytes = encode_utf8_bytes(lo);
    auto hi_bytes = encode_utf8_bytes(hi);
    assert(lo_bytes.size() == hi_bytes.size());
    return byte_range_nfa_recursive(lo_bytes, hi_bytes, 0);
}

grammar_nfa codepoint_range_to_byte_nfa(uint32_t lo, uint32_t hi) {
    if (lo > hi) {
        // Empty range: return NFA that accepts nothing
        grammar_nfa nfa;
        nfa.start  = nfa.add_state();
        nfa.accept = nfa.add_state();
        // No transitions — nothing reaches accept
        return nfa;
    }

    // UTF-8 length boundaries
    static const uint32_t boundaries[] = {0x80, 0x800, 0x10000};
    std::vector<std::pair<uint32_t, uint32_t>> ranges;

    uint32_t cur = lo;
    for (uint32_t b : boundaries) {
        if (cur < b && b <= hi) {
            ranges.push_back({cur, b - 1});
            cur = b;
        }
    }
    ranges.push_back({cur, hi});

    std::vector<grammar_nfa> parts;
    for (auto [r_lo, r_hi] : ranges) {
        parts.push_back(single_length_range_nfa(r_lo, r_hi));
    }

    return grammar_nfa::alternation(std::move(parts));
}

// ============================================================
// Valid UTF-8 character NFA / DFA
// ============================================================

grammar_nfa valid_utf8_char_nfa() {
    // Build NFA that accepts any single valid UTF-8 character
    // This covers:
    //   [\x00-\x7F]                              (1-byte)
    //   [\xC2-\xDF] [\x80-\xBF]                  (2-byte, excluding overlong)
    //   \xE0 [\xA0-\xBF] [\x80-\xBF]             (3-byte, first restricted)
    //   [\xE1-\xEC] [\x80-\xBF] [\x80-\xBF]      (3-byte)
    //   \xED [\x80-\x9F] [\x80-\xBF]             (3-byte, surrogate excluded)
    //   [\xEE-\xEF] [\x80-\xBF] [\x80-\xBF]      (3-byte)
    //   \xF0 [\x90-\xBF] [\x80-\xBF] [\x80-\xBF] (4-byte, first restricted)
    //   [\xF1-\xF3] [\x80-\xBF] [\x80-\xBF] [\x80-\xBF] (4-byte)
    //   \xF4 [\x80-\x8F] [\x80-\xBF] [\x80-\xBF] (4-byte, last restricted)

    auto cont = [](){ return grammar_nfa::byte_match(0x80, 0xBF); };

    std::vector<grammar_nfa> alts;

    // 1-byte: 0x00-0x7F
    alts.push_back(grammar_nfa::byte_match(0x00, 0x7F));

    // 2-byte: C2-DF, 80-BF
    alts.push_back(grammar_nfa::concat(
        grammar_nfa::byte_match(0xC2, 0xDF), cont()));

    // 3-byte: E0, A0-BF, 80-BF
    alts.push_back(grammar_nfa::concat(grammar_nfa::concat(
        grammar_nfa::byte_match(0xE0, 0xE0),
        grammar_nfa::byte_match(0xA0, 0xBF)), cont()));

    // 3-byte: E1-EC, 80-BF, 80-BF
    alts.push_back(grammar_nfa::concat(grammar_nfa::concat(
        grammar_nfa::byte_match(0xE1, 0xEC), cont()), cont()));

    // 3-byte: ED, 80-9F, 80-BF (surrogate pair excluded)
    alts.push_back(grammar_nfa::concat(grammar_nfa::concat(
        grammar_nfa::byte_match(0xED, 0xED),
        grammar_nfa::byte_match(0x80, 0x9F)), cont()));

    // 3-byte: EE-EF, 80-BF, 80-BF
    alts.push_back(grammar_nfa::concat(grammar_nfa::concat(
        grammar_nfa::byte_match(0xEE, 0xEF), cont()), cont()));

    // 4-byte: F0, 90-BF, 80-BF, 80-BF
    alts.push_back(grammar_nfa::concat(grammar_nfa::concat(grammar_nfa::concat(
        grammar_nfa::byte_match(0xF0, 0xF0),
        grammar_nfa::byte_match(0x90, 0xBF)), cont()), cont()));

    // 4-byte: F1-F3, 80-BF, 80-BF, 80-BF
    alts.push_back(grammar_nfa::concat(grammar_nfa::concat(grammar_nfa::concat(
        grammar_nfa::byte_match(0xF1, 0xF3), cont()), cont()), cont()));

    // 4-byte: F4, 80-8F, 80-BF, 80-BF
    alts.push_back(grammar_nfa::concat(grammar_nfa::concat(grammar_nfa::concat(
        grammar_nfa::byte_match(0xF4, 0xF4),
        grammar_nfa::byte_match(0x80, 0x8F)), cont()), cont()));

    return grammar_nfa::alternation(std::move(alts));
}

byte_dfa build_valid_utf8_dfa() {
    auto nfa = valid_utf8_char_nfa();
    auto dfa = byte_dfa::from_nfa(nfa);
    dfa.minimize();
    return dfa;
}

// ============================================================
// NFA -> DFA (subset construction)
// ============================================================

// Compute epsilon closure of a set of NFA states
static std::set<uint32_t> epsilon_closure(
        const grammar_nfa & nfa,
        const std::set<uint32_t> & states) {
    std::set<uint32_t> closure = states;
    std::queue<uint32_t> worklist;
    for (uint32_t s : states) {
        worklist.push(s);
    }
    while (!worklist.empty()) {
        uint32_t s = worklist.front();
        worklist.pop();
        for (uint32_t t : nfa.states[s].epsilon_transitions) {
            if (closure.insert(t).second) {
                worklist.push(t);
            }
        }
    }
    return closure;
}

byte_dfa byte_dfa::from_nfa(const grammar_nfa & nfa) {
    byte_dfa dfa;

    // State 0 is the dead state
    dfa.transitions.emplace_back();
    dfa.transitions[0].fill(0);
    dfa.accept.push_back(false);

    // Map from NFA state sets to DFA state IDs
    std::map<std::set<uint32_t>, uint16_t> state_map;

    // Start state
    std::set<uint32_t> start_set = epsilon_closure(nfa, {nfa.start});
    uint16_t start_id = (uint16_t) dfa.transitions.size();
    dfa.transitions.emplace_back();
    dfa.transitions[start_id].fill(0);
    dfa.accept.push_back(start_set.count(nfa.accept) > 0);
    state_map[start_set] = start_id;
    dfa.start_state = start_id;

    std::queue<std::set<uint32_t>> worklist;
    worklist.push(start_set);

    while (!worklist.empty()) {
        auto current_set = worklist.front();
        worklist.pop();
        uint16_t current_id = state_map[current_set];

        // For each byte value, compute the set of NFA states reachable
        for (int b = 0; b < 256; b++) {
            std::set<uint32_t> next_set;
            for (uint32_t s : current_set) {
                for (const auto & t : nfa.states[s].byte_transitions) {
                    if (b >= t.lo && b <= t.hi) {
                        next_set.insert(t.target);
                    }
                }
            }
            if (next_set.empty()) {
                // Goes to dead state
                continue;
            }
            next_set = epsilon_closure(nfa, next_set);
            if (next_set.empty()) {
                continue;
            }

            auto it = state_map.find(next_set);
            if (it != state_map.end()) {
                dfa.transitions[current_id][b] = it->second;
            } else {
                uint16_t new_id = (uint16_t) dfa.transitions.size();
                // Safety check: don't overflow uint16_t
                if (new_id == 0xFFFF) {
                    // Too many DFA states; stop expanding (will result in incorrect rejection)
                    // In practice this shouldn't happen for typical grammars
                    break;
                }
                dfa.transitions.emplace_back();
                dfa.transitions[new_id].fill(0);
                dfa.accept.push_back(next_set.count(nfa.accept) > 0);
                state_map[next_set] = new_id;
                dfa.transitions[current_id][b] = new_id;
                worklist.push(next_set);
            }
        }
    }

    return dfa;
}

// ============================================================
// DFA minimization (Hopcroft's algorithm)
// ============================================================

void byte_dfa::minimize() {
    size_t n = transitions.size();
    if (n <= 2) {
        return;  // Dead state + at most one other state; nothing to minimize
    }

    // Partition states into accept / non-accept groups
    // State 0 (dead) is always in the non-accept group
    std::vector<uint16_t> state_to_group(n, 0);
    uint16_t num_groups = 2;

    // Group 0 = non-accept, Group 1 = accept
    for (size_t i = 0; i < n; i++) {
        state_to_group[i] = accept[i] ? 1 : 0;
    }

    // Check if all states are in the same group (all accept or all non-accept)
    bool has_accept = false, has_non_accept = false;
    for (size_t i = 0; i < n; i++) {
        if (accept[i]) has_accept = true;
        else has_non_accept = true;
    }
    if (!has_accept || !has_non_accept) {
        return;  // Nothing to minimize
    }

    // Iterative refinement
    bool changed = true;
    while (changed) {
        changed = false;
        std::map<std::pair<uint16_t, std::array<uint16_t, 256>>, uint16_t> signature_map;
        std::vector<uint16_t> new_groups(n, 0);
        uint16_t next_group = 0;

        for (size_t i = 0; i < n; i++) {
            // Build signature: (current group, transition group for each byte)
            std::array<uint16_t, 256> trans_sig;
            for (int b = 0; b < 256; b++) {
                trans_sig[b] = state_to_group[transitions[i][b]];
            }
            auto key = std::make_pair(state_to_group[i], trans_sig);
            auto it = signature_map.find(key);
            if (it == signature_map.end()) {
                signature_map[key] = next_group;
                new_groups[i] = next_group;
                next_group++;
            } else {
                new_groups[i] = it->second;
            }
        }

        if (next_group != num_groups) {
            changed = true;
            num_groups = next_group;
            state_to_group = new_groups;
        }
    }

    // Build minimized DFA
    byte_dfa minimized;
    minimized.transitions.resize(num_groups);
    minimized.accept.resize(num_groups, false);

    // Find which group contains state 0 (dead state)
    uint16_t dead_group = state_to_group[0];

    // Remap groups so dead state is group 0
    std::vector<uint16_t> group_remap(num_groups, 0);
    if (dead_group != 0) {
        // Swap group 0 and dead_group
        uint16_t next = 0;
        for (uint16_t g = 0; g < num_groups; g++) {
            if (g == dead_group) {
                group_remap[g] = 0;
            } else if (g == 0) {
                group_remap[g] = dead_group;
            } else {
                group_remap[g] = g;
            }
        }
        // Re-apply remapping
        for (size_t i = 0; i < n; i++) {
            state_to_group[i] = group_remap[state_to_group[i]];
        }
    }

    // Set transitions and accept states for each group
    // Use the first state in each group as the representative
    std::vector<bool> group_set(num_groups, false);
    for (size_t i = 0; i < n; i++) {
        uint16_t g = state_to_group[i];
        if (!group_set[g]) {
            group_set[g] = true;
            minimized.accept[g] = accept[i];
            for (int b = 0; b < 256; b++) {
                minimized.transitions[g][b] = state_to_group[transitions[i][b]];
            }
        }
    }

    // Dead state must have all-zero transitions and not be accept
    minimized.transitions[0].fill(0);
    minimized.accept[0] = false;

    minimized.start_state = state_to_group[start_state];

    *this = std::move(minimized);
}

// ============================================================
// DFA complement
// ============================================================

void byte_dfa::complement() {
    // After complement, the old dead state (state 0, all self-loop transitions) should
    // become accept — it represents "the input was not matched by the original DFA".
    // But our convention requires state 0 to be dead/non-accept.
    // Fix: move old state 0 content to a new state, then clear state 0 as truly dead.

    // Step 1: Add a new state that takes over old state 0's role
    uint16_t old_dead_replacement = (uint16_t) transitions.size();
    transitions.push_back(transitions[0]);  // copy old dead state's transitions
    accept.push_back(false);                // will be flipped below

    // Step 2: Remap all references from state 0 to old_dead_replacement
    for (size_t i = 0; i < transitions.size(); i++) {
        for (int b = 0; b < 256; b++) {
            if (transitions[i][b] == 0) {
                transitions[i][b] = old_dead_replacement;
            }
        }
    }

    if (start_state == 0) {
        start_state = old_dead_replacement;
    }

    // Step 3: Make state 0 the new dead state (truly unreachable, non-accept)
    transitions[0].fill(0);
    accept[0] = false;

    // Step 4: Flip accept bits for all non-dead states
    for (size_t i = 1; i < accept.size(); i++) {
        accept[i] = !accept[i];
    }
}

// ============================================================
// DFA intersection (product construction)
// ============================================================

byte_dfa byte_dfa::intersect(const byte_dfa & a, const byte_dfa & b) {
    byte_dfa result;

    // State 0 is dead
    result.transitions.emplace_back();
    result.transitions[0].fill(0);
    result.accept.push_back(false);

    size_t na = a.transitions.size();
    size_t nb = b.transitions.size();

    // Map from (a_state, b_state) to result state
    std::map<std::pair<uint16_t, uint16_t>, uint16_t> state_map;
    state_map[{0, 0}] = 0;  // dead × dead = dead

    auto get_or_create = [&](uint16_t sa, uint16_t sb) -> uint16_t {
        auto key = std::make_pair(sa, sb);
        auto it = state_map.find(key);
        if (it != state_map.end()) {
            return it->second;
        }
        uint16_t new_id = (uint16_t) result.transitions.size();
        if (new_id == 0xFFFF) {
            return 0;  // overflow protection
        }
        result.transitions.emplace_back();
        result.transitions[new_id].fill(0);
        result.accept.push_back(
            (sa < na ? a.accept[sa] : false) &&
            (sb < nb ? b.accept[sb] : false));
        state_map[key] = new_id;
        return new_id;
    };

    uint16_t start = get_or_create(a.start_state, b.start_state);
    result.start_state = start;

    std::queue<std::pair<uint16_t, uint16_t>> worklist;
    worklist.push({a.start_state, b.start_state});

    while (!worklist.empty()) {
        auto [sa, sb] = worklist.front();
        worklist.pop();
        uint16_t current = state_map[{sa, sb}];

        for (int byte_val = 0; byte_val < 256; byte_val++) {
            uint16_t na_state = (sa < na) ? a.transitions[sa][byte_val] : 0;
            uint16_t nb_state = (sb < nb) ? b.transitions[sb][byte_val] : 0;

            if (na_state == 0 || nb_state == 0) {
                // One of them goes to dead → product goes to dead
                result.transitions[current][byte_val] = 0;
                continue;
            }

            auto key = std::make_pair(na_state, nb_state);
            bool is_new = (state_map.find(key) == state_map.end());
            uint16_t target = get_or_create(na_state, nb_state);
            result.transitions[current][byte_val] = target;

            if (is_new) {
                worklist.push({na_state, nb_state});
            }
        }
    }

    return result;
}

// ============================================================
// Build DFA for a terminal segment
// ============================================================

// Build NFA for one grammar element (character match / range / negated / wildcard)
// 'pos' points to the start of the element, returns pointer past the element
static grammar_nfa element_to_nfa(
        const llama_grammar_element * & pos,
        bool & is_negated) {
    is_negated = false;

    if (pos->type == LLAMA_GRETYPE_CHAR_ANY) {
        pos++;
        // Skip any trailing CHAR_ALT (shouldn't happen for CHAR_ANY but be safe)
        while (pos->type == LLAMA_GRETYPE_CHAR_ALT) {
            pos++;
        }
        return valid_utf8_char_nfa();
    }

    if (pos->type == LLAMA_GRETYPE_CHAR_NOT) {
        is_negated = true;
    }

    // Collect all the character/range alternatives
    std::vector<grammar_nfa> char_alts;

    bool first = true;
    do {
        uint32_t cp = pos->value;
        if (pos[1].type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
            // Range: [cp..upper]
            uint32_t upper = pos[1].value;
            char_alts.push_back(codepoint_range_to_byte_nfa(cp, upper));
            pos += 2;
        } else {
            // Single code point
            char_alts.push_back(single_codepoint_nfa(cp));
            pos += 1;
        }
        first = false;
    } while (pos->type == LLAMA_GRETYPE_CHAR_ALT);

    return grammar_nfa::alternation(std::move(char_alts));
}

byte_dfa build_segment_dfa(const llama_grammar_element * begin, const llama_grammar_element * end) {
    const llama_grammar_element * pos = begin;

    // Build concatenation of all terminal elements in this segment
    grammar_nfa segment_nfa = grammar_nfa::epsilon_nfa();
    bool first = true;

    while (pos < end) {
        if (pos->type == LLAMA_GRETYPE_END ||
            pos->type == LLAMA_GRETYPE_ALT ||
            pos->type == LLAMA_GRETYPE_RULE_REF) {
            break;
        }

        bool is_negated = false;
        grammar_nfa elem_nfa = element_to_nfa(pos, is_negated);

        if (is_negated) {
            // Build DFA for positive match, complement, intersect with valid-UTF-8
            auto positive_dfa = byte_dfa::from_nfa(elem_nfa);
            positive_dfa.minimize();
            positive_dfa.complement();
            auto utf8_dfa = build_valid_utf8_dfa();
            auto negated_dfa = byte_dfa::intersect(positive_dfa, utf8_dfa);
            negated_dfa.minimize();

            // Convert back to NFA-like form by wrapping in a DFA
            // Actually, we need to concatenate this DFA with the rest of the segment.
            // For simplicity, if we have a negated element, compile it to a standalone DFA
            // and then concatenate NFAs. We can convert a small DFA back to an NFA.

            // Convert DFA back to NFA for concatenation
            grammar_nfa neg_nfa;
            uint32_t base = 0;
            // Create states in NFA corresponding to DFA states
            for (size_t i = 0; i < negated_dfa.transitions.size(); i++) {
                neg_nfa.add_state();
            }
            neg_nfa.start = negated_dfa.start_state;
            // Create a single accept state
            uint32_t nfa_accept = neg_nfa.add_state();
            neg_nfa.accept = nfa_accept;

            // Add epsilon from DFA accept states to NFA accept
            for (size_t i = 0; i < negated_dfa.transitions.size(); i++) {
                if (negated_dfa.accept[i]) {
                    neg_nfa.states[i].epsilon_transitions.push_back(nfa_accept);
                }
            }

            // Add byte transitions
            for (size_t i = 0; i < negated_dfa.transitions.size(); i++) {
                // Group consecutive bytes with same target for efficiency
                int b = 0;
                while (b < 256) {
                    uint16_t target = negated_dfa.transitions[i][b];
                    if (target == 0) {
                        b++;
                        continue;
                    }
                    int start_b = b;
                    while (b < 255 && negated_dfa.transitions[i][b + 1] == target) {
                        b++;
                    }
                    neg_nfa.states[i].byte_transitions.push_back(
                        {(uint8_t)start_b, (uint8_t)b, (uint32_t)target});
                    b++;
                }
            }

            if (first) {
                segment_nfa = std::move(neg_nfa);
                first = false;
            } else {
                segment_nfa = grammar_nfa::concat(std::move(segment_nfa), std::move(neg_nfa));
            }
        } else {
            if (first) {
                segment_nfa = std::move(elem_nfa);
                first = false;
            } else {
                segment_nfa = grammar_nfa::concat(std::move(segment_nfa), std::move(elem_nfa));
            }
        }
    }

    auto dfa = byte_dfa::from_nfa(segment_nfa);
    dfa.minimize();
    return dfa;
}

// ============================================================
// Grammar compilation
// ============================================================

compiled_grammar compiled_grammar::compile(
        const std::vector<std::vector<llama_grammar_element>> & parsed_rules,
        uint32_t start_rule_index) {
    compiled_grammar cg;
    cg.start_rule = start_rule_index;
    cg.rules.resize(parsed_rules.size());

    for (size_t rule_idx = 0; rule_idx < parsed_rules.size(); rule_idx++) {
        const auto & rule = parsed_rules[rule_idx];
        auto & compiled = cg.rules[rule_idx];

        // Walk through the rule, splitting at ALT boundaries into alternates
        const llama_grammar_element * pos = rule.data();
        const llama_grammar_element * rule_end = rule.data() + rule.size();

        while (pos < rule_end) {
            // Start a new alternate
            std::vector<compiled_segment> segments;

            while (pos < rule_end && pos->type != LLAMA_GRETYPE_ALT && pos->type != LLAMA_GRETYPE_END) {
                if (pos->type == LLAMA_GRETYPE_RULE_REF) {
                    compiled_segment seg;
                    seg.type = compiled_segment::RULE_CALL;
                    seg.id   = pos->value;
                    segments.push_back(seg);
                    pos++;
                } else {
                    // Terminal element(s) — find extent of consecutive terminal elements
                    const llama_grammar_element * seg_start = pos;
                    while (pos < rule_end &&
                           pos->type != LLAMA_GRETYPE_RULE_REF &&
                           pos->type != LLAMA_GRETYPE_ALT &&
                           pos->type != LLAMA_GRETYPE_END) {
                        pos++;
                    }
                    // Compile terminal segment to DFA
                    auto dfa = build_segment_dfa(seg_start, pos);
                    uint32_t dfa_id = (uint32_t) cg.dfas.size();
                    cg.dfas.push_back(std::move(dfa));

                    compiled_segment seg;
                    seg.type = compiled_segment::DFA_MATCH;
                    seg.id   = dfa_id;
                    segments.push_back(seg);
                }
            }

            compiled.alternates.push_back(std::move(segments));

            // Skip ALT or END
            if (pos < rule_end) {
                pos++;
            }
        }

        // If rule is empty (just END), add one empty alternate
        if (compiled.alternates.empty()) {
            compiled.alternates.push_back({});
        }
    }

    return cg;
}

// ============================================================
// Runtime: config initialization and advancement
// ============================================================

void compiled_grammar::advance_to_terminal(
        const parse_config & cfg,
        std::vector<parse_config> & out_configs) const {
    const auto & rule = rules[cfg.current.rule_id];

    // Check if we've gone past the end of this alternate's segments
    if (cfg.current.alternate_idx >= rule.alternates.size()) {
        return;  // invalid alternate
    }

    const auto & alt = rule.alternates[cfg.current.alternate_idx];

    if (cfg.current.segment_idx >= alt.size()) {
        // End of this alternate — rule is complete
        if (cfg.call_stack.empty()) {
            // Complete parse — add as a "complete" config
            out_configs.push_back(cfg);
            return;
        }

        // Pop call stack and resume caller
        parse_config resumed = cfg;
        const auto & frame = resumed.call_stack.back();
        resumed.current.rule_id       = frame.rule_id;
        resumed.current.alternate_idx = frame.alternate_idx;
        resumed.current.segment_idx   = frame.segment_idx;
        resumed.current.dfa_state     = frame.dfa_state;
        resumed.call_stack.pop_back();

        // Recurse to continue advancing (might hit another RULE_CALL or another end-of-rule)
        advance_to_terminal(resumed, out_configs);
        return;
    }

    const auto & seg = alt[cfg.current.segment_idx];

    if (seg.type == compiled_segment::DFA_MATCH) {
        // We're at a DFA segment — set up initial DFA state
        parse_config ready = cfg;
        ready.current.dfa_state = dfas[seg.id].start_state;
        out_configs.push_back(ready);
        return;
    }

    // RULE_CALL — push call frame and enter the called rule
    assert(seg.type == compiled_segment::RULE_CALL);

    const auto & called_rule = rules[seg.id];

    // For each alternate of the called rule, create a config
    for (size_t alt_idx = 0; alt_idx < called_rule.alternates.size(); alt_idx++) {
        parse_config new_cfg;
        new_cfg.call_stack = cfg.call_stack;

        // Push return frame: resume at the NEXT segment after this RULE_CALL
        grammar_call_frame frame;
        frame.rule_id       = cfg.current.rule_id;
        frame.alternate_idx = cfg.current.alternate_idx;
        frame.segment_idx   = cfg.current.segment_idx + 1;
        frame.dfa_state     = 0;  // will be set when we reach a DFA segment
        new_cfg.call_stack.push_back(frame);

        new_cfg.current.rule_id       = seg.id;
        new_cfg.current.alternate_idx = (uint8_t) alt_idx;
        new_cfg.current.segment_idx   = 0;
        new_cfg.current.dfa_state     = 0;

        // Recursively advance until we reach a DFA segment or complete
        advance_to_terminal(new_cfg, out_configs);
    }
}

std::vector<parse_config> compiled_grammar::init_configs() const {
    std::vector<parse_config> configs;

    const auto & start = rules[start_rule];
    for (size_t alt_idx = 0; alt_idx < start.alternates.size(); alt_idx++) {
        parse_config cfg;
        cfg.current.rule_id       = start_rule;
        cfg.current.alternate_idx = (uint8_t) alt_idx;
        cfg.current.segment_idx   = 0;
        cfg.current.dfa_state     = 0;

        advance_to_terminal(cfg, configs);
    }

    dedup_configs(configs);
    return configs;
}

void compiled_grammar::dedup_configs(std::vector<parse_config> & configs) {
    // Remove duplicate configs
    std::vector<parse_config> unique;
    unique.reserve(configs.size());
    for (auto & cfg : configs) {
        bool found = false;
        for (const auto & u : unique) {
            if (cfg == u) {
                found = true;
                break;
            }
        }
        if (!found) {
            unique.push_back(std::move(cfg));
        }
    }
    configs = std::move(unique);
}

void compiled_grammar::accept_byte(std::vector<parse_config> & configs, uint8_t byte) const {
    std::vector<parse_config> new_configs;
    new_configs.reserve(configs.size());

    for (auto & cfg : configs) {
        const auto & rule = rules[cfg.current.rule_id];
        const auto & alt = rule.alternates[cfg.current.alternate_idx];

        // If this config is "complete" (past end of segments with empty stack),
        // it doesn't consume bytes
        if (cfg.current.segment_idx >= alt.size()) {
            continue;
        }

        const auto & seg = alt[cfg.current.segment_idx];
        if (seg.type != compiled_segment::DFA_MATCH) {
            // Shouldn't happen if advance_to_terminal was called correctly
            continue;
        }

        const auto & dfa = dfas[seg.id];
        uint16_t next_state = dfa.transitions[cfg.current.dfa_state][byte];

        if (next_state == 0) {
            // Dead state — config dies
            continue;
        }

        if (dfa.accept[next_state]) {
            // Segment complete — advance to next segment
            parse_config advanced = cfg;
            advanced.current.segment_idx++;
            advanced.current.dfa_state = 0;

            // Resolve rule calls / rule completions
            advance_to_terminal(advanced, new_configs);
        } else {
            // Still in the middle of a DFA segment
            parse_config updated = cfg;
            updated.current.dfa_state = next_state;
            new_configs.push_back(std::move(updated));
        }
    }

    dedup_configs(new_configs);
    configs = std::move(new_configs);
}

std::bitset<256> compiled_grammar::get_valid_bytes(const std::vector<parse_config> & configs) const {
    std::bitset<256> result;

    for (const auto & cfg : configs) {
        const auto & rule = rules[cfg.current.rule_id];
        const auto & alt = rule.alternates[cfg.current.alternate_idx];

        if (cfg.current.segment_idx >= alt.size()) {
            // Complete config — no bytes to consume
            continue;
        }

        const auto & seg = alt[cfg.current.segment_idx];
        if (seg.type != compiled_segment::DFA_MATCH) {
            continue;
        }

        const auto & dfa = dfas[seg.id];
        for (int b = 0; b < 256; b++) {
            if (dfa.transitions[cfg.current.dfa_state][b] != 0) {
                result.set(b);
            }
        }
    }

    return result;
}

bool compiled_grammar::any_config_complete(const std::vector<parse_config> & configs) const {
    for (const auto & cfg : configs) {
        const auto & rule = rules[cfg.current.rule_id];
        const auto & alt = rule.alternates[cfg.current.alternate_idx];

        if (cfg.current.segment_idx >= alt.size() && cfg.call_stack.empty()) {
            return true;
        }
    }
    return false;
}
