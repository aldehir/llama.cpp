#include "llama-grammar-dfa.h"
#include "llama-grammar.h"
#include "llama-impl.h"
#include "llama-vocab.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <map>
#include <queue>
#include <set>
#include <unordered_map>

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

grammar_nfa grammar_nfa::kleene_star(grammar_nfa body) {
    grammar_nfa result;
    uint32_t new_start  = result.add_state();
    uint32_t new_accept = result.add_state();
    result.start  = new_start;
    result.accept = new_accept;

    uint32_t offset = result.merge(body);
    uint32_t body_start  = body.start + offset;
    uint32_t body_accept = body.accept + offset;

    // Start can skip (zero iterations) or enter body
    result.states[new_start].epsilon_transitions.push_back(new_accept);
    result.states[new_start].epsilon_transitions.push_back(body_start);
    // After body, can loop or exit
    result.states[body_accept].epsilon_transitions.push_back(body_start);
    result.states[body_accept].epsilon_transitions.push_back(new_accept);

    return result;
}

grammar_nfa grammar_nfa::kleene_plus(grammar_nfa body) {
    // T+ = T with a loop: after accepting, can repeat
    body.states[body.accept].epsilon_transitions.push_back(body.start);
    return body;
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
    static const byte_dfa cached = []() {
        auto nfa = valid_utf8_char_nfa();
        auto dfa = byte_dfa::from_nfa(nfa);
        dfa.minimize();
        return dfa;
    }();
    return cached;
}

// ============================================================
// Exclusion DFA (Aho-Corasick complement)
// ============================================================

byte_dfa build_exclusion_dfa(const std::vector<std::vector<uint8_t>> & needles) {
    // Aho-Corasick trie node
    struct ac_node {
        int children[256];
        int fail;
        bool output;

        ac_node() : fail(0), output(false) {
            std::fill(children, children + 256, -1);
        }
    };

    std::vector<ac_node> nodes;
    nodes.emplace_back();  // root = node 0

    // Insert needles into trie
    for (const auto & needle : needles) {
        if (needle.empty()) {
            // Empty exclusion string = exclude everything = match nothing
            byte_dfa dfa;
            dfa.transitions.resize(2);
            dfa.transitions[0].fill(0);
            dfa.transitions[1].fill(0);
            dfa.accept.resize(2, false);
            dfa.accept_can_continue.resize(2, false);
            dfa.start_state = 1;
            return dfa;
        }
        int cur = 0;
        for (uint8_t b : needle) {
            if (nodes[cur].children[b] == -1) {
                nodes[cur].children[b] = (int)nodes.size();
                nodes.emplace_back();
            }
            cur = nodes[cur].children[b];
        }
        nodes[cur].output = true;
    }

    // Build failure links using BFS
    std::queue<int> bfs_queue;
    for (int b = 0; b < 256; b++) {
        if (nodes[0].children[b] != -1) {
            nodes[nodes[0].children[b]].fail = 0;
            bfs_queue.push(nodes[0].children[b]);
        }
    }

    while (!bfs_queue.empty()) {
        int u = bfs_queue.front();
        bfs_queue.pop();

        // Propagate output flag via suffix link
        if (nodes[nodes[u].fail].output) {
            nodes[u].output = true;
        }

        for (int b = 0; b < 256; b++) {
            int v = nodes[u].children[b];
            if (v == -1) {
                continue;
            }
            // Compute failure link for v: follow u's failure chain
            int f = nodes[u].fail;
            while (f != 0 && nodes[f].children[b] == -1) {
                f = nodes[f].fail;
            }
            if (nodes[f].children[b] != -1 && nodes[f].children[b] != v) {
                nodes[v].fail = nodes[f].children[b];
            } else {
                nodes[v].fail = 0;
            }
            bfs_queue.push(v);
        }
    }

    // Build complement DFA
    // DFA state 0 = dead, DFA state i+1 = AC node i
    int n_nodes = (int)nodes.size();
    byte_dfa dfa;
    dfa.transitions.resize(n_nodes + 1);
    dfa.accept.resize(n_nodes + 1, false);
    dfa.accept_can_continue.resize(n_nodes + 1, false);
    dfa.start_state = 1;  // AC root = DFA state 1

    // Dead state 0
    dfa.transitions[0].fill(0);

    for (int node_idx = 0; node_idx < n_nodes; node_idx++) {
        uint16_t dfa_state = (uint16_t)(node_idx + 1);

        if (nodes[node_idx].output) {
            // Needle matched at this node -> dead in complement
            dfa.transitions[dfa_state].fill(0);
            dfa.accept[dfa_state] = false;
            continue;
        }

        dfa.accept[dfa_state] = true;

        bool has_outgoing = false;
        for (int b = 0; b < 256; b++) {
            // Aho-Corasick goto: follow failure links to find target
            int target = node_idx;
            while (target != 0 && nodes[target].children[b] == -1) {
                target = nodes[target].fail;
            }
            if (nodes[target].children[b] != -1) {
                target = nodes[target].children[b];
            } else {
                target = 0;  // back to root
            }

            if (nodes[target].output) {
                dfa.transitions[dfa_state][b] = 0;  // needle completed -> dead
            } else {
                dfa.transitions[dfa_state][b] = (uint16_t)(target + 1);
                has_outgoing = true;
            }
        }

        dfa.accept_can_continue[dfa_state] = has_outgoing;
    }

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

    // Canonicalize state numbering via BFS from start state.
    // This ensures that two minimal DFAs for the same language are
    // structurally identical (same transitions, same state IDs),
    // enabling deduplication via operator==.
    {
        uint16_t n_min = (uint16_t) minimized.transitions.size();
        std::vector<uint16_t> remap(n_min, UINT16_MAX);
        remap[0] = 0;  // dead state stays 0
        uint16_t next_id = 1;

        // BFS from start state
        std::queue<uint16_t> bfs;
        if (minimized.start_state != 0) {
            remap[minimized.start_state] = next_id++;
            bfs.push(minimized.start_state);
        }

        while (!bfs.empty()) {
            uint16_t s = bfs.front();
            bfs.pop();
            for (int b = 0; b < 256; b++) {
                uint16_t t = minimized.transitions[s][b];
                if (t != 0 && remap[t] == UINT16_MAX) {
                    remap[t] = next_id++;
                    bfs.push(t);
                }
            }
        }

        // Assign IDs to any unreachable states (shouldn't happen for well-formed DFAs)
        for (uint16_t i = 0; i < n_min; i++) {
            if (remap[i] == UINT16_MAX) {
                remap[i] = next_id++;
            }
        }

        // Build canonicalized DFA
        byte_dfa canon;
        canon.transitions.resize(n_min);
        canon.accept.resize(n_min, false);
        for (uint16_t i = 0; i < n_min; i++) {
            canon.accept[remap[i]] = minimized.accept[i];
            for (int b = 0; b < 256; b++) {
                canon.transitions[remap[i]][b] = remap[minimized.transitions[i][b]];
            }
        }
        canon.start_state = remap[minimized.start_state];
        minimized = std::move(canon);
    }

    *this = std::move(minimized);

    // Precompute accept_can_continue: accept[s] && has any outgoing transition
    accept_can_continue.resize(accept.size(), false);
    for (size_t s = 0; s < accept.size(); s++) {
        if (!accept[s]) continue;
        for (int b = 0; b < 256; b++) {
            if (transitions[s][b] != 0) {
                accept_can_continue[s] = true;
                break;
            }
        }
    }
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

bool byte_dfa::operator==(const byte_dfa & other) const {
    return start_state == other.start_state &&
           accept == other.accept &&
           transitions == other.transitions;
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
    } while (pos->type == LLAMA_GRETYPE_CHAR_ALT);

    return grammar_nfa::alternation(std::move(char_alts));
}

// Build an NFA for a sequence of terminal grammar elements (stops at RULE_REF/ALT/END).
// Advances 'pos' past the consumed elements.
static grammar_nfa build_segment_nfa(const llama_grammar_element * & pos,
                                     const llama_grammar_element * end) {
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

            // Convert DFA back to NFA for concatenation
            elem_nfa = grammar_nfa();
            for (size_t i = 0; i < negated_dfa.transitions.size(); i++) {
                elem_nfa.add_state();
            }
            elem_nfa.start = negated_dfa.start_state;
            uint32_t nfa_accept = elem_nfa.add_state();
            elem_nfa.accept = nfa_accept;

            for (size_t i = 0; i < negated_dfa.transitions.size(); i++) {
                if (negated_dfa.accept[i]) {
                    elem_nfa.states[i].epsilon_transitions.push_back(nfa_accept);
                }
            }

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
                    elem_nfa.states[i].byte_transitions.push_back(
                        {(uint8_t)start_b, (uint8_t)b, (uint32_t)target});
                    b++;
                }
            }
        }

        if (first) {
            segment_nfa = std::move(elem_nfa);
            first = false;
        } else {
            segment_nfa = grammar_nfa::concat(std::move(segment_nfa), std::move(elem_nfa));
        }
    }

    return segment_nfa;
}

byte_dfa build_segment_dfa(const llama_grammar_element * begin, const llama_grammar_element * end) {
    const llama_grammar_element * pos = begin;
    auto nfa = build_segment_nfa(pos, end);
    auto dfa = byte_dfa::from_nfa(nfa);
    dfa.minimize();
    return dfa;
}

// ============================================================
// Grammar compilation
// ============================================================

// Helper: add a DFA to the compiled grammar, deduplicating against existing DFAs.
static uint32_t add_or_reuse_dfa(compiled_grammar & cg, byte_dfa && dfa) {
    for (uint32_t i = 0; i < (uint32_t) cg.dfas.size(); i++) {
        if (cg.dfas[i] == dfa) {
            return i;
        }
    }
    uint32_t id = (uint32_t) cg.dfas.size();
    cg.dfas.push_back(std::move(dfa));
    return id;
}

compiled_grammar compiled_grammar::compile(
        const std::vector<std::vector<llama_grammar_element>> & parsed_rules,
        uint32_t start_rule_index) {
    compiled_grammar cg;
    cg.start_rule = start_rule_index;
    cg.rules.resize(parsed_rules.size());

    // --- Detect Kleene star rules ---
    // Pattern: R ::= T... R | ε  (2 alternates, one empty, one = terminals + self-ref)
    // These are generated by the parser for +, *, {n,} repetitions.
    struct kleene_info {
        bool detected = false;
        size_t body_begin = 0;  // index of first terminal element
        size_t body_end = 0;    // index of RULE_REF(self), exclusive
    };
    std::vector<kleene_info> kleene(parsed_rules.size());

    for (size_t i = 0; i < parsed_rules.size(); i++) {
        const auto & rule = parsed_rules[i];
        if (rule.size() < 3) continue;  // need at least: terminal RULE_REF ALT/END

        // Parse alternates: collect [begin, end) index pairs
        struct alt_range { size_t begin; size_t end; };
        std::vector<alt_range> alts;
        size_t alt_begin = 0;
        for (size_t j = 0; j < rule.size(); j++) {
            if (rule[j].type == LLAMA_GRETYPE_ALT || rule[j].type == LLAMA_GRETYPE_END) {
                alts.push_back({alt_begin, j});
                alt_begin = j + 1;
            }
        }
        if (alts.size() != 2) continue;

        // Identify the empty alternate and the body alternate
        int empty_idx = -1;
        for (int a = 0; a < 2; a++) {
            if (alts[a].begin == alts[a].end) {
                empty_idx = a;
            }
        }
        if (empty_idx < 0) continue;
        int body_idx = 1 - empty_idx;

        size_t bb = alts[body_idx].begin;
        size_t be = alts[body_idx].end;
        if (bb >= be) continue;

        // Last element of body must be RULE_REF(self)
        if (rule[be - 1].type != LLAMA_GRETYPE_RULE_REF || rule[be - 1].value != i) continue;

        // Everything before RULE_REF must be terminal elements
        bool all_terminal = true;
        for (size_t j = bb; j < be - 1; j++) {
            if (rule[j].type == LLAMA_GRETYPE_RULE_REF ||
                rule[j].type == LLAMA_GRETYPE_ALT ||
                rule[j].type == LLAMA_GRETYPE_END) {
                all_terminal = false;
                break;
            }
        }
        if (!all_terminal || bb + 1 > be - 1) continue;  // need at least one terminal

        kleene[i] = {true, bb, be - 1};
        LLAMA_LOG_DEBUG("%s: rule %zu detected as Kleene star (body elements [%zu, %zu))\n",
                        __func__, i, bb, be - 1);
    }

    // --- Helper: check if terminal elements match a Kleene star's body ---
    auto elements_match = [&](const llama_grammar_element * seg_start,
                              const llama_grammar_element * seg_end,
                              uint32_t star_rule_idx) -> bool {
        const auto & ki = kleene[star_rule_idx];
        const auto & star_rule = parsed_rules[star_rule_idx];
        size_t seg_len = seg_end - seg_start;
        size_t body_len = ki.body_end - ki.body_begin;
        if (seg_len != body_len) return false;
        for (size_t k = 0; k < seg_len; k++) {
            if (seg_start[k].type  != star_rule[ki.body_begin + k].type ||
                seg_start[k].value != star_rule[ki.body_begin + k].value) {
                return false;
            }
        }
        return true;
    };

    // --- Helper: build a DFA(T+) from terminal elements ---
    auto build_plus_dfa = [&](const llama_grammar_element * begin,
                              const llama_grammar_element * end) -> byte_dfa {
        const llama_grammar_element * pos = begin;
        grammar_nfa body_nfa = build_segment_nfa(pos, end);
        grammar_nfa plus_nfa = grammar_nfa::kleene_plus(std::move(body_nfa));
        auto dfa = byte_dfa::from_nfa(plus_nfa);
        dfa.minimize();
        return dfa;
    };

    // --- Build compiled rules ---
    for (size_t rule_idx = 0; rule_idx < parsed_rules.size(); rule_idx++) {
        const auto & rule = parsed_rules[rule_idx];
        auto & compiled = cg.rules[rule_idx];

        // Kleene star rule: rewrite to DFA(T+) | ε
        if (kleene[rule_idx].detected) {
            const auto & ki = kleene[rule_idx];
            auto dfa = build_plus_dfa(&rule[ki.body_begin], &rule[ki.body_end]);
            uint32_t dfa_id = add_or_reuse_dfa(cg, std::move(dfa));

            // Alt 0: single DFA(T+) segment
            compiled.alternates.push_back({{compiled_segment::DFA_MATCH, dfa_id}});
            // Alt 1: empty (epsilon)
            compiled.alternates.push_back({});
            continue;
        }

        // Normal rule processing
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

                    // Detect Kleene plus pattern: T... followed by RULE_REF(T*)
                    // where the terminals match the Kleene star's body.
                    // Compile directly as DFA(T+) instead of DFA(T) + RULE_CALL(T*).
                    bool is_plus = false;
                    if (pos < rule_end && pos->type == LLAMA_GRETYPE_RULE_REF) {
                        uint32_t called = pos->value;
                        if (called < kleene.size() && kleene[called].detected &&
                            elements_match(seg_start, pos, called)) {
                            auto dfa = build_plus_dfa(seg_start, pos);
                            uint32_t dfa_id = add_or_reuse_dfa(cg, std::move(dfa));
                            segments.push_back({compiled_segment::DFA_MATCH, dfa_id});
                            pos++; // consume the RULE_REF
                            is_plus = true;
                            LLAMA_LOG_DEBUG("%s: rule %zu: detected T+ pattern (star rule %u) -> DFA(T+) id=%u\n",
                                            __func__, rule_idx, called, dfa_id);
                        }
                    }

                    if (!is_plus) {
                        // Normal terminal segment
                        auto dfa = build_segment_dfa(seg_start, pos);
                        uint32_t dfa_id = add_or_reuse_dfa(cg, std::move(dfa));
                        segments.push_back({compiled_segment::DFA_MATCH, dfa_id});
                    }
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

void compiled_grammar::compute_follow_sets() {
    const size_t n_rules = rules.size();
    const size_t n_dfas = dfas.size();

    // Step 1: nullable[rule_id] — true if any alternate is empty or all segments are nullable RULE_CALLs
    std::vector<bool> nullable(n_rules, false);
    for (bool changed = true; changed; ) {
        changed = false;
        for (size_t r = 0; r < n_rules; r++) {
            if (nullable[r]) continue;
            for (const auto & alt : rules[r].alternates) {
                bool alt_nullable = true;
                for (const auto & seg : alt) {
                    if (seg.type == compiled_segment::DFA_MATCH || !nullable[seg.id]) {
                        alt_nullable = false;
                        break;
                    }
                }
                if (alt_nullable) {
                    nullable[r] = true;
                    changed = true;
                    break;
                }
            }
        }
    }

    // Step 2: first_dfas[rule_id] — DFA IDs reachable from rule start
    std::vector<std::set<uint32_t>> first_dfas_set(n_rules);
    for (bool changed = true; changed; ) {
        changed = false;
        for (size_t r = 0; r < n_rules; r++) {
            size_t old_size = first_dfas_set[r].size();
            for (const auto & alt : rules[r].alternates) {
                for (const auto & seg : alt) {
                    if (seg.type == compiled_segment::DFA_MATCH) {
                        first_dfas_set[r].insert(seg.id);
                        break;
                    }
                    // RULE_CALL: add first_dfas of called rule
                    first_dfas_set[r].insert(first_dfas_set[seg.id].begin(),
                                             first_dfas_set[seg.id].end());
                    if (!nullable[seg.id]) break;
                }
            }
            if (first_dfas_set[r].size() != old_size) {
                changed = true;
            }
        }
    }

    // Step 3 & 4: follow_of_rule[rule_id] and dfa_follows[dfa_id]
    // For each segment, collect what can follow it.
    std::vector<std::set<uint32_t>> follow_of_rule(n_rules);
    std::vector<std::set<uint32_t>> dfa_follows_set(n_dfas);

    // Helper: collect DFA IDs following position [from_idx, end) in an alternate.
    // Returns true if end of alternate is reachable (all remaining segments nullable).
    auto collect_following = [&](const std::vector<compiled_segment> & alt, size_t from_idx,
                                 std::set<uint32_t> & target) -> bool {
        for (size_t i = from_idx; i < alt.size(); i++) {
            if (alt[i].type == compiled_segment::DFA_MATCH) {
                target.insert(alt[i].id);
                return false;
            }
            // RULE_CALL: add first_dfas of called rule
            target.insert(first_dfas_set[alt[i].id].begin(),
                          first_dfas_set[alt[i].id].end());
            if (!nullable[alt[i].id]) return false;
        }
        return true;
    };

    for (bool changed = true; changed; ) {
        changed = false;
        for (size_t r = 0; r < n_rules; r++) {
            for (const auto & alt : rules[r].alternates) {
                for (size_t s = 0; s < alt.size(); s++) {
                    std::set<uint32_t> & target =
                        (alt[s].type == compiled_segment::DFA_MATCH)
                            ? dfa_follows_set[alt[s].id]
                            : follow_of_rule[alt[s].id];
                    size_t old_size = target.size();
                    bool reaches_end = collect_following(alt, s + 1, target);
                    if (reaches_end) {
                        target.insert(follow_of_rule[r].begin(), follow_of_rule[r].end());
                    }
                    if (target.size() != old_size) {
                        changed = true;
                    }
                }
            }
        }
    }

    // Convert to sorted vectors
    dfa_follow_dfas.resize(n_dfas);
    for (size_t d = 0; d < n_dfas; d++) {
        dfa_follow_dfas[d].assign(dfa_follows_set[d].begin(), dfa_follows_set[d].end());
    }

    LLAMA_LOG_DEBUG("%s: computed follow sets for %zu DFAs across %zu rules\n",
                    __func__, n_dfas, n_rules);
    for (size_t d = 0; d < n_dfas; d++) {
        if (!dfa_follow_dfas[d].empty()) {
            LLAMA_LOG_DEBUG("%s:   DFA %zu: %zu follow DFAs\n",
                            __func__, d, dfa_follow_dfas[d].size());
        }
    }
}

void compiled_grammar::precompute_token_candidates(const vocab_byte_trie & trie, uint32_t total_vocab_tokens) {
    LLAMA_LOG_INFO("%s: precomputing token candidates for %zu DFAs (vocab size = %u)\n",
                   __func__, dfas.size(), total_vocab_tokens);

    auto t_start = std::chrono::steady_clock::now();

    // Compute follow sets before the per-DFA loop
    compute_follow_sets();

    dfa_candidates.resize(dfas.size());
    for (size_t i = 0; i < dfas.size(); i++) {
        // Build follow DFA pointer list, filtering out nullable follow DFAs
        std::vector<const byte_dfa *> follow_dfa_ptrs;
        bool has_nullable_follow = false;
        for (uint32_t fid : dfa_follow_dfas[i]) {
            if (dfas[fid].accept[dfas[fid].start_state]) {
                has_nullable_follow = true;
                break;
            }
        }
        if (!has_nullable_follow) {
            follow_dfa_ptrs.reserve(dfa_follow_dfas[i].size());
            for (uint32_t fid : dfa_follow_dfas[i]) {
                follow_dfa_ptrs.push_back(&dfas[fid]);
            }
        }
        // If has_nullable_follow or empty follow set, follow_dfa_ptrs stays empty → old behavior

        dfa_candidates[i].precompute(dfas[i], trie, total_vocab_tokens, follow_dfa_ptrs);
    }

    auto t_end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    LLAMA_LOG_INFO("%s: precomputation complete in %.2f ms\n", __func__, ms);
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
        const auto & dfa = dfas[seg.id];
        // We're at a DFA segment — set up initial DFA state
        parse_config ready = cfg;
        ready.current.dfa_state = dfa.start_state;
        out_configs.push_back(ready);

        // If the DFA's start state is accepting (zero-length match possible),
        // also generate a config that has advanced past this segment.
        if (dfa.accept[dfa.start_state]) {
            parse_config advanced = cfg;
            advanced.current.segment_idx++;
            advanced.current.dfa_state = 0;
            advance_to_terminal(advanced, out_configs);
        }
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

        bool is_accept = dfa.accept[next_state];

        if (is_accept) {
            // Segment complete — advance to next segment
            parse_config advanced = cfg;
            advanced.current.segment_idx++;
            advanced.current.dfa_state = 0;

            // Resolve rule calls / rule completions
            advance_to_terminal(advanced, new_configs);
        }

        // Continue in DFA if not accepting, or if accepting but the DFA
        // can still match more (e.g. Kleene plus DFAs with looping accepts).
        bool should_continue = !is_accept || dfa.accept_can_continue[next_state];
        if (should_continue) {
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

// ============================================================
// Vocab trie implementation
// ============================================================

void vocab_trie_node::collect_tokens(std::vector<int32_t> & out) const {
    out.insert(out.end(), tokens.begin(), tokens.end());
    for (int b = 0; b < 256; b++) {
        if (children[b]) {
            children[b]->collect_tokens(out);
        }
    }
}

uint32_t vocab_trie_node::count_tokens() const {
    uint32_t count = (uint32_t) tokens.size();
    for (int b = 0; b < 256; b++) {
        if (children[b]) {
            count += children[b]->count_tokens();
        }
    }
    return count;
}

void vocab_trie_node::cache_subtree_tokens() {
    // Post-order: cache children first so we can reuse their results
    subtree_tokens.clear();
    subtree_tokens.insert(subtree_tokens.end(), tokens.begin(), tokens.end());

    for (int b = 0; b < 256; b++) {
        if (children[b]) {
            children[b]->cache_subtree_tokens();
            subtree_tokens.insert(subtree_tokens.end(),
                                  children[b]->subtree_tokens.begin(),
                                  children[b]->subtree_tokens.end());
        }
    }
    // Sort for deterministic order (useful for reject-set construction)
    std::sort(subtree_tokens.begin(), subtree_tokens.end());
}

void vocab_byte_trie::build(const llama_vocab & vocab) {
    uint32_t n = vocab.n_tokens();
    n_tokens_inserted = 0;

    for (uint32_t id = 0; id < n; id++) {
        if (vocab.is_eog((int32_t)id)) {
            continue;
        }
        const std::string & piece = vocab.token_to_piece((int32_t)id);
        if (piece.empty() || piece[0] == '\0') {
            continue;
        }

        vocab_trie_node * node = &root;
        for (size_t i = 0; i < piece.size(); i++) {
            uint8_t byte = (uint8_t) piece[i];
            if (!node->children[byte]) {
                node->children[byte] = std::make_unique<vocab_trie_node>();
            }
            node = node->children[byte].get();
        }
        node->tokens.push_back((int32_t)id);
        n_tokens_inserted++;
    }

    // Precompute cached subtree token lists for efficient DFA candidate precomputation
    root.cache_subtree_tokens();
}

// ============================================================
// Per-DFA-state candidate precomputation
// ============================================================

// Helper: set bits in a bitset for all tokens in a sorted token list.
static void set_bits_for_tokens(std::vector<uint8_t> & bits, const std::vector<int32_t> & tokens) {
    for (int32_t t : tokens) {
        bits[t / 8] |= (1 << (t % 8));
    }
}

// Recursive helper for walk_context_with_follow: walk the trie with follow DFA
// states, marking tokens as context only where at least one follow DFA is alive.
static void walk_context_follow_recursive(
        const std::vector<const byte_dfa *> & follow_dfas,
        const vocab_trie_node & node,
        const std::vector<uint16_t> & follow_states,
        const std::vector<uint16_t> & accept_origins,
        std::vector<std::vector<uint8_t>> & context_bits) {
    // Mark tokens at this node as context (we only reach here if some follow DFA is alive)
    if (!node.tokens.empty()) {
        for (uint16_t orig : accept_origins) {
            set_bits_for_tokens(context_bits[orig], node.tokens);
        }
    }

    for (int b = 0; b < 256; b++) {
        if (!node.children[b]) continue;

        // Advance all follow DFA states by byte b
        std::vector<uint16_t> next_states(follow_dfas.size());
        bool any_alive = false;
        for (size_t d = 0; d < follow_dfas.size(); d++) {
            next_states[d] = follow_dfas[d]->transitions[follow_states[d]][b];
            if (next_states[d] != 0) {
                any_alive = true;
            }
        }
        if (!any_alive) continue;  // ALL follow DFAs dead — prune subtree

        walk_context_follow_recursive(follow_dfas, *node.children[b], next_states,
                                      accept_origins, context_bits);
    }
}

void dfa_state_candidates::walk_context_with_follow(
        const std::vector<const byte_dfa *> & follow_dfas,
        const vocab_trie_node & node,
        const std::vector<uint16_t> & accept_origins,
        std::vector<std::vector<uint8_t>> & context_bits) {
    // Initialize follow DFA states at their start states
    std::vector<uint16_t> initial_states(follow_dfas.size());
    for (size_t d = 0; d < follow_dfas.size(); d++) {
        initial_states[d] = follow_dfas[d]->start_state;
    }

    // Walk from children of the accept node (the accept byte was already consumed)
    for (int cb = 0; cb < 256; cb++) {
        if (!node.children[cb]) continue;

        // Advance all follow DFA states by byte cb
        std::vector<uint16_t> next_states(follow_dfas.size());
        bool any_alive = false;
        for (size_t d = 0; d < follow_dfas.size(); d++) {
            next_states[d] = follow_dfas[d]->transitions[initial_states[d]][cb];
            if (next_states[d] != 0) {
                any_alive = true;
            }
        }
        if (!any_alive) continue;  // ALL follow DFAs dead — skip subtree

        walk_context_follow_recursive(follow_dfas, *node.children[cb], next_states,
                                      accept_origins, context_bits);
    }
}

void dfa_state_candidates::walk_multi(
        const byte_dfa & dfa,
        const vocab_trie_node & node,
        const std::vector<std::pair<uint16_t, std::vector<uint16_t>>> & active_groups,
        std::vector<std::vector<uint8_t>> & accepted_bits,
        std::vector<std::vector<uint8_t>> & context_bits,
        const std::vector<const byte_dfa *> & follow_dfas) {

    // Tokens terminating at this trie node: guaranteed accepted.
    // The DFA is in a non-dead state and the token is fully consumed
    // (stays within the current DFA segment).
    if (!node.tokens.empty()) {
        for (const auto & group : active_groups) {
            for (uint16_t orig : group.second) {
                set_bits_for_tokens(accepted_bits[orig], node.tokens);
            }
        }
    }

    for (int b = 0; b < 256; b++) {
        if (!node.children[b]) continue;

        // Group active states by their DFA transition on byte b.
        // Key: next dfa_state, Value: list of original starting states.
        // Use a small local map since DFA state counts are typically small.
        std::unordered_map<uint16_t, std::vector<uint16_t>> next_groups;
        std::vector<uint16_t> accept_origins;  // states reaching an accept state

        for (const auto & group : active_groups) {
            uint16_t next_state = dfa.transitions[group.first][b];
            if (next_state == 0) {
                // Dead — prune for all original states in this group
                continue;
            }
            if (dfa.accept[next_state]) {
                // DFA segment completes on this byte
                accept_origins.insert(accept_origins.end(),
                                      group.second.begin(), group.second.end());
                // If the accept state has outgoing transitions (e.g. Kleene star
                // self-loop), keep walking — tokens fully consumed via continued
                // DFA transitions are guaranteed-accept, not context-dependent.
                if (dfa.accept_can_continue[next_state]) {
                    auto & v = next_groups[next_state];
                    v.insert(v.end(), group.second.begin(), group.second.end());
                }
            } else {
                auto & v = next_groups[next_state];
                v.insert(v.end(), group.second.begin(), group.second.end());
            }
        }

        if (!accept_origins.empty()) {
            const auto & child = *node.children[b];

            // Tokens terminating exactly at the child node: DFA accepts on
            // their last byte. These are guaranteed accepted (exact-accept).
            if (!child.tokens.empty()) {
                for (uint16_t orig : accept_origins) {
                    set_bits_for_tokens(accepted_bits[orig], child.tokens);
                }
            }

            // Tokens deeper in the subtree: DFA accepts mid-token, remaining
            // bytes must be processed by subsequent grammar segments.
            // These are context-dependent and need simulation.
            if (follow_dfas.empty()) {
                // No follow info — conservative fallback (mark all as context)
                for (int cb = 0; cb < 256; cb++) {
                    if (!child.children[cb]) continue;
                    for (uint16_t orig : accept_origins) {
                        set_bits_for_tokens(context_bits[orig],
                                            child.children[cb]->subtree_tokens);
                    }
                }
            } else {
                // Use follow DFAs to filter: only mark tokens where at least
                // one follow DFA can process the remaining bytes.
                walk_context_with_follow(follow_dfas, child, accept_origins, context_bits);
            }
        }

        // Recurse with merged groups — states sharing the same next dfa_state
        // only recurse once, sharing all work in the subtree.
        if (!next_groups.empty()) {
            std::vector<std::pair<uint16_t, std::vector<uint16_t>>> merged;
            merged.reserve(next_groups.size());
            for (auto & kv : next_groups) {
                merged.emplace_back(kv.first, std::move(kv.second));
            }
            walk_multi(dfa, *node.children[b], merged, accepted_bits, context_bits, follow_dfas);
        }
    }
}

void dfa_state_candidates::precompute(const byte_dfa & dfa, const vocab_byte_trie & trie,
                                       uint32_t total_vocab_tokens,
                                       const std::vector<const byte_dfa *> & follow_dfas) {
    size_t n_states = dfa.transitions.size();
    states.resize(n_states);

    LLAMA_LOG_DEBUG("%s: processing DFA with %zu states\n", __func__, n_states);

    if (n_states <= 1) {
        // Only state 0 (dead) — nothing to precompute
        return;
    }

    // Allocate two bitsets per DFA state: accepted (guaranteed) and context-dependent.
    size_t bits_size = (total_vocab_tokens + 7) / 8;
    std::vector<std::vector<uint8_t>> accepted_bits(n_states, std::vector<uint8_t>(bits_size, 0));
    std::vector<std::vector<uint8_t>> context_bits(n_states, std::vector<uint8_t>(bits_size, 0));

    // Build initial active groups: each DFA state (except 0) starts as its own group
    std::vector<std::pair<uint16_t, std::vector<uint16_t>>> initial_groups;
    initial_groups.reserve(n_states - 1);
    for (size_t s = 1; s < n_states; s++) {
        initial_groups.push_back({(uint16_t) s, {(uint16_t) s}});
    }

    // Single combined walk processes all DFA states together
    walk_multi(dfa, trie.root, initial_groups, accepted_bits, context_bits, follow_dfas);

    // Convert bitsets to adaptive sets
    size_t n_accept_heavy = 0;
    size_t n_reject_heavy = 0;

    for (size_t s = 1; s < n_states; s++) {
        uint32_t n_accepted = 0;
        uint32_t n_context = 0;
        for (size_t i = 0; i < bits_size; i++) {
            uint8_t v = accepted_bits[s][i];
            while (v) { n_accepted++; v &= v - 1; }
            v = context_bits[s][i];
            while (v) { n_context++; v &= v - 1; }
        }

        auto & state = states[s];

        // Build context set: tokens that are context-dependent but NOT
        // guaranteed-accepted. If a token is fully consumed within the DFA
        // (accepted), it doesn't need runtime simulation even if some path
        // through the DFA also accepts mid-token.
        for (uint32_t t = 0; t < total_vocab_tokens; t++) {
            if ((context_bits[s][t / 8] & (1 << (t % 8))) &&
                !(accepted_bits[s][t / 8] & (1 << (t % 8)))) {
                state.context_set.push_back((int32_t) t);
            }
        }

        // Build adaptive accepted/rejected set
        if (n_accepted > total_vocab_tokens / 2) {
            // Accept-heavy: store reject set (tokens not accepted and not context)
            state.accept_heavy = true;
            n_accept_heavy++;
            // Note: n_accepted and n_context can overlap, so avoid underflow
            for (uint32_t t = 0; t < total_vocab_tokens; t++) {
                if (!(accepted_bits[s][t / 8] & (1 << (t % 8))) &&
                    !(context_bits[s][t / 8] & (1 << (t % 8)))) {
                    state.token_set.push_back((int32_t) t);
                }
            }
        } else {
            // Reject-heavy: store accept set
            state.accept_heavy = false;
            n_reject_heavy++;
            state.token_set.reserve(n_accepted);
            for (uint32_t t = 0; t < total_vocab_tokens; t++) {
                if (accepted_bits[s][t / 8] & (1 << (t % 8))) {
                    state.token_set.push_back((int32_t) t);
                }
            }
        }
    }

    for (size_t s = 1; s < n_states; s++) {
        LLAMA_LOG_DEBUG("%s:   state %zu: accept=%d can_continue=%d token_set=%zu context_set=%zu %s\n",
                        __func__, s, (int)dfa.accept[s], (int)dfa.accept_can_continue[s],
                        states[s].token_set.size(), states[s].context_set.size(),
                        states[s].accept_heavy ? "accept-heavy" : "reject-heavy");
    }
    LLAMA_LOG_DEBUG("%s: done - %zu accept-heavy states, %zu reject-heavy states\n",
                    __func__, n_accept_heavy, n_reject_heavy);
}

