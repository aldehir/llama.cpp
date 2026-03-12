# Grammar Token Caching: Per-DFA-State Precomputed Candidates + Vocab Trie

## Problem

In `llama_grammar_apply_impl` (src/llama-grammar.cpp:754), every candidate token
is validated by cloning the current `parse_config` vector and simulating the DFA
forward byte-by-byte through the token's text. For a vocab of 128k tokens, this
means **128k clone-and-simulate operations per sampling step**. The vast majority
are rejects. We can do much better.

## Core Idea

**Precompute, at each DFA state, which tokens are even possible candidates.** At
runtime, instead of simulating all 128k tokens, we look up the candidate set for
the current state and only simulate those. For a grammar expecting a digit at DFA
state S, the candidate set might contain only ~500 tokens (those starting with
`0`-`9`). We simulate 500 instead of 128k.

The candidate set is an **overapproximation** (superset of truly valid tokens)
because it's computed per-segment without knowledge of cross-segment transitions or
the call stack. The runtime simulation on the candidate set gives the exact answer.

**Two components:**
1. **Vocab trie** — built once from the vocabulary. Enables efficient precomputation
   by sharing prefix walks across tokens.
2. **Per-DFA-state candidate sets** — precomputed at grammar compile time by walking
   the trie × DFA product. Stored in the `compiled_grammar` (immutable, shared).

An optional **adaptive runtime cache** on top avoids even the candidate simulation
on repeated visits to the same grammar state.

---

## Architecture

### Component 1: Vocab Byte Trie

A byte-level trie over all token strings. Built once per vocab (or lazily on first
grammar use). This is the tool for efficient precomputation — not a runtime structure.

```
Location: src/llama-grammar-dfa.h / src/llama-grammar-dfa.cpp
Lifetime: built once, can be shared across grammars using the same vocab
```

#### Data Structure

```cpp
struct vocab_trie_node {
    // Tokens that terminate exactly at this node.
    // A token "abc" has its ID stored at the node reached after walking a→b→c.
    std::vector<llama_token> tokens;

    // Children indexed by next byte.
    // Dense array at shallow depths (0-2) where branching is high.
    // Could switch to sparse at deeper levels, but 256 pointers × 8 bytes = 2KB
    // per node is acceptable given typical trie depth (4-8).
    std::unique_ptr<vocab_trie_node> children[256];
};

struct vocab_byte_trie {
    vocab_trie_node root;

    // Build from vocabulary. Insert every non-EOG, non-empty token.
    void build(const llama_vocab & vocab);

    // Count all tokens in a subtree (for reject_subtree).
    static void collect_tokens(const vocab_trie_node & node,
                               std::vector<llama_token> & out);
};
```

#### Build Algorithm

```
for token_id in 0..vocab.n_tokens():
    piece = vocab.token_to_piece(token_id)
    if piece.empty() or piece[0] == '\0' or vocab.is_eog(token_id):
        continue
    node = &root
    for byte in piece:
        if not node->children[byte]:
            node->children[byte] = make_unique<vocab_trie_node>()
        node = node->children[byte].get()
    node->tokens.push_back(token_id)
```

**Cost:** O(total bytes across all tokens). ~500KB of token text for 128k vocab.
One-time, < 5ms.

### Component 2: Per-DFA-State Candidate Sets (Precomputed)

For each `(dfa_id, dfa_state)` pair in the compiled grammar, precompute the set of
tokens that **don't die** when their bytes are fed through the DFA starting from
that state. This is the key optimization.

```
Location: stored inside compiled_grammar (immutable after compile)
Indexed by: (dfa_id, dfa_state) → vector<llama_token>
```

#### What "Don't Die" Means

A token with bytes `[b0, b1, ..., bN]` is a candidate at DFA state S if:

```
s0 = S
s1 = dfa.transitions[s0][b0]    → must be != 0 (not dead)
s2 = dfa.transitions[s1][b1]    → must be != 0
...
sN = dfa.transitions[s(N-1)][bN] → must be != 0

OR

at any intermediate step si, dfa.accept[si] == true
(segment completes mid-token; remaining bytes go to a different segment)
```

The second condition is critical: if the DFA accepts at byte j < N, the remaining
bytes `[b(j+1), ..., bN]` will be consumed by a *different* DFA segment (after
`advance_to_terminal`). We can't predict which DFA that is at precompute time
(it depends on the call stack), so we **conservatively include** the token. This
makes the candidate set a safe overapproximation.

#### Precomputation Algorithm (Trie × DFA Walk)

For each DFA `d` in `compiled_grammar.dfas`, for each live state `s` (s != 0):

```
function precompute_candidates(dfa d, state s, trie_node node, candidates):
    for byte in 0..255 where node.children[byte] exists:
        next_state = d.transitions[s][byte]

        if next_state == 0:
            // Dead — entire subtree is rejected for this DFA state
            continue

        child = node.children[byte]

        if d.accept[next_state]:
            // DFA segment completes here. All tokens at or below this
            // child are candidates (remaining bytes go to next segment).
            collect_all_tokens(child, candidates)
        else:
            // DFA still in progress — tokens terminating here are candidates
            // (they leave the DFA mid-segment, which is valid if configs allow it)
            // Actually: tokens terminating at a non-accept state are NOT valid
            // for this segment alone. But they could still be valid if another
            // config handles them. To keep the overapproximation safe, include them.
            // Alternatively, exclude them for tighter sets — but then we must
            // ensure the runtime union across configs catches them.
            for token_id in child.tokens:
                candidates.push_back(token_id)

            // Recurse for longer tokens
            precompute_candidates(d, next_state, child, candidates)
```

**Refinement — tighter candidate sets:** We can split candidates into two
categories per (dfa_id, state):
1. **definite candidates**: tokens whose full byte sequence stays alive in the DFA
   (all transitions non-dead, no early accept) AND ends in a non-dead state.
2. **early-accept candidates**: tokens where the DFA accepts mid-token.

At runtime, we always include definite candidates. We include early-accept
candidates only if the grammar has rule calls that could follow (which is almost
always true, so in practice include both).

For simplicity in v1, just compute one combined set.

#### Storage

```cpp
// Inside compiled_grammar:
struct dfa_state_candidates {
    // candidates[state] = sorted vector of token IDs that are possible
    // at that DFA state. State 0 (dead) has empty candidates.
    std::vector<std::vector<llama_token>> candidates;
};

// One per DFA in the compiled grammar
std::vector<dfa_state_candidates> dfa_candidates;
```

**Memory estimate:**
- Typical grammar: 10-30 DFAs, 5-20 states each → 50-600 (dfa, state) pairs
- Average candidate set size: ~1000-5000 tokens (heavily grammar-dependent)
- Storage per set: sorted `vector<llama_token>` = 4 bytes × N
- Total: 600 × 5000 × 4 = ~12 MB worst case
- Typical: ~2-5 MB (most states are restrictive with small candidate sets)

**Optimization — bitset alternative:**
For vocab ≤ 256k tokens, use `std::vector<bool>` (32 KB per state) instead of
sorted vectors. Trades memory for O(1) lookup. With 600 states: 600 × 32KB = 19 MB.
Only worthwhile if candidate sets are large (> 8k tokens). Benchmark both.

#### Precomputation Cost

For each (dfa, state), we walk the trie. But the DFA pruning means we visit far
fewer trie nodes than the full trie. The work is:

```
Total precompute cost = Σ over (dfa, state) of [trie nodes visited for that state]
```

With DFA pruning, each walk visits O(|candidates| × avg_token_length) nodes.
For 600 states with avg 2000 candidates of avg length 4: 600 × 2000 × 4 = ~5M
node visits. At ~10ns per visit: **~50ms total precomputation** — acceptable as
a one-time compile cost.

### Component 3: Runtime Integration

#### Modified `llama_grammar_apply_impl`

```cpp
void llama_grammar_apply_impl(const struct llama_grammar & grammar,
                               llama_token_data_array * cur_p) {
    if (grammar.awaiting_trigger) return;

    bool allow_eog = grammar.compiled->any_config_complete(grammar.configs);

    // Step 1: Compute the candidate union across all active configs.
    //
    // Each config points to a (dfa_id, dfa_state). Look up precomputed
    // candidates for that state. Union them.
    std::vector<bool> is_candidate(grammar.vocab->n_tokens(), false);
    for (const auto & cfg : grammar.configs) {
        const auto & rule = grammar.compiled->rules[cfg.current.rule_id];
        const auto & alt = rule.alternates[cfg.current.alternate_idx];
        if (cfg.current.segment_idx >= alt.size()) continue;
        const auto & seg = alt[cfg.current.segment_idx];
        if (seg.type != compiled_segment::DFA_MATCH) continue;

        const auto & candidates =
            grammar.compiled->dfa_candidates[seg.id].candidates[cfg.current.dfa_state];
        for (llama_token tok : candidates) {
            is_candidate[tok] = true;
        }
    }

    // Step 2: For each token in cur_p, fast-reject non-candidates,
    // then simulate only the candidates.
    for (size_t i = 0; i < cur_p->size; ++i) {
        const llama_token id = cur_p->data[i].id;

        if (grammar.vocab->is_eog(id)) {
            if (!allow_eog) {
                cur_p->data[i].logit = -INFINITY;
            }
            continue;
        }

        const std::string & piece = grammar.vocab->token_to_piece(id);
        if (piece.empty() || piece[0] == 0) {
            cur_p->data[i].logit = -INFINITY;
            continue;
        }

        if (!is_candidate[id]) {
            // Not a candidate at ANY active DFA state → definitely rejected
            cur_p->data[i].logit = -INFINITY;
            continue;
        }

        // Candidate token — must simulate to confirm (handles cross-segment
        // transitions, call stack, multi-config interactions)
        auto sim_configs = grammar.configs;
        bool valid = true;
        for (size_t j = 0; j < piece.size(); j++) {
            grammar.compiled->accept_byte(sim_configs, (uint8_t)piece[j]);
            if (sim_configs.empty()) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            cur_p->data[i].logit = -INFINITY;
        }
    }
}
```

**Why this works:**
- The precomputed candidate sets are a **superset** of truly valid tokens.
  No valid token is ever incorrectly skipped (no false negatives).
- Non-candidate tokens are rejected in O(1) — a single `is_candidate[id]` check.
- Only candidate tokens (~0.1-5% of vocab) go through the full simulation.
- The simulation is the exact same `accept_byte` logic, so correctness is preserved.

### Component 4: Adaptive Runtime Cache (Optional Layer)

On top of precomputed candidates, cache the exact accept/reject results for the
current grammar state to avoid even the candidate simulation on repeated calls.

```cpp
struct grammar_token_cache {
    uint64_t state_hash = 0;

    // Adaptive: stores whichever is smaller (accepted or rejected)
    bool polarity;  // true = set contains accepted tokens
    std::unordered_set<llama_token> token_set;
    bool allow_eog;

    bool is_valid(uint64_t h) const { return state_hash == h && h != 0; }

    bool is_token_accepted(llama_token id) const {
        bool in_set = token_set.count(id) > 0;
        return polarity ? in_set : !in_set;
    }

    void invalidate() { state_hash = 0; }
};
```

**Flow with cache:**
1. Hash current configs.
2. If cache hit → apply cached results directly (O(1) per token).
3. If cache miss → compute candidates via precomputed sets + simulate → populate
   cache with adaptive polarity.

Cache is invalidated in `llama_grammar_accept_impl` when the grammar advances.

---

## Implementation Plan

### Step 1: Vocab Byte Trie

**Files:** `src/llama-grammar-dfa.h`, `src/llama-grammar-dfa.cpp`

1. Define `vocab_trie_node` struct (tokens vector + children[256] array).
2. Define `vocab_byte_trie` struct with `build(const llama_vocab &)`.
3. Implement `build`: iterate all tokens, insert byte-by-byte into trie.
4. Implement `collect_tokens`: recursive helper to gather all token IDs in subtree.
5. Store the trie in `llama_grammar` (or as a shared object if multiple grammars
   use the same vocab).

**Tests:**
- All non-EOG, non-empty tokens are reachable in the trie.
- `collect_tokens` on root returns all inserted tokens.
- Subtree token counts match expected values.

### Step 2: Per-DFA-State Candidate Precomputation

**Files:** `src/llama-grammar-dfa.h`, `src/llama-grammar-dfa.cpp`

1. Add `dfa_state_candidates` struct to header.
2. Add `std::vector<dfa_state_candidates> dfa_candidates` to `compiled_grammar`.
3. Implement `precompute_candidates(dfa, state, trie_node, out)`:
   - Walk trie × DFA in lockstep.
   - Prune dead branches (transition == 0).
   - On DFA accept mid-token: collect entire subtree (conservative inclusion).
   - On non-dead, non-accept: add terminating tokens + recurse.
4. Call precomputation in `compiled_grammar::compile()` after all DFAs are built.
   Requires passing the vocab trie into compile, or running as a post-compile step.
5. Sort each candidate vector for efficient union at runtime (or use bitset).

**Tests:**
- For a digits-only DFA, candidates contain only tokens starting with 0-9.
- For an any-character DFA at start state, candidates = nearly all tokens.
- Candidate set is a superset of brute-force valid tokens (no false negatives).
- Compare brute-force simulation vs candidate-filtered simulation for correctness.

### Step 3: Integration + Candidate-Filtered Apply

**Files:** `src/llama-grammar.cpp`

1. Build vocab trie in `llama_grammar_init_impl` (or cache it per-vocab).
2. Pass trie to `compiled_grammar::compile()` or run precomputation after compile.
3. Replace the loop in `llama_grammar_apply_impl`:
   - Build `is_candidate` bitset from union of precomputed candidate sets.
   - Skip non-candidates with O(1) check.
   - Simulate only candidates.
4. Verify all existing tests pass unchanged.

**Tests:**
- Identical output to current brute-force implementation on all test grammars.
- `test-grammar-integration`, `test-llama-grammar`, `test-json-schema-to-grammar`.
- Performance measurement: log candidate set sizes vs vocab size.

### Step 4: Adaptive Runtime Cache

**Files:** `src/llama-grammar-dfa.h`, `src/llama-grammar.h`, `src/llama-grammar.cpp`

1. Add `grammar_token_cache` struct.
2. Implement FNV-1a `hash_configs`.
3. On cache miss: run candidate-filtered simulation, collect accept/reject results,
   store with adaptive polarity.
4. On cache hit: apply cached results directly.
5. Invalidate in `llama_grammar_accept_impl`.
6. Invalidate on clone (conservative).

**Tests:**
- Cache hit returns identical results to fresh computation.
- Polarity selection is correct for both accept-heavy and reject-heavy states.
- Cache invalidation triggers on state change.

### Step 5: Optimizations (Follow-up)

1. **Trie compaction:** Compress single-child chains into a single node with a byte
   string key. Reduces pointer chasing for multi-byte UTF-8 prefixes.
2. **Bitset candidate storage:** For large candidate sets, `std::vector<bool>` may
   outperform sorted vectors for the union step.
3. **LRU state cache:** Keep cached results for the N most recent grammar states
   instead of just the current one. Helps with grammars that cycle between states
   (e.g., JSON object → key → value → key → value...).
4. **Lazy precomputation:** Only precompute candidates for (dfa, state) pairs that
   are actually reached, not all possible states. Saves compile time for DFAs with
   many states but few reachable paths.

---

## Complexity Analysis

### Current (Brute Force)

```
Per sampling step: O(V × L × C)
  V = vocab size (~128k)
  L = avg token length in bytes (~4)
  C = config count (typically 1-4)
```

### With Precomputed Candidates (Cache Miss)

```
Per sampling step: O(V + A × L × C)
  V = vocab size (for is_candidate bitset build — but A << V, so the union is fast)
  A = |candidate union| (typically 100-5000, << V)
  L, C as above
```

The `is_candidate` bitset build costs O(Σ |candidate_set| for each active config).
With small config counts and candidate sets of ~1000: O(1000 × 2) = O(2000).
The simulation is O(A × L × C) = O(1000 × 4 × 2) = O(8000).
Compare to brute force: O(128000 × 4 × 2) = O(1,024,000). **~100x reduction.**

### With Precomputed Candidates + Cache (Cache Hit)

```
Per sampling step: O(V')
  V' = cur_p->size (apply cached result per token)
  Per-token: O(1) hash set lookup
```

Near-zero overhead on cache hit.

### Precomputation Cost (One-Time)

```
O(D × S × T_visited)
  D = number of DFAs (~10-30)
  S = states per DFA (~5-20)
  T_visited = trie nodes visited per (dfa, state) (depends on DFA selectivity)
```

Estimated: 50-200ms for typical grammars. Runs once at grammar compile time.

---

## Memory Budget

| Component | Size | Notes |
|-----------|------|-------|
| Vocab trie | ~3-5 MB | 256-pointer arrays at each node; depth ~4-8 |
| Candidate sets | ~2-12 MB | Depends on grammar selectivity per state |
| Runtime cache | ~1-32 KB | Adaptive, stores min(accept, reject) |
| `is_candidate` bitset | ~16 KB | Temporary per apply call (vocab/8 bytes) |
| **Total** | **~5-17 MB** | Dominated by trie + candidate sets |

Modest relative to model weights (GB) and KV cache. The trie could be shared across
grammars using the same vocab, amortizing its cost.

---

## Edge Cases

1. **DFA accept at first byte:** Token `"{"` with a DFA that accepts `{` immediately.
   The candidate set includes `"{"` plus all tokens starting with `{` (since the DFA
   accepts mid-token for multi-byte tokens starting with `{`). The runtime simulation
   correctly identifies which longer tokens are actually valid after cross-segment
   handling.

2. **Token spans multiple segments:** Token `": "` where `:` completes one DFA
   segment and `" "` starts another. The precomputed set for the first DFA includes
   this token (DFA accepts at byte 0, so all tokens starting with `:` are
   conservatively included). Runtime simulation handles the segment crossing exactly.

3. **All tokens valid (free-text state):** Candidate set ≈ full vocab. No speedup
   from precomputation at this state, but the runtime cache avoids re-simulation.

4. **Very restrictive state (expect exactly `true`):** Candidate set contains only
   tokens starting with `t`. Maybe 500-1000 tokens. Huge speedup.

5. **Multiple active configs at different DFA states:** Union of candidate sets.
   Still much smaller than full vocab in practice.

6. **DFA state 0 (dead):** Candidate set is empty. No configs should point here
   (dead configs are pruned by `accept_byte`), so this is a no-op.
