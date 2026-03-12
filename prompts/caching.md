# Grammar Token Caching: Vocab Trie + Adaptive DFA State Cache

## Problem

In `llama_grammar_apply_impl` (src/llama-grammar.cpp:754), every candidate token
is validated by cloning the current `parse_config` vector and simulating the DFA
forward byte-by-byte through the token's text. For a vocab of 128k tokens, this
means **128k clone-and-simulate operations per sampling step** — most of which
produce the same reject result. The grammar state (set of `parse_config`s) changes
only when a token is actually accepted, yet we recompute everything from scratch.

## Goals

1. **Vocab trie** — build a byte-level trie over all token strings so that DFA
   simulation can be shared across tokens with common prefixes. One DFA walk down
   a shared prefix validates/invalidates thousands of tokens at once.
2. **Adaptive per-state token cache** — at each grammar state, cache the
   accept/reject result for every token. Use an **adaptive representation** that
   stores whichever set is smaller (accepted tokens or rejected tokens) plus a
   polarity bit, keeping memory proportional to `min(|accept|, |reject|)`.
3. **Cache invalidation** — the cache key is the grammar state (the config set).
   When `llama_grammar_accept_impl` advances the state, the cache is invalidated.

---

## Architecture

### Component 1: Vocab Trie (`vocab_byte_trie`)

A compact byte-level trie over the entire vocabulary built once at grammar init.

```
Location: src/llama-grammar-dfa.h / src/llama-grammar-dfa.cpp
Lifetime: built once per llama_grammar, stored in llama_grammar struct
```

#### Data Structure

```cpp
struct vocab_trie_node {
    // Tokens that terminate exactly at this node
    std::vector<llama_token> tokens;

    // Children indexed by next byte (sparse — use flat_map or sorted vector)
    // For nodes near the root (depth 0-1), up to 256 children are common.
    // For deeper nodes, children are sparse.
    std::array<std::unique_ptr<vocab_trie_node>, 256> children;
    // Alternative for memory efficiency at deeper levels:
    //   std::vector<std::pair<uint8_t, std::unique_ptr<vocab_trie_node>>> children;
    // Benchmark both; the array form is faster for the hot DFA walk.
};

struct vocab_byte_trie {
    vocab_trie_node root;
    uint32_t n_tokens;  // total tokens inserted

    void build(const llama_vocab & vocab);
};
```

#### Build Algorithm

```
for each token_id in [0, vocab.n_tokens()):
    piece = vocab.token_to_piece(token_id)
    if piece is empty or piece[0] == '\0' or vocab.is_eog(token_id):
        skip  // these are handled separately
    walk trie from root, creating nodes as needed
    at the terminal node, append token_id to node.tokens
```

**Complexity:** O(total bytes across all tokens) — one-time cost, typically < 5ms
for 128k tokens.

#### Trie-Guided DFA Walk

The key insight: instead of simulating each token independently, we walk the trie
and the DFA **simultaneously** in a single depth-first traversal.

```
function walk(trie_node, configs):
    // All tokens at this node are VALID (configs is non-empty here)
    for token_id in trie_node.tokens:
        mark token_id as accepted

    for byte in 0..255 where trie_node.children[byte] exists:
        child_configs = clone(configs)
        compiled.accept_byte(child_configs, byte)
        if child_configs is non-empty:
            walk(trie_node.children[byte], child_configs)
        else:
            // Entire subtree is rejected — skip all tokens below
            reject_subtree(trie_node.children[byte])
```

**Why this is fast:**
- Tokens sharing a prefix (e.g., `" the"`, `" them"`, `" then"`, `" there"`)
  share the DFA simulation for their common prefix `" the"`.
- When a DFA prefix leads to a dead state, the **entire subtree** is pruned.
  For a grammar expecting a digit, all ~120k tokens not starting with `0-9`
  are pruned at depth 1.
- Worst case is the same as brute force (no sharing). Best case is O(256) byte
  transitions at the root to reject almost everything.

**Expected improvement:** For structured grammars (JSON, function calls), the
valid token set is typically < 1% of vocab at any given state. The trie walk
will prune 99%+ of tokens at shallow depths (1-3 bytes).

### Component 2: Adaptive Token Cache (`grammar_token_cache`)

After the trie walk computes the full accept/reject partition, cache it.

```
Location: src/llama-grammar-dfa.h / src/llama-grammar-dfa.cpp
Lifetime: stored in llama_grammar struct, invalidated on state change
```

#### Data Structure

```cpp
struct grammar_token_cache {
    // The configs that this cache was computed for.
    // Used for validation — if grammar.configs changes, cache is stale.
    uint64_t state_hash;

    // Adaptive representation:
    // If polarity == true:  token_set contains ACCEPTED tokens (accept-heavy → store accepts)
    // If polarity == false: token_set contains REJECTED tokens (reject-heavy → store rejects)
    // The polarity is chosen so that token_set.size() <= n_tokens / 2.
    bool polarity;  // true = set contains accepted, false = set contains rejected
    std::unordered_set<llama_token> token_set;

    // EOG handling (separate from token set)
    bool allow_eog;

    bool is_valid(uint64_t current_hash) const {
        return state_hash == current_hash;
    }

    bool is_token_accepted(llama_token id) const {
        bool in_set = token_set.count(id) > 0;
        return polarity ? in_set : !in_set;
        // polarity=true  (set=accepted): in_set means accepted
        // polarity=false (set=rejected): NOT in_set means accepted
    }

    void build(bool polarity_, std::unordered_set<llama_token> && set,
               bool allow_eog_, uint64_t hash) {
        polarity = polarity_;
        token_set = std::move(set);
        allow_eog = allow_eog_;
        state_hash = hash;
    }

    void invalidate() { state_hash = 0; }
};
```

#### Adaptive Polarity Selection

After the trie walk produces the set of accepted tokens:

```
n_accepted = count of accepted tokens
n_rejected = n_tokens - n_accepted

if n_accepted <= n_rejected:
    store accepted set, polarity = true     // "accept-heavy" grammar state
else:
    store rejected set, polarity = false    // "reject-heavy" grammar state
```

This ensures the cache always uses `O(min(|accept|, |reject|))` memory.

**Typical case for structured grammars:** At most grammar states, only a few
tokens are valid (e.g., only `{`, `"`, digits). The accepted set is tiny
(~10-1000 tokens), so we store accepts with polarity=true. Memory: ~4-8 KB.

**Typical case at free-text states:** When the grammar allows any UTF-8 text
(e.g., inside a JSON string value), most tokens are valid. The rejected set is
small (only tokens with invalid UTF-8 or control characters). We store rejects
with polarity=false. Memory: ~1-4 KB.

#### State Hashing

The cache key is a hash of the current `parse_config` vector:

```cpp
uint64_t hash_configs(const std::vector<parse_config> & configs) {
    uint64_t h = FNV_OFFSET_BASIS;
    h = fnv1a(h, configs.size());
    for (const auto & cfg : configs) {
        h = fnv1a(h, cfg.current.rule_id);
        h = fnv1a(h, cfg.current.alternate_idx);
        h = fnv1a(h, cfg.current.segment_idx);
        h = fnv1a(h, cfg.current.dfa_state);
        h = fnv1a(h, cfg.call_stack.size());
        for (const auto & frame : cfg.call_stack) {
            h = fnv1a(h, frame.rule_id);
            h = fnv1a(h, frame.alternate_idx);
            h = fnv1a(h, frame.segment_idx);
            h = fnv1a(h, frame.dfa_state);
        }
    }
    return h;
}
```

### Component 3: Integration into `llama_grammar_apply_impl`

The new hot path:

```cpp
void llama_grammar_apply_impl(const struct llama_grammar & grammar,
                               llama_token_data_array * cur_p) {
    if (grammar.awaiting_trigger) return;

    uint64_t state_hash = hash_configs(grammar.configs);

    // Check if cache is still valid for current grammar state
    if (!grammar.token_cache.is_valid(state_hash)) {
        // Cache miss — recompute via trie-guided DFA walk
        std::unordered_set<llama_token> accepted;
        bool allow_eog = grammar.compiled->any_config_complete(grammar.configs);

        grammar.vocab_trie.walk_with_dfa(
            grammar.compiled, grammar.configs, accepted);

        // Choose adaptive polarity
        uint32_t n_total = grammar.vocab->n_tokens();
        bool polarity;
        std::unordered_set<llama_token> cache_set;

        if (accepted.size() <= n_total / 2) {
            polarity = true;
            cache_set = std::move(accepted);
        } else {
            polarity = false;
            // Build rejected set = all non-EOG tokens not in accepted
            for (uint32_t id = 0; id < n_total; id++) {
                if (!grammar.vocab->is_eog(id) && accepted.count(id) == 0) {
                    cache_set.insert(id);
                }
            }
        }

        grammar.token_cache.build(polarity, std::move(cache_set),
                                   allow_eog, state_hash);
    }

    // Apply cached results — O(cur_p->size) with O(1) per-token lookup
    for (size_t i = 0; i < cur_p->size; ++i) {
        const llama_token id = cur_p->data[i].id;

        if (grammar.vocab->is_eog(id)) {
            if (!grammar.token_cache.allow_eog) {
                cur_p->data[i].logit = -INFINITY;
            }
        } else {
            const std::string & piece = grammar.vocab->token_to_piece(id);
            if (piece.empty() || piece[0] == 0) {
                cur_p->data[i].logit = -INFINITY;
            } else if (!grammar.token_cache.is_token_accepted(id)) {
                cur_p->data[i].logit = -INFINITY;
            }
        }
    }
}
```

---

## Implementation Plan

### Step 1: Vocab Trie Construction

**Files:** `src/llama-grammar-dfa.h`, `src/llama-grammar-dfa.cpp`

1. Add `vocab_trie_node` and `vocab_byte_trie` structs to the header.
2. Implement `vocab_byte_trie::build(const llama_vocab & vocab)`:
   - Iterate all token IDs, get `token_to_piece`, insert into trie.
   - Skip EOG tokens, empty tokens, and null-byte tokens.
3. Implement `vocab_byte_trie::walk_with_dfa(...)`:
   - Recursive DFS that walks the trie and DFA in lockstep.
   - Accepts: `compiled_grammar`, `configs`, output `accepted` set.
   - Prune subtrees when `accept_byte` produces empty configs.
4. Add `vocab_byte_trie` field to `llama_grammar` struct.
5. Build the trie in `llama_grammar_init_impl`.

**Tests:**
- Trie construction correctness (all tokens reachable).
- Trie walk produces same accept/reject as brute-force for small grammars.
- Subtree pruning count verification.

### Step 2: Adaptive Token Cache

**Files:** `src/llama-grammar-dfa.h`, `src/llama-grammar-dfa.cpp`

1. Add `grammar_token_cache` struct to the header.
2. Implement `hash_configs` (FNV-1a over config state).
3. Implement `grammar_token_cache::build`, `is_valid`, `is_token_accepted`.
4. Add `grammar_token_cache` field to `llama_grammar` struct (mutable).
5. Invalidate cache in `llama_grammar_accept_impl` after state advancement.

**Tests:**
- Hash stability (same configs → same hash).
- Hash sensitivity (different configs → different hash, no false positives).
- Polarity correctness (accept-heavy and reject-heavy cases).
- Cache hit/miss behavior across sampling iterations.

### Step 3: Integrate into Apply Path

**Files:** `src/llama-grammar.cpp`

1. Replace the per-token simulation loop in `llama_grammar_apply_impl` with
   the cache-check-then-trie-walk pattern shown above.
2. Add cache invalidation call in `llama_grammar_accept_impl`.
3. Update `llama_grammar_clone_impl` to handle cache (invalidate on clone,
   or share if configs match — invalidate is simpler and correct).

**Tests:**
- End-to-end grammar sampling produces identical results to current code.
- Run existing test suites: `test-grammar-integration`, `test-llama-grammar`,
  `test-json-schema-to-grammar`.
- Performance: measure token evaluation time before/after on a real model.

### Step 4: Optimizations (Follow-up)

After correctness is verified:

1. **Trie node compaction:** For long single-child chains (common in multi-byte
   UTF-8 prefixes), compress into a single node with a byte string key.
2. **Bitset alternative for cache:** When vocab < 256k, consider a `std::bitset`
   or `std::vector<bool>` indexed by token ID instead of `unordered_set`.
   Trades memory (16-32 KB fixed) for O(1) lookup without hashing.
3. **Persistent cache across states:** For grammar states that recur (e.g.,
   returning to the same JSON object key state), consider an LRU cache keyed
   by `state_hash` that persists across multiple accept steps.
4. **Parallel trie walk:** The DFS branches are independent — could use a
   thread pool for the 256 children of the root node.

---

## Complexity Analysis

### Current (No Cache, No Trie)

| Operation | Cost |
|-----------|------|
| Per sampling step | O(V × L × C) where V=vocab, L=avg token length, C=config count |
| Config clone | O(C) per token |
| Byte simulation | O(C) per byte per token |
| Dedup | O(C²) per byte per token |
| **Total** | **O(V × L × C²)** per sampling step |

For V=128k, L=4, C=2: ~1M DFA transitions per step.

### With Trie + Cache (Cache Hit)

| Operation | Cost |
|-----------|------|
| Hash check | O(C × S) where S=avg stack depth |
| Per-token lookup | O(1) hash lookup |
| **Total** | **O(C × S + V')** where V'=cur_p->size |

Near-zero cost on cache hit. V' is typically = V but the per-token work is a
single hash table lookup.

### With Trie + Cache (Cache Miss — Trie Walk)

| Operation | Cost |
|-----------|------|
| Trie DFS | O(N_nodes × C) where N_nodes = trie nodes visited |
| Pruning benefit | N_nodes << total trie nodes (dead branches skipped) |
| **Expected** | **O(256 × C + A × L × C)** where A=accepted token count |

For structured grammars where A << V: massive speedup. If only 100 tokens are
valid out of 128k, we visit ~100×4=400 trie nodes instead of 128k×4=512k byte
simulations.

---

## Memory Budget

| Component | Memory |
|-----------|--------|
| Vocab trie nodes | ~2-4 MB (128k tokens, avg 4 bytes each, 256-pointer arrays) |
| Trie token lists | ~512 KB (128k token IDs × 4 bytes) |
| Token cache (accept-heavy) | ~1-8 KB (tiny accepted set) |
| Token cache (reject-heavy) | ~1-8 KB (tiny rejected set) |
| State hash | 8 bytes |
| **Total overhead** | **~3-5 MB** (dominated by trie, amortized across all steps) |

This is modest compared to model weights (several GB) and KV cache.

---

## Edge Cases

1. **Empty grammar / no constraints:** All tokens accepted. Cache stores empty
   rejected set (polarity=false, empty set). Every lookup returns true. Cost: O(1).

2. **Very restrictive grammar (1-2 valid tokens):** Cache stores tiny accepted set
   (polarity=true, 1-2 entries). Trie walk prunes at depth 1 for most branches.

3. **Free-text inside grammar (JSON string values):** Most tokens valid. Cache
   stores small rejected set. Trie walk visits many nodes but that's unavoidable —
   and the cache means we only do it once per state.

4. **Grammar state that recurs:** Same hash → cache hit. No re-walk needed.

5. **Hash collision:** Extremely unlikely with 64-bit FNV-1a over small config
   vectors. If paranoid, add a full config equality check on cache hit.

6. **Grammar clone:** Invalidate cache (safe default). The clone may diverge
   immediately, so sharing the cache is risky.
