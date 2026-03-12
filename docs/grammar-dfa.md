# Grammar Engine: Byte-Level DFA with Token Candidate Caching

## Overview

The grammar engine constrains LLM token sampling to match a BNF-style grammar.
It replaces the legacy NPDA (nondeterministic pushdown automaton) with a
byte-level DFA approach that is simpler, faster, and eliminates partial UTF-8
handling entirely.

The engine has three layers:

1. **Compile-time:** Grammar rules are parsed, split into terminal segments and
   rule calls, and each terminal segment is compiled to a minimized byte-level
   DFA.
2. **Init-time:** A vocab trie is built from all token strings, then walked
   against each DFA state to precompute which tokens are possible candidates.
   These candidate sets use an adaptive representation that stores whichever of
   the accept or reject set is smaller.
3. **Runtime:** At each sampling step, the precomputed candidate sets narrow the
   token space to a small subset that must be simulated. Non-candidates are
   rejected in O(1). Only candidates undergo full DFA simulation.

## Architecture

### Compilation Pipeline

```
Grammar text
  → Parser (llama_grammar_parser)
  → Parsed rules: vector<vector<llama_grammar_element>>
  → Compiler (compiled_grammar::compile):
      For each rule:
        Split alternates at ALT/END boundaries
        Split each alternate into segments at RULE_REF boundaries
        For each terminal segment:
          → Thompson NFA construction (byte-level)
          → Subset construction (NFA → DFA)
          → Hopcroft minimization
          → byte_dfa stored in compiled_grammar.dfas[]
        Rule calls become compiled_segment::RULE_CALL
  → compiled_grammar (immutable, shared across clones)
```

Key property: the grammar operates on **raw bytes**, not Unicode code points.
UTF-8 code point ranges like `[a-z]` or `[α-ω]` are expanded to byte-level NFA
patterns at compile time. Multi-byte UTF-8 sequences become intermediate DFA
states. This eliminates all partial UTF-8 handling.

### Data Structures

#### `byte_dfa`

The core automaton. Each DFA has:
- `transitions[state][byte] → next_state` (state 0 = dead/reject)
- `accept[state]` — whether the state is accepting
- `start_state` — typically state 1

Transition lookup is O(1): a single array index.

#### `compiled_grammar`

Immutable after construction. Contains:
- `dfas[]` — all compiled DFA segments
- `rules[]` — each rule has alternates, each alternate is a sequence of segments
  (`DFA_MATCH` or `RULE_CALL`)
- `dfa_candidates[]` — precomputed per-DFA-state token candidate sets
- `start_rule` — entry point

#### `parse_config`

The runtime state for one active parse path:
- `current` — a `grammar_position` (rule_id, alternate_idx, segment_idx, dfa_state)
- `call_stack` — vector of `grammar_call_frame` for rule call/return

Multiple `parse_config`s may be active simultaneously when the grammar has
ambiguous rule choices (multiple alternates). For well-formed grammars (JSON,
function calls), the config count is typically 1-4.

### Runtime: Token Validation

`llama_grammar_apply_impl` is the hot path, called once per sampling step. It
sets `logit = -INFINITY` for tokens that violate the grammar.

#### Without Caching (Old Approach)

For every candidate token (up to ~128k):
1. Clone the config vector
2. Feed each byte of the token through `accept_byte`
3. If configs go empty, reject the token

Cost: O(V x L x C) where V=vocab size, L=avg token length, C=config count.

#### With Precomputed Candidates (Current Approach)

1. For each active config, look up its `(dfa_id, dfa_state)` in the precomputed
   `dfa_candidates` table to get a token set.
2. Intersect reject sets across all configs to build a `needs_simulation` bitmask.
3. Only simulate tokens where `needs_simulation[id]` is true.
4. Non-candidates are rejected in O(1).

Cost: O(A x L x C) where A = number of candidate tokens (typically 100-5000,
<< V for structured grammars).

### Token Acceptance

When a token is selected, `llama_grammar_accept_impl` feeds its bytes through
`accept_byte` to advance the grammar state. `accept_byte`:
1. For each active config, looks up the DFA transition for the byte.
2. If the transition goes to state 0 (dead), the config dies.
3. If the transition reaches an accept state, the DFA segment is complete —
   `advance_to_terminal` is called to resolve rule calls, rule completions, and
   call stack operations.
4. Otherwise, the config advances to the new DFA state.
5. Configs are deduplicated after each byte.

## Vocab Trie

A byte-level trie over all token strings in the vocabulary. Built once at grammar
initialization.

```
struct vocab_trie_node {
    vector<llama_token> tokens;               // tokens terminating at this node
    unique_ptr<vocab_trie_node> children[256]; // indexed by byte value
};
```

**Purpose:** Enables efficient precomputation of per-DFA-state candidate sets.
Instead of testing every token against every DFA state independently, the trie
allows shared prefix walks: tokens sharing a common prefix share the DFA
simulation for that prefix.

**Build cost:** O(total bytes across all tokens). Typically < 5ms for 128k tokens.

**Memory:** ~3-5 MB for a 128k token vocabulary (dominated by the 256-pointer
arrays at each node).

## Precomputed Token Candidates

For each `(dfa_id, dfa_state)` pair, the engine precomputes which tokens could
possibly survive DFA simulation starting from that state. This is done by walking
the vocab trie and DFA simultaneously:

```
walk(dfa, dfa_state, trie_node, candidates):
    add trie_node.tokens to candidates  // tokens ending here survived so far

    for each byte b where trie_node.children[b] exists:
        next = dfa.transitions[dfa_state][b]
        if next == 0:          skip subtree (dead — prunes thousands of tokens)
        if dfa.accept[next]:   collect ALL tokens in subtree (segment completes,
                               remaining bytes go to a different segment)
        else:                  recurse with (dfa, next, child, candidates)
```

### Adaptive Representation

Each `dfa_state_token_set` stores the smaller of the accept or reject set:

- **Reject-heavy** (most tokens rejected, e.g. grammar expects a digit):
  `accept_heavy = false`, `token_set` = the small accept set.
- **Accept-heavy** (most tokens accepted, e.g. free text in a JSON string):
  `accept_heavy = true`, `token_set` = the small reject set.

This keeps storage proportional to `min(|accept|, |reject|)` rather than always
storing the full candidate set.

### Intersection Across Configs

At runtime, multiple configs may be active with different DFA states. A token
needs simulation if ANY config considers it a candidate. The engine computes
this by intersecting reject sets:

1. Start with `rejected_by_all` = all true.
2. For each config's token set:
   - **Accept-heavy** (token_set = reject set): AND `rejected_by_all` with this
     reject set. Tokens not in the reject set are candidates for this config, so
     clear their `rejected_by_all` bit.
   - **Reject-heavy** (token_set = accept set): clear `rejected_by_all` bits for
     tokens in the accept set (they're candidates for this config).
3. `needs_simulation = !rejected_by_all`

## DFA Operations

### Thompson NFA Construction

Grammar elements (characters, ranges, negation, wildcards) are converted to NFAs
using Thompson's construction:
- `byte_match(lo, hi)` — matches a single byte in [lo, hi]
- `concat(a, b)` — sequential composition
- `alternation([a, b, ...])` — choice via epsilon transitions
- `epsilon_nfa()` — matches the empty string

### UTF-8 Code Point Expansion

Unicode ranges like `[α-ω]` are expanded to byte-level patterns:
1. Split the range at UTF-8 length boundaries (0x80, 0x800, 0x10000).
2. For each sub-range within a single UTF-8 length class, build a byte-level NFA
   using `byte_range_nfa_recursive` which handles lead byte splitting.
3. Combine sub-ranges with alternation.

### Subset Construction (NFA → DFA)

Standard algorithm: compute epsilon closures, build DFA states from sets of NFA
states, expand transitions for all 256 byte values.

### Hopcroft Minimization

Reduces DFA state count by merging equivalent states. Uses iterative partition
refinement: states with the same accept/reject status and identical transition
signatures (modulo partition) are merged.

### DFA Complement and Intersection

Used for negated character classes (`[^...]`):
1. Build DFA for the positive match.
2. Complement it (flip accept states, handling the dead state correctly).
3. Intersect with the valid-UTF-8 DFA (to ensure the negation only matches valid
   UTF-8 characters, not arbitrary byte sequences).

## Files

| File | Purpose |
|------|---------|
| `src/llama-grammar-dfa.h` | All DFA engine structures and interfaces |
| `src/llama-grammar-dfa.cpp` | NFA/DFA construction, trie, precomputation, runtime engine |
| `src/llama-grammar.h` | Grammar struct, parser, internal API |
| `src/llama-grammar.cpp` | Grammar init, apply, accept, clone, parser implementation |
| `tests/test-grammar-dfa.cpp` | DFA engine unit tests |
| `tests/test-grammar-integration.cpp` | End-to-end grammar integration tests |
| `tests/test-llama-grammar.cpp` | Grammar state and validation tests |

## Performance Characteristics

### Compile Time (One-Time)
- NFA construction + subset construction + minimization: milliseconds for typical
  grammars (JSON schemas, function calls).
- Trie construction: < 5ms for 128k vocab.
- Candidate precomputation: 50-200ms depending on grammar complexity and DFA
  state count.

### Token Evaluation (Hot Path)
- **Non-candidates:** O(1) rejection via bitmask lookup.
- **Candidates:** O(L x C) per token for DFA simulation, where L = token byte
  length, C = config count. Typically 100-5000 tokens simulated instead of 128k.
- **Expected speedup:** 10-100x for structured grammars where < 5% of tokens are
  candidates at any given state.

### Memory
- DFA transition tables: 256 x 2 bytes x N states per segment. Typically 5-20
  states per segment, a few KB per rule.
- Vocab trie: ~3-5 MB.
- Candidate sets: 2-12 MB total (depends on grammar; adaptive representation
  minimizes this).
- Compiled grammar is immutable and shared across clones via `shared_ptr`.
