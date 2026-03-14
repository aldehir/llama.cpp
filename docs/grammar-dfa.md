# Grammar Engine: Byte-Level DFA with Token Candidate Caching

## Overview

The grammar engine constrains LLM token sampling to match a BNF-style grammar.
It uses a byte-level DFA approach that eliminates partial UTF-8 handling entirely.

The engine has four layers:

1. **Parse-time:** Grammar text is parsed into an AST, then optimized (terminal
   inlining with size limits, Kleene detection, unreachable rule elimination).
2. **Compile-time:** The AST is compiled to a `compiled_grammar`: terminal
   subtrees become minimized byte-level DFAs, rule references become
   `RULE_CALL` segments. DFAs are canonicalized and deduplicated.
3. **Init-time:** A vocab trie (cached per vocabulary) is walked against each
   DFA state to precompute token candidate sets. Follow-set analysis enables
   context-dependent token filtering. Candidate sets use an adaptive
   representation that stores whichever of the accept or reject set is smaller.
4. **Runtime:** Precomputed candidate sets narrow the token space to a small
   subset that must be simulated. Non-candidates are rejected in O(1).

## Architecture

### Compilation Pipeline

```
Grammar text (GBNF)
  → AST parser (llama_grammar_ast_parser::parse)
  → AST: vector<ast_node_ptr>, one per rule
  → Optimization (ast_parser.optimize):
      1. Classify rules: PURELY_TERMINAL / TRANSITIVELY_TERMINAL / NON_TERMINAL
      2. Inline terminal refs: replace RULE_REFs to terminal rules with their body
         (subject to MAX_INLINE_AST_NODES size limit — see Inlining Size Limit)
      3. Flatten nested SEQUENCE/ALTERNATION nodes
  → Compiler (compile_grammar_from_ast):
      1. Compute reachable rules from start rule (BFS)
      2. For each reachable rule:
           For each alternate (ALTERNATION children or single body):
             Split into segments at RULE_REF boundaries
             For each terminal segment:
               → ast_to_nfa: AST → Thompson NFA (byte-level)
                 Handles LITERAL, CHAR_CLASS, SEQUENCE, ALTERNATION, REPETITION,
                 EXCLUSION (via Aho-Corasick complement DFA)
               → Subset construction (NFA → DFA)
               → Hopcroft minimization + BFS canonicalization
               → Deduplicate: reuse existing DFA if identical (via operator==)
               → byte_dfa stored in compiled_grammar.dfas[]
             Rule calls become compiled_segment::RULE_CALL
      3. Detect Kleene patterns: self-recursive rules like R ::= body R | ε
         are rewritten as DFA(body+), avoiding recursive call overhead
      4. Compute follow sets (FIRST/FOLLOW analysis on compiled grammar)
  → compiled_grammar (immutable, shared across clones via shared_ptr)
```

Key property: the grammar operates on **raw bytes**, not Unicode code points.
UTF-8 code point ranges like `[a-z]` or `[α-ω]` are expanded to byte-level NFA
patterns at compile time. Multi-byte UTF-8 sequences become intermediate DFA
states. This eliminates all partial UTF-8 handling.

### AST

The grammar is first parsed into an abstract syntax tree. Each rule becomes an
`ast_node_ptr` (a `grammar_ast_node`). Node types:

| Type | Description |
|------|-------------|
| `LITERAL` | Exact code points from `"..."` strings |
| `CHAR_CLASS` | Character ranges `[a-z]`, negated `[^...]`, wildcard `.` |
| `RULE_REF` | Reference to a named rule |
| `SEQUENCE` | Concatenation of children |
| `ALTERNATION` | Choice among children |
| `REPETITION` | Child repeated `{min, max}` times (supports `*`, `+`, `?`) |
| `EXCLUSION` | `!("str1" \| "str2")` — matches any string not containing excluded substrings |

#### Terminal Classification

Rules are classified to enable inlining:

- **PURELY_TERMINAL:** No `RULE_REF` nodes anywhere in the subtree.
- **TRANSITIVELY_TERMINAL:** All referenced rules are themselves terminal.
- **NON_TERMINAL:** References non-terminal rules.

Purely terminal and transitively terminal rules are inlined at their call sites,
replacing the `RULE_REF` with a copy of the rule body, subject to the inlining
size limit (see below). This reduces the number of compiled rule calls and
enables larger DFA segments.

#### Inlining Size Limit

To prevent DFA state explosion from cascading inlining, the optimizer enforces
a size limit (`MAX_INLINE_AST_NODES = 50`). Before inlining a terminal rule
reference, the optimizer clones the target rule's AST, recursively inlines its
own references, and measures the final node count. If the fully-expanded AST
exceeds the limit, the `RULE_REF` is kept as-is and compiled as a `RULE_CALL`
segment instead. This ensures that large terminal rules (e.g. JSON schema
rules with many inlined fields) remain as separate compiled rules with their
own DFA segments, rather than being absorbed into a single monolithic DFA.

#### Unreachable Rule Elimination

Before compilation, a BFS from the start rule identifies reachable rules.
Unreachable rules are skipped entirely, reducing compiled grammar size and
precomputation cost.

### Data Structures

#### `byte_dfa`

The core automaton. Each DFA has:
- `transitions[state][byte] → next_state` (state 0 = dead/reject)
- `accept[state]` — whether the state is accepting
- `accept_can_continue[state]` — whether the state accepts AND has outgoing
  transitions (used for context-dependent token classification)
- `start_state` — typically state 1

Transition lookup is O(1): a single array index.

#### `compiled_grammar`

Immutable after construction. Contains:
- `dfas[]` — all compiled DFA segments (deduplicated)
- `rules[]` — each rule has alternates, each alternate is a sequence of segments
  (`DFA_MATCH` or `RULE_CALL`)
- `dfa_candidates[]` — precomputed per-DFA-state token candidate sets
- `dfa_follow_dfas[]` — follow DFA sets for context-dependent filtering:
  `dfa_follow_dfas[dfa_id]` = sorted list of DFA IDs that can follow this DFA
- `start_rule` — entry point

#### `parse_config`

The runtime state for one active parse path:
- `current` — a `grammar_position` (rule_id, alternate_idx, segment_idx, dfa_state)
- `call_stack` — vector of `grammar_call_frame` for rule call/return

Multiple `parse_config`s may be active simultaneously when the grammar has
ambiguous rule choices (multiple alternates). For well-formed grammars (JSON,
function calls), the config count is typically 1-4.

### Kleene Star/Plus Optimization

The compiler detects self-recursive rules that match the Kleene pattern:

```
R ::= body R | ε       →  rewritten as DFA(body+)
```

When a rule has exactly two alternates — one that ends with a self-reference and
one that is empty — the terminal body segments are compiled into a single
`DFA(body+)` using the NFA `kleene_plus` primitive. This eliminates recursive
rule calls at runtime and produces a single DFA that matches one or more
repetitions directly.

### DFA Canonicalization and Deduplication

After Hopcroft minimization, each DFA is **canonicalized** by renumbering states
in BFS order from the start state. This ensures that structurally identical DFAs
always have the same state numbering, enabling exact equality comparison via
`operator==`.

During compilation, each new DFA is compared against all existing DFAs. If an
identical DFA already exists, the existing index is reused. This avoids
redundant precomputation for DFAs that appear in multiple rules (common in
grammars with repeated patterns like JSON keys).

### Runtime: Token Validation

`llama_grammar_apply_impl` is the hot path, called once per sampling step. It
sets `logit = -INFINITY` for tokens that violate the grammar.

1. For each active config, look up its `(dfa_id, dfa_state)` in the precomputed
   `dfa_candidates` table to get a token set.
2. Intersect reject sets across all configs to build a `needs_simulation` set.
3. Add context-dependent tokens to the simulation set.
4. Only simulate tokens that need it.
5. Non-candidates are rejected in O(1).

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

A byte-level trie over all token strings in the vocabulary.

```
struct vocab_trie_node {
    vector<llama_token> tokens;               // tokens terminating at this node
    unique_ptr<vocab_trie_node> children[256]; // indexed by byte value
    vector<int32_t> subtree_tokens;           // cached: all token IDs at or below
};
```

**Caching:** The trie is built once per `llama_vocab` and cached on it via
`vocab->get_grammar_trie()`. All grammars using the same vocabulary share the
same trie instance (thread-safe, returned as `shared_ptr`).

**Subtree caching:** Each trie node precomputes `subtree_tokens` — a sorted list
of all token IDs at or below that node. This enables efficient batch collection
when an entire subtree is accepted or needs context classification.

**Build cost:** O(total bytes across all tokens). Typically < 5ms for 128k tokens.

**Memory:** ~3-5 MB for a 128k token vocabulary (dominated by the 256-pointer
arrays at each node).

## Precomputed Token Candidates

For each `(dfa_id, dfa_state)` pair, the engine precomputes which tokens could
possibly survive DFA simulation starting from that state.

### Three-Way Token Classification

Tokens are classified into three categories per DFA state:

1. **Guaranteed accept:** The token's bytes stay within the DFA without reaching
   an accept state, or end exactly at an accept state. These tokens are known to
   be valid without any further simulation.

2. **Context-dependent:** The DFA reaches an accept state *mid-token* — the
   current DFA segment completes, but the token has remaining bytes that must be
   processed by the *next* segment. Whether the token is valid depends on what
   follows in the grammar. These are stored in a separate `context_set` and
   always require simulation.

3. **Rejected:** The DFA transitions to the dead state (0) during the token.
   These are rejected in O(1).

### Multi-State Walk

Instead of walking the vocab trie independently for each DFA state, the
precomputation walks all states simultaneously using `walk_multi`:

```
walk_multi(dfa, active_groups, trie_node, accepted_bits, context_bits):
    active_groups: map<dfa_state → list<original_starting_states>>

    for each byte b where trie_node.children[b] exists:
        group the active states by their transition on byte b
        for each group (next_state → starting_states):
            if next_state == 0:     skip (dead)
            if dfa.accept[next]:    handle accept (exact vs context)
            else:                   recurse with merged group
```

This amortizes trie walk cost across all DFA states, since states that transition
to the same next state share the recursive walk.

### Follow-Set Filtering for Context Tokens

Context-dependent tokens (where the DFA accepts mid-token) are further filtered
using **follow DFA sets**. For each DFA, the engine precomputes which DFAs can
appear next in the grammar (see FIRST/FOLLOW Sets below). When a token reaches
a DFA accept state with remaining bytes, those bytes are walked through the
follow DFAs. If no follow DFA can match the remaining bytes, the token is
pruned from the context set.

This significantly reduces the number of tokens requiring runtime simulation,
especially for grammars where the follow set is restrictive (e.g., after a JSON
key, only `:` can follow).

### Adaptive Representation

Each `dfa_state_token_set` stores the smaller of the accept or reject set:

- **Reject-heavy** (most tokens rejected, e.g. grammar expects a digit):
  `accept_heavy = false`, `token_set` = the small accept set.
- **Accept-heavy** (most tokens accepted, e.g. free text in a JSON string):
  `accept_heavy = true`, `token_set` = the small reject set.

Context-dependent tokens are stored separately in `context_set`.

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
3. Add all context-dependent tokens to the simulation set.
4. `needs_simulation = !rejected_by_all`

## FIRST/FOLLOW Sets

The engine computes FIRST and FOLLOW sets over the compiled grammar to determine
which DFAs can appear after each DFA. This is used during precomputation to
filter context-dependent tokens.

### Algorithm

1. **Nullable analysis:** Determine which rules can match the empty string
   (a rule is nullable if any alternate has all-nullable segments).

2. **FIRST sets:** For each rule, compute which DFA IDs are reachable from the
   rule's start position. A rule's FIRST set includes the first DFA of each
   alternate, plus (if that segment's rule call is nullable) the DFAs reachable
   through the nullable prefix.

3. **FOLLOW sets:** For each DFA, compute which DFA IDs can appear after it.
   Walk every alternate of every rule: for each segment, the DFAs that follow it
   include the FIRST sets of subsequent segments. If a rule call at the end of
   an alternate is nullable, the FOLLOW set of the calling rule propagates
   through.

4. Fixed-point iteration continues until no FOLLOW set changes.

Result: `dfa_follow_dfas[dfa_id]` = sorted list of DFA IDs that can follow DFA
`dfa_id` in any valid parse.

## DFA Operations

### Thompson NFA Construction

Grammar elements (characters, ranges, negation, wildcards, repetition) are
converted to NFAs using Thompson's construction:
- `byte_match(lo, hi)` — matches a single byte in [lo, hi]
- `concat(a, b)` — sequential composition
- `alternation([a, b, ...])` — choice via epsilon transitions
- `epsilon_nfa()` — matches the empty string
- `kleene_star(a)` — zero or more repetitions
- `kleene_plus(a)` — one or more repetitions

The `ast_to_nfa` function converts AST nodes to NFAs recursively, handling
`REPETITION` nodes by expanding `{min, max}` into the appropriate number of
concatenated copies with optional `kleene_star` suffixes.

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
signatures (modulo partition) are merged. After minimization, states are
renumbered in BFS order from the start state (canonicalization).

### DFA Complement and Intersection

Used for negated character classes (`[^...]`):
1. Build DFA for the positive match.
2. Complement it (flip accept states, handling the dead state correctly).
3. Intersect with the valid-UTF-8 DFA (to ensure the negation only matches valid
   UTF-8 characters, not arbitrary byte sequences).

### Exclusion DFA (Aho-Corasick Complement)

The `!("string1" | "string2")` GBNF syntax matches any string that does not
contain any of the excluded substrings. It compiles directly to a complement
DFA using Aho-Corasick multi-pattern matching:

1. **Trie construction:** Insert the UTF-8 byte sequences of all excluded
   strings into a trie.
2. **Failure links:** Compute Aho-Corasick failure links via BFS, propagating
   output flags through suffix links (so a node that is a suffix of a needle
   is also marked as matching).
3. **Complement DFA:** Map each trie node to a DFA state. States where a
   needle has been fully matched become dead (state 0). All other states are
   accepting with `accept_can_continue = true`.

The resulting DFA has O(sum of needle lengths) states — far smaller than the
hand-written exclusion patterns it replaces. For example, excluding
`" to=functions."` and `"<|message|>"` produces a ~25-state DFA, compared to
hundreds of NFA states from the equivalent 25-alternative pattern.

In `compile_node`, `EXCLUSION` nodes are compiled directly to `byte_dfa`
segments (bypassing NFA construction). In `ast_to_nfa` (when an `EXCLUSION`
appears inside a terminal alternation or repetition), the complement DFA is
converted back to NFA form using the same pattern as negated character classes.

**GBNF syntax:** Only literal strings are allowed inside `!()`. This ensures
the Aho-Corasick construction is always applicable with guaranteed linear
state count. Arbitrary grammar expressions inside negation would require
complementing an arbitrary DFA, which can itself cause state explosion.

## Files

| File | Purpose |
|------|---------|
| `src/llama-grammar-ast.h` | AST node types, parser interface, terminal classification |
| `src/llama-grammar-ast.cpp` | AST parser, optimization passes, AST-to-NFA conversion |
| `src/llama-grammar-dfa.h` | DFA engine structures: byte_dfa, compiled_grammar, trie, candidates |
| `src/llama-grammar-dfa.cpp` | NFA/DFA construction, trie, precomputation, follow sets, runtime engine |
| `src/llama-grammar.h` | Grammar struct, internal API |
| `src/llama-grammar.cpp` | Grammar init, apply, accept, clone |
| `tests/test-grammar-dfa.cpp` | DFA engine unit tests |
| `tests/test-grammar-integration.cpp` | End-to-end grammar integration tests |
| `tests/test-llama-grammar.cpp` | Grammar state and validation tests |

## Performance Characteristics

### Compile Time (One-Time)
- AST parsing + optimization: sub-millisecond for typical grammars.
- NFA construction + subset construction + minimization: milliseconds for typical
  grammars (JSON schemas, function calls).
- DFA canonicalization + deduplication: negligible overhead, saves precomputation
  time by avoiding redundant DFAs.
- Trie construction: < 5ms for 128k vocab (amortized to zero if cached).
- Follow-set computation: sub-millisecond.
- Candidate precomputation: 50-200ms depending on grammar complexity and DFA
  state count. Multi-state walk and follow-set filtering reduce this
  significantly for complex grammars.

### Token Evaluation (Hot Path)
- **Non-candidates:** O(1) rejection via bitmask lookup.
- **Context-dependent tokens:** Require simulation but are filtered by follow
  sets during precomputation, keeping their count small.
- **Candidates:** O(L x C) per token for DFA simulation, where L = token byte
  length, C = config count. Typically 100-5000 tokens simulated instead of 128k.
- **Expected speedup:** 10-100x for structured grammars where < 5% of tokens are
  candidates at any given state.

### Memory
- DFA transition tables: 256 x 2 bytes x N states per segment. Typically 5-20
  states per segment, a few KB per rule.
- Vocab trie: ~3-5 MB (shared across all grammars via vocab cache).
- Candidate sets: 2-12 MB total (depends on grammar; adaptive representation
  and follow-set filtering minimize this).
- Compiled grammar is immutable and shared across clones via `shared_ptr`.
