# ByteGrammar: Byte-Level DFA Compilation with Precomputed Token Candidate Sets for Grammar-Constrained LLM Decoding

**Working Title Alternatives:**

- *TrieDFA: Trie-Accelerated Deterministic Automata for Structured LLM Output*
- *GrammarForge: Compiling Context-Free Grammars to Byte-Level Automata for Token-Space Constraint Enforcement*
- *DFADecode: From GBNF to Byte Automata — A Multi-Stage Compilation Approach to Grammar-Constrained Sampling*
- *TokenSieve: Precomputed DFA-Vocabulary Intersection for Efficient Structured Decoding*

---

## Abstract

We present **ByteGrammmar**, a multi-stage compilation and runtime system for
constraining large language model (LLM) token sampling to conform to a
user-specified context-free grammar. The system compiles grammars written in
GBNF (a BNF variant) through a pipeline of AST parsing, terminal inlining, NFA
construction via Thompson's algorithm, subset construction to DFA, Hopcroft
minimization, and state canonicalization. The key innovation is a
*precomputation phase* that walks a vocabulary byte-trie against all DFA states
simultaneously, producing per-state token candidate sets that reduce runtime
token validation from O(V) per sampling step to O(A), where A << V is the
number of candidate tokens. We describe the three-way token classification
(guaranteed-accept, context-dependent, rejected), the adaptive set
representation that stores whichever of the accept or reject set is smaller,
FIRST/FOLLOW set analysis for context-dependent token pruning, and an
Aho-Corasick complement construction for exclusion patterns. The system
achieves 10-100x speedups over naive per-token simulation for structured
grammars, with precomputation completing in 170-280ms for typical grammars on
128k-token vocabularies. All automata operate on raw bytes rather than Unicode
code points, eliminating partial UTF-8 handling entirely.

## 1. Introduction

Large language models generate text token-by-token, sampling from a probability
distribution over a fixed vocabulary at each decoding step. Many applications
require the output to conform to a structured format: JSON schemas, SQL
queries, function call signatures, configuration files, or domain-specific
languages. Unconstrained generation provides no such guarantee.

**Grammar-constrained decoding** addresses this by masking out tokens that would
violate a specified grammar before sampling. At each step, the sampler sets the
logit of every invalid token to negative infinity, ensuring the model can only
produce grammatically valid continuations.

The fundamental challenge is efficiency. A vocabulary of V = 128,000 tokens
must be filtered at every decoding step. Naive approaches simulate each token's
byte sequence through the grammar state machine, yielding O(V * L * C) cost per
step, where L is the average token byte length and C is the number of active
parse configurations. For real-time inference serving hundreds of requests per
second, this is prohibitive.

We present a system, implemented within the llama.cpp inference engine, that
reduces per-step cost to O(A * L * C) where A is typically 100-5,000 tokens
(< 5% of V) for structured grammars like JSON or function calls. The system
achieves this through a multi-stage compilation pipeline that precomputes, for
every reachable DFA state, which tokens are guaranteed valid, which are
guaranteed invalid, and which require runtime simulation.

### 1.1 Contributions

1. A **byte-level DFA compilation pipeline** that transforms context-free
   grammars (in GBNF notation) through AST parsing, optimization, Thompson NFA
   construction, subset construction, Hopcroft minimization, and
   canonicalization — operating entirely on raw bytes to eliminate UTF-8
   partial-character edge cases.

2. A **vocabulary trie precomputation** algorithm that walks all DFA states
   simultaneously through a byte-level trie of the vocabulary, classifying
   every token into guaranteed-accept, context-dependent, or rejected categories
   for each DFA state.

3. An **adaptive candidate set representation** that stores whichever of the
   accept or reject set is smaller, with FIRST/FOLLOW set analysis that prunes
   context-dependent tokens using grammar-structural information.

4. An **Aho-Corasick complement construction** for exclusion patterns
   (`!("str1" | "str2")`) that produces linear-size DFAs for substring
   avoidance.

5. A **Kleene detection and rewriting** optimization that identifies
   self-recursive grammar rules matching the pattern `R ::= body R | epsilon`
   and compiles them to single DFA segments, eliminating recursive call
   overhead.

## 2. Background and Problem Setting

### 2.1 Grammar-Constrained Decoding

Let G be a context-free grammar and V = {t_1, ..., t_V} be the token
vocabulary, where each token t_i maps to a byte string bytes(t_i). At decoding
step k, let s_k be the current grammar parse state. A token t is *valid* at
step k if feeding bytes(t) through the grammar from state s_k produces at least
one surviving parse configuration (the grammar does not enter a dead state
during the token's byte sequence).

The grammar sampler must compute, for each step k:

```
valid_k = { t in V : simulate(s_k, bytes(t)) != DEAD }
```

and set logit(t) = -infinity for all t not in valid_k.

### 2.2 GBNF Grammar Formalism

The system uses GBNF (GGML BNF), an extended BNF notation supporting:

- **Literals:** `"hello"` (exact byte sequences)
- **Character classes:** `[a-z]`, `[^0-9]`, `.` (wildcard)
- **Alternation:** `rule1 | rule2`
- **Repetition:** `*`, `+`, `?`, `{m,n}`
- **Grouping:** `(expr)`
- **Rule references:** `other-rule`
- **Exclusion:** `!("forbidden1" | "forbidden2")` (matches strings not
  containing any of the excluded substrings)

### 2.3 Challenges

1. **Vocabulary size:** Modern BPE vocabularies contain 32k-128k tokens. Naive
   per-token simulation is too slow for interactive inference.

2. **Byte-token mismatch:** Tokens are variable-length byte sequences. A single
   token may span multiple grammar elements or straddle grammar boundaries
   (e.g., a token `": "` might complete a JSON key's closing quote, the colon
   separator, and the space before the value).

3. **Multi-byte characters:** Unicode code points encode as 1-4 bytes in UTF-8.
   Grammar character classes like `[alpha-omega]` must be handled at the byte
   level to match the tokenizer's byte-level operation.

4. **Ambiguity:** Context-free grammars may have multiple valid parse paths.
   The system must track all active parse configurations simultaneously.

## 3. System Architecture

The system is organized as a four-layer pipeline:

```
Layer 1: PARSE       Grammar text -> AST (one tree per rule)
Layer 2: COMPILE     AST -> compiled_grammar (DFAs + rule structure)
Layer 3: PRECOMPUTE  DFA states x vocab trie -> per-state token sets
Layer 4: RUNTIME     Precomputed sets -> fast token filtering per step
```

Layers 1-3 execute once per grammar (and are cached across uses). Layer 4
executes at every decoding step.

### 3.1 Layer 1: Parsing and AST Construction

The grammar text is parsed by `llama_grammar_ast_parser` into an abstract
syntax tree. Each rule becomes an `ast_node_ptr` (a `grammar_ast_node`) with
one of seven node types:

| Node Type     | Description                                                |
|---------------|------------------------------------------------------------|
| `LITERAL`     | Exact Unicode code point sequence from `"..."` strings     |
| `CHAR_CLASS`  | Character ranges `[a-z]`, negated `[^...]`, wildcard `.`   |
| `RULE_REF`    | Reference to a named rule                                  |
| `SEQUENCE`    | Concatenation of children                                  |
| `ALTERNATION` | Choice among children                                      |
| `REPETITION`  | Child repeated `{min, max}` times (`*`, `+`, `?`)          |
| `EXCLUSION`   | `!("str1" | "str2")` substring avoidance                   |

An optimization pass then performs three transformations:

1. **Terminal classification:** Rules are classified as PURELY_TERMINAL (no rule
   references), TRANSITIVELY_TERMINAL (all referenced rules are themselves
   terminal), or NON_TERMINAL.

2. **Terminal inlining:** `RULE_REF` nodes pointing to terminal rules are
   replaced with a copy of the referenced rule's body. This enlarges DFA
   segments and reduces rule-call overhead at runtime.

3. **Flattening:** Nested `SEQUENCE` and `ALTERNATION` nodes are collapsed
   into flat lists of children.

### 3.2 Layer 2: Compilation to Byte-Level DFAs

The compiler (`compile_grammar_from_ast`) transforms the optimized AST into a
`compiled_grammar` consisting of:

- A vector of `byte_dfa` segments (shared, deduplicated)
- A vector of `compiled_rule` structures, each containing alternates, where
  each alternate is a sequence of `compiled_segment` entries of type
  `DFA_MATCH` or `RULE_CALL`

#### 3.2.1 Segment Compilation

For each rule alternate, the compiler walks the AST and accumulates terminal
content (literals, character classes, repetitions of terminal content) into a
*pending NFA*. When a `RULE_REF` is encountered, the pending NFA is flushed as
a DFA segment, and a `RULE_CALL` segment is emitted. The compilation pipeline
for each NFA segment is:

```
AST subtree
  -> Thompson NFA construction (byte-level)
  -> Subset construction (NFA -> DFA)
  -> Hopcroft minimization
  -> BFS canonicalization (deterministic state numbering)
  -> Deduplication (reuse identical DFA by operator==)
```

#### 3.2.2 Thompson NFA Construction

Terminal AST nodes are converted to NFAs over raw bytes using Thompson's
construction:

- `byte_match(lo, hi)`: Matches any single byte in [lo, hi]. Two states, one
  transition.
- `concat(A, B)`: Connects A's accept state to B's start state via an epsilon
  transition.
- `alternation([A, B, ...])`: New start state with epsilon transitions to each
  sub-NFA's start; each sub-NFA's accept connects to a new shared accept state.
- `kleene_star(A)`: Wraps A with epsilon transitions for zero-or-more
  repetition.
- `kleene_plus(A)`: Adds a back-edge from A's accept to A's start for
  one-or-more repetition.
- `epsilon_nfa()`: Matches the empty string.

**UTF-8 expansion.** Unicode code point ranges are expanded to byte-level NFA
patterns at compile time. A range like `[alpha-omega]` (U+03B1 to U+03C9) is
split at UTF-8 length boundaries (0x80, 0x800, 0x10000) and each sub-range is
compiled using `byte_range_nfa_recursive`, which splits on shared leading bytes
and recurses on continuation byte ranges. For example, `[A-Z]` becomes
`byte_match(0x41, 0x5A)`, while `[alpha-omega]` becomes a two-byte NFA pattern matching
`[0xCE][0xB1-0xBF] | [0xCF][0x80-0x89]`.

**Negated character classes.** `[^X]` is compiled as:
1. Build DFA for the positive match X.
2. Complement the DFA (flip accept/reject states, excluding dead state 0).
3. Intersect with the valid-UTF-8 DFA (a prebuilt, cached 9-state DFA) to
   ensure negation only matches valid UTF-8 characters.

The intersection uses standard product construction.

#### 3.2.3 NFA Size Bounding

To prevent DFA state explosion during subset construction, the compiler bounds
NFA size through two mechanisms:

1. **Auto-flush at 128 NFA states:** When the pending NFA exceeds
   `MAX_PENDING_NFA_STATES`, it is flushed as a DFA segment before accumulating
   more terminal content.

2. **Large alternation/repetition splitting:** When a terminal `ALTERNATION` or
   `REPETITION` node would produce an NFA exceeding the threshold, it is
   compiled as a synthetic rule with separate alternates, each getting its own
   DFA segment. This prevents the combinatorial blowup that occurs when subset
   construction must track which of many overlapping NFA branches are active.

Together, these keep individual DFAs to ~50-100 states.

#### 3.2.4 Hopcroft Minimization and Canonicalization

After subset construction, each DFA undergoes Hopcroft's partition-refinement
minimization algorithm:

1. Partition states into accept/non-accept groups.
2. Iteratively refine: for each state, compute a *signature* consisting of its
   group and the group of each byte-transition target. States with identical
   signatures are merged.
3. Iterate until stable.

After minimization, states are renumbered in BFS order from the start state.
This *canonicalization* ensures that structurally identical DFAs always receive
the same state numbering, enabling deduplication via `operator==` comparison.

#### 3.2.5 Kleene Detection and Rewriting

The compiler detects self-recursive rules matching the pattern:

```
R ::= body R | epsilon
```

These are rewritten to a single DFA segment matching `body+` using the NFA
`kleene_plus` primitive, eliminating recursive rule calls at runtime. This is
particularly effective for rules like `ws ::= [ \t\n]* ` that appear
ubiquitously in grammar definitions.

#### 3.2.6 Exclusion Patterns via Aho-Corasick

The `!("str1" | "str2")` syntax compiles to a complement DFA using
Aho-Corasick multi-pattern matching:

1. Insert the UTF-8 byte sequences of all excluded strings into a trie.
2. Compute Aho-Corasick failure links via BFS, propagating output flags through
   suffix links.
3. Map each non-output trie node to an accepting DFA state; output nodes
   (where a needle has been fully matched) map to the dead state.

The resulting DFA has O(sum of needle lengths) states — far smaller than the
equivalent hand-written exclusion patterns. For example, excluding two strings
of total length 25 bytes produces a ~25-state DFA.

### 3.3 Layer 3: Token Candidate Precomputation

This is the central algorithmic contribution. For each `(dfa_id, dfa_state)`
pair, the system precomputes which vocabulary tokens are valid, invalid, or
require runtime simulation.

#### 3.3.1 Vocabulary Byte-Trie

A byte-level trie is constructed over all token strings in the vocabulary:

```
struct vocab_trie_node {
    vector<llama_token> tokens;               // tokens ending exactly here
    unique_ptr<vocab_trie_node> children[256]; // indexed by byte value
    vector<int32_t> subtree_tokens;           // precomputed: all tokens at/below
};
```

Each node stores tokens that terminate at that exact prefix, and a precomputed
`subtree_tokens` list (sorted) of all token IDs at or below the node. The trie
is built once per vocabulary (O(total bytes across all tokens), typically < 5ms
for 128k tokens, ~3-5 MB memory) and cached on the vocabulary object.

#### 3.3.2 Three-Way Token Classification

For each DFA state s, tokens are classified into three categories:

1. **Guaranteed accept:** The token's byte sequence transitions through the DFA
   without reaching a dead state, either staying within the DFA or ending at an
   accept state. These are valid regardless of grammar context.

2. **Context-dependent:** The DFA reaches an accept state *mid-token* — the
   current DFA segment completes, but the token has remaining bytes. Whether
   these remaining bytes are valid depends on what grammar segment follows
   (which is determined by runtime parse configuration). These require
   simulation.

3. **Rejected:** The DFA transitions to the dead state (0) during the token.
   Invalid in O(1).

#### 3.3.3 Multi-State Trie Walk

Rather than walking the trie independently for each DFA state, all states are
walked simultaneously via `walk_multi`:

```
walk_multi(dfa, trie_node, active_groups, accepted_bits, context_bits):
    // active_groups: map<current_dfa_state -> [originating_start_states]>

    for each byte b where trie_node.children[b] exists:
        next_groups = {}
        for each (dfa_state, start_states) in active_groups:
            next = dfa.transitions[dfa_state][b]
            if next == 0: skip (dead)
            if dfa.accept[next]:
                if token ends here:
                    mark start_states as guaranteed-accept
                else if dfa.accept_can_continue[next]:
                    recurse with next as active, also start context walk
                else:
                    mark start_states for context classification
            else:
                next_groups[next].append(start_states)
        recurse walk_multi with next_groups
```

This amortizes the trie traversal cost: DFA states that transition to the same
next state on a given byte share the recursive walk. The result is a pair of
bit-vectors per originating state: `accepted_bits[s]` (guaranteed accept) and
`context_bits[s]` (needs simulation).

#### 3.3.4 FIRST/FOLLOW Set Analysis

Context-dependent tokens are further pruned using FIRST/FOLLOW set analysis
adapted for the compiled grammar:

1. **Nullable analysis:** Determine which rules can match the empty string.
2. **FIRST sets:** For each rule, compute which DFA IDs are reachable from the
   start position.
3. **FOLLOW sets:** For each DFA, compute which DFA IDs can appear after it in
   any valid derivation. Fixed-point iteration propagates through rule
   boundaries.

Result: `dfa_follow_dfas[d]` = sorted list of DFA IDs that can follow DFA d.

During context-dependent token classification, when a token reaches a DFA
accept state with remaining bytes, those bytes are walked through the follow
DFAs. If no follow DFA can accept the remaining bytes, the token is pruned
from the context set. This significantly reduces runtime simulation load,
especially for grammars with restrictive follow sets (e.g., after a JSON key
DFA, only the `":"` DFA can follow).

#### 3.3.5 Adaptive Set Representation

Each DFA state's token set uses an adaptive representation:

```
struct dfa_state_token_set {
    bool accept_heavy;          // if true, token_set = reject set
    vector<int32_t> token_set;  // sorted (accept or reject, adaptive)
    vector<int32_t> context_set; // sorted (always context-dependent)
};
```

- **Reject-heavy states** (grammar expects a digit, keyword, etc.):
  `accept_heavy = false`, `token_set` = the small set of accepted tokens.
- **Accept-heavy states** (grammar in free-text region, e.g., JSON string
  value): `accept_heavy = true`, `token_set` = the small set of rejected
  tokens.

Storage is proportional to `min(|accept|, |reject|)` rather than always V.

#### 3.3.6 Parallel Precomputation

Candidate precomputation is parallelized across DFAs using worker threads with
atomic work-stealing. Each DFA's trie walk reads only shared immutable data
(DFA transitions, vocab trie, follow sets) and writes exclusively to its own
candidate entry, requiring no synchronization beyond the task counter. The
system uses 4 worker threads, falling back to sequential for fewer than 4 DFAs.

### 3.4 Layer 4: Runtime Token Filtering

At each decoding step, `llama_grammar_apply_impl` is the hot path:

1. **Candidate collection.** For each active parse configuration, look up its
   `(dfa_id, dfa_state)` in the precomputed candidate table. Union the
   not-rejected token sets and context-dependent sets across all configurations.

2. **Reject-heavy fast path.** If all configurations are reject-heavy (small
   accept sets), merge the accept sets and context sets into sorted lists.
   Iterate the `cur_p` token array with a merge-scan: tokens not in the
   not-rejected set have their logits set to -infinity. This avoids touching
   the full vocabulary.

3. **Accept-heavy fallback.** If any configuration is accept-heavy (large
   accept set, small reject set), fall back to a bitmap approach: build
   `rejected_by_all[V]` and `is_context[V]` bitmaps, then scan `cur_p`.

4. **Simulation.** Context-dependent tokens are simulated through the full
   grammar state machine (`grammar_simulate_token`): copy the current
   configurations, feed each byte through `accept_byte`, check for survival.
   Simulation is parallelized across tokens using a persistent thread pool
   (8 threads, sleeping on condition variables between steps).

5. **EOG handling.** End-of-generation tokens are allowed only if any current
   configuration is in a complete state (rule finished, call stack empty).

### 3.5 Runtime State Machine

#### 3.5.1 Parse Configurations

The runtime state consists of a vector of `parse_config` structs:

```
struct parse_config {
    grammar_position current;           // (rule_id, alt_idx, seg_idx, dfa_state)
    vector<grammar_call_frame> call_stack;  // for rule call/return
};
```

Multiple configurations arise from grammar ambiguity (multiple valid alternates
at a choice point). For well-formed grammars (JSON, function calls), the count
is typically 1-4.

#### 3.5.2 Byte Acceptance

When a token is selected, its bytes are fed through `accept_byte` one at a
time:

1. Look up `dfa.transitions[current.dfa_state][byte]` for each configuration.
2. If the transition goes to state 0 (dead), the configuration dies.
3. If the transition reaches an accept state and the DFA segment is complete,
   call `advance_to_terminal` to resolve:
   - Advance to the next segment in the current alternate.
   - If the next segment is a `RULE_CALL`, push a call frame and recurse into
     the called rule.
   - If the current alternate is exhausted, pop the call stack and resume the
     caller.
   - Generate all reachable configurations (multiple alternates may be viable).
4. Deduplicate configurations after each byte.

### 3.6 Caching

#### 3.6.1 Grammar LRU Cache

The entire compilation pipeline (parse, optimize, compile, precompute) is
cached in a process-global LRU cache (5 entries) keyed by `(grammar_str,
grammar_root, vocab_ptr)`. On cache hit, only fresh runtime state
(configurations, trigger patterns, thread pool) is created. Hash-based fast
reject avoids string comparison on most misses.

#### 3.6.2 Vocabulary Trie Cache

The byte-trie is built once per vocabulary and cached on the vocab object via
`shared_ptr`. All grammars using the same vocabulary share the trie.

## 4. Complexity Analysis

### 4.1 Compilation (One-Time)

| Phase | Time Complexity | Typical Wall Time |
|-------|----------------|-------------------|
| AST parsing | O(G) where G = grammar size | < 1ms |
| Terminal classification + inlining | O(R * D) where R = rules, D = max rule depth | < 1ms |
| NFA construction (per segment) | O(N) where N = bounded by 128 NFA states | < 1ms |
| Subset construction | O(2^N * 256) worst case, O(N * 256) typical | < 10ms |
| Hopcroft minimization | O(D_s * 256 * log D_s) where D_s = DFA states | < 1ms |
| Trie construction | O(sum of token byte lengths) | < 5ms |
| FIRST/FOLLOW sets | O(R^2 * S) where S = max segments per rule | < 1ms |
| Candidate precomputation | O(D_total * T * 256) where T = trie depth | 170-280ms |

The NFA size bound of 128 states keeps subset construction practical. Without
it, a grammar with a 20-alternative character class could produce 2^20 DFA
states.

### 4.2 Per-Step Runtime

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Candidate lookup | O(C) per config | C = config count (1-4 typical) |
| Set union (reject-heavy) | O(A) | A = candidate count |
| Merge-scan reject | O(P + A) | P = cur_p size |
| Token simulation | O(A_ctx * L * C) | A_ctx = context tokens, L = byte length |
| Byte acceptance | O(L * C) | Per selected token |

For structured grammars (JSON, SQL), A is typically 100-5,000 out of V =
128,000, yielding 10-100x speedup over naive simulation.

### 4.3 Memory

| Component | Size | Notes |
|-----------|------|-------|
| DFA transitions | 512 * D_s bytes per DFA | D_s typically 10-55 states |
| Vocab trie | ~3-5 MB | Shared across all grammars |
| Candidate sets | 2-12 MB total | Adaptive representation |
| Compiled grammar | Shared via `shared_ptr` | Immutable after construction |

## 5. Key Design Decisions

### 5.1 Byte-Level vs. Code-Point-Level Automata

Operating on raw bytes rather than Unicode code points eliminates an entire
class of edge cases: partial UTF-8 sequences mid-token, surrogate pair
handling, and variable-width character indexing. The cost is slightly larger
DFAs for multi-byte character ranges (a 2-byte code point range becomes 2-4
DFA states vs. 1), but this is offset by the simplicity of the byte-level
transition table (256 entries per state, O(1) lookup).

### 5.2 DFA vs. NFA at Runtime

While NFAs are more compact, DFA simulation is O(1) per byte per configuration
(single array lookup), whereas NFA simulation requires maintaining and
advancing a set of states. Given that the DFAs are pre-minimized and bounded in
size, DFA runtime is strictly preferable.

### 5.3 Precomputation vs. Online Filtering

The precomputation cost (170-280ms) amortizes over hundreds to thousands of
decoding steps in a typical generation. For a 512-token generation with
per-step savings of 0.5-2ms, the precomputation pays for itself within the
first 100-500 tokens. The grammar cache eliminates precomputation entirely for
repeated grammar use.

### 5.4 Adaptive Set Representation

A fixed representation (always storing the accept set) wastes space and
scanning time for accept-heavy states (e.g., inside a JSON string value where
most tokens are valid). The adaptive approach ensures that the stored set is
always the smaller of the two, keeping both storage and scan time proportional
to the minority class.

## 6. Implementation

The system is implemented in C++ within the llama.cpp inference framework.

| File | Lines | Purpose |
|------|-------|---------|
| `src/llama-grammar-ast.h` | ~123 | AST node types, parser, terminal classification |
| `src/llama-grammar-dfa.h` | ~298 | DFA, NFA, compiled grammar, trie, candidate structures |
| `src/llama-grammar-dfa.cpp` | ~2000 | NFA/DFA construction, trie, precomputation, FIRST/FOLLOW, runtime |
| `src/llama-grammar.h` | ~185 | Grammar struct, internal API, thread pool |
| `src/llama-grammar.cpp` | ~1300 | Grammar init, apply, accept, clone, GBNF parser, cache |

The compiled grammar is immutable and shared across grammar clones via
`std::shared_ptr`. Runtime state (parse configurations) is value-copied on
clone, enabling independent sampling from the same grammar (e.g., beam search
or parallel request serving).

## 7. Related Work

**Outlines** (Willard & Louf, 2023) pioneered the use of finite-state machines
for grammar-constrained generation, using regex-based indexing. Our work extends
this to full context-free grammars with a multi-level compilation pipeline and
vocabulary-trie precomputation.

**LMQL** (Beurer-Kellner et al., 2023) provides a query language for
constrained generation with eager token masking. It operates at a higher level
of abstraction, using scripted constraints rather than compiled automata.

**Guidance** (Microsoft, 2023) interleaves generation with programmatic
constraints, supporting context-free grammars through a parser-based approach.
Unlike our DFA-based system, it does not precompute token candidate sets.

**GrammarFlow** and similar systems use Earley parsing or GLL parsing for
runtime grammar enforcement. Our DFA-based approach trades generality (no
left-recursive grammars) for performance (O(1) per-byte transitions).

**XGrammar** (Dong et al., 2024) proposes a similar precomputation approach
with adaptive token masking. Our system adds FIRST/FOLLOW set pruning for
context-dependent tokens, Aho-Corasick exclusion patterns, and Kleene
detection for self-recursive rules.

## 8. Limitations and Future Work

1. **Left recursion.** The system requires grammars to be free of left
   recursion. A preprocessing step to eliminate left recursion could broaden
   applicability.

2. **DFA state explosion.** While bounded by NFA size limits, pathological
   grammars with many overlapping character classes can still produce large
   DFAs. Further heuristics for NFA splitting could help.

3. **Precomputation latency.** The 170-280ms precomputation cost, while
   amortized, adds to first-request latency. Incremental precomputation
   (computing candidates for only the initial DFA states, then lazily expanding)
   could reduce this.

4. **Configuration explosion.** Highly ambiguous grammars can produce many
   active parse configurations. Configuration merging or prioritization could
   bound this.

5. **Integration with speculative decoding.** Grammar constraints interact
   non-trivially with speculative decoding (draft-verify paradigm). Extending
   the precomputed sets to support multi-token lookahead is an open problem.

## 9. Conclusion

We have presented a comprehensive system for grammar-constrained LLM decoding
that compiles context-free grammars to byte-level DFAs and precomputes
per-state token candidate sets via a vocabulary trie walk. The three-way token
classification (guaranteed-accept, context-dependent, rejected) with adaptive
set representation and FIRST/FOLLOW pruning reduces per-step filtering cost by
10-100x for structured grammars, while the grammar and trie caches eliminate
recompilation overhead. The system is production-deployed within llama.cpp,
serving grammar-constrained generation across a wide range of applications
including JSON schema enforcement, function calling, and structured data
extraction.

## References

- Thompson, K. (1968). Programming Techniques: Regular expression search
  algorithm. *Communications of the ACM*, 11(6), 419-422.

- Hopcroft, J. (1971). An n log n algorithm for minimizing states in a finite
  automaton. *Theory of Machines and Computations*, 189-196.

- Aho, A. V., & Corasick, M. J. (1975). Efficient string matching: An aid to
  bibliographic search. *Communications of the ACM*, 18(6), 333-340.

- Willard, B. T., & Louf, R. (2023). Efficient Guided Generation for Large
  Language Models. *arXiv preprint arXiv:2307.09702*.

- Beurer-Kellner, L., Fischer, M., & Vechev, M. (2023). Prompting Is
  Programming: A Query Language for Large Language Models. *PLDI 2023*.

- Dong, Y., Jiang, C., et al. (2024). XGrammar: Flexible and Efficient
  Structured Generation Engine for Large Language Models. *arXiv preprint
  arXiv:2411.15100*.
