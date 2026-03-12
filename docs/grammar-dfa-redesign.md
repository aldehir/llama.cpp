# Grammar Engine Redesign: Byte-Level DFA + External Stack

## Motivation

The current grammar engine in `llama-grammar.cpp` implements a nondeterministic pushdown
automaton (NPDA) that operates on Unicode code points. While correct, this design has two
intertwined sources of complexity:

1. **Partial UTF-8 handling.** LLM tokens can split multi-byte UTF-8 sequences across token
   boundaries. The current code maintains `llama_partial_utf8` state, and the function
   `llama_grammar_match_partial_char` performs speculative range-overlap checks to determine
   whether an incomplete byte sequence *could* complete to a valid code point for the current
   grammar position. This is the most subtle and bug-prone code in the file.

2. **Stack explosion from terminal nondeterminism.** The NPDA represents nondeterminism by
   maintaining a *set* of stacks (`llama_grammar_stacks`). Every alternate in a rule, every
   character class, and every `RULE_REF` to a rule with multiple alternates can multiply the
   stack count. Terminal matching (characters, ranges, negation) and context-free recursion
   (rule calls/returns) are interleaved in the same stack structure, so nondeterminism from
   either source compounds.

The proposed redesign separates these concerns:

- **Terminal matching** (regular) is compiled into byte-level DFAs at grammar-parse time.
- **Rule recursion** (context-free) is handled by a small external call stack.
- **Partial UTF-8 disappears entirely** — the DFA operates on raw bytes, and multi-byte
  sequences are just intermediate DFA states.

---

## Current Architecture

### Data Structures

```
src/llama-grammar.h
```

| Type | Purpose |
|---|---|
| `llama_grammar_element` | `{type, value}` — one grammar instruction. `value` is a Unicode code point or rule ID |
| `llama_grammar_rule` | `vector<llama_grammar_element>` — a single rule (elements + END, alternates separated by ALT) |
| `llama_grammar_stack` | `vector<const llama_grammar_element *>` — a PDA stack; each entry is a pointer into a rule |
| `llama_grammar_stacks` | `vector<llama_grammar_stack>` — **the set of all possible PDA configurations** |
| `llama_partial_utf8` | `{value, n_remain}` — tracks incomplete UTF-8 sequences across token boundaries |
| `llama_grammar_candidate` | `{index, code_points, partial_utf8}` — a token candidate decoded to code points |
| `llama_grammar` | Top-level: owns rules, stacks, partial_utf8, lazy-trigger state |

### Grammar Element Types (`llama_gretype`)

```
LLAMA_GRETYPE_END            (0)  — end of rule definition
LLAMA_GRETYPE_ALT            (1)  — start of alternate
LLAMA_GRETYPE_RULE_REF       (2)  — non-terminal reference
LLAMA_GRETYPE_CHAR           (3)  — terminal: exact code point
LLAMA_GRETYPE_CHAR_NOT       (4)  — terminal: negated char/range
LLAMA_GRETYPE_CHAR_RNG_UPPER (5)  — range upper bound modifier
LLAMA_GRETYPE_CHAR_ALT       (6)  — alternate char modifier
LLAMA_GRETYPE_CHAR_ANY       (7)  — wildcard (.)
```

### Key Functions

**Grammar parsing** (`llama_grammar_parser`):
- `parse()` — recursive descent parser for GBNF text
- `parse_rule()` → `parse_alternates()` → `parse_sequence()` — builds `llama_grammar_rule` vectors
- Repetition (`*`, `+`, `?`, `{m,n}`) is desugared into recursive helper rules at parse time
- `parse_char()` handles Unicode escapes (`\xNN`, `\uNNNN`, `\UNNNNNNNN`) and raw UTF-8

**PDA core** (the runtime):
- `llama_grammar_advance_stack()` (line 696) — recursively resolves `RULE_REF` elements until every resulting stack has a terminal (CHAR/CHAR_NOT/CHAR_ANY) on top. This is the main source of stack multiplication.
- `llama_grammar_match_char()` (line 616) — tests whether a code point matches the terminal at the current stack position (handles ranges, alternates, negation, wildcard).
- `llama_grammar_match_partial_char()` (line 646) — speculative matching for incomplete UTF-8: computes the range `[low, high]` of code points the partial sequence could complete to, and checks for overlap with the grammar position.
- `llama_grammar_accept()` (line 834) — for each stack, if the top matches the character, advance past it and call `advance_stack` to resolve the next position. Replaces the entire stack set.

**Sampling integration** (`llama-sampling.cpp`, lines 1540–1698):
- `llama_grammar_apply_impl()` — for each token candidate, decodes its bytes to code points (using `decode_utf8` with `partial_utf8` state), then calls `llama_grammar_reject_candidates()` to filter. Rejected tokens get logit = -INFINITY.
- `llama_grammar_accept_impl()` — after a token is selected, feeds its code points into `llama_grammar_accept()` one at a time.
- Sampler interface: `apply` (constrain logits), `accept` (advance state), `reset`, `clone`, `free`.

**Left recursion detection** (`llama_grammar_detect_left_recursion`, line 773):
- DFS with "in progress" tracking. Also detects rules that may produce epsilon (empty string) to correctly handle indirect left recursion through nullable prefixes.
- Left-recursive grammars are rejected at init time.

### Pain Points in Current Design

1. **`llama_grammar_match_partial_char` (lines 646–692):** Computes `low = partial_value << (n_remain * 6)` and `high = low | ((1 << (n_remain * 6)) - 1)`, then adjusts for overlong encoding. Must correctly handle the interaction of partial code points with ranges, negation, and alternates. Any bug here silently allows or blocks tokens.

2. **Two `decode_utf8` overloads (lines 18, 33):** The string overload must resume from `partial_utf8` state, handle invalid continuation bytes, and track a new partial state at the end. Used on every token candidate evaluation.

3. **Stack set explosion:** `llama_grammar_advance_stack` is recursive and branches on every alternate of every `RULE_REF`. The dedup check at line 701/741 is `std::find` (linear scan). For grammars with many alternates (common in JSON schema output), stacks can proliferate.

4. **`llama_grammar_clone_impl` (line 1097):** Must remap all stack pointers from old rule storage to new rule storage — an O(stacks * stack_depth * total_rule_elements) operation.

---

## Proposed Architecture: Byte-Level DFA + External Stack

### Overview

```
┌──────────────────────────────────────────────────┐
│                  Grammar Compiler                 │
│                                                   │
│  GBNF text ──→ parse rules ──→ for each rule:    │
│                                  - extract        │
│                                    terminal body  │
│                                  - expand Unicode │
│                                    to UTF-8 bytes │
│                                  - NFA ──→ DFA    │
│                                  - minimize DFA   │
│                                                   │
│  Output: rule_id → (DFA, call_edges, return_info) │
└──────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│                  Runtime Engine                    │
│                                                   │
│  State = (current_rule, dfa_state, call_stack)    │
│                                                   │
│  For each byte:                                   │
│    1. Transition DFA on byte                      │
│    2. If DFA reaches a CALL edge:                 │
│         push (rule, return_dfa_state) onto stack  │
│         enter called rule's DFA at start state    │
│    3. If DFA reaches accept state:                │
│         pop stack, resume caller's DFA            │
│    4. If DFA reaches dead state:                  │
│         reject                                    │
│                                                   │
│  valid_bytes = set of bytes with non-dead         │
│                transitions from current state     │
└──────────────────────────────────────────────────┘
```

### Phase 1: Grammar Compilation

The grammar compiler runs once at `llama_grammar_init_impl` time. It replaces the current
approach of storing raw `llama_grammar_element` arrays and interpreting them at runtime.

#### Step 1.1: Parse GBNF (reuse existing parser)

The existing `llama_grammar_parser` is retained mostly as-is. It produces the same
`llama_grammar_rules` (vector of vector of `llama_grammar_element`). The desugaring of
`*`, `+`, `?`, `{m,n}` into recursive helper rules also stays.

#### Step 1.2: Split each rule into segments

For each rule, identify **segments** separated by `RULE_REF` elements. A segment is a
sequence of terminal elements (CHAR, CHAR_NOT, CHAR_RNG_UPPER, CHAR_ALT, CHAR_ANY)
that can be compiled into a single DFA.

Example rule:
```
json-string ::= "\"" [^"\\]* "\""
```

After parsing, the element sequence is:
```
CHAR('"')  RULE_REF(char_rule)  CHAR('"')  END
```

This has two terminal segments (`"` and `"`) separated by a rule reference. Each segment
compiles to a trivial DFA.

For a rule that is purely terminal with alternates:
```
greeting ::= "hello" | "hi"
```

Element sequence:
```
CHAR('h') CHAR('e') CHAR('l') CHAR('l') CHAR('o') ALT CHAR('h') CHAR('i') END
```

This is a single terminal segment with alternates — compiles to one DFA.

#### Step 1.3: Expand code points to UTF-8 byte sequences

Each terminal element is expanded from a code point match to the corresponding UTF-8 byte
pattern. This is a mechanical transformation:

**Single code point** (CHAR):
- ASCII (U+0000–U+007F): 1-byte sequence → single byte match
- U+0080–U+07FF: 2-byte sequence → match lead byte, then continuation byte
- U+0800–U+FFFF: 3-byte → match 3 specific bytes
- U+10000–U+10FFFF: 4-byte → match 4 specific bytes

**Code point range** (CHAR + CHAR_RNG_UPPER, e.g. `[α-ω]`):
Split the range at UTF-8 length boundaries and lead-byte boundaries. For U+03B1–U+03C9:
```
\xCE [\xB1-\xBF]   |   \xCF [\x80-\x89]
```

Algorithm (well-known, used by RE2 and Rust regex):
1. Split range at boundaries where the UTF-8 encoding length changes (U+0080, U+0800, U+10000)
2. Within each length class, split at lead-byte boundaries
3. Within each lead-byte group, emit a sequence: fixed lead byte + continuation byte ranges

**Negation** (CHAR_NOT):
Compute at the NFA level — build the NFA for the positive match, then complement after
determinization (DFA complement is trivial: flip accept/reject states, intersect with
valid-UTF-8 DFA).

**Wildcard** (CHAR_ANY):
Expand to the valid-UTF-8 automaton:
```
[\x00-\x7F]
| [\xC2-\xDF] [\x80-\xBF]
| \xE0 [\xA0-\xBF] [\x80-\xBF]
| [\xE1-\xEC] [\x80-\xBF] [\x80-\xBF]
| \xED [\x80-\x9F] [\x80-\xBF]
| [\xEE-\xEF] [\x80-\xBF] [\x80-\xBF]
| \xF0 [\x90-\xBF] [\x80-\xBF] [\x80-\xBF]
| [\xF1-\xF3] [\x80-\xBF] [\x80-\xBF] [\x80-\xBF]
| \xF4 [\x80-\x8F] [\x80-\xBF] [\x80-\xBF]
```

This valid-UTF-8 DFA is a fixed ~10-state automaton and can be shared across all rules.

#### Step 1.4: Build NFA from byte-level patterns

For each terminal segment, construct a Thompson NFA over bytes (0x00–0xFF alphabet).
Standard NFA construction:
- Concatenation of byte matches → chain states
- Alternation → epsilon-split to each alternate
- The existing elements map directly:
  - `CHAR(cp)` → concatenation of the UTF-8 byte matches for that code point
  - `CHAR + CHAR_ALT + ...` (character class) → alternation of all member byte patterns
  - `CHAR_NOT + ...` (negated class) → built as positive, then complemented post-determinization
  - `CHAR_RNG_UPPER` → range of byte patterns (from Step 1.3)

#### Step 1.5: Determinize and minimize

Standard NFA → DFA conversion (subset construction), then Hopcroft minimization.

For negated classes (`CHAR_NOT`): determinize the positive DFA first, then complement
(flip accept states), then intersect with the valid-UTF-8 DFA to exclude invalid byte
sequences from the "any character except X" semantics.

The result for each terminal segment is a compact DFA stored as a transition table:
```cpp
struct byte_dfa {
    // transition[state][byte] → next_state (0 = dead/reject state)
    std::vector<std::array<uint16_t, 256>> transition;
    std::vector<bool> accept;  // accept[state] → is this an accept state?
    uint16_t start_state;
};
```

For space efficiency, states with identical transitions for most bytes can use a compressed
representation (e.g., default transition + exception list), but the flat table is the
simplest starting point and gives O(1) byte transitions.

#### Step 1.6: Assemble per-rule structure

Each compiled rule becomes:
```cpp
struct compiled_rule {
    // The sequence of operations to execute for this rule.
    // Each entry is either a DFA segment to match or a rule call.
    struct segment {
        enum type { DFA_MATCH, RULE_CALL } type;
        union {
            uint32_t dfa_id;   // index into shared DFA table
            uint32_t rule_id;  // called rule ID
        };
    };
    std::vector<std::vector<segment>> alternates; // one vector<segment> per alternate
};
```

For rules with alternates that share prefixes, the DFA compilation in Step 1.5 already
handles the merging — the alternates' terminal prefixes are merged into one DFA.

### Phase 2: Runtime Engine

The runtime replaces the current stack-set-based NPDA with a much simpler execution model.

#### State Representation

```cpp
struct grammar_state {
    uint32_t rule_id;        // currently executing rule
    uint8_t  alternate_idx;  // which alternate of the rule
    uint8_t  segment_idx;    // which segment within the alternate
    uint16_t dfa_state;      // current state within the active DFA segment
};

struct grammar_call_frame {
    uint32_t rule_id;
    uint8_t  alternate_idx;
    uint8_t  segment_idx;     // segment to resume AFTER the call returns
    uint16_t dfa_state;       // DFA state to resume (for the next segment)
};

struct llama_grammar {
    const llama_vocab * vocab;

    // Compiled grammar (shared, immutable after init)
    std::vector<byte_dfa>      dfas;
    std::vector<compiled_rule> rules;
    uint32_t                   start_rule;

    // Runtime state (mutable during decoding)
    // Multiple states needed ONLY for rule-call ambiguity (see below)
    struct parse_config {
        grammar_state                    current;
        std::vector<grammar_call_frame>  call_stack;
    };
    std::vector<parse_config> configs;  // set of active configurations

    // Lazy trigger state (unchanged from current design)
    bool                     lazy             = false;
    bool                     awaiting_trigger = false;
    std::string              trigger_buffer;
    std::vector<llama_token> trigger_tokens;
    std::vector<llama_grammar_trigger_pattern> trigger_patterns;

    // NOTE: no more partial_utf8 — the DFA handles byte-level state
};
```

#### Handling Nondeterminism

**Terminal nondeterminism** (character alternates, Unicode ranges, negation): Fully absorbed
by the DFA. A DFA has exactly one active state for any input — no branching.

**Rule-call nondeterminism** (a rule position can call one of several rules): This is the
only remaining source of nondeterminism. It occurs when a rule has the form:
```
root ::= a_rule x | b_rule y
```
and both `a_rule` and `b_rule` can start with the same byte.

This is handled by maintaining multiple `parse_config`s — but these only branch at rule
call points, not at every character. For well-structured grammars (JSON, function calls),
the config count stays very small.

For pathological grammars with heavy rule-call ambiguity, an optional left-factoring
pass at compile time can reduce branching further.

#### Core Operations

**`grammar_accept_byte(byte)`:**
```
for each active config:
    transition DFA on byte
    if DFA reaches dead state:
        remove this config (dead branch)
    if DFA reaches accept state AND at end of segment:
        advance to next segment in the alternate
        if next segment is RULE_CALL:
            push current position onto call_stack
            enter called rule (may branch if called rule has alternates)
        if no more segments (end of rule):
            if call_stack empty: this config represents a complete parse
            else: pop call_stack, resume caller
    if DFA reaches accept state but more bytes could extend the match:
        keep in current DFA state (greedy/continuation)
```

**`grammar_get_valid_bytes()` → `std::bitset<256>`:**
```
result = empty bitset
for each active config:
    for each byte b in 0..255:
        if DFA transition from current state on b != dead:
            result.set(b)
return result
```

This replaces the current `llama_grammar_reject_candidates` flow. Instead of decoding each
token candidate to code points and testing against the PDA, we can:

1. Compute the 256-bit valid-byte mask once per sampling step
2. For each candidate token, check whether its first byte is in the mask
3. For multi-byte tokens, simulate the DFA forward through all bytes to check full validity

**`grammar_apply(cur_p)`:**
```
for each candidate token in cur_p:
    if is_eog(token):
        allow iff any config has empty call_stack and DFA in accept state
    else:
        bytes = token_to_bytes(token)  // raw bytes, no UTF-8 decode needed
        simulate DFA+stack forward through bytes
        if simulation reaches dead state: set logit = -INFINITY
```

**`grammar_accept_token(token)`:**
```
bytes = token_to_bytes(token)
for each byte in bytes:
    grammar_accept_byte(byte)
```

### Phase 3: UTF-8 Byte Pattern Expansion (Detail)

This is the most mechanically complex part of the compiler. The algorithm for expanding a
Unicode code point range `[lo, hi]` to UTF-8 byte-level NFA patterns is:

```
function codepoint_range_to_byte_nfa(lo: uint32, hi: uint32) -> NFA:
    if lo > hi: return empty NFA

    # Split at UTF-8 length boundaries
    ranges = split_at_boundaries(lo, hi, [0x80, 0x800, 0x10000])

    nfa = empty NFA
    for (sub_lo, sub_hi) in ranges:
        nfa = nfa | single_length_range_to_nfa(sub_lo, sub_hi)
    return nfa

function single_length_range_to_nfa(lo: uint32, hi: uint32) -> NFA:
    lo_bytes = encode_utf8(lo)  # e.g., [0xCE, 0xB1] for U+03B1
    hi_bytes = encode_utf8(hi)  # e.g., [0xCF, 0x89] for U+03C9
    n = len(lo_bytes)           # same length since we split at boundaries

    return byte_range_nfa(lo_bytes, hi_bytes, 0)

function byte_range_nfa(lo: byte[], hi: byte[], pos: int) -> NFA:
    if pos == len(lo) - 1:
        return byte_match(lo[pos]..hi[pos])

    if lo[pos] == hi[pos]:
        # Same prefix byte — emit it, recurse on suffix
        return concat(byte_match(lo[pos]),
                      byte_range_nfa(lo, hi, pos + 1))

    # Different prefix bytes — split into three parts:
    # 1. lo[pos] with restricted continuation
    # 2. (lo[pos]+1)..(hi[pos]-1) with full continuation range
    # 3. hi[pos] with restricted continuation
    result = empty NFA

    # Part 1: lo[pos] + (lo[pos+1:]..max continuation)
    max_suffix = [0xBF] * (len(lo) - pos - 1)
    result = result | concat(byte_match(lo[pos]),
                             byte_range_nfa(lo, lo[:pos+1] + max_suffix, pos + 1))

    # Part 2: middle prefix bytes with full continuation range
    if lo[pos] + 1 < hi[pos]:
        min_suffix = [0x80] * (len(lo) - pos - 1)
        full_max   = [0xBF] * (len(lo) - pos - 1)
        result = result | concat(byte_match((lo[pos]+1)..(hi[pos]-1)),
                                 full_continuation_nfa(len(lo) - pos - 1))

    # Part 3: hi[pos] + (min continuation..hi[pos+1:])
    min_suffix = [0x80] * (len(lo) - pos - 1)
    result = result | concat(byte_match(hi[pos]),
                             byte_range_nfa(hi[:pos+1] + min_suffix, hi, pos + 1))

    return result
```

This produces an NFA with O(n * byte_length) states per range, where n is the number of
lead-byte splits. After determinization this collapses into a compact DFA.

### Phase 4: DFA Complement for Negation

For `[^X]` (CHAR_NOT), the procedure is:

1. Build the DFA for the positive match `[X]` using the above byte expansion
2. Build (or reuse) the valid-UTF-8 DFA
3. Compute the product automaton: `valid_utf8 AND NOT positive_match`
   - States = pairs `(utf8_state, match_state)`
   - Accept = `utf8_accept AND NOT match_accept`
4. Minimize the result

The valid-UTF-8 DFA is a fixed ~10-state automaton that only needs to be built once:

```
State 0 (start):
    0x00-0x7F → accept (State 0, self-loop for single-byte)
    0xC2-0xDF → State 1 (expect 1 continuation)
    0xE0      → State 2 (expect 2 cont, first restricted)
    0xE1-0xEC → State 3 (expect 2 cont)
    0xED      → State 4 (expect 2 cont, first restricted for surrogates)
    0xEE-0xEF → State 3
    0xF0      → State 5 (expect 3 cont, first restricted)
    0xF1-0xF3 → State 6 (expect 3 cont)
    0xF4      → State 7 (expect 3 cont, first restricted)
    else      → dead

State 1: 0x80-0xBF → accept (State 0)
State 2: 0xA0-0xBF → State 8
State 3: 0x80-0xBF → State 8
State 4: 0x80-0x9F → State 8
State 5: 0x90-0xBF → State 9
State 6: 0x80-0xBF → State 9
State 7: 0x80-0x8F → State 9
State 8: 0x80-0xBF → accept (State 0)   # 3-byte complete
State 9: 0x80-0xBF → State 8            # 4-byte: one more cont needed
```

---

## What Gets Removed

| Current Code | Replacement |
|---|---|
| `llama_partial_utf8` struct | Gone — DFA tracks byte-level state |
| `decode_utf8` (string overload, lines 33–91) | Gone — tokens fed as raw bytes |
| `decode_utf8` (char overload, lines 18–31) | Retained only in grammar parser for reading GBNF source |
| `llama_grammar_match_partial_char` (lines 646–692) | Gone entirely |
| `llama_grammar_match_char` (lines 616–641) | Replaced by DFA transition lookup |
| `llama_grammar_advance_stack` (lines 696–752) | Replaced by explicit call stack push/pop |
| `llama_grammar_stacks` (set of stacks) | Replaced by `vector<parse_config>` — much smaller |
| `llama_grammar_reject_candidates` (lines 754–771) | Replaced by DFA simulation per token |
| `llama_grammar_reject_candidates_for_stack` (lines 859–912) | Replaced by DFA simulation |
| `llama_grammar_candidate` struct | Gone — no code-point-level candidates |
| `llama_grammar_clone_impl` pointer remapping (lines 1111–1121) | Gone — DFAs are value types, configs are indices |

## What Gets Added

| New Component | Purpose |
|---|---|
| `byte_dfa` struct | Transition table + accept states for one terminal segment |
| `compiled_rule` struct | Per-rule: sequence of (DFA segment, rule call) entries |
| `codepoint_range_to_byte_nfa()` | Expands Unicode ranges to UTF-8 byte NFAs |
| `nfa_to_dfa()` | Subset construction |
| `dfa_minimize()` | Hopcroft minimization |
| `dfa_complement()` | Accept-state flip for negated classes |
| `valid_utf8_dfa()` | Fixed DFA accepting valid UTF-8 sequences |
| `grammar_get_valid_bytes()` | Returns 256-bit mask of valid next bytes |

## What Stays Unchanged

| Component | Why |
|---|---|
| `llama_grammar_parser` | GBNF parsing still produces the same rule representation; it's the input to the new compiler |
| `parse_char`, `parse_hex`, `parse_name`, etc. | Grammar source parsing is unaffected |
| Repetition desugaring (`handle_repetitions`) | `*`, `+`, `?`, `{m,n}` still become recursive helper rules |
| Left recursion detection | Still needed — left-recursive grammars would cause infinite call stack growth |
| Lazy trigger system | Orthogonal to the PDA/DFA change — operates at the token/string level |
| Sampler interface (`apply`, `accept`, `reset`, `clone`, `free`) | Same API contract, different internals |
| Public API (`llama_sampler_init_grammar`, etc.) | No changes to any public-facing function signatures |

---

## Migration Strategy

### Step 1: Add byte-level DFA infrastructure

Add new files (or a new section in `llama-grammar.cpp`):
- `struct byte_dfa` with transition table
- NFA construction from byte ranges
- Subset construction (NFA → DFA)
- Hopcroft minimization
- UTF-8 code point range → byte NFA expansion
- Valid-UTF-8 DFA (hardcoded or generated)
- DFA complement operation

This can be developed and unit-tested in isolation. Key test cases:
- ASCII range `[a-z]` → 1-state DFA
- Multi-byte range `[α-ω]` → small DFA with correct byte transitions
- Negated ASCII `[^a]` → accepts all valid UTF-8 except `a`
- Negated multi-byte `[^α]` → accepts all valid UTF-8 except 0xCE 0xB1
- Wildcard `.` → accepts any single valid UTF-8 character
- Mixed class `[a-zα-ω]` → union of ASCII and 2-byte ranges

### Step 2: Add compiled rule structure and grammar compiler

- Walk parsed rules, split into segments at RULE_REF boundaries
- Compile terminal segments to DFAs using Step 1 infrastructure
- Build `compiled_rule` structures

Test cases: compile known grammars and verify the DFA segment structure.

### Step 3: Implement runtime engine alongside existing code

Add new functions:
- `grammar_accept_byte()`
- `grammar_get_valid_bytes()` → `bitset<256>`
- `grammar_apply_dfa()` (new implementation of `llama_grammar_apply_impl`)
- `grammar_accept_token_dfa()` (new implementation of `llama_grammar_accept_impl`)

Run both old and new engines in parallel on the test suite. Compare outputs for correctness.

### Step 4: Switch over and remove old code

Once the new engine passes all tests:
- Replace `llama_grammar_apply_impl` and `llama_grammar_accept_impl`
- Remove `llama_partial_utf8`, `decode_utf8` (string overload), `llama_grammar_match_partial_char`, `llama_grammar_match_char`, `llama_grammar_advance_stack`, `llama_grammar_reject_candidates`, etc.
- Update `llama_grammar` struct to new state representation
- Update clone/reset operations (now simpler — DFAs are shared, state is just indices + small stack)

### Step 5: Update tests

- `test-grammar-parser.cpp` — parser tests stay, add DFA compilation tests
- `test-grammar-integration.cpp` — string acceptance tests run against new engine
- `test-llama-grammar.cpp` — rewrite stack inspection tests to use new state model
- Add new byte-level DFA unit tests (range expansion, complement, valid-UTF-8)

---

## Performance Expectations

**Grammar compilation** (one-time cost at init): Slower than current approach due to NFA
construction, determinization, and minimization. For typical grammars (JSON schemas), this
should be milliseconds. The result can be cached.

**Token evaluation** (per-token, hot path):
- Current: decode UTF-8 (branch-heavy), match code points against stack set (linear in stack count), speculative partial match
- New: for each candidate token, feed raw bytes through DFA (one table lookup per byte, no branching), check call stack only at segment boundaries
- Expected: **faster** for typical grammars due to O(1) byte transitions and elimination of UTF-8 decode/partial-match overhead

**Memory**: DFA transition tables are 256 entries × 2 bytes × N states. For typical grammars
with small DFAs (5–20 states per segment), this is a few KB total. Much less than the
unbounded stack-set storage of the current design.

**Stack count**: Only branches on rule-call ambiguity, not on terminal ambiguity. For JSON
grammars (the dominant use case), rule-call ambiguity is minimal, so the config count stays
close to 1.

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| DFA state explosion for complex Unicode negation patterns | Monitor DFA sizes; add a state count limit with fallback to current engine |
| Subtle correctness bugs in UTF-8 range expansion | Exhaustive testing: for every code point in a range, verify the DFA accepts exactly its UTF-8 encoding |
| Rule-call ambiguity causing config proliferation | Left-factoring optimization pass; worst case is same as current stack proliferation but limited to rule-call sites |
| Breaking changes to test infrastructure | Parallel execution (Step 3) validates before cutover |
| Performance regression for grammars with many small DFA segments | Segment merging: combine adjacent small segments where possible |

---

## Existing Test Matrix

The following test files must continue to pass:

| File | Scope |
|---|---|
| `test-grammar-parser.cpp` | GBNF parsing → internal rule representation |
| `test-grammar-integration.cpp` | End-to-end: grammar + strings → accept/reject. Covers JSON schema conversion, Unicode, quantifiers, error detection, left recursion |
| `test-llama-grammar.cpp` | Low-level engine: stack states, candidate rejection |
| `test-json-schema-to-grammar.cpp` | JSON schema → GBNF output |
| `test-gbnf-validator.cpp` | GBNF validation |
| `test-grammar-llguidance.cpp` | llguidance integration (separate engine, unaffected) |
