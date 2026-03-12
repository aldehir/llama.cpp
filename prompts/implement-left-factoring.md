# Task: Implement Left-Factoring Optimization for Grammar Rules

## Context

The DFA-based grammar engine in `src/llama-grammar-dfa.{h,cpp}` handles rule-call
nondeterminism by maintaining multiple `parse_config`s. When a rule has alternates
that start with the same rule call (or compatible terminal prefixes), the runtime
creates separate configs for each alternate and prunes dead branches as bytes are
consumed.

For well-structured grammars (JSON, function calls), config count stays small. But
for grammars with heavy rule-call ambiguity, configs can proliferate. A compile-time
left-factoring pass can reduce this by merging common prefixes across alternates.

This implements the optimization mentioned in `docs/grammar-dfa-redesign.md`:
> For pathological grammars with heavy rule-call ambiguity, an optional left-factoring
> pass at compile time can reduce branching further.

## Background: Left Factoring

Given a rule with alternates that share a prefix:
```
root ::= A B x | A B y | A C z
```

Left factoring transforms this to:
```
root ::= A root'
root' ::= B root'' | C z
root'' ::= x | y
```

This reduces the initial config count from 3 to 1 when entering `root`, since all
alternates start with `A`. The branching is deferred to after `A` is matched.

## What to Implement

### 1. Prefix Analysis

Add a function that analyzes alternates of a compiled rule to find common prefixes:

```cpp
// In llama-grammar-dfa.h or a new file

// Represents a prefix shared across multiple alternates
struct shared_prefix {
    std::vector<compiled_segment> prefix;    // common prefix segments
    std::vector<uint32_t> alternate_indices; // which alternates share this prefix
};

// Find shared prefixes across alternates of a rule
std::vector<shared_prefix> find_shared_prefixes(const compiled_rule & rule);
```

Two alternates share a prefix if their leading segments are identical:
- Two `RULE_CALL` segments match if they call the same rule
- Two `DFA_MATCH` segments match if they use the same DFA (by ID)

### 2. Rule Splitting

When shared prefixes are found, split the rule into factored form:

```cpp
// Transform a rule by factoring out common prefixes
// May add new rules to the compiled grammar
void left_factor_rule(compiled_grammar & grammar, uint32_t rule_id);
```

Algorithm:
1. Group alternates by their first segment
2. For each group with >1 alternate:
   a. Find the longest common prefix (sequence of identical segments)
   b. Create a new "suffix" rule containing the remainders
   c. Replace the group with a single alternate: `prefix + RULE_CALL(suffix_rule)`
3. Recurse on newly created suffix rules (they may also be factorable)

### 3. Integration Point

Add left factoring as an optional pass in `compiled_grammar::compile()`:

```cpp
compiled_grammar compiled_grammar::compile(
        const std::vector<std::vector<llama_grammar_element>> & parsed_rules,
        uint32_t start_rule_index) {
    // ... existing compilation code ...

    // Optional: left-factor rules to reduce runtime config proliferation
    cg.left_factor();

    return cg;
}
```

### 4. Terminal Prefix Merging (Advanced)

Beyond rule-call factoring, also merge alternates whose terminal DFA segments accept
overlapping byte sets. For example:
```
root ::= "hello" rest1 | "help" rest2
```

After compilation, both alternates start with a DFA_MATCH segment. If we detect that
two DFA_MATCH segments share a common byte prefix (both start with `h`, `e`, `l`),
we can:
1. Build a merged DFA for the shared prefix ("hel")
2. Create a suffix rule with the divergent parts ("lo" rest1 | "p" rest2)
3. Replace the two alternates with: merged_dfa + RULE_CALL(suffix_rule)

This is more complex because it requires DFA analysis (finding the longest common
prefix of two DFAs), but it eliminates the most common source of config proliferation
in practical grammars.

**DFA prefix extraction algorithm:**
```
Given DFA_A and DFA_B:
1. Simulate both DFAs in lockstep from their start states
2. At each step, compute the set of bytes where both DFAs have non-dead transitions
3. If the sets are identical, this byte position is part of the shared prefix
4. Stop when the sets diverge or either DFA reaches an accept state
5. The shared prefix is the DFA traversed up to the divergence point
```

## Files to Modify

- `src/llama-grammar-dfa.h` — Add `left_factor()` method to `compiled_grammar`,
  helper structs/functions
- `src/llama-grammar-dfa.cpp` — Implement left factoring algorithm
- `tests/test-grammar-dfa.cpp` — Add tests for left factoring

## Test Cases

### Basic left factoring
```
root ::= "a" "b" "x" | "a" "b" "y"
→ After factoring: root starts with one config instead of two
→ Verify: config count at start == 1
→ Verify: both "abx" and "aby" are still accepted
```

### Rule-call factoring
```
root ::= shared "x" | shared "y"
shared ::= [a-z]+
→ After factoring: one config at start
→ Verify: "hellox" and "helloy" accepted, "helloz" rejected
```

### No factoring needed
```
root ::= "a" | "b"
→ Alternates don't share a prefix, no factoring applied
→ Config count unchanged
```

### Deep factoring
```
root ::= A B C "x" | A B C "y" | A B D "z"
→ Factor out A B, then factor the suffix to get A B (C root'' | D "z")
→ Verify all three strings accepted
```

### Verify config count reduction
```cpp
// Before left factoring
auto * g1 = build_grammar("root ::= \"abc\" \"x\" | \"abc\" \"y\" | \"abd\" \"z\"");
auto configs1 = g1->configs;
// configs1.size() should be 3 without factoring

// After left factoring (enabled)
auto * g2 = build_grammar_with_factoring("root ::= \"abc\" \"x\" | \"abc\" \"y\" | \"abd\" \"z\"");
auto configs2 = g2->configs;
// configs2.size() should be 1 (single config for shared "ab" prefix)
```

## Performance Considerations

- Left factoring adds compile-time cost (one pass over all rules)
- For typical grammars (JSON schemas), most rules have few alternates → minimal overhead
- The benefit is runtime: fewer configs = less work per `accept_byte` and `get_valid_bytes`
- Consider making it opt-in via a parameter if compile time is a concern for very large grammars

## Edge Cases

- Empty alternates (`rule ::= A | ε`) — the empty alternate has no prefix to factor
- Single-alternate rules — nothing to factor
- Rules where ALL alternates are identical — collapse to one alternate
- Circular factoring — a rule that factors into itself. Avoid by not factoring rules
  that consist of a single RULE_CALL to themselves
- DFA merging across different UTF-8 byte lengths — the DFA handles this naturally
  since it operates on raw bytes
