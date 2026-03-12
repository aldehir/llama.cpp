# Task: Remove Legacy NPDA Grammar Engine

## Context

The grammar engine has been redesigned from a nondeterministic pushdown automaton (NPDA)
operating on Unicode code points to a byte-level DFA engine with an external call stack.
The new DFA engine is fully implemented and integrated in `src/llama-grammar-dfa.{h,cpp}`.

Currently, both engines run in parallel: the DFA engine handles `apply_impl` and
`accept_str`, while the legacy NPDA engine is kept in sync for backward compatibility
with tests that inspect internal stack state.

This task completes **Step 4** of the migration plan in `docs/grammar-dfa-redesign.md`:
remove the legacy code paths now that the new engine passes all tests.

## What to Remove

### From `src/llama-grammar.h`

1. **`llama_partial_utf8` struct** — DFA tracks byte-level state; no partial UTF-8 needed.
2. **`llama_grammar_candidate` struct** — No code-point-level candidates in DFA engine.
3. **Type aliases that are legacy-only:**
   - `llama_grammar_stack` (`vector<const llama_grammar_element *>`)
   - `llama_grammar_stacks` (`vector<llama_grammar_stack>`)
   - `llama_grammar_candidates` (`vector<llama_grammar_candidate>`)
4. **Legacy accessor functions:**
   - `llama_grammar_get_rules()` — was marked `TODO: remove, needed for tests atm`
   - `llama_grammar_get_stacks()`
5. **`llama_grammar_accept()` (code-point-based)** — replaced by `compiled_grammar::accept_byte()`
6. **`llama_grammar_reject_candidates_for_stack()`** — replaced by DFA simulation
7. **From `llama_grammar` struct:**
   - `rules` field (the raw `llama_grammar_rules`)
   - `stacks` field (the `llama_grammar_stacks`)
   - `partial_utf8` field

### From `src/llama-grammar.cpp`

1. **`decode_utf8` (string overload, ~lines 33–95)** — tokens are fed as raw bytes now.
   Keep the `const char *` overload — it's used by the GBNF parser for reading source.
2. **`llama_grammar_match_char()` (~lines 616–641)** — replaced by DFA transition lookup.
3. **`llama_grammar_match_partial_char()` (~lines 646–692)** — gone entirely.
4. **`llama_grammar_advance_stack()` (~lines 696–752)** — replaced by `advance_to_terminal()`.
5. **`llama_grammar_reject_candidates()` (~lines 754–771)** — replaced by DFA simulation.
6. **`llama_grammar_reject_candidates_for_stack()` (~lines 859–912)** — replaced by DFA simulation.
7. **`llama_grammar_accept()` (code-point)** — replaced by byte-level acceptance.
8. **`llama_grammar_get_rules()` / `llama_grammar_get_stacks()`** — remove accessors.
9. **Legacy fallback paths** in `llama_grammar_apply_impl`, `llama_grammar_accept_impl`,
   and `llama_grammar_accept_str` — remove the `if (!grammar.compiled)` branches and
   the "keep legacy in sync" code in `accept_str`.
10. **Stack pointer remapping** in `llama_grammar_clone_impl` (~lines 1111–1121) — DFA
    configs are value types (indices), no pointer remapping needed.

### From `src/llama-grammar.cpp` init functions

1. Remove the legacy stack initialization code in both `llama_grammar_init_impl` overloads:
   - The `llama_grammar_advance_stack` loop that builds initial `stacks`
   - The `llama_grammar_detect_left_recursion` call (still needed — move it before DFA
     compilation so it rejects left-recursive grammars before the DFA compiler sees them)
2. Simplify the `llama_grammar` struct construction — no more `stacks`, `partial_utf8`,
   or `rules` fields.

## What to Update

### `llama_grammar_clone_impl`
Simplify to just copy the `compiled` shared_ptr and the `configs` vector. No pointer
remapping needed.

### `llama_grammar_apply_impl`
Remove the legacy code path (the `else` branch after `if (grammar.compiled)`). The
compiled grammar is always available now.

### `llama_grammar_accept_str`
Remove the legacy sync code. Only feed bytes through the DFA engine.

### `llama_grammar_accept_impl`
Remove the legacy stack-empty check for EOG tokens. Use only `any_config_complete()`.

## Tests to Update

### `tests/test-grammar-integration.cpp`
The `match_string` function currently uses `llama_grammar_get_stacks()` and
`llama_grammar_accept()` (code-point API). Rewrite to use the DFA engine:
```cpp
static bool match_string(const std::string & input, llama_grammar * grammar) {
    auto configs = grammar->configs;  // copy
    for (uint8_t b : input) {
        grammar->compiled->accept_byte(configs, b);
        if (configs.empty()) return false;
    }
    return grammar->compiled->any_config_complete(configs);
}
```

### `tests/test-llama-grammar.cpp`
This test inspects `llama_grammar_get_stacks()` and calls
`llama_grammar_reject_candidates_for_stack()`. It needs a full rewrite to use DFA
state inspection instead. Key tests to preserve:
- Grammar initialization succeeds for valid grammars
- Correct bytes are accepted/rejected for simple grammars
- Use `get_valid_bytes()` instead of stack inspection

### `tests/test-grammar-parser.cpp`
Parser tests don't use runtime state — they should be unaffected. Verify they still
compile after removing the legacy type aliases (they may reference `llama_grammar_rule`
which should be kept since it's the parser's output format).

## Validation

After making changes:
1. Build: `make -j$(nproc) llama test-grammar-parser test-grammar-integration test-llama-grammar test-grammar-dfa`
2. Run all four test binaries and verify they pass
3. The `test-json-schema-to-grammar` C++ portion should also still pass

## Notes

- Keep `llama_grammar_element`, `llama_gretype`, and `llama_grammar_rule` — these are
  the parser's output format and the compiler's input.
- Keep `llama_grammar_parser` entirely — it's the GBNF parser, orthogonal to the engine.
- Keep `llama_grammar_detect_left_recursion` — it prevents infinite call stack growth
  in the DFA runtime.
- Keep the lazy trigger system unchanged — it operates at the token/string level.
- The `decode_utf8(const char *)` overload (single code point) is used by `parse_char()`
  in the GBNF parser — keep it.
