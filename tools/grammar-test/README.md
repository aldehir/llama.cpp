# llama-grammar-test

A CLI tool to test GBNF grammar correctness and profile logit-masking performance
without running inference. It loads only the vocabulary from a model file and
exercises the full grammar pipeline: parsing, DFA compilation, candidate
precomputation, and per-token logit masking.

## Building

```bash
cmake -B build
cmake --build build --target llama-grammar-test
```

## Usage

```
llama-grammar-test -g GRAMMAR_FILE [-m MODEL] [mode] [options]
```

The model file is used only for its vocabulary (`vocab_only` load). No weights
are loaded and no inference is performed. The model is required for all modes
except `--graph`.

## Modes

### Validate (default)

Test strings against the grammar and report pass/fail. On mismatch, a
byte-by-byte DFA state trace is printed for debugging.

```bash
# Inline test strings
llama-grammar-test -m model.gguf -g grammars/json.gbnf \
  --test-pass '{"key": 42}' \
  --test-pass '{}' \
  --test-fail '{invalid' \
  --test-fail ''

# From a file (one per line, prefixed with + or -)
llama-grammar-test -m model.gguf -g grammars/json.gbnf \
  --test-file tests.txt
```

Test file format:
```
+{"key": 42}
+{}
-{invalid
-
# lines starting with # are ignored
```

Exit code is 0 if all tests pass, 1 if any mismatch.

### Bench

Benchmark each grammar pipeline stage with wall-clock timings and a per-token
breakdown of the logit masking step.

```bash
llama-grammar-test -m model.gguf -g grammars/json.gbnf \
  --bench --bench-iters 1000
```

Example output:
```
Grammar: 78 rules, 23 DFAs (91 states total) | Vocab: 32000 tokens

  Parse:                  0.05 ms
  DFA compile:            1.41 ms  (23 DFAs, 91 states)
  Init (trie+precomp):   81.04 ms

  Logit masking (1000 iters):
    Mean:   0.311 ms/call
    Min:    0.304 ms   Max: 0.325 ms   Stddev: 0.006 ms
    Breakdown:
      Precomp-rejected:  31978 tokens ( 99.9%)
      Simulated:            20 tokens (  0.1%)
        Sim-accepted:        4 tokens
        Sim-rejected:       16 tokens
      EOG:                   1 tokens
      Empty/null:            1 tokens
```

### Candidates

Dump per-DFA-state candidate set analysis: representation type
(accept-heavy vs reject-heavy), set sizes, sample tokens, and memory usage.

```bash
llama-grammar-test -m model.gguf -g grammars/json.gbnf --candidates
```

### Masking report

Categorize every token in the vocabulary by how the grammar handles it at the
current state. Output is tab-separated for easy filtering with `grep`, `sort`,
`awk`, etc.

```bash
# Full report
llama-grammar-test -m model.gguf -g grammars/json.gbnf --masking-report

# Only accepted tokens
llama-grammar-test -m model.gguf -g grammars/json.gbnf --masking-report \
  | grep sim-accept
```

Categories:

| Category | Meaning |
|----------|---------|
| `precomp-reject` | Rejected by precomputed candidate bitmask (fast path) |
| `sim-reject` | Was a candidate but DFA simulation rejected it |
| `sim-accept` | Was a candidate and DFA simulation confirmed it |
| `eog-accept` | End-of-generation token, grammar is complete |
| `eog-reject` | End-of-generation token, grammar is not complete |
| `empty-reject` | Empty or null token piece |

### Graph

Generate a graphviz dot graph of the grammar structure. This mode does not
require a model file. The output includes:

- **Rule call graph**: each rule shown with its alternates (DFA segments and
  rule calls), with edges between rules that reference each other
- **DFA subgraphs**: each DFA as a state machine with byte-range transitions
  (compressed into readable labels like `a-z`, `0-9`)

```bash
# Render as SVG
llama-grammar-test -g grammars/json.gbnf --graph | dot -Tsvg -o json.svg

# Render as PNG
llama-grammar-test -g grammars/arithmetic.gbnf --graph | dot -Tpng -o arith.png

# Just inspect the dot source
llama-grammar-test -g grammars/chess.gbnf --graph
```

The start rule is highlighted with a blue border. Accept states in DFAs use
double circles. Start states in DFAs are bold.

## Common options

| Option | Description |
|--------|-------------|
| `--state STR` | Advance the grammar by this byte prefix before running the selected mode. Useful for inspecting mid-parse states. |
| `--grammar-root NAME` | Start rule name (default: `root`) |
| `--log-disable` | Suppress model loading logs |

## Examples

```bash
# Check what tokens are valid after '{"key": ' in a JSON grammar
llama-grammar-test -m model.gguf -g grammars/json.gbnf \
  --masking-report --state '{"key": ' | grep sim-accept

# Benchmark logit masking mid-parse
llama-grammar-test -m model.gguf -g grammars/json.gbnf \
  --bench --state '{"key": ' --bench-iters 500

# Inspect candidate sets for the arithmetic grammar
llama-grammar-test -m model.gguf -g grammars/arithmetic.gbnf --candidates

# Visualize grammar structure (no model needed)
llama-grammar-test -g grammars/json.gbnf --graph | dot -Tsvg -o json.svg
```
