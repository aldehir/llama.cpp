#!/bin/bash
set -o errexit
cmake --build build -t llama-grammar-test
./build/bin/llama-grammar-test -g grammars/json.gbnf -m ./models/ggml-vocab-llama-bpe.gguf --bench-string '{"a": 1, "b": 2}' --bench-iters 1
