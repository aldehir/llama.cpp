Build llama-grammar-test:

cmake --build build -t llama-grammar-test

Test with bench string:

./build/bin/llama-grammar-test -g grammars/json.gbnf -m ./models/ggml-vocab-llama-bpe.gguf --bench-string '{"a": 1, "b": 2}' --bench-iters 1


or:

./grammar-test.sh

This runs both


Files of interest:

src/llama-grammar.cpp
src/llama-grammar-dfa.cpp
tools/grammar-test/
