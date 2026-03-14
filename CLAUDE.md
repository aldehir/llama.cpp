Build llama-grammar-test:

cmake --build build-user-reldebug -t llama-grammar-test

Test with bench file:

./build-user-reldebug/bin/llama-grammar-test -g grammars/ini.gbnf -m ./models/ggml-vocab-llama-bpe.gguf --bench-file example2.ini --bench-iters 1 --log-disable

Files of interest:

src/llama-grammar.cpp
src/llama-grammar-dfa.cpp
tools/grammar-test/
