#include "testing.h"

#include "llama.h"
#include "common.h"
#include "console.h"

#include "../src/unicode.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

static void load_vocab(testing & t, const std::string & fname, llama_model *& model, llama_context *& ctx) {
    fprintf(stderr, "%s : reading vocab from: '%s'\n", __func__, fname.c_str());

    auto mparams = llama_model_default_params();

    mparams.vocab_only = true;

    model = llama_model_load_from_file(fname.c_str(), mparams);

    if (!t.assert_true("load vocab '" + fname + "'", model != NULL)) {
        fprintf(stderr, "%s: error: failed to load vocab '%s'\n", __func__, fname.c_str());
        return;
    }

    auto cparams = llama_context_default_params();

    ctx = llama_init_from_model(model, cparams);

    if (!t.assert_true("create context for '" + fname + "'", ctx != NULL)) {
        fprintf(stderr, "%s: error: failed to load vocab '%s'\n", __func__, fname.c_str());
    }
}

// every token must tokenize back to itself, the loop stops at the first failure
static void test_tokens(testing & t, llama_context * ctx, const llama_vocab * vocab) {
    const int n_vocab = llama_vocab_n_tokens(vocab);

    t.log("checking " + std::to_string(n_vocab) + " tokens");

    for (int i = 0; i < n_vocab; ++i) {
        std::string str = common_detokenize(ctx, std::vector<int>(1, i), true);
        std::vector<llama_token> tokens = common_tokenize(ctx, str, false, true);
        std::string check = common_detokenize(ctx, tokens);
        const std::string msg = "token " + std::to_string(i) + " '" + str + "'(" + std::to_string(str.length()) + ") roundtrips through tokenize/detokenize";
        if (!t.assert_equal(msg, str, check)) {
            fprintf(stderr, "%s : error: token %d detokenizes to '%s'(%zu) but tokenization of this detokenizes to '%s'(%zu)\n",
                __func__, i, str.c_str(), str.length(), check.c_str(), check.length());
            return;
        }
    }
}

// every unicode codepoint must roundtrip, the threads stop at the first failure
static void test_codepoints(testing & t, llama_context * ctx) {
    const int nthread = std::thread::hardware_concurrency();

    std::vector<std::thread> threads(nthread);

    std::atomic_int errcode = {};

    std::mutex mtx;
    std::vector<std::string> errors;

    t.log("checking codepoints on " + std::to_string(nthread) + " threads");

    for (int i = 0; i < nthread; ++i) {
        threads[i] = std::thread([i, nthread, ctx, &errcode, &mtx, &errors]() {
            for (uint32_t cp = i; !errcode && cp < 0x00110000; cp += nthread) {
                if ((0x0000D800 <= cp && cp <= 0x0000DFFF) ||  // surrogates \p{Cs}
                    (0x00040000 <= cp && cp <= 0x000E0000)) {  // undefined \p{Cn}
                    continue;
                }

                std::string str = unicode_cpt_to_utf8(cp);
                std::vector<llama_token> tokens = common_tokenize(ctx, str, false, true);
                std::string check = common_detokenize(ctx, tokens);
                if (cp != 9601 && str != check) {
                    char msg[256];
                    snprintf(msg, sizeof(msg), "codepoint 0x%x detokenizes to '%s'(%zu) instead of '%s'(%zu)",
                             cp, check.c_str(), check.length(), str.c_str(), str.length());
                    fprintf(stderr, "error: %s\n", msg);
                    std::lock_guard<std::mutex> lock(mtx);
                    errors.push_back(msg);
                    errcode = 3;
                }
            }
        });
    }

    for (auto & th : threads) {
        th.join();
    }

    for (const auto & err : errors) {
        t.assert_true(err, false);
    }
    t.assert_true("all codepoints roundtrip through tokenize/detokenize", errcode == 0);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <vocab-file> [filter]\n", argv[0]);
        return 1;
    }

    const std::string fname = argv[1];

    testing t;
    t.capture_output = true;
    t.apply_env();
    if (argc > 2) {
        t.set_filter(argv[2]);
    }

    llama_model * model = nullptr;
    llama_context * ctx = nullptr;

    llama_backend_init();

    t.test("load", [&](testing & t) {
        load_vocab(t, fname, model, ctx);
    });

    if (ctx != nullptr) {
        const llama_vocab * vocab = llama_model_get_vocab(model);

        //GGML_ASSERT(llama_vocab_type(model) == LLAMA_VOCAB_TYPE_SPM);
        if (llama_vocab_type(vocab) != LLAMA_VOCAB_TYPE_SPM) {
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 99;
        }

#ifdef _WIN32
        // We need this for unicode console support
        console::init(false, false);
        atexit([]() { console::cleanup(); });
#endif

        t.test("tokens", [&](testing & t) {
            test_tokens(t, ctx, vocab);
        });

        t.test("codepoints", [&](testing & t) {
            test_codepoints(t, ctx);
        });
    }

    llama_free(ctx);
    llama_model_free(model);

    llama_backend_free();

    return t.summary();
}
