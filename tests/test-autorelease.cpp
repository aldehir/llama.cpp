// ref: https://github.com/ggml-org/llama.cpp/issues/4952#issuecomment-1892864763

#include "testing.h"

#include <thread>

#include "llama.h"
#include "common.h"

// This creates a new context inside a pthread and then tries to exit cleanly.
int main(int argc, char ** argv) {
    auto * model_path = common_get_model_or_exit(argc, argv);

    testing t;
    t.capture_output = true;
    t.apply_env();
    if (argc > 2) {
        t.set_filter(argv[2]);
    }

    t.test("context_in_thread", [&](testing & t) {
        bool model_ok = false;
        bool ctx_ok   = false;

        std::thread([&]() {
            llama_backend_init();
            auto * model = llama_model_load_from_file(model_path, llama_model_default_params());
            auto * ctx = llama_init_from_model(model, llama_context_default_params());
            model_ok = model != nullptr;
            ctx_ok   = ctx != nullptr;
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
        }).join();

        t.assert_true(std::string("load model '") + model_path + "' in a thread", model_ok);
        t.assert_true("create context in a thread", ctx_ok);
    });

    return t.summary();
}
