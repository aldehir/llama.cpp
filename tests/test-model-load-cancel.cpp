#include "testing.h"

#include "llama.h"
#include "common.h"

#include <cstdlib>

int main(int argc, char *argv[] ) {
    auto * model_path = common_get_model_or_exit(argc, argv);

    testing t;
    t.capture_output = true;
    t.apply_env();
    if (argc > 2) {
        t.set_filter(argv[2]);
    }

    t.test("cancel", [&](testing & t) {
        auto * file = fopen(model_path, "r");
        if (!t.assert_true(std::string("model file '") + model_path + "' exists", file != nullptr)) {
            fprintf(stderr, "no model at '%s' found\n", model_path);
            return;
        }

        fprintf(stderr, "using '%s'\n", model_path);
        fclose(file);

        llama_backend_init();
        auto params = llama_model_params{};
        params.load_mode = LLAMA_LOAD_MODE_NONE;
        params.progress_callback = [](float progress, void * ctx){
            (void) ctx;
            return progress > 0.50;
        };
        auto * model = llama_model_load_from_file(model_path, params);
        llama_backend_free();

        t.assert_true(std::string("loading '") + model_path + "' is cancelled by the progress callback", model == nullptr);
    });

    return t.summary();
}
