#include <string>
#include <iostream>

#include <nlohmann/json.hpp>

#include "jinja/recipe.h"
#include "jinja/runtime.h"
#include "jinja/value.h"

#include "testing.h"

using json = nlohmann::ordered_json;

// Parse a recipe, seed the context (ggml type names as strings + tensor vars),
// run it through the jinja runtime, and read back the resulting `type`.
static std::string eval_recipe(const std::string & src, const json & vars) {
    jinja::program prog = jinja::parse_recipe(src);

    jinja::context ctx(src);
    for (const char * name : { "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0" }) {
        ctx.set_val(name, jinja::mk_val<jinja::value_string>(std::string(name)));
    }
    jinja::global_from_json(ctx, vars, false);

    jinja::runtime rt(ctx);
    rt.execute(prog);

    return ctx.get_val("type")->as_string().str();
}

static void test_floor_division(testing & t);
static void test_basic_statements(testing & t);
static void test_control_flow(testing & t);
static void test_parse_errors(testing & t);
static void test_q4_k_m(testing & t);

int main(int argc, char * argv[]) {
    testing t(std::cout);
    t.verbose = true;

    for (int i = 1; i < argc; i++) {
        t.set_filter(argv[i]);
    }

    t.test("floor division", test_floor_division);
    t.test("basic statements", test_basic_statements);
    t.test("control flow", test_control_flow);
    t.test("parse errors", test_parse_errors);
    t.test("Q4_K_M transcription", test_q4_k_m);

    return t.summary();
}

static void test_floor_division(testing & t) {
    // // is integer floor division: 7 // 2 == 3 (float / would give 3.5)
    t.assert_equal("7 // 2 == 3", std::string("Q6_K"),
        eval_recipe("default Q4_K\nif 7 // 2 == 3\n  type = Q6_K\nendif\n", json::object()));

    // the threshold lands exactly where C++ integer division would: 80 // 8 == 10
    t.assert_equal("n // 8 threshold", std::string("Q6_K"),
        eval_recipe("default Q4_K\nif n // 8 == 10\n  type = Q6_K\nendif\n", {{"n", 80}}));

    // modulo on integers stays integer
    t.assert_equal("10 % 3 == 1", std::string("Q6_K"),
        eval_recipe("default Q4_K\nif 10 % 3 == 1\n  type = Q6_K\nendif\n", json::object()));
}

static void test_basic_statements(testing & t) {
    t.assert_equal("default only", std::string("Q4_K"),
        eval_recipe("default Q4_K\n", json::object()));

    t.assert_equal("type reassignment, last wins", std::string("Q8_0"),
        eval_recipe("default Q4_K\ntype = Q6_K\ntype = Q8_0\n", json::object()));

    // `set` binds a name, then it can be read later
    t.assert_equal("set then read", std::string("Q6_K"),
        eval_recipe("default Q4_K\nset hi = layer > 10\nif hi\n  type = Q6_K\nendif\n", {{"layer", 20}}));

    // reading the mutable `type` in a guard
    t.assert_equal("guard reads type", std::string("Q5_K"),
        eval_recipe("default Q4_K\nif type == Q4_K\n  type = Q5_K\nendif\n", json::object()));
}

static void test_control_flow(testing & t) {
    const char * src =
        "default Q4_K\n"
        "if category == \"A\"\n"
        "  type = Q6_K\n"
        "elif category == \"B\"\n"
        "  type = Q5_K\n"
        "else\n"
        "  type = Q8_0\n"
        "endif\n";

    t.assert_equal("if branch",   std::string("Q6_K"), eval_recipe(src, {{"category", "A"}}));
    t.assert_equal("elif branch", std::string("Q5_K"), eval_recipe(src, {{"category", "B"}}));
    t.assert_equal("else branch", std::string("Q8_0"), eval_recipe(src, {{"category", "C"}}));

    // nested if + and/or/not precedence
    t.assert_equal("and/or/not", std::string("Q6_K"),
        eval_recipe("default Q4_K\nif not done and (a or b)\n  type = Q6_K\nendif\n",
            {{"done", false}, {"a", false}, {"b", true}}));

    // line comments are ignored
    t.assert_equal("comments", std::string("Q6_K"),
        eval_recipe("# pick a type\ndefault Q4_K\ntype = Q6_K  # override\n", json::object()));
}

static void test_parse_errors(testing & t) {
    auto throws = [](const std::string & src) {
        try {
            jinja::parse_recipe(src);
        } catch (const jinja::recipe_exception &) {
            return true;
        }
        return false;
    };

    // bare `/` is deliberately not offered (only `//`)
    t.assert_true("bare / rejected", throws("default Q4_K\ntype = n / 2\n"));
    // no for/macro/function-call productions: totality by construction
    t.assert_true("for rejected", throws("default Q4_K\nfor x in y\n  type = Q6_K\nendfor\n"));
    // dangling block
    t.assert_true("missing endif rejected", throws("default Q4_K\nif a\n  type = Q6_K\n"));
}

// Full Q4_K_M recipe, transcribed from llama_tensor_get_type_impl.
static const char * Q4_K_M = R"(# Q4_K_M -- transcribed from llama_tensor_get_type_impl
default Q4_K

set more_bits = layer < n_layer // 8
             or layer >= 7 * n_layer // 8
             or (layer - n_layer // 8) % 3 == 2

if category == "ATTENTION_WV"
    if more_bits
        type = Q6_K
    endif
    if model_type == "70B" and (type == Q3_K or type == Q4_K)
        type = Q5_K
    endif
    if n_expert == 8
        type = Q8_0
    endif
endif

if category == "FFN_DOWN"
    if arch == "falcon"
        if layer < n_layer // 16
            type = Q6_K
        elif more_bits
            type = Q5_K
        else
            type = Q4_K
        endif
    else
        if more_bits
            type = Q6_K
        endif
    endif
endif
)";

static void test_q4_k_m(testing & t) {
    auto wv = [&](int layer, int n_layer, const std::string & model_type, int n_expert) {
        return eval_recipe(Q4_K_M, {
            {"category", "ATTENTION_WV"}, {"layer", layer}, {"n_layer", n_layer},
            {"arch", "llama"}, {"model_type", model_type}, {"n_expert", n_expert},
        });
    };
    auto ffn = [&](int layer, int n_layer, const std::string & arch) {
        return eval_recipe(Q4_K_M, {
            {"category", "FFN_DOWN"}, {"layer", layer}, {"n_layer", n_layer},
            {"arch", arch}, {"model_type", "7B"}, {"n_expert", 1},
        });
    };

    // ATTENTION_WV
    t.assert_equal("wv more_bits -> Q6_K",  std::string("Q6_K"), wv(0,  80, "7B",  1));
    t.assert_equal("wv 70B -> Q5_K",        std::string("Q5_K"), wv(40, 80, "70B", 1));
    t.assert_equal("wv n_expert=8 -> Q8_0", std::string("Q8_0"), wv(40, 80, "7B",  8));
    t.assert_equal("wv plain -> Q4_K",      std::string("Q4_K"), wv(40, 80, "7B",  1));
    // 70B + more_bits: type becomes Q6_K first, so the Q3_K/Q4_K guard fails -> stays Q6_K
    t.assert_equal("wv 70B + more_bits",    std::string("Q6_K"), wv(0,  80, "70B", 1));

    // FFN_DOWN, falcon: the elif chain reproduces legacy first-1/16 -> Q6_K
    t.assert_equal("falcon first 1/16 -> Q6_K", std::string("Q6_K"), ffn(2,  80, "falcon"));
    t.assert_equal("falcon more_bits -> Q5_K",  std::string("Q5_K"), ffn(8,  80, "falcon"));
    t.assert_equal("falcon plain -> Q4_K",      std::string("Q4_K"), ffn(40, 80, "falcon"));

    // FFN_DOWN, non-falcon
    t.assert_equal("ffn more_bits -> Q6_K", std::string("Q6_K"), ffn(2,  80, "llama"));
    t.assert_equal("ffn middle third",      std::string("Q6_K"), ffn(12, 80, "llama")); // (12-10)%3==2
    t.assert_equal("ffn plain -> Q4_K",     std::string("Q4_K"), ffn(40, 80, "llama"));

    // unmatched category falls through to the default
    t.assert_equal("other category -> Q4_K", std::string("Q4_K"),
        eval_recipe(Q4_K_M, {
            {"category", "ATTENTION_QKV"}, {"layer", 0}, {"n_layer", 80},
            {"arch", "llama"}, {"model_type", "7B"}, {"n_expert", 1},
        }));
}
