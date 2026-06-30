#pragma once

#include "ggml.h"

#include <memory>
#include <string>

namespace jinja { struct program; }

// Canonical tensor categories. The string values are what the host assigns to
// `category` per tensor; recipes refer to them as named constants, e.g.
// `if category == FFN_DOWN`.
namespace quant_category {
    inline constexpr const char * OUTPUT      = "output";
    inline constexpr const char * TOKEN_EMBD  = "token_embd";
    inline constexpr const char * ATTN_QKV    = "attn_qkv";
    inline constexpr const char * ATTN_KV_B   = "attn_kv_b";
    inline constexpr const char * ATTN_V      = "attn_v";
    inline constexpr const char * ATTN_K      = "attn_k";
    inline constexpr const char * ATTN_Q      = "attn_q";
    inline constexpr const char * ATTN_OUTPUT = "attn_output";
    inline constexpr const char * FFN_UP      = "ffn_up";
    inline constexpr const char * FFN_GATE    = "ffn_gate";
    inline constexpr const char * FFN_DOWN    = "ffn_down";
}

inline constexpr const char * const QUANT_RECIPE_CATEGORIES[] = {
    quant_category::OUTPUT,
    quant_category::TOKEN_EMBD,
    quant_category::ATTN_QKV,
    quant_category::ATTN_KV_B,
    quant_category::ATTN_V,
    quant_category::ATTN_K,
    quant_category::ATTN_Q,
    quant_category::ATTN_OUTPUT,
    quant_category::FFN_UP,
    quant_category::FFN_GATE,
    quant_category::FFN_DOWN,
};

// Per-tensor context seeded into a recipe before evaluation. Mirrors the
// information the quantizer derives internally (category, layer position,
// architecture, expert count, ...).
struct quant_recipe_tensor {
    std::string category;    // tensor category, e.g. "attn_v", "ffn_down"
    std::string arch;        // model architecture, e.g. "llama"
    std::string model_type;  // size label, e.g. "70B" (empty if unknown)

    int  layer          = -1; // absolute layer index parsed from the tensor name
    int  n_layer        = 0;  // total number of layers (block_count)
    int  n_expert       = 0;  // number of experts (0 for dense models)
    int  n_gqa          = 0;  // grouped-query attention factor (n_head / n_head_kv)
    int  category_index = 0;  // position of this tensor within its own category
    int  category_count = 0;  // total tensors in this category
    bool has_imatrix    = false;
};

// A parsed quantization recipe.
//
// A recipe is a small, total-by-construction evaluation dialect that is parsed
// into a jinja program and executed once per tensor to choose a ggml_type. See
// quant-recipe.cpp for the grammar.
class quant_recipe {
public:
    quant_recipe();
    ~quant_recipe();
    quant_recipe(quant_recipe &&) noexcept;
    quant_recipe & operator=(quant_recipe &&) noexcept;

    // Parse a recipe from source text. Throws std::runtime_error on error.
    static quant_recipe parse(const std::string & src);

    // Load and parse a recipe from a file. Throws std::runtime_error on error.
    static quant_recipe load(const std::string & path);

    // Evaluate the recipe for a tensor. Returns GGML_TYPE_COUNT if the recipe
    // leaves `type` unset or assigns an unknown type name.
    ggml_type eval(const quant_recipe_tensor & tensor) const;

private:
    std::unique_ptr<jinja::program> prog_;
};
