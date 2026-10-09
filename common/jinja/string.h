#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "utils.h"

namespace jinja {

struct string_part {
    // kinds tag where a part came from, as a bitmask, e.g. to differentiate between user input and template strings
    // transformations should handle them as follows:
    // - one-to-one (e.g., uppercase, lowercase): preserve kinds
    // - one-to-many (e.g., strip): all resulting parts keep the kinds of the input string
    // - many-to-one (e.g., concat): the resulting part keeps the kinds shared by ALL input parts
    enum kinds : uint32_t {
        KIND_NONE       = 0,
        KIND_INPUT      = 1 << 0, // user input, may skip parsing special tokens
        KIND_GEN_PROMPT = 1 << 1, // emitted by a branch taken on add_generation_prompt
    };

    std::string val;
    uint32_t kind = KIND_NONE;

    bool is_input() const { return kind & KIND_INPUT; }

    bool is_uppercase() const;
    bool is_lowercase() const;
};

struct string {
    std::vector<string_part> parts;
    string() = default;
    string(const std::string & v, uint32_t kind = string_part::KIND_NONE) {
        parts.push_back({v, kind});
    }
    string(int v) {
        parts.push_back({std::to_string(v)});
    }
    string(double v) {
        parts.push_back({std::to_string(v)});
    }

    // add a kind to all parts
    void mark_kind(uint32_t kind);

    std::string str() const;
    size_t length() const;
    void hash_update(hasher & hash) const noexcept;
    // the kinds shared by ALL parts
    uint32_t common_kind() const;
    bool is_uppercase() const;
    bool is_lowercase() const;

    // add the kinds shared by ALL parts of other
    void mark_kind_based_on(const string & other);

    string & append(const string & other);

    // in-place transformations

    string uppercase();
    string lowercase();
    string capitalize();
    string titlecase();
    string strip(bool left, bool right, std::optional<const std::string_view> chars = std::nullopt);
};

} // namespace jinja
