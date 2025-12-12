#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace pcre_ast {

// Forward declarations
struct Node;
using NodePtr = std::shared_ptr<Node>;
using NodeList = std::vector<NodePtr>;

// Character range for character classes
struct CharRange {
    uint32_t start;
    uint32_t end;

    CharRange(uint32_t c) : start(c), end(c) {}
    CharRange(uint32_t s, uint32_t e) : start(s), end(e) {}

    bool is_single() const { return start == end; }
};

// Unicode property types
enum class UnicodeProperty {
    Letter,        // \p{L}
    Number,        // \p{N}
    Whitespace,    // \s or \p{Z}
    Punctuation,   // \p{P}
    Symbol,        // \p{S}
    Mark,          // \p{M}
    Han,           // \p{Han}
    Uppercase,     // \p{Lu}
    Lowercase,     // \p{Ll}
    Titlecase,     // \p{Lt}
    Modifier,      // \p{Lm}
    Other,         // \p{Lo}
};

// AST Node types

// Matches a literal string
struct Literal {
    std::string value;
};

// Matches any single character (.)
struct AnyChar {};

// Matches a character class [abc], [a-z], [^abc]
struct CharClass {
    std::vector<CharRange> ranges;
    std::vector<UnicodeProperty> properties;
    std::vector<UnicodeProperty> excluded_properties;  // For set subtraction like [^\p{Han}]
    bool negated = false;
    bool includes_specific_chars = false;  // true if ranges contains specific chars, not just properties
};

// Unicode property shorthand \p{L}, \p{N}, etc.
struct UnicodePropertyMatch {
    UnicodeProperty property;
    bool negated = false;  // \P{...}
};

// Whitespace shortcuts \s, \S
struct WhitespaceMatch {
    bool negated = false;  // \S
};

// Word character shortcuts \w, \W
struct WordMatch {
    bool negated = false;  // \W
};

// Digit shortcuts \d, \D
struct DigitMatch {
    bool negated = false;  // \D
};

// Special character matches
struct SpecialChar {
    enum Type {
        CarriageReturn,  // \r
        Newline,         // \n
        Tab,             // \t
        FormFeed,        // \f
        VerticalTab,     // \v
        Null,            // \0
    };
    Type type;
};

// Sequence of nodes (concatenation)
struct Sequence {
    NodeList children;
};

// Alternation (a|b|c)
struct Alternation {
    NodeList alternatives;
};

// Quantifiers
struct Quantifier {
    NodePtr child;
    int min_count;
    int max_count;  // -1 for unbounded
    bool greedy = true;
    bool possessive = false;
};

// Non-capturing group (?:...)
struct NonCapturingGroup {
    NodePtr child;
};

// Capturing group (...)
struct CapturingGroup {
    NodePtr child;
    int group_number = 0;
};

// Flags group (?i:...), (?-i:...), etc.
struct FlagsGroup {
    NodePtr child;
    bool case_insensitive = false;
    bool multiline = false;
    bool dotall = false;
};

// Positive lookahead (?=...)
struct PositiveLookahead {
    NodePtr child;
};

// Negative lookahead (?!...)
struct NegativeLookahead {
    NodePtr child;
};

// Positive lookbehind (?<=...)
struct PositiveLookbehind {
    NodePtr child;
};

// Negative lookbehind (?<!...)
struct NegativeLookbehind {
    NodePtr child;
};

// Start anchor ^
struct StartAnchor {};

// End anchor $
struct EndAnchor {};

// Word boundary \b, \B
struct WordBoundary {
    bool negated = false;  // \B
};

// Empty match (epsilon)
struct Empty {};

// The main AST node variant
using NodeVariant = std::variant<
    Literal,
    AnyChar,
    CharClass,
    UnicodePropertyMatch,
    WhitespaceMatch,
    WordMatch,
    DigitMatch,
    SpecialChar,
    Sequence,
    Alternation,
    Quantifier,
    NonCapturingGroup,
    CapturingGroup,
    FlagsGroup,
    PositiveLookahead,
    NegativeLookahead,
    PositiveLookbehind,
    NegativeLookbehind,
    StartAnchor,
    EndAnchor,
    WordBoundary,
    Empty
>;

struct Node {
    NodeVariant value;

    template<typename T>
    Node(T&& v) : value(std::forward<T>(v)) {}

    template<typename T>
    bool is() const { return std::holds_alternative<T>(value); }

    template<typename T>
    const T& as() const { return std::get<T>(value); }

    template<typename T>
    T& as() { return std::get<T>(value); }
};

// Helper functions for creating nodes
template<typename T, typename... Args>
NodePtr make_node(Args&&... args) {
    return std::make_shared<Node>(T{std::forward<Args>(args)...});
}

inline NodePtr make_literal(const std::string& s) {
    return make_node<Literal>(s);
}

inline NodePtr make_any() {
    return make_node<AnyChar>();
}

inline NodePtr make_sequence(NodeList children) {
    if (children.size() == 1) {
        return children[0];
    }
    return make_node<Sequence>(std::move(children));
}

inline NodePtr make_alternation(NodeList alternatives) {
    if (alternatives.size() == 1) {
        return alternatives[0];
    }
    return make_node<Alternation>(std::move(alternatives));
}

inline NodePtr make_quantifier(NodePtr child, int min, int max, bool greedy = true) {
    auto q = Quantifier{};
    q.child = child;
    q.min_count = min;
    q.max_count = max;
    q.greedy = greedy;
    return std::make_shared<Node>(q);
}

inline NodePtr make_optional(NodePtr child, bool greedy = true) {
    return make_quantifier(child, 0, 1, greedy);
}

inline NodePtr make_star(NodePtr child, bool greedy = true) {
    return make_quantifier(child, 0, -1, greedy);
}

inline NodePtr make_plus(NodePtr child, bool greedy = true) {
    return make_quantifier(child, 1, -1, greedy);
}

inline NodePtr make_repeat(NodePtr child, int n) {
    return make_quantifier(child, n, n);
}

inline NodePtr make_repeat_range(NodePtr child, int min, int max) {
    return make_quantifier(child, min, max);
}

inline NodePtr make_char_class(std::vector<CharRange> ranges, bool negated = false) {
    auto cc = CharClass{};
    cc.ranges = std::move(ranges);
    cc.negated = negated;
    cc.includes_specific_chars = true;
    return std::make_shared<Node>(cc);
}

inline NodePtr make_unicode_property(UnicodeProperty prop, bool negated = false) {
    return make_node<UnicodePropertyMatch>(prop, negated);
}

inline NodePtr make_non_capturing_group(NodePtr child) {
    auto g = NonCapturingGroup{};
    g.child = child;
    return std::make_shared<Node>(g);
}

inline NodePtr make_flags_group(NodePtr child, bool case_insensitive) {
    auto g = FlagsGroup{};
    g.child = child;
    g.case_insensitive = case_insensitive;
    return std::make_shared<Node>(g);
}

inline NodePtr make_positive_lookahead(NodePtr child) {
    auto la = PositiveLookahead{};
    la.child = child;
    return std::make_shared<Node>(la);
}

inline NodePtr make_negative_lookahead(NodePtr child) {
    auto la = NegativeLookahead{};
    la.child = child;
    return std::make_shared<Node>(la);
}

// Debug/printing functions
std::string node_to_string(const NodePtr& node, int indent = 0);
std::string unicode_property_name(UnicodeProperty prop);

}  // namespace pcre_ast
