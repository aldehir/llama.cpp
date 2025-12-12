#include "pcre-parser.h"
#include "pcre-ast.h"

#include <cctype>
#include <cstdint>
#include <sstream>
#include <unordered_map>

namespace {
// Helper functions to avoid std::isdigit ambiguity with char type
inline bool is_digit(char c) { return c >= '0' && c <= '9'; }
inline bool is_xdigit(char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}
inline char to_lower(char c) {
    return (c >= 'A' && c <= 'Z') ? (c + ('a' - 'A')) : c;
}
}  // namespace

namespace pcre_parser {

using namespace pcre_ast;

// Recursive descent parser for PCRE regex
class Parser {
    std::string input_;
    size_t pos_ = 0;
    int group_count_ = 0;

public:
    explicit Parser(const std::string& pattern) : input_(pattern) {}

    NodePtr parse() {
        auto result = parse_alternation();
        if (pos_ < input_.size()) {
            throw ParseError("Unexpected character", pos_, remaining());
        }
        return result;
    }

private:
    // Get remaining input for error messages
    std::string remaining() const {
        return input_.substr(pos_, std::min(size_t(20), input_.size() - pos_));
    }

    bool at_end() const {
        return pos_ >= input_.size();
    }

    char peek(size_t offset = 0) const {
        if (pos_ + offset >= input_.size()) return '\0';
        return input_[pos_ + offset];
    }

    char advance() {
        if (at_end()) return '\0';
        return input_[pos_++];
    }

    bool match(char c) {
        if (peek() == c) {
            advance();
            return true;
        }
        return false;
    }

    bool match(const std::string& s) {
        if (input_.compare(pos_, s.size(), s) == 0) {
            pos_ += s.size();
            return true;
        }
        return false;
    }

    void expect(char c) {
        if (!match(c)) {
            std::ostringstream oss;
            oss << "Expected '" << c << "'";
            throw ParseError(oss.str(), pos_, remaining());
        }
    }

    // Check if we're at a position that ends an alternation branch
    bool at_alternation_end() const {
        if (at_end()) return true;
        char c = peek();
        return c == ')' || c == '|';
    }

    // Parse alternation: a|b|c
    NodePtr parse_alternation() {
        NodeList alternatives;
        alternatives.push_back(parse_sequence());

        while (match('|')) {
            alternatives.push_back(parse_sequence());
        }

        return make_alternation(std::move(alternatives));
    }

    // Parse sequence of atoms: abc
    NodePtr parse_sequence() {
        NodeList children;

        while (!at_alternation_end()) {
            auto atom = parse_quantified();
            if (atom) {
                children.push_back(atom);
            } else {
                break;
            }
        }

        if (children.empty()) {
            return make_node<Empty>();
        }

        return make_sequence(std::move(children));
    }

    // Parse atom with optional quantifier
    NodePtr parse_quantified() {
        auto atom = parse_atom();
        if (!atom) return nullptr;

        // Check for quantifiers
        if (match('*')) {
            bool greedy = !match('?');
            return make_star(atom, greedy);
        }
        if (match('+')) {
            bool greedy = !match('?');
            return make_plus(atom, greedy);
        }
        if (match('?')) {
            bool greedy = !match('?');
            return make_optional(atom, greedy);
        }
        if (match('{')) {
            return parse_repetition(atom);
        }

        return atom;
    }

    // Parse repetition: {n}, {n,}, {n,m}
    NodePtr parse_repetition(NodePtr atom) {
        int min_count = parse_number();
        int max_count = min_count;

        if (match(',')) {
            if (peek() == '}') {
                max_count = -1;  // Unbounded
            } else {
                max_count = parse_number();
            }
        }

        expect('}');
        bool greedy = !match('?');

        return make_quantifier(atom, min_count, max_count, greedy);
    }

    int parse_number() {
        int result = 0;
        bool found = false;
        while (is_digit(peek())) {
            found = true;
            result = result * 10 + (advance() - '0');
        }
        if (!found) {
            throw ParseError("Expected number", pos_, remaining());
        }
        return result;
    }

    // Parse single atom
    NodePtr parse_atom() {
        if (at_end() || at_alternation_end()) {
            return nullptr;
        }

        char c = peek();

        // Group
        if (c == '(') {
            return parse_group();
        }

        // Character class
        if (c == '[') {
            return parse_char_class();
        }

        // Escape sequence
        if (c == '\\') {
            return parse_escape();
        }

        // Any character
        if (c == '.') {
            advance();
            return make_any();
        }

        // Start anchor
        if (c == '^') {
            advance();
            return make_node<StartAnchor>();
        }

        // End anchor
        if (c == '$') {
            advance();
            return make_node<EndAnchor>();
        }

        // Special characters that should not be treated as literals
        if (c == '*' || c == '+' || c == '?' || c == '{' || c == '}' ||
            c == ')' || c == ']') {
            return nullptr;
        }

        // Literal character
        advance();
        return make_literal(std::string(1, c));
    }

    // Parse group: (...), (?:...), (?i:...), (?=...), (?!...)
    NodePtr parse_group() {
        expect('(');

        if (match('?')) {
            // Special group
            if (match(':')) {
                // Non-capturing group
                auto child = parse_alternation();
                expect(')');
                return make_non_capturing_group(child);
            }

            if (match('=')) {
                // Positive lookahead
                auto child = parse_alternation();
                expect(')');
                return make_positive_lookahead(child);
            }

            if (match('!')) {
                // Negative lookahead
                auto child = parse_alternation();
                expect(')');
                return make_negative_lookahead(child);
            }

            if (match('<')) {
                if (match('=')) {
                    // Positive lookbehind
                    auto child = parse_alternation();
                    expect(')');
                    auto lb = PositiveLookbehind{};
                    lb.child = child;
                    return std::make_shared<Node>(lb);
                }
                if (match('!')) {
                    // Negative lookbehind
                    auto child = parse_alternation();
                    expect(')');
                    auto lb = NegativeLookbehind{};
                    lb.child = child;
                    return std::make_shared<Node>(lb);
                }
                throw ParseError("Unknown lookbehind type", pos_, remaining());
            }

            // Check for flags like (?i:...) or (?-i:...)
            bool case_insensitive = false;
            bool has_flags = false;

            while (peek() != ':' && peek() != ')' && !at_end()) {
                if (match('i')) {
                    case_insensitive = true;
                    has_flags = true;
                } else if (match('-')) {
                    if (match('i')) {
                        case_insensitive = false;
                        has_flags = true;
                    }
                } else if (match('m') || match('s') || match('x')) {
                    // Ignore these flags for now
                    has_flags = true;
                } else {
                    throw ParseError("Unknown group modifier", pos_, remaining());
                }
            }

            if (has_flags && match(':')) {
                auto child = parse_alternation();
                expect(')');
                return make_flags_group(child, case_insensitive);
            }

            // Inline flags without content (?i) - apply to rest of pattern
            if (has_flags && match(')')) {
                // Return empty node with flags - this is a simplification
                return make_node<Empty>();
            }

            throw ParseError("Unrecognized group syntax", pos_, remaining());
        }

        // Capturing group
        int group_num = ++group_count_;
        auto child = parse_alternation();
        expect(')');

        auto g = CapturingGroup{};
        g.child = child;
        g.group_number = group_num;
        return std::make_shared<Node>(g);
    }

    // Parse character class: [abc], [a-z], [^abc]
    NodePtr parse_char_class() {
        expect('[');

        CharClass cc;
        cc.negated = match('^');

        // Handle special case of ] as first char in class
        if (peek() == ']') {
            cc.ranges.push_back(CharRange(']'));
            cc.includes_specific_chars = true;
            advance();
        }

        while (!at_end() && peek() != ']') {
            // Check for POSIX class or Unicode property inside []
            if (peek() == '\\') {
                advance();
                if (peek() == 'p' || peek() == 'P') {
                    bool negated = (peek() == 'P');
                    advance();
                    auto prop = parse_unicode_property_name();
                    if (negated) {
                        cc.excluded_properties.push_back(prop);
                    } else {
                        cc.properties.push_back(prop);
                    }
                    continue;
                }

                // Handle escape in character class
                auto escaped = parse_escape_in_class();
                if (auto* sr = std::get_if<CharRange>(&escaped)) {
                    cc.ranges.push_back(*sr);
                    cc.includes_specific_chars = true;
                } else if (auto* prop = std::get_if<UnicodeProperty>(&escaped)) {
                    cc.properties.push_back(*prop);
                }
                continue;
            }

            // Check for && set intersection (used in some PCRE patterns)
            if (peek() == '&' && peek(1) == '&') {
                advance();
                advance();
                // Parse the intersected set and use it as exclusion
                if (match('[')) {
                    bool inner_negated = match('^');
                    while (!at_end() && peek() != ']') {
                        if (peek() == '\\') {
                            advance();
                            if (peek() == 'p' || peek() == 'P') {
                                bool prop_negated = (peek() == 'P');
                                advance();
                                auto prop = parse_unicode_property_name();
                                if (inner_negated != prop_negated) {
                                    cc.properties.push_back(prop);
                                } else {
                                    cc.excluded_properties.push_back(prop);
                                }
                                continue;
                            }
                            // Handle other escapes
                            auto escaped = parse_escape_in_class();
                            if (auto* sr = std::get_if<CharRange>(&escaped)) {
                                // For intersection, exclusion depends on inner_negated
                                if (inner_negated) {
                                    cc.ranges.push_back(*sr);
                                    cc.includes_specific_chars = true;
                                }
                            }
                        } else {
                            uint32_t ch = parse_char_class_char();
                            if (inner_negated) {
                                cc.ranges.push_back(CharRange(ch));
                                cc.includes_specific_chars = true;
                            }
                        }
                    }
                    expect(']');
                }
                continue;
            }

            // Regular character or range
            uint32_t start = parse_char_class_char();

            if (peek() == '-' && peek(1) != ']') {
                advance();  // consume '-'
                uint32_t end = parse_char_class_char();
                cc.ranges.push_back(CharRange(start, end));
            } else {
                cc.ranges.push_back(CharRange(start));
            }
            cc.includes_specific_chars = true;
        }

        expect(']');
        return std::make_shared<Node>(cc);
    }

    uint32_t parse_char_class_char() {
        if (peek() == '\\') {
            advance();
            return parse_escaped_char();
        }
        return static_cast<uint32_t>(static_cast<unsigned char>(advance()));
    }

    uint32_t parse_escaped_char() {
        char c = advance();
        switch (c) {
            case 'n': return '\n';
            case 'r': return '\r';
            case 't': return '\t';
            case 'f': return '\f';
            case 'v': return '\v';
            case '0': return '\0';
            case 'x': return parse_hex_escape();
            case 'u': return parse_unicode_escape();
            default: return static_cast<uint32_t>(static_cast<unsigned char>(c));
        }
    }

    uint32_t parse_hex_escape() {
        // \xNN
        uint32_t result = 0;
        for (int i = 0; i < 2 && is_xdigit(peek()); ++i) {
            char c = advance();
            result = result * 16 + (is_digit(c) ? c - '0' : to_lower(c) - 'a' + 10);
        }
        return result;
    }

    uint32_t parse_unicode_escape() {
        // \uNNNN or \u{NNNN}
        if (match('{')) {
            uint32_t result = 0;
            while (is_xdigit(peek())) {
                char c = advance();
                result = result * 16 + (is_digit(c) ? c - '0' : to_lower(c) - 'a' + 10);
            }
            expect('}');
            return result;
        }

        uint32_t result = 0;
        for (int i = 0; i < 4 && is_xdigit(peek()); ++i) {
            char c = advance();
            result = result * 16 + (is_digit(c) ? c - '0' : to_lower(c) - 'a' + 10);
        }
        return result;
    }

    // Result type for escape sequences in character classes
    using EscapeInClassResult = std::variant<CharRange, UnicodeProperty>;

    EscapeInClassResult parse_escape_in_class() {
        char c = peek();

        // Shorthand classes inside []
        if (c == 'd') {
            advance();
            return UnicodeProperty::Number;
        }
        if (c == 's') {
            advance();
            return UnicodeProperty::Whitespace;
        }
        if (c == 'w') {
            advance();
            return UnicodeProperty::Letter;  // Simplified - actually [a-zA-Z0-9_]
        }

        // Unicode property
        if (c == 'p' || c == 'P') {
            advance();
            return parse_unicode_property_name();
        }

        // Regular escaped char
        return CharRange(parse_escaped_char());
    }

    // Parse escape sequence
    NodePtr parse_escape() {
        expect('\\');
        char c = peek();

        // Unicode property
        if (c == 'p' || c == 'P') {
            bool negated = (c == 'P');
            advance();
            auto prop = parse_unicode_property_name();
            return make_unicode_property(prop, negated);
        }

        // Shorthand classes
        if (c == 's') {
            advance();
            return make_node<WhitespaceMatch>(false);
        }
        if (c == 'S') {
            advance();
            return make_node<WhitespaceMatch>(true);
        }
        if (c == 'd') {
            advance();
            return make_node<DigitMatch>(false);
        }
        if (c == 'D') {
            advance();
            return make_node<DigitMatch>(true);
        }
        if (c == 'w') {
            advance();
            return make_node<WordMatch>(false);
        }
        if (c == 'W') {
            advance();
            return make_node<WordMatch>(true);
        }

        // Word boundary
        if (c == 'b') {
            advance();
            return make_node<WordBoundary>(false);
        }
        if (c == 'B') {
            advance();
            return make_node<WordBoundary>(true);
        }

        // Special characters
        if (c == 'r') {
            advance();
            return make_node<SpecialChar>(SpecialChar::CarriageReturn);
        }
        if (c == 'n') {
            advance();
            return make_node<SpecialChar>(SpecialChar::Newline);
        }
        if (c == 't') {
            advance();
            return make_node<SpecialChar>(SpecialChar::Tab);
        }
        if (c == 'f') {
            advance();
            return make_node<SpecialChar>(SpecialChar::FormFeed);
        }
        if (c == 'v') {
            advance();
            return make_node<SpecialChar>(SpecialChar::VerticalTab);
        }
        if (c == '0') {
            advance();
            return make_node<SpecialChar>(SpecialChar::Null);
        }

        // Hex escape
        if (c == 'x') {
            advance();
            uint32_t ch = parse_hex_escape();
            return make_literal(std::string(1, static_cast<char>(ch)));
        }

        // Unicode escape
        if (c == 'u') {
            advance();
            uint32_t ch = parse_unicode_escape();
            // Convert to UTF-8
            std::string utf8;
            if (ch < 0x80) {
                utf8.push_back(static_cast<char>(ch));
            } else if (ch < 0x800) {
                utf8.push_back(static_cast<char>(0xC0 | (ch >> 6)));
                utf8.push_back(static_cast<char>(0x80 | (ch & 0x3F)));
            } else if (ch < 0x10000) {
                utf8.push_back(static_cast<char>(0xE0 | (ch >> 12)));
                utf8.push_back(static_cast<char>(0x80 | ((ch >> 6) & 0x3F)));
                utf8.push_back(static_cast<char>(0x80 | (ch & 0x3F)));
            } else {
                utf8.push_back(static_cast<char>(0xF0 | (ch >> 18)));
                utf8.push_back(static_cast<char>(0x80 | ((ch >> 12) & 0x3F)));
                utf8.push_back(static_cast<char>(0x80 | ((ch >> 6) & 0x3F)));
                utf8.push_back(static_cast<char>(0x80 | (ch & 0x3F)));
            }
            return make_literal(utf8);
        }

        // Escaped metacharacter
        advance();
        return make_literal(std::string(1, c));
    }

    UnicodeProperty parse_unicode_property_name() {
        static const std::unordered_map<std::string, UnicodeProperty> property_map = {
            {"L", UnicodeProperty::Letter},
            {"Letter", UnicodeProperty::Letter},
            {"N", UnicodeProperty::Number},
            {"Number", UnicodeProperty::Number},
            {"P", UnicodeProperty::Punctuation},
            {"Punctuation", UnicodeProperty::Punctuation},
            {"S", UnicodeProperty::Symbol},
            {"Symbol", UnicodeProperty::Symbol},
            {"M", UnicodeProperty::Mark},
            {"Mark", UnicodeProperty::Mark},
            {"Z", UnicodeProperty::Whitespace},
            {"Separator", UnicodeProperty::Whitespace},
            {"Han", UnicodeProperty::Han},
            {"Lu", UnicodeProperty::Uppercase},
            {"Uppercase_Letter", UnicodeProperty::Uppercase},
            {"Ll", UnicodeProperty::Lowercase},
            {"Lowercase_Letter", UnicodeProperty::Lowercase},
            {"Lt", UnicodeProperty::Titlecase},
            {"Titlecase_Letter", UnicodeProperty::Titlecase},
            {"Lm", UnicodeProperty::Modifier},
            {"Modifier_Letter", UnicodeProperty::Modifier},
            {"Lo", UnicodeProperty::Other},
            {"Other_Letter", UnicodeProperty::Other},
        };

        expect('{');
        std::string name;
        while (!at_end() && peek() != '}') {
            name += advance();
        }
        expect('}');

        auto it = property_map.find(name);
        if (it != property_map.end()) {
            return it->second;
        }

        // Unknown property - default to Letter as fallback
        return UnicodeProperty::Letter;
    }
};

}  // namespace pcre_parser

// Implementation of debug functions from pcre-ast.h
namespace pcre_ast {

std::string unicode_property_name(UnicodeProperty prop) {
    switch (prop) {
        case UnicodeProperty::Letter: return "Letter";
        case UnicodeProperty::Number: return "Number";
        case UnicodeProperty::Whitespace: return "Whitespace";
        case UnicodeProperty::Punctuation: return "Punctuation";
        case UnicodeProperty::Symbol: return "Symbol";
        case UnicodeProperty::Mark: return "Mark";
        case UnicodeProperty::Han: return "Han";
        case UnicodeProperty::Uppercase: return "Uppercase";
        case UnicodeProperty::Lowercase: return "Lowercase";
        case UnicodeProperty::Titlecase: return "Titlecase";
        case UnicodeProperty::Modifier: return "Modifier";
        case UnicodeProperty::Other: return "Other";
        default: return "Unknown";
    }
}

std::string node_to_string(const NodePtr& node, int indent) {
    if (!node) return std::string(indent, ' ') + "(null)\n";

    std::string prefix(indent, ' ');
    std::ostringstream oss;

    std::visit([&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;

        if constexpr (std::is_same_v<T, Literal>) {
            oss << prefix << "Literal(\"" << arg.value << "\")\n";
        }
        else if constexpr (std::is_same_v<T, AnyChar>) {
            oss << prefix << "AnyChar\n";
        }
        else if constexpr (std::is_same_v<T, CharClass>) {
            oss << prefix << "CharClass(" << (arg.negated ? "negated" : "") << ")\n";
            for (const auto& r : arg.ranges) {
                if (r.is_single()) {
                    oss << prefix << "  char: " << r.start << "\n";
                } else {
                    oss << prefix << "  range: " << r.start << "-" << r.end << "\n";
                }
            }
            for (const auto& p : arg.properties) {
                oss << prefix << "  property: " << unicode_property_name(p) << "\n";
            }
        }
        else if constexpr (std::is_same_v<T, UnicodePropertyMatch>) {
            oss << prefix << "UnicodeProperty(" << unicode_property_name(arg.property);
            if (arg.negated) oss << ", negated";
            oss << ")\n";
        }
        else if constexpr (std::is_same_v<T, WhitespaceMatch>) {
            oss << prefix << "Whitespace(" << (arg.negated ? "negated" : "") << ")\n";
        }
        else if constexpr (std::is_same_v<T, WordMatch>) {
            oss << prefix << "Word(" << (arg.negated ? "negated" : "") << ")\n";
        }
        else if constexpr (std::is_same_v<T, DigitMatch>) {
            oss << prefix << "Digit(" << (arg.negated ? "negated" : "") << ")\n";
        }
        else if constexpr (std::is_same_v<T, SpecialChar>) {
            oss << prefix << "SpecialChar(" << static_cast<int>(arg.type) << ")\n";
        }
        else if constexpr (std::is_same_v<T, Sequence>) {
            oss << prefix << "Sequence\n";
            for (const auto& child : arg.children) {
                oss << node_to_string(child, indent + 2);
            }
        }
        else if constexpr (std::is_same_v<T, Alternation>) {
            oss << prefix << "Alternation\n";
            for (const auto& alt : arg.alternatives) {
                oss << node_to_string(alt, indent + 2);
            }
        }
        else if constexpr (std::is_same_v<T, Quantifier>) {
            oss << prefix << "Quantifier{" << arg.min_count << ",";
            if (arg.max_count < 0) {
                oss << "inf";
            } else {
                oss << arg.max_count;
            }
            oss << "}" << (arg.greedy ? "" : "?") << "\n";
            oss << node_to_string(arg.child, indent + 2);
        }
        else if constexpr (std::is_same_v<T, NonCapturingGroup>) {
            oss << prefix << "NonCapturingGroup\n";
            oss << node_to_string(arg.child, indent + 2);
        }
        else if constexpr (std::is_same_v<T, CapturingGroup>) {
            oss << prefix << "CapturingGroup(" << arg.group_number << ")\n";
            oss << node_to_string(arg.child, indent + 2);
        }
        else if constexpr (std::is_same_v<T, FlagsGroup>) {
            oss << prefix << "FlagsGroup(" << (arg.case_insensitive ? "i" : "") << ")\n";
            oss << node_to_string(arg.child, indent + 2);
        }
        else if constexpr (std::is_same_v<T, PositiveLookahead>) {
            oss << prefix << "PositiveLookahead\n";
            oss << node_to_string(arg.child, indent + 2);
        }
        else if constexpr (std::is_same_v<T, NegativeLookahead>) {
            oss << prefix << "NegativeLookahead\n";
            oss << node_to_string(arg.child, indent + 2);
        }
        else if constexpr (std::is_same_v<T, PositiveLookbehind>) {
            oss << prefix << "PositiveLookbehind\n";
            oss << node_to_string(arg.child, indent + 2);
        }
        else if constexpr (std::is_same_v<T, NegativeLookbehind>) {
            oss << prefix << "NegativeLookbehind\n";
            oss << node_to_string(arg.child, indent + 2);
        }
        else if constexpr (std::is_same_v<T, StartAnchor>) {
            oss << prefix << "StartAnchor\n";
        }
        else if constexpr (std::is_same_v<T, EndAnchor>) {
            oss << prefix << "EndAnchor\n";
        }
        else if constexpr (std::is_same_v<T, WordBoundary>) {
            oss << prefix << "WordBoundary(" << (arg.negated ? "negated" : "") << ")\n";
        }
        else if constexpr (std::is_same_v<T, Empty>) {
            oss << prefix << "Empty\n";
        }
    }, node->value);

    return oss.str();
}

}  // namespace pcre_ast

namespace pcre_parser {

NodePtr parse(const std::string& pattern) {
    Parser parser(pattern);
    return parser.parse();
}

ParseResult parse_with_result(const std::string& pattern) {
    ParseResult result;
    try {
        Parser parser(pattern);
        result.ast = parser.parse();
        result.success = true;
    } catch (const ParseError& e) {
        result.success = false;
        result.error_message = e.what();
        result.error_position = e.position;
    }
    return result;
}

}  // namespace pcre_parser
