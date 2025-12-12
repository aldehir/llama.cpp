#include "pcre-parser.h"
#include "pcre-ast.h"
#include "peg-parser.h"

#include <cstdint>
#include <sstream>
#include <unordered_map>

namespace pcre_parser {

using namespace pcre_ast;

// Build the PEG grammar for PCRE regex parsing
static common_peg_arena build_pcre_grammar() {
    return build_peg_parser([](common_peg_parser_builder & p) {
        // Helper: digits
        auto digit = p.chars("[0-9]");
        auto digits = p.chars("[0-9]", 1, -1);
        auto hex_digit = p.chars("[0-9a-fA-F]");

        // Unicode property name: {L}, {Letter}, {Han}, etc.
        auto prop_name = p.rule("prop-name",
            p.literal("{") + p.chars("[a-zA-Z_]", 1, -1) + p.literal("}"));

        // Unicode property: \p{L} or \P{L}
        auto unicode_prop = p.rule("unicode-prop",
            p.literal("\\") + p.chars("[pP]") + p.ref("prop-name"));

        // Shorthand classes: \s, \S, \d, \D, \w, \W
        auto shorthand_class = p.rule("shorthand",
            p.literal("\\") + p.chars("[sSdDwWbB]"));

        // Special escape chars: \n, \r, \t, \f, \v, \0
        auto special_escape = p.rule("special-escape",
            p.literal("\\") + p.chars("[nrtfv0]"));

        // Hex escape: \xNN
        auto hex_escape = p.rule("hex-escape",
            p.literal("\\x") + p.repeat(hex_digit, 1, 2));

        // Unicode escape: \uNNNN or \u{NNNN}
        auto unicode_escape = p.rule("unicode-escape",
            p.literal("\\u") + (
                (p.literal("{") + p.chars("[0-9a-fA-F]", 1, 6) + p.literal("}")) |
                p.repeat(hex_digit, 4, 4)
            ));

        // Escaped metacharacter: \. \* \+ etc
        auto escaped_meta = p.rule("escaped-meta",
            p.literal("\\") + p.any());

        // Any escape sequence (ordered by specificity)
        auto escape = p.rule("escape",
            unicode_prop | shorthand_class | special_escape |
            hex_escape | unicode_escape | escaped_meta);

        // Character class range element: a-z or single char
        auto cc_escape = p.rule("cc-escape",
            p.literal("\\") + p.any());

        auto cc_char = p.rule("cc-char",
            cc_escape | (p.negate(p.chars("[\\]\\\\]")) + p.any()));

        auto cc_range = p.rule("cc-range",
            p.tag("start", p.ref("cc-char")) +
            p.literal("-") +
            p.tag("end", p.ref("cc-char")));

        auto cc_prop = p.rule("cc-prop",
            p.literal("\\") + p.chars("[pP]") + p.ref("prop-name"));

        auto cc_shorthand = p.rule("cc-shorthand",
            p.literal("\\") + p.chars("[sSdDwW]"));

        // Set intersection: &&[...]
        auto cc_intersection = p.rule("cc-intersection",
            p.literal("&&") + p.literal("[") +
            p.optional(p.literal("^")) +
            p.zero_or_more(p.ref("cc-element")) +
            p.literal("]"));

        auto cc_element = p.rule("cc-element",
            cc_intersection | cc_prop | cc_shorthand | cc_range | p.ref("cc-char"));

        // Character class: [abc], [^abc], [a-z]
        auto char_class = p.rule("char-class",
            p.literal("[") +
            p.tag("negated", p.optional(p.literal("^"))) +
            p.zero_or_more(p.ref("cc-element")) +
            p.literal("]"));

        // Quantifier suffix
        auto quant_range = p.rule("quant-range",
            p.literal("{") +
            p.tag("min", digits) +
            p.optional(p.literal(",") + p.tag("max", p.optional(digits))) +
            p.literal("}"));

        auto quantifier = p.rule("quantifier",
            p.chars("[*+?]") | p.ref("quant-range"));

        auto lazy_marker = p.optional(p.literal("?"));

        // Anchors
        auto start_anchor = p.rule("start-anchor", p.literal("^"));
        auto end_anchor = p.rule("end-anchor", p.literal("$"));

        // Any char
        auto any_char = p.rule("any-char", p.literal("."));

        // Literal char (not a metachar)
        auto literal_char = p.rule("literal",
            p.negate(p.chars("[|()\\[\\]{}*+?^$.\\\\]")) + p.any());

        // Forward reference for recursive grammar
        auto alternation_ref = p.ref("alternation");

        // Group types
        auto non_capturing = p.rule("non-capturing",
            p.literal("(?:") + alternation_ref + p.literal(")"));

        auto positive_lookahead = p.rule("pos-lookahead",
            p.literal("(?=") + alternation_ref + p.literal(")"));

        auto negative_lookahead = p.rule("neg-lookahead",
            p.literal("(?!") + alternation_ref + p.literal(")"));

        auto positive_lookbehind = p.rule("pos-lookbehind",
            p.literal("(?<=") + alternation_ref + p.literal(")"));

        auto negative_lookbehind = p.rule("neg-lookbehind",
            p.literal("(?<!") + alternation_ref + p.literal(")"));

        // Flags group: (?i:...), (?-i:...), (?imsx:...)
        auto flags = p.chars("[imsx-]", 0, -1);
        auto flags_group = p.rule("flags-group",
            p.literal("(?") +
            p.tag("flags", flags) +
            p.literal(":") +
            alternation_ref +
            p.literal(")"));

        // Capturing group
        auto capturing_group = p.rule("capturing",
            p.literal("(") +
            p.negate(p.literal("?")) +
            alternation_ref +
            p.literal(")"));

        // Any group type
        auto group = p.rule("group",
            flags_group | non_capturing | positive_lookahead | negative_lookahead |
            positive_lookbehind | negative_lookbehind | capturing_group);

        // Atom: the smallest matchable unit
        auto atom = p.rule("atom",
            group | char_class | escape | any_char |
            start_anchor | end_anchor | literal_char);

        // Quantified atom
        auto quantified = p.rule("quantified",
            p.tag("atom", p.ref("atom")) +
            p.optional(p.tag("quant", quantifier) + p.tag("lazy", lazy_marker)));

        // Sequence of quantified atoms
        auto sequence = p.rule("sequence",
            p.one_or_more(p.ref("quantified")));

        // Alternation: sequence | sequence | ...
        auto alternation = p.rule("alternation",
            p.ref("sequence") +
            p.zero_or_more(p.literal("|") + p.ref("sequence")));

        p.set_root(alternation);
        return alternation;
    });
}

// Global grammar instance (built once)
static common_peg_arena& get_grammar() {
    static common_peg_arena grammar = build_pcre_grammar();
    return grammar;
}

// AST converter: walks the PEG parse tree and builds pcre_ast nodes
class AstConverter {
    const common_peg_ast_arena& peg_ast_;
    const std::string& input_;
    int group_count_ = 0;

public:
    AstConverter(const common_peg_ast_arena& ast, const std::string& input)
        : peg_ast_(ast), input_(input) {}

    NodePtr convert(const common_peg_parse_result& result) {
        if (result.nodes.empty()) {
            return make_node<Empty>();
        }
        return convert_node(result.nodes[0]);
    }

private:
    NodePtr convert_node(common_peg_ast_id id) {
        const auto& node = peg_ast_.get(id);
        const std::string& rule = node.rule;
        std::string text(node.text);

        if (rule == "alternation") {
            return convert_alternation(node);
        }
        if (rule == "sequence") {
            return convert_sequence(node);
        }
        if (rule == "quantified") {
            return convert_quantified(node);
        }
        if (rule == "atom") {
            return convert_atom(node);
        }
        if (rule == "group") {
            return convert_group(node);
        }
        if (rule == "char-class") {
            return convert_char_class(node);
        }
        if (rule == "escape") {
            return convert_escape(text);
        }
        if (rule == "unicode-prop") {
            return convert_unicode_prop(text);
        }
        if (rule == "shorthand") {
            return convert_shorthand(text);
        }
        if (rule == "special-escape") {
            return convert_special_escape(text);
        }
        if (rule == "hex-escape") {
            return convert_hex_escape(text);
        }
        if (rule == "unicode-escape") {
            return convert_unicode_escape(text);
        }
        if (rule == "escaped-meta") {
            return convert_escaped_meta(text);
        }
        if (rule == "literal") {
            return make_literal(text);
        }
        if (rule == "any-char") {
            return make_any();
        }
        if (rule == "start-anchor") {
            return make_node<StartAnchor>();
        }
        if (rule == "end-anchor") {
            return make_node<EndAnchor>();
        }
        if (rule == "capturing") {
            return convert_capturing(node);
        }
        if (rule == "non-capturing") {
            return convert_non_capturing(node);
        }
        if (rule == "pos-lookahead") {
            return convert_pos_lookahead(node);
        }
        if (rule == "neg-lookahead") {
            return convert_neg_lookahead(node);
        }
        if (rule == "pos-lookbehind") {
            return convert_pos_lookbehind(node);
        }
        if (rule == "neg-lookbehind") {
            return convert_neg_lookbehind(node);
        }
        if (rule == "flags-group") {
            return convert_flags_group(node);
        }

        // For unrecognized rules, try to convert children
        if (!node.children.empty()) {
            if (node.children.size() == 1) {
                return convert_node(node.children[0]);
            }
            NodeList children;
            for (auto child_id : node.children) {
                children.push_back(convert_node(child_id));
            }
            return make_sequence(std::move(children));
        }

        // Fallback to literal
        return make_literal(text);
    }

    NodePtr convert_alternation(const common_peg_ast_node& node) {
        NodeList alternatives;
        for (auto child_id : node.children) {
            const auto& child = peg_ast_.get(child_id);
            if (child.rule == "sequence") {
                alternatives.push_back(convert_node(child_id));
            }
        }
        if (alternatives.empty()) {
            return make_node<Empty>();
        }
        return make_alternation(std::move(alternatives));
    }

    NodePtr convert_sequence(const common_peg_ast_node& node) {
        NodeList children;
        for (auto child_id : node.children) {
            children.push_back(convert_node(child_id));
        }
        return make_sequence(std::move(children));
    }

    NodePtr convert_quantified(const common_peg_ast_node& node) {
        NodePtr atom_node;
        std::string quant_text;
        bool lazy = false;

        for (auto child_id : node.children) {
            const auto& child = peg_ast_.get(child_id);
            if (child.tag == "atom" || child.rule == "atom") {
                atom_node = convert_node(child_id);
            } else if (child.tag == "quant" || child.rule == "quantifier" || child.rule == "quant-range") {
                quant_text = std::string(child.text);
            } else if (child.tag == "lazy") {
                lazy = !child.text.empty();
            } else if (!child.children.empty()) {
                // Nested structure - recurse
                for (auto nested_id : child.children) {
                    const auto& nested = peg_ast_.get(nested_id);
                    if (nested.tag == "atom" || nested.rule == "atom") {
                        atom_node = convert_node(nested_id);
                    } else if (nested.tag == "quant") {
                        quant_text = std::string(nested.text);
                    } else if (nested.tag == "lazy") {
                        lazy = !nested.text.empty();
                    }
                }
            }
        }

        if (!atom_node) {
            // No separate atom found, convert the whole node text
            atom_node = convert_atom_text(std::string(node.text));
        }

        if (quant_text.empty()) {
            return atom_node;
        }

        return apply_quantifier(atom_node, quant_text, !lazy);
    }

    NodePtr convert_atom_text(const std::string& text) {
        if (text.empty()) return make_node<Empty>();
        if (text == ".") return make_any();
        if (text == "^") return make_node<StartAnchor>();
        if (text == "$") return make_node<EndAnchor>();
        if (text[0] == '\\') return convert_escape(text);
        if (text[0] == '[') {
            // Would need to re-parse char class - simplified
            return make_literal(text);
        }
        return make_literal(text);
    }

    NodePtr convert_atom(const common_peg_ast_node& node) {
        if (!node.children.empty()) {
            return convert_node(node.children[0]);
        }
        return convert_atom_text(std::string(node.text));
    }

    NodePtr apply_quantifier(NodePtr child, const std::string& quant, bool greedy) {
        if (quant == "*") return make_star(child, greedy);
        if (quant == "+") return make_plus(child, greedy);
        if (quant == "?") return make_optional(child, greedy);

        // Parse {n}, {n,}, {n,m}
        if (quant[0] == '{') {
            size_t comma = quant.find(',');
            size_t close = quant.find('}');
            int min_count = 0, max_count = 0;

            if (comma == std::string::npos) {
                // {n}
                min_count = max_count = std::stoi(quant.substr(1, close - 1));
            } else {
                // {n,} or {n,m}
                min_count = std::stoi(quant.substr(1, comma - 1));
                if (comma + 1 == close) {
                    max_count = -1;  // Unbounded
                } else {
                    max_count = std::stoi(quant.substr(comma + 1, close - comma - 1));
                }
            }
            return make_quantifier(child, min_count, max_count, greedy);
        }

        return child;
    }

    NodePtr convert_group(const common_peg_ast_node& node) {
        if (!node.children.empty()) {
            return convert_node(node.children[0]);
        }
        return make_node<Empty>();
    }

    NodePtr convert_capturing(const common_peg_ast_node& node) {
        int group_num = ++group_count_;
        NodePtr child = make_node<Empty>();

        for (auto child_id : node.children) {
            const auto& c = peg_ast_.get(child_id);
            if (c.rule == "alternation") {
                child = convert_node(child_id);
                break;
            }
        }

        auto g = CapturingGroup{};
        g.child = child;
        g.group_number = group_num;
        return std::make_shared<Node>(g);
    }

    NodePtr convert_non_capturing(const common_peg_ast_node& node) {
        NodePtr child = make_node<Empty>();
        for (auto child_id : node.children) {
            const auto& c = peg_ast_.get(child_id);
            if (c.rule == "alternation") {
                child = convert_node(child_id);
                break;
            }
        }
        return make_non_capturing_group(child);
    }

    NodePtr convert_pos_lookahead(const common_peg_ast_node& node) {
        NodePtr child = make_node<Empty>();
        for (auto child_id : node.children) {
            const auto& c = peg_ast_.get(child_id);
            if (c.rule == "alternation") {
                child = convert_node(child_id);
                break;
            }
        }
        return make_positive_lookahead(child);
    }

    NodePtr convert_neg_lookahead(const common_peg_ast_node& node) {
        NodePtr child = make_node<Empty>();
        for (auto child_id : node.children) {
            const auto& c = peg_ast_.get(child_id);
            if (c.rule == "alternation") {
                child = convert_node(child_id);
                break;
            }
        }
        return make_negative_lookahead(child);
    }

    NodePtr convert_pos_lookbehind(const common_peg_ast_node& node) {
        NodePtr child = make_node<Empty>();
        for (auto child_id : node.children) {
            const auto& c = peg_ast_.get(child_id);
            if (c.rule == "alternation") {
                child = convert_node(child_id);
                break;
            }
        }
        auto lb = PositiveLookbehind{};
        lb.child = child;
        return std::make_shared<Node>(lb);
    }

    NodePtr convert_neg_lookbehind(const common_peg_ast_node& node) {
        NodePtr child = make_node<Empty>();
        for (auto child_id : node.children) {
            const auto& c = peg_ast_.get(child_id);
            if (c.rule == "alternation") {
                child = convert_node(child_id);
                break;
            }
        }
        auto lb = NegativeLookbehind{};
        lb.child = child;
        return std::make_shared<Node>(lb);
    }

    NodePtr convert_flags_group(const common_peg_ast_node& node) {
        bool case_insensitive = false;
        NodePtr child = make_node<Empty>();

        for (auto child_id : node.children) {
            const auto& c = peg_ast_.get(child_id);
            if (c.tag == "flags") {
                std::string flags_str(c.text);
                case_insensitive = (flags_str.find('i') != std::string::npos);
            } else if (c.rule == "alternation") {
                child = convert_node(child_id);
            }
        }
        return make_flags_group(child, case_insensitive);
    }

    NodePtr convert_char_class(const common_peg_ast_node& node) {
        CharClass cc;
        std::string text(node.text);

        // Simple parsing of the character class text
        size_t pos = 1;  // Skip opening [
        if (pos < text.size() && text[pos] == '^') {
            cc.negated = true;
            pos++;
        }

        while (pos < text.size() && text[pos] != ']') {
            if (text[pos] == '\\' && pos + 1 < text.size()) {
                char next = text[pos + 1];
                if (next == 'p' || next == 'P') {
                    // Unicode property
                    bool negated = (next == 'P');
                    pos += 2;
                    if (pos < text.size() && text[pos] == '{') {
                        size_t end = text.find('}', pos);
                        if (end != std::string::npos) {
                            std::string prop_name = text.substr(pos + 1, end - pos - 1);
                            auto prop = get_unicode_property(prop_name);
                            if (negated) {
                                cc.excluded_properties.push_back(prop);
                            } else {
                                cc.properties.push_back(prop);
                            }
                            pos = end + 1;
                            continue;
                        }
                    }
                } else if (next == 's') {
                    cc.properties.push_back(UnicodeProperty::Whitespace);
                    pos += 2;
                    continue;
                } else if (next == 'd') {
                    cc.properties.push_back(UnicodeProperty::Number);
                    pos += 2;
                    continue;
                } else if (next == 'w') {
                    cc.properties.push_back(UnicodeProperty::Letter);
                    pos += 2;
                    continue;
                } else {
                    // Escaped char
                    uint32_t ch = parse_escape_char(text, pos);
                    cc.ranges.push_back(CharRange(ch));
                    cc.includes_specific_chars = true;
                    continue;
                }
            }

            // Check for && intersection
            if (text[pos] == '&' && pos + 1 < text.size() && text[pos + 1] == '&') {
                pos += 2;
                // Skip the intersected set for now
                if (pos < text.size() && text[pos] == '[') {
                    int depth = 1;
                    pos++;
                    while (pos < text.size() && depth > 0) {
                        if (text[pos] == '[') depth++;
                        else if (text[pos] == ']') depth--;
                        pos++;
                    }
                }
                continue;
            }

            // Regular char or range
            uint32_t start_ch = static_cast<uint32_t>(static_cast<unsigned char>(text[pos]));
            pos++;

            if (pos < text.size() && text[pos] == '-' && pos + 1 < text.size() && text[pos + 1] != ']') {
                pos++;  // Skip -
                uint32_t end_ch = static_cast<uint32_t>(static_cast<unsigned char>(text[pos]));
                pos++;
                cc.ranges.push_back(CharRange(start_ch, end_ch));
            } else {
                cc.ranges.push_back(CharRange(start_ch));
            }
            cc.includes_specific_chars = true;
        }

        return std::make_shared<Node>(cc);
    }

    uint32_t parse_escape_char(const std::string& text, size_t& pos) {
        pos++;  // Skip backslash
        if (pos >= text.size()) return '\\';

        char c = text[pos++];
        switch (c) {
            case 'n': return '\n';
            case 'r': return '\r';
            case 't': return '\t';
            case 'f': return '\f';
            case 'v': return '\v';
            case '0': return '\0';
            case 'x': {
                uint32_t result = 0;
                for (int i = 0; i < 2 && pos < text.size(); ++i) {
                    char h = text[pos];
                    if ((h >= '0' && h <= '9') || (h >= 'a' && h <= 'f') || (h >= 'A' && h <= 'F')) {
                        result = result * 16 + (h >= 'a' ? h - 'a' + 10 : h >= 'A' ? h - 'A' + 10 : h - '0');
                        pos++;
                    } else break;
                }
                return result;
            }
            default: return static_cast<uint32_t>(static_cast<unsigned char>(c));
        }
    }

    NodePtr convert_escape(const std::string& text) {
        if (text.size() < 2) return make_literal(text);

        char c = text[1];

        // Unicode property
        if ((c == 'p' || c == 'P') && text.size() > 3 && text[2] == '{') {
            return convert_unicode_prop(text);
        }

        // Shorthand classes
        if (c == 's') return make_node<WhitespaceMatch>(false);
        if (c == 'S') return make_node<WhitespaceMatch>(true);
        if (c == 'd') return make_node<DigitMatch>(false);
        if (c == 'D') return make_node<DigitMatch>(true);
        if (c == 'w') return make_node<WordMatch>(false);
        if (c == 'W') return make_node<WordMatch>(true);
        if (c == 'b') return make_node<WordBoundary>(false);
        if (c == 'B') return make_node<WordBoundary>(true);

        // Special chars
        if (c == 'r') return make_node<SpecialChar>(SpecialChar::CarriageReturn);
        if (c == 'n') return make_node<SpecialChar>(SpecialChar::Newline);
        if (c == 't') return make_node<SpecialChar>(SpecialChar::Tab);
        if (c == 'f') return make_node<SpecialChar>(SpecialChar::FormFeed);
        if (c == 'v') return make_node<SpecialChar>(SpecialChar::VerticalTab);
        if (c == '0') return make_node<SpecialChar>(SpecialChar::Null);

        // Hex escape
        if (c == 'x') return convert_hex_escape(text);

        // Unicode escape
        if (c == 'u') return convert_unicode_escape(text);

        // Escaped metachar
        return make_literal(std::string(1, c));
    }

    NodePtr convert_unicode_prop(const std::string& text) {
        bool negated = (text[1] == 'P');
        size_t start = text.find('{');
        size_t end = text.find('}');
        if (start != std::string::npos && end != std::string::npos) {
            std::string name = text.substr(start + 1, end - start - 1);
            auto prop = get_unicode_property(name);
            return make_unicode_property(prop, negated);
        }
        return make_literal(text);
    }

    NodePtr convert_shorthand(const std::string& text) {
        if (text.size() < 2) return make_literal(text);
        return convert_escape(text);
    }

    NodePtr convert_special_escape(const std::string& text) {
        return convert_escape(text);
    }

    NodePtr convert_hex_escape(const std::string& text) {
        if (text.size() < 3) return make_literal(text);
        uint32_t val = 0;
        for (size_t i = 2; i < text.size(); i++) {
            char h = text[i];
            if ((h >= '0' && h <= '9') || (h >= 'a' && h <= 'f') || (h >= 'A' && h <= 'F')) {
                val = val * 16 + (h >= 'a' ? h - 'a' + 10 : h >= 'A' ? h - 'A' + 10 : h - '0');
            }
        }
        return make_literal(std::string(1, static_cast<char>(val)));
    }

    NodePtr convert_unicode_escape(const std::string& text) {
        uint32_t val = 0;
        size_t start = 2;
        if (text.size() > 3 && text[2] == '{') {
            start = 3;
            for (size_t i = start; i < text.size() && text[i] != '}'; i++) {
                char h = text[i];
                if ((h >= '0' && h <= '9') || (h >= 'a' && h <= 'f') || (h >= 'A' && h <= 'F')) {
                    val = val * 16 + (h >= 'a' ? h - 'a' + 10 : h >= 'A' ? h - 'A' + 10 : h - '0');
                }
            }
        } else {
            for (size_t i = start; i < text.size() && i < start + 4; i++) {
                char h = text[i];
                if ((h >= '0' && h <= '9') || (h >= 'a' && h <= 'f') || (h >= 'A' && h <= 'F')) {
                    val = val * 16 + (h >= 'a' ? h - 'a' + 10 : h >= 'A' ? h - 'A' + 10 : h - '0');
                }
            }
        }

        // Convert to UTF-8
        std::string utf8;
        if (val < 0x80) {
            utf8.push_back(static_cast<char>(val));
        } else if (val < 0x800) {
            utf8.push_back(static_cast<char>(0xC0 | (val >> 6)));
            utf8.push_back(static_cast<char>(0x80 | (val & 0x3F)));
        } else if (val < 0x10000) {
            utf8.push_back(static_cast<char>(0xE0 | (val >> 12)));
            utf8.push_back(static_cast<char>(0x80 | ((val >> 6) & 0x3F)));
            utf8.push_back(static_cast<char>(0x80 | (val & 0x3F)));
        } else {
            utf8.push_back(static_cast<char>(0xF0 | (val >> 18)));
            utf8.push_back(static_cast<char>(0x80 | ((val >> 12) & 0x3F)));
            utf8.push_back(static_cast<char>(0x80 | ((val >> 6) & 0x3F)));
            utf8.push_back(static_cast<char>(0x80 | (val & 0x3F)));
        }
        return make_literal(utf8);
    }

    NodePtr convert_escaped_meta(const std::string& text) {
        if (text.size() < 2) return make_literal(text);
        return make_literal(std::string(1, text[1]));
    }

    static UnicodeProperty get_unicode_property(const std::string& name) {
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

        auto it = property_map.find(name);
        if (it != property_map.end()) {
            return it->second;
        }
        return UnicodeProperty::Letter;  // Default fallback
    }
};

// Public API implementation
NodePtr parse(const std::string& pattern) {
    auto& grammar = get_grammar();
    common_peg_parse_context ctx(pattern);
    auto result = grammar.parse(ctx);

    if (result.fail()) {
        throw ParseError("Parse failed", result.end, pattern.substr(result.end, 20));
    }

    AstConverter converter(ctx.ast, pattern);
    return converter.convert(result);
}

ParseResult parse_with_result(const std::string& pattern) {
    ParseResult result;
    try {
        result.ast = parse(pattern);
        result.success = true;
    } catch (const ParseError& e) {
        result.success = false;
        result.error_message = e.what();
        result.error_position = e.position;
    }
    return result;
}

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
