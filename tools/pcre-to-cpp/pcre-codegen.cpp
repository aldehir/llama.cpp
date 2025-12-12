#include "pcre-codegen.h"
#include "pcre-ast.h"

#include <cstdio>
#include <sstream>
#include <unordered_set>

namespace pcre_codegen {

using namespace pcre_ast;

// Code generator that produces standalone C++ regex matching code
class CodeGenerator {
    const CodegenOptions& opts_;
    std::ostringstream out_;
    int indent_level_ = 0;
    int label_counter_ = 0;
    std::unordered_set<std::string> needed_helpers_;

public:
    explicit CodeGenerator(const CodegenOptions& opts) : opts_(opts) {}

    std::string generate(const NodePtr& ast) {
        // Analyze AST to determine needed helpers
        analyze_needed_helpers(ast);

        // Always need cpt_len for the main split loop
        needed_helpers_.insert("cpt_from_utf8");

        // Generate header
        if (opts_.generate_header) {
            generate_header();
        }

        // Generate unicode helpers if needed
        if (opts_.inline_unicode_helpers) {
            generate_unicode_helpers();
        }

        // Generate the main split function
        generate_split_function(ast);

        // Generate main if requested
        if (opts_.generate_main) {
            generate_main();
        }

        return out_.str();
    }

    std::string generate_matcher_body(const NodePtr& ast) {
        analyze_needed_helpers(ast);
        std::ostringstream body;
        // Generate just the inner matching logic
        generate_match_logic(ast, body, 0);
        return body.str();
    }

private:
    void indent() { indent_level_++; }
    void dedent() { indent_level_--; }

    std::string ind() const {
        std::string result;
        for (int i = 0; i < indent_level_; i++) {
            result += opts_.indent;
        }
        return result;
    }

    void line(const std::string& s = "") {
        if (s.empty()) {
            out_ << "\n";
        } else {
            out_ << ind() << s << "\n";
        }
    }

    void analyze_needed_helpers(const NodePtr& node) {
        if (!node) return;

        std::visit([this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;

            if constexpr (std::is_same_v<T, UnicodePropertyMatch>) {
                needed_helpers_.insert("unicode_flags");
                needed_helpers_.insert("cpt_from_utf8");
            }
            else if constexpr (std::is_same_v<T, WhitespaceMatch>) {
                needed_helpers_.insert("unicode_flags");
                needed_helpers_.insert("cpt_from_utf8");
            }
            else if constexpr (std::is_same_v<T, DigitMatch>) {
                needed_helpers_.insert("unicode_flags");
                needed_helpers_.insert("cpt_from_utf8");
            }
            else if constexpr (std::is_same_v<T, WordMatch>) {
                needed_helpers_.insert("unicode_flags");
                needed_helpers_.insert("cpt_from_utf8");
            }
            else if constexpr (std::is_same_v<T, CharClass>) {
                if (!arg.properties.empty() || !arg.excluded_properties.empty()) {
                    needed_helpers_.insert("unicode_flags");
                }
                needed_helpers_.insert("cpt_from_utf8");
                if (arg.properties.size() > 0) {
                    for (auto p : arg.properties) {
                        if (p == UnicodeProperty::Han) {
                            needed_helpers_.insert("is_han");
                        }
                    }
                }
            }
            else if constexpr (std::is_same_v<T, FlagsGroup>) {
                if (arg.case_insensitive) {
                    needed_helpers_.insert("tolower");
                }
                analyze_needed_helpers(arg.child);
            }
            else if constexpr (std::is_same_v<T, Sequence>) {
                for (const auto& child : arg.children) {
                    analyze_needed_helpers(child);
                }
            }
            else if constexpr (std::is_same_v<T, Alternation>) {
                for (const auto& alt : arg.alternatives) {
                    analyze_needed_helpers(alt);
                }
            }
            else if constexpr (std::is_same_v<T, Quantifier>) {
                analyze_needed_helpers(arg.child);
            }
            else if constexpr (std::is_same_v<T, NonCapturingGroup>) {
                analyze_needed_helpers(arg.child);
            }
            else if constexpr (std::is_same_v<T, CapturingGroup>) {
                analyze_needed_helpers(arg.child);
            }
            else if constexpr (std::is_same_v<T, PositiveLookahead>) {
                analyze_needed_helpers(arg.child);
            }
            else if constexpr (std::is_same_v<T, NegativeLookahead>) {
                analyze_needed_helpers(arg.child);
            }
            else if constexpr (std::is_same_v<T, AnyChar>) {
                needed_helpers_.insert("cpt_from_utf8");
            }
        }, node->value);
    }

    void generate_header() {
        line("#include <cstdint>");
        line("#include <cstddef>");
        line("#include <string>");
        line("#include <vector>");
        line();

        if (!opts_.namespace_name.empty()) {
            line("namespace " + opts_.namespace_name + " {");
            line();
        }
    }

    void generate_unicode_helpers() {
        // UTF-8 decoding
        if (needed_helpers_.count("cpt_from_utf8")) {
            line("// UTF-8 codepoint extraction");
            line("static inline uint32_t cpt_from_utf8(const char* s, size_t len, size_t& pos) {");
            indent();
            line("if (pos >= len) return 0xFFFFFFFF;");
            line("uint8_t c = static_cast<uint8_t>(s[pos]);");
            line("if ((c & 0x80) == 0) {");
            indent();
            line("pos++;");
            line("return c;");
            dedent();
            line("}");
            line("if ((c & 0xE0) == 0xC0 && pos + 1 < len) {");
            indent();
            line("uint32_t result = (c & 0x1F) << 6;");
            line("result |= (static_cast<uint8_t>(s[pos + 1]) & 0x3F);");
            line("pos += 2;");
            line("return result;");
            dedent();
            line("}");
            line("if ((c & 0xF0) == 0xE0 && pos + 2 < len) {");
            indent();
            line("uint32_t result = (c & 0x0F) << 12;");
            line("result |= (static_cast<uint8_t>(s[pos + 1]) & 0x3F) << 6;");
            line("result |= (static_cast<uint8_t>(s[pos + 2]) & 0x3F);");
            line("pos += 3;");
            line("return result;");
            dedent();
            line("}");
            line("if ((c & 0xF8) == 0xF0 && pos + 3 < len) {");
            indent();
            line("uint32_t result = (c & 0x07) << 18;");
            line("result |= (static_cast<uint8_t>(s[pos + 1]) & 0x3F) << 12;");
            line("result |= (static_cast<uint8_t>(s[pos + 2]) & 0x3F) << 6;");
            line("result |= (static_cast<uint8_t>(s[pos + 3]) & 0x3F);");
            line("pos += 4;");
            line("return result;");
            dedent();
            line("}");
            line("pos++;");
            line("return 0xFFFD; // Replacement character");
            dedent();
            line("}");
            line();

            line("// Peek at codepoint without advancing");
            line("static inline uint32_t peek_cpt(const char* s, size_t len, size_t pos) {");
            indent();
            line("size_t tmp = pos;");
            line("return cpt_from_utf8(s, len, tmp);");
            dedent();
            line("}");
            line();

            line("// Get UTF-8 byte length for codepoint at position");
            line("static inline size_t cpt_len(const char* s, size_t len, size_t pos) {");
            indent();
            line("if (pos >= len) return 0;");
            line("uint8_t c = static_cast<uint8_t>(s[pos]);");
            line("if ((c & 0x80) == 0) return 1;");
            line("if ((c & 0xE0) == 0xC0) return 2;");
            line("if ((c & 0xF0) == 0xE0) return 3;");
            line("if ((c & 0xF8) == 0xF0) return 4;");
            line("return 1;");
            dedent();
            line("}");
            line();
        }

        // Unicode property flags
        if (needed_helpers_.count("unicode_flags")) {
            line("// Unicode codepoint flags");
            line("struct CptFlags {");
            indent();
            line("bool is_letter = false;");
            line("bool is_number = false;");
            line("bool is_whitespace = false;");
            line("bool is_punctuation = false;");
            line("bool is_symbol = false;");
            line("bool is_mark = false;");
            line("bool is_uppercase = false;");
            line("bool is_lowercase = false;");
            line();
            line("uint32_t as_uint() const {");
            indent();
            line("return (is_letter ? 1 : 0) | (is_number ? 2 : 0) |");
            line("       (is_whitespace ? 4 : 0) | (is_punctuation ? 8 : 0) |");
            line("       (is_symbol ? 16 : 0) | (is_mark ? 32 : 0);");
            dedent();
            line("}");
            dedent();
            line("};");
            line();

            line("// Get flags for a codepoint (simplified - full impl would use tables)");
            line("static inline CptFlags get_cpt_flags(uint32_t cpt) {");
            indent();
            line("CptFlags f;");
            line("// ASCII letters");
            line("if ((cpt >= 'A' && cpt <= 'Z') || (cpt >= 'a' && cpt <= 'z')) {");
            indent();
            line("f.is_letter = true;");
            line("f.is_uppercase = (cpt >= 'A' && cpt <= 'Z');");
            line("f.is_lowercase = (cpt >= 'a' && cpt <= 'z');");
            dedent();
            line("}");
            line("// ASCII digits");
            line("else if (cpt >= '0' && cpt <= '9') {");
            indent();
            line("f.is_number = true;");
            dedent();
            line("}");
            line("// ASCII whitespace");
            line("else if (cpt == ' ' || cpt == '\\t' || cpt == '\\n' || cpt == '\\r' || cpt == '\\f' || cpt == '\\v') {");
            indent();
            line("f.is_whitespace = true;");
            dedent();
            line("}");
            line("// ASCII punctuation");
            line("else if ((cpt >= '!' && cpt <= '/') || (cpt >= ':' && cpt <= '@') ||");
            line("         (cpt >= '[' && cpt <= '`') || (cpt >= '{' && cpt <= '~')) {");
            indent();
            line("f.is_punctuation = true;");
            dedent();
            line("}");
            line("// Extended Latin letters (Latin-1 Supplement)");
            line("else if ((cpt >= 0xC0 && cpt <= 0xD6) || (cpt >= 0xD8 && cpt <= 0xF6) || (cpt >= 0xF8 && cpt <= 0xFF)) {");
            indent();
            line("f.is_letter = true;");
            line("f.is_uppercase = (cpt >= 0xC0 && cpt <= 0xD6) || (cpt >= 0xD8 && cpt <= 0xDE);");
            line("f.is_lowercase = (cpt >= 0xDF && cpt <= 0xF6) || (cpt >= 0xF8 && cpt <= 0xFF);");
            dedent();
            line("}");
            line("// Latin Extended-A");
            line("else if (cpt >= 0x100 && cpt <= 0x17F) {");
            indent();
            line("f.is_letter = true;");
            dedent();
            line("}");
            line("// Latin Extended-B");
            line("else if (cpt >= 0x180 && cpt <= 0x24F) {");
            indent();
            line("f.is_letter = true;");
            dedent();
            line("}");
            line("// Greek and Coptic");
            line("else if (cpt >= 0x370 && cpt <= 0x3FF) {");
            indent();
            line("f.is_letter = true;");
            dedent();
            line("}");
            line("// Cyrillic");
            line("else if (cpt >= 0x400 && cpt <= 0x4FF) {");
            indent();
            line("f.is_letter = true;");
            dedent();
            line("}");
            line("// Arabic");
            line("else if (cpt >= 0x600 && cpt <= 0x6FF) {");
            indent();
            line("f.is_letter = true;");
            dedent();
            line("}");
            line("// Hebrew");
            line("else if (cpt >= 0x590 && cpt <= 0x5FF) {");
            indent();
            line("f.is_letter = true;");
            dedent();
            line("}");
            line("// Hiragana");
            line("else if (cpt >= 0x3040 && cpt <= 0x309F) {");
            indent();
            line("f.is_letter = true;");
            dedent();
            line("}");
            line("// Katakana");
            line("else if (cpt >= 0x30A0 && cpt <= 0x30FF) {");
            indent();
            line("f.is_letter = true;");
            dedent();
            line("}");
            line("// Non-breaking space and other Unicode whitespace");
            line("else if (cpt == 0xA0 || cpt == 0x1680 || (cpt >= 0x2000 && cpt <= 0x200A) ||");
            line("         cpt == 0x2028 || cpt == 0x2029 || cpt == 0x202F || cpt == 0x205F || cpt == 0x3000) {");
            indent();
            line("f.is_whitespace = true;");
            dedent();
            line("}");
            line("// General category detection for higher codepoints");
            line("else if (cpt >= 0x10000) {");
            indent();
            line("// SMP letters (supplementary)");
            line("if ((cpt >= 0x10000 && cpt <= 0x1007F) || // Linear B Syllabary");
            line("    (cpt >= 0x10080 && cpt <= 0x100FF) || // Linear B Ideograms");
            line("    (cpt >= 0x10300 && cpt <= 0x1032F) || // Old Italic");
            line("    (cpt >= 0x10400 && cpt <= 0x1044F) || // Deseret");
            line("    (cpt >= 0x1D400 && cpt <= 0x1D7FF)) { // Mathematical Alphanumeric");
            indent();
            line("f.is_letter = true;");
            dedent();
            line("}");
            dedent();
            line("}");
            line("return f;");
            dedent();
            line("}");
            line();
        }

        // Han character detection
        if (needed_helpers_.count("is_han")) {
            line("// Check if codepoint is Han (CJK)");
            line("static inline bool is_han(uint32_t cpt) {");
            indent();
            line("// CJK Unified Ideographs");
            line("if (cpt >= 0x4E00 && cpt <= 0x9FFF) return true;");
            line("// CJK Extension A");
            line("if (cpt >= 0x3400 && cpt <= 0x4DBF) return true;");
            line("// CJK Extension B");
            line("if (cpt >= 0x20000 && cpt <= 0x2A6DF) return true;");
            line("// CJK Extension C");
            line("if (cpt >= 0x2A700 && cpt <= 0x2B73F) return true;");
            line("// CJK Extension D");
            line("if (cpt >= 0x2B740 && cpt <= 0x2B81F) return true;");
            line("// CJK Extension E");
            line("if (cpt >= 0x2B820 && cpt <= 0x2CEAF) return true;");
            line("// CJK Extension F");
            line("if (cpt >= 0x2CEB0 && cpt <= 0x2EBEF) return true;");
            line("// CJK Compatibility Ideographs");
            line("if (cpt >= 0xF900 && cpt <= 0xFAFF) return true;");
            line("// CJK Compatibility Ideographs Supplement");
            line("if (cpt >= 0x2F800 && cpt <= 0x2FA1F) return true;");
            line("return false;");
            dedent();
            line("}");
            line();
        }

        // Case conversion
        if (needed_helpers_.count("tolower")) {
            line("// Simple ASCII tolower");
            line("static inline uint32_t to_lower(uint32_t cpt) {");
            indent();
            line("if (cpt >= 'A' && cpt <= 'Z') return cpt + 32;");
            line("return cpt;");
            dedent();
            line("}");
            line();
        }
    }

    void generate_split_function(const NodePtr& ast) {
        line("// Generated regex splitter");
        line("// Splits input string into tokens based on regex matches");
        line("std::vector<std::string> " + opts_.function_name + "(const std::string& text) {");
        indent();
        line("std::vector<std::string> tokens;");
        line("const char* s = text.c_str();");
        line("const size_t len = text.size();");
        line("size_t pos = 0;");
        line("size_t token_start = 0;");
        line();

        line("while (pos < len) {");
        indent();
        line("size_t match_start = pos;");
        line("size_t match_end = pos;");
        line("bool matched = false;");
        line();

        // Generate the matching code
        generate_match_toplevel(ast);

        line();
        line("if (matched && match_end > match_start) {");
        indent();
        line("// Add any unmatched prefix as a token");
        line("if (token_start < match_start) {");
        indent();
        line("tokens.push_back(text.substr(token_start, match_start - token_start));");
        dedent();
        line("}");
        line("// Add the matched token");
        line("tokens.push_back(text.substr(match_start, match_end - match_start));");
        line("token_start = match_end;");
        line("pos = match_end;");
        dedent();
        line("} else {");
        indent();
        line("// No match, advance by one codepoint");
        line("pos += cpt_len(s, len, pos);");
        dedent();
        line("}");
        dedent();
        line("}");
        line();

        line("// Add any remaining text as final token");
        line("if (token_start < len) {");
        indent();
        line("tokens.push_back(text.substr(token_start));");
        dedent();
        line("}");
        line();
        line("return tokens;");
        dedent();
        line("}");
        line();
    }

    void generate_match_toplevel(const NodePtr& ast) {
        // For alternation at top level, we try each branch
        if (ast->is<Alternation>()) {
            const auto& alt = ast->as<Alternation>();
            line("// Try each alternative");
            for (size_t i = 0; i < alt.alternatives.size(); i++) {
                line("{");
                indent();
                line("size_t try_pos = pos;");
                std::ostringstream match_code;
                generate_match_code(alt.alternatives[i], match_code, "try_pos", "s", "len");
                out_ << match_code.str();
                line("if (try_pos > pos) {");
                indent();
                line("match_end = try_pos;");
                line("matched = true;");
                dedent();
                line("}");
                dedent();
                line("}");
                if (i < alt.alternatives.size() - 1) {
                    line("if (!matched) {");
                    indent();
                }
            }
            // Close all the if (!matched) blocks
            for (size_t i = 1; i < alt.alternatives.size(); i++) {
                dedent();
                line("}");
            }
        } else {
            line("{");
            indent();
            line("size_t try_pos = pos;");
            std::ostringstream match_code;
            generate_match_code(ast, match_code, "try_pos", "s", "len");
            out_ << match_code.str();
            line("if (try_pos > pos) {");
            indent();
            line("match_end = try_pos;");
            line("matched = true;");
            dedent();
            line("}");
            dedent();
            line("}");
        }
    }

    void generate_match_code(const NodePtr& node, std::ostringstream& out,
                             const std::string& pos_var, const std::string& str_var,
                             const std::string& len_var) {
        if (!node) return;

        std::visit([&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;

            if constexpr (std::is_same_v<T, Literal>) {
                generate_literal_match(arg, out, pos_var, str_var, len_var);
            }
            else if constexpr (std::is_same_v<T, AnyChar>) {
                out << ind() << "if (" << pos_var << " < " << len_var << ") {\n";
                out << ind() << opts_.indent << pos_var << " += cpt_len(" << str_var << ", " << len_var << ", " << pos_var << ");\n";
                out << ind() << "}\n";
            }
            else if constexpr (std::is_same_v<T, CharClass>) {
                generate_char_class_match(arg, out, pos_var, str_var, len_var);
            }
            else if constexpr (std::is_same_v<T, UnicodePropertyMatch>) {
                generate_unicode_property_match(arg, out, pos_var, str_var, len_var);
            }
            else if constexpr (std::is_same_v<T, WhitespaceMatch>) {
                generate_whitespace_match(arg, out, pos_var, str_var, len_var);
            }
            else if constexpr (std::is_same_v<T, DigitMatch>) {
                generate_digit_match(arg, out, pos_var, str_var, len_var);
            }
            else if constexpr (std::is_same_v<T, WordMatch>) {
                generate_word_match(arg, out, pos_var, str_var, len_var);
            }
            else if constexpr (std::is_same_v<T, SpecialChar>) {
                generate_special_char_match(arg, out, pos_var, str_var, len_var);
            }
            else if constexpr (std::is_same_v<T, Sequence>) {
                generate_sequence_match(arg, out, pos_var, str_var, len_var);
            }
            else if constexpr (std::is_same_v<T, Alternation>) {
                generate_alternation_match(arg, out, pos_var, str_var, len_var);
            }
            else if constexpr (std::is_same_v<T, Quantifier>) {
                generate_quantifier_match(arg, out, pos_var, str_var, len_var);
            }
            else if constexpr (std::is_same_v<T, NonCapturingGroup>) {
                generate_match_code(arg.child, out, pos_var, str_var, len_var);
            }
            else if constexpr (std::is_same_v<T, CapturingGroup>) {
                generate_match_code(arg.child, out, pos_var, str_var, len_var);
            }
            else if constexpr (std::is_same_v<T, FlagsGroup>) {
                // For case-insensitive, we'd need to modify literal matching
                // For now, just process the child
                generate_match_code(arg.child, out, pos_var, str_var, len_var);
            }
            else if constexpr (std::is_same_v<T, PositiveLookahead>) {
                generate_positive_lookahead(arg, out, pos_var, str_var, len_var);
            }
            else if constexpr (std::is_same_v<T, NegativeLookahead>) {
                generate_negative_lookahead(arg, out, pos_var, str_var, len_var);
            }
            else if constexpr (std::is_same_v<T, Empty>) {
                // Empty matches nothing, advances nothing
            }
            else if constexpr (std::is_same_v<T, StartAnchor>) {
                out << ind() << "// Start anchor - only match at start\n";
                out << ind() << "if (" << pos_var << " != 0) { " << pos_var << " = " << len_var << " + 1; }\n";
            }
            else if constexpr (std::is_same_v<T, EndAnchor>) {
                out << ind() << "// End anchor - only match at end\n";
                out << ind() << "if (" << pos_var << " != " << len_var << ") { " << pos_var << " = " << len_var << " + 1; }\n";
            }
        }, node->value);
    }

    void generate_literal_match(const Literal& lit, std::ostringstream& out,
                                const std::string& pos_var, const std::string& str_var,
                                const std::string& len_var) {
        out << ind() << "// Match literal \"" << escape_string(lit.value) << "\"\n";
        out << ind() << "if (" << pos_var << " + " << lit.value.size() << " <= " << len_var;
        for (size_t i = 0; i < lit.value.size(); i++) {
            out << " && " << str_var << "[" << pos_var << " + " << i << "] == ";
            out << "'" << escape_char(lit.value[i]) << "'";
        }
        out << ") {\n";
        out << ind() << opts_.indent << pos_var << " += " << lit.value.size() << ";\n";
        out << ind() << "}\n";
    }

    void generate_char_class_match(const CharClass& cc, std::ostringstream& out,
                                   const std::string& pos_var, const std::string& str_var,
                                   const std::string& len_var) {
        out << ind() << "// Match character class\n";
        out << ind() << "if (" << pos_var << " < " << len_var << ") {\n";
        indent();
        out << ind() << "uint32_t cpt = peek_cpt(" << str_var << ", " << len_var << ", " << pos_var << ");\n";

        if (!cc.properties.empty() || !cc.excluded_properties.empty()) {
            out << ind() << "CptFlags flags = get_cpt_flags(cpt);\n";
        }

        out << ind() << "bool in_class = ";

        std::vector<std::string> conditions;

        // Add range conditions
        for (const auto& r : cc.ranges) {
            std::ostringstream cond;
            if (r.is_single()) {
                cond << "(cpt == " << r.start << ")";
            } else {
                cond << "(cpt >= " << r.start << " && cpt <= " << r.end << ")";
            }
            conditions.push_back(cond.str());
        }

        // Add property conditions
        for (const auto& prop : cc.properties) {
            conditions.push_back(get_property_condition(prop, false));
        }

        if (conditions.empty()) {
            out << "false";
        } else {
            for (size_t i = 0; i < conditions.size(); i++) {
                if (i > 0) out << " || ";
                out << conditions[i];
            }
        }
        out << ";\n";

        // Handle excluded properties
        for (const auto& prop : cc.excluded_properties) {
            out << ind() << "in_class = in_class && !(" << get_property_condition(prop, false) << ");\n";
        }

        if (cc.negated) {
            out << ind() << "in_class = !in_class;\n";
        }

        out << ind() << "if (in_class) {\n";
        out << ind() << opts_.indent << pos_var << " += cpt_len(" << str_var << ", " << len_var << ", " << pos_var << ");\n";
        out << ind() << "}\n";
        dedent();
        out << ind() << "}\n";
    }

    void generate_unicode_property_match(const UnicodePropertyMatch& pm, std::ostringstream& out,
                                         const std::string& pos_var, const std::string& str_var,
                                         const std::string& len_var) {
        out << ind() << "// Match unicode property\n";
        out << ind() << "if (" << pos_var << " < " << len_var << ") {\n";
        indent();
        out << ind() << "uint32_t cpt = peek_cpt(" << str_var << ", " << len_var << ", " << pos_var << ");\n";

        if (pm.property == UnicodeProperty::Han) {
            out << ind() << "if (" << (pm.negated ? "!" : "") << "is_han(cpt)) {\n";
        } else {
            out << ind() << "CptFlags flags = get_cpt_flags(cpt);\n";
            out << ind() << "if (" << (pm.negated ? "!" : "") << "(" << get_property_condition(pm.property, false) << ")) {\n";
        }
        out << ind() << opts_.indent << pos_var << " += cpt_len(" << str_var << ", " << len_var << ", " << pos_var << ");\n";
        out << ind() << "}\n";
        dedent();
        out << ind() << "}\n";
    }

    void generate_whitespace_match(const WhitespaceMatch& wm, std::ostringstream& out,
                                   const std::string& pos_var, const std::string& str_var,
                                   const std::string& len_var) {
        out << ind() << "// Match whitespace\n";
        out << ind() << "if (" << pos_var << " < " << len_var << ") {\n";
        indent();
        out << ind() << "uint32_t cpt = peek_cpt(" << str_var << ", " << len_var << ", " << pos_var << ");\n";
        out << ind() << "CptFlags flags = get_cpt_flags(cpt);\n";
        out << ind() << "if (" << (wm.negated ? "!" : "") << "flags.is_whitespace) {\n";
        out << ind() << opts_.indent << pos_var << " += cpt_len(" << str_var << ", " << len_var << ", " << pos_var << ");\n";
        out << ind() << "}\n";
        dedent();
        out << ind() << "}\n";
    }

    void generate_digit_match(const DigitMatch& dm, std::ostringstream& out,
                              const std::string& pos_var, const std::string& str_var,
                              const std::string& len_var) {
        out << ind() << "// Match digit\n";
        out << ind() << "if (" << pos_var << " < " << len_var << ") {\n";
        indent();
        out << ind() << "uint32_t cpt = peek_cpt(" << str_var << ", " << len_var << ", " << pos_var << ");\n";
        out << ind() << "CptFlags flags = get_cpt_flags(cpt);\n";
        out << ind() << "if (" << (dm.negated ? "!" : "") << "flags.is_number) {\n";
        out << ind() << opts_.indent << pos_var << " += cpt_len(" << str_var << ", " << len_var << ", " << pos_var << ");\n";
        out << ind() << "}\n";
        dedent();
        out << ind() << "}\n";
    }

    void generate_word_match(const WordMatch& wm, std::ostringstream& out,
                             const std::string& pos_var, const std::string& str_var,
                             const std::string& len_var) {
        out << ind() << "// Match word character\n";
        out << ind() << "if (" << pos_var << " < " << len_var << ") {\n";
        indent();
        out << ind() << "uint32_t cpt = peek_cpt(" << str_var << ", " << len_var << ", " << pos_var << ");\n";
        out << ind() << "CptFlags flags = get_cpt_flags(cpt);\n";
        out << ind() << "bool is_word = flags.is_letter || flags.is_number || cpt == '_';\n";
        out << ind() << "if (" << (wm.negated ? "!" : "") << "is_word) {\n";
        out << ind() << opts_.indent << pos_var << " += cpt_len(" << str_var << ", " << len_var << ", " << pos_var << ");\n";
        out << ind() << "}\n";
        dedent();
        out << ind() << "}\n";
    }

    void generate_special_char_match(const SpecialChar& sc, std::ostringstream& out,
                                     const std::string& pos_var, const std::string& str_var,
                                     const std::string& len_var) {
        char c = '\0';
        switch (sc.type) {
            case SpecialChar::CarriageReturn: c = '\r'; break;
            case SpecialChar::Newline: c = '\n'; break;
            case SpecialChar::Tab: c = '\t'; break;
            case SpecialChar::FormFeed: c = '\f'; break;
            case SpecialChar::VerticalTab: c = '\v'; break;
            case SpecialChar::Null: c = '\0'; break;
            default: break;
        }
        out << ind() << "// Match special char\n";
        out << ind() << "if (" << pos_var << " < " << len_var << " && " << str_var << "[" << pos_var << "] == '";
        out << escape_char(c) << "') {\n";
        out << ind() << opts_.indent << pos_var << "++;\n";
        out << ind() << "}\n";
    }

    void generate_sequence_match(const Sequence& seq, std::ostringstream& out,
                                  const std::string& pos_var, const std::string& str_var,
                                  const std::string& len_var) {
        if (seq.children.empty()) {
            return;
        }

        int label = label_counter_++;
        std::string start_var = "seq_start_" + std::to_string(label);

        out << ind() << "// Sequence - all parts must match\n";
        out << ind() << "{\n";
        indent();
        out << ind() << "size_t " << start_var << " = " << pos_var << ";\n";
        out << ind() << "bool seq_ok = true;\n";

        for (size_t i = 0; i < seq.children.size(); i++) {
            out << ind() << "if (seq_ok) {\n";
            indent();
            out << ind() << "size_t before = " << pos_var << ";\n";
            generate_match_code(seq.children[i], out, pos_var, str_var, len_var);
            out << ind() << "if (" << pos_var << " == before) seq_ok = false;\n";
            dedent();
            out << ind() << "}\n";
        }

        out << ind() << "if (!seq_ok) " << pos_var << " = " << start_var << "; // Reset on failure\n";
        dedent();
        out << ind() << "}\n";
    }

    void generate_alternation_match(const Alternation& alt, std::ostringstream& out,
                                    const std::string& pos_var, const std::string& str_var,
                                    const std::string& len_var) {
        out << ind() << "// Alternation - try each branch\n";
        out << ind() << "{\n";
        indent();
        out << ind() << "size_t alt_start = " << pos_var << ";\n";
        out << ind() << "size_t best_end = alt_start;\n";

        for (size_t i = 0; i < alt.alternatives.size(); i++) {
            out << ind() << "{\n";
            indent();
            out << ind() << "size_t try_pos = alt_start;\n";
            generate_match_code(alt.alternatives[i], out, "try_pos", str_var, len_var);
            out << ind() << "if (try_pos > best_end) best_end = try_pos;\n";
            dedent();
            out << ind() << "}\n";
        }

        out << ind() << pos_var << " = best_end;\n";
        dedent();
        out << ind() << "}\n";
    }

    void generate_quantifier_match(const Quantifier& q, std::ostringstream& out,
                                   const std::string& pos_var, const std::string& str_var,
                                   const std::string& len_var) {
        int label = label_counter_++;
        std::string count_var = "count_" + std::to_string(label);
        std::string start_var = "qstart_" + std::to_string(label);

        out << ind() << "// Quantifier {" << q.min_count << ",";
        if (q.max_count < 0) {
            out << "inf";
        } else {
            out << q.max_count;
        }
        out << "}\n";

        out << ind() << "{\n";
        indent();
        out << ind() << "int " << count_var << " = 0;\n";
        out << ind() << "size_t " << start_var << " = " << pos_var << ";\n";

        if (q.max_count < 0) {
            out << ind() << "while (true) {\n";
        } else {
            out << ind() << "while (" << count_var << " < " << q.max_count << ") {\n";
        }
        indent();
        out << ind() << "size_t before = " << pos_var << ";\n";
        generate_match_code(q.child, out, pos_var, str_var, len_var);
        out << ind() << "if (" << pos_var << " == before) break; // No progress\n";
        out << ind() << count_var << "++;\n";
        dedent();
        out << ind() << "}\n";

        // Check minimum count
        out << ind() << "if (" << count_var << " < " << q.min_count << ") {\n";
        out << ind() << opts_.indent << pos_var << " = " << start_var << "; // Reset on min count failure\n";
        out << ind() << "}\n";
        dedent();
        out << ind() << "}\n";
    }

    void generate_positive_lookahead(const PositiveLookahead& la, std::ostringstream& out,
                                     const std::string& pos_var, const std::string& str_var,
                                     const std::string& len_var) {
        int label = label_counter_++;
        std::string save_var = "la_save_" + std::to_string(label);

        out << ind() << "// Positive lookahead\n";
        out << ind() << "{\n";
        indent();
        out << ind() << "size_t " << save_var << " = " << pos_var << ";\n";
        generate_match_code(la.child, out, pos_var, str_var, len_var);
        out << ind() << "bool la_matched = (" << pos_var << " > " << save_var << ");\n";
        out << ind() << pos_var << " = " << save_var << "; // Restore position\n";
        out << ind() << "if (!la_matched) " << pos_var << " = " << len_var << " + 1; // Force failure\n";
        dedent();
        out << ind() << "}\n";
    }

    void generate_negative_lookahead(const NegativeLookahead& la, std::ostringstream& out,
                                     const std::string& pos_var, const std::string& str_var,
                                     const std::string& len_var) {
        int label = label_counter_++;
        std::string save_var = "la_save_" + std::to_string(label);

        out << ind() << "// Negative lookahead\n";
        out << ind() << "{\n";
        indent();
        out << ind() << "size_t " << save_var << " = " << pos_var << ";\n";
        generate_match_code(la.child, out, pos_var, str_var, len_var);
        out << ind() << "bool la_matched = (" << pos_var << " > " << save_var << ");\n";
        out << ind() << pos_var << " = " << save_var << "; // Restore position\n";
        out << ind() << "if (la_matched) " << pos_var << " = " << len_var << " + 1; // Force failure if lookahead matched\n";
        dedent();
        out << ind() << "}\n";
    }

    std::string get_property_condition(UnicodeProperty prop, bool negated) {
        std::string cond;
        switch (prop) {
            case UnicodeProperty::Letter:
                cond = "flags.is_letter";
                break;
            case UnicodeProperty::Number:
                cond = "flags.is_number";
                break;
            case UnicodeProperty::Whitespace:
                cond = "flags.is_whitespace";
                break;
            case UnicodeProperty::Punctuation:
                cond = "flags.is_punctuation";
                break;
            case UnicodeProperty::Symbol:
                cond = "flags.is_symbol";
                break;
            case UnicodeProperty::Mark:
                cond = "flags.is_mark";
                break;
            case UnicodeProperty::Uppercase:
                cond = "flags.is_uppercase";
                break;
            case UnicodeProperty::Lowercase:
                cond = "flags.is_lowercase";
                break;
            case UnicodeProperty::Han:
                cond = "is_han(cpt)";
                break;
            default:
                cond = "flags.is_letter";  // Fallback
        }
        if (negated) {
            return "!(" + cond + ")";
        }
        return cond;
    }

    void generate_main() {
        line("int main(int argc, char** argv) {");
        indent();
        line("if (argc < 2) {");
        indent();
        line("fprintf(stderr, \"Usage: %s <text>\\n\", argv[0]);");
        line("return 1;");
        dedent();
        line("}");
        line();
        line("auto tokens = " + opts_.function_name + "(argv[1]);");
        line("for (const auto& token : tokens) {");
        indent();
        line("printf(\"[%s]\\n\", token.c_str());");
        dedent();
        line("}");
        line("return 0;");
        dedent();
        line("}");
    }

    void generate_match_logic(const NodePtr& ast, std::ostringstream& out, int /*depth*/) {
        // Simplified version for embedding
        generate_match_code(ast, out, "pos", "s", "len");
    }

    static std::string escape_char(char c) {
        switch (c) {
            case '\n': return "\\n";
            case '\r': return "\\r";
            case '\t': return "\\t";
            case '\f': return "\\f";
            case '\v': return "\\v";
            case '\0': return "\\0";
            case '\\': return "\\\\";
            case '\'': return "\\'";
            case '"': return "\\\"";
            default:
                if (c < 32 || c > 126) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\x%02x", static_cast<unsigned char>(c));
                    return buf;
                }
                return std::string(1, c);
        }
    }

    static std::string escape_string(const std::string& s) {
        std::string result;
        for (char c : s) {
            result += escape_char(c);
        }
        return result;
    }
};

std::string generate(const NodePtr& ast, const CodegenOptions& options) {
    CodeGenerator gen(options);
    return gen.generate(ast);
}

std::string generate_matcher_body(const NodePtr& ast, const CodegenOptions& options) {
    CodeGenerator gen(options);
    return gen.generate_matcher_body(ast);
}

}  // namespace pcre_codegen
