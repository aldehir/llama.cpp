#include "json-schema-to-grammar.h"
#include "common.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <deque>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using json = nlohmann::ordered_json;

static std::string build_repetition(const std::string & item_rule, int min_items, int max_items, const std::string & separator_rule = "") {
    auto has_max = max_items != std::numeric_limits<int>::max();

    if (max_items == 0) {
        return "";
    }
    if (min_items == 0 && max_items == 1) {
        return item_rule + "?";
    }

    if (separator_rule.empty()) {
        if (min_items == 1 && !has_max) {
            return item_rule + "+";
        }
        if (min_items == 0 && !has_max) {
            return item_rule + "*";
        }
        return item_rule + "{" + std::to_string(min_items) + "," + (has_max ? std::to_string(max_items) : "") + "}";
    }

    auto result = item_rule + " " + build_repetition("(" + separator_rule + " " + item_rule + ")", min_items == 0 ? 0 : min_items - 1, has_max ? max_items - 1 : max_items);
    if (min_items == 0) {
        result = "(" + result + ")?";
    }
    return result;
}

static void build_min_max_int(int64_t min_value, int64_t max_value, std::stringstream & out, int decimals_left = 16, bool top_level = true) {
    auto has_min = min_value != std::numeric_limits<int64_t>::min();
    auto has_max = max_value != std::numeric_limits<int64_t>::max();

    auto digit_range = [&](char from, char to) {
        out << "[";
        if (from == to) {
            out << from;
        } else {
            out << from << "-" << to;
        }
        out << "]";
    };
    auto more_digits = [&](int min_digits, int max_digits) {
        out << "[0-9]";
        if (min_digits == max_digits && min_digits == 1) {
            return;
        }
        out << "{";
        out << min_digits;
        if (max_digits != min_digits) {
            out << ",";
            if (max_digits != std::numeric_limits<int>::max()) {
                out << max_digits;
            }
        }
        out << "}";
    };
    std::function<void(const std::string_view &, const std::string_view &)> uniform_range =
        [&](const std::string_view & from, const std::string_view & to) {
            size_t i = 0;
            while (i < from.length() && i < to.length() && from[i] == to[i]) {
                i++;
            }
            if (i > 0) {
                out << "\"" << from.substr(0, i) << "\"";
            }
            if (i < from.length() && i < to.length()) {
                if (i > 0) {
                    out << " ";
                }
                auto sub_len = from.length() - i - 1;
                if (sub_len > 0) {
                    auto from_sub = from.substr(i + 1);
                    auto to_sub = to.substr(i + 1);
                    auto sub_zeros = string_repeat("0", sub_len);
                    auto sub_nines = string_repeat("9", sub_len);

                    auto to_reached = false;
                    out << "(";
                    if (from_sub == sub_zeros) {
                        digit_range(from[i], to[i] - 1);
                        out << " ";
                        more_digits(sub_len, sub_len);
                    } else {
                        out << "[" << from[i] << "] ";
                        out << "(";
                        uniform_range(from_sub, sub_nines);
                        out << ")";
                        if (from[i] < to[i] - 1) {
                            out << " | ";
                            if (to_sub == sub_nines) {
                                digit_range(from[i] + 1, to[i]);
                                to_reached = true;
                            } else {
                                digit_range(from[i] + 1, to[i] - 1);
                            }
                            out << " ";
                            more_digits(sub_len, sub_len);
                        }
                    }
                    if (!to_reached) {
                        out << " | ";
                        digit_range(to[i], to[i]);
                        out << " ";
                        uniform_range(sub_zeros, to_sub);
                    }
                    out << ")";
                } else {
                    out << "[" << from[i] << "-" << to[i] << "]";
                }
            }
        };

    if (has_min && has_max) {
        if (min_value < 0 && max_value < 0) {
            out << "\"-\" (";
            build_min_max_int(-max_value, -min_value, out, decimals_left, /* top_level= */ true);
            out << ")";
            return;
        }

        if (min_value < 0) {
            out << "\"-\" (";
            build_min_max_int(0, -min_value, out, decimals_left, /* top_level= */ true);
            out << ") | ";
            min_value = 0;
        }

        auto min_s = std::to_string(min_value);
        auto max_s = std::to_string(max_value);
        auto min_digits = min_s.length();
        auto max_digits = max_s.length();

        for (auto digits = min_digits; digits < max_digits; digits++) {
            uniform_range(min_s, string_repeat("9", digits));
            min_s = "1" + string_repeat("0", digits);
            out << " | ";
        }
        uniform_range(min_s, max_s);
        return;
    }

    auto less_decimals = std::max(decimals_left - 1, 1);

    if (has_min) {
        if (min_value < 0) {
            out << "\"-\" (";
            build_min_max_int(std::numeric_limits<int64_t>::min(), -min_value, out, decimals_left, /* top_level= */ false);
            out << ") | [0] | [1-9] ";
            more_digits(0, decimals_left - 1);
        } else if (min_value == 0) {
            if (top_level) {
                out << "[0] | [1-9] ";
                more_digits(0, less_decimals);
            } else {
                more_digits(1, decimals_left);
            }
        } else if (min_value <= 9) {
            char c = '0' + min_value;
            auto range_start = top_level ? '1' : '0';
            if (c > range_start) {
                digit_range(range_start, c - 1);
                out << " ";
                more_digits(1, less_decimals);
                out << " | ";
            }
            digit_range(c, '9');
            out << " ";
            more_digits(0, less_decimals);
        } else {
            auto min_s = std::to_string(min_value);
            auto len = min_s.length();
            auto c = min_s[0];

            if (c > '1') {
                digit_range(top_level ? '1' : '0', c - 1);
                out << " ";
                more_digits(len, less_decimals);
                out << " | ";
            }
            digit_range(c, c);
            out << " (";
            build_min_max_int(std::stoll(min_s.substr(1)), std::numeric_limits<int64_t>::max(), out, less_decimals, /* top_level= */ false);
            out << ")";
            if (c < '9') {
                out << " | ";
                digit_range(c + 1, '9');
                out << " ";
                more_digits(len - 1, less_decimals);
            }
        }
        return;
    }

    if (has_max) {
        if (max_value >= 0) {
            if (top_level) {
                out << "\"-\" [1-9] ";
                more_digits(0, less_decimals);
                out << " | ";
            }
            build_min_max_int(0, max_value, out, decimals_left, /* top_level= */ true);
        } else {
            out << "\"-\" (";
            build_min_max_int(-max_value, std::numeric_limits<int64_t>::max(), out, decimals_left, /* top_level= */ false);
            out << ")";
        }
        return;
    }

    throw std::runtime_error("At least one of min_value or max_value must be set");
}

const std::string SPACE_RULE = "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}";

struct BuiltinRule {
    std::string content;
    std::vector<std::string> deps;
};

static std::unordered_map<std::string, BuiltinRule> PRIMITIVE_RULES = {
    {"boolean", {"(\"true\" | \"false\") space", {}}},
    {"decimal-part", {"[0-9]{1,16}", {}}},
    {"integral-part", {"[0] | [1-9] [0-9]{0,15}", {}}},
    {"number", {"(\"-\"? integral-part) (\".\" decimal-part)? ([eE] [-+]? integral-part)? space", {"integral-part", "decimal-part"}}},
    {"integer", {"(\"-\"? integral-part) space", {"integral-part"}}},
    {"value", {"object | array | string | number | boolean | null", {"object", "array", "string", "number", "boolean", "null"}}},
    {"object", {"\"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? \"}\" space", {"string", "value"}}},
    {"array", {"\"[\" space ( value (\",\" space value)* )? \"]\" space", {"value"}}},
    {"uuid", {"\"\\\"\" [0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12} \"\\\"\" space", {}}},
    {"char",   {"[^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})", {}}},
    {"string", {"\"\\\"\" char* \"\\\"\" space", {"char"}}},
    {"null", {"\"null\" space", {}}},
};

static std::unordered_map<std::string, BuiltinRule> STRING_FORMAT_RULES = {
    {"date", {"[0-9]{4} \"-\" ( \"0\" [1-9] | \"1\" [0-2] ) \"-\" ( \"0\" [1-9] | [1-2] [0-9] | \"3\" [0-1] )", {}}},
    {"time", {"([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9] \":\" [0-5] [0-9] ( \".\" [0-9]{3} )? ( \"Z\" | ( \"+\" | \"-\" ) ( [01] [0-9] | \"2\" [0-3] ) \":\" [0-5] [0-9] )", {}}},
    {"date-time", {"date \"T\" time", {"date", "time"}}},
    {"date-string", {"\"\\\"\" date \"\\\"\" space", {"date"}}},
    {"time-string", {"\"\\\"\" time \"\\\"\" space", {"time"}}},
    {"date-time-string", {"\"\\\"\" date-time \"\\\"\" space", {"date-time"}}}
};

static bool is_reserved_name(const std::string & name) {
    static const std::unordered_set<std::string> RESERVED_NAMES = [] {
        std::unordered_set<std::string> s;
        s.insert("root");
        for (const auto & p : PRIMITIVE_RULES) {
            s.insert(p.first);
        }
        for (const auto & p : STRING_FORMAT_RULES) {
            s.insert(p.first);
        }
        return s;
    }();
    return RESERVED_NAMES.find(name) != RESERVED_NAMES.end();
}

static std::regex INVALID_RULE_CHARS_RE("[^a-zA-Z0-9-]+");
static std::regex GRAMMAR_LITERAL_ESCAPE_RE("[\r\n\"\\\\]");
static std::regex GRAMMAR_RANGE_LITERAL_ESCAPE_RE("[\r\n\"\\]\\-\\\\]");
static std::unordered_map<char, std::string> GRAMMAR_LITERAL_ESCAPES = {
    {'\r', "\\r"}, {'\n', "\\n"}, {'"', "\\\""}, {'-', "\\-"}, {']', "\\]"}, {'\\', "\\\\"}
};

static std::unordered_set<char> NON_LITERAL_SET = {'|', '.', '(', ')', '[', ']', '{', '}', '*', '+', '?'};
static std::unordered_set<char> ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS = {'^', '$', '.', '[', ']', '(', ')', '|', '{', '}', '*', '+', '?'};

static std::string replacePattern(const std::string & input, const std::regex & regex, const std::function<std::string(const std::smatch  &)> & replacement) {
    std::smatch match;
    std::string result;

    std::string::const_iterator searchStart(input.cbegin());
    std::string::const_iterator searchEnd(input.cend());

    while (std::regex_search(searchStart, searchEnd, match, regex)) {
        result.append(searchStart, searchStart + match.position());
        result.append(replacement(match));
        searchStart = match.suffix().first;
    }

    result.append(searchStart, searchEnd);

    return result;
}

static std::string format_literal(const std::string & literal) {
    std::string escaped = replacePattern(literal, GRAMMAR_LITERAL_ESCAPE_RE, [&](const std::smatch & match) {
        char c = match.str()[0];
        return GRAMMAR_LITERAL_ESCAPES.at(c);
    });
    return "\"" + escaped + "\"";
}

std::string gbnf_format_literal(const std::string & literal) { return format_literal(literal); }

// Schema intermediate representation.
//
// Normalization lowers a JSON schema into these nodes, resolving structural
// keywords (oneOf/anyOf, type arrays, allOf merging) up front. Emission walks
// the nodes to produce GBNF. Type queries such as
// common_schema_info::resolves_to_string() also operate on the nodes.

using common_schema_node_id = size_t;

enum common_schema_node_kind {
    COMMON_SCHEMA_NODE_REF,        // $ref to a schema tracked in _refs
    COMMON_SCHEMA_NODE_UNION,      // oneOf / anyOf / array of types
    COMMON_SCHEMA_NODE_CONST,      // const value
    COMMON_SCHEMA_NODE_ENUM,       // enum values (or allOf enum intersection)
    COMMON_SCHEMA_NODE_OBJECT,     // object with properties (or merged allOf)
    COMMON_SCHEMA_NODE_TUPLE,      // fixed-length array (items array / prefixItems)
    COMMON_SCHEMA_NODE_ARRAY,      // homogeneous array with optional bounds
    COMMON_SCHEMA_NODE_PATTERN,    // string constrained by a regex pattern
    COMMON_SCHEMA_NODE_FORMAT,     // string format (uuid, date, time, ...)
    COMMON_SCHEMA_NODE_STRING,     // string with length bounds
    COMMON_SCHEMA_NODE_INTEGER,    // integer with value bounds
    COMMON_SCHEMA_NODE_PRIMITIVE,  // unconstrained primitive type
    COMMON_SCHEMA_NODE_INVALID,    // unrecognized schema, emits an error
};

enum common_schema_node_additional {
    COMMON_SCHEMA_ADDITIONAL_NONE,    // no additional properties
    COMMON_SCHEMA_ADDITIONAL_ANY,     // additionalProperties: true
    COMMON_SCHEMA_ADDITIONAL_SCHEMA,  // additionalProperties: { ... }
};

struct common_schema_node {
    common_schema_node_kind kind = COMMON_SCHEMA_NODE_INVALID;

    std::string ref;                                // REF
    std::vector<common_schema_node_id> children;    // UNION alternatives, TUPLE items
    std::vector<nlohmann::ordered_json> values;     // CONST (single value), ENUM

    // OBJECT
    std::vector<std::pair<std::string, common_schema_node_id>> props;
    std::unordered_set<std::string> required;
    common_schema_node_additional additional = COMMON_SCHEMA_ADDITIONAL_NONE;
    common_schema_node_id additional_node = 0;

    // Original allOf members, kept for type queries on the merged node
    std::vector<common_schema_node_id> allof_members;

    common_schema_node_id item = 0;                   // ARRAY
    int min_items = 0;                                // ARRAY
    int max_items = std::numeric_limits<int>::max();  // ARRAY

    int min_len = 0;                                // STRING
    int max_len = std::numeric_limits<int>::max();  // STRING

    int64_t min_value = std::numeric_limits<int64_t>::min();  // INTEGER
    int64_t max_value = std::numeric_limits<int64_t>::max();  // INTEGER

    std::string pattern;       // PATTERN
    std::string name;          // FORMAT format name, PRIMITIVE rule name
    bool wrap = false;         // PRIMITIVE: reference the primitive from a named rule
    bool string_hint = false;  // schema had string-only keywords the node kind doesn't capture
    std::string message;       // INVALID
};

class common_schema_converter {
private:
    friend std::string build_grammar(const std::function<void(const common_grammar_builder &)> & cb, const common_grammar_options & options);
    std::function<json(const std::string &)> _fetch_json;
    bool _dotall;
    std::map<std::string, std::string> _rules;
    std::unordered_map<std::string, json> _refs;
    std::unordered_set<std::string> _refs_being_resolved;
    std::vector<std::string> _errors;
    std::vector<std::string> _warnings;
    // deque so node references stay valid while normalization appends
    std::deque<common_schema_node> _nodes;
    std::unordered_map<std::string, common_schema_node_id> _def_nodes;

    common_schema_node_id _add_node(common_schema_node && node) {
        _nodes.push_back(std::move(node));
        return _nodes.size() - 1;
    }

    // Normalizes (and memoizes) the schema referenced by ref.
    common_schema_node_id _normalize_def(const std::string & ref) {
        auto it = _def_nodes.find(ref);
        if (it != _def_nodes.end()) {
            return it->second;
        }
        auto id = normalize(_refs[ref]);
        _def_nodes.emplace(ref, id);
        return id;
    }

    std::string _add_rule(const std::string & name, const std::string & rule) {
        std::string esc_name = regex_replace(name, INVALID_RULE_CHARS_RE, "-");
        if (_rules.find(esc_name) == _rules.end() || _rules[esc_name] == rule) {
            _rules[esc_name] = rule;
            return esc_name;
        }
        int i = 0;
        while (_rules.find(esc_name + std::to_string(i)) != _rules.end() && _rules[esc_name + std::to_string(i)] != rule) {
            i++;
        }
        std::string key = esc_name + std::to_string(i);
        _rules[key] = rule;
        return key;
    }

    std::string _generate_union_rule(const std::string & name, const std::vector<common_schema_node_id> & alt_nodes) {
        std::vector<std::string> rules;
        rules.reserve(alt_nodes.size());
        for (size_t i = 0; i < alt_nodes.size(); i++) {
            rules.push_back(emit(alt_nodes[i], name + (name.empty() ? "alternative-" : "-") + std::to_string(i)));
        }
        return string_join(rules, " | ");
    }

    std::string _visit_pattern(const std::string & pattern, const std::string & name) {
        if (!(pattern.front() == '^' && pattern.back() == '$')) {
            _errors.push_back("Pattern must start with '^' and end with '$'");
            return "";
        }
        std::string sub_pattern = pattern.substr(1, pattern.length() - 2);
        std::unordered_map<std::string, std::string> sub_rule_ids;

        size_t i = 0;
        size_t length = sub_pattern.length();

        using literal_or_rule = std::pair<std::string, bool>;
        auto to_rule = [&](const literal_or_rule & ls) {
            auto is_literal = ls.second;
            auto s = ls.first;
            return is_literal ? "\"" + s + "\"" : s;
        };
        std::function<literal_or_rule()> transform = [&]() -> literal_or_rule {
            size_t start = i;
            std::vector<literal_or_rule> seq;

            auto get_dot = [&]() {
                std::string rule;
                if (_dotall) {
                    rule = "[\\U00000000-\\U0010FFFF]";
                } else {
                    rule = "[^\\x0A\\x0D]";
                }
                return _add_rule("dot", rule);
            };

            // Joins the sequence, merging consecutive literals together.
            auto join_seq = [&]() {
                std::vector<literal_or_rule> ret;

                std::string literal;
                auto flush_literal = [&]() {
                    if (literal.empty()) {
                        return false;
                    }
                    ret.emplace_back(literal, true);
                    literal.clear();
                    return true;
                };

                for (const auto & item : seq) {
                    auto is_literal = item.second;
                    if (is_literal) {
                        literal += item.first;
                    } else {
                        flush_literal();
                        ret.push_back(item);
                    }
                }
                flush_literal();

                std::vector<std::string> results;
                results.reserve(ret.size());
                for (const auto & item : ret) {
                    results.push_back(to_rule(item));
                }
                return std::make_pair(string_join(results, " "), false);
            };

            while (i < length) {
                char c = sub_pattern[i];
                if (c == '.') {
                    seq.emplace_back(get_dot(), false);
                    i++;
                } else if (c == '(') {
                    i++;
                    if (i < length && sub_pattern[i] == '?') {
                        if (i + 1 < length && sub_pattern[i + 1] == ':') {
                            i += 2; // skip "?:" for non-capturing group, treat as regular group
                        } else {
                            // lookahead/lookbehind (?=, ?!, ?<=, ?<!) - not supported
                            _warnings.push_back("Unsupported pattern syntax");
                            // skip to matching ')' to avoid UB on empty seq
                            int depth = 1;
                            while (i < length && depth > 0) {
                                if (sub_pattern[i] == '\\' && i + 1 < length) {
                                    i += 2; // skip escaped character
                                } else {
                                    if (sub_pattern[i] == '(') depth++;
                                    else if (sub_pattern[i] == ')') depth--;
                                    i++;
                                }
                            }
                            continue;
                        }
                    }
                    seq.emplace_back("(" + to_rule(transform()) + ")", false);
                } else if (c == ')') {
                    i++;
                    if (start > 0 && sub_pattern[start - 1] != '(' && (start < 2 || sub_pattern[start - 2] != '?' || sub_pattern[start - 1] != ':')) {
                        _errors.push_back("Unbalanced parentheses");
                    }
                    return join_seq();
                } else if (c == '[') {
                    std::string square_brackets = std::string(1, c);
                    i++;
                    while (i < length && sub_pattern[i] != ']') {
                        if (sub_pattern[i] == '\\') {
                            square_brackets += sub_pattern.substr(i, 2);
                            i += 2;
                        } else {
                            square_brackets += sub_pattern[i];
                            i++;
                        }
                    }
                    if (i >= length) {
                        _errors.push_back("Unbalanced square brackets");
                    }
                    square_brackets += ']';
                    i++;
                    seq.emplace_back(square_brackets, false);
                } else if (c == '|') {
                    seq.emplace_back("|", false);
                    i++;
                } else if (c == '*' || c == '+' || c == '?') {
                    seq.back() = std::make_pair(to_rule(seq.back()) + c, false);
                    i++;
                } else if (c == '{') {
                    std::string curly_brackets = std::string(1, c);
                    i++;
                    while (i < length && sub_pattern[i] != '}') {
                        curly_brackets += sub_pattern[i];
                        i++;
                    }
                    if (i >= length) {
                        _errors.push_back("Unbalanced curly brackets");
                    }
                    curly_brackets += '}';
                    i++;
                    auto nums = string_split(curly_brackets.substr(1, curly_brackets.length() - 2), ",");
                    int min_times = 0;
                    int max_times = std::numeric_limits<int>::max();
                    try {
                        if (nums.size() == 1) {
                            min_times = max_times = std::stoi(nums[0]);
                        } else if (nums.size() != 2) {
                            _errors.push_back("Wrong number of values in curly brackets");
                        } else {
                            if (!nums[0].empty()) {
                                min_times = std::stoi(nums[0]);
                            }
                            if (!nums[1].empty()) {
                                max_times = std::stoi(nums[1]);
                            }
                        }
                    } catch (const std::invalid_argument & e) {
                        _errors.push_back("Invalid number in curly brackets");
                        return std::make_pair("", false);
                    }
                    auto &last = seq.back();
                    auto &sub = last.first;
                    auto sub_is_literal = last.second;

                    if (!sub_is_literal) {
                        std::string & sub_id = sub_rule_ids[sub];
                        if (sub_id.empty()) {
                            sub_id = _add_rule(name + "-" + std::to_string(sub_rule_ids.size()), sub);
                        }
                        sub = sub_id;
                    }
                    seq.back().first = build_repetition(
                        sub_is_literal ? "\"" + sub + "\"" : sub,
                        min_times,
                        max_times,
                        ""
                    );
                    seq.back().second = false;
                } else {
                    std::string literal;
                    auto is_non_literal = [&](char c) {
                        return NON_LITERAL_SET.find(c) != NON_LITERAL_SET.end();
                    };
                    while (i < length) {
                        if (sub_pattern[i] == '\\' && i < length - 1) {
                            char next = sub_pattern[i + 1];
                            if (ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS.find(next) != ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS.end()) {
                                i++;
                                literal += sub_pattern[i];
                                i++;
                            } else {
                                literal += sub_pattern.substr(i, 2);
                                i += 2;
                            }
                        } else if (sub_pattern[i] == '"') {
                            literal += "\\\"";
                            i++;
                        } else if (!is_non_literal(sub_pattern[i]) &&
                                (i == length - 1 || literal.empty() || sub_pattern[i + 1] == '.' || !is_non_literal(sub_pattern[i + 1]))) {
                            literal += sub_pattern[i];
                            i++;
                        } else {
                            break;
                        }
                    }
                    if (!literal.empty()) {
                        seq.emplace_back(literal, true);
                    }
                }
            }
            return join_seq();
        };
        return _add_rule(name, "\"\\\"\" (" + to_rule(transform()) + ") \"\\\"\" space");
    }

    /*
        Returns a rule that matches a JSON string that is none of the provided strings

        not_strings({"a"})
            -> ["] ( [a] char+ | [^"a] char* )? ["] space
        not_strings({"and", "also"})
            -> ["] ( [a] ([l] ([s] ([o] char+ | [^"o] char*) | [^"s] char*) | [n] ([d] char+ | [^"d] char*) | [^"ln] char*) | [^"a] char* )? ["] space
    */
    std::string _not_strings(const std::vector<std::string> & strings) {

        struct TrieNode {
            std::map<char, TrieNode> children;
            bool is_end_of_string;

            TrieNode() : is_end_of_string(false) {}

            void insert(const std::string & string) {
                auto *node = this;
                for (char c : string) {
                    node = &node->children[c];
                }
                node->is_end_of_string = true;
            }
        };

        TrieNode trie;
        for (const auto & s : strings) {
            trie.insert(s);
        }

        std::string char_rule = _add_primitive("char", PRIMITIVE_RULES.at("char"));
        std::ostringstream out;
        out << "[\"] ( ";
        std::function<void(const TrieNode &)> visit = [&](const TrieNode & node) {
            std::ostringstream rejects;
            auto first = true;
            for (const auto & kv : node.children) {
                rejects << kv.first;
                if (first) {
                    first = false;
                } else {
                    out << " | ";
                }
                out << "[" << kv.first << "]";
                if (!kv.second.children.empty()) {
                    out << " (";
                    visit(kv.second);
                    out << ")";
                } else if (kv.second.is_end_of_string) {
                    out << " " << char_rule << "+";
                }
            }
            if (!node.children.empty()) {
                if (!first) {
                    out << " | ";
                }
                out << "[^\"" << rejects.str() << "] " << char_rule << "*";
            }
        };
        visit(trie);

        out << " )";
        if (!trie.is_end_of_string) {
            out << "?";
        }
        out << " [\"] space";
        return out.str();
    }

    std::string _resolve_ref(const std::string & ref) {
        auto it = ref.find('#');
        std::string ref_fragment = it != std::string::npos ? ref.substr(it + 1) : ref;
        static const std::regex nonalphanumeric_regex(R"([^a-zA-Z0-9-]+)");
        std::string ref_name = "ref" + std::regex_replace(ref_fragment, nonalphanumeric_regex, "-");
        if (_rules.find(ref_name) == _rules.end() && _refs_being_resolved.find(ref) == _refs_being_resolved.end()) {
            _refs_being_resolved.insert(ref);
            ref_name = emit(_normalize_def(ref), ref_name);
            _refs_being_resolved.erase(ref);
        }
        return ref_name;
    }

    std::string _build_object_rule(
        const std::vector<std::pair<std::string, common_schema_node_id>> & properties,
        const std::unordered_set<std::string> & required,
        const std::string & name,
        common_schema_node_additional additional,
        common_schema_node_id additional_node)
    {
        std::vector<std::string> required_props;
        std::vector<std::string> optional_props;
        std::unordered_map<std::string, std::string> prop_kv_rule_names;
        std::vector<std::string> prop_names;
        for (const auto & kv : properties) {
            const auto &prop_name = kv.first;
            const auto &prop_node = kv.second;

            std::string prop_rule_name = emit(prop_node, name + (name.empty() ? "" : "-") + prop_name);
            prop_kv_rule_names[prop_name] = _add_rule(
                name + (name.empty() ? "" : "-") + prop_name + "-kv",
                format_literal(json(prop_name).dump()) + " space \":\" space " + prop_rule_name
            );
            if (required.find(prop_name) != required.end()) {
                required_props.push_back(prop_name);
            } else {
                optional_props.push_back(prop_name);
            }
            prop_names.push_back(prop_name);
        }
        if (additional != COMMON_SCHEMA_ADDITIONAL_NONE) {
            std::string sub_name = name + (name.empty() ? "" : "-") + "additional";
            std::string value_rule =
                additional == COMMON_SCHEMA_ADDITIONAL_SCHEMA ? emit(additional_node, sub_name + "-value")
                : _add_primitive("value", PRIMITIVE_RULES.at("value"));

            auto key_rule =
                prop_names.empty() ? _add_primitive("string", PRIMITIVE_RULES.at("string"))
                : _add_rule(sub_name + "-k", _not_strings(prop_names));
            std::string kv_rule = _add_rule(sub_name + "-kv", key_rule + " \":\" space " + value_rule);
            prop_kv_rule_names["*"] = kv_rule;
            optional_props.push_back("*");
        }

        std::string rule = "\"{\" space ";
        for (size_t i = 0; i < required_props.size(); i++) {
            if (i > 0) {
                rule += " \",\" space ";
            }
            rule += prop_kv_rule_names[required_props[i]];
        }

        if (!optional_props.empty()) {
            rule += " (";
            if (!required_props.empty()) {
                rule += " \",\" space ( ";
            }

            std::function<std::string(const std::vector<std::string> &, bool)> get_recursive_refs = [&](const std::vector<std::string> & ks, bool first_is_optional) {
                std::string res;
                if (ks.empty()) {
                    return res;
                }
                const std::string& k = ks[0];
                std::string kv_rule_name = prop_kv_rule_names[k];
                std::string comma_ref = "( \",\" space " + kv_rule_name + " )";
                if (first_is_optional) {
                    res = comma_ref + (k == "*" ? "*" : "?");
                } else {
                    res = kv_rule_name + (k == "*" ? " " + comma_ref + "*" : "");
                }
                if (ks.size() > 1) {
                    res += " " + _add_rule(
                        name + (name.empty() ? "" : "-") + k + "-rest",
                        get_recursive_refs(std::vector<std::string>(ks.begin() + 1, ks.end()), true)
                    );
                }
                return res;
            };

            for (size_t i = 0; i < optional_props.size(); i++) {
                if (i > 0) {
                    rule += " | ";
                }
                rule += get_recursive_refs(std::vector<std::string>(optional_props.begin() + i, optional_props.end()), false);
            }
            if (!required_props.empty()) {
                rule += " )";
            }
            rule += " )?";
        }

        rule += " \"}\" space";

        return rule;
    }

    std::string _add_primitive(const std::string & name, const BuiltinRule & rule) {
        auto n = _add_rule(name, rule.content);
        for (const auto & dep : rule.deps) {
            BuiltinRule dep_rule;
            auto it = PRIMITIVE_RULES.find(dep);
            if (it == PRIMITIVE_RULES.end()) {
                it = STRING_FORMAT_RULES.find(dep);
                if (it == STRING_FORMAT_RULES.end()) {
                    _errors.push_back("Rule " + dep + " not known");
                    continue;
                }
            }
            if (_rules.find(dep) == _rules.end()) {
                _add_primitive(dep, it->second);
            }
        }
        return n;
    }

public:
    common_schema_converter(
        const std::function<json(const std::string &)> & fetch_json,
        bool dotall)
          : _fetch_json(fetch_json), _dotall(dotall)
    {
        _rules["space"] = SPACE_RULE;
    }

    void resolve_refs(json & schema, const std::string & url) {
        /*
        * Resolves all $ref fields in the given schema, fetching any remote schemas,
        * replacing each $ref with absolute reference URL and populates _refs with the
        * respective referenced (sub)schema dictionaries.
        */
        std::function<void(json &)> visit_refs = [&](json & n) {
            if (n.is_array()) {
                for (auto & x : n) {
                    visit_refs(x);
                }
            } else if (n.is_object()) {
                if (n.contains("$ref")) {
                    std::string ref = n["$ref"];
                    if (_refs.find(ref) == _refs.end()) {
                        json target;
                        if (ref.find("https://") == 0) {
                            std::string base_url = ref.substr(0, ref.find('#'));
                            auto it = _refs.find(base_url);
                            if (it != _refs.end()) {
                                target = it->second;
                            } else {
                                // Fetch the referenced schema and resolve its refs
                                auto referenced = _fetch_json(ref);
                                resolve_refs(referenced, base_url);
                                _refs[base_url] = referenced;
                            }
                            if (ref.find('#') == std::string::npos || ref.substr(ref.find('#') + 1).empty()) {
                                return;
                            }
                        } else if (ref.find("#/") == 0) {
                            target = schema;
                            n["$ref"] = url + ref;
                            ref = url + ref;
                        } else {
                            _errors.push_back("Unsupported ref: " + ref);
                            return;
                        }
                        std::string pointer = ref.substr(ref.find('#') + 1);
                        std::vector<std::string> tokens = string_split(pointer, "/");
                        for (size_t i = 1; i < tokens.size(); ++i) {
                            const std::string& sel = tokens[i];
                            if (target.is_object() && target.contains(sel)) {
                                target = target[sel];
                            } else if (target.is_array()) {
                                size_t sel_index;
                                try {
                                    sel_index = std::stoull(sel);
                                } catch (const std::invalid_argument & e) {
                                    sel_index = target.size();
                                }
                                if (sel_index >= target.size()) {
                                    _errors.push_back("Error resolving ref " + ref + ": " + sel + " not in " + target.dump());
                                    return;
                                }
                                target = target[sel_index];
                            } else {
                                _errors.push_back("Error resolving ref " + ref + ": " + sel + " not in " + target.dump());
                                return;
                            }
                        }
                        _refs[ref] = target;
                    }
                } else {
                    for (const auto & kv : n.items()) {
                        visit_refs(kv.value());
                    }
                }
            }
        };

        visit_refs(schema);
    }

    static std::string _generate_constant_rule(const json & value) {
        return format_literal(value.dump());
    }

    std::string visit(const json & schema, const std::string & name) {
        return emit(normalize(schema), name);
    }

    // Lowers a JSON schema into a normalized node. Unrecognized schemas become
    // INVALID nodes so the error is reported at emission, preserving ordering.
    common_schema_node_id normalize(const json & schema) {
        json schema_type = schema.contains("type") ? schema["type"] : json();
        std::string schema_format = schema.contains("format") ? schema["format"].get<std::string>() : "";

        common_schema_node node;

        if (schema.contains("$ref")) {
            node.kind = COMMON_SCHEMA_NODE_REF;
            node.ref = schema["$ref"];
            return _add_node(std::move(node));
        }
        if (schema.contains("oneOf") || schema.contains("anyOf")) {
            const json & alts = schema.contains("oneOf") ? schema["oneOf"] : schema["anyOf"];
            node.kind = COMMON_SCHEMA_NODE_UNION;
            for (const auto & alt : alts) {
                node.children.push_back(normalize(alt));
            }
            return _add_node(std::move(node));
        }
        if (schema_type.is_array()) {
            node.kind = COMMON_SCHEMA_NODE_UNION;
            for (const auto & t : schema_type) {
                json schema_copy(schema);
                schema_copy["type"] = t;
                node.children.push_back(normalize(schema_copy));
            }
            return _add_node(std::move(node));
        }
        if (schema.contains("const")) {
            node.kind = COMMON_SCHEMA_NODE_CONST;
            node.values.push_back(schema["const"]);
            return _add_node(std::move(node));
        }
        if (schema.contains("enum")) {
            node.kind = COMMON_SCHEMA_NODE_ENUM;
            for (const auto & v : schema["enum"]) {
                node.values.push_back(v);
            }
            return _add_node(std::move(node));
        }
        if ((schema_type.is_null() || schema_type == "object")
                && (schema.contains("properties") ||
                    (schema.contains("additionalProperties") && schema["additionalProperties"] != true))) {
            node.kind = COMMON_SCHEMA_NODE_OBJECT;
            if (schema.contains("required") && schema["required"].is_array()) {
                for (const auto & item : schema["required"]) {
                    if (item.is_string()) {
                        node.required.insert(item.get<std::string>());
                    }
                }
            }
            if (schema.contains("properties")) {
                for (const auto & prop : schema["properties"].items()) {
                    node.props.emplace_back(prop.key(), normalize(prop.value()));
                }
            }
            if (schema.contains("additionalProperties")) {
                const json & additional = schema["additionalProperties"];
                if (additional.is_boolean() && additional.get<bool>()) {
                    node.additional = COMMON_SCHEMA_ADDITIONAL_ANY;
                } else if (additional.is_object()) {
                    node.additional = COMMON_SCHEMA_ADDITIONAL_SCHEMA;
                    node.additional_node = normalize(additional);
                }
            }
            return _add_node(std::move(node));
        }
        if ((schema_type.is_null() || schema_type == "object" || schema_type == "string") && schema.contains("allOf")) {
            // allOf is merged into a single object (or an enum intersection)
            std::map<std::string, std::pair<json, size_t>> enum_values;
            node.string_hint = schema_type == "string";
            std::function<void(const json &, bool)> add_component = [&](const json & comp_schema, bool is_required) {
                if (comp_schema.contains("$ref")) {
                    add_component(_refs[comp_schema["$ref"]], is_required);
                } else if (comp_schema.contains("properties")) {
                    for (const auto & prop : comp_schema["properties"].items()) {
                        node.props.emplace_back(prop.key(), normalize(prop.value()));
                        if (is_required) {
                            node.required.insert(prop.key());
                        }
                    }
                } else if (comp_schema.contains("enum")) {
                    for (const auto & v : comp_schema["enum"]) {
                        const auto rule = _generate_constant_rule(v);
                        auto it = enum_values.find(rule);
                        if (it == enum_values.end()) {
                            enum_values.emplace(rule, std::make_pair(v, 1));
                        } else {
                            it->second.second += 1;
                        }
                    }
                } else {
                  // todo warning
                }
            };
            for (const auto & t : schema["allOf"]) {
                node.allof_members.push_back(normalize(t));
                if (t.contains("anyOf")) {
                    for (const auto & tt : t["anyOf"]) {
                        add_component(tt, false);
                    }
                } else {
                    add_component(t, true);
                }
            }
            if (!enum_values.empty()) {
                std::vector<json> intersection;
                for (const auto & p : enum_values) {
                    if (p.second.second == schema["allOf"].size()) {
                        intersection.push_back(p.second.first);
                    }
                }
                if (!intersection.empty()) {
                    node.kind = COMMON_SCHEMA_NODE_ENUM;
                    node.values = std::move(intersection);
                    node.props.clear();
                    node.required.clear();
                    return _add_node(std::move(node));
                }
            }
            node.kind = COMMON_SCHEMA_NODE_OBJECT;
            return _add_node(std::move(node));
        }
        if ((schema_type.is_null() || schema_type == "array") && (schema.contains("items") || schema.contains("prefixItems"))) {
            const json & items = schema.contains("items") ? schema["items"] : schema["prefixItems"];
            if (items.is_array()) {
                node.kind = COMMON_SCHEMA_NODE_TUPLE;
                for (const auto & item : items) {
                    node.children.push_back(normalize(item));
                }
                return _add_node(std::move(node));
            }
            node.kind = COMMON_SCHEMA_NODE_ARRAY;
            node.item = normalize(items);
            node.min_items = schema.contains("minItems") ? schema["minItems"].get<int>() : 0;
            if (schema.contains("maxItems") && schema["maxItems"].is_number_integer()) {
                node.max_items = schema["maxItems"].get<int>();
            }
            return _add_node(std::move(node));
        }
        if ((schema_type.is_null() || schema_type == "string") && schema.contains("pattern")) {
            node.kind = COMMON_SCHEMA_NODE_PATTERN;
            node.pattern = schema["pattern"];
            return _add_node(std::move(node));
        }
        if ((schema_type.is_null() || schema_type == "string") && std::regex_match(schema_format, std::regex("^uuid[1-5]?$"))) {
            node.kind = COMMON_SCHEMA_NODE_FORMAT;
            node.name = schema_format;
            return _add_node(std::move(node));
        }
        if ((schema_type.is_null() || schema_type == "string") && STRING_FORMAT_RULES.find(schema_format + "-string") != STRING_FORMAT_RULES.end()) {
            node.kind = COMMON_SCHEMA_NODE_FORMAT;
            node.name = schema_format;
            return _add_node(std::move(node));
        }
        if (schema_type == "string" && (schema.contains("minLength") || schema.contains("maxLength"))) {
            node.kind = COMMON_SCHEMA_NODE_STRING;
            node.min_len = schema.contains("minLength") ? schema["minLength"].get<int>() : 0;
            if (schema.contains("maxLength")) {
                node.max_len = schema["maxLength"].get<int>();
            }
            return _add_node(std::move(node));
        }
        if (schema_type == "integer" && (schema.contains("minimum") || schema.contains("exclusiveMinimum") || schema.contains("maximum") || schema.contains("exclusiveMaximum"))) {
            node.kind = COMMON_SCHEMA_NODE_INTEGER;
            if (schema.contains("minimum")) {
                node.min_value = schema["minimum"].get<int64_t>();
            } else if (schema.contains("exclusiveMinimum")) {
                node.min_value = schema["exclusiveMinimum"].get<int64_t>() + 1;
            }
            if (schema.contains("maximum")) {
                node.max_value = schema["maximum"].get<int64_t>();
            } else if (schema.contains("exclusiveMaximum")) {
                node.max_value = schema["exclusiveMaximum"].get<int64_t>() - 1;
            }
            return _add_node(std::move(node));
        }
        if (schema.empty() || schema_type == "object") {
            node.kind = COMMON_SCHEMA_NODE_PRIMITIVE;
            node.name = "object";
            node.wrap = true;
            return _add_node(std::move(node));
        }
        if (schema_type.is_null() && schema.is_object()) {
            // No type constraint and no recognized structural keywords (e.g. {"description": "..."}).
            // Per JSON Schema semantics this is equivalent to {} and accepts any value.
            node.kind = COMMON_SCHEMA_NODE_PRIMITIVE;
            node.name = "value";
            node.wrap = true;
            node.string_hint = schema.contains("pattern") || schema.contains("minLength") || schema.contains("maxLength") ||
                               schema_format == "uri" || schema_format == "email" || schema_format == "hostname" ||
                               schema_format == "ipv4" || schema_format == "ipv6" || schema_format.find("uuid") == 0;
            return _add_node(std::move(node));
        }
        if (!schema_type.is_string() || PRIMITIVE_RULES.find(schema_type.get<std::string>()) == PRIMITIVE_RULES.end()) {
            node.kind = COMMON_SCHEMA_NODE_INVALID;
            node.message = "Unrecognized schema: " + schema.dump();
            return _add_node(std::move(node));
        }
        // TODO: support minimum, maximum, exclusiveMinimum, exclusiveMaximum at least for zero
        node.kind = COMMON_SCHEMA_NODE_PRIMITIVE;
        node.name = schema_type.get<std::string>();
        return _add_node(std::move(node));
    }

    // Emits GBNF rules for a normalized node, returning the rule name
    std::string emit(common_schema_node_id id, const std::string & name) {
        const auto & node = _nodes[id];
        std::string rule_name = is_reserved_name(name) ? name + "-" : name.empty() ? "root" : name;

        switch (node.kind) {
            case COMMON_SCHEMA_NODE_REF:
                return _add_rule(rule_name, _resolve_ref(node.ref));
            case COMMON_SCHEMA_NODE_UNION:
                return _add_rule(rule_name, _generate_union_rule(name, node.children));
            case COMMON_SCHEMA_NODE_CONST:
                return _add_rule(rule_name, _generate_constant_rule(node.values.front()) + " space");
            case COMMON_SCHEMA_NODE_ENUM: {
                std::vector<std::string> enum_values;
                enum_values.reserve(node.values.size());
                for (const auto & v : node.values) {
                    enum_values.push_back(_generate_constant_rule(v));
                }
                return _add_rule(rule_name, "(" + string_join(enum_values, " | ") + ") space");
            }
            case COMMON_SCHEMA_NODE_OBJECT:
                return _add_rule(rule_name,
                    _build_object_rule(node.props, node.required, name, node.additional, node.additional_node));
            case COMMON_SCHEMA_NODE_TUPLE: {
                std::string rule = "\"[\" space ";
                for (size_t i = 0; i < node.children.size(); i++) {
                    if (i > 0) {
                        rule += " \",\" space ";
                    }
                    rule += emit(node.children[i], name + (name.empty() ? "" : "-") + "tuple-" + std::to_string(i));
                }
                rule += " \"]\" space";
                return _add_rule(rule_name, rule);
            }
            case COMMON_SCHEMA_NODE_ARRAY: {
                std::string item_rule_name = emit(node.item, name + (name.empty() ? "" : "-") + "item");
                return _add_rule(rule_name, "\"[\" space " + build_repetition(item_rule_name, node.min_items, node.max_items, "\",\" space") + " \"]\" space");
            }
            case COMMON_SCHEMA_NODE_PATTERN:
                return _visit_pattern(node.pattern, rule_name);
            case COMMON_SCHEMA_NODE_FORMAT: {
                if (std::regex_match(node.name, std::regex("^uuid[1-5]?$"))) {
                    return _add_primitive(rule_name == "root" ? "root" : node.name, PRIMITIVE_RULES.at("uuid"));
                }
                auto prim_name = node.name + "-string";
                return _add_rule(rule_name, _add_primitive(prim_name, STRING_FORMAT_RULES.at(prim_name)));
            }
            case COMMON_SCHEMA_NODE_STRING: {
                std::string char_rule = _add_primitive("char", PRIMITIVE_RULES.at("char"));
                return _add_rule(rule_name, "\"\\\"\" " + build_repetition(char_rule, node.min_len, node.max_len) + " \"\\\"\" space");
            }
            case COMMON_SCHEMA_NODE_INTEGER: {
                std::stringstream out;
                out << "(";
                build_min_max_int(node.min_value, node.max_value, out);
                out << ") space";
                return _add_rule(rule_name, out.str());
            }
            case COMMON_SCHEMA_NODE_PRIMITIVE:
                if (node.wrap) {
                    return _add_rule(rule_name, _add_primitive(node.name, PRIMITIVE_RULES.at(node.name)));
                }
                return _add_primitive(rule_name == "root" ? "root" : node.name, PRIMITIVE_RULES.at(node.name));
            case COMMON_SCHEMA_NODE_INVALID:
                _errors.push_back(node.message);
                return "";
        }
        return "";
    }

    // Determines if a node can resolve to a string type through any path.
    // Some models emit raw string values rather than JSON-encoded strings for string parameters.
    // If any branch of the node (via UNION, REF, etc.) permits a string, this returns true,
    // allowing callers to handle the value as a raw string for simplicity.
    bool resolves_to_string(common_schema_node_id id, std::unordered_set<std::string> & visited_refs) {
        const auto & node = _nodes[id];

        auto all_members_resolve_to_string = [&]() {
            for (auto member : node.allof_members) {
                if (!resolves_to_string(member, visited_refs)) {
                    return false;
                }
            }
            return true;
        };

        switch (node.kind) {
            case COMMON_SCHEMA_NODE_REF: {
                if (visited_refs.find(node.ref) != visited_refs.end()) {
                    // Circular reference, assume not a string to be safe
                    return false;
                }
                visited_refs.insert(node.ref);
                if (_refs.find(node.ref) == _refs.end()) {
                    return false;
                }
                return resolves_to_string(_normalize_def(node.ref), visited_refs);
            }
            case COMMON_SCHEMA_NODE_UNION:
                for (auto child : node.children) {
                    if (resolves_to_string(child, visited_refs)) {
                        return true;
                    }
                }
                return false;
            case COMMON_SCHEMA_NODE_CONST:
                return node.values.front().is_string();
            case COMMON_SCHEMA_NODE_ENUM:
                if (node.string_hint) {
                    return true;
                }
                if (!node.allof_members.empty()) {
                    // allOf: all members must be compatible with string type
                    return all_members_resolve_to_string();
                }
                for (const auto & v : node.values) {
                    if (v.is_string()) {
                        return true;
                    }
                }
                return false;
            case COMMON_SCHEMA_NODE_OBJECT:
                if (node.string_hint) {
                    return true;
                }
                if (!node.allof_members.empty()) {
                    return all_members_resolve_to_string();
                }
                return false;
            case COMMON_SCHEMA_NODE_PATTERN:
            case COMMON_SCHEMA_NODE_FORMAT:
            case COMMON_SCHEMA_NODE_STRING:
                return true;
            case COMMON_SCHEMA_NODE_PRIMITIVE:
                return node.name == "string" || node.string_hint;
            case COMMON_SCHEMA_NODE_TUPLE:
            case COMMON_SCHEMA_NODE_ARRAY:
            case COMMON_SCHEMA_NODE_INTEGER:
            case COMMON_SCHEMA_NODE_INVALID:
                return false;
        }
        return false;
    }

    void check_errors() {
        if (!_errors.empty()) {
            throw std::invalid_argument("JSON schema conversion failed:\n" + string_join(_errors, "\n"));
        }
        if (!_warnings.empty()) {
            fprintf(stderr, "WARNING: JSON schema conversion was incomplete: %s\n", string_join(_warnings, "; ").c_str());
        }
    }

    std::string format_grammar() {
        std::stringstream ss;
        for (const auto & kv : _rules) {
            ss << kv.first << " ::= " << kv.second << '\n';
        }
        return ss.str();
    }
};

// common_schema_info implementation (pimpl)

common_schema_info::common_schema_info()
    : impl_(std::make_unique<common_schema_converter>(
        [](const std::string &) { return json(); },
        false)) {}

common_schema_info::~common_schema_info() = default;

common_schema_info::common_schema_info(common_schema_info &&) noexcept = default;
common_schema_info & common_schema_info::operator=(common_schema_info &&) noexcept = default;

void common_schema_info::resolve_refs(nlohmann::ordered_json & schema) {
    impl_->resolve_refs(schema, "");
}

// Determines if a JSON schema can resolve to a string type through any path.
// Some models emit raw string values rather than JSON-encoded strings for string parameters.
// If any branch of the schema (via oneOf, anyOf, $ref, etc.) permits a string, this returns
// true, allowing callers to handle the value as a raw string for simplicity.
bool common_schema_info::resolves_to_string(const nlohmann::ordered_json & schema) {
    if (!schema.is_object()) {
        return false;
    }
    std::unordered_set<std::string> visited_refs;
    return impl_->resolves_to_string(impl_->normalize(schema), visited_refs);
}

std::string json_schema_to_grammar(const json & schema, bool force_gbnf) {
#ifdef LLAMA_USE_LLGUIDANCE
    if (!force_gbnf) {
        return "%llguidance {}\nstart: %json " + schema.dump();
    }
#else
    (void)force_gbnf;
#endif // LLAMA_USE_LLGUIDANCE
    return build_grammar([&](const common_grammar_builder & callbacks) {
        auto copy = schema;
        callbacks.resolve_refs(copy);
        callbacks.add_schema("", copy);
    });
}

std::string build_grammar(const std::function<void(const common_grammar_builder &)> & cb, const common_grammar_options & options) {
    common_schema_converter converter([&](const std::string &) { return json(); }, options.dotall);
    common_grammar_builder builder {
        /* .add_rule = */ [&](const std::string & name, const std::string & rule) {
            return converter._add_rule(name, rule);
        },
        /* .add_schema = */ [&](const std::string & name, const nlohmann::ordered_json & schema) {
            return converter.visit(schema, name == "root" ? "" : name);
        },
        /* .resolve_refs = */ [&](nlohmann::ordered_json & schema) {
            converter.resolve_refs(schema, "");
        }
    };
    cb(builder);
    converter.check_errors();
    return converter.format_grammar();
}
