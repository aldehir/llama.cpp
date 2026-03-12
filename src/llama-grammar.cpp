#include "llama-grammar.h"
#include "llama-grammar-dfa.h"

#include "llama-impl.h"
#include "llama-vocab.h"
#include "llama-sampling.h"

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>

#define MAX_REPETITION_THRESHOLD 2000
//
// helpers
//

// NOTE: assumes valid utf8 (but checks for overrun)
static std::pair<uint32_t, const char *> decode_utf8(const char * src) {
    static const int lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t  first_byte = static_cast<uint8_t>(*src);
    uint8_t  highbits   = first_byte >> 4;
    int      len        = lookup[highbits];
    uint8_t  mask       = (1 << (8 - len)) - 1;
    uint32_t value      = first_byte & mask;
    const char * end    = src + len; // may overrun!
    const char * pos    = src + 1;
    for ( ; pos < end && *pos; pos++) {
        value = (value << 6) + (static_cast<uint8_t>(*pos) & 0x3F);
    }
    return std::make_pair(value, pos);
}

static bool is_digit_char(char c) {
    return '0' <= c && c <= '9';
}

static bool is_word_char(char c) {
    return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '-' || is_digit_char(c);
}

static std::pair<uint32_t, const char *> parse_hex(const char * src, int size) {
    const char * pos   = src;
    const char * end   = src + size;
    uint32_t     value = 0;
    for ( ; pos < end && *pos; pos++) {
        value <<= 4;
        char c = *pos;
        if ('a' <= c && c <= 'f') {
            value += c - 'a' + 10;
        } else if ('A' <= c && c <= 'F') {
            value += c - 'A' + 10;
        } else if ('0' <= c && c <= '9') {
            value += c - '0';
        } else {
            break;
        }
    }
    if (pos != end) {
        throw std::runtime_error("expecting " + std::to_string(size) + " hex chars at " + src);
    }
    return std::make_pair(value, pos);
}

static const char * parse_space(const char * src, bool newline_ok) {
    const char * pos = src;
    while (*pos == ' ' || *pos == '\t' || *pos == '#' ||
            (newline_ok && (*pos == '\r' || *pos == '\n'))) {
        if (*pos == '#') {
            while (*pos && *pos != '\r' && *pos != '\n') {
                pos++;
            }
        } else {
            pos++;
        }
    }
    return pos;
}

static const char * parse_name(const char * src) {
    const char * pos = src;
    while (is_word_char(*pos)) {
        pos++;
    }
    if (pos == src) {
        throw std::runtime_error(std::string("expecting name at ") + src);
    }
    return pos;
}

static const char * parse_int(const char * src) {
    const char * pos = src;
    while (is_digit_char(*pos)) {
        pos++;
    }
    if (pos == src) {
        throw std::runtime_error(std::string("expecting integer at ") + src);
    }
    return pos;
}

static std::pair<uint32_t, const char *> parse_char(const char * src) {
    if (*src == '\\') {
        switch (src[1]) {
            case 'x': return parse_hex(src + 2, 2);
            case 'u': return parse_hex(src + 2, 4);
            case 'U': return parse_hex(src + 2, 8);
            case 't': return std::make_pair('\t', src + 2);
            case 'r': return std::make_pair('\r', src + 2);
            case 'n': return std::make_pair('\n', src + 2);
            case '\\':
            case '"':
            case '[':
            case ']':
                      return std::make_pair(src[1], src + 2);
            default:
                      throw std::runtime_error(std::string("unknown escape at ") + src);
        }
    } else if (*src) {
        return decode_utf8(src);
    }
    throw std::runtime_error("unexpected end of input");
}

static void print_grammar_char(FILE * file, uint32_t c) {
    if (0x20 <= c && c <= 0x7f) {
        fprintf(file, "%c", static_cast<char>(c));
    } else {
        // cop out of encoding UTF-8
        fprintf(file, "<U+%04X>", c);
    }
}

static bool is_char_element(llama_grammar_element elem) {
    switch (elem.type) {
        case LLAMA_GRETYPE_CHAR:           return true;
        case LLAMA_GRETYPE_CHAR_NOT:       return true;
        case LLAMA_GRETYPE_CHAR_ALT:       return true;
        case LLAMA_GRETYPE_CHAR_RNG_UPPER: return true;
        case LLAMA_GRETYPE_CHAR_ANY:       return true;
        default:                           return false;
    }
}

static void print_rule_binary(FILE * file, const llama_grammar_rule & rule) {
    for (auto elem : rule) {
        switch (elem.type) {
            case LLAMA_GRETYPE_END:            fprintf(file, "END");            break;
            case LLAMA_GRETYPE_ALT:            fprintf(file, "ALT");            break;
            case LLAMA_GRETYPE_RULE_REF:       fprintf(file, "RULE_REF");       break;
            case LLAMA_GRETYPE_CHAR:           fprintf(file, "CHAR");           break;
            case LLAMA_GRETYPE_CHAR_NOT:       fprintf(file, "CHAR_NOT");       break;
            case LLAMA_GRETYPE_CHAR_RNG_UPPER: fprintf(file, "CHAR_RNG_UPPER"); break;
            case LLAMA_GRETYPE_CHAR_ALT:       fprintf(file, "CHAR_ALT");       break;
            case LLAMA_GRETYPE_CHAR_ANY:       fprintf(file, "CHAR_ANY");       break;
        }
        switch (elem.type) {
            case LLAMA_GRETYPE_END:
            case LLAMA_GRETYPE_ALT:
            case LLAMA_GRETYPE_RULE_REF:
                fprintf(file, "(%u) ", elem.value);
                break;
            case LLAMA_GRETYPE_CHAR:
            case LLAMA_GRETYPE_CHAR_NOT:
            case LLAMA_GRETYPE_CHAR_RNG_UPPER:
            case LLAMA_GRETYPE_CHAR_ALT:
            case LLAMA_GRETYPE_CHAR_ANY:
                fprintf(file, "(\"");
                print_grammar_char(file, elem.value);
                fprintf(file, "\") ");
                break;
        }
    }
    fprintf(file, "\n");
}

static void print_rule(
        FILE     * file,
        uint32_t   rule_id,
        const llama_grammar_rule & rule,
        const std::map<uint32_t, std::string> & symbol_id_names) {
    if (rule.empty() || rule.back().type != LLAMA_GRETYPE_END) {
        throw std::runtime_error(
            "malformed rule, does not end with LLAMA_GRETYPE_END: " + std::to_string(rule_id));
    }
    fprintf(file, "%s ::= ", symbol_id_names.at(rule_id).c_str());
    for (size_t i = 0, end = rule.size() - 1; i < end; i++) {
        llama_grammar_element elem = rule[i];
        switch (elem.type) {
            case LLAMA_GRETYPE_END:
                throw std::runtime_error(
                    "unexpected end of rule: " + std::to_string(rule_id) + "," +
                    std::to_string(i));
            case LLAMA_GRETYPE_ALT:
                fprintf(file, "| ");
                break;
            case LLAMA_GRETYPE_RULE_REF:
                fprintf(file, "%s ", symbol_id_names.at(elem.value).c_str());
                break;
            case LLAMA_GRETYPE_CHAR:
                fprintf(file, "[");
                print_grammar_char(file, elem.value);
                break;
            case LLAMA_GRETYPE_CHAR_NOT:
                fprintf(file, "[^");
                print_grammar_char(file, elem.value);
                break;
            case LLAMA_GRETYPE_CHAR_RNG_UPPER:
                if (i == 0 || !is_char_element(rule[i - 1])) {
                    throw std::runtime_error(
                        "LLAMA_GRETYPE_CHAR_RNG_UPPER without preceding char: " +
                        std::to_string(rule_id) + "," + std::to_string(i));
                }
                fprintf(file, "-");
                print_grammar_char(file, elem.value);
                break;
            case LLAMA_GRETYPE_CHAR_ALT:
                if (i == 0 || !is_char_element(rule[i - 1])) {
                    throw std::runtime_error(
                        "LLAMA_GRETYPE_CHAR_ALT without preceding char: " +
                        std::to_string(rule_id) + "," + std::to_string(i));
                }
                print_grammar_char(file, elem.value);
                break;
            case LLAMA_GRETYPE_CHAR_ANY:
                fprintf(file, ".");
                break;
        }
        if (is_char_element(elem)) {
            switch (rule[i + 1].type) {
                case LLAMA_GRETYPE_CHAR_ALT:
                case LLAMA_GRETYPE_CHAR_RNG_UPPER:
                case LLAMA_GRETYPE_CHAR_ANY:
                    break;
                default:
                    fprintf(file, "] ");
            }
        }
    }
    fprintf(file, "\n");
}

//
// implementation
//

uint32_t llama_grammar_parser::get_symbol_id(const char * src, size_t len) {
    uint32_t next_id = static_cast<uint32_t>(symbol_ids.size());
    auto result = symbol_ids.emplace(std::string(src, len), next_id);
    return result.first->second;
}

uint32_t llama_grammar_parser::generate_symbol_id(const std::string & base_name) {
    uint32_t next_id = static_cast<uint32_t>(symbol_ids.size());
    symbol_ids[base_name + '_' + std::to_string(next_id)] = next_id;
    return next_id;
}

void llama_grammar_parser::add_rule(uint32_t rule_id, const llama_grammar_rule & rule) {
    if (rules.size() <= rule_id) {
        rules.resize(rule_id + 1);
    }
    rules[rule_id] = rule;
}

const char * llama_grammar_parser::parse_alternates(
        const char        * src,
        const std::string & rule_name,
        uint32_t            rule_id,
        bool                is_nested) {
    llama_grammar_rule rule;
    const char * pos = parse_sequence(src, rule_name, rule, is_nested);
    while (*pos == '|') {
        rule.push_back({LLAMA_GRETYPE_ALT, 0});
        pos = parse_space(pos + 1, true);
        pos = parse_sequence(pos, rule_name, rule, is_nested);
    }
    rule.push_back({LLAMA_GRETYPE_END, 0});
    add_rule(rule_id, rule);
    return pos;
}

const char * llama_grammar_parser::parse_sequence(
        const char         * src,
        const std::string  & rule_name,
        llama_grammar_rule & rule,
        bool               is_nested) {
    size_t last_sym_start = rule.size();
    const char * pos = src;

    // use UINT64_MAX as the empty value because we aligned to the proper uint64_t type so -1 can't be used
    // (though it's technically the same as -1 now)
    auto handle_repetitions = [&](uint64_t min_times, uint64_t max_times) {
        bool no_max = max_times == UINT64_MAX;
        if (last_sym_start == rule.size()) {
            throw std::runtime_error(std::string("expecting preceding item to */+/?/{ at ") + pos);
        }

        // apply transformation to previous symbol (last_sym_start to end) according to
        // the following rewrite rules:
        // S{m,n} --> S S S (m times) S'(n-m)
        //            S'(x)   ::= S S'(x-1) |
        //            (... n-m definitions of these S' rules ...)
        //            S'(1)   ::= S |
        // S{m,} -->  S S S (m times) S'
        //            S'     ::= S S' |
        // S*     --> S{0,}
        //        --> S'     ::= S S' |
        // S+     --> S{1,}
        //        --> S S'
        //            S'     ::= S S' |
        // S?     --> S{0,1}
        //        --> S'
        //            S'     ::= S |

        llama_grammar_rule prev_rule(rule.begin() + last_sym_start, rule.end());
        if (min_times == 0) {
            rule.resize(last_sym_start);
        } else {
            // Repeat the previous elements (min_times - 1) times
            for (uint64_t i = 1; i < min_times; i++) {
                rule.insert(rule.end(), prev_rule.begin(), prev_rule.end());
            }
        }

        uint32_t last_rec_rule_id = 0;
        auto n_opt = no_max ? 1 : max_times - min_times;

        llama_grammar_rule rec_rule(prev_rule);
        for (uint64_t i = 0; i < n_opt; i++) {
            rec_rule.resize(prev_rule.size());
            uint32_t rec_rule_id = generate_symbol_id( rule_name);
            if (i > 0 || no_max) {
                rec_rule.push_back({LLAMA_GRETYPE_RULE_REF, no_max ? rec_rule_id : last_rec_rule_id});
            }
            rec_rule.push_back({LLAMA_GRETYPE_ALT, 0});
            rec_rule.push_back({LLAMA_GRETYPE_END, 0});
            add_rule( rec_rule_id, rec_rule);
            last_rec_rule_id = rec_rule_id;
        }
        if (n_opt > 0) {
            rule.push_back({LLAMA_GRETYPE_RULE_REF, last_rec_rule_id});
        }
    };

    while (*pos) {
        if (*pos == '"') { // literal string
            pos++;
            last_sym_start = rule.size();
            while (*pos != '"') {
                if (!*pos) {
                    throw std::runtime_error("unexpected end of input");
                }
                auto char_pair = parse_char(pos);
                     pos       = char_pair.second;
                rule.push_back({LLAMA_GRETYPE_CHAR, char_pair.first});
            }
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '[') { // char range(s)
            pos++;
            enum llama_gretype start_type = LLAMA_GRETYPE_CHAR;
            if (*pos == '^') {
                pos++;
                start_type = LLAMA_GRETYPE_CHAR_NOT;
            }
            last_sym_start = rule.size();
            while (*pos != ']') {
                if (!*pos) {
                    throw std::runtime_error("unexpected end of input");
                }
                auto char_pair = parse_char(pos);
                     pos       = char_pair.second;
                enum llama_gretype type = last_sym_start < rule.size()
                    ? LLAMA_GRETYPE_CHAR_ALT
                    : start_type;

                rule.push_back({type, char_pair.first});
                if (pos[0] == '-' && pos[1] != ']') {
                    if (!pos[1]) {
                        throw std::runtime_error("unexpected end of input");
                    }
                    auto endchar_pair = parse_char(pos + 1);
                         pos          = endchar_pair.second;
                    rule.push_back({LLAMA_GRETYPE_CHAR_RNG_UPPER, endchar_pair.first});
                }
            }
            pos = parse_space(pos + 1, is_nested);
        } else if (is_word_char(*pos)) { // rule reference
            const char * name_end    = parse_name(pos);
            uint32_t ref_rule_id = get_symbol_id(pos, name_end - pos);
            pos = parse_space(name_end, is_nested);
            last_sym_start = rule.size();
            rule.push_back({LLAMA_GRETYPE_RULE_REF, ref_rule_id});
        } else if (*pos == '(') { // grouping
            // parse nested alternates into synthesized rule
            pos = parse_space(pos + 1, true);
            uint32_t sub_rule_id = generate_symbol_id(rule_name);
            pos = parse_alternates(pos, rule_name, sub_rule_id, true);
            last_sym_start = rule.size();
            // output reference to synthesized rule
            rule.push_back({LLAMA_GRETYPE_RULE_REF, sub_rule_id});
            if (*pos != ')') {
                throw std::runtime_error(std::string("expecting ')' at ") + pos);
            }
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '.') { // any char
            last_sym_start = rule.size();
            rule.push_back({LLAMA_GRETYPE_CHAR_ANY, 0});
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '*') {
            pos = parse_space(pos + 1, is_nested);
            handle_repetitions(0, -1);
        } else if (*pos == '+') {
            pos = parse_space(pos + 1, is_nested);
            handle_repetitions(1, -1);
        } else if (*pos == '?') {
            pos = parse_space(pos + 1, is_nested);
            handle_repetitions(0, 1);
        } else if (*pos == '{') {
            pos = parse_space(pos + 1, is_nested);

            if (!is_digit_char(*pos)) {
                throw std::runtime_error(std::string("expecting an int at ") + pos);
            }
            const char * int_end = parse_int(pos);
            uint64_t min_times = std::stoul(std::string(pos, int_end - pos));
            pos = parse_space(int_end, is_nested);

            uint64_t max_times = UINT64_MAX; // default: no max limit

            if (*pos == '}') {
                max_times = min_times;
                pos = parse_space(pos + 1, is_nested);
            } else if (*pos == ',') {
                pos = parse_space(pos + 1, is_nested);

                if (is_digit_char(*pos)) {
                    const char * int_end = parse_int(pos);
                    max_times = std::stoul(std::string(pos, int_end - pos));
                    pos = parse_space(int_end, is_nested);
                }

                if (*pos != '}') {
                    throw std::runtime_error(std::string("expecting '}' at ") + pos);
                }
                pos = parse_space(pos + 1, is_nested);
            } else {
                throw std::runtime_error(std::string("expecting ',' at ") + pos);
            }
            bool has_max = max_times != UINT64_MAX;
            if (min_times > MAX_REPETITION_THRESHOLD || (has_max && max_times > MAX_REPETITION_THRESHOLD)) {
                throw std::runtime_error(std::string("number of repetitions exceeds sane defaults, please reduce the number of repetitions"));
            }
            handle_repetitions(min_times, max_times);
        } else {
            break;
        }
    }
    return pos;
}

const char * llama_grammar_parser::parse_rule(const char * src) {
    const char * name_end = parse_name(src);
    const char * pos      = parse_space(name_end, false);
    size_t       name_len = name_end - src;
    uint32_t     rule_id  = get_symbol_id(src, name_len);
    const std::string name(src, name_len);

    if (!(pos[0] == ':' && pos[1] == ':' && pos[2] == '=')) {
        throw std::runtime_error(std::string("expecting ::= at ") + pos);
    }
    pos = parse_space(pos + 3, true);

    pos = parse_alternates(pos, name, rule_id, false);

    if (*pos == '\r') {
        pos += pos[1] == '\n' ? 2 : 1;
    } else if (*pos == '\n') {
        pos++;
    } else if (*pos) {
        throw std::runtime_error(std::string("expecting newline or end at ") + pos);
    }
    return parse_space(pos, true);
}

bool llama_grammar_parser::parse(const char * src) {
    try {
        const char * pos = parse_space(src, true);
        while (*pos) {
            pos = parse_rule(pos);
        }
        // Validate the state to ensure that all rules are defined
        for (const auto & rule : rules) {
            if (rule.empty()) {
                throw std::runtime_error("Undefined rule");
            }
            for (const auto & elem : rule) {
                if (elem.type == LLAMA_GRETYPE_RULE_REF) {
                    // Ensure that the rule at that location exists
                    if (elem.value >= rules.size() || rules[elem.value].empty()) {
                        // Get the name of the rule that is missing
                        for (const auto & kv : symbol_ids) {
                            if (kv.second == elem.value) {
                                throw std::runtime_error("Undefined rule identifier '" + kv.first + "'");
                            }
                        }
                    }
                }
            }
        }
    } catch (const std::exception & err) {
        fprintf(stderr, "%s: error parsing grammar: %s\n\n%s\n", __func__, err.what(), src);
        rules.clear();
        return false;
    }

    return true;
}

void llama_grammar_parser::print(FILE * file) {
    try {
        std::map<uint32_t, std::string> symbol_id_names;
        for (const auto & kv : symbol_ids) {
            symbol_id_names[kv.second] = kv.first;
        }
        for (size_t i = 0, end = rules.size(); i < end; i++) {
            // fprintf(file, "%zu: ", i);
            // print_rule_binary(file, rules[i]);
            print_rule(file, uint32_t(i), rules[i], symbol_id_names);
            // fprintf(file, "\n");
        }
    } catch (const std::exception & err) {
        fprintf(stderr, "\n%s: error printing grammar: %s\n", __func__, err.what());
    }
}

std::vector<const llama_grammar_element *> llama_grammar_parser::c_rules() const {
    std::vector<const llama_grammar_element *> ret;
    ret.reserve(rules.size());
    for (const auto & rule : rules) {
        ret.push_back(rule.data());
    }
    return ret;
}

// returns true iff pos points to the end of one of the definitions of a rule
static bool llama_grammar_is_end_of_sequence(const llama_grammar_element * pos) {
    switch (pos->type) {
        case LLAMA_GRETYPE_END: return true;  // NOLINT
        case LLAMA_GRETYPE_ALT: return true;  // NOLINT
        default:                return false;
    }
}

static bool llama_grammar_detect_left_recursion(
        const llama_grammar_rules & rules,
        size_t rule_index,
        std::vector<bool> * rules_visited,
        std::vector<bool> * rules_in_progress,
        std::vector<bool> * rules_may_be_empty) {
    if ((*rules_in_progress)[rule_index]) {
        return true;
    }

    (*rules_in_progress)[rule_index] = true;

    const llama_grammar_rule & rule = rules[rule_index];

    // First check if the rule might produce the empty string. This could be done combined with the second
    // step but it's more readable as two steps.
    bool at_rule_start = true;
    for (size_t i = 0; i < rule.size(); i++) {
        if (llama_grammar_is_end_of_sequence(&rule[i])) {
            if (at_rule_start) {
                (*rules_may_be_empty)[rule_index] = true;
                break;
            }
            at_rule_start = true;
        } else {
            at_rule_start = false;
        }
    }

    // Second, recurse into leftmost nonterminals (or next-leftmost as long as the previous nonterminal may
    // be empty)
    bool recurse_into_nonterminal = true;
    for (size_t i = 0; i < rule.size(); i++) {
        if (rule[i].type == LLAMA_GRETYPE_RULE_REF && recurse_into_nonterminal) {
            if (llama_grammar_detect_left_recursion(rules, (size_t)rule[i].value, rules_visited, rules_in_progress, rules_may_be_empty)) {
                return true;
            }
            if (!((*rules_may_be_empty)[(size_t)rule[i].value])) {
                recurse_into_nonterminal = false;
            }
        } else if (llama_grammar_is_end_of_sequence(&rule[i])) {
            recurse_into_nonterminal = true;
        } else {
            recurse_into_nonterminal = false;
        }
    }

    (*rules_in_progress)[rule_index] = false;
    (*rules_visited)[rule_index] = true;

    return false;
}

////////////////////

struct llama_grammar * llama_grammar_init_impl(
        const struct llama_vocab * vocab,
        const llama_grammar_element ** rules,
        size_t n_rules,
        size_t start_rule_index) {
    const llama_grammar_element * pos;

    // copy rule definitions into vectors
    llama_grammar_rules vec_rules(n_rules);
    for (size_t i = 0; i < n_rules; i++) {
        for (pos = rules[i]; pos->type != LLAMA_GRETYPE_END; pos++) {
            vec_rules[i].push_back(*pos);
        }
        vec_rules[i].push_back({LLAMA_GRETYPE_END, 0});
    }

    // Check for left recursion
    std::vector<bool> rules_visited(n_rules);
    std::vector<bool> rules_in_progress(n_rules);
    std::vector<bool> rules_may_be_empty(n_rules);
    for (size_t i = 0; i < n_rules; i++) {
        if (rules_visited[i]) {
            continue;
        }
        if (llama_grammar_detect_left_recursion(vec_rules, i, &rules_visited, &rules_in_progress, &rules_may_be_empty)) {
            LLAMA_LOG_ERROR("unsupported grammar, left recursion detected for nonterminal at index %zu", i);
            return nullptr;
        }
    }

    // Compile DFA-based grammar
    auto cg = std::make_shared<compiled_grammar>(
        compiled_grammar::compile(vec_rules, (uint32_t)start_rule_index));
    auto dfa_configs = cg->init_configs();

    return new llama_grammar {
        vocab,
        /* .compiled = */         std::move(cg),
        /* .configs = */          std::move(dfa_configs),
        /* .lazy =*/              false,
        /* .awaiting_trigger = */ false,
        /* .trigger_buffer = */   "",
        /* .trigger_tokens   = */ {},
        /* .trigger_patterns    = */ {},
    };
}

struct llama_grammar * llama_grammar_init_impl(
        const struct llama_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                              bool lazy,
                     const char ** trigger_patterns,
                            size_t num_trigger_patterns,
               const llama_token * trigger_tokens,
                            size_t num_trigger_tokens) {
    llama_grammar_parser parser;

    // if there is a grammar, parse it
    // rules will be empty (default) if there are parse errors
    if (!parser.parse(grammar_str) || parser.rules.empty()) {
        fprintf(stderr, "%s: failed to parse grammar\n", __func__);
        return nullptr;
    }

    // Ensure that there is a "root" node.
    if (parser.symbol_ids.find("root") == parser.symbol_ids.end()) {
        fprintf(stderr, "%s: grammar does not contain a 'root' symbol\n", __func__);
        return nullptr;
    }

    std::vector<const llama_grammar_element *> grammar_rules(parser.c_rules());

    const size_t n_rules = grammar_rules.size();
    const size_t start_rule_index = parser.symbol_ids.at(grammar_root);

    const llama_grammar_element * pos;

    // copy rule definitions into vectors
    llama_grammar_rules vec_rules(n_rules);
    for (size_t i = 0; i < n_rules; i++) {
        for (pos = grammar_rules[i]; pos->type != LLAMA_GRETYPE_END; pos++) {
            vec_rules[i].push_back(*pos);
        }
        vec_rules[i].push_back({LLAMA_GRETYPE_END, 0});
    }

    // Check for left recursion
    std::vector<bool> rules_visited(n_rules);
    std::vector<bool> rules_in_progress(n_rules);
    std::vector<bool> rules_may_be_empty(n_rules);
    for (size_t i = 0; i < n_rules; i++) {
        if (rules_visited[i]) {
            continue;
        }
        if (llama_grammar_detect_left_recursion(vec_rules, i, &rules_visited, &rules_in_progress, &rules_may_be_empty)) {
            LLAMA_LOG_ERROR("unsupported grammar, left recursion detected for nonterminal at index %zu", i);
            return nullptr;
        }
    }

    std::vector<llama_token>    vec_trigger_tokens;
    std::vector<llama_grammar_trigger_pattern> vec_trigger_patterns;
    for (size_t i = 0; i < num_trigger_tokens; i++) {
        GGML_ASSERT(trigger_tokens != nullptr);
        vec_trigger_tokens.push_back(trigger_tokens[i]);
    }
    for (size_t i = 0; i < num_trigger_patterns; i++) {
        GGML_ASSERT(trigger_patterns != nullptr);
        auto & trigger = vec_trigger_patterns.emplace_back();
        trigger.pattern = trigger_patterns[i];
        trigger.regex = std::regex(trigger.pattern);
    }

    // Compile DFA-based grammar
    auto cg = std::make_shared<compiled_grammar>(
        compiled_grammar::compile(vec_rules, (uint32_t)start_rule_index));
    auto dfa_configs = cg->init_configs();

    return new llama_grammar {
        vocab,
        /* .compiled = */         std::move(cg),
        /* .configs = */          std::move(dfa_configs),
        /* .lazy = */             lazy,
        /* .awaiting_trigger = */ lazy,
        /* .trigger_buffer = */   "",
        std::move(vec_trigger_tokens),
        std::move(vec_trigger_patterns),
    };
}

void llama_grammar_free_impl(struct llama_grammar * grammar) {
    if (grammar == nullptr) {
        return;
    }

    delete grammar;
}

struct llama_grammar * llama_grammar_clone_impl(const struct llama_grammar & grammar) {
    return new llama_grammar {
        grammar.vocab,
        grammar.compiled,    // shared_ptr: compiled grammar is immutable, shared across clones
        grammar.configs,     // configs are value types, safe to copy
        grammar.lazy,
        grammar.awaiting_trigger,
        grammar.trigger_buffer,
        grammar.trigger_tokens,
        grammar.trigger_patterns,
    };
}

void llama_grammar_apply_impl(const struct llama_grammar & grammar, llama_token_data_array * cur_p) {
    GGML_ASSERT(grammar.vocab != nullptr);

    if (grammar.awaiting_trigger) {
        return;
    }

    // DFA-based token validation
    bool allow_eog = grammar.compiled->any_config_complete(grammar.configs);

    for (size_t i = 0; i < cur_p->size; ++i) {
        const llama_token id = cur_p->data[i].id;
        const std::string & piece = grammar.vocab->token_to_piece(id);

        if (grammar.vocab->is_eog(id)) {
            if (!allow_eog) {
                cur_p->data[i].logit = -INFINITY;
            }
        } else if (piece.empty() || piece[0] == 0) {
            cur_p->data[i].logit = -INFINITY;
        } else {
            // Simulate DFA forward through all bytes of the token
            auto sim_configs = grammar.configs;
            bool valid = true;
            for (size_t j = 0; j < piece.size(); j++) {
                grammar.compiled->accept_byte(sim_configs, (uint8_t)piece[j]);
                if (sim_configs.empty()) {
                    valid = false;
                    break;
                }
            }
            if (!valid) {
                cur_p->data[i].logit = -INFINITY;
            }
        }
    }
}

void llama_grammar_accept_impl(struct llama_grammar & grammar, llama_token token) {
    GGML_ASSERT(grammar.vocab != nullptr);

    const auto & piece = grammar.vocab->token_to_piece(token);

    if (grammar.awaiting_trigger) {
        if (std::find(grammar.trigger_tokens.begin(), grammar.trigger_tokens.end(), token) != grammar.trigger_tokens.end()) {
            grammar.awaiting_trigger = false;
            grammar.trigger_buffer.clear();
            llama_grammar_accept_str(grammar, piece);
            LLAMA_LOG_DEBUG("Grammar triggered on token %u (`%s`)", token, piece.c_str());
            return;
        } else {
            grammar.trigger_buffer += piece;

            std::smatch match;
            for (const auto & trigger_pattern : grammar.trigger_patterns) {
                if (std::regex_match(grammar.trigger_buffer, match, trigger_pattern.regex)) {
                    grammar.awaiting_trigger = false;
                    // get from the first matched capturing group to the end of the string
                    size_t start = std::string::npos;
                    for (auto i = 1u; i < match.size(); i++) {
                        if (match.length(i) > 0) {
                            start = match.position(i);
                            break;
                        }
                    }
                    if (start == std::string::npos) {
                        start = match.position(0);
                    }
                    auto constrained_str = grammar.trigger_buffer.substr(start);
                    grammar.trigger_buffer.clear();
                    llama_grammar_accept_str(grammar, constrained_str);
                    LLAMA_LOG_DEBUG("Grammar triggered on regex: '%s'\n", constrained_str.c_str());
                    return;
                }
            }
            LLAMA_LOG_DEBUG("Grammar still awaiting trigger after token %d (`%s`)\n", token, piece.c_str());
            return;
        }
    }

    if (grammar.vocab->is_eog(token)) {
        if (grammar.compiled->any_config_complete(grammar.configs)) {
            return;
        }
        GGML_ABORT("fatal error");
    }

    llama_grammar_accept_str(grammar, piece);
}

void llama_grammar_accept_str(struct llama_grammar & grammar, const std::string & piece) {
    // DFA-based: feed raw bytes through the engine
    for (size_t i = 0; i < piece.size(); i++) {
        grammar.compiled->accept_byte(grammar.configs, (uint8_t)piece[i]);
        if (grammar.configs.empty()) {
            throw std::runtime_error("Unexpected empty grammar configs after accepting byte in piece: " + piece);
        }
    }
}
