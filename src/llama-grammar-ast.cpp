#include "llama-grammar-ast.h"
#include "llama-grammar-dfa.h"
#include "llama-impl.h"

#include <cassert>
#include <cstdio>
#include <functional>
#include <stdexcept>

#define MAX_REPETITION_THRESHOLD 2000
#define MAX_INLINE_AST_NODES 50
#define MAX_PENDING_NFA_STATES 128

// ============================================================
// AST node methods
// ============================================================

ast_node_ptr grammar_ast_node::clone() const {
    auto n = std::make_unique<grammar_ast_node>();
    n->type = type;
    n->code_points = code_points;
    n->ranges = ranges;
    n->negated = negated;
    n->any_char = any_char;
    n->rule_name = rule_name;
    n->rule_id = rule_id;
    n->min_rep = min_rep;
    n->max_rep = max_rep;
    n->excluded_strings = excluded_strings;

    for (const auto & c : children) {
        n->children.push_back(c->clone());
    }
    if (child) {
        n->child = child->clone();
    }
    return n;
}

bool grammar_ast_node::is_purely_terminal() const {
    switch (type) {
        case LITERAL:
        case CHAR_CLASS:
        case EXCLUSION:
            return true;
        case RULE_REF:
            return false;
        case SEQUENCE:
        case ALTERNATION:
            for (const auto & c : children) {
                if (!c->is_purely_terminal()) {
                    return false;
                }
            }
            return true;
        case REPETITION:
            return child && child->is_purely_terminal();
    }
    return false;
}

size_t grammar_ast_node::node_count() const {
    size_t count = 1;
    for (const auto & c : children) {
        count += c->node_count();
    }
    if (child) {
        count += child->node_count();
    }
    return count;
}

// ============================================================
// Lexer helpers (duplicated from llama-grammar.cpp to keep
// the AST parser self-contained)
// ============================================================

static std::pair<uint32_t, const char *> decode_utf8(const char * src) {
    static const int lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t  first_byte = static_cast<uint8_t>(*src);
    uint8_t  highbits   = first_byte >> 4;
    int      len        = lookup[highbits];
    uint8_t  mask       = (1 << (8 - len)) - 1;
    uint32_t value      = first_byte & mask;
    const char * end    = src + len;
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

static const char * parse_int_str(const char * src) {
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

// ============================================================
// AST parser
// ============================================================

uint32_t llama_grammar_ast_parser::get_symbol_id(const char * src, size_t len) {
    uint32_t next_id = static_cast<uint32_t>(symbol_ids.size());
    auto result = symbol_ids.emplace(std::string(src, len), next_id);
    return result.first->second;
}

const char * llama_grammar_ast_parser::parse_alternates(
        const char * src,
        const std::string & rule_name,
        bool is_nested,
        ast_node_ptr & out) {
    ast_node_ptr first;
    const char * pos = parse_sequence(src, rule_name, is_nested, first);

    if (*pos != '|') {
        out = std::move(first);
        return pos;
    }

    std::vector<ast_node_ptr> branches;
    branches.push_back(std::move(first));

    while (*pos == '|') {
        pos = parse_space(pos + 1, true);
        ast_node_ptr branch;
        pos = parse_sequence(pos, rule_name, is_nested, branch);
        branches.push_back(std::move(branch));
    }

    auto alt = std::make_unique<grammar_ast_node>();
    alt->type = grammar_ast_node::ALTERNATION;
    alt->children = std::move(branches);
    out = std::move(alt);
    return pos;
}

const char * llama_grammar_ast_parser::parse_sequence(
        const char * src,
        const std::string & rule_name,
        bool is_nested,
        ast_node_ptr & out) {
    std::vector<ast_node_ptr> items;
    const char * pos = src;

    while (*pos) {
        ast_node_ptr item;

        if (*pos == '"') {
            // Literal string
            pos++;
            auto lit = std::make_unique<grammar_ast_node>();
            lit->type = grammar_ast_node::LITERAL;
            while (*pos != '"') {
                if (!*pos) {
                    throw std::runtime_error("unexpected end of input");
                }
                auto char_pair = parse_char(pos);
                pos = char_pair.second;
                lit->code_points.push_back(char_pair.first);
            }
            pos = parse_space(pos + 1, is_nested);
            item = std::move(lit);
        } else if (*pos == '[') {
            // Character class
            pos++;
            auto cc = std::make_unique<grammar_ast_node>();
            cc->type = grammar_ast_node::CHAR_CLASS;
            if (*pos == '^') {
                cc->negated = true;
                pos++;
            }
            while (*pos != ']') {
                if (!*pos) {
                    throw std::runtime_error("unexpected end of input");
                }
                auto char_pair = parse_char(pos);
                pos = char_pair.second;
                uint32_t lo = char_pair.first;
                uint32_t hi = lo;
                if (pos[0] == '-' && pos[1] != ']') {
                    if (!pos[1]) {
                        throw std::runtime_error("unexpected end of input");
                    }
                    auto end_pair = parse_char(pos + 1);
                    pos = end_pair.second;
                    hi = end_pair.first;
                }
                cc->ranges.push_back({lo, hi});
            }
            pos = parse_space(pos + 1, is_nested);
            item = std::move(cc);
        } else if (is_word_char(*pos)) {
            // Rule reference
            const char * name_end = parse_name(pos);
            uint32_t ref_id = get_symbol_id(pos, name_end - pos);
            auto ref = std::make_unique<grammar_ast_node>();
            ref->type = grammar_ast_node::RULE_REF;
            ref->rule_name = std::string(pos, name_end - pos);
            ref->rule_id = ref_id;
            pos = parse_space(name_end, is_nested);
            item = std::move(ref);
        } else if (*pos == '(') {
            // Parenthesized group -> inline AST node, no synthetic rule
            pos = parse_space(pos + 1, true);
            pos = parse_alternates(pos, rule_name, true, item);
            if (*pos != ')') {
                throw std::runtime_error(std::string("expecting ')' at ") + pos);
            }
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '!') {
            // Exclusion: !("literal1" | "literal2" | ...)
            pos++;
            if (*pos != '(') {
                throw std::runtime_error(std::string("expecting '(' after '!' at ") + pos);
            }
            pos = parse_space(pos + 1, true);

            auto excl = std::make_unique<grammar_ast_node>();
            excl->type = grammar_ast_node::EXCLUSION;

            // Parse first literal string
            if (*pos != '"') {
                throw std::runtime_error(
                    std::string("expecting '\"' inside !() (only literals allowed) at ") + pos);
            }
            {
                pos++;
                std::vector<uint32_t> str;
                while (*pos != '"') {
                    if (!*pos) {
                        throw std::runtime_error("unexpected end of input");
                    }
                    auto char_pair = parse_char(pos);
                    pos = char_pair.second;
                    str.push_back(char_pair.first);
                }
                pos = parse_space(pos + 1, true);
                if (str.empty()) {
                    throw std::runtime_error("empty string in exclusion");
                }
                excl->excluded_strings.push_back(std::move(str));
            }

            // Parse additional pipe-separated literals
            while (*pos == '|') {
                pos = parse_space(pos + 1, true);
                if (*pos != '"') {
                    throw std::runtime_error(
                        std::string("expecting '\"' inside !() (only literals allowed) at ") + pos);
                }
                pos++;
                std::vector<uint32_t> str;
                while (*pos != '"') {
                    if (!*pos) {
                        throw std::runtime_error("unexpected end of input");
                    }
                    auto char_pair = parse_char(pos);
                    pos = char_pair.second;
                    str.push_back(char_pair.first);
                }
                pos = parse_space(pos + 1, true);
                if (str.empty()) {
                    throw std::runtime_error("empty string in exclusion");
                }
                excl->excluded_strings.push_back(std::move(str));
            }

            if (*pos != ')') {
                throw std::runtime_error(std::string("expecting ')' at ") + pos);
            }
            pos = parse_space(pos + 1, is_nested);
            item = std::move(excl);
        } else if (*pos == '.') {
            // Any char
            auto cc = std::make_unique<grammar_ast_node>();
            cc->type = grammar_ast_node::CHAR_CLASS;
            cc->any_char = true;
            pos = parse_space(pos + 1, is_nested);
            item = std::move(cc);
        } else if (*pos == '*' || *pos == '+' || *pos == '?' || *pos == '{') {
            // Repetition operators - handled below after item is set
            // But we need a preceding item
            if (items.empty()) {
                throw std::runtime_error(std::string("expecting preceding item to */+/?/{ at ") + pos);
            }
            // Wrap the last item in a REPETITION node
            ast_node_ptr last = std::move(items.back());
            items.pop_back();

            auto rep = std::make_unique<grammar_ast_node>();
            rep->type = grammar_ast_node::REPETITION;
            rep->child = std::move(last);

            if (*pos == '*') {
                rep->min_rep = 0;
                rep->max_rep = UINT64_MAX;
                pos = parse_space(pos + 1, is_nested);
            } else if (*pos == '+') {
                rep->min_rep = 1;
                rep->max_rep = UINT64_MAX;
                pos = parse_space(pos + 1, is_nested);
            } else if (*pos == '?') {
                rep->min_rep = 0;
                rep->max_rep = 1;
                pos = parse_space(pos + 1, is_nested);
            } else {
                // {n}, {n,}, {n,m}
                pos = parse_space(pos + 1, is_nested);
                if (!is_digit_char(*pos)) {
                    throw std::runtime_error(std::string("expecting an int at ") + pos);
                }
                const char * int_end = parse_int_str(pos);
                uint64_t min_times = std::stoul(std::string(pos, int_end - pos));
                pos = parse_space(int_end, is_nested);

                uint64_t max_times = UINT64_MAX;

                if (*pos == '}') {
                    max_times = min_times;
                    pos = parse_space(pos + 1, is_nested);
                } else if (*pos == ',') {
                    pos = parse_space(pos + 1, is_nested);
                    if (is_digit_char(*pos)) {
                        const char * int_end2 = parse_int_str(pos);
                        max_times = std::stoul(std::string(pos, int_end2 - pos));
                        pos = parse_space(int_end2, is_nested);
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
                    throw std::runtime_error("number of repetitions exceeds sane defaults");
                }

                rep->min_rep = min_times;
                rep->max_rep = max_times;
            }

            items.push_back(std::move(rep));
            continue;  // don't push item again
        } else {
            break;  // end of sequence (hit '|', ')', newline, etc.)
        }

        items.push_back(std::move(item));
    }

    if (items.empty()) {
        // Empty sequence (epsilon)
        auto seq = std::make_unique<grammar_ast_node>();
        seq->type = grammar_ast_node::SEQUENCE;
        out = std::move(seq);
    } else if (items.size() == 1) {
        out = std::move(items[0]);
    } else {
        auto seq = std::make_unique<grammar_ast_node>();
        seq->type = grammar_ast_node::SEQUENCE;
        seq->children = std::move(items);
        out = std::move(seq);
    }

    return pos;
}

const char * llama_grammar_ast_parser::parse_rule(const char * src) {
    const char * name_end = parse_name(src);
    const char * pos      = parse_space(name_end, false);
    size_t       name_len = name_end - src;
    uint32_t     rule_id  = get_symbol_id(src, name_len);
    const std::string name(src, name_len);

    if (!(pos[0] == ':' && pos[1] == ':' && pos[2] == '=')) {
        throw std::runtime_error(std::string("expecting ::= at ") + pos);
    }
    pos = parse_space(pos + 3, true);

    ast_node_ptr node;
    pos = parse_alternates(pos, name, false, node);

    // Store the rule
    if (rules.size() <= rule_id) {
        rules.resize(rule_id + 1);
    }
    rules[rule_id] = std::move(node);

    if (*pos == '\r') {
        pos += pos[1] == '\n' ? 2 : 1;
    } else if (*pos == '\n') {
        pos++;
    } else if (*pos) {
        throw std::runtime_error(std::string("expecting newline or end at ") + pos);
    }
    return parse_space(pos, true);
}

bool llama_grammar_ast_parser::parse(const char * src) {
    try {
        const char * pos = parse_space(src, true);
        while (*pos) {
            pos = parse_rule(pos);
        }
        // Validate: all rules must be defined
        for (size_t i = 0; i < rules.size(); i++) {
            if (!rules[i]) {
                // Find rule name
                for (const auto & kv : symbol_ids) {
                    if (kv.second == i) {
                        throw std::runtime_error("Undefined rule identifier '" + kv.first + "'");
                    }
                }
                throw std::runtime_error("Undefined rule");
            }
        }
    } catch (const std::exception & err) {
        fprintf(stderr, "%s: error parsing grammar: %s\n", __func__, err.what());
        rules.clear();
        return false;
    }
    return true;
}

// ============================================================
// Optimization passes
// ============================================================

void llama_grammar_ast_parser::classify_terminals() {
    rule_classes.resize(rules.size(), terminal_class::UNKNOWN);

    // Pass 1: mark purely terminal rules
    for (size_t i = 0; i < rules.size(); i++) {
        if (rules[i] && rules[i]->is_purely_terminal()) {
            rule_classes[i] = terminal_class::PURELY_TERMINAL;
        }
    }

    // Pass 2: fixed-point iteration for transitively terminal
    std::vector<bool> in_progress(rules.size(), false);
    for (size_t i = 0; i < rules.size(); i++) {
        if (rule_classes[i] == terminal_class::UNKNOWN) {
            check_transitively_terminal(static_cast<uint32_t>(i), in_progress);
        }
    }

    // Log classification results
    for (size_t i = 0; i < rules.size(); i++) {
        const char * cls = "unknown";
        switch (rule_classes[i]) {
            case terminal_class::PURELY_TERMINAL:       cls = "purely_terminal"; break;
            case terminal_class::TRANSITIVELY_TERMINAL: cls = "transitively_terminal"; break;
            case terminal_class::NON_TERMINAL:          cls = "non_terminal"; break;
            case terminal_class::UNKNOWN:               cls = "unknown"; break;
        }
        // Find rule name
        for (const auto & kv : symbol_ids) {
            if (kv.second == i) {
                LLAMA_LOG_DEBUG("%s: rule '%s' (id=%zu) classified as %s\n",
                                __func__, kv.first.c_str(), i, cls);
                break;
            }
        }
    }
}

bool llama_grammar_ast_parser::check_transitively_terminal(
        uint32_t rule_id, std::vector<bool> & in_progress) {
    if (rule_classes[rule_id] == terminal_class::PURELY_TERMINAL ||
        rule_classes[rule_id] == terminal_class::TRANSITIVELY_TERMINAL) {
        return true;
    }
    if (rule_classes[rule_id] == terminal_class::NON_TERMINAL) {
        return false;
    }
    if (in_progress[rule_id]) {
        // Recursive reference -> non-terminal
        rule_classes[rule_id] = terminal_class::NON_TERMINAL;
        return false;
    }

    in_progress[rule_id] = true;

    // Check all RULE_REFs in this rule's AST
    std::function<bool(const grammar_ast_node &)> check_node;
    check_node = [&](const grammar_ast_node & node) -> bool {
        switch (node.type) {
            case grammar_ast_node::LITERAL:
            case grammar_ast_node::CHAR_CLASS:
            case grammar_ast_node::EXCLUSION:
                return true;
            case grammar_ast_node::RULE_REF:
                return check_transitively_terminal(node.rule_id, in_progress);
            case grammar_ast_node::SEQUENCE:
            case grammar_ast_node::ALTERNATION:
                for (const auto & c : node.children) {
                    if (!check_node(*c)) {
                        return false;
                    }
                }
                return true;
            case grammar_ast_node::REPETITION:
                return node.child && check_node(*node.child);
        }
        return false;
    };

    bool is_terminal = rules[rule_id] && check_node(*rules[rule_id]);
    in_progress[rule_id] = false;

    if (is_terminal) {
        rule_classes[rule_id] = terminal_class::TRANSITIVELY_TERMINAL;
    } else {
        rule_classes[rule_id] = terminal_class::NON_TERMINAL;
    }
    return is_terminal;
}

void llama_grammar_ast_parser::inline_refs_in_node(ast_node_ptr & node) {
    if (!node) return;

    switch (node->type) {
        case grammar_ast_node::LITERAL:
        case grammar_ast_node::CHAR_CLASS:
        case grammar_ast_node::EXCLUSION:
            break;
        case grammar_ast_node::RULE_REF:
            if (node->rule_id < rule_classes.size() &&
                (rule_classes[node->rule_id] == terminal_class::PURELY_TERMINAL ||
                 rule_classes[node->rule_id] == terminal_class::TRANSITIVELY_TERMINAL)) {
                // Clone and recursively inline to measure final size
                auto candidate = rules[node->rule_id]->clone();
                inline_refs_in_node(candidate);
                size_t final_size = candidate->node_count();
                if (final_size > MAX_INLINE_AST_NODES) {
                    LLAMA_LOG_DEBUG("%s: skipping inline of rule '%s' (id=%u, %zu nodes > %d limit)\n",
                                    __func__, node->rule_name.c_str(), node->rule_id,
                                    final_size, MAX_INLINE_AST_NODES);
                    break;
                }
                LLAMA_LOG_DEBUG("%s: inlining terminal rule '%s' (id=%u, %zu nodes)\n",
                                __func__, node->rule_name.c_str(), node->rule_id, final_size);
                node = std::move(candidate);
            }
            break;
        case grammar_ast_node::SEQUENCE:
        case grammar_ast_node::ALTERNATION:
            for (auto & c : node->children) {
                inline_refs_in_node(c);
            }
            break;
        case grammar_ast_node::REPETITION:
            inline_refs_in_node(node->child);
            break;
    }
}

void llama_grammar_ast_parser::inline_terminal_refs() {
    for (size_t i = 0; i < rules.size(); i++) {
        if (rule_classes[i] != terminal_class::PURELY_TERMINAL) {
            inline_refs_in_node(rules[i]);
        }
    }
}

void llama_grammar_ast_parser::flatten_nested(ast_node_ptr & node) {
    if (!node) return;

    switch (node->type) {
        case grammar_ast_node::LITERAL:
        case grammar_ast_node::CHAR_CLASS:
        case grammar_ast_node::RULE_REF:
        case grammar_ast_node::EXCLUSION:
            break;
        case grammar_ast_node::SEQUENCE: {
            // Flatten nested sequences
            std::vector<ast_node_ptr> flat;
            for (auto & c : node->children) {
                flatten_nested(c);
                if (c->type == grammar_ast_node::SEQUENCE) {
                    for (auto & gc : c->children) {
                        flat.push_back(std::move(gc));
                    }
                } else {
                    flat.push_back(std::move(c));
                }
            }
            node->children = std::move(flat);
            if (node->children.size() == 1) {
                node = std::move(node->children[0]);
            }
            break;
        }
        case grammar_ast_node::ALTERNATION: {
            // Flatten nested alternations
            std::vector<ast_node_ptr> flat;
            for (auto & c : node->children) {
                flatten_nested(c);
                if (c->type == grammar_ast_node::ALTERNATION) {
                    for (auto & gc : c->children) {
                        flat.push_back(std::move(gc));
                    }
                } else {
                    flat.push_back(std::move(c));
                }
            }
            node->children = std::move(flat);
            if (node->children.size() == 1) {
                node = std::move(node->children[0]);
            }
            break;
        }
        case grammar_ast_node::REPETITION:
            flatten_nested(node->child);
            break;
    }
}

void llama_grammar_ast_parser::optimize() {
    classify_terminals();
    inline_terminal_refs();
    for (auto & rule : rules) {
        flatten_nested(rule);
    }
}

// ============================================================
// AST to NFA conversion (for purely terminal subtrees)
// ============================================================

grammar_nfa ast_to_nfa(const grammar_ast_node & node) {
    switch (node.type) {
        case grammar_ast_node::LITERAL: {
            if (node.code_points.empty()) {
                return grammar_nfa::epsilon_nfa();
            }
            auto bytes = encode_utf8_bytes(node.code_points[0]);
            grammar_nfa result = grammar_nfa::byte_match(bytes[0], bytes[0]);
            for (size_t i = 1; i < bytes.size(); i++) {
                result = grammar_nfa::concat(std::move(result),
                                             grammar_nfa::byte_match(bytes[i], bytes[i]));
            }
            for (size_t cp_idx = 1; cp_idx < node.code_points.size(); cp_idx++) {
                auto cp_bytes = encode_utf8_bytes(node.code_points[cp_idx]);
                for (size_t i = 0; i < cp_bytes.size(); i++) {
                    result = grammar_nfa::concat(std::move(result),
                                                 grammar_nfa::byte_match(cp_bytes[i], cp_bytes[i]));
                }
            }
            return result;
        }

        case grammar_ast_node::CHAR_CLASS: {
            if (node.any_char) {
                return valid_utf8_char_nfa();
            }

            std::vector<grammar_nfa> alts;
            for (const auto & r : node.ranges) {
                alts.push_back(codepoint_range_to_byte_nfa(r.lo, r.hi));
            }
            grammar_nfa char_nfa = grammar_nfa::alternation(std::move(alts));

            if (node.negated) {
                // Complement: build DFA, complement, intersect with valid UTF-8
                auto positive_dfa = byte_dfa::from_nfa(char_nfa);
                positive_dfa.minimize();
                positive_dfa.complement();
                auto utf8_dfa = build_valid_utf8_dfa();
                auto negated_dfa = byte_dfa::intersect(positive_dfa, utf8_dfa);
                negated_dfa.minimize();

                // Convert back to NFA
                grammar_nfa neg_nfa;
                for (size_t i = 0; i < negated_dfa.transitions.size(); i++) {
                    neg_nfa.add_state();
                }
                neg_nfa.start = negated_dfa.start_state;
                uint32_t nfa_accept = neg_nfa.add_state();
                neg_nfa.accept = nfa_accept;

                for (size_t i = 0; i < negated_dfa.transitions.size(); i++) {
                    if (negated_dfa.accept[i]) {
                        neg_nfa.states[i].epsilon_transitions.push_back(nfa_accept);
                    }
                }
                for (size_t i = 0; i < negated_dfa.transitions.size(); i++) {
                    int b = 0;
                    while (b < 256) {
                        uint16_t target = negated_dfa.transitions[i][b];
                        if (target == 0) { b++; continue; }
                        int start_b = b;
                        while (b < 255 && negated_dfa.transitions[i][b + 1] == target) {
                            b++;
                        }
                        neg_nfa.states[i].byte_transitions.push_back(
                            {(uint8_t)start_b, (uint8_t)b, (uint32_t)target});
                        b++;
                    }
                }
                return neg_nfa;
            }

            return char_nfa;
        }

        case grammar_ast_node::SEQUENCE: {
            if (node.children.empty()) {
                return grammar_nfa::epsilon_nfa();
            }
            grammar_nfa result = ast_to_nfa(*node.children[0]);
            for (size_t i = 1; i < node.children.size(); i++) {
                result = grammar_nfa::concat(std::move(result), ast_to_nfa(*node.children[i]));
            }
            return result;
        }

        case grammar_ast_node::ALTERNATION: {
            std::vector<grammar_nfa> parts;
            for (const auto & c : node.children) {
                parts.push_back(ast_to_nfa(*c));
            }
            return grammar_nfa::alternation(std::move(parts));
        }

        case grammar_ast_node::REPETITION: {
            assert(node.child);
            grammar_nfa child_nfa = ast_to_nfa(*node.child);

            if (node.min_rep == 0 && node.max_rep == UINT64_MAX) {
                // *
                return grammar_nfa::kleene_star(std::move(child_nfa));
            }
            if (node.min_rep == 1 && node.max_rep == UINT64_MAX) {
                // +
                return grammar_nfa::kleene_plus(std::move(child_nfa));
            }
            if (node.min_rep == 0 && node.max_rep == 1) {
                // ?
                std::vector<grammar_nfa> parts;
                parts.push_back(std::move(child_nfa));
                parts.push_back(grammar_nfa::epsilon_nfa());
                return grammar_nfa::alternation(std::move(parts));
            }

            // General {n,m} case: unroll
            grammar_nfa result = grammar_nfa::epsilon_nfa();

            // Mandatory repetitions
            for (uint64_t i = 0; i < node.min_rep; i++) {
                grammar_nfa copy = ast_to_nfa(*node.child);
                result = grammar_nfa::concat(std::move(result), std::move(copy));
            }

            if (node.max_rep == UINT64_MAX) {
                // {n,} = n mandatory + kleene_star
                grammar_nfa star = grammar_nfa::kleene_star(ast_to_nfa(*node.child));
                result = grammar_nfa::concat(std::move(result), std::move(star));
            } else {
                // {n,m} = n mandatory + (m-n) optional
                for (uint64_t i = node.min_rep; i < node.max_rep; i++) {
                    grammar_nfa copy = ast_to_nfa(*node.child);
                    std::vector<grammar_nfa> parts;
                    parts.push_back(std::move(copy));
                    parts.push_back(grammar_nfa::epsilon_nfa());
                    grammar_nfa opt = grammar_nfa::alternation(std::move(parts));
                    result = grammar_nfa::concat(std::move(result), std::move(opt));
                }
            }

            return result;
        }

        case grammar_ast_node::EXCLUSION: {
            // Build exclusion DFA from code point strings, then convert to NFA
            std::vector<std::vector<uint8_t>> needles;
            for (const auto & str : node.excluded_strings) {
                std::vector<uint8_t> bytes;
                for (uint32_t cp : str) {
                    auto cp_bytes = encode_utf8_bytes(cp);
                    bytes.insert(bytes.end(), cp_bytes.begin(), cp_bytes.end());
                }
                needles.push_back(std::move(bytes));
            }
            auto dfa = build_exclusion_dfa(needles);
            dfa.minimize();

            // Convert DFA back to NFA (same pattern as negated char classes)
            grammar_nfa nfa;
            for (size_t i = 0; i < dfa.transitions.size(); i++) {
                nfa.add_state();
            }
            nfa.start = dfa.start_state;
            uint32_t nfa_accept = nfa.add_state();
            nfa.accept = nfa_accept;

            for (size_t i = 0; i < dfa.transitions.size(); i++) {
                if (dfa.accept[i]) {
                    nfa.states[i].epsilon_transitions.push_back(nfa_accept);
                }
            }
            for (size_t i = 0; i < dfa.transitions.size(); i++) {
                int b = 0;
                while (b < 256) {
                    uint16_t target = dfa.transitions[i][b];
                    if (target == 0) { b++; continue; }
                    int start_b = b;
                    while (b < 255 && dfa.transitions[i][b + 1] == target) {
                        b++;
                    }
                    nfa.states[i].byte_transitions.push_back(
                        {(uint8_t)start_b, (uint8_t)b, (uint32_t)target});
                    b++;
                }
            }
            return nfa;
        }

        case grammar_ast_node::RULE_REF:
            // Should not be called on non-terminal nodes
            assert(false && "ast_to_nfa called on RULE_REF node");
            return grammar_nfa::epsilon_nfa();
    }

    return grammar_nfa::epsilon_nfa();
}

// ============================================================
// AST to compiled_grammar compilation
// ============================================================

// Helper: add a DFA to the compiled grammar, deduplicating against existing DFAs.
static uint32_t add_or_reuse_dfa(compiled_grammar & cg, byte_dfa && dfa) {
    for (uint32_t i = 0; i < (uint32_t)cg.dfas.size(); i++) {
        if (cg.dfas[i] == dfa) {
            return i;
        }
    }
    uint32_t id = (uint32_t)cg.dfas.size();
    cg.dfas.push_back(std::move(dfa));
    return id;
}

// Compile a single AST node within a sequence context, accumulating terminal
// NFA fragments and flushing them as DFA segments when non-terminal nodes
// are encountered.
static void compile_node(
        compiled_grammar & cg,
        const grammar_ast_node & node,
        grammar_nfa & pending_nfa,
        bool & has_pending,
        std::vector<compiled_segment> & segments);

// Flush pending terminal NFA as a DFA segment
static void flush_pending(
        compiled_grammar & cg,
        grammar_nfa & pending_nfa,
        bool & has_pending,
        std::vector<compiled_segment> & segments) {
    if (!has_pending) return;
    auto dfa = byte_dfa::from_nfa(pending_nfa);
    dfa.minimize();
    uint32_t dfa_id = add_or_reuse_dfa(cg, std::move(dfa));
    segments.push_back({compiled_segment::DFA_MATCH, dfa_id});
    has_pending = false;
    pending_nfa = grammar_nfa();
}

// Append terminal NFA to pending, flushing first if it's getting too large
static void append_terminal(
        compiled_grammar & cg,
        grammar_nfa & pending_nfa,
        bool & has_pending,
        std::vector<compiled_segment> & segments,
        grammar_nfa && nfa) {
    if (has_pending && pending_nfa.states.size() > MAX_PENDING_NFA_STATES) {
        flush_pending(cg, pending_nfa, has_pending, segments);
    }
    if (!has_pending) {
        pending_nfa = std::move(nfa);
        has_pending = true;
    } else {
        pending_nfa = grammar_nfa::concat(std::move(pending_nfa), std::move(nfa));
    }
}

static void compile_node(
        compiled_grammar & cg,
        const grammar_ast_node & node,
        grammar_nfa & pending_nfa,
        bool & has_pending,
        std::vector<compiled_segment> & segments) {
    switch (node.type) {
        case grammar_ast_node::LITERAL:
        case grammar_ast_node::CHAR_CLASS:
            append_terminal(cg, pending_nfa, has_pending, segments, ast_to_nfa(node));
            break;

        case grammar_ast_node::EXCLUSION: {
            // Build exclusion DFA directly as its own segment (avoids NFA round-trip)
            flush_pending(cg, pending_nfa, has_pending, segments);
            std::vector<std::vector<uint8_t>> needles;
            for (const auto & str : node.excluded_strings) {
                std::vector<uint8_t> bytes;
                for (uint32_t cp : str) {
                    auto cp_bytes = encode_utf8_bytes(cp);
                    bytes.insert(bytes.end(), cp_bytes.begin(), cp_bytes.end());
                }
                needles.push_back(std::move(bytes));
            }
            auto dfa = build_exclusion_dfa(needles);
            dfa.minimize();
            uint32_t dfa_id = add_or_reuse_dfa(cg, std::move(dfa));
            segments.push_back({compiled_segment::DFA_MATCH, dfa_id});
            break;
        }

        case grammar_ast_node::RULE_REF:
            // Non-terminal: flush pending terminals, emit RULE_CALL
            flush_pending(cg, pending_nfa, has_pending, segments);
            segments.push_back({compiled_segment::RULE_CALL, node.rule_id});
            break;

        case grammar_ast_node::SEQUENCE:
            for (const auto & c : node.children) {
                compile_node(cg, *c, pending_nfa, has_pending, segments);
            }
            break;

        case grammar_ast_node::REPETITION:
            if (node.child->is_purely_terminal()) {
                // Terminal repetition: build NFA, check size
                grammar_nfa rep_nfa = ast_to_nfa(node);
                if (rep_nfa.states.size() <= MAX_PENDING_NFA_STATES) {
                    append_terminal(cg, pending_nfa, has_pending, segments, std::move(rep_nfa));
                    break;
                }
                // NFA too large: fall through to synthetic rule path
            }
            {
                // Non-terminal repetition: can't merge into NFA
                // Must create a synthetic rule for the repetition
                flush_pending(cg, pending_nfa, has_pending, segments);

                if (node.max_rep == UINT64_MAX) {
                    uint32_t body_rule_id = (uint32_t)cg.rules.size();
                    cg.rules.emplace_back();
                    // Unbounded: body_rule ::= child body_rule | epsilon
                    grammar_nfa child_pending;
                    bool child_has_pending = false;
                    std::vector<compiled_segment> child_segs;
                    compile_node(cg, *node.child, child_pending, child_has_pending, child_segs);
                    flush_pending(cg, child_pending, child_has_pending, child_segs);
                    child_segs.push_back({compiled_segment::RULE_CALL, body_rule_id});

                    // Build alternates locally, then assign (avoids reference invalidation)
                    std::vector<std::vector<compiled_segment>> body_alts;
                    body_alts.push_back(std::move(child_segs));
                    body_alts.push_back({}); // epsilon
                    cg.rules[body_rule_id].alternates = std::move(body_alts);

                    // Emit min_rep inline copies + call to body_rule
                    for (uint64_t i = 0; i < node.min_rep; i++) {
                        compile_node(cg, *node.child, pending_nfa, has_pending, segments);
                    }
                    flush_pending(cg, pending_nfa, has_pending, segments);
                    segments.push_back({compiled_segment::RULE_CALL, body_rule_id});
                } else {
                    // Bounded: {n,m}
                    for (uint64_t i = 0; i < node.min_rep; i++) {
                        compile_node(cg, *node.child, pending_nfa, has_pending, segments);
                    }
                    uint64_t n_opt = node.max_rep - node.min_rep;
                    if (n_opt > 0) {
                        std::vector<uint32_t> opt_ids;
                        for (uint64_t i = 0; i < n_opt; i++) {
                            opt_ids.push_back((uint32_t)cg.rules.size());
                            cg.rules.emplace_back();
                        }

                        for (uint64_t i = 0; i < n_opt; i++) {
                            grammar_nfa opt_pending;
                            bool opt_has_pending = false;
                            std::vector<compiled_segment> opt_segs;
                            compile_node(cg, *node.child, opt_pending, opt_has_pending, opt_segs);
                            flush_pending(cg, opt_pending, opt_has_pending, opt_segs);
                            if (i + 1 < n_opt) {
                                opt_segs.push_back({compiled_segment::RULE_CALL, opt_ids[i + 1]});
                            }
                            // Assign alternates locally to avoid invalidation
                            std::vector<std::vector<compiled_segment>> opt_alts;
                            opt_alts.push_back(std::move(opt_segs));
                            opt_alts.push_back({}); // epsilon
                            cg.rules[opt_ids[i]].alternates = std::move(opt_alts);
                        }

                        flush_pending(cg, pending_nfa, has_pending, segments);
                        segments.push_back({compiled_segment::RULE_CALL, opt_ids[0]});
                    }
                }
            }
            break;

        case grammar_ast_node::ALTERNATION:
            if (node.is_purely_terminal()) {
                // Terminal alternation: build NFA, check size before merging
                grammar_nfa alt_nfa = ast_to_nfa(node);
                if (alt_nfa.states.size() <= MAX_PENDING_NFA_STATES) {
                    append_terminal(cg, pending_nfa, has_pending, segments, std::move(alt_nfa));
                    break;
                }
                // NFA too large: fall through to synthetic rule path
            }
            {
                // Mixed alternation: flush pending, create synthetic rule
                flush_pending(cg, pending_nfa, has_pending, segments);

                uint32_t alt_rule_id = (uint32_t)cg.rules.size();
                cg.rules.emplace_back();

                std::vector<std::vector<compiled_segment>> alt_alts;
                for (const auto & branch : node.children) {
                    grammar_nfa branch_pending;
                    bool branch_has_pending = false;
                    std::vector<compiled_segment> branch_segs;
                    compile_node(cg, *branch, branch_pending, branch_has_pending, branch_segs);
                    flush_pending(cg, branch_pending, branch_has_pending, branch_segs);
                    alt_alts.push_back(std::move(branch_segs));
                }
                cg.rules[alt_rule_id].alternates = std::move(alt_alts);

                segments.push_back({compiled_segment::RULE_CALL, alt_rule_id});
            }
            break;
    }
}

// BFS on AST RULE_REFs to find rules reachable from start_rule
static std::vector<bool> compute_reachable(
        const std::vector<ast_node_ptr> & rules,
        uint32_t start_rule_index) {
    std::vector<bool> reachable(rules.size(), false);
    std::vector<uint32_t> queue;
    reachable[start_rule_index] = true;
    queue.push_back(start_rule_index);

    // Collect RULE_REFs from an AST node
    std::function<void(const grammar_ast_node &)> collect_refs;
    collect_refs = [&](const grammar_ast_node & node) {
        switch (node.type) {
            case grammar_ast_node::RULE_REF:
                if (!reachable[node.rule_id]) {
                    reachable[node.rule_id] = true;
                    queue.push_back(node.rule_id);
                }
                break;
            case grammar_ast_node::SEQUENCE:
            case grammar_ast_node::ALTERNATION:
                for (const auto & c : node.children) {
                    collect_refs(*c);
                }
                break;
            case grammar_ast_node::REPETITION:
                if (node.child) collect_refs(*node.child);
                break;
            case grammar_ast_node::LITERAL:
            case grammar_ast_node::CHAR_CLASS:
            case grammar_ast_node::EXCLUSION:
                break;
        }
    };

    while (!queue.empty()) {
        uint32_t rid = queue.back();
        queue.pop_back();
        if (rules[rid]) {
            collect_refs(*rules[rid]);
        }
    }

    return reachable;
}

compiled_grammar compile_grammar_from_ast(
        const std::vector<ast_node_ptr> & rules,
        uint32_t start_rule_index) {
    compiled_grammar cg;
    cg.start_rule = start_rule_index;
    cg.rules.resize(rules.size());

    // Determine which rules are reachable from the start rule
    auto reachable = compute_reachable(rules, start_rule_index);

    size_t n_skipped = 0;
    for (size_t i = 0; i < rules.size(); i++) {
        if (rules[i] && !reachable[i]) n_skipped++;
    }
    if (n_skipped > 0) {
        LLAMA_LOG_INFO("%s: skipping %zu unreachable rules\n", __func__, n_skipped);
    }

    for (size_t rule_idx = 0; rule_idx < rules.size(); rule_idx++) {
        const auto & ast = rules[rule_idx];
        if (!ast || !reachable[rule_idx]) continue;

        // Collect alternates into a local vector first, because compile_node
        // may append synthetic rules to cg.rules, invalidating references.
        std::vector<std::vector<compiled_segment>> alternates;

        if (ast->type == grammar_ast_node::ALTERNATION) {
            for (const auto & branch : ast->children) {
                grammar_nfa pending_nfa;
                bool has_pending = false;
                std::vector<compiled_segment> segments;
                compile_node(cg, *branch, pending_nfa, has_pending, segments);
                flush_pending(cg, pending_nfa, has_pending, segments);
                alternates.push_back(std::move(segments));
            }
        } else {
            grammar_nfa pending_nfa;
            bool has_pending = false;
            std::vector<compiled_segment> segments;
            compile_node(cg, *ast, pending_nfa, has_pending, segments);
            flush_pending(cg, pending_nfa, has_pending, segments);
            alternates.push_back(std::move(segments));
        }

        if (alternates.empty()) {
            alternates.push_back({});
        }

        cg.rules[rule_idx].alternates = std::move(alternates);
    }

    return cg;
}
