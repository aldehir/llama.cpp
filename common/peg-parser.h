#pragma once

#include <nlohmann/json_fwd.hpp>

#include <memory>
#include <unordered_map>
#include <string>
#include <string_view>
#include <functional>
#include <vector>
#include <variant>

// non-thread-safe
void common_peg_debug_set(bool enabled);

struct common_grammar_builder;

class common_peg_parser_builder;

using common_peg_parser_id = size_t;
constexpr common_peg_parser_id COMMON_PEG_INVALID_PARSER_ID = static_cast<common_peg_parser_id>(-1);

using common_peg_ast_id = size_t;
constexpr common_peg_ast_id COMMON_PEG_INVALID_AST_ID = static_cast<common_peg_ast_id>(-1);

using common_peg_value_id = size_t;
constexpr common_peg_value_id COMMON_PEG_INVALID_VALUE_ID = static_cast<common_peg_value_id>(-1);

// Lightweight wrapper around common_peg_parser_id for convenience
class common_peg_parser {
    common_peg_parser_id id_;
    common_peg_parser_builder & builder_;

  public:
    common_peg_parser(const common_peg_parser & other) : id_(other.id_), builder_(other.builder_) {}
    common_peg_parser(common_peg_parser_id id, common_peg_parser_builder & builder) : id_(id), builder_(builder) {}

    common_peg_parser & operator=(const common_peg_parser & other);
    common_peg_parser & operator+=(const common_peg_parser & other);
    common_peg_parser & operator|=(const common_peg_parser & other);

    operator common_peg_parser_id() const { return id_; }
    common_peg_parser_id id() const { return id_; }

    common_peg_parser_builder & builder() const { return builder_; }

    // Creates a sequence
    common_peg_parser operator+(const common_peg_parser & other) const;

    // Creates a sequence separated by spaces.
    common_peg_parser operator<<(const common_peg_parser & other) const;

    // Creates a choice
    common_peg_parser operator|(const common_peg_parser & other) const;

    common_peg_parser operator+(const char * str) const;
    common_peg_parser operator+(const std::string & str) const;
    common_peg_parser operator<<(const char * str) const;
    common_peg_parser operator<<(const std::string & str) const;
    common_peg_parser operator|(const char * str) const;
    common_peg_parser operator|(const std::string & str) const;
};

common_peg_parser operator+(const char * str, const common_peg_parser & p);
common_peg_parser operator+(const std::string & str, const common_peg_parser & p);
common_peg_parser operator<<(const char * str, const common_peg_parser & p);
common_peg_parser operator<<(const std::string & str, const common_peg_parser & p);
common_peg_parser operator|(const char * str, const common_peg_parser & p);
common_peg_parser operator|(const std::string & str, const common_peg_parser & p);

enum common_peg_parse_result_type {
    COMMON_PEG_PARSE_RESULT_FAIL            = 0,
    COMMON_PEG_PARSE_RESULT_SUCCESS         = 1,
    COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT = 2,
    COMMON_PEG_PARSE_RESULT_FATAL           = 3,
};

const char * common_peg_parse_result_type_name(common_peg_parse_result_type type);

struct common_peg_value {
    virtual ~common_peg_value() = default;
};

using common_peg_value_ptr = std::shared_ptr<common_peg_value>;

class common_peg_value_store {
    struct stored_value {
        common_peg_value_ptr value;
        std::string rule_name;
    };
    std::vector<stored_value> values_;

  public:
    common_peg_value_id add(common_peg_value_ptr val, const std::string & rule_name = "") {
        common_peg_value_id id = values_.size();
        values_.push_back({std::move(val), rule_name});
        return id;
    }

    common_peg_value_ptr get(common_peg_value_id id) const {
        if (id == COMMON_PEG_INVALID_VALUE_ID || id >= values_.size()) {
            return nullptr;
        }
        return values_[id].value;
    }

    std::string_view rule(common_peg_value_id id) const {
        if (id == COMMON_PEG_INVALID_VALUE_ID || id >= values_.size()) {
            return "";
        }
        return values_[id].rule_name;
    }

    template <typename T>
    std::shared_ptr<T> get_as(common_peg_value_id id) const {
        return std::dynamic_pointer_cast<T>(get(id));
    }

    void clear() { values_.clear(); }
    size_t size() const { return values_.size(); }
};

class common_peg_semantic_context {
    std::string_view text_;
    const std::vector<common_peg_value_id>& child_ids_;
    const common_peg_value_store& store_;

  public:
    common_peg_semantic_context(
        std::string_view text,
        const std::vector<common_peg_value_id>& children,
        const common_peg_value_store& store
    ) : text_(text), child_ids_(children), store_(store) {}

    std::string_view text() const { return text_; }

    size_t size() const { return child_ids_.size(); }
    bool empty() const { return child_ids_.empty(); }

    template <typename T>
    std::shared_ptr<T> get(size_t i) const {
        if (i >= child_ids_.size()) {
            return nullptr;
        }
        return store_.get_as<T>(child_ids_[i]);
    }

    template <typename T>
    std::shared_ptr<T> find(size_t start = 0) const {
        for (size_t i = start; i < child_ids_.size(); i++) {
            if (auto p = store_.get_as<T>(child_ids_[i])) {
                return p;
            }
        }
        return nullptr;
    }

    template<typename T, typename Base = T>
    std::vector<std::shared_ptr<Base>> slice(size_t start, size_t end = SIZE_MAX) const {
        std::vector<std::shared_ptr<Base>> result;
        size_t actual_end = std::min(end, child_ids_.size());
        for (size_t i = start; i < actual_end; i++) {
            if (auto p = store_.get_as<T>(child_ids_[i])) {
                result.push_back(std::move(p));
            }
        }
        return result;
    }

    template<typename T, typename Base = T>
    std::vector<std::shared_ptr<Base>> all() const {
        return slice<T, Base>(0);
    }

    common_peg_value_ptr raw(size_t i) const {
        if (i >= child_ids_.size()) {
            return nullptr;
        }
        return store_.get(child_ids_[i]);
    }

    common_peg_value_ptr single() const {
        return (child_ids_.size() == 1) ? store_.get(child_ids_[0]) : nullptr;
    }

    template <typename K, typename V = K>
    std::pair<std::shared_ptr<K>, std::shared_ptr<V>> pair() const {
        return { get<K>(0), get<V>(1) };
    }

    std::string_view rule(size_t i) const {
        if (i >= child_ids_.size()) {
            return "";
        }
        return store_.rule(child_ids_[i]);
    }
};

struct common_peg_ast_node {
    common_peg_ast_id id;
    std::string rule;
    std::string tag;
    size_t start;
    size_t end;
    std::string_view text;
    std::vector<common_peg_ast_id> children;

    bool is_partial = false;
};

struct common_peg_parse_result;

using common_peg_semantic_action = std::function<
    common_peg_value_ptr(const common_peg_semantic_context& ctx)
>;

using common_peg_ast_visitor = std::function<void(const common_peg_ast_node & node)>;

class common_peg_ast_arena {
    std::vector<common_peg_ast_node> nodes_;
  public:
    common_peg_ast_id add_node(
        const std::string & rule,
        const std::string & tag,
        size_t start,
        size_t end,
        std::string_view text,
        std::vector<common_peg_ast_id> children,
        bool is_partial = false
    ) {
        common_peg_ast_id id = nodes_.size();
        nodes_.push_back({id, rule, tag, start, end, text, std::move(children), is_partial});
        return id;
    }

    const common_peg_ast_node & get(common_peg_ast_id id) const { return nodes_.at(id); }

    size_t size() const { return nodes_.size(); }

    void clear() { nodes_.clear(); }

    void visit(common_peg_ast_id id, const common_peg_ast_visitor & visitor) const;
    void visit(const common_peg_parse_result & result, const common_peg_ast_visitor & visitor) const;
};

struct common_peg_parse_result {
    common_peg_parse_result_type type = COMMON_PEG_PARSE_RESULT_FAIL;
    size_t start = 0;
    size_t end = 0;

    std::vector<common_peg_ast_id> nodes;
    std::vector<common_peg_value_id> values;

    common_peg_parse_result() = default;

    common_peg_parse_result(common_peg_parse_result_type type, size_t start)
        : type(type), start(start), end(start) {}

    common_peg_parse_result(common_peg_parse_result_type type, size_t start, size_t end)
        : type(type), start(start), end(end) {}

    common_peg_parse_result(common_peg_parse_result_type type, size_t start, size_t end, std::vector<common_peg_ast_id> nodes)
        : type(type), start(start), end(end), nodes(std::move(nodes)) {}

    common_peg_parse_result(common_peg_parse_result_type type, size_t start, size_t end, std::vector<common_peg_ast_id> nodes, std::vector<common_peg_value_id> values)
        : type(type), start(start), end(end), nodes(std::move(nodes)), values(std::move(values)) {}

    bool fail() const { return type == COMMON_PEG_PARSE_RESULT_FAIL; }
    bool fatal() const { return type == COMMON_PEG_PARSE_RESULT_FATAL; }
    bool need_more_input() const { return type == COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT; }
    bool success() const { return type == COMMON_PEG_PARSE_RESULT_SUCCESS; }
};

struct common_peg_parse_error {
    size_t position;
    std::string message;
    std::string rule;
};

class common_peg_cache {
    struct key_hash {
        size_t operator()(const std::pair<common_peg_parser_id, size_t> & k) const {
            // Combine parser id and position into a single hash
            return std::hash<size_t>()(k.first) ^ (std::hash<size_t>()(k.second) << 1);
        }
    };

    std::unordered_map<std::pair<common_peg_parser_id, size_t>, common_peg_parse_result, key_hash> cache_;

  public:
    bool get(common_peg_parser_id id, size_t pos, common_peg_parse_result & result) const {
        auto it = cache_.find({id, pos});
        if (it != cache_.end()) {
            result = it->second;
            return true;
        }
        return false;
    }

    void put(common_peg_parser_id id, size_t pos, const common_peg_parse_result & result) {
        cache_[{id, pos}] = result;
    }

    void clear() { cache_.clear(); }
    size_t size() const { return cache_.size(); }
};

struct common_peg_parse_context {
    std::string input;
    bool is_partial;
    common_peg_ast_arena ast;
    common_peg_value_store values;

    int parse_depth;

    std::vector<common_peg_parse_error> errors;
    size_t furthest_position = 0;

    std::unique_ptr<common_peg_cache> cache;

    common_peg_parse_context()
        : is_partial(false), parse_depth(0) {}

    common_peg_parse_context(const std::string & input)
        : input(input), is_partial(false), parse_depth(0) {}

    common_peg_parse_context(const std::string & input, bool is_partial)
        : input(input), is_partial(is_partial), parse_depth(0) {}

    void add_error(size_t pos, const std::string & message, const std::string & rule = "");

    void enable_cache() { cache = std::make_unique<common_peg_cache>(); }
    std::unique_ptr<common_peg_cache> release_cache() { return std::move(cache); }
    void set_cache(std::unique_ptr<common_peg_cache> c) { cache = std::move(c); }
};

class common_peg_arena;

// Parser variants
struct common_peg_epsilon_parser {};

struct common_peg_start_parser {};

struct common_peg_end_parser {};

struct common_peg_literal_parser {
    std::string literal;
};

struct common_peg_sequence_parser {
    std::vector<common_peg_parser_id> children;
};

struct common_peg_choice_parser {
    std::vector<common_peg_parser_id> children;
};

struct common_peg_repetition_parser {
    common_peg_parser_id child;
    int min_count;
    int max_count;  // -1 for unbounded
};

struct common_peg_and_parser {
    common_peg_parser_id child;
};

struct common_peg_not_parser {
    common_peg_parser_id child;
};

struct common_peg_any_parser {};

struct common_peg_space_parser {};

struct common_peg_chars_parser {
    struct char_range {
        uint32_t start;
        uint32_t end;
        bool contains(uint32_t codepoint) const { return codepoint >= start && codepoint <= end; }
    };

    std::string pattern;
    std::vector<char_range> ranges;
    bool negated;
    int min_count;
    int max_count;  // -1 for unbounded
};

struct common_peg_json_string_parser {};

struct common_peg_until_parser {
    std::vector<std::string> delimiters;
};

struct common_peg_schema_parser {
    common_peg_parser_id child;
    std::string name;
    std::shared_ptr<nlohmann::ordered_json> schema;

    // Indicates if the GBNF should accept a raw string that matches the schema.
    bool raw;
};

struct common_peg_rule_parser {
    std::string name;
    common_peg_parser_id child;
    bool trigger;
    common_peg_semantic_action action;  // optional semantic action
};

struct common_peg_ref_parser {
    std::string name;
};

struct common_peg_atomic_parser {
    common_peg_parser_id child;
};

struct common_peg_tag_parser {
    common_peg_parser_id child;
    std::string tag;
};

struct common_peg_cut_parser {};

struct common_peg_expect_parser {
    common_peg_parser_id child;
    std::string message;
};

// Variant holding all parser types
using common_peg_parser_variant = std::variant<
    common_peg_epsilon_parser,
    common_peg_start_parser,
    common_peg_end_parser,
    common_peg_literal_parser,
    common_peg_sequence_parser,
    common_peg_choice_parser,
    common_peg_repetition_parser,
    common_peg_and_parser,
    common_peg_not_parser,
    common_peg_any_parser,
    common_peg_space_parser,
    common_peg_chars_parser,
    common_peg_json_string_parser,
    common_peg_until_parser,
    common_peg_schema_parser,
    common_peg_rule_parser,
    common_peg_ref_parser,
    common_peg_atomic_parser,
    common_peg_tag_parser,
    common_peg_cut_parser,
    common_peg_expect_parser
>;

class common_peg_arena {
    std::vector<common_peg_parser_variant> parsers_;
    std::unordered_map<std::string, common_peg_parser_id> rules_;
    common_peg_parser_id root_ = COMMON_PEG_INVALID_PARSER_ID;

  public:
    const common_peg_parser_variant & get(common_peg_parser_id id) const { return parsers_.at(id); }
    common_peg_parser_variant & get(common_peg_parser_id id) { return parsers_.at(id); }

    size_t size() const { return parsers_.size(); }
    bool empty() const { return parsers_.empty(); }

    common_peg_parser_id get_rule(const std::string & name) const;
    bool has_rule(const std::string & name) const { return rules_.find(name) != rules_.end(); }

    common_peg_parser_id root() const { return root_; }
    void set_root(common_peg_parser_id id) { root_ = id; }

    common_peg_parse_result parse(common_peg_parse_context & ctx, size_t start = 0) const;
    common_peg_parse_result parse(common_peg_parser_id id, common_peg_parse_context & ctx, size_t start) const;

    void resolve_refs();

    void build_grammar(const common_grammar_builder & builder, bool lazy = false) const;

    std::string dump(common_peg_parser_id id) const;

    nlohmann::json to_json() const;
    static common_peg_arena from_json(const nlohmann::json & j);

    std::string save() const;
    void load(const std::string & data);

    friend class common_peg_parser_builder;

  private:
    common_peg_parser_id add_parser(common_peg_parser_variant parser);
    void add_rule(const std::string & name, common_peg_parser_id id);

    common_peg_parser_id resolve_ref(common_peg_parser_id id);
};

class common_peg_parser_builder {
    friend class common_peg_parser;

    common_peg_arena arena_;
    common_peg_parser_id token_space_ = COMMON_PEG_INVALID_PARSER_ID;
    common_peg_parser_id word_boundary_ = COMMON_PEG_INVALID_PARSER_ID;

    common_peg_parser wrap(common_peg_parser_id id) { return common_peg_parser(id, *this); }
    common_peg_parser add(const common_peg_parser_variant & p) { return wrap(arena_.add_parser(p)); }

    // flattens left side if it's a choice
    common_peg_parser choice_flatten_left(common_peg_parser_id left, common_peg_parser_id right);

  public:
    common_peg_parser_builder();

    // Match nothing, always succeed.
    //   S -> ε
    common_peg_parser eps() { return add(common_peg_epsilon_parser{}); }

    // Matches the start of the input.
    //   S -> ^
    common_peg_parser start() { return add(common_peg_start_parser{}); }

    // Matches the end of the input.
    //   S -> $
    common_peg_parser end() { return add(common_peg_end_parser{}); }

    // Matches an exact literal string.
    //   S -> "hello"
    common_peg_parser literal(const std::string & literal) { return add(common_peg_literal_parser{literal}); }

    // Matches a sequence of parsers in order, all must succeed.
    //   S -> A B C
    common_peg_parser sequence() { return add(common_peg_sequence_parser{}); }
    common_peg_parser sequence(const std::vector<common_peg_parser_id> & parsers);
    common_peg_parser sequence(const std::vector<common_peg_parser> & parsers);
    common_peg_parser sequence(std::initializer_list<common_peg_parser> parsers);

    // Matches the first parser that succeeds from a list of alternatives.
    //   S -> A | B | C
    common_peg_parser choice() { return add(common_peg_choice_parser{}); }
    common_peg_parser choice(const std::vector<common_peg_parser_id> & parsers);
    common_peg_parser choice(const std::vector<common_peg_parser> & parsers);
    common_peg_parser choice(std::initializer_list<common_peg_parser> parsers);

    // Matches one or more repetitions of a parser.
    //   S -> A+
    common_peg_parser one_or_more(const common_peg_parser & p) { return repeat(p, 1, -1); }

    // Matches zero or more repetitions of a parser, always succeeds.
    //   S -> A*
    common_peg_parser zero_or_more(const common_peg_parser & p) { return repeat(p, 0, -1); }

    // Matches zero or one occurrence of a parser, always succeeds.
    //   S -> A?
    common_peg_parser optional(const common_peg_parser & p) { return repeat(p, 0, 1); }

    // Positive lookahead: succeeds if child parser succeeds, consumes no input.
    //   S -> &A
    common_peg_parser peek(const common_peg_parser & p) { return add(common_peg_and_parser{p}); }

    // Negative lookahead: succeeds if child parser fails, consumes no input.
    //   S -> !A
    common_peg_parser negate(const common_peg_parser & p) { return add(common_peg_not_parser{p}); }

    // Matches any single character.
    //   S -> .
    common_peg_parser any() { return add(common_peg_any_parser{}); }

    // Matches between min and max repetitions of characters from a character class.
    //   S -> [a-z]{m,n}
    //
    // Use -1 for max to represent unbounded repetition (equivalent to {m,})
    common_peg_parser chars(const std::string & classes, int min = 1, int max = -1);

    // Creates a lightweight reference to a named rule (resolved during build()).
    // Use this for forward references in recursive grammars.
    //   expr_ref -> expr
    common_peg_parser ref(const std::string & name) { return add(common_peg_ref_parser{name}); }

    // Matches zero or more whitespace characters (space, tab, newline).
    //   S -> [ \t\n]*
    common_peg_parser space() { return add(common_peg_space_parser{}); }

    // Sets the whitespace parser to use for token() and keyword().
    // If not set, token() returns the parser unchanged.
    void set_token_space(const common_peg_parser & p) { token_space_ = p.id(); }

    // Sets the word boundary parser for keyword().
    // keyword() will use negate(word_boundary) to ensure the keyword isn't a prefix.
    // If not set, keyword() behaves like token().
    void set_word_boundary(const common_peg_parser & p) { word_boundary_ = p.id(); }

    // Wraps a parser to consume trailing whitespace (as configured by set_token_space).
    //   token(p) -> p token_space
    common_peg_parser token(const common_peg_parser & p);
    common_peg_parser token(const std::string & literal);

    // Creates a keyword parser: matches literal, checks word boundary, consumes trailing whitespace.
    //   keyword(kw) -> kw !word_boundary token_space
    common_peg_parser keyword(const std::string & kw);

    // Matches all characters until a delimiter is found (delimiter not consumed).
    //   S -> (!delim .)*
    common_peg_parser until(const std::string & delimiter) { return add(common_peg_until_parser{{delimiter}}); }

    // Matches all characters until one of the delimiters in the list is found (delimiter not consumed).
    //   S -> (!delim .)*
    common_peg_parser until_one_of(const std::vector<std::string> & delimiters) { return add(common_peg_until_parser{delimiters}); }

    // Matches everything
    //   S -> .*
    common_peg_parser rest() { return until_one_of({}); }

    // Matches between min and max repetitions of a parser (inclusive).
    //   S -> A{m,n}
    // Use -1 for max to represent unbounded repetition (equivalent to {m,})
    common_peg_parser repeat(const common_peg_parser & p, int min, int max) { return add(common_peg_repetition_parser{p, min,max}); }

    // Matches exactly n repetitions of a parser.
    //   S -> A{n}
    common_peg_parser repeat(const common_peg_parser & p, int n) { return repeat(p, n, n); }

    // Creates a complete JSON parser supporting objects, arrays, strings, numbers, booleans, and null.
    //   value -> object | array | string | number | true | false | null
    common_peg_parser json();
    common_peg_parser json_object();
    common_peg_parser json_string();
    common_peg_parser json_array();
    common_peg_parser json_number();
    common_peg_parser json_bool();
    common_peg_parser json_null();

    // Matches JSON string content without the surrounding quotes.
    // Useful for extracting content within a JSON string.
    common_peg_parser json_string_content();

    // Matches a JSON object member with a key and associated parser as the
    // value.
    common_peg_parser json_member(const std::string & key, const common_peg_parser & p);

    // Wraps a parser with JSON schema metadata for grammar generation.
    // Used internally to convert JSON schemas to GBNF grammar rules.
    common_peg_parser schema(const common_peg_parser & p, const std::string & name, const nlohmann::ordered_json & schema, bool raw = false);

    // Creates a named rule, stores it in the grammar, and returns a ref.
    // If trigger=true, marks this rule as an entry point for lazy grammar generation.
    //   auto json = p.rule("json", json_obj | json_arr | ...)
    common_peg_parser rule(const std::string & name, const common_peg_parser & p, bool trigger = false);

    // Creates a named rule using a builder function, and returns a ref.
    // If trigger=true, marks this rule as an entry point for lazy grammar generation.
    //   auto json = p.rule("json", [&]() { return json_object() | json_array() | ... })
    common_peg_parser rule(const std::string & name, const std::function<common_peg_parser()> & builder, bool trigger = false);

    common_peg_parser rule(const std::string & name, const common_peg_parser & p, bool trigger, common_peg_semantic_action action);
    common_peg_parser rule(const std::string & name, const std::function<common_peg_parser()> & builder, bool trigger, common_peg_semantic_action action);

    common_peg_parser action_rule(const std::string & name, const common_peg_parser & p, common_peg_semantic_action action) {
        return rule(name, p, false, std::move(action));
    }
    common_peg_parser action_rule(const std::string & name, const std::function<common_peg_parser()> & builder, common_peg_semantic_action action) {
        return rule(name, builder, false, std::move(action));
    }

    // Creates a trigger rule. When generating a lazy grammar from the parser,
    // only trigger rules and descendents are emitted.
    common_peg_parser trigger_rule(const std::string & name, const common_peg_parser & p) { return rule(name, p, true); }
    common_peg_parser trigger_rule(const std::string & name, const std::function<common_peg_parser()> & builder) { return rule(name, builder, true); }

    // Creates an atomic parser. Atomic parsers do not create an AST node if
    // the child results in a partial parse, i.e. NEEDS_MORE_INPUT. This is
    // intended for situations where partial output is undesirable.
    common_peg_parser atomic(const common_peg_parser & p) { return add(common_peg_atomic_parser{p}); }

    // Tags create nodes in the generated AST for semantic purposes.
    // Unlike rules, you can tag multiple nodes with the same tag.
    common_peg_parser tag(const std::string & tag, const common_peg_parser & p) { return add(common_peg_tag_parser{p.id(), tag}); }

    // Cut commits to the current choice branch, preventing backtracking on subsequent failure.
    // Used in sequences: seq(a, cut(), b)
    common_peg_parser cut() { return add(common_peg_cut_parser{}); }

    // Expect wraps a parser with an error message that is recorded on failure.
    common_peg_parser expect(const common_peg_parser & p, const std::string & message);
    common_peg_parser expect(const std::string & literal, const std::string & message);

    void set_root(const common_peg_parser & p);

    common_peg_arena build();
};

// Helper function for building parsers
common_peg_arena build_peg_parser(const std::function<common_peg_parser(common_peg_parser_builder & builder)> & fn);
