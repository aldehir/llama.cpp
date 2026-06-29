#include "recipe.h"
#include "lexer.h"
#include "runtime.h"

#include <cassert>
#include <cctype>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace jinja {
namespace recipe {

// The recipe dialect is an evaluation frontend for the jinja engine. It shares
// jinja's token, AST and runtime, but uses a much smaller, total-by-construction
// grammar (no for/macro/calls, floor division only):
//
//   program := ("default" TYPE)? stmt*
//   stmt    := "type" "=" expr
//            | "set" IDENT "=" expr
//            | "if" expr stmt* ("elif" expr stmt*)* ("else" stmt*)? "endif"
//   expr    := or
//   or      := and (("or") and)*
//   and     := not (("and") not)*
//   not     := "not" not | cmp
//   cmp     := add (("=="|"!="|"<"|"<="|">"|">=") add)?
//   add     := mul (("+"|"-") mul)*
//   mul     := un  (("*"|"//"|"%") un)*
//   un      := "-" un | atom
//   atom    := INT | STRING | TYPE | IDENT | "(" expr ")"

namespace {

const std::map<char, char> escape_chars = {
    {'n', '\n'}, {'t', '\t'}, {'r', '\r'}, {'b', '\b'}, {'f', '\f'},
    {'v', '\v'}, {'\\', '\\'}, {'\'', '\''}, {'\"', '\"'},
};

bool is_word_start(char c) {
    return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
}

bool is_word(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

bool is_digit(char c) {
    return std::isdigit(static_cast<unsigned char>(c));
}

// Words that may not be used as a plain expression atom (operators and block
// keywords). NOTE: "type" is intentionally absent - it doubles as a readable
// binding, e.g. `type == Q4_K`.
bool is_reserved_atom(const std::string & w) {
    static const std::set<std::string> reserved = {
        "default", "set", "if", "elif", "else", "endif", "and", "or", "not",
    };
    return reserved.count(w) != 0;
}

// Words that may not be a "set"/"type" assignment target.
bool is_reserved_name(const std::string & w) {
    static const std::set<std::string> reserved = {
        "default", "set", "type", "if", "elif", "else", "endif",
        "and", "or", "not",
    };
    return reserved.count(w) != 0;
}

} // namespace

lexer_result tokenize(const std::string & source) {
    std::vector<token> tokens;
    const std::string & src = source; // kept as-is for error reporting
    const size_t n = src.size();
    size_t pos = 0;

    while (pos < n) {
        const char c = src[pos];

        // whitespace (including newlines) is insignificant
        if (std::isspace(static_cast<unsigned char>(c))) {
            ++pos;
            continue;
        }

        // line comment: # ... <newline>
        if (c == '#') {
            while (pos < n && src[pos] != '\n') {
                ++pos;
            }
            continue;
        }

        const size_t start = pos;

        // identifiers / keywords
        if (is_word_start(c)) {
            std::string word;
            while (pos < n && is_word(src[pos])) {
                word += src[pos++];
            }
            tokens.push_back({token::identifier, std::move(word), start});
            continue;
        }

        // integer literals
        if (is_digit(c)) {
            std::string num;
            while (pos < n && is_digit(src[pos])) {
                num += src[pos++];
            }
            tokens.push_back({token::numeric_literal, std::move(num), start});
            continue;
        }

        // string literals
        if (c == '\'' || c == '"') {
            const char quote = c;
            ++pos; // opening quote
            std::string str;
            while (pos < n && src[pos] != quote) {
                if (src[pos] == '\\') {
                    ++pos;
                    if (pos >= n) {
                        throw recipe_exception("unexpected end of input after escape character", source, pos);
                    }
                    const char escaped = src[pos++];
                    auto it = escape_chars.find(escaped);
                    if (it == escape_chars.end()) {
                        throw recipe_exception(std::string("unknown escape character \\") + escaped, source, pos);
                    }
                    str += it->second;
                } else {
                    str += src[pos++];
                }
            }
            if (pos >= n) {
                throw recipe_exception("unterminated string literal", source, start);
            }
            ++pos; // closing quote
            tokens.push_back({token::string_literal, std::move(str), start});
            continue;
        }

        // operators
        const std::string two = src.substr(pos, 2);
        if (two == "//") {
            tokens.push_back({token::multiplicative_binary_operator, "//", start});
            pos += 2;
            continue;
        }
        if (two == "==" || two == "!=" || two == "<=" || two == ">=") {
            tokens.push_back({token::comparison_binary_operator, two, start});
            pos += 2;
            continue;
        }

        switch (c) {
            case '<':
            case '>':
                tokens.push_back({token::comparison_binary_operator, std::string(1, c), start});
                ++pos;
                continue;
            case '+':
            case '-':
                tokens.push_back({token::additive_binary_operator, std::string(1, c), start});
                ++pos;
                continue;
            case '*':
            case '%':
                tokens.push_back({token::multiplicative_binary_operator, std::string(1, c), start});
                ++pos;
                continue;
            case '=':
                tokens.push_back({token::equals, "=", start});
                ++pos;
                continue;
            case '(':
                tokens.push_back({token::open_paren, "(", start});
                ++pos;
                continue;
            case ')':
                tokens.push_back({token::close_paren, ")", start});
                ++pos;
                continue;
            case '/':
                throw recipe_exception("bare '/' is not allowed; use '//' for floor division", source, pos);
            default:
                throw recipe_exception(std::string("unexpected character: ") + c, source, pos);
        }
    }

    return {std::move(tokens), src};
}

class recipe_parser {
    const std::vector<token> & tokens;
    const std::string & source;
    const type_predicate & is_type;
    size_t current = 0;

public:
    recipe_parser(const std::vector<token> & t, const std::string & src, const type_predicate & is_type)
        : tokens(t), source(src), is_type(is_type) {}

    program parse() {
        statements body;

        // optional leading `default TYPE`, seeds the initial `type`
        if (is_identifier("default")) {
            const size_t kw = current;
            ++current; // consume 'default'
            auto type_val = parse_type_literal("default");
            body.push_back(mk_stmt<set_statement>(kw, mk_ident(kw, "type"), std::move(type_val), statements{}));
        }

        while (!is(token::eof)) {
            body.push_back(parse_statement());
        }
        return program(std::move(body));
    }

private:
    template<typename T, typename... Args>
    std::unique_ptr<T> mk_stmt(size_t start_pos, Args &&... args) {
        auto ptr = std::make_unique<T>(std::forward<Args>(args)...);
        assert(start_pos < tokens.size());
        ptr->pos = tokens[start_pos].pos;
        return ptr;
    }

    statement_ptr mk_ident(size_t start_pos, const std::string & name) {
        return mk_stmt<identifier>(start_pos, name);
    }

    const token & peek(size_t offset = 0) const {
        if (current + offset >= tokens.size()) {
            static const token end_token{token::eof, "", 0};
            return end_token;
        }
        return tokens[current + offset];
    }

    const token & next() {
        if (current >= tokens.size()) {
            throw recipe_exception("unexpected end of input", source, tokens.empty() ? 0 : tokens.back().pos);
        }
        return tokens[current++];
    }

    bool is(token::type type) const {
        return peek().t == type;
    }

    bool is_identifier(const std::string & name) const {
        return peek().t == token::identifier && peek().value == name;
    }

    token expect(token::type type, const std::string & error) {
        const auto & t = peek();
        if (t.t != type) {
            throw recipe_exception(error + " (got '" + t.value + "')", source, t.pos);
        }
        ++current;
        return t;
    }

    // Statements

    statement_ptr parse_statement() {
        const token & t = peek();
        if (t.t != token::identifier) {
            throw recipe_exception("expected a statement (type, set or if)", source, t.pos);
        }
        if (t.value == "set") {
            return parse_set_statement();
        }
        if (t.value == "if") {
            return parse_if_statement();
        }
        if (t.value == "type") {
            return parse_type_assignment();
        }
        if (t.value == "default") {
            throw recipe_exception("'default' must appear before any other statement", source, t.pos);
        }
        throw recipe_exception("unexpected token: " + t.value, source, t.pos);
    }

    statement_ptr parse_type_assignment() {
        const size_t kw = current;
        ++current; // consume 'type'
        expect(token::equals, "expected '=' after 'type'");
        auto val = parse_expression();
        return mk_stmt<set_statement>(kw, mk_ident(kw, "type"), std::move(val), statements{});
    }

    statement_ptr parse_set_statement() {
        const size_t kw = current;
        ++current; // consume 'set'
        const token & name = peek();
        if (name.t != token::identifier || is_reserved_name(name.value)) {
            throw recipe_exception("expected a variable name after 'set'", source, name.pos);
        }
        const size_t name_pos = current;
        ++current; // consume IDENT
        expect(token::equals, "expected '=' in set statement");
        auto val = parse_expression();
        return mk_stmt<set_statement>(kw, mk_ident(name_pos, name.value), std::move(val), statements{});
    }

    statement_ptr parse_if_statement() {
        const size_t kw = current;
        ++current; // consume 'if'
        auto node = parse_if_body(kw);
        if (!is_identifier("endif")) {
            throw recipe_exception("expected 'endif'", source, peek().pos);
        }
        ++current; // consume the single closing 'endif'
        return node;
    }

    // parses an if/elif chain after the 'if'/'elif' keyword is consumed; the
    // whole chain is closed by a single 'endif' handled by parse_if_statement
    statement_ptr parse_if_body(size_t kw) {
        auto test = parse_expression();

        statements body;
        while (!is_identifier("elif") && !is_identifier("else") && !is_identifier("endif")) {
            if (is(token::eof)) {
                throw recipe_exception("expected 'endif'", source, peek().pos);
            }
            body.push_back(parse_statement());
        }

        statements alternate;
        if (is_identifier("elif")) {
            const size_t epos = current;
            ++current; // consume 'elif'
            alternate.push_back(parse_if_body(epos)); // nested, shares closing endif
        } else if (is_identifier("else")) {
            ++current; // consume 'else'
            while (!is_identifier("endif")) {
                if (is(token::eof)) {
                    throw recipe_exception("expected 'endif'", source, peek().pos);
                }
                alternate.push_back(parse_statement());
            }
        }
        return mk_stmt<if_statement>(kw, std::move(test), std::move(body), std::move(alternate));
    }

    // Expressions

    statement_ptr parse_expression() {
        return parse_or();
    }

    statement_ptr parse_or() {
        auto left = parse_and();
        while (is_identifier("or")) {
            const size_t p = current;
            token op = next();
            left = mk_stmt<binary_expression>(p, op, std::move(left), parse_and());
        }
        return left;
    }

    statement_ptr parse_and() {
        auto left = parse_not();
        while (is_identifier("and")) {
            const size_t p = current;
            token op = next();
            left = mk_stmt<binary_expression>(p, op, std::move(left), parse_not());
        }
        return left;
    }

    statement_ptr parse_not() {
        if (is_identifier("not")) {
            const size_t p = current;
            token op = next();
            return mk_stmt<unary_expression>(p, op, parse_not());
        }
        return parse_comparison();
    }

    statement_ptr parse_comparison() {
        auto left = parse_additive();
        if (is(token::comparison_binary_operator)) {
            const size_t p = current;
            token op = next();
            left = mk_stmt<binary_expression>(p, op, std::move(left), parse_additive());
        }
        return left;
    }

    statement_ptr parse_additive() {
        auto left = parse_multiplicative();
        while (is(token::additive_binary_operator)) {
            const size_t p = current;
            token op = next();
            left = mk_stmt<binary_expression>(p, op, std::move(left), parse_multiplicative());
        }
        return left;
    }

    statement_ptr parse_multiplicative() {
        auto left = parse_unary();
        while (is(token::multiplicative_binary_operator)) {
            const size_t p = current;
            token op = next();
            left = mk_stmt<binary_expression>(p, op, std::move(left), parse_unary());
        }
        return left;
    }

    statement_ptr parse_unary() {
        if (is(token::additive_binary_operator) && peek().value == "-") {
            const size_t p = current;
            token op = next();
            return mk_stmt<unary_expression>(p, op, parse_unary());
        }
        return parse_atom();
    }

    statement_ptr parse_atom() {
        const token & t = peek();
        const size_t p = current;
        switch (t.t) {
            case token::numeric_literal:
                ++current;
                return mk_stmt<integer_literal>(p, std::stoll(t.value));
            case token::string_literal:
                ++current;
                return mk_stmt<string_literal>(p, t.value);
            case token::open_paren: {
                ++current; // consume '('
                auto expr = parse_expression();
                expect(token::close_paren, "expected ')'");
                return expr;
            }
            case token::identifier: {
                if (is_reserved_atom(t.value)) {
                    throw recipe_exception("unexpected keyword in expression: " + t.value, source, t.pos);
                }
                ++current;
                if (is_type && is_type(t.value)) {
                    return mk_stmt<string_literal>(p, t.value); // TYPE -> string literal
                }
                return mk_stmt<identifier>(p, t.value);
            }
            default:
                throw recipe_exception("unexpected token in expression: " + t.value, source, t.pos);
        }
    }

    statement_ptr parse_type_literal(const char * after) {
        const token & t = peek();
        if (t.t != token::identifier || is_reserved_atom(t.value)) {
            throw recipe_exception(std::string("expected a quant type after '") + after + "'", source, t.pos);
        }
        if (is_type && !is_type(t.value)) {
            throw recipe_exception("unknown quant type: " + t.value, source, t.pos);
        }
        const size_t p = current;
        ++current;
        return mk_stmt<string_literal>(p, t.value);
    }
};

program parse(const std::string & source, const type_predicate & is_type) {
    lexer_result lex = tokenize(source);
    return recipe_parser(lex.tokens, lex.source, is_type).parse();
}

} // namespace recipe
} // namespace jinja
