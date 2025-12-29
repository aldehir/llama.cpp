/*
 * # Jinja2 Template PEG Grammar
 *
 * ## Lexical Rules
 *
 * keyword        <- ('if' / 'elif' / 'else' / 'endif' / 'for' / 'endfor' / 'in' /
 *                   'set' / 'endset' / 'macro' / 'endmacro' / 'call' / 'endcall' /
 *                   'filter' / 'endfilter' / 'break' / 'continue' / 'and' / 'or' /
 *                   'not' / 'is' / 'true' / 'false' / 'True' / 'False' / 'none' / 'None') !ident-cont
 * ident-start    <- [a-zA-Z_]
 * ident-cont     <- [a-zA-Z0-9_]
 * identifier     <- !keyword ident-start ident-cont*
 *
 * ## Literals
 *
 * integer        <- [0-9]+
 * float          <- [0-9]+ '.' [0-9]+
 * escape         <- '\\' .
 * double-string  <- '"' (escape / (!'\"' !'\\' .))* '"'
 * single-string  <- "'" (escape / (!"'" !'\\' .))* "'"
 * string         <- double-string / single-string
 * boolean        <- 'true' / 'false' / 'True' / 'False'
 * none-literal   <- 'none' / 'None'
 * array-literal  <- '[' (expr (',' expr)*)? ']'
 * kv-pair        <- expr ':' expr
 * object-literal <- '{' (kv-pair (',' kv-pair)*)? '}'
 *
 * ## Primary & Postfix Expressions
 *
 * primary        <- float / integer / string / boolean / none-literal /
 *                   array-literal / object-literal / identifier / '(' expr ')'
 * member-access  <- '.' identifier
 * subscript-access <- '[' (expr? ':' expr? (':' expr?)? / expr) ']'
 * spread-arg     <- '*' expr
 * kwarg          <- identifier '=' expr
 * arg            <- spread-arg / kwarg / expr
 * arg-list       <- arg (',' arg)*
 * call-access    <- '(' arg-list? ')'
 * postfix        <- primary (member-access / subscript-access / call-access)*
 *
 * ## Filters & Tests
 *
 * filter-call    <- identifier call-access?
 * filter-expr    <- postfix ('|' filter-call)*
 * test-suffix    <- 'is' 'not'? identifier
 * test-expr      <- filter-expr test-suffix*
 *
 * ## Operators (by precedence, lowest to highest)
 *
 * unary          <- ('+' / '-' / 'not') unary / test-expr
 * mult-op        <- '//' / '*' / '/' / '%'
 * multiplicative <- unary (mult-op unary)*
 * add-op         <- '+' / '-'
 * additive       <- multiplicative (add-op multiplicative)*
 * comp-op        <- '==' / '!=' / '<=' / '>=' / '<' / '>' / 'not' 'in' / 'in'
 * comparison     <- additive (comp-op additive)*
 * and-op         <- 'and'
 * and-expr       <- comparison (and-op comparison)*
 * or-op          <- 'or'
 * or-expr        <- and-expr (or-op and-expr)*
 * if-expr        <- or-expr ('if' or-expr ('else' if-expr)?)?
 * expr           <- if-expr
 *
 * ## Expression Sequences
 *
 * expr-seq       <- expr (',' expr)*
 * primary-seq    <- primary (',' primary)*
 *
 * ## Template Elements
 *
 * open-tag       <- '{{' / '{%' / '{#'
 *
 * # When lstrip_blocks is enabled:
 * text-lstrip    <- (!open-tag .)+ &('{%' / '{#')   # strips trailing line whitespace
 * text-normal    <- (!open-tag .)+
 * text           <- text-lstrip / text-normal
 *
 * # When lstrip_blocks is disabled:
 * text           <- (!open-tag .)+
 *
 * comment        <- '{#' (!'#}' .)* '#}'
 * expression     <- '{{' expr '}}'
 *
 * ## Control Statements
 *
 * # When trim_blocks is enabled, block tags consume an optional trailing '\n' after '%}'
 *
 * break-stmt     <- '{%' 'break' '%}' '\n'?
 * continue-stmt  <- '{%' 'continue' '%}' '\n'?
 * set-inline     <- '{%' 'set' expr-seq '=' expr-seq '%}' '\n'?
 * set-block      <- '{%' 'set' identifier '%}' '\n'? template-body '{%' 'endset' '%}' '\n'?
 * set-stmt       <- set-inline / set-block
 * elif-clause    <- '{%' 'elif' expr '%}' '\n'? template-body
 * else-clause    <- '{%' 'else' '%}' '\n'? template-body
 * if-stmt        <- '{%' 'if' expr '%}' '\n'? template-body elif-clause* else-clause? '{%' 'endif' '%}' '\n'?
 * for-else       <- '{%' 'else' '%}' '\n'? template-body
 * for-stmt       <- '{%' 'for' primary-seq 'in' expr '%}' '\n'? template-body for-else? '{%' 'endfor' '%}' '\n'?
 * macro-stmt     <- '{%' 'macro' identifier '(' arg-list? ')' '%}' '\n'? template-body '{%' 'endmacro' '%}' '\n'?
 * call-stmt      <- '{%' 'call' ('(' arg-list? ')')? identifier '(' arg-list? ')' '%}' '\n'?
 *                   template-body
 *                   '{%' 'endcall' '%}' '\n'?
 * filter-stmt    <- '{%' 'filter' filter-call '%}' '\n'? template-body '{%' 'endfilter' '%}' '\n'?
 * statement      <- if-stmt / for-stmt / set-stmt / macro-stmt / call-stmt /
 *                   filter-stmt / break-stmt / continue-stmt
 *
 * ## Template Structure
 *
 * template-body    <- template-element*
 * template-element <- statement / expression / comment / text
 * template         <- template-element* end
 */

#include "jinja-peg-parser.h"
#include "peg-parser.h"

#include <stdexcept>
#include <string>
#include <memory>

namespace jinja {

using semantic_context = common_peg_semantic_context;

static std::string unescape_string(const std::string& s) {
    std::string result;
    result.reserve(s.size());
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            switch (s[++i]) {
                case 'n':  result += '\n'; break;
                case 't':  result += '\t'; break;
                case 'r':  result += '\r'; break;
                case '\\': result += '\\'; break;
                case '"':  result += '"';  break;
                case '\'': result += '\''; break;
                default:   result += s[i]; break;
            }
        } else {
            result += s[i];
        }
    }
    return result;
}

static std::string_view strip_trailing_line_whitespace(std::string_view sv) {
    if (sv.empty()) {
        return sv;
    }
    size_t i = sv.size();
    while (i > 0 && (sv[i - 1] == ' ' || sv[i - 1] == '\t')) {
        i--;
    }
    if (i == 0 || sv[i - 1] == '\n') {
        return sv.substr(0, i);
    }
    return sv;
}

// Intermediate semantic value types

struct call_access_op : common_peg_value {
    statements args;
    call_access_op(statements a) : args(std::move(a)) {}
};

struct operator_token : common_peg_value {
    std::string op;
    operator_token(std::string o) : op(std::move(o)) {}
};

struct key_value_pair : common_peg_value {
    statement_ptr key;
    statement_ptr value;
    key_value_pair(statement_ptr k, statement_ptr v) : key(std::move(k)), value(std::move(v)) {}
};

struct test_op : common_peg_value {
    bool negate;
    statement_ptr test;
    test_op(bool n, statement_ptr t) : negate(n), test(std::move(t)) {}
};

struct else_clause : common_peg_value {
    statements body;
    else_clause(statements b) : body(std::move(b)) {}
};

struct elif_clause : common_peg_value {
    statement_ptr test;
    statements body;
    elif_clause(statement_ptr t, statements b) : test(std::move(t)), body(std::move(b)) {}
};


// Parser

static common_peg_parser build(common_peg_parser_builder & p, const jinja_parse_options & options) {
    auto sp = p.space();

    p.set_token_space(sp);
    p.set_word_boundary(p.chars("[a-zA-Z0-9_]", 1, 1));

    auto open_block = p.token("{%");
    auto close_block = p.expect("%}", "expected '%}'");
    auto close_block_trimmed = options.trim_blocks
        ? (close_block + p.optional(p.literal("\n")))
        : close_block;

    auto open_expr = p.token("{{");
    auto close_expr = p.expect("}}", "expected '}}'");

    auto open_comment = p.literal("{#");

    auto comma = p.token(",");
    auto comma_sep = sp + comma;
    auto colon = p.token(":");
    auto dot = p.token(".");
    auto pipe = p.token("|");

    auto open_paren = p.token("(");
    auto close_paren = p.expect(")", "expected ')'");

    auto open_bracket = p.token("[");
    auto close_bracket = p.expect("]", "expected ']'");

    auto open_brace = p.token("{");
    auto close_brace = p.expect("}", "expected '}'");

    auto backslash = p.literal("\\");
    auto escape_seq = backslash + p.any();

    auto double_quote = p.literal("\"");
    auto single_quote = p.literal("'");

    auto sep_list = [&](const std::string & item) {
        return p.ref(item) + p.zero_or_more(comma_sep + p.ref(item));
    };

    p.rule("keyword", (
            p.literal("if") | "elif" | "else" | "endif" |
            "for" | "endfor" | "in" |
            "set" | "endset" |
            "macro" | "endmacro" |
            "call" | "endcall" |
            "filter" | "endfilter" |
            "break" | "continue" |
            "and" | "or" | "not" | "is" |
            "true" | "false" | "True" | "False" |
            "none" | "None"
        ) + p.negate(p.ref("ident-cont")));

    p.rule("ident-start", p.chars("[a-zA-Z_]", 1, 1));
    p.rule("ident-cont", p.chars("[a-zA-Z0-9_]", 1, 1));
    p.action_rule("identifier",
        p.negate(p.ref("keyword")) + p.ref("ident-start") + p.zero_or_more(p.ref("ident-cont")),
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            return std::make_shared<identifier>(std::string(ctx.text()));
        });

    p.action_rule("integer", p.chars("[0-9]", 1, -1),
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            return std::make_shared<integer_literal>(std::stoll(std::string(ctx.text())));
        });

    p.action_rule("float",
        p.chars("[0-9]", 1, -1) + p.literal(".") + p.chars("[0-9]", 1, -1),
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            return std::make_shared<float_literal>(std::stod(std::string(ctx.text())));
        });

    auto action_string = [](const semantic_context & ctx) -> common_peg_value_ptr {
        std::string content(ctx.text());
        if (content.size() >= 2) {
            content = content.substr(1, content.size() - 2);
        }
        return std::make_shared<string_literal>(unescape_string(content));
    };

    p.action_rule("double-string",
        double_quote + p.cut() + p.zero_or_more(
            escape_seq | (p.negate(double_quote) + p.negate(backslash) + p.any())
        ) + p.expect("\"", "unterminated string"),
        action_string);
    p.action_rule("single-string",
        single_quote + p.cut() + p.zero_or_more(
            escape_seq | (p.negate(single_quote) + p.negate(backslash) + p.any())
        ) + p.expect("'", "unterminated string"),
        action_string);
    p.rule("string", p.ref("double-string") | p.ref("single-string"));

    p.action_rule("boolean", p.literal("true") | "false" | "True" | "False",
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            std::string s(ctx.text());
            int64_t value = (s == "true" || s == "True") ? 1 : 0;
            return std::make_shared<integer_literal>(value);
        });

    p.action_rule("none-literal", p.literal("none") | "None",
        [](const semantic_context &) -> common_peg_value_ptr {
            return std::make_shared<integer_literal>(0);
        });

    p.action_rule("array-literal", open_bracket + p.cut() + p.optional(sep_list("expr")) + sp + close_bracket,
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            return std::make_shared<array_literal>(ctx.all<expression, statement>());
        });

    p.action_rule("kv-pair", p.ref("expr") + colon + p.ref("expr"),
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            auto [key, val] = ctx.pair<expression>();
            if (key && val) {
                return std::make_shared<key_value_pair>(key, val);
            }
            return nullptr;
        });

    p.action_rule("object-literal", open_brace + p.cut() + p.optional(sep_list("kv-pair")) + sp + close_brace,
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            std::vector<std::pair<statement_ptr, statement_ptr>> pairs;
            for (auto& kv : ctx.all<key_value_pair>()) {
                pairs.push_back({std::move(kv->key), std::move(kv->value)});
            }
            return std::make_shared<object_literal>(std::move(pairs));
        });

    p.rule("primary",
        p.ref("float")
        | p.ref("integer")
        | p.ref("string")
        | p.ref("boolean")
        | p.ref("none-literal")
        | p.ref("array-literal")
        | p.ref("object-literal")
        | p.ref("identifier")
        | (open_paren + p.cut() + p.ref("expr") + sp + close_paren));

    p.action_rule("member-access", dot + p.ref("identifier"),
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            return ctx.raw(0);
        });

    p.action_rule("subscript-access",
        open_bracket + p.cut() + (
            (p.optional(p.ref("expr")) + colon + p.optional(p.ref("expr")) + p.optional(colon + p.optional(p.ref("expr"))))
            | p.ref("expr")
        ) + sp + close_bracket,
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            std::string s(ctx.text());
            bool is_slice = s.find(':') != std::string::npos;
            if (is_slice) {
                statement_ptr start;
                statement_ptr stop;
                statement_ptr step;
                size_t idx = 0;
                if (idx < ctx.size()) {
                    start = ctx.get<expression>(idx++);
                }
                if (idx < ctx.size()) {
                    stop = ctx.get<expression>(idx++);
                }
                if (idx < ctx.size()) {
                    step = ctx.get<expression>(idx++);
                }
                return std::make_shared<slice_expression>(std::move(start), std::move(stop), std::move(step));
            }
            return ctx.get<expression>(0);
        });

    p.action_rule("spread-arg", p.token("*") + p.ref("expr"),
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            auto arg = ctx.get<expression>(0);
            return std::make_shared<spread_expression>(std::move(arg));
        });

    p.action_rule("kwarg", p.ref("identifier") + p.token("=") + p.ref("expr"),
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            auto [key, val] = ctx.pair<expression>();
            return std::make_shared<keyword_argument_expression>(std::move(key), std::move(val));
        });

    p.rule("arg", p.ref("spread-arg") | p.ref("kwarg") | p.ref("expr"));

    p.action_rule("arg-list", sep_list("arg"),
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            return std::make_shared<call_access_op>(ctx.all<expression, statement>());
        });

    p.action_rule("call-access", open_paren + p.cut() + p.optional(p.ref("arg-list")) + sp + close_paren,
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            if (auto call_op = ctx.find<call_access_op>()) {
                return call_op;
            }
            return std::make_shared<call_access_op>(statements{});
        });

    p.action_rule("postfix",
        p.ref("primary") + p.zero_or_more(
            p.ref("member-access") | p.ref("subscript-access") | p.ref("call-access")),
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            if (ctx.empty()) {
                return nullptr;
            }
            auto base = ctx.get<expression>(0);
            if (!base) {
                return nullptr;
            }
            for (size_t i = 1; i < ctx.size(); i++) {
                auto rule = ctx.rule(i);
                if (rule == "member-access") {
                    base = std::make_shared<member_expression>(std::move(base), ctx.get<expression>(i), false);
                } else if (rule == "subscript-access") {
                    base = std::make_shared<member_expression>(std::move(base), ctx.get<expression>(i), true);
                } else if (rule == "call-access") {
                    auto c = ctx.get<call_access_op>(i);
                    base = std::make_shared<call_expression>(std::move(base), std::move(c->args));
                }
            }
            return base;
        });

    p.action_rule("filter-call", p.ref("identifier") + p.optional(p.ref("call-access")),
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            auto id = ctx.get<expression>(0);
            if (ctx.size() > 1) {
                if (auto call = ctx.get<call_access_op>(1)) {
                    return std::make_shared<call_expression>(std::move(id), std::move(call->args));
                }
            }
            return id;
        });

    p.action_rule("filter-expr",
        p.ref("postfix") + p.zero_or_more(sp + pipe + p.ref("filter-call")),
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            if (ctx.empty()) {
                return nullptr;
            }
            auto base = ctx.get<expression>(0);
            if (!base) {
                return nullptr;
            }
            for (size_t i = 1; i < ctx.size(); i++) {
                if (ctx.rule(i) == "filter-call") {
                    base = std::make_shared<filter_expression>(std::move(base), ctx.get<expression>(i));
                }
            }
            return base;
        });

    p.action_rule("test-suffix", p.keyword("is") + p.optional(p.keyword("not")) + p.ref("identifier"),
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            if (ctx.empty()) {
                return nullptr;
            }
            std::string s(ctx.text());
            bool negate = s.find("not") != std::string::npos;
            auto test_id = ctx.get<expression>(0);
            return std::make_shared<test_op>(negate, test_id);
        });

    p.action_rule("test-expr", p.ref("filter-expr") + p.zero_or_more(sp + p.ref("test-suffix")),
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            if (ctx.empty()) {
                return nullptr;
            }
            auto base = ctx.get<expression>(0);
            if (!base) {
                return nullptr;
            }
            for (size_t i = 1; i < ctx.size(); i++) {
                if (auto t = ctx.get<test_op>(i)) {
                    base = std::make_shared<test_expression>(std::move(base), t->negate, std::move(t->test));
                }
            }
            return base;
        });

    p.action_rule("unary", ((p.literal("+") | "-" | "not") + sp + p.ref("unary")) | p.ref("test-expr"),
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            if (ctx.empty()) {
                return nullptr;
            }
            std::string s(ctx.text());
            while (!s.empty() && std::isspace(s.front())) {
                s.erase(0, 1);
            }
            if (s.size() >= 3 && s.substr(0, 3) == "not") {
                auto arg = ctx.get<expression>(0);
                if (arg) {
                    token op;
                    op.value = "not";
                    return std::make_shared<unary_expression>(op, arg);
                }
            } else if (!s.empty() && (s[0] == '+' || s[0] == '-')) {
                auto arg = ctx.get<expression>(0);
                if (arg) {
                    token op;
                    op.value = std::string(1, s[0]);
                    return std::make_shared<unary_expression>(op, arg);
                }
            }
            return ctx.raw(0);
        });

    auto action_operator = [](const semantic_context & ctx) -> common_peg_value_ptr {
        return std::make_shared<operator_token>(std::string(ctx.text()));
    };

    p.action_rule("mult-op", p.literal("//") | "*" | "/" | "%", action_operator);
    p.action_rule("add-op", p.literal("+") | "-", action_operator);
    p.action_rule("and-op", p.literal("and"), action_operator);
    p.action_rule("or-op", p.literal("or"), action_operator);

    auto action_binary_chain = [](const semantic_context & ctx) -> common_peg_value_ptr {
        if (ctx.empty()) {
            return nullptr;
        }
        auto result = ctx.get<expression>(0);
        if (!result) {
            return nullptr;
        }
        for (size_t i = 1; i + 1 < ctx.size(); i += 2) {
            auto op_tok = ctx.get<operator_token>(i);
            auto right = ctx.get<expression>(i + 1);
            if (op_tok && right) {
                token op;
                op.value = op_tok->op;
                result = std::make_shared<binary_expression>(op, result, right);
            }
        }
        return result;
    };
    p.action_rule("multiplicative", p.ref("unary") + p.zero_or_more(sp + p.ref("mult-op") + sp + p.ref("unary")), action_binary_chain);

    p.action_rule("additive", p.ref("multiplicative") + p.zero_or_more(sp + p.ref("add-op") + sp + p.ref("multiplicative")), action_binary_chain);

    p.action_rule("comp-op",
        p.literal("==") | "!=" | "<=" | ">=" | "<" | ">"
        | (p.literal("not") + sp + "in") | "in",
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            std::string s(ctx.text());
            if (s.find("not") != std::string::npos) {
                return std::make_shared<operator_token>("not in");
            }
            return std::make_shared<operator_token>(s);
        });

    p.action_rule("comparison", p.ref("additive") + p.zero_or_more(sp + p.ref("comp-op") + sp + p.ref("additive")), action_binary_chain);

    p.action_rule("and-expr", p.ref("comparison") + p.zero_or_more(sp + p.ref("and-op") + sp + p.ref("comparison")), action_binary_chain);

    p.action_rule("or-expr", p.ref("and-expr") + p.zero_or_more(sp + p.ref("or-op") + sp + p.ref("and-expr")), action_binary_chain);

    p.action_rule("if-expr",
        p.ref("or-expr") + p.optional(sp + p.keyword("if") + p.ref("or-expr") + p.optional(sp + p.keyword("else") + p.ref("if-expr"))),
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            if (ctx.empty()) {
                return nullptr;
            }
            if (ctx.size() == 1) {
                return ctx.raw(0);
            }
            auto value_if_true = ctx.get<expression>(0);
            auto test = ctx.get<expression>(1);
            if (ctx.size() >= 3) {
                auto value_if_false = ctx.get<expression>(2);
                return std::make_shared<ternary_expression>(std::move(test), std::move(value_if_true), std::move(value_if_false));
            }
            return std::make_shared<select_expression>(std::move(value_if_true), std::move(test));
        });

    p.rule("expr", p.ref("if-expr"));

    p.action_rule("expr-seq", sep_list("expr"),
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            if (auto s = ctx.single()) {
                return s;
            }
            return std::make_shared<tuple_literal>(ctx.all<expression, statement>());
        });

    p.action_rule("primary-seq", sep_list("primary"),
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            if (auto s = ctx.single()) {
                return s;
            }
            return std::make_shared<tuple_literal>(ctx.all<expression, statement>());
        });

    p.rule("open-tag", p.literal("{{") | "{%" | "{#");

    // Use until_one_of for efficient scanning
    auto text_until = p.until_one_of({"{{", "{%", "{#"});

    if (options.lstrip_blocks) {
        p.action_rule("text-lstrip",
            p.negate(p.ref("open-tag")) + p.any() + text_until + p.peek(p.literal("{%") | p.literal("{#")),
            [](const semantic_context & ctx) -> common_peg_value_ptr {
                return std::make_shared<string_literal>(std::string(strip_trailing_line_whitespace(ctx.text())));
            });

        p.action_rule("text-normal",
            p.negate(p.ref("open-tag")) + p.any() + text_until,
            [](const semantic_context & ctx) -> common_peg_value_ptr {
                return std::make_shared<string_literal>(std::string(ctx.text()));
            });

        // Try lstrip first (more specific), then fall back to normal
        p.rule("text", p.ref("text-lstrip") | p.ref("text-normal"));
    } else {
        p.action_rule("text", p.negate(p.ref("open-tag")) + p.any() + text_until,
            [](const semantic_context & ctx) -> common_peg_value_ptr {
                return std::make_shared<string_literal>(std::string(ctx.text()));
            });
    }

    p.action_rule("comment", open_comment + p.cut() + p.until("#}") + p.expect("#}", "unterminated comment"),
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            std::string content(ctx.text());
            if (content.size() >= 4) {
                content = content.substr(2, content.size() - 4);
            }
            return std::make_shared<comment_statement>(content);
        });

    p.rule("expression", open_expr + p.cut() + p.ref("expr") + sp + close_expr);

    p.action_rule("break-stmt", open_block + p.keyword("break") + p.cut() + close_block_trimmed,
        [](const semantic_context &) -> common_peg_value_ptr {
            return std::make_shared<break_statement>();
        });

    p.action_rule("continue-stmt", open_block + p.keyword("continue") + p.cut() + close_block_trimmed,
        [](const semantic_context &) -> common_peg_value_ptr {
            return std::make_shared<continue_statement>();
        });

    p.action_rule("set-inline",
        open_block + p.keyword("set") + p.ref("expr-seq") + sp + p.token("=") + p.cut() + p.ref("expr-seq") + sp + close_block,
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            auto [assignee, value] = ctx.pair<expression>();
            if (!assignee || !value) {
                return nullptr;
            }
            return std::make_shared<set_statement>(std::move(assignee), std::move(value), statements{});
        });

    p.action_rule("set-block",
        open_block + p.keyword("set") + p.ref("identifier") + sp + close_block + p.cut()
        + p.ref("template-body")
        + open_block + p.keyword("endset") + close_block_trimmed,
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            if (ctx.empty()) {
                return nullptr;
            }
            auto assignee = ctx.get<expression>(0);
            return std::make_shared<set_statement>(std::move(assignee), nullptr, ctx.slice<statement>(1));
        });

    p.rule("set-stmt", p.ref("set-inline") | p.ref("set-block"));

    p.action_rule("elif-clause", open_block + p.keyword("elif") + p.cut() + p.ref("expr") + sp + close_block_trimmed + p.ref("template-body"),
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            if (ctx.empty()) {
                return nullptr;
            }
            auto test = ctx.get<expression>(0);
            return std::make_shared<elif_clause>(std::move(test), ctx.slice<statement>(1));
        });

    p.action_rule("else-clause", open_block + p.keyword("else") + p.cut() + close_block_trimmed + p.ref("template-body"),
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            return std::make_shared<else_clause>(ctx.all<statement>());
        });

    p.action_rule("if-stmt",
        open_block + p.keyword("if") + p.cut() + p.ref("expr") + sp + close_block_trimmed
        + p.ref("template-body")
        + p.zero_or_more(p.ref("elif-clause"))
        + p.optional(p.ref("else-clause"))
        + open_block + p.keyword("endif") + close_block_trimmed,
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            if (ctx.empty()) {
                return nullptr;
            }
            auto test = ctx.get<expression>(0);
            auto body = ctx.slice<statement>(1);
            auto elifs = ctx.slice<elif_clause>(2);
            auto else_ = ctx.find<else_clause>(2);

            // Build the if-elif-else chain from the end backwards
            statements alternate;
            if (else_) {
                alternate = std::move(else_->body);
            }

            for (auto it = elifs.rbegin(); it != elifs.rend(); ++it) {
                auto elif_stmt = std::make_shared<if_statement>(
                    std::move((*it)->test),
                    std::move((*it)->body),
                    std::move(alternate)
                );
                alternate = statements{elif_stmt};
            }

            return std::make_shared<if_statement>(std::move(test), std::move(body), std::move(alternate));
        });

    p.rule("for-else", open_block + p.keyword("else") + p.cut() + close_block_trimmed + p.ref("template-body"));

    p.action_rule("for-stmt",
        open_block + p.keyword("for") + p.cut() + p.ref("primary-seq") + sp + p.keyword("in") + p.ref("expr") + sp + close_block_trimmed
        + p.ref("template-body")
        + p.optional(p.ref("for-else"))
        + open_block + p.keyword("endfor") + close_block_trimmed,
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            if (ctx.size() < 2) {
                return nullptr;
            }
            auto loopvar = ctx.get<expression>(0);
            auto iterable = ctx.get<expression>(1);
            auto body = ctx.slice<statement>(2);
            statements default_block;  // TODO: for-else not being collected
            return std::make_shared<for_statement>(std::move(loopvar), std::move(iterable), std::move(body), std::move(default_block));
        });

    p.action_rule("macro-stmt",
        open_block + p.keyword("macro") + p.cut() + p.ref("identifier") + open_paren + p.cut() + p.optional(p.ref("arg-list")) + sp + close_paren + sp + close_block_trimmed
        + p.ref("template-body")
        + open_block + p.keyword("endmacro") + close_block_trimmed,
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            if (ctx.empty()) {
                return nullptr;
            }
            auto name = ctx.get<expression>(0);
            statements args;
            if (auto call_op = ctx.find<call_access_op>()) {
                args = std::move(call_op->args);
            }
            auto body = ctx.slice<statement>(1);
            return std::make_shared<macro_statement>(std::move(name), std::move(args), std::move(body));
        });

    p.action_rule("call-stmt",
        open_block + p.keyword("call") + p.cut()
            + p.optional(open_paren + p.cut() + p.optional(p.ref("arg-list")) + sp + close_paren)
            + sp + p.ref("identifier") + open_paren + p.cut() + p.optional(p.ref("arg-list")) + sp + close_paren
            + sp + close_block_trimmed
        + p.ref("template-body")
        + open_block + p.keyword("endcall") + close_block_trimmed,
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            if (ctx.empty()) {
                return nullptr;
            }
            // First call_access_op is caller args (optional), second is call args
            auto call_ops = ctx.all<call_access_op>();
            statements caller_args;
            statements call_args;
            if (call_ops.size() >= 2) {
                caller_args = std::move(call_ops[0]->args);
                call_args = std::move(call_ops[1]->args);
            } else if (call_ops.size() == 1) {
                call_args = std::move(call_ops[0]->args);
            }
            auto callee = ctx.find<expression>();
            auto body = ctx.slice<statement>(0);
            auto call_expr = std::make_shared<call_expression>(std::move(callee), std::move(call_args));
            return std::make_shared<call_statement>(std::move(call_expr), std::move(caller_args), std::move(body));
        });

    p.action_rule("filter-stmt",
        open_block + p.keyword("filter") + p.cut() + p.ref("filter-call") + sp + close_block_trimmed
        + p.ref("template-body")
        + open_block + p.keyword("endfilter") + close_block_trimmed,
        [](const semantic_context & ctx) -> common_peg_value_ptr {
            statement_ptr filter = ctx.get<expression>(0);
            return std::make_shared<filter_statement>(std::move(filter), ctx.slice<statement>(1));
        });

    p.rule("statement",
        p.ref("if-stmt") | p.ref("for-stmt") | p.ref("set-stmt") | p.ref("macro-stmt")
        | p.ref("call-stmt") | p.ref("filter-stmt") | p.ref("break-stmt") | p.ref("continue-stmt"));

    p.rule("template-body", p.zero_or_more(p.ref("template-element")));

    p.rule("template-element",
        p.ref("statement") | p.ref("expression") | p.ref("comment") | p.ref("text"));

    return p.rule("template", p.zero_or_more(p.ref("template-element")) + p.end());
}

program parse_with_peg(const std::string & input, const jinja_parse_options & options) {
    // Build parser with the given options
    auto parser = build_peg_parser([&](common_peg_parser_builder & p) {
        return build(p, options);
    });

    common_peg_parse_context ctx(input);
    ctx.enable_cache();
    auto result = parser.parse(ctx);

    if (result.fail()) {
        std::string message = "unexpected input";
        size_t pos = ctx.furthest_position;
        if (!ctx.errors.empty()) {
            const auto & err = ctx.errors.back();
            pos = err.position;
            message = err.message;
        }
        throw std::runtime_error("PEG parse failed at position " + std::to_string(pos) + ": " + message);
    }

    statements body;
    for (auto& val : result.values) {
        if (auto s = ctx.values.get_as<statement>(val)) {
            body.push_back(s);
        }
    }
    return program(std::move(body));
}

} // namespace jinja
