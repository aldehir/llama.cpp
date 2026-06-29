#include "recipe.h"

#include "peg-parser.h"

#include <memory>
#include <string>
#include <vector>

namespace jinja {

// ---------------------------------------------------------------------------
// Grammar
//
//   program := ws ("default" NAME)? stmt* EOF
//   stmt    := "type" "=" expr
//            | "set" NAME "=" expr
//            | "if" expr stmt* ("elif" expr stmt*)* ("else" stmt*)? "endif"
//   expr    := or
//   or      := and  ("or"  and)*
//   and     := not  ("and" not)*
//   not     := "not" not | cmp
//   cmp     := add  (("=="|"!="|"<="|">="|"<"|">") add)?
//   add     := mul  (("+"|"-") mul)*
//   mul     := un   (("*"|"//"|"%") un)*
//   un      := "-" un | atom
//   atom    := INT | STRING | NAME | "(" expr ")"
//
// Comments start with '#' and run to end of line. Whitespace (including
// newlines) is insignificant; blocks close with explicit "endif".
// ---------------------------------------------------------------------------

static common_peg_arena build_recipe_grammar() {
    return build_peg_parser([](common_peg_parser_builder & b) {
        // whitespace + line comments, shared and node-less (plain combinators)
        auto comment  = b.literal("#") + b.until("\n");
        auto ws_chars = b.chars("[ \t\r\n]", 1, -1);
        auto ws       = b.zero_or_more(b.choice({ ws_chars, comment }));

        auto word_char = b.chars("[A-Za-z0-9_]", 1, 1);

        // lexeme helpers: a token followed by trailing whitespace
        auto sym = [&](const std::string & s) { return b.literal(s) + ws; };
        auto kw  = [&](const std::string & s) { return b.literal(s) + b.negate(word_char) + ws; };

        // an operator token, tagged so the tree walk can recover it
        auto op    = [&](const std::string & s) { return b.tag("op", b.literal(s)) + ws; };
        auto op_kw = [&](const std::string & s) { return b.tag("op", b.literal(s)) + b.negate(word_char) + ws; };

        // captured terminals (the rule node text is the bare token, no ws)
        auto name_lit = b.rule("name", b.chars("[A-Za-z_]", 1, 1) + b.chars("[A-Za-z0-9_]", 0, -1));
        auto int_lit  = b.rule("int", b.chars("[0-9]", 1, -1));
        auto str_lit  = b.rule("string", [&]() {
            return b.choice({
                b.literal("\"") + b.string_content('"')  + b.literal("\""),
                b.literal("'")  + b.string_content('\'') + b.literal("'"),
            });
        });

        // reserve operator words so they are not consumed as a bare name
        auto reserved = (b.literal("and") | b.literal("or") | b.literal("not")) + b.negate(word_char);

        // expression grammar, lowest precedence first
        auto expr = b.ref("or-expr");

        b.rule("un-expr", [&]() {
            return b.choice({
                op("-") + b.ref("un-expr"),
                int_lit + ws,
                str_lit + ws,
                b.negate(reserved) + name_lit + ws,
                sym("(") + expr + sym(")"),
            });
        });

        auto mul_op = b.choice({ op("*"), op("//"), op("%") });
        b.rule("mul-expr", [&]() {
            return b.ref("un-expr") + b.zero_or_more(mul_op + b.ref("un-expr"));
        });

        auto add_op = b.choice({ op("+"), op("-") });
        b.rule("add-expr", [&]() {
            return b.ref("mul-expr") + b.zero_or_more(add_op + b.ref("mul-expr"));
        });

        auto cmp_op = b.choice({ op("=="), op("!="), op("<="), op(">="), op("<"), op(">") });
        b.rule("cmp-expr", [&]() {
            return b.ref("add-expr") + b.optional(cmp_op + b.ref("add-expr"));
        });

        b.rule("not-expr", [&]() {
            return b.choice({ op_kw("not") + b.ref("not-expr"), b.ref("cmp-expr") });
        });

        b.rule("and-expr", [&]() {
            return b.ref("not-expr") + b.zero_or_more(op_kw("and") + b.ref("not-expr"));
        });

        b.rule("or-expr", [&]() {
            return b.ref("and-expr") + b.zero_or_more(op_kw("or") + b.ref("and-expr"));
        });

        // statements
        common_peg_parser stmt = b.choice({ b.ref("assign-type"), b.ref("assign-set"), b.ref("if-stmt") });

        b.rule("assign-type", [&]() { return kw("type") + sym("=") + expr; });
        b.rule("assign-set",  [&]() { return kw("set") + name_lit + ws + sym("=") + expr; });

        b.rule("if-stmt", [&]() {
            auto block = [&](const std::string & key) { return b.tag(key, b.zero_or_more(stmt)); };
            auto elif  = b.rule("elif-clause", [&]() { return kw("elif") + b.tag("cond", expr) + block("body"); });
            auto els   = b.rule("else-clause", [&]() { return kw("else") + block("body"); });
            return kw("if") + b.tag("cond", expr) + block("body")
                 + b.zero_or_more(elif)
                 + b.optional(els)
                 + kw("endif");
        });

        auto default_clause = b.rule("default", [&]() { return kw("default") + name_lit + ws; });

        return b.rule("program", [&]() {
            return ws + b.optional(default_clause) + b.zero_or_more(stmt) + b.end();
        });
    });
}

// ---------------------------------------------------------------------------
// Tree walk: PEG parse tree -> jinja AST
// ---------------------------------------------------------------------------

namespace {

token make_op_token(const std::string & value, size_t pos) {
    token t;
    t.value = value;
    t.pos   = pos;
    if (value == "and" || value == "or" || value == "not") {
        t.t = token::identifier;
    } else if (value == "+" || value == "-") {
        t.t = token::additive_binary_operator;
    } else if (value == "*" || value == "//" || value == "%") {
        t.t = token::multiplicative_binary_operator;
    } else {
        t.t = token::comparison_binary_operator;
    }
    return t;
}

struct recipe_builder {
    const common_peg_ast_arena & ast;
    const std::string & src;

    const common_peg_ast_node & get(common_peg_ast_id id) const { return ast.get(id); }

    [[noreturn]] void fail(const std::string & msg, size_t pos) const {
        throw recipe_exception(msg, src, pos);
    }

    program build_program(const common_peg_ast_node & node) const {
        program prog;
        for (auto cid : node.children) {
            const auto & child = get(cid);
            if (child.rule == "default") {
                const auto & name = get(child.children.at(0));
                prog.body.push_back(set_type(std::make_unique<identifier>(std::string(name.text))));
            } else {
                prog.body.push_back(build_stmt(child));
            }
        }
        return prog;
    }

    statements build_block(const common_peg_ast_node & body_tag) const {
        statements out;
        for (auto cid : body_tag.children) {
            out.push_back(build_stmt(get(cid)));
        }
        return out;
    }

    statement_ptr build_stmt(const common_peg_ast_node & node) const {
        if (node.rule == "assign-type") {
            return set_type(build_expr(get(node.children.at(0))));
        }
        if (node.rule == "assign-set") {
            std::string var = std::string(get(node.children.at(0)).text);
            auto val = build_expr(get(node.children.at(1)));
            return std::make_unique<set_statement>(std::make_unique<identifier>(var), std::move(val), statements{});
        }
        if (node.rule == "if-stmt") {
            return build_if(node);
        }
        fail("unexpected statement node '" + node.rule + "'", node.start);
    }

    statement_ptr build_if(const common_peg_ast_node & node) const {
        // children: cond, body, elif-clause*, else-clause?
        auto test = build_expr(get(get(node.children.at(0)).children.at(0)));
        auto body = build_block(get(node.children.at(1)));

        std::vector<const common_peg_ast_node *> elifs;
        const common_peg_ast_node * else_clause = nullptr;
        for (size_t i = 2; i < node.children.size(); ++i) {
            const auto & child = get(node.children[i]);
            if (child.rule == "elif-clause") {
                elifs.push_back(&child);
            } else if (child.rule == "else-clause") {
                else_clause = &child;
            }
        }

        // collapse elif/else into a nested chain of if_statements (last to first)
        statements alternate;
        if (else_clause) {
            alternate = build_block(get(else_clause->children.at(0)));
        }
        for (auto it = elifs.rbegin(); it != elifs.rend(); ++it) {
            const auto & clause = **it;
            auto etest = build_expr(get(get(clause.children.at(0)).children.at(0)));
            auto ebody = build_block(get(clause.children.at(1)));
            statements nested;
            nested.push_back(std::make_unique<if_statement>(std::move(etest), std::move(ebody), std::move(alternate)));
            alternate = std::move(nested);
        }

        return std::make_unique<if_statement>(std::move(test), std::move(body), std::move(alternate));
    }

    statement_ptr build_expr(const common_peg_ast_node & node) const {
        const std::string & r = node.rule;
        if (r == "or-expr" || r == "and-expr" || r == "cmp-expr" || r == "add-expr" || r == "mul-expr") {
            return build_binary_chain(node);
        }
        if (r == "not-expr" || r == "un-expr") {
            return build_unary(node);
        }
        if (r == "name") {
            return std::make_unique<identifier>(std::string(node.text));
        }
        if (r == "int") {
            return std::make_unique<integer_literal>(std::stoll(std::string(node.text)));
        }
        if (r == "string") {
            return std::make_unique<string_literal>(unquote(node.text));
        }
        fail("unexpected expression node '" + r + "'", node.start);
    }

    // left-associative fold of `operand (op operand)*`
    statement_ptr build_binary_chain(const common_peg_ast_node & node) const {
        const auto & ch = node.children;
        statement_ptr lhs = build_expr(get(ch.at(0)));
        for (size_t i = 1; i + 1 < ch.size(); i += 2) {
            const auto & op_node = get(ch[i]);
            auto rhs = build_expr(get(ch[i + 1]));
            lhs = std::make_unique<binary_expression>(
                make_op_token(std::string(op_node.text), op_node.start), std::move(lhs), std::move(rhs));
        }
        return lhs;
    }

    statement_ptr build_unary(const common_peg_ast_node & node) const {
        const auto & ch = node.children;
        if (ch.size() == 2 && get(ch[0]).tag == "op") {
            const auto & op_node = get(ch[0]);
            auto arg = build_expr(get(ch[1]));
            return std::make_unique<unary_expression>(
                make_op_token(std::string(op_node.text), op_node.start), std::move(arg));
        }
        return build_expr(get(ch.at(0)));
    }

    static statement_ptr set_type(statement_ptr value) {
        return std::make_unique<set_statement>(std::make_unique<identifier>("type"), std::move(value), statements{});
    }

    static std::string unquote(std::string_view text) {
        if (text.size() >= 2) {
            return std::string(text.substr(1, text.size() - 2));
        }
        return std::string(text);
    }
};

} // namespace

program parse_recipe(const std::string & src) {
    static const common_peg_arena arena = build_recipe_grammar();

    common_peg_parse_context ctx(src);
    auto result = arena.parse(ctx);
    if (!result.success() || result.nodes.empty()) {
        throw recipe_exception("failed to parse recipe", src, result.end);
    }

    recipe_builder builder{ ctx.ast, src };
    return builder.build_program(ctx.ast.get(result.nodes.front()));
}

} // namespace jinja
