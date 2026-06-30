#include "quant-recipe.h"

#include "peg-parser.h"

#include "jinja/runtime.h"
#include "jinja/value.h"

#include <algorithm>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Recipe evaluation dialect
//
//   program := ws stmt* EOF
//   stmt    := NAME "=" expr
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
// newlines) is insignificant; blocks close with explicit "endif". The dialect
// deliberately omits for/macro/calls and bare '/' so a recipe always
// terminates and integer thresholds use floor division.
//
// A recipe is parsed with the common PEG parser, then the resulting parse tree
// is walked to emit a jinja::program that the jinja runtime executes unchanged.
// ---------------------------------------------------------------------------

namespace {

common_peg_arena build_recipe_grammar() {
    return build_peg_parser([](common_peg_parser_builder & b) {
        // whitespace + line comments, shared and node-less (plain combinators)
        auto comment  = b.literal("#") + b.until("\n");
        auto ws_chars = b.chars("[ \t\r\n]", 1, -1);
        auto ws       = b.zero_or_more(b.choice({ ws_chars, comment }));

        auto word_char = b.chars("[A-Za-z0-9_]", 1, 1);

        // lexeme helpers: a token followed by trailing whitespace
        auto sym = [&](const std::string & s) { return b.literal(s) + ws; };
        auto kw  = [&](const std::string & s) { return b.literal(s) + b.negate(word_char) + ws; };

        // operator tokens, tagged so the tree walk can recover them
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

        // reserve keywords so they are not consumed as a bare name
        auto reserved = (b.literal("and")  | b.literal("or")   | b.literal("not") |
                         b.literal("if")   | b.literal("elif") | b.literal("else") | b.literal("endif"))
                      + b.negate(word_char);

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

        // statements (if-stmt first so "if" is never read as an assignment target)
        common_peg_parser stmt = b.choice({ b.ref("if-stmt"), b.ref("assign") });

        b.rule("assign", [&]() { return b.negate(reserved) + name_lit + ws + sym("=") + expr; });

        b.rule("if-stmt", [&]() {
            auto block = [&](const std::string & key) { return b.tag(key, b.zero_or_more(stmt)); };
            auto elif  = b.rule("elif-clause", [&]() { return kw("elif") + b.tag("cond", expr) + block("body"); });
            auto els   = b.rule("else-clause", [&]() { return kw("else") + block("body"); });
            return kw("if") + b.tag("cond", expr) + block("body")
                 + b.zero_or_more(elif)
                 + b.optional(els)
                 + kw("endif");
        });

        return b.rule("program", [&]() {
            return ws + b.zero_or_more(stmt) + b.end();
        });
    });
}

// ---------------------------------------------------------------------------
// Tree walk: PEG parse tree -> jinja AST
// ---------------------------------------------------------------------------

jinja::token make_op_token(const std::string & value, size_t pos) {
    jinja::token t;
    t.value = value;
    t.pos   = pos;
    if (value == "and" || value == "or" || value == "not") {
        t.t = jinja::token::identifier;
    } else if (value == "+" || value == "-") {
        t.t = jinja::token::additive_binary_operator;
    } else if (value == "*" || value == "//" || value == "%") {
        t.t = jinja::token::multiplicative_binary_operator;
    } else {
        t.t = jinja::token::comparison_binary_operator;
    }
    return t;
}

struct recipe_builder {
    const common_peg_ast_arena & ast;
    const std::string & src;

    const common_peg_ast_node & get(common_peg_ast_id id) const { return ast.get(id); }

    [[noreturn]] void fail(const std::string & msg, size_t pos) const {
        throw std::runtime_error(jinja::fmt_error_with_source("recipe", msg, src, pos));
    }

    jinja::program build_program(const common_peg_ast_node & node) const {
        jinja::program prog;
        for (auto cid : node.children) {
            prog.body.push_back(build_stmt(get(cid)));
        }
        return prog;
    }

    jinja::statements build_block(const common_peg_ast_node & body_tag) const {
        jinja::statements out;
        for (auto cid : body_tag.children) {
            out.push_back(build_stmt(get(cid)));
        }
        return out;
    }

    jinja::statement_ptr build_stmt(const common_peg_ast_node & node) const {
        if (node.rule == "assign") {
            std::string var = std::string(get(node.children.at(0)).text);
            auto val = build_expr(get(node.children.at(1)));
            return std::make_unique<jinja::set_statement>(
                std::make_unique<jinja::identifier>(var), std::move(val), jinja::statements{});
        }
        if (node.rule == "if-stmt") {
            return build_if(node);
        }
        fail("unexpected statement node '" + node.rule + "'", node.start);
    }

    jinja::statement_ptr build_if(const common_peg_ast_node & node) const {
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
        jinja::statements alternate;
        if (else_clause) {
            alternate = build_block(get(else_clause->children.at(0)));
        }
        for (auto it = elifs.rbegin(); it != elifs.rend(); ++it) {
            const auto & clause = **it;
            auto etest = build_expr(get(get(clause.children.at(0)).children.at(0)));
            auto ebody = build_block(get(clause.children.at(1)));
            jinja::statements nested;
            nested.push_back(std::make_unique<jinja::if_statement>(
                std::move(etest), std::move(ebody), std::move(alternate)));
            alternate = std::move(nested);
        }

        return std::make_unique<jinja::if_statement>(std::move(test), std::move(body), std::move(alternate));
    }

    jinja::statement_ptr build_expr(const common_peg_ast_node & node) const {
        const std::string & r = node.rule;
        if (r == "or-expr" || r == "and-expr" || r == "cmp-expr" || r == "add-expr" || r == "mul-expr") {
            return build_binary_chain(node);
        }
        if (r == "not-expr" || r == "un-expr") {
            return build_unary(node);
        }
        if (r == "name") {
            return std::make_unique<jinja::identifier>(std::string(node.text));
        }
        if (r == "int") {
            return std::make_unique<jinja::integer_literal>(std::stoll(std::string(node.text)));
        }
        if (r == "string") {
            return std::make_unique<jinja::string_literal>(unquote(node.text));
        }
        fail("unexpected expression node '" + r + "'", node.start);
    }

    // left-associative fold of `operand (op operand)*`
    jinja::statement_ptr build_binary_chain(const common_peg_ast_node & node) const {
        const auto & ch = node.children;
        jinja::statement_ptr lhs = build_expr(get(ch.at(0)));
        for (size_t i = 1; i + 1 < ch.size(); i += 2) {
            const auto & op_node = get(ch[i]);
            auto rhs = build_expr(get(ch[i + 1]));
            lhs = std::make_unique<jinja::binary_expression>(
                make_op_token(std::string(op_node.text), op_node.start), std::move(lhs), std::move(rhs));
        }
        return lhs;
    }

    jinja::statement_ptr build_unary(const common_peg_ast_node & node) const {
        const auto & ch = node.children;
        if (ch.size() == 2 && get(ch[0]).tag == "op") {
            const auto & op_node = get(ch[0]);
            auto arg = build_expr(get(ch[1]));
            return std::make_unique<jinja::unary_expression>(
                make_op_token(std::string(op_node.text), op_node.start), std::move(arg));
        }
        return build_expr(get(ch.at(0)));
    }

    static std::string unquote(std::string_view text) {
        if (text.size() >= 2) {
            return std::string(text.substr(1, text.size() - 2));
        }
        return std::string(text);
    }
};

jinja::program parse_recipe_program(const std::string & src) {
    static const common_peg_arena arena = build_recipe_grammar();

    common_peg_parse_context ctx(src);
    auto result = arena.parse(ctx);
    if (!result.success() || result.nodes.empty()) {
        throw std::runtime_error(jinja::fmt_error_with_source("recipe", "failed to parse recipe", src, result.end));
    }

    recipe_builder builder{ ctx.ast, src };
    return builder.build_program(ctx.ast.get(result.nodes.front()));
}

std::string to_upper(const std::string & s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return (char) std::toupper(c); });
    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// quant_recipe
// ---------------------------------------------------------------------------

quant_recipe::quant_recipe() = default;
quant_recipe::~quant_recipe() = default;
quant_recipe::quant_recipe(quant_recipe &&) noexcept = default;
quant_recipe & quant_recipe::operator=(quant_recipe &&) noexcept = default;

// Built-in definitions prepended to every recipe. Authored in the recipe
// dialect itself (not magic host state) so the behavior stays transparent and
// an author can still override these names locally.
static const char * RECIPE_PRELUDE =
    "more_bits = layer < n_layer // 8\n"
    "         or layer >= 7 * n_layer // 8\n"
    "         or (layer - n_layer // 8) % 3 == 2\n";

quant_recipe quant_recipe::parse(const std::string & src) {
    quant_recipe r;

    // prelude statements run first, so the names they bind are available to the
    // recipe; a later assignment in the recipe overrides them.
    jinja::program prog = parse_recipe_program(RECIPE_PRELUDE);
    jinja::program user = parse_recipe_program(src);
    for (auto & stmt : user.body) {
        prog.body.push_back(std::move(stmt));
    }

    r.prog_ = std::make_unique<jinja::program>(std::move(prog));
    return r;
}

quant_recipe quant_recipe::load(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("failed to open recipe file: " + path);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return parse(ss.str());
}

ggml_type quant_recipe::eval(const quant_recipe_tensor & tensor) const {
    jinja::context ctx;

    // seed the ggml type names as strings so bare type tokens resolve.
    // both the canonical name (q4_K) and an upper-cased alias (Q4_K) are bound.
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const char * name = ggml_type_name((ggml_type) i);
        if (!name) {
            continue;
        }
        auto val = jinja::mk_val<jinja::value_string>(std::string(name));
        ctx.set_val(std::string(name), val);
        ctx.set_val(to_upper(name), val);
    }

    // seed the category names as constants (canonical and upper-cased alias)
    for (const char * cat : QUANT_RECIPE_CATEGORIES) {
        auto val = jinja::mk_val<jinja::value_string>(std::string(cat));
        ctx.set_val(cat, val);
        ctx.set_val(to_upper(cat), val);
    }

    // seed the per-tensor context
    ctx.set_val("category",       jinja::mk_val<jinja::value_string>(tensor.category));
    ctx.set_val("arch",           jinja::mk_val<jinja::value_string>(tensor.arch));
    ctx.set_val("model_type",     jinja::mk_val<jinja::value_string>(tensor.model_type));
    ctx.set_val("layer",          jinja::mk_val<jinja::value_int>(tensor.layer));
    ctx.set_val("n_layer",        jinja::mk_val<jinja::value_int>(tensor.n_layer));
    ctx.set_val("n_expert",       jinja::mk_val<jinja::value_int>(tensor.n_expert));
    ctx.set_val("n_gqa",          jinja::mk_val<jinja::value_int>(tensor.n_gqa));
    ctx.set_val("category_index", jinja::mk_val<jinja::value_int>(tensor.category_index));
    ctx.set_val("category_count", jinja::mk_val<jinja::value_int>(tensor.category_count));
    ctx.set_val("has_imatrix",    jinja::mk_val<jinja::value_bool>(tensor.has_imatrix));

    jinja::runtime rt(ctx);
    rt.execute(*prog_);

    auto type_val = ctx.get_val("type");
    if (type_val->is_undefined()) {
        return GGML_TYPE_COUNT;
    }

    std::string name = type_val->as_string().str();
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const char * tn = ggml_type_name((ggml_type) i);
        if (tn && name == tn) {
            return (ggml_type) i;
        }
    }
    return GGML_TYPE_COUNT;
}
