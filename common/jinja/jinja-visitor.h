#pragma once

#include "jinja-vm.h"
#include <stdexcept>

namespace jinja {

/**
 * Abstract visitor interface for traversing the AST
 * Each method returns a result type (template parameter)
 *
 * Uses dynamic dispatch via dynamic_cast to avoid modifying
 * the existing AST classes.
 */
template<typename Result>
struct ASTVisitor {
    virtual ~ASTVisitor() = default;

    // Statements
    virtual Result visit(program& node) = 0;
    virtual Result visit(if_statement& node) = 0;
    virtual Result visit(for_statement& node) = 0;
    virtual Result visit(set_statement& node) = 0;
    virtual Result visit(macro_statement& node) = 0;
    virtual Result visit(comment_statement& node) = 0;
    virtual Result visit(break_statement& node) = 0;
    virtual Result visit(continue_statement& node) = 0;
    virtual Result visit(filter_statement& node) = 0;
    virtual Result visit(call_statement& node) = 0;

    // Expressions
    virtual Result visit(member_expression& node) = 0;
    virtual Result visit(call_expression& node) = 0;
    virtual Result visit(identifier& node) = 0;
    virtual Result visit(integer_literal& node) = 0;
    virtual Result visit(float_literal& node) = 0;
    virtual Result visit(string_literal& node) = 0;
    virtual Result visit(array_literal& node) = 0;
    virtual Result visit(tuple_literal& node) = 0;
    virtual Result visit(object_literal& node) = 0;
    virtual Result visit(binary_expression& node) = 0;
    virtual Result visit(filter_expression& node) = 0;
    virtual Result visit(select_expression& node) = 0;
    virtual Result visit(test_expression& node) = 0;
    virtual Result visit(unary_expression& node) = 0;
    virtual Result visit(slice_expression& node) = 0;
    virtual Result visit(keyword_argument_expression& node) = 0;
    virtual Result visit(spread_expression& node) = 0;
    virtual Result visit(ternary_expression& node) = 0;

    /**
     * Dispatcher - routes to the appropriate visit method based on runtime type
     */
    Result dispatch(statement& node) {
        // Statements (check more specific types first)
        if (auto* p = dynamic_cast<program*>(&node)) return visit(*p);
        if (auto* p = dynamic_cast<if_statement*>(&node)) return visit(*p);
        if (auto* p = dynamic_cast<for_statement*>(&node)) return visit(*p);
        if (auto* p = dynamic_cast<set_statement*>(&node)) return visit(*p);
        if (auto* p = dynamic_cast<macro_statement*>(&node)) return visit(*p);
        if (auto* p = dynamic_cast<comment_statement*>(&node)) return visit(*p);
        if (auto* p = dynamic_cast<break_statement*>(&node)) return visit(*p);
        if (auto* p = dynamic_cast<continue_statement*>(&node)) return visit(*p);
        if (auto* p = dynamic_cast<filter_statement*>(&node)) return visit(*p);
        if (auto* p = dynamic_cast<call_statement*>(&node)) return visit(*p);

        // Expressions (check more specific types before base types)
        // tuple_literal extends array_literal, so check it first
        if (auto* p = dynamic_cast<tuple_literal*>(&node)) return visit(*p);
        if (auto* p = dynamic_cast<array_literal*>(&node)) return visit(*p);

        if (auto* p = dynamic_cast<member_expression*>(&node)) return visit(*p);
        if (auto* p = dynamic_cast<call_expression*>(&node)) return visit(*p);
        if (auto* p = dynamic_cast<identifier*>(&node)) return visit(*p);
        if (auto* p = dynamic_cast<integer_literal*>(&node)) return visit(*p);
        if (auto* p = dynamic_cast<float_literal*>(&node)) return visit(*p);
        if (auto* p = dynamic_cast<string_literal*>(&node)) return visit(*p);
        if (auto* p = dynamic_cast<object_literal*>(&node)) return visit(*p);
        if (auto* p = dynamic_cast<binary_expression*>(&node)) return visit(*p);
        if (auto* p = dynamic_cast<filter_expression*>(&node)) return visit(*p);
        if (auto* p = dynamic_cast<select_expression*>(&node)) return visit(*p);
        if (auto* p = dynamic_cast<test_expression*>(&node)) return visit(*p);
        if (auto* p = dynamic_cast<unary_expression*>(&node)) return visit(*p);
        if (auto* p = dynamic_cast<slice_expression*>(&node)) return visit(*p);
        if (auto* p = dynamic_cast<keyword_argument_expression*>(&node)) return visit(*p);
        if (auto* p = dynamic_cast<spread_expression*>(&node)) return visit(*p);
        if (auto* p = dynamic_cast<ternary_expression*>(&node)) return visit(*p);

        throw std::runtime_error("Unknown AST node type: " + node.type());
    }

    /**
     * Convenience method to dispatch on a statement_ptr
     */
    Result dispatch(const statement_ptr& ptr) {
        if (!ptr) {
            throw std::runtime_error("Cannot dispatch on null statement");
        }
        return dispatch(*ptr);
    }
};

/**
 * Base class for visitors that don't need to return a value.
 * All visit methods have default empty implementations.
 */
struct DefaultASTVisitor : public ASTVisitor<void> {
    // Statements - default empty implementations
    void visit(program& node) override {
        for (auto& stmt : node.body) dispatch(*stmt);
    }
    void visit(if_statement& node) override {
        dispatch(*node.test);
        for (auto& stmt : node.body) dispatch(*stmt);
        for (auto& stmt : node.alternate) dispatch(*stmt);
    }
    void visit(for_statement& node) override {
        dispatch(*node.loopvar);
        dispatch(*node.iterable);
        for (auto& stmt : node.body) dispatch(*stmt);
        for (auto& stmt : node.default_block) dispatch(*stmt);
    }
    void visit(set_statement& node) override {
        dispatch(*node.assignee);
        if (node.val) dispatch(*node.val);
        for (auto& stmt : node.body) dispatch(*stmt);
    }
    void visit(macro_statement& node) override {
        dispatch(*node.name);
        for (auto& arg : node.args) dispatch(*arg);
        for (auto& stmt : node.body) dispatch(*stmt);
    }
    void visit(comment_statement&) override {}
    void visit(break_statement&) override {}
    void visit(continue_statement&) override {}
    void visit(filter_statement& node) override {
        dispatch(*node.filter);
        for (auto& stmt : node.body) dispatch(*stmt);
    }
    void visit(call_statement& node) override {
        dispatch(*node.call);
        for (auto& arg : node.caller_args) dispatch(*arg);
        for (auto& stmt : node.body) dispatch(*stmt);
    }

    // Expressions - default empty implementations
    void visit(member_expression& node) override {
        dispatch(*node.object);
        dispatch(*node.property);
    }
    void visit(call_expression& node) override {
        dispatch(*node.callee);
        for (auto& arg : node.args) dispatch(*arg);
    }
    void visit(identifier&) override {}
    void visit(integer_literal&) override {}
    void visit(float_literal&) override {}
    void visit(string_literal&) override {}
    void visit(array_literal& node) override {
        for (auto& item : node.val) dispatch(*item);
    }
    void visit(tuple_literal& node) override {
        for (auto& item : node.val) dispatch(*item);
    }
    void visit(object_literal& node) override {
        for (auto& [key, val] : node.val) {
            dispatch(*key);
            dispatch(*val);
        }
    }
    void visit(binary_expression& node) override {
        dispatch(*node.left);
        dispatch(*node.right);
    }
    void visit(filter_expression& node) override {
        if (node.operand) dispatch(*node.operand);
        dispatch(*node.filter);
    }
    void visit(select_expression& node) override {
        dispatch(*node.lhs);
        dispatch(*node.test);
    }
    void visit(test_expression& node) override {
        dispatch(*node.operand);
        dispatch(*node.test);
    }
    void visit(unary_expression& node) override {
        dispatch(*node.argument);
    }
    void visit(slice_expression& node) override {
        if (node.start_expr) dispatch(*node.start_expr);
        if (node.stop_expr) dispatch(*node.stop_expr);
        if (node.step_expr) dispatch(*node.step_expr);
    }
    void visit(keyword_argument_expression& node) override {
        dispatch(*node.key);
        dispatch(*node.val);
    }
    void visit(spread_expression& node) override {
        dispatch(*node.argument);
    }
    void visit(ternary_expression& node) override {
        dispatch(*node.condition);
        dispatch(*node.true_expr);
        dispatch(*node.false_expr);
    }
};

} // namespace jinja
