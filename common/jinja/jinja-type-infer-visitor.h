#pragma once

#include "jinja-visitor.h"
#include "jinja-type-system.h"
#include "jinja-constraints.h"
#include <map>
#include <string>
#include <stack>
#include <memory>
#include <set>

namespace jinja {
namespace types {

/**
 * Type environment - maps variable names to their types
 * Supports nested scopes via parent pointer
 */
struct type_environment {
    std::map<std::string, TypePtr> bindings;
    type_environment* parent = nullptr;

    type_environment() = default;
    explicit type_environment(type_environment* p) : parent(p) {}

    TypePtr lookup(const std::string& name) const {
        auto it = bindings.find(name);
        if (it != bindings.end()) return it->second;
        if (parent) return parent->lookup(name);
        return nullptr;
    }

    void bind(const std::string& name, TypePtr type) {
        bindings[name] = std::move(type);
    }

    // Get all bindings (including from parent scopes)
    std::map<std::string, TypePtr> all_bindings() const {
        std::map<std::string, TypePtr> result;
        if (parent) result = parent->all_bindings();
        for (const auto& [k, v] : bindings) {
            result[k] = v;
        }
        return result;
    }

    // Get only root-level bindings (no inherited)
    const std::map<std::string, TypePtr>& local_bindings() const {
        return bindings;
    }
};

/**
 * Type guard - captures type narrowing from test expressions like "x is string"
 */
struct type_guard {
    std::string variable;
    TypePtr object_type;
    std::string field_name;
    TypePtr narrowed_type;
    bool negated = false;

    type_guard() = default;

    type_guard(const std::string& var, TypePtr type, bool neg = false)
        : variable(var), narrowed_type(std::move(type)), negated(neg) {}

    type_guard(TypePtr obj_type, const std::string& field, TypePtr type, bool neg = false)
        : object_type(std::move(obj_type)), field_name(field), narrowed_type(std::move(type)), negated(neg) {}

    bool is_field_guard() const { return object_type != nullptr && !field_name.empty(); }
    bool is_variable_guard() const { return !variable.empty(); }
    bool valid() const { return (is_variable_guard() || is_field_guard()) && narrowed_type != nullptr; }
};

struct inference_context {
    bool in_conditional = false;
    bool in_loop = false;
    bool in_output = false;
    int depth = 0;

    inference_context enter_conditional() const {
        return inference_context{true, in_loop, in_output, depth + 1};
    }

    inference_context enter_loop() const {
        return inference_context{in_conditional, true, in_output, depth + 1};
    }

    inference_context enter_output() const {
        return inference_context{in_conditional, in_loop, true, depth};
    }
};

/**
 * Type inference visitor - generates constraints from AST
 */
struct type_inference_visitor : public ast_visitor<TypePtr> {
    constraint_set& constraints;
    type_environment* env;
    inference_context ctx;

    std::vector<std::unique_ptr<type_environment>> env_stack;
    type_environment* root_env;

    type_inference_visitor(constraint_set& cs, type_environment* root)
        : constraints(cs), env(root), ctx(), root_env(root) {}

    type_guard extract_type_guard(statement& test_node, bool negated = false) {
        if (auto* unary = dynamic_cast<unary_expression*>(&test_node)) {
            if (unary->op.value == "not") {
                return extract_type_guard(*unary->argument, !negated);
            }
        }

        if (auto* test = dynamic_cast<test_expression*>(&test_node)) {
            std::string var_name;
            TypePtr member_object_type;
            std::string member_field_name;

            if (auto* id = dynamic_cast<identifier*>(test->operand.get())) {
                var_name = id->val;
            } else if (auto* member = dynamic_cast<member_expression*>(test->operand.get())) {
                member_object_type = dispatch(*test->operand);

                if (!member->computed) {
                    if (auto* prop_id = dynamic_cast<identifier*>(member->property.get())) {
                        member_field_name = prop_id->val;
                    }
                } else {
                    if (auto* str_lit = dynamic_cast<string_literal*>(member->property.get())) {
                        member_field_name = str_lit->val;
                    }
                }

                if (member_field_name.empty()) {
                    return type_guard{};
                }
            } else {
                return type_guard{};
            }

            std::string test_name;
            if (auto* test_id = dynamic_cast<identifier*>(test->test.get())) {
                test_name = test_id->val;
            } else if (auto* call = dynamic_cast<call_expression*>(test->test.get())) {
                if (auto* callee_id = dynamic_cast<identifier*>(call->callee.get())) {
                    test_name = callee_id->val;
                }
            }

            if (test->negate) {
                negated = !negated;
            }

            TypePtr narrowed_type = nullptr;
            if (test_name == "string") {
                narrowed_type = make_string();
            } else if (test_name == "number" || test_name == "integer") {
                narrowed_type = make_int();
            } else if (test_name == "float") {
                narrowed_type = make_float();
            } else if (test_name == "boolean") {
                narrowed_type = make_bool();
            } else if (test_name == "none" || test_name == "null") {
                narrowed_type = make_null();
            } else if (test_name == "defined") {
                if (!negated) {
                    auto current = env->lookup(var_name);
                    if (current) {
                        narrowed_type = current;
                    }
                }
            } else if (test_name == "iterable" || test_name == "sequence") {
                narrowed_type = make_array(fresh());
            } else if (test_name == "mapping") {
                narrowed_type = make_object(true);
            }

            if (narrowed_type) {
                if (!var_name.empty()) {
                    return type_guard(var_name, narrowed_type, negated);
                } else if (!member_field_name.empty()) {
                    return type_guard(member_object_type, member_field_name, narrowed_type, negated);
                }
            }
        }

        if (auto* id = dynamic_cast<identifier*>(&test_node)) {
            auto current = env->lookup(id->val);
            if (current && !negated) {
                return type_guard(id->val, current, false);
            }
        }

        return type_guard{};
    }

    void apply_type_guard(const type_guard& guard) {
        if (!guard.valid() || guard.negated) return;

        if (guard.is_variable_guard()) {
            auto existing = env->lookup(guard.variable);
            env->bind(guard.variable, guard.narrowed_type);
            if (existing) {
                constraints.add_type_alternative(existing, guard.narrowed_type,
                    "type guard narrowing " + guard.variable);
            }
        } else if (guard.is_field_guard()) {
            constraints.add_type_alternative(guard.object_type, guard.narrowed_type,
                "type guard narrowing field " + guard.field_name);
        }
    }

    type_guard invert_guard(const type_guard& guard) const {
        if (!guard.valid()) return type_guard{};

        if (guard.is_variable_guard()) {
            return type_guard(guard.variable, guard.narrowed_type, !guard.negated);
        } else if (guard.is_field_guard()) {
            return type_guard(guard.object_type, guard.field_name, guard.narrowed_type, !guard.negated);
        }
        return type_guard{};
    }

    void push_scope() {
        auto new_env = std::make_unique<type_environment>(env);
        env = new_env.get();
        env_stack.push_back(std::move(new_env));
    }

    void pop_scope() {
        if (!env_stack.empty()) {
            env_stack.pop_back();
            env = env_stack.empty() ? root_env : env_stack.back().get();
        }
    }

    TypePtr fresh() { return make_typevar(type_var_generator::fresh()); }

    std::string source_loc(const statement& node) {
        return "pos:" + std::to_string(node.pos);
    }

    TypePtr visit(program& node) override {
        for (auto& stmt : node.body) {
            visit_body_statement(*stmt);
        }
        return make_null();
    }

    TypePtr visit(if_statement& node) override {
        dispatch(*node.test);
        auto guard = extract_type_guard(*node.test);

        auto old_ctx = ctx;
        ctx = ctx.enter_conditional();

        push_scope();
        if (guard.valid() && !guard.negated) {
            apply_type_guard(guard);
        }
        for (auto& stmt : node.body) {
            visit_body_statement(*stmt);
        }
        auto true_branch_bindings = env->local_bindings();
        pop_scope();

        push_scope();
        if (guard.valid()) {
            auto inverted = invert_guard(guard);
            apply_type_guard(inverted);
        }
        for (auto& stmt : node.alternate) {
            visit_body_statement(*stmt);
        }
        auto false_branch_bindings = env->local_bindings();
        pop_scope();

        merge_branch_bindings(true_branch_bindings, false_branch_bindings);

        ctx = old_ctx;
        return make_null();
    }

    void merge_branch_bindings(
        const std::map<std::string, TypePtr>& true_bindings,
        const std::map<std::string, TypePtr>& false_bindings)
    {
        std::set<std::string> all_vars;
        for (const auto& [name, _] : true_bindings) all_vars.insert(name);
        for (const auto& [name, _] : false_bindings) all_vars.insert(name);

        for (const auto& var_name : all_vars) {
            auto true_it = true_bindings.find(var_name);
            auto false_it = false_bindings.find(var_name);

            TypePtr true_type = (true_it != true_bindings.end()) ? true_it->second : nullptr;
            TypePtr false_type = (false_it != false_bindings.end()) ? false_it->second : nullptr;

            TypePtr original_type = env->lookup(var_name);

            TypePtr merged_type;

            if (true_type && false_type) {
                if (true_type == false_type || (true_type && false_type && true_type->equals(false_type))) {
                    merged_type = true_type;
                } else {
                    auto u = std::make_shared<union_type>();
                    u->add_alternative(true_type);
                    u->add_alternative(false_type);
                    merged_type = u;
                }
            } else if (true_type) {
                if (original_type && original_type != true_type && !original_type->equals(true_type)) {
                    auto u = std::make_shared<union_type>();
                    u->add_alternative(true_type);
                    u->add_alternative(original_type);
                    merged_type = u;
                } else {
                    merged_type = true_type;
                }
            } else if (false_type) {
                if (original_type && original_type != false_type && !original_type->equals(false_type)) {
                    auto u = std::make_shared<union_type>();
                    u->add_alternative(false_type);
                    u->add_alternative(original_type);
                    merged_type = u;
                } else {
                    merged_type = false_type;
                }
            }

            if (merged_type) {
                env->bind(var_name, merged_type);
            }
        }
    }

    TypePtr visit(for_statement& node) override {
        auto iterable_type = dispatch(*node.iterable);
        auto element_type = fresh();
        constraints.add_iterable(iterable_type, element_type, "for loop at " + source_loc(node));

        push_scope();
        auto old_ctx = ctx;
        ctx = ctx.enter_loop();

        if (is_stmt<identifier>(node.loopvar)) {
            auto* id = cast_stmt<identifier>(node.loopvar);
            env->bind(id->val, element_type);
        } else if (is_stmt<tuple_literal>(node.loopvar)) {
            auto* tuple = cast_stmt<tuple_literal>(node.loopvar);
            for (size_t i = 0; i < tuple->val.size(); ++i) {
                if (is_stmt<identifier>(tuple->val[i])) {
                    auto* id = cast_stmt<identifier>(tuple->val[i]);
                    auto var_type = fresh();
                    env->bind(id->val, var_type);
                    constraints.add_array_element(element_type, var_type,
                        "tuple unpack element " + std::to_string(i));
                }
            }
        }

        auto loop_obj = std::make_shared<object_type>(false);
        loop_obj->add_field("index", make_int());
        loop_obj->add_field("index0", make_int());
        loop_obj->add_field("first", make_bool());
        loop_obj->add_field("last", make_bool());
        loop_obj->add_field("length", make_int());
        loop_obj->add_field("revindex", make_int());
        loop_obj->add_field("revindex0", make_int());
        loop_obj->add_field("previtem", make_optional(element_type));
        loop_obj->add_field("nextitem", make_optional(element_type));
        env->bind("loop", loop_obj);

        for (auto& stmt : node.body) {
            visit_body_statement(*stmt);
        }

        ctx = old_ctx;
        pop_scope();

        for (auto& stmt : node.default_block) {
            visit_body_statement(*stmt);
        }

        return make_null();
    }

    void visit_body_statement(statement& stmt) {
        bool is_output_expr = dynamic_cast<expression*>(&stmt) != nullptr &&
                              !dynamic_cast<if_statement*>(&stmt) &&
                              !dynamic_cast<for_statement*>(&stmt) &&
                              !dynamic_cast<set_statement*>(&stmt) &&
                              !dynamic_cast<macro_statement*>(&stmt) &&
                              !dynamic_cast<filter_statement*>(&stmt) &&
                              !dynamic_cast<call_statement*>(&stmt);

        if (is_output_expr) {
            auto old_ctx = ctx;
            ctx = ctx.enter_output();
            auto expr_type = dispatch(stmt);
            constraints.add_output_coercion(expr_type, "output expression at " + source_loc(stmt));
            ctx = old_ctx;
        } else {
            dispatch(stmt);
        }
    }

    TypePtr visit(set_statement& node) override {
        TypePtr rhs_type;
        if (node.val) {
            rhs_type = dispatch(*node.val);
        } else if (!node.body.empty()) {
            for (auto& stmt : node.body) dispatch(*stmt);
            rhs_type = make_string();
        } else {
            rhs_type = make_null();
        }

        if (is_stmt<identifier>(node.assignee)) {
            env->bind(cast_stmt<identifier>(node.assignee)->val, rhs_type);
        } else if (is_stmt<tuple_literal>(node.assignee)) {
            auto* tuple = cast_stmt<tuple_literal>(node.assignee);
            for (size_t i = 0; i < tuple->val.size(); ++i) {
                if (is_stmt<identifier>(tuple->val[i])) {
                    auto* id = cast_stmt<identifier>(tuple->val[i]);
                    auto var_type = fresh();
                    env->bind(id->val, var_type);
                    constraints.add_array_element(rhs_type, var_type,
                        "set tuple unpack " + std::to_string(i));
                }
            }
        } else if (is_stmt<member_expression>(node.assignee)) {
            dispatch(*node.assignee);
        }

        return make_null();
    }

    TypePtr visit(macro_statement& node) override {
        std::vector<TypePtr> param_types;
        push_scope();

        for (auto& arg : node.args) {
            auto param_type = fresh();
            param_types.push_back(param_type);

            if (is_stmt<identifier>(arg)) {
                env->bind(cast_stmt<identifier>(arg)->val, param_type);
            } else if (is_stmt<keyword_argument_expression>(arg)) {
                auto* kwarg = cast_stmt<keyword_argument_expression>(arg);
                if (is_stmt<identifier>(kwarg->key)) {
                    env->bind(cast_stmt<identifier>(kwarg->key)->val, param_type);
                }
            }
        }

        for (auto& stmt : node.body) dispatch(*stmt);
        pop_scope();

        if (is_stmt<identifier>(node.name)) {
            env->bind(cast_stmt<identifier>(node.name)->val,
                std::make_shared<function_type>(std::move(param_types), make_string()));
        }

        return make_null();
    }

    TypePtr visit(comment_statement&) override {
        return make_null();
    }

    TypePtr visit(break_statement&) override {
        return make_null();
    }

    TypePtr visit(continue_statement&) override {
        return make_null();
    }

    TypePtr visit(filter_statement& node) override {
        for (auto& stmt : node.body) {
            dispatch(*stmt);
        }
        dispatch(*node.filter);
        return make_string();
    }

    TypePtr visit(call_statement& node) override {
        dispatch(*node.call);
        for (auto& arg : node.caller_args) dispatch(*arg);
        push_scope();
        env->bind("caller", make_function({}, make_string()));
        for (auto& stmt : node.body) dispatch(*stmt);
        pop_scope();
        return make_null();
    }

    TypePtr visit(identifier& node) override {
        auto type = env->lookup(node.val);
        if (type) return type;

        auto var_type = fresh();
        root_env->bind(node.val, var_type);
        return var_type;
    }

    TypePtr visit(member_expression& node) override {
        auto object_type = dispatch(*node.object);

        if (!node.computed) {
            if (is_stmt<identifier>(node.property)) {
                auto* prop_id = cast_stmt<identifier>(node.property);
                auto field_type = fresh();
                constraints.add_field(object_type, prop_id->val, field_type,
                    ctx.in_conditional, "member access ." + prop_id->val + " at " + source_loc(node));
                return field_type;
            }
        } else {
            if (is_stmt<slice_expression>(node.property)) {
                auto* slice = cast_stmt<slice_expression>(node.property);
                if (slice->start_expr) dispatch(*slice->start_expr);
                if (slice->stop_expr) dispatch(*slice->stop_expr);
                if (slice->step_expr) dispatch(*slice->step_expr);
                return object_type;
            }

            dispatch(*node.property);

            if (is_stmt<string_literal>(node.property)) {
                auto* str_lit = cast_stmt<string_literal>(node.property);
                auto field_type = fresh();
                constraints.add_field(object_type, str_lit->val, field_type,
                    ctx.in_conditional, "dict access ['" + str_lit->val + "'] at " + source_loc(node));
                return field_type;
            }

            if (is_stmt<integer_literal>(node.property)) {
                auto elem_type = fresh();
                constraints.add_array_element(object_type, elem_type, "array index at " + source_loc(node));
                return elem_type;
            }

            auto result_type = fresh();
            constraints.add_array_element(object_type, result_type, "dynamic access at " + source_loc(node));
            return result_type;
        }

        return fresh();
    }

    TypePtr visit(call_expression& node) override {
        // Check if this is a string method call like content.startswith(...)
        if (is_stmt<member_expression>(node.callee)) {
            auto* member = cast_stmt<member_expression>(node.callee);
            if (!member->computed && is_stmt<identifier>(member->property)) {
                auto* method_id = cast_stmt<identifier>(member->property);
                const std::string& method_name = method_id->val;

                // Known string methods
                static const std::set<std::string> string_methods_bool = {
                    "startswith", "endswith", "isalpha", "isdigit", "isalnum",
                    "isspace", "isupper", "islower", "isnumeric", "isdecimal"
                };
                static const std::set<std::string> string_methods_string = {
                    "strip", "lstrip", "rstrip", "lower", "upper", "title",
                    "capitalize", "swapcase", "center", "ljust", "rjust",
                    "zfill", "replace", "format"
                };
                static const std::set<std::string> string_methods_int = {
                    "find", "rfind", "index", "rindex", "count"
                };
                static const std::set<std::string> string_methods_array_string = {
                    "split", "rsplit", "splitlines"
                };

                bool is_string_method = string_methods_bool.count(method_name) ||
                                        string_methods_string.count(method_name) ||
                                        string_methods_int.count(method_name) ||
                                        string_methods_array_string.count(method_name);

                if (is_string_method) {
                    auto object_type = dispatch(*member->object);
                    constraints.add_equality(object_type, make_string(),
                        "string method ." + method_name + "() at " + source_loc(node));

                    for (auto& arg : node.args) {
                        dispatch(*arg);
                    }

                    if (string_methods_bool.count(method_name)) {
                        return make_bool();
                    } else if (string_methods_string.count(method_name)) {
                        return make_string();
                    } else if (string_methods_int.count(method_name)) {
                        return make_int();
                    } else if (string_methods_array_string.count(method_name)) {
                        return make_array(make_string());
                    }
                }
            }
        }

        auto callee_type = dispatch(*node.callee);
        std::vector<TypePtr> arg_types;

        for (auto& arg : node.args) {
            arg_types.push_back(dispatch(*arg));
        }

        auto return_type = fresh();
        constraints.add_callable(callee_type, std::move(arg_types), return_type,
            "call at " + source_loc(node));

        return return_type;
    }

    TypePtr visit(integer_literal&) override {
        return make_int();
    }

    TypePtr visit(float_literal&) override {
        return make_float();
    }

    TypePtr visit(string_literal&) override {
        return make_string();
    }

    TypePtr visit(array_literal& node) override {
        auto elem_type = fresh();
        for (auto& item : node.val) {
            auto item_type = dispatch(*item);
            constraints.add_equality(elem_type, item_type,
                "array element at " + source_loc(node));
        }
        return make_array(elem_type);
    }

    TypePtr visit(tuple_literal& node) override {
        return visit(static_cast<array_literal&>(node));
    }

    TypePtr visit(object_literal& node) override {
        auto obj = std::make_shared<object_type>(false);
        for (auto& [key_stmt, val_stmt] : node.val) {
            auto val_type = dispatch(*val_stmt);
            if (is_stmt<string_literal>(key_stmt)) {
                auto* str = cast_stmt<string_literal>(key_stmt);
                obj->add_field(str->val, val_type);
            } else if (is_stmt<identifier>(key_stmt)) {
                auto* id = cast_stmt<identifier>(key_stmt);
                obj->add_field(id->val, val_type);
            }
        }
        return obj;
    }

    TypePtr visit(binary_expression& node) override {
        auto left_type = dispatch(*node.left);
        auto right_type = dispatch(*node.right);
        const std::string& op = node.op.value;

        if (op == "==" || op == "!=" || op == "<" || op == ">" ||
            op == "<=" || op == ">=" || op == "in" || op == "not in") {

            bool is_containment = (op == "in" || op == "not in");
            bool is_equality = (op == "==" || op == "!=");

            if (!is_containment) {
                if (is_stmt<string_literal>(node.left)) {
                    auto* str_lit = cast_stmt<string_literal>(node.left);
                    if (is_equality) {
                        constraints.add_literal(right_type, make_literal_string(str_lit->val),
                            "equality with string literal at " + source_loc(node));
                    } else {
                        constraints.add_equality(right_type, make_string(),
                            "comparison with string literal at " + source_loc(node));
                    }
                } else if (is_stmt<string_literal>(node.right)) {
                    auto* str_lit = cast_stmt<string_literal>(node.right);
                    if (is_equality) {
                        constraints.add_literal(left_type, make_literal_string(str_lit->val),
                            "equality with string literal at " + source_loc(node));
                    } else {
                        constraints.add_equality(left_type, make_string(),
                            "comparison with string literal at " + source_loc(node));
                    }
                }

                if (is_stmt<integer_literal>(node.left)) {
                    auto* int_lit = cast_stmt<integer_literal>(node.left);
                    if (is_equality) {
                        constraints.add_literal(right_type, make_literal_int(int_lit->val),
                            "equality with int literal at " + source_loc(node));
                    } else {
                        constraints.add_equality(right_type, make_int(),
                            "comparison with int literal at " + source_loc(node));
                    }
                } else if (is_stmt<integer_literal>(node.right)) {
                    auto* int_lit = cast_stmt<integer_literal>(node.right);
                    if (is_equality) {
                        constraints.add_literal(left_type, make_literal_int(int_lit->val),
                            "equality with int literal at " + source_loc(node));
                    } else {
                        constraints.add_equality(left_type, make_int(),
                            "comparison with int literal at " + source_loc(node));
                    }
                }
            }

            return make_bool();
        }

        if (op == "and" || op == "or") {
            auto u = std::make_shared<union_type>();
            u->add_alternative(left_type);
            u->add_alternative(right_type);
            return u;
        }

        if (op == "~") {
            constraints.add_string_operand(left_type, "string concat lhs at " + source_loc(node));
            constraints.add_string_operand(right_type, "string concat rhs at " + source_loc(node));
            return make_string();
        }

        if (op == "+") {
            if (is_stmt<string_literal>(node.left)) {
                constraints.add_output_coercion(right_type, "string concat rhs at " + source_loc(node));
                return make_string();
            } else if (is_stmt<string_literal>(node.right)) {
                constraints.add_output_coercion(left_type, "string concat lhs at " + source_loc(node));
                return make_string();
            }
            return fresh();
        }

        if (op == "-" || op == "*" || op == "/" || op == "%" || op == "//") {
            return fresh();
        }

        return fresh();
    }

    TypePtr visit(filter_expression& node) override {
        TypePtr input_type = node.operand ? dispatch(*node.operand) : make_string();

        if (is_stmt<identifier>(node.filter)) {
            const auto& name = cast_stmt<identifier>(node.filter)->val;

            // tojson accepts any type - mark input as 'any'
            if (name == "tojson") {
                constraints.add_equality(input_type, make_any(),
                    "tojson input accepts any type at " + source_loc(node));
                return make_string();
            }

            if (name == "trim" || name == "strip" ||
                name == "upper" || name == "lower" || name == "title" ||
                name == "capitalize" || name == "safe" || name == "e" ||
                name == "escape" || name == "string" || name == "indent") {
                return make_string();
            }

            if (name == "length" || name == "count" || name == "int" || name == "abs") {
                return make_int();
            }

            if (name == "float" || name == "round") return make_float();
            if (name == "bool") return make_bool();
            if (name == "default" || name == "d") return input_type;

            if (name == "list" || name == "sort" || name == "reverse" || name == "unique" || name == "slice") {
                auto elem = fresh();
                constraints.add_iterable(input_type, elem, "filter " + name + " at " + source_loc(node));
                return make_array(elem);
            }

            if (name == "batch") {
                auto elem = fresh();
                constraints.add_iterable(input_type, elem, "filter batch at " + source_loc(node));
                return make_array(make_array(elem));
            }

            if (name == "first" || name == "last" || name == "random") {
                auto elem = fresh();
                constraints.add_array_element(input_type, elem, "filter " + name + " at " + source_loc(node));
                return elem;
            }

            if (name == "join") {
                auto elem = fresh();
                constraints.add_iterable(input_type, elem, "filter join at " + source_loc(node));
                return make_string();
            }

            if (name == "select" || name == "reject" || name == "selectattr" || name == "rejectattr" || name == "map") {
                auto elem = fresh();
                constraints.add_iterable(input_type, elem, "filter " + name + " at " + source_loc(node));
                return make_array(elem);
            }
        }

        if (is_stmt<call_expression>(node.filter)) {
            auto* call = cast_stmt<call_expression>(node.filter);
            if (is_stmt<identifier>(call->callee)) {
                const auto& name = cast_stmt<identifier>(call->callee)->val;

                auto find_kwarg = [&](const std::string& key) -> statement* {
                    for (auto& arg : call->args) {
                        if (auto* kw = cast_stmt<keyword_argument_expression>(arg)) {
                            if (auto* kid = cast_stmt<identifier>(kw->key)) {
                                if (kid->val == key) return kw->val.get();
                            }
                        }
                    }
                    return nullptr;
                };

                for (auto& arg : call->args) dispatch(*arg);

                if (name == "tojson") {
                    constraints.add_equality(input_type, make_any(),
                        "tojson input accepts any type at " + source_loc(node));
                    return make_string();
                }

                if (name == "trim" || name == "indent" || name == "replace") {
                    return make_string();
                }

                if (name == "join") {
                    auto elem = fresh();
                    constraints.add_iterable(input_type, elem, "filter join at " + source_loc(node));
                    return make_string();
                }

                if (name == "default" || name == "d") return input_type;

                if (name == "slice" || name == "sort" || name == "reverse" || name == "unique" || name == "list") {
                    auto elem = fresh();
                    constraints.add_iterable(input_type, elem, "filter " + name + " at " + source_loc(node));
                    return make_array(elem);
                }

                if (name == "batch") {
                    auto elem = fresh();
                    constraints.add_iterable(input_type, elem, "filter batch at " + source_loc(node));
                    return make_array(make_array(elem));
                }

                if (name == "map") {
                    auto elem = fresh();
                    constraints.add_iterable(input_type, elem, "filter map at " + source_loc(node));
                    if (auto* attr_arg = find_kwarg("attribute")) {
                        if (auto* attr_str = dynamic_cast<string_literal*>(attr_arg)) {
                            auto attr_type = fresh();
                            constraints.add_field(elem, attr_str->val, attr_type, false,
                                "map attribute '" + attr_str->val + "' at " + source_loc(node));
                            return make_array(attr_type);
                        }
                    }
                    return make_array(elem);
                }

                if (name == "selectattr" || name == "rejectattr") {
                    auto elem = fresh();
                    constraints.add_iterable(input_type, elem, "filter " + name + " at " + source_loc(node));
                    if (!call->args.empty()) {
                        if (auto* attr_str = cast_stmt<string_literal>(call->args[0])) {
                            auto attr_type = fresh();
                            constraints.add_field(elem, attr_str->val, attr_type, false,
                                name + " attribute '" + attr_str->val + "' at " + source_loc(node));
                        }
                    }
                    return make_array(elem);
                }

                if (name == "select" || name == "reject") {
                    auto elem = fresh();
                    constraints.add_iterable(input_type, elem, "filter " + name + " at " + source_loc(node));
                    return make_array(elem);
                }

                if (name == "first" || name == "last" || name == "random") {
                    auto elem = fresh();
                    constraints.add_array_element(input_type, elem, "filter " + name + " at " + source_loc(node));
                    return elem;
                }
            }
        }

        return fresh();
    }

    TypePtr visit(select_expression& node) override {
        auto input_type = dispatch(*node.lhs);
        dispatch(*node.test);
        auto elem = fresh();
        constraints.add_iterable(input_type, elem, "select expression at " + source_loc(node));
        return make_array(elem);
    }

    TypePtr visit(test_expression& node) override {
        dispatch(*node.operand);
        return make_bool();
    }

    TypePtr visit(unary_expression& node) override {
        auto arg_type = dispatch(*node.argument);
        if (node.op.value == "not") return make_bool();
        if (node.op.value == "-") return arg_type;
        return fresh();
    }

    TypePtr visit(slice_expression& node) override {
        if (node.start_expr) dispatch(*node.start_expr);
        if (node.stop_expr) dispatch(*node.stop_expr);
        if (node.step_expr) dispatch(*node.step_expr);
        return fresh();
    }

    TypePtr visit(keyword_argument_expression& node) override {
        return dispatch(*node.val);
    }

    TypePtr visit(spread_expression& node) override {
        return dispatch(*node.argument);
    }

    TypePtr visit(ternary_expression& node) override {
        dispatch(*node.condition);
        auto old_ctx = ctx;
        ctx = ctx.enter_conditional();
        auto true_type = dispatch(*node.true_expr);
        auto false_type = dispatch(*node.false_expr);
        ctx = old_ctx;
        auto u = std::make_shared<union_type>();
        u->add_alternative(true_type);
        u->add_alternative(false_type);
        return u;
    }
};

} // namespace types
} // namespace jinja
