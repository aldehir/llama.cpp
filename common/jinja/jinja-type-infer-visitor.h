#pragma once

#include "jinja-visitor.h"
#include "jinja-type-system.h"
#include "jinja-constraints.h"
#include <map>
#include <string>
#include <stack>
#include <memory>

namespace jinja {
namespace types {

/**
 * Type environment - maps variable names to their types
 * Supports nested scopes via parent pointer
 */
struct TypeEnvironment {
    std::map<std::string, TypePtr> bindings;
    TypeEnvironment* parent = nullptr;

    TypeEnvironment() = default;
    explicit TypeEnvironment(TypeEnvironment* p) : parent(p) {}

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
 * Context tracking for conditional access analysis
 */
struct InferenceContext {
    bool in_conditional = false;  // Inside if/ternary
    bool in_loop = false;         // Inside for loop
    int depth = 0;                // Nesting depth

    InferenceContext enter_conditional() const {
        return InferenceContext{true, in_loop, depth + 1};
    }

    InferenceContext enter_loop() const {
        return InferenceContext{in_conditional, true, depth + 1};
    }
};

/**
 * Type inference visitor - generates constraints from AST
 */
struct TypeInferenceVisitor : public ASTVisitor<TypePtr> {
    ConstraintSet& constraints;
    TypeEnvironment* env;
    InferenceContext ctx;

    // Stack of environments for scoping
    std::vector<std::unique_ptr<TypeEnvironment>> env_stack;

    // Root environment (kept separate for final result extraction)
    TypeEnvironment* root_env;

    TypeInferenceVisitor(ConstraintSet& cs, TypeEnvironment* root)
        : constraints(cs), env(root), ctx(), root_env(root) {}

    // Scope management
    void push_scope() {
        auto new_env = std::make_unique<TypeEnvironment>(env);
        env = new_env.get();
        env_stack.push_back(std::move(new_env));
    }

    void pop_scope() {
        if (!env_stack.empty()) {
            env_stack.pop_back();
            env = env_stack.empty() ? root_env : env_stack.back().get();
        }
    }

    // Fresh type variable
    TypePtr fresh() {
        return make_typevar(TypeVarGenerator::fresh());
    }

    // Get source location string for debugging
    std::string source_loc(const statement& node) {
        return "pos:" + std::to_string(node.pos);
    }

    // === Statement visitors ===

    TypePtr visit(program& node) override {
        for (auto& stmt : node.body) {
            dispatch(*stmt);
        }
        return make_null();
    }

    TypePtr visit(if_statement& node) override {
        // Visit test expression
        auto test_type = dispatch(*node.test);

        // Visit body in conditional context
        auto old_ctx = ctx;
        ctx = ctx.enter_conditional();

        for (auto& stmt : node.body) {
            dispatch(*stmt);
        }
        for (auto& stmt : node.alternate) {
            dispatch(*stmt);
        }

        ctx = old_ctx;
        return make_null();
    }

    TypePtr visit(for_statement& node) override {
        // Analyze iterable
        auto iterable_type = dispatch(*node.iterable);
        auto element_type = fresh();

        constraints.add_iterable(iterable_type, element_type,
            "for loop at " + source_loc(node));

        // Create new scope for loop variables
        push_scope();
        auto old_ctx = ctx;
        ctx = ctx.enter_loop();

        // Bind loop variable(s)
        if (is_stmt<identifier>(node.loopvar)) {
            auto* id = cast_stmt<identifier>(node.loopvar);
            env->bind(id->val, element_type);
        } else if (is_stmt<tuple_literal>(node.loopvar)) {
            auto* tuple = cast_stmt<tuple_literal>(node.loopvar);
            // Element must be a tuple/array for unpacking
            for (size_t i = 0; i < tuple->val.size(); ++i) {
                if (is_stmt<identifier>(tuple->val[i])) {
                    auto* id = cast_stmt<identifier>(tuple->val[i]);
                    auto var_type = fresh();
                    env->bind(id->val, var_type);
                    // Add constraint: element_type[i] = var_type
                    constraints.add_array_element(element_type, var_type,
                        "tuple unpack element " + std::to_string(i));
                }
            }
        }

        // Built-in loop object
        auto loop_obj = std::make_shared<ObjectType>(false);  // not extensible
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

        // Visit body
        for (auto& stmt : node.body) {
            dispatch(*stmt);
        }

        ctx = old_ctx;
        pop_scope();

        // Visit default block (outside loop scope)
        for (auto& stmt : node.default_block) {
            dispatch(*stmt);
        }

        return make_null();
    }

    TypePtr visit(set_statement& node) override {
        TypePtr rhs_type;
        if (node.val) {
            rhs_type = dispatch(*node.val);
        } else if (!node.body.empty()) {
            // Block set: {% set x %}...{% endset %}
            for (auto& stmt : node.body) {
                dispatch(*stmt);
            }
            rhs_type = make_string();  // Block set produces string
        } else {
            rhs_type = make_null();
        }

        if (is_stmt<identifier>(node.assignee)) {
            auto* id = cast_stmt<identifier>(node.assignee);
            env->bind(id->val, rhs_type);
        } else if (is_stmt<tuple_literal>(node.assignee)) {
            // Tuple unpacking in set
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
            // Setting a field: {% set obj.field = value %}
            dispatch(*node.assignee);
        }

        return make_null();
    }

    TypePtr visit(macro_statement& node) override {
        // Create function type
        std::vector<TypePtr> param_types;
        push_scope();

        for (auto& arg : node.args) {
            auto param_type = fresh();
            param_types.push_back(param_type);

            if (is_stmt<identifier>(arg)) {
                auto* id = cast_stmt<identifier>(arg);
                env->bind(id->val, param_type);
            } else if (is_stmt<keyword_argument_expression>(arg)) {
                auto* kwarg = cast_stmt<keyword_argument_expression>(arg);
                if (is_stmt<identifier>(kwarg->key)) {
                    auto* id = cast_stmt<identifier>(kwarg->key);
                    env->bind(id->val, param_type);
                }
            }
        }

        // Visit body
        for (auto& stmt : node.body) {
            dispatch(*stmt);
        }

        pop_scope();

        // Bind macro name in outer scope
        if (is_stmt<identifier>(node.name)) {
            auto* name_id = cast_stmt<identifier>(node.name);
            auto func_type = std::make_shared<FunctionType>(
                std::move(param_types), make_string());
            env->bind(name_id->val, func_type);
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
        for (auto& arg : node.caller_args) {
            dispatch(*arg);
        }
        push_scope();
        // Bind 'caller' function
        env->bind("caller", make_function({}, make_string()));
        for (auto& stmt : node.body) {
            dispatch(*stmt);
        }
        pop_scope();
        return make_null();
    }

    // === Expression visitors ===

    TypePtr visit(identifier& node) override {
        auto type = env->lookup(node.val);
        if (type) return type;

        // Unknown identifier - create a type variable for it
        // This is a "root" variable that needs to be inferred from usage
        auto var_type = fresh();
        // Bind in root environment so it persists
        root_env->bind(node.val, var_type);
        return var_type;
    }

    TypePtr visit(member_expression& node) override {
        auto object_type = dispatch(*node.object);

        if (!node.computed) {
            // dot access: obj.field
            if (is_stmt<identifier>(node.property)) {
                auto* prop_id = cast_stmt<identifier>(node.property);
                auto field_type = fresh();

                constraints.add_field(object_type, prop_id->val, field_type,
                    ctx.in_conditional, "member access ." + prop_id->val + " at " + source_loc(node));

                return field_type;
            }
        } else {
            // bracket access: obj[key]
            if (is_stmt<slice_expression>(node.property)) {
                // Slicing: obj[start:stop:step]
                auto* slice = cast_stmt<slice_expression>(node.property);
                if (slice->start_expr) dispatch(*slice->start_expr);
                if (slice->stop_expr) dispatch(*slice->stop_expr);
                if (slice->step_expr) dispatch(*slice->step_expr);
                // Slice returns same array type
                return object_type;
            }

            auto key_type = dispatch(*node.property);

            // Check if key is a string literal (dictionary access)
            if (is_stmt<string_literal>(node.property)) {
                auto* str_lit = cast_stmt<string_literal>(node.property);
                auto field_type = fresh();
                constraints.add_field(object_type, str_lit->val, field_type,
                    ctx.in_conditional, "dict access ['" + str_lit->val + "'] at " + source_loc(node));
                return field_type;
            }

            // Integer access (array indexing)
            if (is_stmt<integer_literal>(node.property)) {
                auto elem_type = fresh();
                constraints.add_array_element(object_type, elem_type,
                    "array index at " + source_loc(node));
                return elem_type;
            }

            // Dynamic access - could be array or object
            auto result_type = fresh();
            // Assume array access for now
            constraints.add_array_element(object_type, result_type,
                "dynamic access at " + source_loc(node));
            return result_type;
        }

        return fresh();
    }

    TypePtr visit(call_expression& node) override {
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
        // Treat same as array for type inference
        return visit(static_cast<array_literal&>(node));
    }

    TypePtr visit(object_literal& node) override {
        auto obj = std::make_shared<ObjectType>(false);
        for (auto& [key_stmt, val_stmt] : node.val) {
            auto val_type = dispatch(*val_stmt);
            // Key could be string literal or identifier
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

        // Comparison operators return bool
        if (op == "==" || op == "!=" || op == "<" || op == ">" ||
            op == "<=" || op == ">=" || op == "in" || op == "not in") {
            return make_bool();
        }

        // Logical operators
        if (op == "and" || op == "or") {
            // Short-circuit: result type depends on operands
            auto u = std::make_shared<UnionType>();
            u->add_alternative(left_type);
            u->add_alternative(right_type);
            return u;
        }

        // String concatenation
        if (op == "~") {
            return make_string();
        }

        // Arithmetic operators: +, -, *, /, %, //
        // + can also be string concatenation or array concatenation
        if (op == "+" || op == "-" || op == "*" || op == "/" ||
            op == "%" || op == "//") {
            // Could be numeric or string - return fresh type variable
            return fresh();
        }

        return fresh();
    }

    TypePtr visit(filter_expression& node) override {
        TypePtr input_type;
        if (node.operand) {
            input_type = dispatch(*node.operand);
        } else {
            input_type = make_string();  // From filter_statement
        }

        // Handle common filters with known types
        if (is_stmt<identifier>(node.filter)) {
            auto* filter_id = cast_stmt<identifier>(node.filter);
            const auto& name = filter_id->val;

            // String-returning filters
            if (name == "tojson" || name == "trim" || name == "strip" ||
                name == "upper" || name == "lower" || name == "title" ||
                name == "capitalize" || name == "safe" || name == "e" ||
                name == "escape" || name == "string" || name == "indent") {
                return make_string();
            }

            // Integer-returning filters
            if (name == "length" || name == "count" || name == "int" || name == "abs") {
                return make_int();
            }

            // Float-returning filters
            if (name == "float" || name == "round") {
                return make_float();
            }

            // Bool-returning filters
            if (name == "bool") {
                return make_bool();
            }

            // Array-returning filters
            if (name == "list" || name == "sort" || name == "reverse" ||
                name == "unique" || name == "batch" || name == "slice") {
                return make_array(fresh());
            }

            // Element-returning filters (from array)
            if (name == "first" || name == "last" || name == "random") {
                auto elem = fresh();
                constraints.add_array_element(input_type, elem,
                    "filter " + name + " at " + source_loc(node));
                return elem;
            }

            // Default filter
            if (name == "default" || name == "d") {
                return input_type;  // Returns input or default value
            }

            // Join returns string
            if (name == "join") {
                return make_string();
            }

            // Map/select return arrays
            if (name == "map" || name == "select" || name == "reject" ||
                name == "selectattr" || name == "rejectattr") {
                return make_array(fresh());
            }
        }

        // Filter with call expression
        if (is_stmt<call_expression>(node.filter)) {
            auto* call = cast_stmt<call_expression>(node.filter);
            if (is_stmt<identifier>(call->callee)) {
                auto* filter_id = cast_stmt<identifier>(call->callee);
                const auto& name = filter_id->val;

                // Visit arguments
                for (auto& arg : call->args) {
                    dispatch(*arg);
                }

                // Apply same filter type rules
                if (name == "tojson" || name == "trim" || name == "indent" ||
                    name == "join" || name == "replace") {
                    return make_string();
                }
                if (name == "default" || name == "d") {
                    return input_type;
                }
                if (name == "slice" || name == "batch") {
                    return make_array(fresh());
                }
            }
        }

        // Unknown filter - return fresh type
        return fresh();
    }

    TypePtr visit(select_expression& node) override {
        dispatch(*node.lhs);
        dispatch(*node.test);
        return make_array(fresh());
    }

    TypePtr visit(test_expression& node) override {
        auto operand_type = dispatch(*node.operand);

        // "is defined" and "is not defined" are common patterns
        if (is_stmt<identifier>(node.test)) {
            auto* test_id = cast_stmt<identifier>(node.test);
            // These tests return bool
            return make_bool();
        }

        return make_bool();
    }

    TypePtr visit(unary_expression& node) override {
        auto arg_type = dispatch(*node.argument);

        if (node.op.value == "not") {
            return make_bool();
        }
        if (node.op.value == "-") {
            // Numeric negation - could be int or float
            return arg_type;
        }

        return fresh();
    }

    TypePtr visit(slice_expression& node) override {
        if (node.start_expr) dispatch(*node.start_expr);
        if (node.stop_expr) dispatch(*node.stop_expr);
        if (node.step_expr) dispatch(*node.step_expr);
        return fresh();  // Slice of array/string
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

        // Result is union of both branches
        auto u = std::make_shared<UnionType>();
        u->add_alternative(true_type);
        u->add_alternative(false_type);
        return u;
    }
};

} // namespace types
} // namespace jinja
