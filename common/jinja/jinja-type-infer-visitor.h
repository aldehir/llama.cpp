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
 * Type guard - captures type narrowing information from test expressions
 * Example: "x is string" produces type_guard{x, string}
 */
struct type_guard {
    std::string variable;  // The variable being narrowed
    TypePtr narrowed_type; // The type it's narrowed to
    bool negated = false;  // True if this is from "is not" or "not (x is ...)"

    type_guard() = default;
    type_guard(const std::string& var, TypePtr type, bool neg = false)
        : variable(var), narrowed_type(std::move(type)), negated(neg) {}

    bool valid() const { return !variable.empty() && narrowed_type != nullptr; }
};

/**
 * Context tracking for conditional access analysis
 */
struct inference_context {
    bool in_conditional = false;  // Inside if/ternary
    bool in_loop = false;         // Inside for loop
    bool in_output = false;       // Inside output expression {{ }}
    int depth = 0;                // Nesting depth

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

    // Stack of environments for scoping
    std::vector<std::unique_ptr<type_environment>> env_stack;

    // Root environment (kept separate for final result extraction)
    type_environment* root_env;

    type_inference_visitor(constraint_set& cs, type_environment* root)
        : constraints(cs), env(root), ctx(), root_env(root) {}

    // === Type guard extraction ===

    /**
     * Extract a type guard from a test expression like "x is string"
     * Returns an invalid guard if no narrowing can be inferred
     */
    type_guard extract_type_guard(statement& test_node, bool negated = false) {
        // Handle "not (x is y)" - extract inner test and flip negation
        if (auto* unary = dynamic_cast<unary_expression*>(&test_node)) {
            if (unary->op.value == "not") {
                return extract_type_guard(*unary->argument, !negated);
            }
        }

        // Handle "x is y" test expressions
        if (auto* test = dynamic_cast<test_expression*>(&test_node)) {
            // Get the variable being tested
            std::string var_name;
            if (auto* id = dynamic_cast<identifier*>(test->operand.get())) {
                var_name = id->val;
            } else {
                return type_guard{}; // Can only narrow simple identifiers for now
            }

            // Get the test name
            std::string test_name;
            if (auto* test_id = dynamic_cast<identifier*>(test->test.get())) {
                test_name = test_id->val;
            } else if (auto* call = dynamic_cast<call_expression*>(test->test.get())) {
                if (auto* callee_id = dynamic_cast<identifier*>(call->callee.get())) {
                    test_name = callee_id->val;
                }
            }

            // Handle "is not" by flipping negation
            if (test->negate) {
                negated = !negated;
            }

            // Map test names to types
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
                // "is defined" means variable exists and is non-null
                // For negated (is not defined), we can't narrow
                if (!negated) {
                    // Variable is defined - mark as non-null by giving it its current type
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
                return type_guard(var_name, narrowed_type, negated);
            }
        }

        // Handle truthiness check: "if x" or "if x.field"
        // This implies x (or x.field) is truthy/non-null
        if (auto* id = dynamic_cast<identifier*>(&test_node)) {
            // "if x" - x is truthy, meaning it's defined and non-null
            // We don't narrow the type, but we know it's not null
            // Return guard with current type to indicate "is defined"
            auto current = env->lookup(id->val);
            if (current && !negated) {
                return type_guard(id->val, current, false);
            }
        }

        return type_guard{}; // No narrowing possible
    }

    /**
     * Apply a type guard to the current environment
     * Creates a new binding with the narrowed type
     */
    void apply_type_guard(const type_guard& guard) {
        if (!guard.valid()) return;

        if (guard.negated) {
            // For negated guards like "x is not string", we can't easily narrow
            // In the future, we could create exclusion types
            return;
        }

        // Bind the variable to its narrowed type in current scope
        env->bind(guard.variable, guard.narrowed_type);

        // Also add an equality constraint to help the solver
        auto existing = root_env->lookup(guard.variable);
        if (existing) {
            constraints.add_equality(existing, guard.narrowed_type,
                "type guard narrowing " + guard.variable);
        }
    }

    // Scope management
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

    // Fresh type variable
    TypePtr fresh() {
        return make_typevar(type_var_generator::fresh());
    }

    // Get source location string for debugging
    std::string source_loc(const statement& node) {
        return "pos:" + std::to_string(node.pos);
    }

    // === Statement visitors ===

    TypePtr visit(program& node) override {
        for (auto& stmt : node.body) {
            visit_body_statement(*stmt);
        }
        return make_null();
    }

    TypePtr visit(if_statement& node) override {
        // Visit test expression first (generates constraints)
        auto test_type = dispatch(*node.test);

        // Extract type guard from the test
        auto guard = extract_type_guard(*node.test);

        auto old_ctx = ctx;
        ctx = ctx.enter_conditional();

        // Track bindings before visiting branches
        // We need to merge bindings that happen in either branch

        // Visit body with type guard applied (true branch)
        push_scope();
        if (guard.valid() && !guard.negated) {
            apply_type_guard(guard);
        }
        for (auto& stmt : node.body) {
            visit_body_statement(*stmt);
        }
        // Capture true branch bindings before popping
        auto true_branch_bindings = env->local_bindings();
        pop_scope();

        // Visit alternate with inverted guard (false branch)
        push_scope();
        if (guard.valid()) {
            // In the else branch, the guard is negated
            type_guard inverted_guard(guard.variable, guard.narrowed_type, !guard.negated);
            // For now, we don't apply negated guards (would need exclusion types)
            // But we still create the scope for proper scoping
        }
        for (auto& stmt : node.alternate) {
            visit_body_statement(*stmt);
        }
        // Capture false branch bindings before popping
        auto false_branch_bindings = env->local_bindings();
        pop_scope();

        // Merge branch bindings into the parent scope
        // For variables set in either branch, create union types
        merge_branch_bindings(true_branch_bindings, false_branch_bindings);

        ctx = old_ctx;
        return make_null();
    }

    /**
     * Merge bindings from if/else branches into the parent scope
     * Creates union types for variables that could have different types
     */
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

            // Get original type from parent scope (if exists)
            TypePtr original_type = env->lookup(var_name);

            TypePtr merged_type;

            if (true_type && false_type) {
                // Set in both branches - union of both
                if (true_type == false_type || (true_type && false_type && true_type->equals(false_type))) {
                    merged_type = true_type;
                } else {
                    auto u = std::make_shared<union_type>();
                    u->add_alternative(true_type);
                    u->add_alternative(false_type);
                    merged_type = u;
                }
            } else if (true_type) {
                // Set only in true branch - union with original (or just true_type if no original)
                if (original_type && original_type != true_type && !original_type->equals(true_type)) {
                    auto u = std::make_shared<union_type>();
                    u->add_alternative(true_type);
                    u->add_alternative(original_type);
                    merged_type = u;
                } else {
                    merged_type = true_type;
                }
            } else if (false_type) {
                // Set only in false branch - union with original
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
        auto loop_obj = std::make_shared<object_type>(false);  // not extensible
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

        // Visit body - expressions in loop body are output expressions
        for (auto& stmt : node.body) {
            visit_body_statement(*stmt);
        }

        ctx = old_ctx;
        pop_scope();

        // Visit default block (outside loop scope)
        for (auto& stmt : node.default_block) {
            visit_body_statement(*stmt);
        }

        return make_null();
    }

    /**
     * Helper to visit a statement that may contain output expressions
     * Adds output coercion constraints for expression statements
     */
    void visit_body_statement(statement& stmt) {
        // Check if this is an expression statement (output context)
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
            auto ft = std::make_shared<function_type>(
                std::move(param_types), make_string());
            env->bind(name_id->val, ft);
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
        auto obj = std::make_shared<object_type>(false);
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
        // Also infer operand types from literal comparisons
        if (op == "==" || op == "!=" || op == "<" || op == ">" ||
            op == "<=" || op == ">=" || op == "in" || op == "not in") {

            // For 'in'/'not in', the container can be string, array, or object
            // so we cannot infer container type from element type
            // Example: "abc" in x could mean x is string ("abcdef") or object ({"abc": 1})
            bool is_containment_check = (op == "in" || op == "not in");

            // If comparing with a string literal, the other operand is likely a string
            // (only for equality/ordering comparisons, not containment checks)
            if (!is_containment_check) {
                if (is_stmt<string_literal>(node.left)) {
                    constraints.add_equality(right_type, make_string(),
                        "comparison with string literal at " + source_loc(node));
                } else if (is_stmt<string_literal>(node.right)) {
                    constraints.add_equality(left_type, make_string(),
                        "comparison with string literal at " + source_loc(node));
                }

                // If comparing with an integer literal, the other operand is likely int
                if (is_stmt<integer_literal>(node.left)) {
                    constraints.add_equality(right_type, make_int(),
                        "comparison with int literal at " + source_loc(node));
                } else if (is_stmt<integer_literal>(node.right)) {
                    constraints.add_equality(left_type, make_int(),
                        "comparison with int literal at " + source_loc(node));
                }
            }

            return make_bool();
        }

        // Logical operators
        if (op == "and" || op == "or") {
            // Short-circuit: result type depends on operands
            auto u = std::make_shared<union_type>();
            u->add_alternative(left_type);
            u->add_alternative(right_type);
            return u;
        }

        // String concatenation - operands are coerced to strings
        if (op == "~") {
            constraints.add_string_operand(left_type,
                "string concat lhs at " + source_loc(node));
            constraints.add_string_operand(right_type,
                "string concat rhs at " + source_loc(node));
            return make_string();
        }

        // The + operator can be string concat, array concat, or numeric addition
        if (op == "+") {
            // If either operand is a string literal, it's string concatenation
            if (is_stmt<string_literal>(node.left)) {
                constraints.add_equality(right_type, make_string(),
                    "string concat (+ with string literal) at " + source_loc(node));
                return make_string();
            } else if (is_stmt<string_literal>(node.right)) {
                constraints.add_equality(left_type, make_string(),
                    "string concat (+ with string literal) at " + source_loc(node));
                return make_string();
            }
            // Otherwise could be numeric or array concat - return fresh type variable
            return fresh();
        }

        // Arithmetic operators: -, *, /, %, // are numeric only
        if (op == "-" || op == "*" || op == "/" || op == "%" || op == "//") {
            // These are numeric operations
            return fresh();  // Could be int or float
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

            // Array-preserving filters (maintain element type)
            if (name == "list" || name == "sort" || name == "reverse" || name == "unique") {
                auto elem = fresh();
                constraints.add_iterable(input_type, elem,
                    "filter " + name + " at " + source_loc(node));
                return make_array(elem);
            }

            // Slice without args returns array with same element type
            if (name == "slice") {
                auto elem = fresh();
                constraints.add_iterable(input_type, elem,
                    "filter slice at " + source_loc(node));
                return make_array(elem);
            }

            // Batch returns array of arrays
            if (name == "batch") {
                auto elem = fresh();
                constraints.add_iterable(input_type, elem,
                    "filter batch at " + source_loc(node));
                return make_array(make_array(elem));
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

            // Join returns string (input must be iterable)
            if (name == "join") {
                auto elem = fresh();
                constraints.add_iterable(input_type, elem,
                    "filter join at " + source_loc(node));
                return make_string();
            }

            // Filtering filters (preserve element type)
            if (name == "select" || name == "reject") {
                auto elem = fresh();
                constraints.add_iterable(input_type, elem,
                    "filter " + name + " at " + source_loc(node));
                return make_array(elem);
            }

            // Attribute filtering filters (preserve element type)
            if (name == "selectattr" || name == "rejectattr") {
                auto elem = fresh();
                constraints.add_iterable(input_type, elem,
                    "filter " + name + " at " + source_loc(node));
                return make_array(elem);
            }

            // Map without args - preserve element type
            if (name == "map") {
                auto elem = fresh();
                constraints.add_iterable(input_type, elem,
                    "filter map at " + source_loc(node));
                return make_array(elem);
            }
        }

        // Filter with call expression (filter has arguments)
        if (is_stmt<call_expression>(node.filter)) {
            auto* call = cast_stmt<call_expression>(node.filter);
            if (is_stmt<identifier>(call->callee)) {
                auto* filter_id = cast_stmt<identifier>(call->callee);
                const auto& name = filter_id->val;

                // Helper to find keyword argument value
                auto find_kwarg = [&](const std::string& key) -> statement* {
                    for (auto& arg : call->args) {
                        if (auto* kw = cast_stmt<keyword_argument_expression>(arg)) {
                            if (auto* kid = cast_stmt<identifier>(kw->key)) {
                                if (kid->val == key) {
                                    return kw->val.get();
                                }
                            }
                        }
                    }
                    return nullptr;
                };

                // Visit arguments
                for (auto& arg : call->args) {
                    dispatch(*arg);
                }

                // String-returning filters (most don't need input constraints)
                if (name == "tojson" || name == "trim" || name == "indent" ||
                    name == "replace") {
                    return make_string();
                }

                // Join returns string and requires iterable input
                if (name == "join") {
                    auto elem = fresh();
                    constraints.add_iterable(input_type, elem,
                        "filter join at " + source_loc(node));
                    return make_string();
                }

                // Default filter - returns input type
                if (name == "default" || name == "d") {
                    return input_type;
                }

                // Slice with args - preserve element type
                if (name == "slice") {
                    auto elem = fresh();
                    constraints.add_iterable(input_type, elem,
                        "filter slice at " + source_loc(node));
                    return make_array(elem);
                }

                // Batch - returns array of arrays
                if (name == "batch") {
                    auto elem = fresh();
                    constraints.add_iterable(input_type, elem,
                        "filter batch at " + source_loc(node));
                    return make_array(make_array(elem));
                }

                // Map with attribute - extract attribute type from elements
                if (name == "map") {
                    auto elem = fresh();
                    constraints.add_iterable(input_type, elem,
                        "filter map at " + source_loc(node));

                    // Check for attribute='...' argument
                    if (auto* attr_arg = find_kwarg("attribute")) {
                        if (auto* attr_str = dynamic_cast<string_literal*>(attr_arg)) {
                            // map(attribute='name') extracts .name from each element
                            auto attr_type = fresh();
                            constraints.add_field(elem, attr_str->val, attr_type, false,
                                "map attribute '" + attr_str->val + "' at " + source_loc(node));
                            return make_array(attr_type);
                        }
                    }
                    // map without attribute - preserve element type
                    return make_array(elem);
                }

                // selectattr/rejectattr with attribute - preserve element type
                if (name == "selectattr" || name == "rejectattr") {
                    auto elem = fresh();
                    constraints.add_iterable(input_type, elem,
                        "filter " + name + " at " + source_loc(node));

                    // First positional arg is the attribute name
                    if (!call->args.empty()) {
                        if (auto* attr_str = cast_stmt<string_literal>(call->args[0])) {
                            // Add constraint that elements have this attribute
                            auto attr_type = fresh();
                            constraints.add_field(elem, attr_str->val, attr_type, false,
                                name + " attribute '" + attr_str->val + "' at " + source_loc(node));
                        }
                    }
                    return make_array(elem);
                }

                // select/reject with test - preserve element type
                if (name == "select" || name == "reject") {
                    auto elem = fresh();
                    constraints.add_iterable(input_type, elem,
                        "filter " + name + " at " + source_loc(node));
                    return make_array(elem);
                }

                // sort/reverse/unique/list - preserve element type
                if (name == "sort" || name == "reverse" || name == "unique" || name == "list") {
                    auto elem = fresh();
                    constraints.add_iterable(input_type, elem,
                        "filter " + name + " at " + source_loc(node));
                    return make_array(elem);
                }

                // first/last/random - return element type
                if (name == "first" || name == "last" || name == "random") {
                    auto elem = fresh();
                    constraints.add_array_element(input_type, elem,
                        "filter " + name + " at " + source_loc(node));
                    return elem;
                }
            }
        }

        // Unknown filter - return fresh type
        return fresh();
    }

    TypePtr visit(select_expression& node) override {
        auto input_type = dispatch(*node.lhs);
        dispatch(*node.test);
        // Select expression filters elements, preserving element type
        auto elem = fresh();
        constraints.add_iterable(input_type, elem,
            "select expression at " + source_loc(node));
        return make_array(elem);
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
        auto u = std::make_shared<union_type>();
        u->add_alternative(true_type);
        u->add_alternative(false_type);
        return u;
    }
};

} // namespace types
} // namespace jinja
