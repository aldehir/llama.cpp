#pragma once

#include "jinja-type-system.h"
#include <variant>
#include <vector>
#include <string>

namespace jinja {
namespace types {

/**
 * Type constraints generated during AST traversal
 */

// Equality constraint: T1 = T2
struct equality_constraint {
    TypePtr left;
    TypePtr right;
    std::string source;  // Debug info: where this constraint came from

    std::string to_string() const {
        return (left ? left->to_string() : "null") + " = " +
               (right ? right->to_string() : "null") +
               " (from: " + source + ")";
    }
};

// Field access constraint: T1 has field 'name' of type T2
struct has_field_constraint {
    TypePtr object_type;
    std::string field_name;
    TypePtr field_type;
    bool optional;  // true if accessed in conditional context
    std::string source;

    std::string to_string() const {
        return (object_type ? object_type->to_string() : "null") + "." + field_name +
               (optional ? "?" : "") + " : " +
               (field_type ? field_type->to_string() : "null") +
               " (from: " + source + ")";
    }
};

// Array element constraint: T1 is array with element type T2
struct array_element_constraint {
    TypePtr array_type;
    TypePtr element_type;
    std::string source;

    std::string to_string() const {
        return (array_type ? array_type->to_string() : "null") + "[*] : " +
               (element_type ? element_type->to_string() : "null") +
               " (from: " + source + ")";
    }
};

// Iterable constraint: T can be iterated (array or object)
struct iterable_constraint {
    TypePtr type;
    TypePtr element_type;  // Type of iteration variable
    std::string source;

    std::string to_string() const {
        return (type ? type->to_string() : "null") + " is iterable<" +
               (element_type ? element_type->to_string() : "null") + ">" +
               " (from: " + source + ")";
    }
};

// Callable constraint: T is callable with return type R
struct callable_constraint {
    TypePtr callee_type;
    std::vector<TypePtr> arg_types;
    TypePtr return_type;
    std::string source;

    std::string to_string() const {
        std::string args;
        for (size_t i = 0; i < arg_types.size(); ++i) {
            if (i > 0) args += ", ";
            args += arg_types[i] ? arg_types[i]->to_string() : "null";
        }
        return (callee_type ? callee_type->to_string() : "null") + "(" + args + ") -> " +
               (return_type ? return_type->to_string() : "null") +
               " (from: " + source + ")";
    }
};

// Output coercion constraint: T is used in output context (will be coerced to string)
// This is a "soft" constraint - it suggests the type should be string-like
// but doesn't force it (since any type can be stringified in Jinja)
struct output_coercion_constraint {
    TypePtr type;
    std::string source;

    std::string to_string() const {
        return (type ? type->to_string() : "null") + " -> output" +
               " (from: " + source + ")";
    }
};

// String operand constraint: T is used in string context (e.g., ~ operator)
// This is stronger than output coercion - the value is definitely a string
struct string_operand_constraint {
    TypePtr type;
    std::string source;

    std::string to_string() const {
        return (type ? type->to_string() : "null") + " : string" +
               " (from: " + source + ")";
    }
};

// Literal constraint: T is compared to a literal value
// Used to infer literal types like literal<"assistant", "user", ...>
struct literal_constraint {
    TypePtr type;           // The type being constrained
    TypePtr literal_value;  // A literal_type with the specific value
    std::string source;

    std::string to_string() const {
        return (type ? type->to_string() : "null") + " == " +
               (literal_value ? literal_value->to_string() : "null") +
               " (from: " + source + ")";
    }
};

// Union of all constraint types
using constraint = std::variant<
    equality_constraint,
    has_field_constraint,
    array_element_constraint,
    iterable_constraint,
    callable_constraint,
    output_coercion_constraint,
    string_operand_constraint,
    literal_constraint
>;

/**
 * Constraint set - collects all constraints from template analysis
 */
struct constraint_set {
    std::vector<constraint> constraints;

    void add(constraint c) {
        constraints.push_back(std::move(c));
    }

    void add_equality(TypePtr left, TypePtr right, const std::string& source) {
        constraints.push_back(equality_constraint{std::move(left), std::move(right), source});
    }

    void add_field(TypePtr obj, const std::string& field, TypePtr type,
                   bool optional, const std::string& source) {
        constraints.push_back(has_field_constraint{
            std::move(obj), field, std::move(type), optional, source
        });
    }

    void add_array_element(TypePtr arr, TypePtr elem, const std::string& source) {
        constraints.push_back(array_element_constraint{std::move(arr), std::move(elem), source});
    }

    void add_iterable(TypePtr type, TypePtr elem, const std::string& source) {
        constraints.push_back(iterable_constraint{std::move(type), std::move(elem), source});
    }

    void add_callable(TypePtr callee, std::vector<TypePtr> args, TypePtr ret,
                      const std::string& source) {
        constraints.push_back(callable_constraint{
            std::move(callee), std::move(args), std::move(ret), source
        });
    }

    void add_output_coercion(TypePtr type, const std::string& source) {
        constraints.push_back(output_coercion_constraint{std::move(type), source});
    }

    void add_string_operand(TypePtr type, const std::string& source) {
        constraints.push_back(string_operand_constraint{std::move(type), source});
    }

    void add_literal(TypePtr type, TypePtr literal_value, const std::string& source) {
        constraints.push_back(literal_constraint{std::move(type), std::move(literal_value), source});
    }

    void clear() {
        constraints.clear();
    }

    size_t size() const {
        return constraints.size();
    }

    std::string to_string() const {
        std::string result = "Constraints (" + std::to_string(constraints.size()) + "):\n";
        for (const auto& c : constraints) {
            result += "  ";
            std::visit([&](const auto& cst) {
                result += cst.to_string();
            }, c);
            result += "\n";
        }
        return result;
    }
};

} // namespace types
} // namespace jinja
