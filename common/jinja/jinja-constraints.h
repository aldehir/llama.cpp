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
struct EqualityConstraint {
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
struct HasFieldConstraint {
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
struct ArrayElementConstraint {
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
struct IterableConstraint {
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
struct CallableConstraint {
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

// Union of all constraint types
using Constraint = std::variant<
    EqualityConstraint,
    HasFieldConstraint,
    ArrayElementConstraint,
    IterableConstraint,
    CallableConstraint
>;

/**
 * Constraint set - collects all constraints from template analysis
 */
struct ConstraintSet {
    std::vector<Constraint> constraints;

    void add(Constraint c) {
        constraints.push_back(std::move(c));
    }

    void add_equality(TypePtr left, TypePtr right, const std::string& source) {
        constraints.push_back(EqualityConstraint{std::move(left), std::move(right), source});
    }

    void add_field(TypePtr obj, const std::string& field, TypePtr type,
                   bool optional, const std::string& source) {
        constraints.push_back(HasFieldConstraint{
            std::move(obj), field, std::move(type), optional, source
        });
    }

    void add_array_element(TypePtr arr, TypePtr elem, const std::string& source) {
        constraints.push_back(ArrayElementConstraint{std::move(arr), std::move(elem), source});
    }

    void add_iterable(TypePtr type, TypePtr elem, const std::string& source) {
        constraints.push_back(IterableConstraint{std::move(type), std::move(elem), source});
    }

    void add_callable(TypePtr callee, std::vector<TypePtr> args, TypePtr ret,
                      const std::string& source) {
        constraints.push_back(CallableConstraint{
            std::move(callee), std::move(args), std::move(ret), source
        });
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
            std::visit([&](const auto& constraint) {
                result += constraint.to_string();
            }, c);
            result += "\n";
        }
        return result;
    }
};

} // namespace types
} // namespace jinja
