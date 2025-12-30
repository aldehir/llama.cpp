#pragma once

#include "jinja-type-system.h"
#include "jinja-constraints.h"
#include <map>
#include <vector>
#include <set>
#include <stdexcept>

namespace jinja {
namespace types {

/**
 * Constraint solver using unification
 */
struct constraint_solver {
    // Substitution: maps type variable names to their resolved types
    std::map<std::string, TypePtr> subst;

    // Errors encountered during solving
    std::vector<std::string> errors;

    // Type variables that were used in output context (soft constraint to string)
    std::set<std::string> output_coerced_types;

    // Type variables that were used in string operand context (hard constraint to string)
    std::set<std::string> string_operand_types;

    // Literal values collected for type variables (for building union literal types)
    // Maps type variable name -> set of literal values
    std::map<std::string, std::vector<TypePtr>> literal_values;

    // Type alternatives collected from type guards (for building union types)
    // Maps type variable name -> set of alternative types
    std::map<std::string, std::vector<TypePtr>> type_alternatives;

    // Maximum number of literals before collapsing to base type (e.g., string)
    static constexpr size_t MAX_LITERALS = 10;

    /**
     * Solve all constraints
     * Returns true if successful, false if errors occurred
     */
    bool solve(const constraint_set& constraints) {
        errors.clear();
        output_coerced_types.clear();
        string_operand_types.clear();
        literal_values.clear();
        type_alternatives.clear();

        for (const auto& constraint : constraints.constraints) {
            std::visit([this](const auto& c) { solve_constraint(c); }, constraint);
        }

        // Finalization pass: resolve remaining type variables
        finalize_types();

        return errors.empty();
    }

    /**
     * Finalization pass - resolve unbound type variables based on usage context
     */
    void finalize_types() {
        // First, handle type alternatives (from type guards like "x is string")
        // If a type variable has alternatives AND is bound to something else, create a union
        for (const auto& [var_name, alternatives] : type_alternatives) {
            if (alternatives.empty()) continue;

            // Find what this type variable is currently bound to
            auto it = subst.find(var_name);
            if (it != subst.end()) {
                // Already bound to something - create union with alternatives
                TypePtr current_type = it->second;

                // Skip if already a union that includes the alternative
                if (auto* existing_union = as_type<union_type>(current_type)) {
                    for (const auto& alt : alternatives) {
                        existing_union->add_alternative(alt);
                    }
                } else {
                    // Create a new union with current type and all alternatives
                    auto u = std::make_shared<union_type>();
                    u->add_alternative(current_type);
                    for (const auto& alt : alternatives) {
                        u->add_alternative(alt);
                    }
                    subst[var_name] = u;
                }
            } else {
                // Unbound - if only one alternative, use it; otherwise create union
                if (alternatives.size() == 1) {
                    subst[var_name] = alternatives[0];
                } else {
                    auto u = std::make_shared<union_type>();
                    for (const auto& alt : alternatives) {
                        u->add_alternative(alt);
                    }
                    subst[var_name] = u;
                }
            }
        }

        // Then, aggregate literals from all equivalent type variables
        // Type variables get unified during constraint solving, so literals may be
        // spread across multiple variable names that all resolve to the same root
        std::map<std::string, std::vector<TypePtr>> aggregated_literals;

        for (const auto& [var_name, literals] : literal_values) {
            // Find the root type variable for this var_name by following the chain
            std::string root_var = var_name;
            auto it = subst.find(var_name);
            while (it != subst.end()) {
                if (auto* tv = as_type<type_variable>(it->second)) {
                    root_var = tv->name;
                    it = subst.find(tv->name);
                } else {
                    // Bound to a concrete type, not a type variable
                    root_var.clear();  // Signal that this is resolved
                    break;
                }
            }

            if (root_var.empty()) {
                continue;  // Already resolved to concrete type
            }

            // Aggregate literals under the root variable
            auto& root_literals = aggregated_literals[root_var];
            for (const auto& lit : literals) {
                bool found = false;
                for (const auto& existing : root_literals) {
                    if (existing->equals(lit)) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    root_literals.push_back(lit);
                }
            }
        }

        // Now resolve using aggregated literals
        for (const auto& [var_name, literals] : aggregated_literals) {
            if (subst.find(var_name) == subst.end() && !literals.empty()) {
                if (literals.size() == 1) {
                    // Single literal - use it directly
                    subst[var_name] = literals[0];
                } else if (literals.size() <= MAX_LITERALS) {
                    // Multiple literals - create union of literals
                    auto u = std::make_shared<union_type>();
                    for (const auto& lit : literals) {
                        u->add_alternative(lit);
                    }
                    subst[var_name] = u;
                } else {
                    // Too many literals - collapse to base type
                    if (auto* lit = as_type<literal_type>(literals[0])) {
                        subst[var_name] = lit->base_type();
                    }
                }
            }
        }

        // Then, resolve type variables used as string operands
        for (const auto& var_name : string_operand_types) {
            if (subst.find(var_name) == subst.end()) {
                subst[var_name] = make_string();
            }
        }

        // Finally, resolve type variables used in output context to string
        // (only if still unbound after other constraints)
        for (const auto& var_name : output_coerced_types) {
            if (subst.find(var_name) == subst.end()) {
                subst[var_name] = make_string();
            }
        }
    }

    /**
     * Apply substitution to resolve all type variables in a type
     */
    TypePtr apply(const TypePtr& type) const {
        std::set<const Type*> visited;
        return apply_impl(type, visited);
    }

    TypePtr apply_impl(const TypePtr& type, std::set<const Type*>& visited) const {
        if (!type) return nullptr;

        // Cycle detection - if we've seen this exact type pointer, return it as-is
        if (visited.count(type.get()) > 0) {
            return type;
        }
        visited.insert(type.get());

        if (auto* tv = as_type<type_variable>(type)) {
            auto it = subst.find(tv->name);
            if (it != subst.end()) {
                return apply_impl(it->second, visited);
            }
            return type;
        }

        if (auto* arr = as_type<array_type>(type)) {
            auto result = std::make_shared<array_type>();
            if (arr->element_type) {
                result->element_type = apply_impl(arr->element_type, visited);
            }
            return result;
        }

        if (auto* obj = as_type<object_type>(type)) {
            auto result = std::make_shared<object_type>(obj->extensible);
            for (const auto& [name, field] : obj->fields) {
                result->add_field(name, apply_impl(field.type, visited), field.optional);
            }
            return result;
        }

        if (auto* u = as_type<union_type>(type)) {
            auto result = std::make_shared<union_type>();
            for (const auto& alt : u->alternatives) {
                result->add_alternative(apply_impl(alt, visited));
            }
            return result;
        }

        if (auto* func = as_type<function_type>(type)) {
            std::vector<TypePtr> params;
            for (const auto& p : func->param_types) {
                params.push_back(apply_impl(p, visited));
            }
            return std::make_shared<function_type>(
                std::move(params), apply_impl(func->return_type, visited));
        }

        return type;  // Primitives are unchanged
    }

    /**
     * Get fully resolved types from a type environment
     */
    std::map<std::string, TypePtr> get_resolved_types(
        const std::map<std::string, TypePtr>& env) const
    {
        std::map<std::string, TypePtr> result;
        for (const auto& [name, type] : env) {
            result[name] = apply(type);
        }
        return result;
    }

private:
    /**
     * Follow substitution chain to get the actual stored type (no copying)
     * Returns the final type and the last type variable name in the chain
     */
    std::pair<TypePtr, std::string> follow_chain(const TypePtr& type) {
        TypePtr current = type;
        std::string last_var;
        while (auto* tv = as_type<type_variable>(current)) {
            last_var = tv->name;
            auto it = subst.find(tv->name);
            if (it != subst.end()) {
                current = it->second;
            } else {
                break;  // Unbound type variable
            }
        }
        return {current, last_var};
    }

    // === Constraint solving methods ===

    void solve_constraint(const equality_constraint& c) {
        if (!unify(c.left, c.right)) {
            errors.push_back("Type mismatch: " +
                (c.left ? c.left->to_string() : "null") + " != " +
                (c.right ? c.right->to_string() : "null") +
                " at " + c.source);
        }
    }

    void solve_constraint(const has_field_constraint& c) {
        auto [current, last_var] = follow_chain(c.object_type);

        if (auto* tv = as_type<type_variable>(current)) {
            // Object type is still unknown - create object type with field
            auto new_obj = std::make_shared<object_type>(true);
            new_obj->add_field(c.field_name, c.field_type, c.optional);
            subst[tv->name] = new_obj;
        } else if (auto* existing_obj = as_type<object_type>(current)) {
            // Modify the existing object directly (not a copy!)
            add_field_to_object(existing_obj, c.field_name, c.field_type, c.optional, c.source);
        } else if (auto* ut = as_type<union_type>(current)) {
            // Union type - add field constraint to ALL alternatives
            for (auto& alt : ut->alternatives) {
                has_field_constraint alt_constraint{alt, c.field_name, c.field_type, c.optional, c.source};
                solve_constraint(alt_constraint);
            }
        } else {
            // Non-object type with field access - create an object type
            auto fallback_obj = std::make_shared<object_type>(true);
            fallback_obj->add_field(c.field_name, c.field_type, c.optional);

            // Update the last type variable to point to this object
            if (!last_var.empty()) {
                subst[last_var] = fallback_obj;
            }
        }
    }

    /**
     * Helper to add a field to an object type, handling unification
     */
    void add_field_to_object(object_type* obj, const std::string& field_name,
                             TypePtr field_type, bool optional, const std::string& source) {
        auto* existing = obj->get_field(field_name);
        if (existing) {
            // Unify field types
            if (!unify(existing->type, field_type)) {
                errors.push_back("Field type mismatch for '" + field_name +
                    "' at " + source);
            }
            // If accessed unconditionally anywhere, mark as required
            if (!optional) {
                existing->optional = false;
            }
        } else {
            // Add new field to the original object
            obj->add_field(field_name, field_type, optional);
        }
    }

    void solve_constraint(const array_element_constraint& c) {
        auto [current, last_var] = follow_chain(c.array_type);

        if (auto* tv = as_type<type_variable>(current)) {
            // Unbound type variable - create array type
            subst[tv->name] = make_array(c.element_type);
        } else if (auto* arr = as_type<array_type>(current)) {
            // Unify element types directly on the original array
            if (arr->element_type) {
                unify(arr->element_type, c.element_type);
            } else {
                arr->element_type = c.element_type;
            }
        } else if (auto* ut = as_type<union_type>(current)) {
            // Union type - add constraint to ALL alternatives
            for (auto& alt : ut->alternatives) {
                array_element_constraint alt_constraint{alt, c.element_type, c.source};
                solve_constraint(alt_constraint);
            }
        }
        // Object used as array is fine - element_type represents value type
    }

    void solve_constraint(const iterable_constraint& c) {
        auto [current, last_var] = follow_chain(c.type);

        if (auto* tv = as_type<type_variable>(current)) {
            // Unbound type variable - create array type
            subst[tv->name] = make_array(c.element_type);
        } else if (auto* arr = as_type<array_type>(current)) {
            // Unify element types directly on the original array
            if (arr->element_type) {
                unify(arr->element_type, c.element_type);
            } else {
                arr->element_type = c.element_type;
            }
        } else if (auto* ut = as_type<union_type>(current)) {
            // Union type - add iterable constraint to ALL alternatives
            for (auto& alt : ut->alternatives) {
                iterable_constraint alt_constraint{alt, c.element_type, c.source};
                solve_constraint(alt_constraint);
            }
        }
        // Object iteration yields values - element_type could be anything
    }

    void solve_constraint(const callable_constraint& c) {
        auto resolved = apply(c.callee_type);

        if (auto* func = as_type<function_type>(resolved)) {
            // Unify return type
            unify(func->return_type, c.return_type);
            // Unify parameter types
            size_t min_params = std::min(func->param_types.size(), c.arg_types.size());
            for (size_t i = 0; i < min_params; ++i) {
                unify(func->param_types[i], c.arg_types[i]);
            }
        } else if (auto* tv = as_type<type_variable>(resolved)) {
            // Create function type
            subst[tv->name] = make_function(c.arg_types, c.return_type);
        }
        // Built-in functions/filters are handled by returning fresh types
    }

    void solve_constraint(const output_coercion_constraint& c) {
        // Output coercion is a soft constraint - it marks types that should be
        // string-coercible. We track these for the finalization pass.
        auto resolved = apply(c.type);

        if (auto* tv = as_type<type_variable>(resolved)) {
            // Mark this type variable as used in output context
            // During finalization, unbound type vars with output coercion
            // will default to string
            output_coerced_types.insert(tv->name);
        }
    }

    void solve_constraint(const string_operand_constraint& c) {
        // String operand is a strong constraint - the value is used as a string
        auto resolved = apply(c.type);

        if (auto* tv = as_type<type_variable>(resolved)) {
            // If type variable is unbound, bind to string
            if (subst.find(tv->name) == subst.end()) {
                subst[tv->name] = make_string();
            }
        }
        // If already bound to something else, that's fine - Jinja will coerce
    }

    void solve_constraint(const literal_constraint& c) {
        // Literal constraint: type is compared to a literal value
        // Collect literal values per type variable for later resolution
        auto resolved = apply(c.type);

        if (auto* tv = as_type<type_variable>(resolved)) {
            // Collect this literal value for the type variable
            auto& literals = literal_values[tv->name];

            // Only add if not already present (avoid duplicates)
            bool found = false;
            for (const auto& existing : literals) {
                if (existing->equals(c.literal_value)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                literals.push_back(c.literal_value);
            }
        } else if (auto* obj = as_type<object_type>(resolved)) {
            // Type is already resolved to an object - this means we have a nested field
            // that's being compared. We should trace back to find the field type variable.
            // For now, we just ignore - the field type will be handled separately.
        }
        // If already bound to something concrete, check compatibility
        // (but be lenient - Jinja allows comparing any types)
    }

    void solve_constraint(const type_alternative_constraint& c) {
        // Type alternative constraint: type could be the alternative
        // This is used for type guards like "x is string" where x could be string or other types
        auto resolved = apply(c.type);

        if (auto* tv = as_type<type_variable>(resolved)) {
            // Collect this alternative for the type variable
            auto& alternatives = type_alternatives[tv->name];

            // Only add if not already present (avoid duplicates)
            bool found = false;
            for (const auto& existing : alternatives) {
                if (existing->equals(c.alternative)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                alternatives.push_back(c.alternative);
            }
        }
        // If already bound to something concrete, the alternative is already captured
    }

    // === Unification algorithm ===

    /**
     * Unify two types - make them equivalent
     * Returns true if successful
     * Uses follow_chain to avoid copying structured types
     */
    bool unify(TypePtr t1, TypePtr t2) {
        if (!t1 || !t2) return true;  // Null types unify with anything

        // Follow chains without copying to get the actual stored types
        auto [resolved1, var1] = follow_chain(t1);
        auto [resolved2, var2] = follow_chain(t2);

        // Same type (pointer equality for shared objects)
        if (resolved1 == resolved2) return true;
        if (resolved1->equals(resolved2)) return true;

        // Type variable on left (still unbound)
        if (auto* tv1 = as_type<type_variable>(resolved1)) {
            return unify_var(tv1->name, resolved2);
        }

        // Type variable on right (still unbound)
        if (auto* tv2 = as_type<type_variable>(resolved2)) {
            return unify_var(tv2->name, resolved1);
        }

        // Both arrays - unify element types
        if (auto* arr1 = as_type<array_type>(resolved1)) {
            if (auto* arr2 = as_type<array_type>(resolved2)) {
                if (arr1->element_type && arr2->element_type) {
                    return unify(arr1->element_type, arr2->element_type);
                }
                // If either is null, unification succeeds
                if (arr1->element_type) {
                    arr2->element_type = arr1->element_type;
                } else if (arr2->element_type) {
                    arr1->element_type = arr2->element_type;
                }
                return true;
            }
        }

        // Both objects - merge fields into the first one
        if (auto* obj1 = as_type<object_type>(resolved1)) {
            if (auto* obj2 = as_type<object_type>(resolved2)) {
                // Merge obj2's fields into obj1
                if (!unify_objects(obj1, obj2)) return false;
                // Point var2 to obj1 so all references see the merged object
                if (!var2.empty()) {
                    subst[var2] = resolved1;
                }
                return true;
            }
        }

        // Both unions - unify alternatives
        if (auto* u1 = as_type<union_type>(resolved1)) {
            if (auto* u2 = as_type<union_type>(resolved2)) {
                // Merge alternatives
                for (const auto& alt : u2->alternatives) {
                    u1->add_alternative(alt);
                }
                return true;
            }
        }

        // Union on left, non-union on right: all alternatives must unify with right
        // e.g., (string | 'T99) = string means string='string' AND 'T99=string
        if (auto* u1 = as_type<union_type>(resolved1)) {
            bool all_ok = true;
            for (const auto& alt : u1->alternatives) {
                if (!unify(alt, resolved2)) {
                    all_ok = false;
                }
            }
            return all_ok;
        }

        // Union on right, non-union on left: same logic, reversed
        if (auto* u2 = as_type<union_type>(resolved2)) {
            bool all_ok = true;
            for (const auto& alt : u2->alternatives) {
                if (!unify(resolved1, alt)) {
                    all_ok = false;
                }
            }
            return all_ok;
        }

        // Both functions
        if (auto* f1 = as_type<function_type>(resolved1)) {
            if (auto* f2 = as_type<function_type>(resolved2)) {
                if (f1->param_types.size() != f2->param_types.size()) {
                    return false;
                }
                for (size_t i = 0; i < f1->param_types.size(); ++i) {
                    if (!unify(f1->param_types[i], f2->param_types[i])) {
                        return false;
                    }
                }
                return unify(f1->return_type, f2->return_type);
            }
        }

        // Primitives must match exactly
        if (auto* p1 = as_type<primitive_type>(resolved1)) {
            if (auto* p2 = as_type<primitive_type>(resolved2)) {
                return p1->prim_kind == p2->prim_kind;
            }
        }

        // Different type kinds - can't unify (but be lenient)
        return false;
    }

    bool unify_var(const std::string& var, TypePtr type) {
        // Occurs check - prevent infinite types
        if (occurs_in(var, type)) {
            // Don't report error, just skip to avoid infinite loops
            return true;
        }
        subst[var] = type;
        return true;
    }

    bool unify_objects(object_type* obj1, object_type* obj2) {
        // Merge fields from obj2 into obj1
        for (const auto& [name, field] : obj2->fields) {
            auto* existing = obj1->get_field(name);
            if (existing) {
                // Unify field types
                if (!unify(existing->type, field.type)) {
                    return false;
                }
                // Field is optional only if optional in both
                existing->optional = existing->optional && field.optional;
            } else {
                obj1->add_field(name, field.type, field.optional);
            }
        }
        return true;
    }

    /**
     * Occurs check - checks if type variable appears in type
     * Used to prevent infinite types like T = array<T>
     */
    bool occurs_in(const std::string& var, const TypePtr& type) {
        if (!type) return false;

        if (auto* tv = as_type<type_variable>(type)) {
            if (tv->name == var) return true;
            auto it = subst.find(tv->name);
            if (it != subst.end()) {
                return occurs_in(var, it->second);
            }
            return false;
        }

        if (auto* arr = as_type<array_type>(type)) {
            return arr->element_type && occurs_in(var, arr->element_type);
        }

        if (auto* obj = as_type<object_type>(type)) {
            for (const auto& [_, field] : obj->fields) {
                if (occurs_in(var, field.type)) return true;
            }
            return false;
        }

        if (auto* u = as_type<union_type>(type)) {
            for (const auto& alt : u->alternatives) {
                if (occurs_in(var, alt)) return true;
            }
            return false;
        }

        if (auto* func = as_type<function_type>(type)) {
            for (const auto& param : func->param_types) {
                if (occurs_in(var, param)) return true;
            }
            return occurs_in(var, func->return_type);
        }

        return false;  // Primitives don't contain type variables
    }
};

} // namespace types
} // namespace jinja
