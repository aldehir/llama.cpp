#pragma once

#include "jinja-type-system.h"
#include "jinja-constraints.h"
#include <map>
#include <vector>
#include <set>
#include <stdexcept>

namespace jinja {
namespace types {

struct constraint_solver {
    std::map<std::string, TypePtr> subst;
    std::vector<std::string> errors;
    std::set<std::string> output_coerced_types;
    std::set<std::string> string_operand_types;
    std::map<std::string, std::vector<TypePtr>> literal_values;
    std::map<std::string, std::vector<TypePtr>> type_alternatives;
    static constexpr size_t MAX_LITERALS = 10;

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

    void finalize_types() {
        // Handle type alternatives from type guards
        for (const auto& [var_name, alternatives] : type_alternatives) {
            if (alternatives.empty()) continue;

            auto it = subst.find(var_name);
            if (it != subst.end()) {
                if (auto* existing_union = as_type<union_type>(it->second)) {
                    for (const auto& alt : alternatives) {
                        existing_union->add_alternative(alt);
                    }
                } else {
                    auto u = std::make_shared<union_type>();
                    u->add_alternative(it->second);
                    for (const auto& alt : alternatives) {
                        u->add_alternative(alt);
                    }
                    subst[var_name] = u;
                }
            } else {
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

        // Aggregate literals from equivalent type variables
        std::map<std::string, std::vector<TypePtr>> aggregated_literals;

        for (const auto& [var_name, literals] : literal_values) {
            std::string root_var = var_name;
            auto it = subst.find(var_name);
            while (it != subst.end()) {
                if (auto* tv = as_type<type_variable>(it->second)) {
                    root_var = tv->name;
                    it = subst.find(tv->name);
                } else {
                    root_var.clear();
                    break;
                }
            }

            if (root_var.empty()) continue;

            auto& root_literals = aggregated_literals[root_var];
            for (const auto& lit : literals) {
                bool found = false;
                for (const auto& existing : root_literals) {
                    if (existing->equals(lit)) { found = true; break; }
                }
                if (!found) root_literals.push_back(lit);
            }
        }

        for (const auto& [var_name, literals] : aggregated_literals) {
            if (subst.find(var_name) == subst.end() && !literals.empty()) {
                if (literals.size() == 1) {
                    subst[var_name] = literals[0];
                } else if (literals.size() <= MAX_LITERALS) {
                    auto u = std::make_shared<union_type>();
                    for (const auto& lit : literals) u->add_alternative(lit);
                    subst[var_name] = u;
                } else {
                    if (auto* lit = as_type<literal_type>(literals[0])) {
                        subst[var_name] = lit->base_type();
                    }
                }
            }
        }

        for (const auto& var_name : string_operand_types) {
            if (subst.find(var_name) == subst.end()) {
                subst[var_name] = make_string();
            }
        }

        for (const auto& var_name : output_coerced_types) {
            if (subst.find(var_name) == subst.end()) {
                subst[var_name] = make_string();
            }
        }
    }

    TypePtr apply(const TypePtr& type) const {
        std::set<const Type*> visited;
        return apply_impl(type, visited);
    }

    TypePtr apply_impl(const TypePtr& type, std::set<const Type*>& visited) const {
        if (!type) return nullptr;
        if (visited.count(type.get()) > 0) return type;
        visited.insert(type.get());

        if (auto* tv = as_type<type_variable>(type)) {
            auto it = subst.find(tv->name);
            if (it != subst.end()) return apply_impl(it->second, visited);
            // Leave unresolved type variables as-is for now
            // (unknown conversion happens later if needed)
            return type;
        }

        if (auto* arr = as_type<array_type>(type)) {
            auto result = std::make_shared<array_type>();
            if (arr->element_type) result->element_type = apply_impl(arr->element_type, visited);
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
            for (const auto& alt : u->alternatives) result->add_alternative(apply_impl(alt, visited));
            return result;
        }

        if (auto* func = as_type<function_type>(type)) {
            std::vector<TypePtr> params;
            for (const auto& p : func->param_types) params.push_back(apply_impl(p, visited));
            return std::make_shared<function_type>(std::move(params), apply_impl(func->return_type, visited));
        }

        return type;
    }

    std::map<std::string, TypePtr> get_resolved_types(const std::map<std::string, TypePtr>& env) const {
        std::map<std::string, TypePtr> result;
        for (const auto& [name, type] : env) {
            auto resolved = apply(type);
            result[name] = convert_typevars_to_unknown(resolved);
        }
        return result;
    }

    // Convert remaining unresolved type variables to 'unknown' type
    TypePtr convert_typevars_to_unknown(const TypePtr& type) const {
        if (!type) return nullptr;

        if (auto* tv = as_type<type_variable>(type)) {
            return make_unknown(tv->name);
        }

        if (auto* arr = as_type<array_type>(type)) {
            auto result = std::make_shared<array_type>();
            if (arr->element_type) {
                result->element_type = convert_typevars_to_unknown(arr->element_type);
            }
            return result;
        }

        if (auto* obj = as_type<object_type>(type)) {
            auto result = std::make_shared<object_type>(obj->extensible);
            for (const auto& [name, field] : obj->fields) {
                result->add_field(name, convert_typevars_to_unknown(field.type), field.optional);
            }
            return result;
        }

        if (auto* u = as_type<union_type>(type)) {
            auto result = std::make_shared<union_type>();
            for (const auto& alt : u->alternatives) {
                result->add_alternative(convert_typevars_to_unknown(alt));
            }
            return result;
        }

        if (auto* func = as_type<function_type>(type)) {
            std::vector<TypePtr> params;
            for (const auto& p : func->param_types) {
                params.push_back(convert_typevars_to_unknown(p));
            }
            return std::make_shared<function_type>(
                std::move(params),
                convert_typevars_to_unknown(func->return_type)
            );
        }

        return type;
    }

private:
    std::pair<TypePtr, std::string> follow_chain(const TypePtr& type) {
        TypePtr current = type;
        std::string last_var;
        while (auto* tv = as_type<type_variable>(current)) {
            last_var = tv->name;
            auto it = subst.find(tv->name);
            if (it != subst.end()) {
                current = it->second;
            } else {
                break;
            }
        }
        return {current, last_var};
    }

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
            auto new_obj = std::make_shared<object_type>(true);
            new_obj->add_field(c.field_name, c.field_type, c.optional);
            subst[tv->name] = new_obj;
        } else if (auto* existing_obj = as_type<object_type>(current)) {
            add_field_to_object(existing_obj, c.field_name, c.field_type, c.optional, c.source);
        } else if (auto* ut = as_type<union_type>(current)) {
            for (auto& alt : ut->alternatives) {
                solve_constraint(has_field_constraint{alt, c.field_name, c.field_type, c.optional, c.source});
            }
        } else if (!last_var.empty()) {
            auto fallback_obj = std::make_shared<object_type>(true);
            fallback_obj->add_field(c.field_name, c.field_type, c.optional);
            subst[last_var] = fallback_obj;
        }
    }

    void add_field_to_object(object_type* obj, const std::string& field_name,
                             TypePtr field_type, bool optional, const std::string& source) {
        auto* existing = obj->get_field(field_name);
        if (existing) {
            if (!unify(existing->type, field_type)) {
                errors.push_back("Field type mismatch for '" + field_name + "' at " + source);
            }
            if (!optional) existing->optional = false;
        } else {
            obj->add_field(field_name, field_type, optional);
        }
    }

    void solve_constraint(const array_element_constraint& c) {
        auto [current, last_var] = follow_chain(c.array_type);

        if (auto* tv = as_type<type_variable>(current)) {
            subst[tv->name] = make_array(c.element_type);
        } else if (auto* arr = as_type<array_type>(current)) {
            if (arr->element_type) {
                unify(arr->element_type, c.element_type);
            } else {
                arr->element_type = c.element_type;
            }
        } else if (auto* ut = as_type<union_type>(current)) {
            for (auto& alt : ut->alternatives) {
                solve_constraint(array_element_constraint{alt, c.element_type, c.source});
            }
        }
    }

    void solve_constraint(const iterable_constraint& c) {
        auto [current, last_var] = follow_chain(c.type);

        if (auto* tv = as_type<type_variable>(current)) {
            subst[tv->name] = make_array(c.element_type);
        } else if (auto* arr = as_type<array_type>(current)) {
            if (arr->element_type) {
                unify(arr->element_type, c.element_type);
            } else {
                arr->element_type = c.element_type;
            }
        } else if (auto* ut = as_type<union_type>(current)) {
            for (auto& alt : ut->alternatives) {
                solve_constraint(iterable_constraint{alt, c.element_type, c.source});
            }
        }
    }

    void solve_constraint(const callable_constraint& c) {
        auto resolved = apply(c.callee_type);

        if (auto* func = as_type<function_type>(resolved)) {
            unify(func->return_type, c.return_type);
            size_t min_params = std::min(func->param_types.size(), c.arg_types.size());
            for (size_t i = 0; i < min_params; ++i) {
                unify(func->param_types[i], c.arg_types[i]);
            }
        } else if (auto* tv = as_type<type_variable>(resolved)) {
            subst[tv->name] = make_function(c.arg_types, c.return_type);
        }
    }

    void solve_constraint(const output_coercion_constraint& c) {
        auto resolved = apply(c.type);
        if (auto* tv = as_type<type_variable>(resolved)) {
            output_coerced_types.insert(tv->name);
        }
    }

    void solve_constraint(const string_operand_constraint& c) {
        auto resolved = apply(c.type);
        if (auto* tv = as_type<type_variable>(resolved)) {
            if (subst.find(tv->name) == subst.end()) {
                subst[tv->name] = make_string();
            }
        }
    }

    void solve_constraint(const literal_constraint& c) {
        auto resolved = apply(c.type);
        if (auto* tv = as_type<type_variable>(resolved)) {
            auto& literals = literal_values[tv->name];
            bool found = false;
            for (const auto& existing : literals) {
                if (existing->equals(c.literal_value)) { found = true; break; }
            }
            if (!found) literals.push_back(c.literal_value);
        }
    }

    void solve_constraint(const type_alternative_constraint& c) {
        auto resolved = apply(c.type);
        if (auto* tv = as_type<type_variable>(resolved)) {
            auto& alternatives = type_alternatives[tv->name];
            bool found = false;
            for (const auto& existing : alternatives) {
                if (existing->equals(c.alternative)) { found = true; break; }
            }
            if (!found) alternatives.push_back(c.alternative);
        }
    }

    bool unify(TypePtr t1, TypePtr t2) {
        if (!t1 || !t2) return true;

        auto [resolved1, var1] = follow_chain(t1);
        auto [resolved2, var2] = follow_chain(t2);

        if (resolved1 == resolved2) return true;
        if (resolved1->equals(resolved2)) return true;

        // any_type absorbs any other type
        // If unifying with 'any', the type variable becomes 'any'
        if (is_type<any_type>(resolved1)) {
            if (!var2.empty()) subst[var2] = resolved1;
            return true;
        }
        if (is_type<any_type>(resolved2)) {
            if (!var1.empty()) subst[var1] = resolved2;
            return true;
        }

        if (auto* tv1 = as_type<type_variable>(resolved1)) {
            return unify_var(tv1->name, resolved2);
        }
        if (auto* tv2 = as_type<type_variable>(resolved2)) {
            return unify_var(tv2->name, resolved1);
        }

        if (auto* arr1 = as_type<array_type>(resolved1)) {
            if (auto* arr2 = as_type<array_type>(resolved2)) {
                if (arr1->element_type && arr2->element_type) {
                    return unify(arr1->element_type, arr2->element_type);
                }
                if (arr1->element_type) arr2->element_type = arr1->element_type;
                else if (arr2->element_type) arr1->element_type = arr2->element_type;
                return true;
            }
        }

        if (auto* obj1 = as_type<object_type>(resolved1)) {
            if (auto* obj2 = as_type<object_type>(resolved2)) {
                if (!unify_objects(obj1, obj2)) return false;
                if (!var2.empty()) subst[var2] = resolved1;
                return true;
            }
        }

        if (auto* u1 = as_type<union_type>(resolved1)) {
            if (auto* u2 = as_type<union_type>(resolved2)) {
                for (const auto& alt : u2->alternatives) u1->add_alternative(alt);
                return true;
            }
        }

        if (auto* u1 = as_type<union_type>(resolved1)) {
            bool all_ok = true;
            for (const auto& alt : u1->alternatives) {
                if (!unify(alt, resolved2)) all_ok = false;
            }
            return all_ok;
        }

        if (auto* u2 = as_type<union_type>(resolved2)) {
            bool all_ok = true;
            for (const auto& alt : u2->alternatives) {
                if (!unify(resolved1, alt)) all_ok = false;
            }
            return all_ok;
        }

        if (auto* f1 = as_type<function_type>(resolved1)) {
            if (auto* f2 = as_type<function_type>(resolved2)) {
                if (f1->param_types.size() != f2->param_types.size()) return false;
                for (size_t i = 0; i < f1->param_types.size(); ++i) {
                    if (!unify(f1->param_types[i], f2->param_types[i])) return false;
                }
                return unify(f1->return_type, f2->return_type);
            }
        }

        if (auto* p1 = as_type<primitive_type>(resolved1)) {
            if (auto* p2 = as_type<primitive_type>(resolved2)) {
                return p1->prim_kind == p2->prim_kind;
            }
        }

        return false;
    }

    bool unify_var(const std::string& var, TypePtr type) {
        if (occurs_in(var, type)) return true;
        subst[var] = type;
        return true;
    }

    bool unify_objects(object_type* obj1, object_type* obj2) {
        for (const auto& [name, field] : obj2->fields) {
            auto* existing = obj1->get_field(name);
            if (existing) {
                if (!unify(existing->type, field.type)) return false;
                existing->optional = existing->optional && field.optional;
            } else {
                obj1->add_field(name, field.type, field.optional);
            }
        }
        return true;
    }

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
