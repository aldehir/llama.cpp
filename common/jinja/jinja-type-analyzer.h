#pragma once

#include "jinja-type-system.h"
#include "jinja-constraints.h"
#include "jinja-type-infer-visitor.h"
#include "jinja-type-solver.h"
#include "jinja-parser.h"

#include <string>
#include <map>
#include <set>
#include <sstream>

namespace jinja {
namespace types {

/**
 * Result of type inference analysis
 */
struct InferenceResult {
    // Inferred types for top-level variables
    std::map<std::string, TypePtr> variable_types;

    // All generated constraints (for debugging)
    ConstraintSet constraints;

    // Any errors encountered
    std::vector<std::string> errors;

    bool success() const { return errors.empty(); }

    /**
     * Pretty print the inferred schema
     */
    std::string to_string() const {
        std::ostringstream oss;
        oss << "=== Inferred Template Schema ===\n";

        for (const auto& [name, type] : variable_types) {
            oss << name << ": " << (type ? type->to_string() : "unknown") << "\n";
        }

        if (!errors.empty()) {
            oss << "\n=== Errors ===\n";
            for (const auto& err : errors) {
                oss << "  - " << err << "\n";
            }
        }

        return oss.str();
    }

    /**
     * Get type for a specific variable
     */
    TypePtr get_type(const std::string& name) const {
        auto it = variable_types.find(name);
        return it != variable_types.end() ? it->second : nullptr;
    }

    /**
     * Check if a variable exists in the inferred schema
     */
    bool has_variable(const std::string& name) const {
        return variable_types.find(name) != variable_types.end();
    }

    /**
     * Get all variable names
     */
    std::vector<std::string> variable_names() const {
        std::vector<std::string> names;
        for (const auto& [name, _] : variable_types) {
            names.push_back(name);
        }
        return names;
    }

    /**
     * Debug output: print all constraints
     */
    std::string constraints_to_string() const {
        return constraints.to_string();
    }
};

/**
 * Main entry point for type inference
 */
class TypeAnalyzer {
public:
    TypeAnalyzer() = default;

    /**
     * Analyze a parsed Jinja program and infer types
     */
    InferenceResult analyze(program& ast) {
        InferenceResult result;
        TypeVarGenerator::reset();

        // Create root environment
        TypeEnvironment root_env;

        // Add built-in globals
        setup_builtins(root_env);

        // Generate constraints
        TypeInferenceVisitor visitor(result.constraints, &root_env);
        visitor.dispatch(ast);

        // Solve constraints
        ConstraintSolver solver;
        if (!solver.solve(result.constraints)) {
            result.errors = solver.errors;
        }

        // Extract final types (resolve all type variables)
        auto resolved = solver.get_resolved_types(root_env.local_bindings());

        // Filter to only show "interesting" variables (not built-ins)
        filter_builtins(resolved);

        result.variable_types = std::move(resolved);

        return result;
    }

private:
    void setup_builtins(TypeEnvironment& env) {
        // Standard Jinja built-ins
        env.bind("true", make_bool());
        env.bind("false", make_bool());
        env.bind("none", make_null());

        // Built-in functions (these return specific types)
        env.bind("range", make_function({make_int()}, make_array(make_int())));
        env.bind("dict", make_function({}, make_object()));
        env.bind("cycler", make_function({}, make_object()));
        env.bind("joiner", make_function({}, make_function({}, make_string())));
        env.bind("namespace", make_function({}, make_object()));
    }

    void filter_builtins(std::map<std::string, TypePtr>& types) {
        static const std::set<std::string> builtins = {
            "true", "false", "none", "loop",
            "range", "dict", "cycler", "joiner", "namespace"
        };

        for (const auto& name : builtins) {
            types.erase(name);
        }
    }
};

// ============================================================================
// Convenience functions
// ============================================================================

/**
 * Analyze a parsed program and infer types
 */
inline InferenceResult infer_types(program& ast) {
    TypeAnalyzer analyzer;
    return analyzer.analyze(ast);
}

/**
 * Parse and analyze a template source string
 */
inline InferenceResult infer_types_from_source(const std::string& source) {
    lexer lex;
    preprocess_options opts;
    auto lexer_res = lex.tokenize(source, opts);
    program ast = parse_from_tokens(lexer_res);
    return infer_types(ast);
}

/**
 * Parse and analyze a template source with custom preprocessing options
 */
inline InferenceResult infer_types_from_source(
    const std::string& source,
    const preprocess_options& opts)
{
    lexer lex;
    auto lexer_res = lex.tokenize(source, opts);
    program ast = parse_from_tokens(lexer_res);
    return infer_types(ast);
}

// ============================================================================
// Type query utilities
// ============================================================================

/**
 * Check if an object type has a specific field
 */
inline bool has_field(const TypePtr& type, const std::string& field_name) {
    if (auto* obj = as_type<ObjectType>(type)) {
        return obj->has_field(field_name);
    }
    return false;
}

/**
 * Check if a field is optional in an object type
 */
inline bool is_field_optional(const TypePtr& type, const std::string& field_name) {
    if (auto* obj = as_type<ObjectType>(type)) {
        auto* field = obj->get_field(field_name);
        return field ? field->optional : true;
    }
    return true;
}

/**
 * Get the element type of an array
 */
inline TypePtr get_element_type(const TypePtr& type) {
    if (auto* arr = as_type<ArrayType>(type)) {
        return arr->element_type;
    }
    return nullptr;
}

/**
 * Get a field type from an object
 */
inline TypePtr get_field_type(const TypePtr& type, const std::string& field_name) {
    if (auto* obj = as_type<ObjectType>(type)) {
        auto* field = obj->get_field(field_name);
        return field ? field->type : nullptr;
    }
    return nullptr;
}

/**
 * Check if type represents a message array (common pattern in chat templates)
 * Returns true if type is array<{role: string, content: string, ...}>
 */
inline bool is_message_array(const TypePtr& type) {
    if (auto* arr = as_type<ArrayType>(type)) {
        if (arr->element_type) {
            if (auto* obj = as_type<ObjectType>(arr->element_type)) {
                // Check for common message fields
                bool has_role = obj->has_field("role");
                bool has_content = obj->has_field("content");
                return has_role || has_content;
            }
        }
    }
    return false;
}

/**
 * Check if messages support tool_calls
 */
inline bool supports_tool_calls(const TypePtr& type) {
    if (auto* arr = as_type<ArrayType>(type)) {
        if (arr->element_type) {
            if (auto* obj = as_type<ObjectType>(arr->element_type)) {
                return obj->has_field("tool_calls");
            }
        }
    }
    return false;
}

/**
 * Check if messages support reasoning_content
 */
inline bool supports_reasoning_content(const TypePtr& type) {
    if (auto* arr = as_type<ArrayType>(type)) {
        if (arr->element_type) {
            if (auto* obj = as_type<ObjectType>(arr->element_type)) {
                return obj->has_field("reasoning_content");
            }
        }
    }
    return false;
}

} // namespace types
} // namespace jinja
