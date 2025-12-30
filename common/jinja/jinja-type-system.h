#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <set>

namespace jinja {
namespace types {

// Forward declarations
struct Type;
using TypePtr = std::shared_ptr<Type>;

// Type variable name generator
struct TypeVarGenerator {
    static inline int counter = 0;
    static std::string fresh() { return "T" + std::to_string(++counter); }
    static void reset() { counter = 0; }
};

/**
 * Base type class - all types inherit from this
 */
struct Type {
    virtual ~Type() = default;
    virtual std::string to_string() const = 0;
    virtual bool equals(const TypePtr& other) const = 0;
    virtual TypePtr clone() const = 0;
    virtual std::string kind() const = 0;
};

/**
 * Primitive types: int, float, string, bool, null
 */
struct PrimitiveType : public Type {
    enum Kind { Int, Float, String, Bool, Null };
    Kind prim_kind;

    explicit PrimitiveType(Kind k) : prim_kind(k) {}

    std::string to_string() const override {
        switch (prim_kind) {
            case Int:    return "int";
            case Float:  return "float";
            case String: return "string";
            case Bool:   return "bool";
            case Null:   return "null";
        }
        return "unknown_primitive";
    }

    bool equals(const TypePtr& other) const override {
        auto* p = dynamic_cast<PrimitiveType*>(other.get());
        return p && p->prim_kind == prim_kind;
    }

    TypePtr clone() const override {
        return std::make_shared<PrimitiveType>(prim_kind);
    }

    std::string kind() const override { return "primitive"; }
};

/**
 * Array type with element type
 * Example: array<string> or array<{content: string, role: string}>
 */
struct ArrayType : public Type {
    TypePtr element_type;

    explicit ArrayType(TypePtr elem = nullptr) : element_type(std::move(elem)) {}

    std::string to_string() const override {
        return "array<" + (element_type ? element_type->to_string() : "unknown") + ">";
    }

    bool equals(const TypePtr& other) const override {
        auto* a = dynamic_cast<ArrayType*>(other.get());
        if (!a) return false;
        if (!element_type && !a->element_type) return true;
        if (!element_type || !a->element_type) return false;
        return element_type->equals(a->element_type);
    }

    TypePtr clone() const override {
        return std::make_shared<ArrayType>(element_type ? element_type->clone() : nullptr);
    }

    std::string kind() const override { return "array"; }
};

/**
 * Object type with named fields (structural typing)
 * Example: {content: string, role: string, tool_calls?: array<...>}
 */
struct ObjectType : public Type {
    struct Field {
        std::string name;
        TypePtr type;
        bool optional;  // true if field might not exist (accessed in conditional)

        Field() : optional(false) {}
        Field(const std::string& n, TypePtr t, bool opt = false)
            : name(n), type(std::move(t)), optional(opt) {}
    };

    std::map<std::string, Field> fields;
    bool extensible;  // true if object may have additional unknown fields

    explicit ObjectType(bool ext = true) : extensible(ext) {}

    void add_field(const std::string& name, TypePtr type, bool optional = false) {
        fields[name] = Field{name, std::move(type), optional};
    }

    bool has_field(const std::string& name) const {
        return fields.find(name) != fields.end();
    }

    Field* get_field(const std::string& name) {
        auto it = fields.find(name);
        return it != fields.end() ? &it->second : nullptr;
    }

    const Field* get_field(const std::string& name) const {
        auto it = fields.find(name);
        return it != fields.end() ? &it->second : nullptr;
    }

    std::string to_string() const override {
        std::string result = "{";
        bool first = true;
        for (const auto& [name, field] : fields) {
            if (!first) result += ", ";
            first = false;
            result += name;
            if (field.optional) result += "?";
            result += ": " + (field.type ? field.type->to_string() : "unknown");
        }
        result += "}";
        return result;
    }

    bool equals(const TypePtr& other) const override {
        auto* o = dynamic_cast<ObjectType*>(other.get());
        if (!o) return false;
        if (fields.size() != o->fields.size()) return false;
        for (const auto& [name, field] : fields) {
            auto it = o->fields.find(name);
            if (it == o->fields.end()) return false;
            if (field.optional != it->second.optional) return false;
            if (!field.type || !it->second.type) {
                if (field.type != it->second.type) return false;
            } else if (!field.type->equals(it->second.type)) {
                return false;
            }
        }
        return true;
    }

    TypePtr clone() const override {
        auto obj = std::make_shared<ObjectType>(extensible);
        for (const auto& [name, field] : fields) {
            obj->fields[name] = Field{name, field.type ? field.type->clone() : nullptr, field.optional};
        }
        return obj;
    }

    std::string kind() const override { return "object"; }
};

/**
 * Union type: represents "type A or type B"
 * Example: string | null (optional string)
 */
struct UnionType : public Type {
    std::vector<TypePtr> alternatives;

    UnionType() = default;
    explicit UnionType(std::vector<TypePtr> alts) : alternatives(std::move(alts)) {}

    void add_alternative(TypePtr type) {
        // Don't add duplicates
        for (const auto& alt : alternatives) {
            if (alt->equals(type)) return;
        }
        alternatives.push_back(std::move(type));
    }

    std::string to_string() const override {
        std::string result;
        for (size_t i = 0; i < alternatives.size(); ++i) {
            if (i > 0) result += " | ";
            result += alternatives[i]->to_string();
        }
        return result.empty() ? "never" : result;
    }

    bool equals(const TypePtr& other) const override {
        auto* u = dynamic_cast<UnionType*>(other.get());
        if (!u) return false;
        if (alternatives.size() != u->alternatives.size()) return false;
        // Order-independent comparison
        for (const auto& alt : alternatives) {
            bool found = false;
            for (const auto& ualt : u->alternatives) {
                if (alt->equals(ualt)) { found = true; break; }
            }
            if (!found) return false;
        }
        return true;
    }

    TypePtr clone() const override {
        auto u = std::make_shared<UnionType>();
        for (const auto& alt : alternatives) {
            u->alternatives.push_back(alt->clone());
        }
        return u;
    }

    std::string kind() const override { return "union"; }
};

/**
 * Type variable: represents an unknown type to be solved
 * Used during constraint solving
 */
struct TypeVariable : public Type {
    std::string name;
    TypePtr bound;  // If solved, this holds the concrete type

    explicit TypeVariable(const std::string& n) : name(n), bound(nullptr) {}

    bool is_bound() const { return bound != nullptr; }

    TypePtr resolve() const {
        if (!bound) return nullptr;
        // Follow chain of type variables
        if (auto* tv = dynamic_cast<TypeVariable*>(bound.get())) {
            return tv->resolve();
        }
        return bound;
    }

    std::string to_string() const override {
        if (bound) return bound->to_string();
        return "'" + name;
    }

    bool equals(const TypePtr& other) const override {
        if (bound) return bound->equals(other);
        auto* tv = dynamic_cast<TypeVariable*>(other.get());
        return tv && tv->name == name;
    }

    TypePtr clone() const override {
        auto tv = std::make_shared<TypeVariable>(name);
        if (bound) tv->bound = bound->clone();
        return tv;
    }

    std::string kind() const override { return "type_variable"; }
};

/**
 * Function type (for macros and filters)
 */
struct FunctionType : public Type {
    std::vector<TypePtr> param_types;
    TypePtr return_type;

    FunctionType() = default;
    FunctionType(std::vector<TypePtr> params, TypePtr ret)
        : param_types(std::move(params)), return_type(std::move(ret)) {}

    std::string to_string() const override {
        std::string result = "(";
        for (size_t i = 0; i < param_types.size(); ++i) {
            if (i > 0) result += ", ";
            result += param_types[i] ? param_types[i]->to_string() : "unknown";
        }
        result += ") -> " + (return_type ? return_type->to_string() : "unknown");
        return result;
    }

    bool equals(const TypePtr& other) const override {
        auto* f = dynamic_cast<FunctionType*>(other.get());
        if (!f) return false;
        if (param_types.size() != f->param_types.size()) return false;
        for (size_t i = 0; i < param_types.size(); ++i) {
            if (!param_types[i] || !f->param_types[i]) {
                if (param_types[i] != f->param_types[i]) return false;
            } else if (!param_types[i]->equals(f->param_types[i])) {
                return false;
            }
        }
        if (!return_type || !f->return_type) {
            return return_type == f->return_type;
        }
        return return_type->equals(f->return_type);
    }

    TypePtr clone() const override {
        std::vector<TypePtr> params;
        for (const auto& p : param_types) {
            params.push_back(p ? p->clone() : nullptr);
        }
        return std::make_shared<FunctionType>(std::move(params),
            return_type ? return_type->clone() : nullptr);
    }

    std::string kind() const override { return "function"; }
};

// ============================================================================
// Factory functions
// ============================================================================

inline TypePtr make_int() {
    return std::make_shared<PrimitiveType>(PrimitiveType::Int);
}

inline TypePtr make_float() {
    return std::make_shared<PrimitiveType>(PrimitiveType::Float);
}

inline TypePtr make_string() {
    return std::make_shared<PrimitiveType>(PrimitiveType::String);
}

inline TypePtr make_bool() {
    return std::make_shared<PrimitiveType>(PrimitiveType::Bool);
}

inline TypePtr make_null() {
    return std::make_shared<PrimitiveType>(PrimitiveType::Null);
}

inline TypePtr make_array(TypePtr elem = nullptr) {
    return std::make_shared<ArrayType>(std::move(elem));
}

inline TypePtr make_object(bool extensible = true) {
    return std::make_shared<ObjectType>(extensible);
}

inline TypePtr make_union() {
    return std::make_shared<UnionType>();
}

inline TypePtr make_typevar(const std::string& name) {
    return std::make_shared<TypeVariable>(name);
}

inline TypePtr make_function(std::vector<TypePtr> params, TypePtr ret) {
    return std::make_shared<FunctionType>(std::move(params), std::move(ret));
}

// Helper for optional types (T | null)
inline TypePtr make_optional(TypePtr type) {
    auto u = std::make_shared<UnionType>();
    u->add_alternative(std::move(type));
    u->add_alternative(make_null());
    return u;
}

// ============================================================================
// Type casting helpers
// ============================================================================

template<typename T>
inline T* as_type(const TypePtr& ptr) {
    return dynamic_cast<T*>(ptr.get());
}

template<typename T>
inline bool is_type(const TypePtr& ptr) {
    return dynamic_cast<T*>(ptr.get()) != nullptr;
}

} // namespace types
} // namespace jinja
