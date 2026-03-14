#pragma once

#include "llama.h"
#include "llama-grammar-dfa.h"

#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>

struct llama_vocab;

// grammar element type
enum llama_gretype {
    // end of rule definition
    LLAMA_GRETYPE_END            = 0,

    // start of alternate definition for rule
    LLAMA_GRETYPE_ALT            = 1,

    // non-terminal element: reference to rule
    LLAMA_GRETYPE_RULE_REF       = 2,

    // terminal element: character (code point)
    LLAMA_GRETYPE_CHAR           = 3,

    // inverse char(s) ([^a], [^a-b] [^abc])
    LLAMA_GRETYPE_CHAR_NOT       = 4,

    // modifies a preceding LLAMA_GRETYPE_CHAR or LLAMA_GRETYPE_CHAR_ALT to
    // be an inclusive range ([a-z])
    LLAMA_GRETYPE_CHAR_RNG_UPPER = 5,

    // modifies a preceding LLAMA_GRETYPE_CHAR or
    // LLAMA_GRETYPE_CHAR_RNG_UPPER to add an alternate char to match ([ab], [a-zA])
    LLAMA_GRETYPE_CHAR_ALT       = 6,

    // any character (.)
    LLAMA_GRETYPE_CHAR_ANY       = 7,
};

typedef struct llama_grammar_element {
    enum llama_gretype type;
    uint32_t           value; // Unicode code point or rule ID
} llama_grammar_element;

using llama_grammar_rule  = std::vector<      llama_grammar_element>;
using llama_grammar_rules = std::vector<llama_grammar_rule>;

struct llama_grammar_parser {
    std::map<std::string, uint32_t> symbol_ids;

    llama_grammar_rules rules;

    std::vector<const llama_grammar_element *> c_rules() const;

    uint32_t get_symbol_id(const char * src, size_t len);
    uint32_t generate_symbol_id(const std::string & base_name);

    void add_rule(uint32_t rule_id, const llama_grammar_rule & rule);

    const char * parse_alternates(
            const char        * src,
            const std::string & rule_name,
            uint32_t            rule_id,
            bool                is_nested);

    const char * parse_sequence(
            const char         * src,
            const std::string  & rule_name,
            llama_grammar_rule & rule,
            bool               is_nested);

    const char * parse_rule(const char * src);

    bool parse(const char * src);
    void print(FILE * file);
};

struct llama_grammar_trigger_pattern {
    std::string pattern;
    std::regex  regex;
};

// Persistent thread pool for parallelizing grammar_simulate_token() calls.
// Workers sleep on a condition variable between apply() calls — no thread
// creation/destruction per token.
struct grammar_thread_pool {
    static constexpr int N_THREADS = 8;

    grammar_thread_pool();
    ~grammar_thread_pool();

    // Run fn(i) for i in [0, n_items) across worker threads, blocking until done.
    // Falls back to single-threaded for small n_items.
    void run(const std::function<void(size_t, size_t)> & fn, size_t n_items);

    grammar_thread_pool(const grammar_thread_pool &) = delete;
    grammar_thread_pool & operator=(const grammar_thread_pool &) = delete;

private:
    std::vector<std::thread>    workers;
    std::mutex                  mtx;
    std::condition_variable     cv_work;   // workers wait on this
    std::condition_variable     cv_done;   // caller waits on this

    // Work descriptor — set by run(), read by workers
    std::function<void(size_t, size_t)> work_fn;
    size_t                      work_lo[N_THREADS] = {};
    size_t                      work_hi[N_THREADS] = {};

    int                         n_active  = 0;     // workers still running
    int                         n_workers = 0;     // workers to wake
    bool                        stop      = false;
};

struct llama_grammar {
    // note: allow null vocab for testing (not great)
    const llama_vocab * vocab;

    // DFA-based engine
    std::shared_ptr<compiled_grammar> compiled;  // immutable after init, shared across clones
    std::vector<parse_config>         configs;    // mutable runtime state

    // Vocab trie for efficient prefix-based candidate lookup (shared across clones)
    std::shared_ptr<vocab_byte_trie>  trie;

    // lazy grammars wait for trigger words or tokens before constraining the sampling.
    // we still have trigger_tokens for non-lazy grammars to force printing of special trigger tokens.
    // (useful e.g. for tool_choice=required)
    bool                     lazy             = false;
    bool                     awaiting_trigger = false; // Initialized to true for lazy grammars only
    std::string              trigger_buffer;           // Output buffered by lazy grammar. Will be cleared once trigger is found.
    std::vector<llama_token> trigger_tokens;           // Tokens that trigger a lazy grammar, or tokens to force printing of (even if special).
    std::vector<llama_grammar_trigger_pattern>
                             trigger_patterns;         // Regular expressions that trigger a lazy grammar. Must be a full match of the entire generated
                                                       // string, and the grammar will be given the string from the first match group onwards.

    // Thread pool for parallel grammar simulation (shared across clones)
    std::shared_ptr<grammar_thread_pool> pool;
};

//
// internal API
//

// note: needed for tests (not great)
struct llama_grammar * llama_grammar_init_impl(
        const struct llama_vocab * vocab,
        const llama_grammar_element ** rules,
        size_t n_rules,
        size_t start_rule_index);

struct llama_grammar * llama_grammar_init_impl(
        const struct llama_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                              bool lazy,
                     const char ** trigger_patterns,
                            size_t num_trigger_patterns,
               const llama_token * trigger_tokens,
                            size_t num_trigger_tokens);

void llama_grammar_free_impl(struct llama_grammar * grammar);

struct llama_grammar * llama_grammar_clone_impl(const struct llama_grammar & grammar);

// TODO: move the API below as member functions of llama_grammar
void llama_grammar_apply_impl(
        const struct llama_grammar & grammar,
            llama_token_data_array * cur_p);

void llama_grammar_accept_impl(
              struct llama_grammar & grammar,
                       llama_token   token);

void llama_grammar_accept_str(
              struct llama_grammar & grammar,
                 const std::string & piece);
