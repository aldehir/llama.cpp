#pragma once

#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

struct testing_state;

// one node of the test tree: test() and bench() create a child node, hand it to the body, and roll its results up into the parent when it finishes
struct testing {
    // options, copied into a subtest when it starts
    bool verbose         = false;
    bool throw_exception = false;
    // capture fd 1 and 2 while a test runs and print the output only if the test fails
    bool capture_output  = false;

    // results of this node and everything below it
    int tests      = 0;
    int assertions = 0;
    int failures   = 0;
    int exceptions = 0;
    int skipped    = 0;

    std::string name;
    testing *   parent = nullptr;
    std::vector<std::unique_ptr<testing>> subtests;

    static constexpr std::size_t status_column = 80;

    explicit testing(std::ostream & os = std::cout);
    ~testing();

    testing(const testing &) = delete;
    testing & operator=(const testing &) = delete;

    // where the framework itself writes; bypasses the capture redirect
    std::ostream & stream() const;

    int depth() const;
    std::string indent() const;
    std::string full_name() const;

    void log(const std::string & msg);
    // LLAMA_TEST_VERBOSE=1 turns on log(), LLAMA_TEST_CAPTURE=0 shows test output live, LLAMA_TEST_FILTER sets the filter
    void apply_env();
    // run only the tests whose dotted full name matches; a matched test runs its whole subtree
    void set_filter(const std::string & re);
    void skip(const std::string & reason = "");

    template <typename F>
    void test(const std::string & test_name, F f) {
        run_test(test_name, [&](testing & t) { f(t); });
    }

    template <typename F>
    void test(F f) {
        run_test(next_unnamed("test #"), [&](testing & t) { f(t); });
    }

    template <typename F>
    void bench(const std::string & test_name, F f, int iterations = 100) {
        run_bench(test_name, [&] { f(); }, iterations);
    }

    template <typename F>
    void bench(F f, int iterations = 100) {
        run_bench(next_unnamed("bench #"), [&] { f(); }, iterations);
    }

    // Assertions
    bool assert_true(bool cond) {
        return assert_true("", cond);
    }

    bool assert_true(const std::string & msg, bool cond);

    template <typename A, typename B>
    bool assert_equal(const A & expected, const B & actual) {
        return assert_equal("", expected, actual);
    }

    template <typename A, typename B>
    bool assert_equal(const std::string & msg, const A & expected, const B & actual) {
        ++assertions;
        if (!(actual == expected)) {
            ++failures;
            std::ostream & out = stream();
            out << indent() << "ASSERT EQUAL FAILED";
            if (!msg.empty()) {
                out << " : " << msg;
            }
            out << "\n";

            out << indent() << "  expected: " << expected << "\n";
            out << indent() << "  actual  : " << actual << "\n";
            return false;
        }
        return true;
    }

    // Print summary and return an exit code
    int summary() const;

private:
    testing(testing & p, const std::string & n);

    std::string next_unnamed(const char * prefix);

    void run_test(const std::string & test_name, const std::function<void(testing &)> & body);
    void run_bench(const std::string & test_name, const std::function<void()> & body, int iterations);
    void run_guarded(const std::function<void()> & body, const char * ctx);

    testing * begin(const std::string & test_name, const std::string & label);
    std::string end_capture();
    void roll_up();
    void finish(const std::string & label, const std::string & extra);

    void print_result(const std::string & label, const std::string & extra, bool was_skipped) const;
    void print_captured(const std::string & captured) const;
    void collect_failed(std::vector<std::string> & names) const;

    std::shared_ptr<testing_state> state;

    bool        matched        = false;
    bool        skip_requested = false;
    std::string skip_reason;

    bool        capturing     = false;
    std::size_t capture_start = 0;
};
