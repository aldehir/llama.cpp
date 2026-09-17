#pragma once

#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

struct testing_state;

enum testing_style {
    TESTING_STYLE_DOTS, // one mark per test, failures reported with their output at the end
    TESTING_STYLE_TREE, // one line per test with its result as it finishes
};

// one node of the test tree: test() and bench() create a child node, hand it to the body, and roll its results up into the parent when it finishes
struct testing {
    // options, copied into a subtest when it starts
    bool verbose         = false;
    bool throw_exception = false;
    // capture fd 1 and 2 while a test runs and print the output only if the test fails
    bool capture_output  = false;
    testing_style style  = TESTING_STYLE_DOTS;

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
    // where messages about this test go: the console in tree style, the failure report in dots style
    std::ostream & report_stream();

    int depth() const;
    std::string indent() const;
    std::string full_name() const;

    // in dots style the message is kept and shown only if the test fails
    void log(const std::string & msg);
    // LLAMA_TEST_VERBOSE=1 turns on log(), LLAMA_TEST_CAPTURE=0 shows test output live, LLAMA_TEST_FILTER sets the filter, LLAMA_TEST_STYLE=tree|dots picks the output style
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
            std::ostream & out = report_stream();
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

    int own_failures() const;
    int own_exceptions() const;

    void print_mark(char mark) const;
    void print_result(const std::string & label, const std::string & extra, bool was_skipped) const;
    void print_captured(const std::string & captured) const;
    void print_failures(const std::vector<const testing *> & failed) const;
    void collect_failed(std::vector<const testing *> & failed) const;

    std::shared_ptr<testing_state> state;

    bool        matched        = false;
    bool        skip_requested = false;
    std::string skip_reason;

    bool        capturing     = false;
    std::size_t capture_start = 0;

    // dots style keeps the messages and the captured output of a failed test for the report
    std::ostringstream report;
    std::string        captured;
};
