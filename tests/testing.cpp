#include "testing.h"

#include "log.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <regex>
#include <streambuf>
#include <thread>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <cerrno>
#include <unistd.h>
#endif

namespace {

#ifdef _WIN32
int fd_dup(int fd)                               { return _dup(fd); }
int fd_dup2(int from, int to)                    { return _dup2(from, to); }
int fd_close(int fd)                             { return _close(fd); }
int fd_pipe(int fds[2])                          { return _pipe(fds, 1 << 16, _O_BINARY); }
int fd_read(int fd, char * buf, size_t n)        { return _read(fd, buf, (unsigned) n); }
int fd_write(int fd, const char * buf, size_t n) { return _write(fd, buf, (unsigned) n); }
bool fd_is_tty(int fd)                           { return _isatty(fd) != 0; }
#else
int fd_dup(int fd)                               { return dup(fd); }
int fd_dup2(int from, int to)                    { return dup2(from, to); }
int fd_close(int fd)                             { return close(fd); }
int fd_pipe(int fds[2])                          { return pipe(fds); }
int fd_read(int fd, char * buf, size_t n)        { return (int) read(fd, buf, n); }
int fd_write(int fd, const char * buf, size_t n) { return (int) write(fd, buf, n); }
bool fd_is_tty(int fd)                           { return isatty(fd) != 0; }
#endif

// ANSI colors are used only on a terminal; Windows consoles need virtual terminal processing switched on
bool fd_wants_color(int fd) {
    if (fd < 0 || !fd_is_tty(fd) || getenv("NO_COLOR")) {
        return false;
    }
#ifdef _WIN32
    HANDLE h = (HANDLE) _get_osfhandle(fd);
    DWORD mode = 0;
    if (!GetConsoleMode(h, &mode)) {
        return false;
    }
    return SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0;
#else
    return true;
#endif
}

// a title centered in a line of fill characters, as in the pytest report
std::string rule(const std::string & title, char fill) {
    const size_t width = testing::status_column;
    std::string  text  = " " + title + " ";
    if (text.size() + 2 > width) {
        return text;
    }
    size_t left = (width - text.size()) / 2;
    return std::string(left, fill) + text + std::string(width - text.size() - left, fill);
}

void fd_write_all(int fd, const char * buf, size_t n) {
    while (n > 0) {
        int w = fd_write(fd, buf, n);
        if (w < 0) {
#ifndef _WIN32
            if (errno == EINTR) {
                continue;
            }
#endif
            return;
        }
        buf += w;
        n   -= (size_t) w;
    }
}

// drain the C and C++ standard stream buffers before a redirect switch
void flush_std() {
    std::cout.flush();
    std::cerr.flush();
    std::clog.flush();
    fflush(stdout);
    fflush(stderr);
}

// unbuffered stream backend that writes straight to a file descriptor
struct fd_streambuf : std::streambuf {
    int fd;

    explicit fd_streambuf(int fd) : fd(fd) {}

    int_type overflow(int_type ch) override {
        if (!traits_type::eq_int_type(ch, traits_type::eof())) {
            char c = traits_type::to_char_type(ch);
            fd_write_all(fd, &c, 1);
        }
        return traits_type::not_eof(ch);
    }

    std::streamsize xsputn(const char * s, std::streamsize n) override {
        fd_write_all(fd, s, (size_t) n);
        return n;
    }
};

// written into the pipe after a test body; once the reader sees it, everything the body wrote is in the buffer
const char   SENTINEL[]   = "\x1f\x1e" "testing-capture-sync" "\x1e\x1f";
const size_t SENTINEL_LEN = sizeof(SENTINEL) - 1;

// redirects fd 1 and 2 into a pipe that one long-lived thread drains into a buffer; nested tests mark their start and cut out their own segment when they finish
struct testing_capture {
    bool ok = false;

    int pipe_rd   = -1;
    int pipe_wr   = -1;
    int saved_out = -1;
    int saved_err = -1;
    int depth     = 0;

    // the original stdout (or stderr) descriptor, for the framework output
    fd_streambuf console_buf{-1};
    std::ostream console{&console_buf};

    std::thread             reader;
    std::mutex              mtx;
    std::condition_variable cv;
    std::string             buffer;
    uint64_t                sent        = 0;
    uint64_t                acked       = 0;
    bool                    stopping    = false;
    bool                    reader_done = false;

    explicit testing_capture(bool console_is_stderr) {
        int fds[2];
        if (fd_pipe(fds) != 0) {
            return;
        }
        pipe_rd   = fds[0];
        pipe_wr   = fds[1];
        saved_out = fd_dup(1);
        saved_err = fd_dup(2);
        if (saved_out < 0 || saved_err < 0) {
            close_all();
            return;
        }
        console_buf.fd = console_is_stderr ? saved_err : saved_out;
        reader = std::thread([this] { reader_loop(); });
        ok = true;
    }

    ~testing_capture() {
        if (depth > 0) {
            depth = 1;
            pop();
        }
        {
            std::lock_guard<std::mutex> lock(mtx);
            stopping = true;
        }
        if (reader.joinable()) {
            if (!reader_done) {
                fd_write_all(pipe_wr, SENTINEL, SENTINEL_LEN);
            }
            reader.join();
        }
        close_all();
    }

    void close_all() {
        for (int * fd : { &pipe_rd, &pipe_wr, &saved_out, &saved_err }) {
            if (*fd >= 0) {
                fd_close(*fd);
                *fd = -1;
            }
        }
    }

    void push() {
        if (depth++ == 0) {
            flush_std();
            fd_dup2(pipe_wr, 1);
            fd_dup2(pipe_wr, 2);
        }
    }

    void pop() {
        if (--depth == 0) {
            flush_std();
            fd_dup2(saved_out, 1);
            fd_dup2(saved_err, 2);

            // anything that arrived after the last take() belongs to no test; print it rather than lose it
            std::string leftover;
            {
                std::lock_guard<std::mutex> lock(mtx);
                leftover.swap(buffer);
            }
            if (!leftover.empty()) {
                console << leftover;
            }
        }
    }

    size_t mark() {
        sync();
        std::lock_guard<std::mutex> lock(mtx);
        return buffer.size();
    }

    std::string take(size_t from) {
        sync();
        std::lock_guard<std::mutex> lock(mtx);
        std::string segment = buffer.substr(from);
        buffer.resize(from);
        return segment;
    }

    void sync() {
        flush_std();
        // the common logger writes from its own thread; pause joins that thread after it drains its queue
        common_log_pause(common_log_main());
        common_log_resume(common_log_main());
        uint64_t expected;
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (reader_done) {
                return;
            }
            expected = ++sent;
        }
        fd_write_all(pipe_wr, SENTINEL, SENTINEL_LEN);
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] { return acked >= expected || reader_done; });
    }

    void reader_loop() {
        char chunk[4096];
        for (;;) {
            int n = fd_read(pipe_rd, chunk, sizeof(chunk));
            if (n < 0) {
#ifndef _WIN32
                if (errno == EINTR) {
                    continue;
                }
#endif
                break;
            }
            if (n == 0) {
                break;
            }

            std::lock_guard<std::mutex> lock(mtx);
            // a sentinel may straddle two reads, so rescan the tail of the previous one
            size_t scan = buffer.size() > SENTINEL_LEN - 1 ? buffer.size() - (SENTINEL_LEN - 1) : 0;
            buffer.append(chunk, (size_t) n);
            bool stop = false;
            for (size_t p; (p = buffer.find(SENTINEL, scan)) != std::string::npos; scan = p) {
                buffer.erase(p, SENTINEL_LEN);
                ++acked;
                stop = stopping;
            }
            cv.notify_all();
            if (stop) {
                break;
            }
        }
        std::lock_guard<std::mutex> lock(mtx);
        reader_done = true;
        cv.notify_all();
    }
};

} // namespace

// shared by every node of one tree
struct testing_state {
    std::ostream & out;
    std::regex     filter;
    std::string    filter_text;
    bool           filter_tests = false;
    int            unnamed      = 0;

    std::unique_ptr<testing_capture> capture;

    // dots style state
    size_t marks_on_line = 0;
    bool   color         = false;
    bool   color_checked = false;

    explicit testing_state(std::ostream & os) : out(os) {}

    bool out_is_console() const {
        return &out == &std::cout || &out == &std::cerr;
    }

    int console_fd() const {
        if (capture) {
            return capture->console_buf.fd;
        }
        return &out == &std::cout ? 1 : &out == &std::cerr ? 2 : -1;
    }

    bool use_color() {
        if (!color_checked) {
            color_checked = true;
            color = out_is_console() && fd_wants_color(console_fd());
        }
        return color;
    }

    std::string colored(const std::string & text, const char * code) {
        return use_color() ? std::string("\x1b[") + code + "m" + text + "\x1b[0m" : text;
    }

    void end_marks(std::ostream & os) {
        if (marks_on_line > 0) {
            os << "\n";
            marks_on_line = 0;
        }
    }

    testing_capture * get_capture() {
        if (!capture) {
            capture.reset(new testing_capture(&out == &std::cerr));
        }
        return capture->ok ? capture.get() : nullptr;
    }
};

testing::testing(std::ostream & os) : state(std::make_shared<testing_state>(os)) {}

testing::testing(testing & p, const std::string & n) :
    verbose(p.verbose),
    throw_exception(p.throw_exception),
    capture_output(p.capture_output),
    style(p.style),
    name(n),
    parent(&p),
    state(p.state) {}

testing::~testing() = default;

std::ostream & testing::stream() const {
    if (state->capture && state->out_is_console()) {
        flush_std();
        return state->capture->console;
    }
    return state->out;
}

std::ostream & testing::report_stream() {
    if (style == TESTING_STYLE_DOTS) {
        return report;
    }
    return stream();
}

int testing::depth() const {
    int d = 0;
    for (const testing * p = parent; p; p = p->parent) {
        ++d;
    }
    return d;
}

std::string testing::indent() const {
    int d = depth();
    if (d <= 1 || style == TESTING_STYLE_DOTS) {
        return "";
    }
    return std::string((d - 1) * 2, ' ');
}

std::string testing::full_name() const {
    std::string result;
    for (const testing * p = this; p->parent; p = p->parent) {
        result = result.empty() ? p->name : p->name + "." + result;
    }
    return result;
}

void testing::log(const std::string & msg) {
    if (verbose) {
        report_stream() << indent() << "  " << msg << "\n";
    }
}

void testing::apply_env() {
    if (const char * v = getenv("LLAMA_TEST_VERBOSE")) {
        verbose = std::string(v) == "1";
    }
    if (const char * s = getenv("LLAMA_TEST_STYLE")) {
        if (std::string(s) == "tree") {
            style = TESTING_STYLE_TREE;
        } else if (std::string(s) == "dots") {
            style = TESTING_STYLE_DOTS;
        }
    }
    if (const char * c = getenv("LLAMA_TEST_CAPTURE")) {
        capture_output = std::string(c) != "0";
    }
    if (const char * f = getenv("LLAMA_TEST_FILTER")) {
        set_filter(f);
    }
}

void testing::set_filter(const std::string & re) {
    state->filter       = std::regex(re);
    state->filter_text  = re;
    state->filter_tests = true;
}

void testing::skip(const std::string & reason) {
    skip_requested = true;
    skip_reason    = reason;
}

std::string testing::next_unnamed(const char * prefix) {
    return prefix + std::to_string(++state->unnamed);
}

void testing::run_guarded(const std::function<void()> & body, const char * ctx) {
    try {
        body();
    } catch (const std::exception & e) {
        ++failures;
        ++exceptions;
        report_stream() << indent() << "UNHANDLED EXCEPTION (" << ctx << "): " << e.what() << "\n";
        if (throw_exception) {
            throw;
        }
    } catch (...) {
        ++failures;
        ++exceptions;
        report_stream() << indent() << "UNHANDLED EXCEPTION (" << ctx << "): unknown\n";
        if (throw_exception) {
            throw;
        }
    }
}

testing * testing::begin(const std::string & test_name, const std::string & label) {
    std::string parent_name = full_name();
    std::string full        = parent_name.empty() ? test_name : parent_name + "." + test_name;

    // under a matched ancestor everything runs; otherwise the test must match itself or lie on the path to a literal dotted filter
    bool child_matched = matched;
    if (state->filter_tests && !child_matched) {
        if (std::regex_match(full, state->filter)) {
            child_matched = true;
        } else if (state->filter_text.compare(0, full.size() + 1, full + ".") != 0) {
            return nullptr;
        }
    }

    subtests.emplace_back(new testing(*this, test_name));
    testing * child = subtests.back().get();
    child->matched = child_matched;

    if (style == TESTING_STYLE_TREE) {
        stream() << child->indent() << label << "\n";
    }

    if (child->capture_output) {
        testing_capture * cap = state->get_capture();
        if (cap) {
            cap->push();
            child->capturing     = true;
            child->capture_start = cap->mark();
        }
    }
    return child;
}

std::string testing::end_capture() {
    if (!capturing) {
        return "";
    }
    capturing = false;
    std::string captured = state->capture->take(capture_start);
    state->capture->pop();
    return captured;
}

void testing::roll_up() {
    parent->tests      += tests + 1;
    parent->assertions += assertions;
    parent->failures   += failures;
    parent->exceptions += exceptions;
    parent->skipped    += skipped;
}

void testing::finish(const std::string & label, const std::string & extra) {
    std::string output = end_capture();

    bool was_skipped = skip_requested && failures == 0;
    if (was_skipped) {
        ++skipped;
    }

    if (style == TESTING_STYLE_DOTS) {
        if (failures > 0) {
            captured = output;
        }
        // a failed test gets a mark, a bench prints its result since that is its point, any other leaf gets a mark
        if (own_failures() > 0) {
            print_mark(own_exceptions() > 0 ? 'E' : 'F');
        } else if (!extra.empty()) {
            std::ostream & out = stream();
            state->end_marks(out);
            out << label << " (" << extra << ")\n";
        } else if (subtests.empty()) {
            print_mark(was_skipped ? 's' : '.');
        }
    } else {
        if (failures > 0 && !output.empty()) {
            print_captured(output);
        }
        print_result(label, was_skipped ? skip_reason : extra, was_skipped);
    }
    roll_up();
}

int testing::own_failures() const {
    int own = failures;
    for (const auto & child : subtests) {
        own -= child->failures;
    }
    return own;
}

int testing::own_exceptions() const {
    int own = exceptions;
    for (const auto & child : subtests) {
        own -= child->exceptions;
    }
    return own;
}

void testing::print_mark(char mark) const {
    std::ostream & out = stream();

    const char * code = mark == '.' ? "32" : mark == 's' ? "33" : "31";
    out << state->colored(std::string(1, mark), code);
    if (++state->marks_on_line >= status_column) {
        state->end_marks(out);
    }
    out.flush();
}

void testing::run_test(const std::string & test_name, const std::function<void(testing &)> & body) {
    testing * child = begin(test_name, test_name);
    if (!child) {
        return;
    }

    try {
        child->run_guarded([&] { body(*child); }, "test");
    } catch (...) {
        child->end_capture();
        child->roll_up();
        throw;
    }

    child->finish(test_name, "");
}

void testing::run_bench(const std::string & test_name, const std::function<void()> & body, int iterations) {
    std::string label = "[bench] " + test_name;

    testing * child = begin(test_name, label);
    if (!child) {
        return;
    }

    using clock = std::chrono::high_resolution_clock;

    std::chrono::microseconds duration(0);

    try {
        child->run_guarded([&] {
            for (auto i = 0; i < iterations; i++) {
                auto start = clock::now();
                body();
                duration += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - start);
            }
        }, "bench");
    } catch (...) {
        child->end_capture();
        child->roll_up();
        throw;
    }

    auto avg_elapsed   = duration.count() / iterations;
    auto avg_elapsed_s = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count() / iterations;
    auto rate = (avg_elapsed_s > 0.0) ? (1.0 / avg_elapsed_s) : 0.0;

    std::string extra =
        "n=" + std::to_string(iterations) +
        " avg=" + std::to_string(avg_elapsed) + "us" +
        " rate=" + std::to_string(int(rate)) + "/s";

    child->finish(label, extra);
}

void testing::print_result(const std::string & label, const std::string & extra, bool was_skipped) const {
    std::string line = indent() + label;

    std::string details;
    if (assertions > 0) {
        if (failures == 0) {
            details = std::to_string(assertions) + " assertion(s)";
        } else {
            details = std::to_string(failures) + " of " +
                      std::to_string(assertions) + " assertion(s) failed";
        }
    }
    if (!extra.empty()) {
        if (!details.empty()) {
            details += ", ";
        }
        details += extra;
    }

    if (!details.empty()) {
        line += " (" + details + ")";
    }

    std::string status = failures != 0 ? "[FAIL]" : (was_skipped ? "[SKIP]" : "[PASS]");

    if (line.size() + 1 < status_column) {
        line.append(status_column - line.size(), ' ');
    } else {
        line.push_back(' ');
    }

    stream() << line << status << "\n";
}

void testing::print_captured(const std::string & captured) const {
    std::ostream & out = stream();
    std::string    pad = indent() + "  ";

    out << pad << "--- captured output ---\n";
    for (size_t pos = 0; pos < captured.size();) {
        size_t nl = captured.find('\n', pos);
        std::string line = captured.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        out << pad << line << "\n";
        if (nl == std::string::npos) {
            break;
        }
        pos = nl + 1;
    }
}

// in completion order, so a subtest comes before the test that contains it
void testing::collect_failed(std::vector<const testing *> & failed) const {
    for (const auto & child : subtests) {
        child->collect_failed(failed);
    }
    if (own_failures() > 0 && parent) {
        failed.push_back(this);
    }
}

void testing::print_failures(const std::vector<const testing *> & failed) const {
    std::ostream & out = stream();

    out << state->colored(rule("FAILURES", '='), "31") << "\n";
    for (const testing * node : failed) {
        out << state->colored(rule(node->full_name(), '_'), "31") << "\n";
        out << node->report.str();
        if (!node->captured.empty()) {
            node->print_captured(node->captured);
        }
        out << "\n";
    }
}

bool testing::assert_true(const std::string & msg, bool cond) {
    ++assertions;
    if (!cond) {
        ++failures;
        std::ostream & out = report_stream();
        out << indent() << "ASSERTION FAILED";
        if (!msg.empty()) {
            out << " : " << msg;
        }
        out << "\n";
        return false;
    }
    return true;
}

int testing::summary() const {
    std::ostream & out = stream();

    std::vector<const testing *> failed;
    collect_failed(failed);

    if (style == TESTING_STYLE_DOTS) {
        state->end_marks(out);
        out << "\n";
        if (!failed.empty()) {
            print_failures(failed);
        }
    } else {
        out << "\n";
        if (!failed.empty()) {
            out << "failed tests:\n";
            for (const testing * node : failed) {
                out << "  " << node->full_name() << "\n";
            }
            out << "\n";
        }
    }
    out << "tests      : " << tests << "\n";
    out << "assertions : " << assertions << "\n";
    out << "failures   : " << failures << "\n";
    out << "exceptions : " << exceptions << "\n";
    out << "skipped    : " << skipped << "\n";
    return failures == 0 ? 0 : 1;
}
