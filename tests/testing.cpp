#include "testing.h"

#include "log.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <mutex>
#include <regex>
#include <streambuf>
#include <thread>

#ifdef _WIN32
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
#else
int fd_dup(int fd)                               { return dup(fd); }
int fd_dup2(int from, int to)                    { return dup2(from, to); }
int fd_close(int fd)                             { return close(fd); }
int fd_pipe(int fds[2])                          { return pipe(fds); }
int fd_read(int fd, char * buf, size_t n)        { return (int) read(fd, buf, n); }
int fd_write(int fd, const char * buf, size_t n) { return (int) write(fd, buf, n); }
#endif

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
    bool           filter_tests = false;
    int            unnamed      = 0;

    std::unique_ptr<testing_capture> capture;

    explicit testing_state(std::ostream & os) : out(os) {}

    bool out_is_console() const {
        return &out == &std::cout || &out == &std::cerr;
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

int testing::depth() const {
    int d = 0;
    for (const testing * p = parent; p; p = p->parent) {
        ++d;
    }
    return d;
}

std::string testing::indent() const {
    int d = depth();
    if (d <= 1) {
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
        stream() << indent() << "  " << msg << "\n";
    }
}

void testing::set_filter(const std::string & re) {
    state->filter       = std::regex(re);
    state->filter_tests = true;
}

void testing::skip(const std::string & reason) {
    skip_requested = true;
    skip_reason    = reason;
}

std::string testing::next_unnamed(const char * prefix) {
    return prefix + std::to_string(++state->unnamed);
}

bool testing::should_run(const std::string & full) const {
    if (state->filter_tests) {
        if (!std::regex_match(full, state->filter)) {
            return false;
        }
    }
    return true;
}

void testing::run_guarded(const std::function<void()> & body, const char * ctx) {
    try {
        body();
    } catch (const std::exception & e) {
        ++failures;
        ++exceptions;
        stream() << indent() << "UNHANDLED EXCEPTION (" << ctx << "): " << e.what() << "\n";
        if (throw_exception) {
            throw;
        }
    } catch (...) {
        ++failures;
        ++exceptions;
        stream() << indent() << "UNHANDLED EXCEPTION (" << ctx << "): unknown\n";
        if (throw_exception) {
            throw;
        }
    }
}

testing * testing::begin(const std::string & test_name, const std::string & label) {
    std::string parent_name = full_name();
    if (!should_run(parent_name.empty() ? test_name : parent_name + "." + test_name)) {
        return nullptr;
    }

    subtests.emplace_back(new testing(*this, test_name));
    testing * child = subtests.back().get();

    stream() << child->indent() << label << "\n";

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
    std::string captured = end_capture();

    bool was_skipped = skip_requested && failures == 0;
    if (was_skipped) {
        ++skipped;
    }

    if (failures > 0 && !captured.empty()) {
        print_captured(captured);
    }
    print_result(label, was_skipped ? skip_reason : extra, was_skipped);
    roll_up();
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

void testing::collect_failed(std::vector<std::string> & names) const {
    int own = failures;
    for (const auto & child : subtests) {
        own -= child->failures;
    }
    if (own > 0 && parent) {
        names.push_back(full_name());
    }
    for (const auto & child : subtests) {
        child->collect_failed(names);
    }
}

bool testing::assert_true(const std::string & msg, bool cond) {
    ++assertions;
    if (!cond) {
        ++failures;
        std::ostream & out = stream();
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

    std::vector<std::string> failed;
    collect_failed(failed);

    out << "\n";
    if (!failed.empty()) {
        out << "failed tests:\n";
        for (const auto & failed_name : failed) {
            out << "  " << failed_name << "\n";
        }
        out << "\n";
    }
    out << "tests      : " << tests << "\n";
    out << "assertions : " << assertions << "\n";
    out << "failures   : " << failures << "\n";
    out << "exceptions : " << exceptions << "\n";
    out << "skipped    : " << skipped << "\n";
    return failures == 0 ? 0 : 1;
}
