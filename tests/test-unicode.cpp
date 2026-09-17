#include "../src/unicode.h"

#include "testing.h"

#include <cstdio>
#include <string>
#include <vector>

static void test_regex_split(testing & t) {
    t.test("symbol run then word", [](testing & t) {
        const std::vector<std::string> regex_exprs = {
            "[~][A-Za-z]+| ?[\\p{S}]+|\\s+",
        };
        const std::vector<std::string> expected = { " ~", "foo" };
        const auto actual = unicode_regex_split(" ~foo", regex_exprs, false);

        auto join = [](const std::vector<std::string> & pieces) {
            std::string out;
            for (const auto & piece : pieces) {
                out += " [" + piece + "]";
            }
            return out;
        };

        if (actual != expected) {
            fprintf(stderr, "unexpected split:%s\n", join(actual).c_str());
        }
        t.assert_true("\" ~foo\" splits into" + join(expected) + ", got" + join(actual), actual == expected);
    });
}

int main(int argc, char ** argv) {
    testing t;
    t.capture_output = true;
    t.apply_env();
    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("regex_split", test_regex_split);

    return t.summary();
}
