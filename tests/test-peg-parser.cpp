#include <cstdlib>
#include <string>
#include <iostream>

#include "peg-parser/tests.h"

int main(int argc, char *argv[]) {
    testing t(std::cout);
    if (argc >= 2) {
        t.set_filter(argv[1]);
    }

    t.capture_output = true;
    t.apply_env();

    t.test("basic", test_basic);
    t.test("unicode", test_unicode);
    t.test("json", test_json_parser);
    t.test("gbnf", test_gbnf_generation);
    t.test("serialization", test_json_serialization);
    t.test("python-dict", test_python_dict_parser);

    return t.summary();
}
