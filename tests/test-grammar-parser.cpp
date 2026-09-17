#include "llama.h"

// TODO: shold not include libllama sources
#include "../src/llama-grammar.h"

#include "testing.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <utility>
#include <vector>

static const char * type_str(llama_gretype type) {
    switch (type) {
        case LLAMA_GRETYPE_CHAR: return "LLAMA_GRETYPE_CHAR";
        case LLAMA_GRETYPE_CHAR_NOT: return "LLAMA_GRETYPE_CHAR_NOT";
        case LLAMA_GRETYPE_CHAR_ALT: return "LLAMA_GRETYPE_CHAR_ALT";
        case LLAMA_GRETYPE_CHAR_RNG_UPPER: return "LLAMA_GRETYPE_CHAR_RNG_UPPER";
        case LLAMA_GRETYPE_RULE_REF: return "LLAMA_GRETYPE_RULE_REF";
        case LLAMA_GRETYPE_ALT: return "LLAMA_GRETYPE_ALT";
        case LLAMA_GRETYPE_END: return "LLAMA_GRETYPE_END";
        default: return "?";
    }
}

static void verify_parsing(testing & t, const char *grammar_bytes, const std::vector<std::pair<std::string, uint32_t>> expected, const std::vector<llama_grammar_element> &expected_rules) {
    uint32_t index = 0;
    llama_grammar_parser parsed_grammar;
    parsed_grammar.parse(grammar_bytes);

    std::map<uint32_t, std::string> symbol_names;
    for (auto it = parsed_grammar.symbol_ids.begin(); it != parsed_grammar.symbol_ids.end(); ++it) {
        symbol_names[it->second] = it->first;
    }

    auto print_all = [&]() {
        fprintf(stderr, "    verify_parsing(R\"\"\"(%s)\"\"\", {\n", grammar_bytes);
        for (auto it = parsed_grammar.symbol_ids.begin(); it != parsed_grammar.symbol_ids.end(); ++it) {
            fprintf(stderr, "        {\"%s\", %u},\n", it->first.c_str(), it->second);
        }
        fprintf(stderr, "    }, {\n");
        for (size_t i_rule = 0; i_rule < parsed_grammar.rules.size(); i_rule++) {
            fprintf(stderr, "        // %s (index %zu)\n", symbol_names[i_rule].c_str(), i_rule);
            auto & rule = parsed_grammar.rules[i_rule];
            for (uint32_t i = 0; i < rule.size(); i++) {
                std::string rule_str;
                fprintf(stderr, "        {%s, ", type_str(rule[i].type));
                if (rule[i].type == LLAMA_GRETYPE_CHAR || rule[i].type == LLAMA_GRETYPE_CHAR_ALT ||
                    rule[i].type == LLAMA_GRETYPE_CHAR_NOT || rule[i].type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
                    char c = rule[i].value;
                    if (c == '\n') {
                        fprintf(stderr, "'\\n'");
                    } else if (c == '\t') {
                        fprintf(stderr, "'\\t'");
                    } else if (c == '\r') {
                        fprintf(stderr, "'\\r'");
                    } else if (c == '\0') {
                        fprintf(stderr, "'\\0'");
                    } else {
                        fprintf(stderr, "'%c'", c);
                    }
                } else if (rule[i].type == LLAMA_GRETYPE_RULE_REF) {
                    fprintf(stderr, "/* %s */ %u", symbol_names[rule[i].value].c_str(), rule[i].value);
                } else {
                    fprintf(stderr, "%u", rule[i].value);
                }
                fprintf(stderr, "},\n");
            }
        }
        fprintf(stderr, "    });\n");
    };

    if (getenv("TEST_GRAMMAR_PARSER_PRINT_ALL")) {
        print_all();
        fprintf(stderr, "\n");
        return;
    }

    fprintf(stderr, "Testing grammar:%s\n", grammar_bytes);

    if (!t.assert_equal("symbol count", expected.size(), parsed_grammar.symbol_ids.size())) {
        fprintf(stderr, "Code to update expectation (set TEST_GRAMMAR_PARSER_PRINT_ALL=1 to print all):\n");
        print_all();
        return;
    }

    for (auto it = parsed_grammar.symbol_ids.begin(); it != parsed_grammar.symbol_ids.end(); ++it)
    {
        std::string key = it->first;
        uint32_t value = it->second;
        std::pair<std::string, uint32_t> expected_pair = expected[index];

        // pretty print error message before asserting
        if (expected_pair.first != key || expected_pair.second != value)
        {
            fprintf(stderr, "index: %u\n", index);
            fprintf(stderr, "expected_pair: %s, %u\n", expected_pair.first.c_str(), expected_pair.second);
            fprintf(stderr, "actual_pair: %s, %u\n", key.c_str(), value);
            fprintf(stderr, "expected_pair != actual_pair\n");
            fprintf(stderr, "Code to update expectation (set TEST_GRAMMAR_PARSER_PRINT_ALL=1 to print all):\n");
            print_all();
        }

        if (!t.assert_true("symbol " + std::to_string(index) + " is " + expected_pair.first + " = " + std::to_string(expected_pair.second) + ", got " + key + " = " + std::to_string(value),
                           expected_pair.first == key && expected_pair.second == value)) {
            return;
        }

        index++;
    }

    index = 0;
    for (auto rule : parsed_grammar.rules)
    {
        // compare rule to expected rule
        for (uint32_t i = 0; i < rule.size(); i++)
        {
            llama_grammar_element element = rule[i];

            // fail on a surplus element instead of reading past the expectation
            if (index >= expected_rules.size()) {
                t.assert_true("element " + std::to_string(index) + " is not in the expectation", false);
                print_all();
                return;
            }

            llama_grammar_element expected_element = expected_rules[index];

            // pretty print error message before asserting
            if (expected_element.type != element.type || expected_element.value != element.value)
            {
                fprintf(stderr, "index: %u\n", index);
                fprintf(stderr, "expected_element: %s, %u\n", type_str(expected_element.type), expected_element.value);
                fprintf(stderr, "actual_element: %s, %u\n", type_str(element.type), element.value);
                fprintf(stderr, "expected_element != actual_element\n");
                fprintf(stderr, "all elements:\n");
                fprintf(stderr, "Code to update expectation (set TEST_GRAMMAR_PARSER_PRINT_ALL=1 to print all):\n");
                print_all();
            }

            if (!t.assert_true("element " + std::to_string(index) + " is " + type_str(expected_element.type) + " " + std::to_string(expected_element.value) + ", got " + type_str(element.type) + " " + std::to_string(element.value),
                               expected_element.type == element.type && expected_element.value == element.value)) {
                return;
            }
            index++;
        }
    }
}

static void verify_failure(testing & t, const char * grammar_bytes) {
    fprintf(stderr, "Testing expected failure:%s\n", grammar_bytes);
    llama_grammar_parser result;
    result.parse(grammar_bytes);
    t.assert_true(std::string("grammar should fail to parse:") + grammar_bytes, result.rules.empty());
}

static void test_failures(testing & t) {
    t.test("empty repetition bounds", [](testing & t) {
        verify_failure(t, R"""(
            root ::= "a"{,}"
        )""");
    });

    t.test("nested repetition overflow", [](testing & t) {
        verify_failure(t, R"""(
            root ::= (((((([^x]*){0,99}){0,99}){0,99}){0,99}){0,99}){0,99}
        )""");
    });

    t.test("missing repetition minimum", [](testing & t) {
        verify_failure(t, R"""(
            root ::= "a"{,10}"
        )""");
    });

    t.test("exact repetition too large", [](testing & t) {
        verify_failure(t, R"""(
            root ::= "a"{5000}
        )""");
    });

    t.test("min repetition too large", [](testing & t) {
        verify_failure(t, R"""(
            root ::= "a"{5000,}
        )""");
    });

    t.test("min max repetition too large", [](testing & t) {
        verify_failure(t, R"""(
            root ::= "a"{5000,6000}
        )""");
    });
}

static void test_parsing(testing & t) {
    t.test("max repetition at limit", [](testing & t) {
        verify_parsing(t, R"""(
            root ::= "a"{0,5000}
        )""", {
            {"root", 0},
            {"root_1", 1},
        }, {
            // root (index 0)
            {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
            {LLAMA_GRETYPE_END, 0},
            // root_1 (index 1)
            {LLAMA_GRETYPE_CHAR, 'a'},
            {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
        });
    });

    t.test("min and max repetition at limit", [](testing & t) {
        verify_parsing(t, R"""(
            root ::= "a"{3,5000}
        )""", {
            {"root", 0},
            {"root_1", 1},
        }, {
            // root (index 0)
            {LLAMA_GRETYPE_CHAR, 'a'},
            {LLAMA_GRETYPE_CHAR, 'a'},
            {LLAMA_GRETYPE_CHAR, 'a'},
            {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
            {LLAMA_GRETYPE_END, 0},
            // root_1 (index 1)
            {LLAMA_GRETYPE_CHAR, 'a'},
            {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
        });
    });

    t.test("single char", [](testing & t) {
        verify_parsing(t, R"""(
            root  ::= "a"
        )""", {
            {"root", 0},
        }, {
            // root (index 0)
            {LLAMA_GRETYPE_CHAR, 'a'},
            {LLAMA_GRETYPE_END, 0},
        });
    });

    t.test("alternates and char classes", [](testing & t) {
        verify_parsing(t, R"""(
            root  ::= "a" | [bdx-z] | [^1-3]
        )""", {
            {"root", 0},
        }, {
            // root (index 0)
            {LLAMA_GRETYPE_CHAR, 'a'},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_CHAR, 'b'},
            {LLAMA_GRETYPE_CHAR_ALT, 'd'},
            {LLAMA_GRETYPE_CHAR_ALT, 'x'},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 'z'},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_CHAR_NOT, '1'},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, '3'},
            {LLAMA_GRETYPE_END, 0},
        });
    });

    t.test("plus on rule ref", [](testing & t) {
        verify_parsing(t, R"""(
            root  ::= a+
            a     ::= "a"
        )""", {
            {"a", 1},
            {"root", 0},
            {"root_2", 2},
        }, {
            // root (index 0)
            {LLAMA_GRETYPE_RULE_REF, /* a */ 1},
            {LLAMA_GRETYPE_RULE_REF, /* root_2 */ 2},
            {LLAMA_GRETYPE_END, 0},
            // a (index 1)
            {LLAMA_GRETYPE_CHAR, 'a'},
            {LLAMA_GRETYPE_END, 0},
            // root_2 (index 2)
            {LLAMA_GRETYPE_RULE_REF, /* a */ 1},
            {LLAMA_GRETYPE_RULE_REF, /* root_2 */ 2},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
        });
    });

    t.test("plus on char", [](testing & t) {
        verify_parsing(t, R"""(
            root  ::= "a"+
        )""", {
            {"root", 0},
            {"root_1", 1},
        }, {
            // root (index 0)
            {LLAMA_GRETYPE_CHAR, 'a'},
            {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
            {LLAMA_GRETYPE_END, 0},
            // root_1 (index 1)
            {LLAMA_GRETYPE_CHAR, 'a'},
            {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
        });
    });

    t.test("optional rule ref", [](testing & t) {
        verify_parsing(t, R"""(
            root  ::= a?
            a     ::= "a"
        )""", {
            {"a", 1},
            {"root", 0},
            {"root_2", 2},
        }, {
            // root (index 0)
            {LLAMA_GRETYPE_RULE_REF, /* root_2 */ 2},
            {LLAMA_GRETYPE_END, 0},
            // a (index 1)
            {LLAMA_GRETYPE_CHAR, 'a'},
            {LLAMA_GRETYPE_END, 0},
            // root_2 (index 2)
            {LLAMA_GRETYPE_RULE_REF, /* a */ 1},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
        });
    });

    t.test("optional char", [](testing & t) {
        verify_parsing(t, R"""(
            root  ::= "a"?
        )""", {
            {"root", 0},
            {"root_1", 1},
        }, {
            // root (index 0)
            {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
            {LLAMA_GRETYPE_END, 0},
            // root_1 (index 1)
            {LLAMA_GRETYPE_CHAR, 'a'},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
        });
    });

    t.test("star on rule ref", [](testing & t) {
        verify_parsing(t, R"""(
            root  ::= a*
            a     ::= "a"
        )""", {
            {"a", 1},
            {"root", 0},
            {"root_2", 2},
        }, {
            // root (index 0)
            {LLAMA_GRETYPE_RULE_REF, /* root_2 */ 2},
            {LLAMA_GRETYPE_END, 0},
            // a (index 1)
            {LLAMA_GRETYPE_CHAR, 'a'},
            {LLAMA_GRETYPE_END, 0},
            // root_2 (index 2)
            {LLAMA_GRETYPE_RULE_REF, /* a */ 1},
            {LLAMA_GRETYPE_RULE_REF, /* root_2 */ 2},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
        });
    });

    t.test("star on char", [](testing & t) {
        verify_parsing(t, R"""(
            root  ::= "a"*
        )""", {
            {"root", 0},
            {"root_1", 1},
        }, {
            // root (index 0)
            {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
            {LLAMA_GRETYPE_END, 0},
            // root_1 (index 1)
            {LLAMA_GRETYPE_CHAR, 'a'},
            {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
        });
    });

    t.test("exact repetition", [](testing & t) {
        verify_parsing(t, R"""(
            root  ::= "a"{2}
        )""", {
            {"root", 0},
        }, {
            // root (index 0)
            {LLAMA_GRETYPE_CHAR, 'a'},
            {LLAMA_GRETYPE_CHAR, 'a'},
            {LLAMA_GRETYPE_END, 0},
        });
    });

    t.test("min repetition", [](testing & t) {
        verify_parsing(t, R"""(
            root  ::= "a"{2,}
        )""", {
            {"root", 0},
            {"root_1", 1},
        }, {
            // root (index 0)
            {LLAMA_GRETYPE_CHAR, 'a'},
            {LLAMA_GRETYPE_CHAR, 'a'},
            {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
            {LLAMA_GRETYPE_END, 0},
            // root_1 (index 1)
            {LLAMA_GRETYPE_CHAR, 'a'},
            {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
        });
    });

    t.test("exact repetition with space", [](testing & t) {
        verify_parsing(t, R"""(
            root  ::= "a"{ 4}
        )""", {
            {"root", 0},
        }, {
            // root (index 0)
            {LLAMA_GRETYPE_CHAR, 'a'},
            {LLAMA_GRETYPE_CHAR, 'a'},
            {LLAMA_GRETYPE_CHAR, 'a'},
            {LLAMA_GRETYPE_CHAR, 'a'},
            {LLAMA_GRETYPE_END, 0},
        });
    });

    t.test("min max repetition", [](testing & t) {
        verify_parsing(t, R"""(
            root  ::= "a"{2,4}
        )""", {
            {"root", 0},
            {"root_1", 1},
            {"root_2", 2},
        }, {
            // root (index 0)
            {LLAMA_GRETYPE_CHAR, 'a'},
            {LLAMA_GRETYPE_CHAR, 'a'},
            {LLAMA_GRETYPE_RULE_REF, /* root_2 */ 2},
            {LLAMA_GRETYPE_END, 0},
            // root_1 (index 1)
            {LLAMA_GRETYPE_CHAR, 'a'},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
            // root_2 (index 2)
            {LLAMA_GRETYPE_CHAR, 'a'},
            {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
        });
    });

    t.test("expression grammar", [](testing & t) {
        verify_parsing(t, R"""(
            root  ::= (expr "=" term "\n")+
            expr  ::= term ([-+*/] term)*
            term  ::= [0-9]+
        )""", {
            {"expr", 2},
            {"expr_5", 5},
            {"expr_6", 6},
            {"root", 0},
            {"root_1", 1},
            {"root_4", 4},
            {"term", 3},
            {"term_7", 7},
        }, {
            // root (index 0)
            {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
            {LLAMA_GRETYPE_RULE_REF, /* root_4 */ 4},
            {LLAMA_GRETYPE_END, 0},
            // root_1 (index 1)
            {LLAMA_GRETYPE_RULE_REF, /* expr */ 2},
            {LLAMA_GRETYPE_CHAR, '='},
            {LLAMA_GRETYPE_RULE_REF, /* term */ 3},
            {LLAMA_GRETYPE_CHAR, '\n'},
            {LLAMA_GRETYPE_END, 0},
            // expr (index 2)
            {LLAMA_GRETYPE_RULE_REF, /* term */ 3},
            {LLAMA_GRETYPE_RULE_REF, /* expr_6 */ 6},
            {LLAMA_GRETYPE_END, 0},
            // term (index 3)
            {LLAMA_GRETYPE_CHAR, '0'},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, '9'},
            {LLAMA_GRETYPE_RULE_REF, /* term_7 */ 7},
            {LLAMA_GRETYPE_END, 0},
            // root_4 (index 4)
            {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
            {LLAMA_GRETYPE_RULE_REF, /* root_4 */ 4},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
            // expr_5 (index 5)
            {LLAMA_GRETYPE_CHAR, '-'},
            {LLAMA_GRETYPE_CHAR_ALT, '+'},
            {LLAMA_GRETYPE_CHAR_ALT, '*'},
            {LLAMA_GRETYPE_CHAR_ALT, '/'},
            {LLAMA_GRETYPE_RULE_REF, /* term */ 3},
            {LLAMA_GRETYPE_END, 0},
            // expr_6 (index 6)
            {LLAMA_GRETYPE_RULE_REF, /* expr_5 */ 5},
            {LLAMA_GRETYPE_RULE_REF, /* expr_6 */ 6},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
            // term_7 (index 7)
            {LLAMA_GRETYPE_CHAR, '0'},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, '9'},
            {LLAMA_GRETYPE_RULE_REF, /* term_7 */ 7},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
        });
    });

    t.test("expression grammar with whitespace", [](testing & t) {
        verify_parsing(t, R"""(
            root  ::= (expr "=" ws term "\n")+
            expr  ::= term ([-+*/] term)*
            term  ::= ident | num | "(" ws expr ")" ws
            ident ::= [a-z] [a-z0-9_]* ws
            num   ::= [0-9]+ ws
            ws    ::= [ \t\n]*
        )""", {
            {"expr", 2},
            {"expr_6", 6},
            {"expr_7", 7},
            {"ident", 8},
            {"ident_10", 10},
            {"num", 9},
            {"num_11", 11},
            {"root", 0},
            {"root_1", 1},
            {"root_5", 5},
            {"term", 4},
            {"ws", 3},
            {"ws_12", 12},
        }, {
            // root (index 0)
            {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
            {LLAMA_GRETYPE_RULE_REF, /* root_5 */ 5},
            {LLAMA_GRETYPE_END, 0},
            // root_1 (index 1)
            {LLAMA_GRETYPE_RULE_REF, /* expr */ 2},
            {LLAMA_GRETYPE_CHAR, '='},
            {LLAMA_GRETYPE_RULE_REF, /* ws */ 3},
            {LLAMA_GRETYPE_RULE_REF, /* term */ 4},
            {LLAMA_GRETYPE_CHAR, '\n'},
            {LLAMA_GRETYPE_END, 0},
            // expr (index 2)
            {LLAMA_GRETYPE_RULE_REF, /* term */ 4},
            {LLAMA_GRETYPE_RULE_REF, /* expr_7 */ 7},
            {LLAMA_GRETYPE_END, 0},
            // ws (index 3)
            {LLAMA_GRETYPE_RULE_REF, /* ws_12 */ 12},
            {LLAMA_GRETYPE_END, 0},
            // term (index 4)
            {LLAMA_GRETYPE_RULE_REF, /* ident */ 8},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_RULE_REF, /* num */ 9},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_CHAR, '('},
            {LLAMA_GRETYPE_RULE_REF, /* ws */ 3},
            {LLAMA_GRETYPE_RULE_REF, /* expr */ 2},
            {LLAMA_GRETYPE_CHAR, ')'},
            {LLAMA_GRETYPE_RULE_REF, /* ws */ 3},
            {LLAMA_GRETYPE_END, 0},
            // root_5 (index 5)
            {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
            {LLAMA_GRETYPE_RULE_REF, /* root_5 */ 5},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
            // expr_6 (index 6)
            {LLAMA_GRETYPE_CHAR, '-'},
            {LLAMA_GRETYPE_CHAR_ALT, '+'},
            {LLAMA_GRETYPE_CHAR_ALT, '*'},
            {LLAMA_GRETYPE_CHAR_ALT, '/'},
            {LLAMA_GRETYPE_RULE_REF, /* term */ 4},
            {LLAMA_GRETYPE_END, 0},
            // expr_7 (index 7)
            {LLAMA_GRETYPE_RULE_REF, /* expr_6 */ 6},
            {LLAMA_GRETYPE_RULE_REF, /* expr_7 */ 7},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
            // ident (index 8)
            {LLAMA_GRETYPE_CHAR, 'a'},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 'z'},
            {LLAMA_GRETYPE_RULE_REF, /* ident_10 */ 10},
            {LLAMA_GRETYPE_RULE_REF, /* ws */ 3},
            {LLAMA_GRETYPE_END, 0},
            // num (index 9)
            {LLAMA_GRETYPE_CHAR, '0'},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, '9'},
            {LLAMA_GRETYPE_RULE_REF, /* num_11 */ 11},
            {LLAMA_GRETYPE_RULE_REF, /* ws */ 3},
            {LLAMA_GRETYPE_END, 0},
            // ident_10 (index 10)
            {LLAMA_GRETYPE_CHAR, 'a'},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 'z'},
            {LLAMA_GRETYPE_CHAR_ALT, '0'},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, '9'},
            {LLAMA_GRETYPE_CHAR_ALT, '_'},
            {LLAMA_GRETYPE_RULE_REF, /* ident_10 */ 10},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
            // num_11 (index 11)
            {LLAMA_GRETYPE_CHAR, '0'},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, '9'},
            {LLAMA_GRETYPE_RULE_REF, /* num_11 */ 11},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
            // ws_12 (index 12)
            {LLAMA_GRETYPE_CHAR, ' '},
            {LLAMA_GRETYPE_CHAR_ALT, '\t'},
            {LLAMA_GRETYPE_CHAR_ALT, '\n'},
            {LLAMA_GRETYPE_RULE_REF, /* ws_12 */ 12},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
        });
    });

    // <[1000]> = "<think>"
    // <[1001]> = "</think>"
    t.test("token elements", [](testing & t) {
        verify_parsing(t, R"""(
            root  ::= <[1000]> !<[1001]> <[1001]>
        )""", {
            {"root", 0}
        }, {
            // root (index 0)
            {LLAMA_GRETYPE_TOKEN, 1000},
            {LLAMA_GRETYPE_TOKEN_NOT, 1001},
            {LLAMA_GRETYPE_TOKEN, 1001},
            {LLAMA_GRETYPE_END, 0},
        });
    });
}

int main(int argc, char ** argv) {
    testing t;
    t.capture_output = true;
    t.apply_env();
    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("failure", test_failures);
    t.test("parsing", test_parsing);

    return t.summary();
}
