// Temporary investigation test for https://github.com/ggml-org/llama.cpp/issues/25967
//
// Reported symptom: "duplicate GBNF rules with large tool lists (gpt-oss) ->
// failed to parse grammar". This test rebuilds the reporter's 17-tool request
// against the gpt-oss template and shows:
//   1. The generated grammar NEVER contains duplicate rule definitions
//      (checked over many sequential and concurrent applications).
//   2. The grammar genuinely fails llama's GBNF parser, and the culprit is the
//      agent_manager tool alone: its prompt property declares
//      "minLength": 1, "maxLength": 100000, which json-schema-to-grammar
//      converts to `char{1,100000}` - above MAX_REPETITION_THRESHOLD (2000)
//      in src/llama-grammar.cpp, so parsing throws "number of repetitions
//      exceeds sane defaults" and the server returns "failed to parse grammar".
//
// Build (after building llama-common):
//   g++ -std=c++17 -Icommon -Iinclude -Iggml/include -Ivendor -Isrc \
//       tests/test-issue-25967-repro.cpp -Lbuild/bin -lllama-common -lllama \
//       -lggml -lggml-base -Wl,-rpath,$PWD/build/bin -o test-issue-25967-repro
// Run from the repo root (needs models/templates/openai-gpt-oss-120b.jinja):
//   ./test-issue-25967-repro
#include "chat.h"

#include "llama-grammar.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <cstdio>
#include <fstream>
#include <map>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::ordered_json;


#include <utility>
#include <vector>

static const std::vector<std::pair<const char *, const char *>> ISSUE_25967_TOOLS = {

{"agent_manager", R"json({
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "mode": {"type": "string", "enum": ["worktree", "local"], "description": "Use wor"},
    "versions": {"anyOf": [{"type": "boolean"}, {"type": "null"}], "description": "Set tru"},
    "tasks": {
      "minItems": 1, "maxItems": 20, "description": "Agent", "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "prompt": {"type": "string", "description": "Initia"},
          "name": {"type": "string", "description": "Short"},
          "branchName": {"type": "string", "description": "Git br"},
          "model": {"type": "string", "description": "Option"},
          "variant": {"type": "string", "description": "Option"}
        }
      }
    },
    "action": {"type": "string", "enum": ["list", "prompt"], "description": "Operati"},
    "filter": {
      "anyOf": [
        {
          "type": "object",
          "properties": {
            "sectionIDs": {"maxItems": 100, "type": "array", "items": {"type": "string"}},
            "states": {"maxItems": 5, "type": "array", "items": {"type": "string", "enum": ["idle", "busy", "retry", "offline", "waiting"]}}
          }
        },
        {"type": "null"}
      ]
    },
    "sessionID": {"pattern": "^ses$", "type": "string"},
    "prompt": {"minLength": 1, "maxLength": 100000, "type": "string"}
  }
})json"},

{"agent_manager_models", R"json({
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "query": {"type": "string", "description": "Case-i"},
    "offset": {"minimum": 0, "type": "integer", "maximum": 1000000, "description": "Result"},
    "limit": {"minimum": 1, "type": "integer", "maximum": 1000000, "description": "Maximum"}
  }
})json"},

{"background_process", R"json({
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "action": {"type": "string", "enum": ["start", "list", "status", "logs", "stop", "restart"], "description": "Operati"},
    "command": {"type": "string", "description": "Required"},
    "id": {"type": "string", "description": "Required"},
    "workdir": {"type": "string", "description": "Working"},
    "description": {"type": "string", "description": "Short l"},
    "ready": {
      "type": "object",
      "properties": {
        "pattern": {"type": "string"},
        "port": {"minimum": -1000000, "exclusiveMinimum": 0, "type": "integer", "maximum": 1000000},
        "timeout": {"minimum": -1000000, "exclusiveMinimum": 0, "type": "integer", "maximum": 1000000}
      },
      "description": "Optional"
    },
    "inherit": {"type": "boolean", "description": "For sub"},
    "persistent": {"type": "boolean", "description": "Keep th"}
  },
  "required": ["action"]
})json"},

{"bash", R"json({
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "command": {"type": "string", "description": "The com"},
    "timeout": {"minimum": -1000000, "exclusiveMinimum": 0, "type": "integer", "maximum": 1000000, "description": "Optional"},
    "workdir": {"type": "string", "description": "The wor"},
    "description": {"type": "string", "description": "Recomm"}
  },
  "required": ["command"]
})json"},

{"edit", R"json({
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "filePath": {"type": "string", "description": "The abs"},
    "oldString": {"type": "string", "description": "The tex"},
    "newString": {"type": "string", "description": "The tex"},
    "replaceAll": {"type": "boolean", "description": "Replace"}
  },
  "required": ["filePath", "oldString", "newString"]
})json"},

{"glob", R"json({
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "pattern": {"type": "string", "description": "The glo"},
    "path": {"type": "string", "description": "The dir"}
  },
  "required": ["pattern"]
})json"},

{"grep", R"json({
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "pattern": {"type": "string", "description": "The reg"},
    "path": {"type": "string", "description": "The dir"},
    "include": {"type": "string", "description": "File pa"}
  },
  "required": ["pattern"]
})json"},

{"kilo_local_recall", R"json({
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "mode": {"type": "string", "enum": ["search", "read"], "description": "search"},
    "query": {"type": "string", "description": "Terms t"},
    "sessionID": {"type": "string", "description": "Session"},
    "limit": {"type": "number", "description": "Maximum"}
  },
  "required": ["mode"]
})json"},

{"plan_exit", R"json({
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "path": {"type": "string", "description": "Option"}
  }
})json"},

{"question", R"json({
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "questions": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "question": {"type": "string", "description": "Complet"},
          "header": {"type": "string", "description": "Very sh"},
          "options": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "label": {"type": "string", "description": "Display"},
                "description": {"type": "string", "description": "Explana"},
                "labelKey": {"type": "string", "description": "Optiona"},
                "descriptionKey": {"type": "string", "description": "Optiona"},
                "mode": {"type": "string", "description": "Optiona"}
              },
              "required": ["label", "description"]
            },
            "description": "Availab"
          },
          "multiple": {"type": "boolean", "description": "Allow s"},
          "questionKey": {"type": "string", "description": "Optiona"},
          "headerKey": {"type": "string", "description": "Optiona"}
        },
        "required": ["question", "header", "options"]
      },
      "description": "Questio"
    }
  },
  "required": ["questions"]
})json"},

{"read", R"json({
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "filePath": {"type": "string", "description": "The abs"},
    "offset": {"minimum": 0, "type": "integer", "maximum": 1000000, "description": "The lin"},
    "limit": {"minimum": 0, "type": "integer", "maximum": 1000000, "description": "The max"}
  },
  "required": ["filePath"]
})json"},

{"skill", R"json({
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "name": {"type": "string", "description": "The nam"}
  },
  "required": ["name"]
})json"},

{"suggest", R"json({
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "suggest": {"type": "string", "description": "Short s"},
    "actions": {
      "minItems": 1, "maxItems": 2, "description": "Availab", "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "label": {"type": "string", "description": "Button"},
          "description": {"type": "string", "description": "Brief"},
          "prompt": {"type": "string", "description": "Synthet"}
        },
        "required": ["label", "prompt"]
      }
    }
  },
  "required": ["suggest", "actions"]
})json"},

{"task", R"json({
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "description": {"type": "string", "description": "A short"},
    "prompt": {"type": "string", "description": "The tas"},
    "subagent_type": {"type": "string", "description": "The typ"},
    "task_id": {"type": "string", "description": "This sh"},
    "command": {"type": "string", "description": "The com"}
  },
  "required": ["description", "prompt", "subagent_type"]
})json"},

{"todowrite", R"json({
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "todos": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "content": {"type": "string", "description": "Brief d"},
          "status": {"type": "string", "description": "Current"},
          "priority": {"type": "string", "description": "Priorit"}
        },
        "required": ["content", "status", "priority"]
      },
      "description": "The upd"
    }
  },
  "required": ["todos"]
})json"},

{"webfetch", R"json({
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "url": {"type": "string", "description": "The URL"},
    "format": {"type": "string", "enum": ["text", "markdown", "html"], "description": "The for", "default": "markdown"},
    "timeout": {"type": "number", "description": "Optiona"}
  },
  "required": ["url"]
})json"},

{"write", R"json({
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "content": {"type": "string", "description": "The con"},
    "filePath": {"type": "string", "description": "The abs"}
  },
  "required": ["content", "filePath"]
})json"},

};


static std::string read_file(const std::string & path) {
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "failed to open %s\n", path.c_str());
        exit(1);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// opencode-like tool set (17 tools), including the two tools named in the
// issue (todowrite, question) with nested array-of-object schemas.
static std::vector<common_chat_tool> make_tools() {
    std::vector<common_chat_tool> tools;
    for (const auto & [name, params] : ISSUE_25967_TOOLS) {
        tools.push_back({name, std::string("Tool ") + name, params});
    }
    return tools;
}

struct dup_report {
    std::vector<std::string> lines;
};

// Returns duplicated "name ::= ..." definitions (same name defined 2+ times).
static dup_report find_duplicates(const std::string & grammar) {
    dup_report rep;
    std::map<std::string, std::vector<std::string>> defs;
    std::istringstream ss(grammar);
    std::string line;
    static const std::regex def_re(R"(^([A-Za-z0-9_-]+) ::= )");
    while (std::getline(ss, line)) {
        std::smatch m;
        if (std::regex_search(line, m, def_re)) {
            defs[m[1].str()].push_back(line);
        }
    }
    for (const auto & [name, lines] : defs) {
        if (lines.size() > 1) {
            for (const auto & l : lines) {
                rep.lines.push_back(l);
            }
        }
    }
    return rep;
}

int main(int argc, char ** argv) {
    const char * tmpl_path = argc > 1 ? argv[1] : "models/templates/openai-gpt-oss-120b.jinja";
    const int    iters     = argc > 2 ? atoi(argv[2]) : 50;
    const int    threads   = argc > 3 ? atoi(argv[3]) : 4;

    auto tmpl_src = read_file(tmpl_path);
    common_chat_templates_ptr tmpls(common_chat_templates_init(/* model= */ nullptr, tmpl_src));

    common_chat_templates_inputs inputs;
    inputs.use_jinja = true;
    inputs.tools = make_tools();
    inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    common_chat_msg msg;
    msg.role = "user";
    msg.content = "hello";
    inputs.messages = {msg};

    std::string baseline;

    // Phase 0: isolate which tool breaks grammar parsing
    printf("== phase 0: per-tool grammar parse ==\n");
    for (const auto & tool : inputs.tools) {
        auto single = inputs;
        single.tools = {tool};
        auto params = common_chat_templates_apply(tmpls.get(), single);
        llama_grammar_parser gparser;
        bool ok = gparser.parse(params.grammar.c_str());
        printf("  %-22s parse=%s\n", tool.name.c_str(), ok ? "OK" : "FAILED");
    }
    {
        auto rest = inputs;
        rest.tools.clear();
        for (const auto & tool : inputs.tools) {
            if (tool.name != "agent_manager") {
                rest.tools.push_back(tool);
            }
        }
        auto params = common_chat_templates_apply(tmpls.get(), rest);
        llama_grammar_parser gparser;
        bool ok = gparser.parse(params.grammar.c_str());
        printf("  all 16 except agent_manager: parse=%s\n\n", ok ? "OK" : "FAILED");
    }

    // Phase 1: sequential determinism + duplicate check
    printf("== phase 1: %d sequential applications ==\n", iters);
    int seq_dups = 0, seq_nondet = 0;
    for (int i = 0; i < iters; i++) {
        auto params = common_chat_templates_apply(tmpls.get(), inputs);
        if (i == 0) {
            baseline = params.grammar;
            printf("grammar: %zu bytes, lazy path\n", params.grammar.size());
            llama_grammar_parser gparser;
            bool ok = gparser.parse(params.grammar.c_str());
            printf("llama grammar parse: %s (%zu rules)\n", ok ? "OK" : "FAILED", gparser.rules.size());
        } else if (params.grammar != baseline) {
            seq_nondet++;
            printf("iteration %d: grammar DIFFERS from baseline!\n", i);
        }
        auto rep = find_duplicates(params.grammar);
        if (!rep.lines.empty()) {
            seq_dups++;
            printf("iteration %d: DUPLICATE definitions:\n", i);
            for (const auto & l : rep.lines) {
                printf("  %s\n", l.c_str());
            }
        }
    }
    printf("sequential: %d/%d runs with duplicates, %d nondeterministic\n\n", seq_dups, iters, seq_nondet);

    // Phase 2: concurrent applications sharing tmpls (mirrors llama-server
    // HTTP worker threads sharing common_chat_templates)
    printf("== phase 2: %d threads x %d concurrent applications ==\n", threads, iters);
    std::atomic<int> conc_dups{0}, conc_nondet{0};
    std::mutex print_mtx;
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; t++) {
        pool.emplace_back([&, t]() {
            for (int i = 0; i < iters; i++) {
                auto params = common_chat_templates_apply(tmpls.get(), inputs);
                if (params.grammar != baseline) {
                    conc_nondet++;
                    std::lock_guard<std::mutex> lock(print_mtx);
                    printf("thread %d iteration %d: grammar DIFFERS from baseline\n", t, i);
                }
                auto rep = find_duplicates(params.grammar);
                if (!rep.lines.empty()) {
                    conc_dups++;
                    std::lock_guard<std::mutex> lock(print_mtx);
                    printf("thread %d iteration %d: DUPLICATE definitions:\n", t, i);
                    for (const auto & l : rep.lines) {
                        printf("  %s\n", l.c_str());
                    }
                }
            }
        });
    }
    for (auto & th : pool) {
        th.join();
    }
    printf("concurrent: %d runs with duplicates, %d nondeterministic\n", conc_dups.load(), conc_nondet.load());

    return (seq_dups || conc_dups.load()) ? 42 : 0;
}
