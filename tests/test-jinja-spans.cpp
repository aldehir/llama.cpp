// compares the generation prompt marked during a single render against the
// add_generation_prompt on/off diff, over every template in models/templates

#include "chat.h"
#include "chat-auto-parser.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>



static json conversation(const std::string & name) {
    if (name == "user") {
        return json::array({
            { { "role", "user" }, { "content", "Hello" } },
        });
    }
    if (name == "system_user") {
        return json::array({
            { { "role", "system" }, { "content", "You are helpful." } },
            { { "role", "user" }, { "content", "Hello" } },
        });
    }
    if (name == "multi_turn") {
        return json::array({
            { { "role", "user" }, { "content", "Hello" } },
            { { "role", "assistant" }, { "content", "Hi there" }, { "reasoning_content", "greet back" } },
            { { "role", "user" }, { "content", "How are you?" } },
        });
    }
    if (name == "assistant_last") {
        return json::array({
            { { "role", "user" }, { "content", "Hello" } },
            { { "role", "assistant" }, { "content", "Hi there" }, { "reasoning_content", "greet back" } },
        });
    }
    if (name == "tool_result") {
        return json::array({
            { { "role", "user" }, { "content", "Weather in Paris?" } },
            { { "role", "assistant" }, { "content", "" }, { "tool_calls", json::array({
                { { "id", "call_1" }, { "type", "function" }, { "function", {
                    { "name", "get_weather" }, { "arguments", { { "city", "Paris" } } } } } },
            }) } },
            { { "role", "tool" }, { "tool_call_id", "call_1" }, { "content", "sunny" } },
        });
    }
    // long history, for timing
    json msgs = json::array({ { { "role", "system" }, { "content", "You are helpful." } } });
    for (int i = 0; i < 50; i++) {
        msgs.push_back({ { "role", "user" }, { "content", "Question number " + std::to_string(i) + " about something long enough to matter." } });
        msgs.push_back({ { "role", "assistant" }, { "content", "Answer number " + std::to_string(i) + " with a reasonably sized body of text." } });
    }
    msgs.push_back({ { "role", "user" }, { "content", "Last question" } });
    return msgs;
}

static json tools() {
    return json::array({
        { { "type", "function" }, { "function", {
            { "name", "get_weather" },
            { "description", "Get the weather" },
            { "parameters", { { "type", "object" }, { "properties", { { "city", { { "type", "string" } } } } }, { "required", json::array({ "city" }) } } },
        } } },
    });
}

// the generation prompt as found before marking: the suffix left after the common prefix of renders with
// add_generation_prompt off and on
static std::string diff_generation_prompt(const common_chat_template & tmpl, autoparser::generation_params params) {
    params.add_generation_prompt = false;
    std::string off = common_chat_template_direct_apply(tmpl, params);
    params.add_generation_prompt = true;
    std::string on  = common_chat_template_direct_apply(tmpl, params);

    size_t n = 0;
    while (n < off.size() && n < on.size() && off[n] == on[n]) {
        n++;
    }
    return on.substr(n);
}

// templates where the diff or the render without add_generation_prompt disagree with the marked render, as intended
static const std::vector<std::string> known_differences = {
    "microsoft-Phi-3.5-mini-instruct.jinja", // the common prefix swallows the leading '<' of <|assistant|>, and the
                                             // eos_token emitted without the flag is cut
    "openai-gpt-oss-120b.jinja",             // same for <|start|>, and a final assistant turn ends with <|end|>
                                             // instead of the <|return|> meant for training
    "llm-jp-llm-jp-4.1-8b-thinking.jinja",   // same
    "llama-cpp-rwkv-world.jinja",            // defaults an undefined add_generation_prompt to true, so the render
                                             // without it still has the generation prompt
    "upstage-Solar-Open-100B.jinja",         // same
    "NVIDIA-Nemotron-Nano-v2.jinja",         // a final assistant turn is rendered open inside the generation prompt
                                             // branch, continuing it is handled by continue_final_message instead
};

static std::string read_file(const std::filesystem::path & p) {
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char ** argv) {
    std::filesystem::path dir = argc > 1 ? argv[1] : "models/templates";

    std::vector<std::filesystem::path> files;
    for (const auto & e : std::filesystem::directory_iterator(dir)) {
        if (e.path().extension() == ".jinja") {
            files.push_back(e.path());
        }
    }
    std::sort(files.begin(), files.end());

    const std::vector<std::string> convs = { "user", "system_user", "multi_turn", "assistant_last", "tool_result" };

    int n_match = 0, n_known = 0, n_mismatch = 0, n_error = 0, n_prompt_mismatch = 0;
    double t_diff = 0, t_marked = 0;

    for (const auto & path : files) {
        const std::string name = path.filename().string();
        std::unique_ptr<common_chat_template> tmpl;
        try {
            tmpl = std::make_unique<common_chat_template>(read_file(path), "<s>", "</s>");
        } catch (const std::exception & e) {
            printf("SKIP  %-55s parse: %s\n", name.c_str(), e.what());
            continue;
        }

        for (const auto & conv : convs) {
            for (bool thinking : { true, false }) {
                autoparser::generation_params params;
                params.messages              = conversation(conv);
                params.add_generation_prompt = true;
                params.enable_thinking       = thinking;
                if (conv == "tool_result") {
                    params.tools = tools();
                }

                std::string expected_prompt, expected_prompt_off, expected_gen;
                common_chat_template_rendered marked, marked_off;
                try {
                    expected_prompt = common_chat_template_direct_apply(*tmpl, params);
                    expected_gen    = diff_generation_prompt(*tmpl, params);
                    marked          = common_chat_template_render(*tmpl, params);
                    params.add_generation_prompt = false;
                    expected_prompt_off = common_chat_template_direct_apply(*tmpl, params);
                    marked_off      = common_chat_template_render(*tmpl, params);
                } catch (const std::exception &) {
                    n_error++;
                    continue;
                }

                if (marked.prompt != expected_prompt) {
                    n_prompt_mismatch++;
                    printf("FAIL  %-55s %-12s thinking=%d prompt mismatch\n", name.c_str(), conv.c_str(), thinking);
                }
                // without add_generation_prompt the prompt is cut at the marker, but the generation prompt is still reported
                const bool known = std::find(known_differences.begin(), known_differences.end(), name) != known_differences.end();
                if (!known && (marked_off.prompt != expected_prompt_off || marked_off.generation_prompt != marked.generation_prompt)) {
                    n_prompt_mismatch++;
                    json d = { { "off", expected_prompt_off.substr(std::min(expected_prompt_off.size(), marked_off.prompt.size() > 80 ? marked_off.prompt.size() - 80 : 0)) },
                               { "cut", marked_off.prompt.substr(marked_off.prompt.size() > 80 ? marked_off.prompt.size() - 80 : 0) } };
                    printf("FAIL  %-55s %-12s thinking=%d off prompt mismatch %s\n", name.c_str(), conv.c_str(), thinking, d.dump().c_str());
                }
                if (marked.generation_prompt == expected_gen) {
                    n_match++;
                } else if (known) {
                    n_known++;
                } else {
                    n_mismatch++;
                    json d = { { "diff", expected_gen }, { "marked", marked.generation_prompt } };
                    printf("DIFF  %-55s %-12s thinking=%d %s\n", name.c_str(), conv.c_str(), thinking, d.dump().c_str());
                }
            }
        }

        // timing: today's 3 renders (prompt + off + on) vs 1 marked render
        autoparser::generation_params params;
        params.messages              = conversation("long");
        params.add_generation_prompt = true;
        const int iters = 20;
        try {
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; i++) {
                (void) common_chat_template_direct_apply(*tmpl, params);
                (void) diff_generation_prompt(*tmpl, params);
            }
            auto t1 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; i++) {
                (void) common_chat_template_render(*tmpl, params);
            }
            auto t2 = std::chrono::steady_clock::now();
            t_diff   += std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
            t_marked += std::chrono::duration<double, std::milli>(t2 - t1).count() / iters;
        } catch (const std::exception &) {
        }
    }

    printf("\ncases: match=%d known=%d mismatch=%d error=%d prompt_mismatch=%d\n", n_match, n_known, n_mismatch, n_error, n_prompt_mismatch);
    printf("per-request render time, 101-message history, summed over %zu templates:\n", files.size());
    printf("  3 renders (today): %8.2f ms\n", t_diff);
    printf("  1 marked render  : %8.2f ms  (%.0f%% saved)\n", t_marked, t_diff > 0 ? 100.0 * (1.0 - t_marked / t_diff) : 0.0);

    return n_mismatch == 0 && n_prompt_mismatch == 0 ? 0 : 1;
}
