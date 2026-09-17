#include "ggml.h"
#include "llama.h"

#include "testing.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern struct llama_sampler * llama_sampler_init_dry_testing(float dry_multiplier, float dry_base, int32_t dry_allowed_length, int32_t dry_penalty_last_n, const std::vector<std::vector<llama_token>>& seq_breakers);

static void dump(const llama_token_data_array * cur_p) {
    for (size_t i = 0; i < cur_p->size; i++) {
        printf("%d: %f (%f)\n", cur_p->data[i].id, cur_p->data[i].p, cur_p->data[i].logit);
    }
}

#define DUMP(__cur_p) do { printf("%s:%d (%s)\n", __FILE__, __LINE__, __func__); dump((__cur_p)); printf("-\n"); } while(0)

static std::string fmt(float x) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%g", x);
    return buf;
}

static std::string fmt(const std::vector<float> & xs) {
    std::string out = "[";
    for (size_t i = 0; i < xs.size(); i++) {
        out += (i ? " " : "") + fmt(xs[i]);
    }
    return out + "]";
}

static std::string fmt(const std::vector<llama_token> & xs) {
    std::string out = "[";
    for (size_t i = 0; i < xs.size(); i++) {
        out += (i ? " " : "") + std::to_string(xs[i]);
    }
    return out + "]";
}

static std::string fmt(const std::vector<std::vector<llama_token>> & xss) {
    std::string out = "[";
    for (size_t i = 0; i < xss.size(); i++) {
        out += (i ? " " : "") + fmt(xss[i]);
    }
    return out + "]";
}

struct sampler_tester {
    sampler_tester(size_t n_vocab) {
        cur.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < (llama_token)n_vocab; token_id++) {
            const float logit = logf(token_id);
            cur.emplace_back(llama_token_data{token_id, logit, 0.0f});
        }

        cur_p = llama_token_data_array { cur.data(), cur.size(), -1, false };
    }

    sampler_tester(const std::vector<float> & probs, const std::vector<float> & probs_expected) : probs_expected(probs_expected) {
        cur.reserve(probs.size());
        for (llama_token token_id = 0; token_id < (llama_token)probs.size(); token_id++) {
            const float logit = logf(probs[token_id]);
            cur.emplace_back(llama_token_data{token_id, logit, probs[token_id]});
        }

        cur_p = llama_token_data_array { cur.data(), cur.size(), -1, false };
    }

    void apply(llama_sampler * sampler) {
        llama_sampler_apply(sampler, &cur_p);
        llama_sampler_free(sampler);
    }

    void check(testing & t) {
        if (!t.assert_equal("candidate count", probs_expected.size(), cur_p.size)) {
            return;
        }
        for (size_t i = 0; i < cur_p.size; i++) {
            t.assert_true("p[" + std::to_string(i) + "] is " + fmt(probs_expected[i]) + ", got " + fmt(cur_p.data[i].p),
                          fabs(cur_p.data[i].p - probs_expected[i]) < 1e-5);
        }
    }

    llama_token_data_array cur_p;

private:
    const std::vector<float> probs_expected;

    std::vector<llama_token_data> cur;
};

static llama_token sample_dist(testing & t, llama_sampler * sampler, const std::vector<float> & logits) {
    std::vector<llama_token_data> cur;
    for (llama_token token_id = 0; token_id < (llama_token) logits.size(); ++token_id) {
        cur.push_back({ token_id, logits[token_id], 0.0f });
    }

    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(sampler, &cur_p);
    if (!t.assert_true("dist selects a candidate in range, selected=" + std::to_string(cur_p.selected) + " size=" + std::to_string(cur_p.size),
                       cur_p.selected >= 0 && (size_t) cur_p.selected < cur_p.size)) {
        return LLAMA_TOKEN_NULL;
    }
    return cur_p.data[cur_p.selected].id;
}

static void test_dist_singleton_rng(testing & t) {
    t.test("singleton rng", [](testing & t) {
        llama_sampler * singleton = llama_sampler_init_dist(4242);
        llama_sampler * control   = llama_sampler_init_dist(4242);

        sample_dist(t, singleton, { 0.0f });
        sample_dist(t, control,   { 0.0f, 0.0f });

        const std::vector<float> logits(256, 0.0f);
        for (int i = 0; i < 4; ++i) {
            const llama_token expected = sample_dist(t, control,   logits);
            const llama_token actual   = sample_dist(t, singleton, logits);
            t.assert_equal("draw " + std::to_string(i) + " matches the control sampler", expected, actual);
        }

        llama_sampler_free(singleton);
        llama_sampler_free(control);
    });
}

static void test_temp(testing & t, const std::vector<float> & probs, const std::vector<float> & probs_expected, float temp) {
    t.test("probs=" + fmt(probs) + " temp=" + fmt(temp), [&](testing & t) {
        sampler_tester tester(probs, probs_expected);

        DUMP(&tester.cur_p);
        tester.apply(llama_sampler_init_temp(temp));
        tester.apply(llama_sampler_init_dist(0));
        DUMP(&tester.cur_p);

        tester.check(t);
    });
}

static void test_temp_ext(testing & t, const std::vector<float> & probs, const std::vector<float> & probs_expected, float temp, float delta, float exponent) {
    t.test("probs=" + fmt(probs) + " temp=" + fmt(temp) + " delta=" + fmt(delta) + " exponent=" + fmt(exponent), [&](testing & t) {
        sampler_tester tester(probs, probs_expected);

        DUMP(&tester.cur_p);
        tester.apply(llama_sampler_init_temp_ext(temp, delta, exponent));
        tester.apply(llama_sampler_init_dist (0));
        DUMP(&tester.cur_p);

        tester.check(t);
    });
}

static void test_top_k(testing & t, const std::vector<float> & probs, const std::vector<float> & probs_expected, int k) {
    t.test("probs=" + fmt(probs) + " k=" + std::to_string(k), [&](testing & t) {
        sampler_tester tester(probs, probs_expected);

        DUMP(&tester.cur_p);
        tester.apply(llama_sampler_init_top_k(k));
        tester.apply(llama_sampler_init_dist (0));
        DUMP(&tester.cur_p);

        tester.check(t);
    });
}

static void test_top_p(testing & t, const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    t.test("probs=" + fmt(probs) + " p=" + fmt(p), [&](testing & t) {
        sampler_tester tester(probs, probs_expected);

        DUMP(&tester.cur_p);
        tester.apply(llama_sampler_init_top_p(p, 0));
        tester.apply(llama_sampler_init_dist (0));
        DUMP(&tester.cur_p);

        tester.check(t);
    });
}

static void test_min_p(testing & t, const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    t.test("probs=" + fmt(probs) + " p=" + fmt(p), [&](testing & t) {
        sampler_tester tester(probs, probs_expected);

        DUMP(&tester.cur_p);
        tester.apply(llama_sampler_init_min_p(p, 0));
        tester.apply(llama_sampler_init_dist (0));
        DUMP(&tester.cur_p);

        tester.check(t);
    });
}

static void test_xtc(testing & t, const std::vector<float> & probs, const std::vector<float> & probs_expected, float p, float threshold) {
    t.test("probs=" + fmt(probs) + " p=" + fmt(p) + " t=" + fmt(threshold), [&](testing & t) {
        sampler_tester tester(probs, probs_expected);

        DUMP(&tester.cur_p);
        tester.apply(llama_sampler_init_xtc(p, threshold, 0, 0));
        DUMP(&tester.cur_p);

        tester.check(t);
    });
}

static void test_typical(testing & t, const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    t.test("probs=" + fmt(probs) + " p=" + fmt(p), [&](testing & t) {
        sampler_tester tester(probs, probs_expected);

        DUMP(&tester.cur_p);
        tester.apply(llama_sampler_init_typical(p, 0));
        DUMP(&tester.cur_p);

        tester.check(t);
    });
}

static void test_penalties(
    testing & t,
    const std::vector<float> & probs, const std::vector<llama_token> & last_tokens,
    const std::vector<float> & probs_expected, float repeat_penalty, float alpha_frequency, float alpha_presence
) {
    GGML_ASSERT(probs.size() == probs_expected.size());

    t.test("probs=" + fmt(probs) + " last=" + fmt(last_tokens) + " repeat=" + fmt(repeat_penalty) + " freq=" + fmt(alpha_frequency) + " pres=" + fmt(alpha_presence), [&](testing & t) {
        sampler_tester tester(probs, probs_expected);

        auto * sampler = llama_sampler_init_penalties((int32_t) probs.size(), (int32_t) last_tokens.size(), repeat_penalty, alpha_frequency, alpha_presence);

        for (size_t i = 0; i < last_tokens.size(); i++) {
            llama_sampler_accept(sampler, last_tokens[i]);
        }

        DUMP(&tester.cur_p);
        tester.apply(sampler);
        tester.apply(llama_sampler_init_dist(0));
        DUMP(&tester.cur_p);

        tester.check(t);
    });
}

static void test_dry(
    testing & t,
    const std::vector<float> & probs, const std::vector<llama_token> & last_tokens,
    const std::vector<float> & expected_probs, float dry_multiplier, float dry_base,
    int dry_allowed_length, int dry_penalty_last_n,
    const std::vector<std::vector<llama_token>> & seq_breakers
) {
    GGML_ASSERT(probs.size() == expected_probs.size());

    t.test("probs=" + fmt(probs) + " last=" + fmt(last_tokens) + " mult=" + fmt(dry_multiplier) + " base=" + fmt(dry_base) + " allowed=" + std::to_string(dry_allowed_length) + " last_n=" + std::to_string(dry_penalty_last_n) + " breakers=" + fmt(seq_breakers), [&](testing & t) {
        sampler_tester tester(probs, expected_probs);

        auto * sampler = llama_sampler_init_dry_testing(dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n, seq_breakers);

        for (size_t i = 0; i < last_tokens.size(); i++) {
            llama_sampler_accept(sampler, last_tokens[i]);
        }

        DUMP(&tester.cur_p);
        tester.apply(sampler);
        tester.apply(llama_sampler_init_dist(0));
        DUMP(&tester.cur_p);
        tester.check(t);
    });
}

static void test_top_n_sigma(testing & t, const std::vector<float> & probs, const std::vector<float> & probs_expected, int n) {
    t.test("probs=" + fmt(probs) + " n=" + std::to_string(n), [&](testing & t) {
        sampler_tester tester(probs, probs_expected);

        DUMP(&tester.cur_p);
        tester.apply(llama_sampler_init_top_n_sigma(n));
        tester.apply(llama_sampler_init_dist (0));
        DUMP(&tester.cur_p);

        tester.check(t);
    });
}

static void test_sampler_queue(testing & t, const size_t n_vocab, const std::string & samplers_sequence, const int top_k, const float top_p, const float min_p
) {
    char name[128];
    snprintf(name, sizeof(name), "%s n_vocab=%zu top_k=%d top_p=%g min_p=%g", samplers_sequence.c_str(), n_vocab, top_k, top_p, min_p);

    t.test(name, [&](testing & t) {
        sampler_tester tester(n_vocab);

              llama_token min_token_id = 0;
        const llama_token max_token_id = n_vocab - 1;

        for (auto s : samplers_sequence) {
            switch (s) {
                case 'k': tester.apply(llama_sampler_init_top_k(top_k)); break;
                case 'y': GGML_ABORT("typical test not implemented");
                case 'p': tester.apply(llama_sampler_init_top_p(top_p, 1)); break;
                case 'm': tester.apply(llama_sampler_init_min_p(min_p, 1)); break;
                case 't': GGML_ABORT("temperature test not implemented");
                default : GGML_ABORT("Unknown sampler");
            }

            tester.apply(llama_sampler_init_dist(0));

            auto & cur_p = tester.cur_p;

            const int size = cur_p.size;

            const std::string step = std::string("after '") + s + "'";

            if (s == 'k') {
                const int expected_size = std::min(size, top_k);
                min_token_id = std::max(min_token_id, (llama_token)(n_vocab - top_k));

                if (!t.assert_equal(step + " candidate count", expected_size, size)) {
                    return;
                }
                t.assert_equal(step + " first token id", max_token_id, cur_p.data[0].id);
                t.assert_equal(step + " last token id", min_token_id, cur_p.data[expected_size-1].id);
            } else if (s == 'p') {
                const int softmax_divisor = n_vocab * (n_vocab-1) / 2 - min_token_id * (min_token_id-1) / 2;
                const int softmax_numerator_target = ceilf(top_p * softmax_divisor);

                    min_token_id  = n_vocab;
                int expected_size = 0;
                int cumsum        = 0;
                do { // do-while because always at least one token is sampled
                    min_token_id--;
                    expected_size++;

                    cumsum += min_token_id;
                } while (cumsum < softmax_numerator_target);

                // token 0 has p == 0, need special consideration for cumsum because top_p immediately returns
                if (min_token_id == 1) {
                    min_token_id--;
                    expected_size += 1;
                }

                if (!t.assert_equal(step + " candidate count", expected_size, size)) {
                    return;
                }
                t.assert_true(step + " first token id is " + std::to_string(max_token_id) + " when sorted, got " + std::to_string(cur_p.data[0].id),
                              !cur_p.sorted || cur_p.data[0].id == max_token_id);
                t.assert_true(step + " last token id is " + std::to_string(min_token_id) + " when sorted, got " + std::to_string(cur_p.data[expected_size-1].id),
                              !cur_p.sorted || cur_p.data[expected_size-1].id == min_token_id);
            } else if (s == 'm') {
                int expected_size = ceilf((1.0f - min_p) * n_vocab);
                expected_size = std::max(expected_size, 1);
                expected_size = std::min(expected_size, size);

                min_token_id = floorf(min_p * n_vocab);
                min_token_id = std::max(min_token_id, 1);
                min_token_id = std::max(min_token_id, (llama_token)(n_vocab - size));
                min_token_id = std::min(min_token_id, (llama_token)(n_vocab - 1));

                if (!t.assert_equal(step + " candidate count", expected_size, size)) {
                    return;
                }
                t.assert_true(step + " first token id is " + std::to_string(max_token_id) + " when sorted, got " + std::to_string(cur_p.data[0].id),
                              !cur_p.sorted || cur_p.data[0].id == max_token_id);
                t.assert_true(step + " last token id is " + std::to_string(min_token_id) + " when sorted, got " + std::to_string(cur_p.data[expected_size-1].id),
                              !cur_p.sorted || cur_p.data[expected_size-1].id == min_token_id);
            } else {
                GGML_ABORT("fatal error");
            }
        }
    });
}

static void bench(testing & t, llama_sampler * cnstr, const char * cnstr_name, const std::vector<llama_token_data> & data, int n_iter) {
    std::vector<llama_token_data> cur(data.size());
    std::copy(data.begin(), data.end(), cur.begin());
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(cnstr, &cur_p);
    llama_sampler_reset(cnstr);
    t.bench(cnstr_name, [&] {
        std::copy(data.begin(), data.end(), cur.begin());
        llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
        llama_sampler_apply(cnstr, &cur_p);
        llama_sampler_reset(cnstr);
    }, n_iter);
    llama_sampler_free(cnstr);
}

#define BENCH(__t, __cnstr, __data, __n_iter) bench((__t), (__cnstr), #__cnstr, (__data), (__n_iter))

static void test_perf(testing & t) {
    const int n_vocab = 1 << 17;

    std::vector<llama_token_data> data;

    data.reserve(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        const float logit = 2.0f*((double)(rand())/RAND_MAX - 0.5);
        data.emplace_back(llama_token_data{i, logit, 0.0f});
    }

    BENCH(t, llama_sampler_init_top_k  (40),                     data, 32);
    BENCH(t, llama_sampler_init_top_p  (0.8f, 1),                data, 32);
    BENCH(t, llama_sampler_init_min_p  (0.2f, 1),                data, 32);
    BENCH(t, llama_sampler_init_typical(0.5f, 1),                data, 32);
    BENCH(t, llama_sampler_init_xtc    (1.0f, 0.1f, 1, 1),       data, 32);
}

int main(int argc, char ** argv) {
    testing t;
    t.capture_output = true;
    t.apply_env();
    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("dist", test_dist_singleton_rng);

    t.test("temp", [](testing & t) {
        test_temp(t, {0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f);
        test_temp(t, {0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.0f, 1.0f}, 0.0f);
    });

    t.test("temp_ext", [](testing & t) {
        test_temp_ext(t, {0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f, 0.0f, 1.0f);
        test_temp_ext(t, {0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.0f, 1.0f}, 0.0f, 0.0f, 1.0f);
    });

    t.test("top_k", [](testing & t) {
        test_top_k(t, {0.1f, 0.2f, 0.3f, 0.4f}, {1.0f}, 1);
        test_top_k(t, {0.1f, 0.2f, 0.3f, 0.4f}, {0.44444f, 0.33333f, 0.22222f}, 3);
        test_top_k(t, {0.1f, 0.2f, 0.3f, 0.4f}, {0.4f, 0.3f, 0.2f, 0.1f}, 4);
        test_top_k(t, {0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 0);
    });

    t.test("top_p", [](testing & t) {
        test_top_p(t, {0.1f, 0.2f, 0.3f, 0.4f}, {1.0f}, 0);
        test_top_p(t, {0.1f, 0.2f, 0.3f, 0.4f}, {0.571429f, 0.428571f}, 0.7f);
        test_top_p(t, {0.1f, 0.2f, 0.3f, 0.4f}, {0.44444f, 0.33333f, 0.22222f}, 0.8f);
        test_top_p(t, {0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f);
    });

    t.test("min_p", [](testing & t) {
        test_min_p(t, {0.1f, 0.2f, 0.3f, 0.4f}, {0.1f/1.0f, 0.2f/1.0f, 0.3f/1.0f, 0.4f/1.0f}, 0.00f);
        test_min_p(t, {0.1f, 0.2f, 0.3f, 0.4f}, {0.1f/1.0f, 0.2f/1.0f, 0.3f/1.0f, 0.4f/1.0f}, 0.24f);
        test_min_p(t, {0.1f, 0.2f, 0.3f, 0.4f}, {0.2f/0.9f, 0.3f/0.9f, 0.4f/0.9f},            0.26f);
        test_min_p(t, {0.1f, 0.2f, 0.3f, 0.4f}, {0.2f/0.9f, 0.3f/0.9f, 0.4f/0.9f},            0.49f);
        test_min_p(t, {0.1f, 0.2f, 0.3f, 0.4f}, {0.3f/0.7f, 0.4f/0.7f},                       0.51f);
        test_min_p(t, {0.1f, 0.2f, 0.3f, 0.4f}, {0.3f/0.7f, 0.4f/0.7f},                       0.74f);
        test_min_p(t, {0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  0.76f);
        test_min_p(t, {0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  1.00f);
        test_min_p(t, {0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  1.05f);
    });

    t.test("xtc", [](testing & t) {
        t.test("should", [](testing & t) {
            test_xtc(t, {0.4f, 0.3f, 0.2f, 0.1f},   {0.1f},                                0.99f, 0.09f);
            test_xtc(t, {0.4f, 0.3f, 0.2f, 0.1f},   {0.2f, 0.1f},                          0.99f, 0.19f);
            test_xtc(t, {0.4f, 0.3f, 0.2f, 0.1f},   {0.3f, 0.2f, 0.1f},                    0.99f, 0.29f);
        });

        t.test("should not", [](testing & t) {
            test_xtc(t, {0.4f, 0.3f, 0.2f, 0.1f},   {0.4f, 0.3f, 0.2f, 0.1f},              0.99f, 0.39f);
        });
    });

    t.test("typical", [](testing & t) {
        test_typical(t, {0.97f, 0.01f, 0.01f, 0.01f}, {0.97f},            0.5f);
        test_typical(t, {0.4f, 0.2f, 0.2f, 0.2f},     {0.2f, 0.2f, 0.2f}, 0.5f);
    });

    t.test("penalties", [](testing & t) {
        test_penalties(t, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0}, {0, 0.25f, 0.25f, 0.25f, 0.25f},   50.0f, 0.0f, 0.0f);
        test_penalties(t, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2}, {0, 0, 0, 0.5f, 0.5f},       50.0f, 0.0f, 0.0f);
        test_penalties(t, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 0}, {0, 0, 0, 0.5f, 0.5f}, 50.0f, 0.0f, 0.0f);

        test_penalties(t, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0},             {0.000011f, 0.249997f, 0.249997f, 0.249997f, 0.249997f}, 1.0f, 5.0f, 5.0f);
        test_penalties(t, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2},       {0.000023f, 0.000023f, 0.000023f, 0.499966f, 0.499966f}, 1.0f, 5.0f, 5.0f);
        test_penalties(t, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 0}, {0.000000f, 0.000023f, 0.000023f, 0.499977f, 0.499977f}, 1.0f, 5.0f, 5.0f);
    });

    t.test("dry", [](testing & t) {
        test_dry(t, {0.25f, 0.25f, 0.25f, 0.25f}, {0, 1}, {0.25f, 0.25f, 0.25f, 0.25f}, 1.0f, 1.1f, 2, 4, {});
        test_dry(t, {0.25f, 0.25f, 0.25f, 0.25f}, {0, 1, 2, 0, 1}, {0.296923f, 0.296923f, 0.109232f, 0.296923f}, 1.0f, 1.1f, 2, 5, {});
        test_dry(t, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 3, 4, 0, 1}, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, 1.0f, 1.1f, 2, 6, {{3}});
        test_dry(t, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 1}, {0.241818f, 0.241818f, 0.032727f, 0.241818f, 0.241818f}, 2.0f, 1.1f, 2, 5, {});
        test_dry(t, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 3, 4, 0, 1}, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, 1.0f, 1.1f, 4, 7, {});
    });

    t.test("top_n_sigma", [](testing & t) {
        test_top_n_sigma(t, {0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.428571f, 0.571429f}, 1.00f);
        test_top_n_sigma(t, {0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 0.00f); // top_n_sigma == 0 now represents a no-op rather than greedy decoding as of PR#13345
        test_top_n_sigma(t, {0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 3.00f);
    });

    t.test("sampler queue", [](testing & t) {
        test_sampler_queue(t, 10000, "k", 10000, 1.0f, 1.0f);
        test_sampler_queue(t, 10000, "k",     1, 1.0f, 1.0f);
        test_sampler_queue(t, 10000, "p", 10000, 1.0f, 1.0f);
        test_sampler_queue(t, 10000, "p", 10000, 0.0f, 1.0f);
        test_sampler_queue(t, 10000, "m", 10000, 1.0f, 1.0f);
        test_sampler_queue(t, 10000, "m", 10000, 1.0f, 1e-12);

        test_sampler_queue(t, 10000, "k",   100, 1.0000f, 1.0f);
        test_sampler_queue(t, 10000, "p", 10000, 0.0003f, 1.0f);
        test_sampler_queue(t, 10000, "p", 10000, 0.8000f, 1.0f);
        test_sampler_queue(t, 10000, "m", 10000, 1.0000f, 9997.9f/9999.0f);
        test_sampler_queue(t, 10000, "m", 10000, 1.0000f, 0.1f);

        test_sampler_queue(t, 10000, "kp", 100, 0.8f, 0.1f);
        test_sampler_queue(t, 10000, "km", 100, 0.8f, 0.1f);
        test_sampler_queue(t, 10000, "pk", 100, 0.8f, 0.1f);
        test_sampler_queue(t, 10000, "pm", 100, 0.8f, 0.1f);
        test_sampler_queue(t, 10000, "mk", 100, 0.8f, 0.1f);
        test_sampler_queue(t, 10000, "mp", 100, 0.8f, 9997.9f/9999.0f);
        test_sampler_queue(t, 10000, "mp", 100, 0.8f, 0.1f);

        test_sampler_queue(t, 10000, "kpm", 100, 0.8f, 0.1f);
        test_sampler_queue(t, 10000, "kmp", 100, 0.8f, 0.1f);
        test_sampler_queue(t, 10000, "pkm", 100, 0.8f, 0.1f);
        test_sampler_queue(t, 10000, "pmk", 100, 0.8f, 0.1f);
        test_sampler_queue(t, 10000, "mkp", 100, 0.8f, 0.1f);
        test_sampler_queue(t, 10000, "mpk", 100, 0.8f, 0.1f);
    });

    t.test("perf", test_perf);

    return t.summary();
}
