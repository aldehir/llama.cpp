// These tests may take a long time!
// They are to prove that conversion from double to float of various functions in ggml.c doesn't affect the result.
// This is done by checking all finite (non-NaN, non-infinite) floats.

#include "testing.h"

#if !defined(__riscv) && !defined(__s390__) && !defined(__ARM_NEON)
#include <immintrin.h>
#endif
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#endif

// ggml.c::quantize_row_q4_0_ref
inline static uint8_t round_orig(float v0) { return ((int8_t) (round(v0))) + 8; }

// ggml.c::ggml_silu_f32
inline static float silu_orig(float x) {
    return x/(1.0 + exp(-x));
}

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

// ggml.c::quantize_row_q4_0_ref
inline static uint8_t round_float(float v0) { return (int8_t)roundf(v0) + 8; }

// ggml.c::ggml_silu_f32
inline static float silu_float(float x) {
    return x/(1.0f + expf(-x));
}

// the loops visit billions of values, so mismatches are counted and reported once with the first offending input
static void test_round(testing & t) {
    uint64_t n_mismatch = 0;
    uint32_t first      = 0;

    uint32_t x = UINT32_MAX;
    do {
        float f;
        memcpy(&f, &x, sizeof(x));
        if (!(!std::isfinite(f) || (round_orig(f) == round_float(f)))) {
            if (n_mismatch == 0) {
                first = x;
            }
            n_mismatch++;
        }
    } while (x--);

    char msg[128];
    snprintf(msg, sizeof(msg), "round_orig == round_float for all finite floats (mismatches=%llu, first bits=0x%08x)",
             (unsigned long long) n_mismatch, first);
    t.assert_true(msg, n_mismatch == 0);
}

static void test_silu_fp16(testing & t) {
#ifdef __F16C__
    // GELU and SILU implementations are used with a FP16 lookup table.
    // The original and float-only results are not equal for all inputs after converting to FP16.
    // GELU is an approximation anyway (tanh), not tested here.
    // For SILU, verify that the results are at least the closest floating point numbers, if the FP16 values don't match.
    uint64_t n_mismatch = 0;
    uint32_t first      = 0;

    for (uint32_t x = 0; x <= UINT16_MAX; x++) {
        float f = _cvtsh_ss(x);
        const float so = silu_orig(f);
        const float sf = silu_float(f);
        if (!(   (_cvtss_sh(so, 0) == _cvtss_sh(sf, 0))
              || (nextafterf(so, sf) == sf)
              || (nextafterf(sf, so) == so))) {
            if (n_mismatch == 0) {
                first = x;
            }
            n_mismatch++;
        }
    }

    char msg[128];
    snprintf(msg, sizeof(msg), "silu_orig and silu_float agree within one ulp for all fp16 (mismatches=%llu, first fp16=0x%04x)",
             (unsigned long long) n_mismatch, first);
    t.assert_true(msg, n_mismatch == 0);
#else
    t.skip("requires F16C");
#endif
}

int main(int argc, char ** argv) {
    testing t;
    t.capture_output = true;
    t.apply_env();

    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("round",     test_round);
    t.test("silu_fp16", test_silu_fp16);

    return t.summary();
}
