/**
 * @file test_ternary_packing.c
 * @brief Unit tests & CPU cycle benchmark for C11 ternary packing primitives.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#include "protocol_ternary.h"

#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
static inline uint64_t read_tsc(void) {
    return __rdtsc();
}
#else
static inline uint64_t read_tsc(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
#endif

/* =========================================================================
 * Test 1: 5-Trit Exhaustive Roundtrip Test (3^5 = 243 combinations)
 * ========================================================================= */
static void test_5trits_roundtrip(void) {
    printf("[TEST] Running 5-Trit Exhaustive Roundtrip Test (3^5 = 243)... ");
    uint32_t count = 0;

    for (int t0 = -1; t0 <= 1; t0++) {
        for (int t1 = -1; t1 <= 1; t1++) {
            for (int t2 = -1; t2 <= 1; t2++) {
                for (int t3 = -1; t3 <= 1; t3++) {
                    for (int t4 = -1; t4 <= 1; t4++) {
                        zcrudp_trit_t orig[5] = { (zcrudp_trit_t)t0, (zcrudp_trit_t)t1,
                                                 (zcrudp_trit_t)t2, (zcrudp_trit_t)t3,
                                                 (zcrudp_trit_t)t4 };

                        uint8_t packed = zcrudp_pack_5trits(orig);
                        assert(packed < ZCRUDP_TRIT_MAX_5P);

                        zcrudp_trit_t unpacked[5];
                        zcrudp_unpack_5trits(packed, unpacked);

                        for (int i = 0; i < 5; i++) {
                            assert(orig[i] == unpacked[i]);
                        }
                        count++;
                    }
                }
            }
        }
    }

    assert(count == 243);
    printf("PASSED (%u combinations verified)\n", count);
}

/* =========================================================================
 * Test 2: 10-Trit Exhaustive Roundtrip Test (3^10 = 59,049 combinations)
 * ========================================================================= */
static void test_10trits_roundtrip(void) {
    printf("[TEST] Running 10-Trit Exhaustive Roundtrip Test (3^10 = 59,049)... ");
    uint32_t count = 0;

    for (uint32_t idx = 0; idx < ZCRUDP_TRIT_MAX_10P; idx++) {
        zcrudp_trit_t orig[10];
        uint32_t temp = idx;
        for (int i = 0; i < 10; i++) {
            orig[i] = (zcrudp_trit_t)(temp % 3U) - 1;
            temp /= 3U;
        }

        uint16_t packed = zcrudp_pack_10trits(orig);
        assert(packed == idx);
        assert(packed < ZCRUDP_TRIT_MAX_10P);

        zcrudp_trit_t unpacked[10];
        zcrudp_unpack_10trits(packed, unpacked);

        for (int i = 0; i < 10; i++) {
            assert(orig[i] == unpacked[i]);
        }
        count++;
    }

    assert(count == 59049);
    printf("PASSED (%u combinations verified)\n", count);
}

/* =========================================================================
 * Test 3: TFV Record Structure Integration Test
 * ========================================================================= */
static void test_tfv_record_integration(void) {
    printf("[TEST] Running TFV Record Integration Test... ");

    zcrudp_trit_t trits[10] = { +1, -1, 0, +1, +1, -1, 0, -1, +1, 0 };
    uint16_t packed_val = zcrudp_pack_10trits(trits);

    zcrudp_tfv_record_t rec = {
        .type = ZCRUDP_TFV_TYPE_TERNARY_PACK10,
        .flags = 0x01,
        .value = packed_val
    };

    assert(sizeof(rec) == 4);
    assert(rec.type == ZCRUDP_TFV_TYPE_TERNARY_PACK10);
    assert(rec.flags == 0x01);

    zcrudp_trit_t unpacked[10];
    zcrudp_unpack_10trits(rec.value, unpacked);

    for (int i = 0; i < 10; i++) {
        assert(trits[i] == unpacked[i]);
    }

    printf("PASSED (TFV size = %zu bytes)\n", sizeof(rec));
}

/* =========================================================================
 * Test 4: Leaky Integrator Reconstruction Test
 * ========================================================================= */
static void test_leaky_integrator(void) {
    printf("[TEST] Running Leaky Integrator Decay & Step Test... ");

    uint8_t prev = 200;
    uint8_t neutral = 128;
    uint8_t gamma_q8 = 243; /* ~0.95 */
    uint8_t delta = 10;

    /* Step with trit 0: should decay towards neutral */
    uint8_t next_decay = zcrudp_leaky_integrator_step(prev, 0, delta, gamma_q8, neutral);
    assert(next_decay < prev);
    assert(next_decay > neutral);

    /* Step with trit +1 */
    uint8_t next_plus = zcrudp_leaky_integrator_step(128, +1, delta, gamma_q8, neutral);
    assert(next_plus == 138);

    /* Step with trit -1 */
    uint8_t next_minus = zcrudp_leaky_integrator_step(128, -1, delta, gamma_q8, neutral);
    assert(next_minus == 118);

    printf("PASSED\n");
}

/* =========================================================================
 * Benchmark: CPU Cycles & Throughput Benchmark
 * ========================================================================= */
static void benchmark_packing(void) {
    printf("[BENCHMARK] Measuring CPU performance for packing/unpacking...\n");

    const uint32_t ITERS = 1000000;
    zcrudp_trit_t trits[10] = { +1, -1, 0, +1, -1, 0, +1, -1, +1, -1 };
    volatile uint16_t dummy_packed = 0;
    zcrudp_trit_t dummy_unpacked[10];

    /* Warmup */
    for (uint32_t i = 0; i < 10000; i++) {
        dummy_packed = zcrudp_pack_10trits(trits);
    }

    /* Benchmark Pack 10-Trits */
    uint64_t start_cycles = read_tsc();
    for (uint32_t i = 0; i < ITERS; i++) {
        trits[0] = (zcrudp_trit_t)((i & 1) ? 1 : -1);
        dummy_packed = zcrudp_pack_10trits(trits);
    }
    uint64_t end_cycles = read_tsc();
    double pack_cycles_per_op = (double)(end_cycles - start_cycles) / ITERS;
    double pack_cycles_per_trit = pack_cycles_per_op / 10.0;

    /* Benchmark Unpack 10-Trits */
    start_cycles = read_tsc();
    for (uint32_t i = 0; i < ITERS; i++) {
        zcrudp_unpack_10trits((uint16_t)(dummy_packed + (i & 0xFF)), dummy_unpacked);
    }
    end_cycles = read_tsc();
    double unpack_cycles_per_op = (double)(end_cycles - start_cycles) / ITERS;
    double unpack_cycles_per_trit = unpack_cycles_per_op / 10.0;

    printf("  -> Pack 10-trits   : %.2f cycles/op (%.2f cycles/trit)\n", pack_cycles_per_op, pack_cycles_per_trit);
    printf("  -> Unpack 10-trits : %.2f cycles/op (%.2f cycles/trit)\n", unpack_cycles_per_op, unpack_cycles_per_trit);
}

int main(void) {
    printf("=== zcrudp Ternary Video Profile Test Suite ===\n\n");

    test_5trits_roundtrip();
    test_10trits_roundtrip();
    test_tfv_record_integration();
    test_leaky_integrator();
    printf("\n");
    benchmark_packing();

    printf("\n=== ALL TESTS PASSED SUCCESSFULLY ===\n");
    return 0;
}
