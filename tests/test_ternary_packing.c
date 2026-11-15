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
    timespec_get(&ts, TIME_UTC);
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
 * Test 5: Checked Unpack, Dual 5-Trit Pair & All-Zero Constants
 * ========================================================================= */
static void test_checked_and_pair_packing(void) {
    printf("[TEST] Running Checked Unpack & Dual 5-Trit Pair Test... ");

    zcrudp_trit_t zeros[10] = { 0 };
    assert(zcrudp_pack_5trits(zeros) == ZCRUDP_TRIT_ALL_ZERO_5P);
    assert(zcrudp_pack_10trits(zeros) == ZCRUDP_TRIT_ALL_ZERO_10P);

    zcrudp_trit_t out5[5];
    assert(zcrudp_unpack_5trits_checked(0, out5) == true);
    assert(zcrudp_unpack_5trits_checked(242, out5) == true);
    assert(zcrudp_unpack_5trits_checked(243, out5) == false);
    assert(zcrudp_unpack_5trits_checked(255, out5) == false);

    zcrudp_trit_t out10[10];
    assert(zcrudp_unpack_10trits_checked(0, out10) == true);
    assert(zcrudp_unpack_10trits_checked(59048, out10) == true);
    assert(zcrudp_unpack_10trits_checked(59049, out10) == false);
    assert(zcrudp_unpack_10trits_checked(65535, out10) == false);

    zcrudp_trit_t pair_in[10] = { -1, 0, +1, -1, +1, +1, +1, 0, -1, 0 };
    uint16_t pair_word = zcrudp_pack_5trits_pair(pair_in);
    assert(zcrudp_unpack_5trits_pair_checked(pair_word, out10) == true);
    for (int i = 0; i < 10; i++) {
        assert(pair_in[i] == out10[i]);
    }
    /* Corrupt high byte (>= 243) and low byte (>= 243) */
    assert(zcrudp_unpack_5trits_pair_checked((uint16_t)(243U << 8), out10) == false);
    assert(zcrudp_unpack_5trits_pair_checked(243U, out10) == false);

    printf("PASSED\n");
}

/* =========================================================================
 * Test 6: Wire Serialization Roundtrip (Little-Endian)
 * ========================================================================= */
static void test_wire_serialization(void) {
    printf("[TEST] Running Little-Endian Wire Serialization Test... ");

    zcrudp_tfv_record_t rec_in = {
        .type = ZCRUDP_TFV_TYPE_TERNARY_SKIP,
        .flags = 0xAB,
        .value = 0x1234
    };
    uint8_t buf4[4];
    zcrudp_tfv_write_le(buf4, &rec_in);
    assert(buf4[0] == ZCRUDP_TFV_TYPE_TERNARY_SKIP);
    assert(buf4[1] == 0xAB);
    assert(buf4[2] == 0x34);
    assert(buf4[3] == 0x12);

    zcrudp_tfv_record_t rec_out;
    zcrudp_tfv_read_le(buf4, &rec_out);
    assert(rec_out.type == rec_in.type);
    assert(rec_out.flags == rec_in.flags);
    assert(rec_out.value == rec_in.value);

    zcrudp_ternary_slice_header_t hdr_in = {
        .slice_index = 42,
        .start_pixel = 12800,
        .count = 64,
        .flags = 0x01,
        .reserved = 0x00
    };
    uint8_t buf8[8];
    zcrudp_slice_header_write_le(buf8, &hdr_in);
    zcrudp_ternary_slice_header_t hdr_out;
    zcrudp_slice_header_read_le(buf8, &hdr_out);
    assert(hdr_out.slice_index == hdr_in.slice_index);
    assert(hdr_out.start_pixel == hdr_in.start_pixel);
    assert(hdr_out.count == hdr_in.count);
    assert(hdr_out.flags == hdr_in.flags);
    assert(hdr_out.reserved == hdr_in.reserved);

    printf("PASSED\n");
}

/* =========================================================================
 * Test 7: Closed-Loop Encoder + Leaky Decoder with Simulated Packet Loss
 * ========================================================================= */
static void test_closed_loop_packet_loss_recovery(void) {
    printf("[TEST] Running Closed-Loop + 10%% Packet Loss Recovery Test... ");

    enum { N_PIXELS = 100 };
    uint8_t source[N_PIXELS];
    uint8_t enc_ref[N_PIXELS];
    uint8_t dec_canvas[N_PIXELS];
    zcrudp_trit_t trits[N_PIXELS];

    const uint8_t neutral = 128;
    const uint8_t gamma_q8 = 243; /* ~0.95 */
    const uint8_t delta = 12;
    const uint8_t threshold = 6;

    for (size_t i = 0; i < N_PIXELS; i++) {
        source[i] = (uint8_t)(40 + (i % 180));
        enc_ref[i] = neutral;
        dec_canvas[i] = neutral;
    }

    /* Warmup 30 frames with 0% loss -> encoder & decoder must match identically */
    for (int f = 0; f < 30; f++) {
        for (size_t i = 0; i < N_PIXELS; i++) {
            trits[i] = zcrudp_encode_step(source[i], &enc_ref[i], threshold, delta, gamma_q8, neutral);
        }
        zcrudp_leaky_integrator_batch(dec_canvas, trits, N_PIXELS, delta, gamma_q8, neutral);
    }
    for (size_t i = 0; i < N_PIXELS; i++) {
        assert(enc_ref[i] == dec_canvas[i]);
    }

    /* Inject 5-frame burst loss (decoder receives 0s) and measure error */
    zcrudp_trit_t lost_trits[N_PIXELS] = { 0 };
    for (int f = 0; f < 5; f++) {
        for (size_t i = 0; i < N_PIXELS; i++) {
            trits[i] = zcrudp_encode_step(source[i], &enc_ref[i], threshold, delta, gamma_q8, neutral);
        }
        zcrudp_leaky_integrator_batch(dec_canvas, lost_trits, N_PIXELS, delta, gamma_q8, neutral);
    }

    uint32_t err_after_loss = 0;
    for (size_t i = 0; i < N_PIXELS; i++) {
        int32_t d = (int32_t)enc_ref[i] - (int32_t)dec_canvas[i];
        err_after_loss += (uint32_t)(d < 0 ? -d : d);
    }
    assert(err_after_loss > 0);

    /* Recover over 60 frames: leaky integrator decays residual drift exponentially */
    for (int f = 0; f < 60; f++) {
        for (size_t i = 0; i < N_PIXELS; i++) {
            trits[i] = zcrudp_encode_step(source[i], &enc_ref[i], threshold, delta, gamma_q8, neutral);
        }
        zcrudp_leaky_integrator_batch(dec_canvas, trits, N_PIXELS, delta, gamma_q8, neutral);
    }

    uint32_t err_after_recovery = 0;
    for (size_t i = 0; i < N_PIXELS; i++) {
        int32_t d = (int32_t)enc_ref[i] - (int32_t)dec_canvas[i];
        err_after_recovery += (uint32_t)(d < 0 ? -d : d);
    }
    assert(err_after_recovery < err_after_loss / 4);

    printf("PASSED (MAE sum %u -> %u)\n", err_after_loss, err_after_recovery);
}

/* =========================================================================
 * Benchmark: CPU Cycles & Throughput Benchmark (Non-Elided)
 * ========================================================================= */
static inline uint64_t read_ns(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void benchmark_packing(void) {
    printf("[BENCHMARK] Measuring CPU performance for packing/unpacking/integrator...\n");

    const uint32_t ITERS = 1000000;
    zcrudp_trit_t trits[10] = { +1, -1, 0, +1, -1, 0, +1, -1, +1, -1 };
    volatile uint64_t sink = 0;
    uint64_t acc = 0;
    zcrudp_trit_t unpacked[10];

    /* Warmup */
    for (uint32_t i = 0; i < 10000; i++) {
        acc += zcrudp_pack_10trits(trits);
    }

    /* Benchmark Pack 10-Trits */
    uint64_t t0_ns = read_ns();
    uint64_t start_cycles = read_tsc();
    for (uint32_t i = 0; i < ITERS; i++) {
        trits[0] = (zcrudp_trit_t)((i & 1) ? 1 : -1);
        acc += zcrudp_pack_10trits(trits);
    }
    uint64_t end_cycles = read_tsc();
    uint64_t t1_ns = read_ns();
    double pack_cycles_per_op = (double)(end_cycles - start_cycles) / ITERS;
    double pack_mpix_s = ((double)ITERS * 10.0) / ((double)(t1_ns - t0_ns) * 1e-3);

    /* Benchmark Unpack 10-Trits */
    uint16_t seed = (uint16_t)(acc % ZCRUDP_TRIT_MAX_10P);
    t0_ns = read_ns();
    start_cycles = read_tsc();
    for (uint32_t i = 0; i < ITERS; i++) {
        uint16_t w = (uint16_t)((seed + i) % ZCRUDP_TRIT_MAX_10P);
        zcrudp_unpack_10trits(w, unpacked);
        acc += (uint64_t)(unpacked[0] + unpacked[9]);
    }
    end_cycles = read_tsc();
    t1_ns = read_ns();
    double unpack_cycles_per_op = (double)(end_cycles - start_cycles) / ITERS;
    double unpack_mpix_s = ((double)ITERS * 10.0) / ((double)(t1_ns - t0_ns) * 1e-3);

    /* Benchmark Leaky Integrator Batch (1000 pixels per iter) */
    uint8_t canvas[1000];
    zcrudp_trit_t batch_trits[1000];
    for (size_t i = 0; i < 1000; i++) {
        canvas[i] = (uint8_t)(i & 0xFF);
        batch_trits[i] = (zcrudp_trit_t)((i % 3) - 1);
    }
    const uint32_t BATCH_ITERS = 100000;
    t0_ns = read_ns();
    start_cycles = read_tsc();
    for (uint32_t i = 0; i < BATCH_ITERS; i++) {
        zcrudp_leaky_integrator_batch(canvas, batch_trits, 1000, 8, 243, 128);
        acc += canvas[i % 1000];
    }
    end_cycles = read_tsc();
    t1_ns = read_ns();
    double batch_mpix_s = ((double)BATCH_ITERS * 1000.0) / ((double)(t1_ns - t0_ns) * 1e-3);
    double batch_cycles_per_px = (double)(end_cycles - start_cycles) / ((double)BATCH_ITERS * 1000.0);

    sink = acc;
    (void)sink;

    printf("  -> Pack 10-trits   : %.2f cycles/op (%.2f cycles/trit) | %.1f MPixels/s\n",
           pack_cycles_per_op, pack_cycles_per_op / 10.0, pack_mpix_s);
    printf("  -> Unpack 10-trits : %.2f cycles/op (%.2f cycles/trit) | %.1f MPixels/s\n",
           unpack_cycles_per_op, unpack_cycles_per_op / 10.0, unpack_mpix_s);
    printf("  -> Leaky Batch     : %.2f cycles/px                     | %.1f MPixels/s\n",
           batch_cycles_per_px, batch_mpix_s);
}

int main(void) {
    printf("=== zcrudp Ternary Video Profile Test Suite ===\n\n");

    test_5trits_roundtrip();
    test_10trits_roundtrip();
    test_tfv_record_integration();
    test_leaky_integrator();
    test_checked_and_pair_packing();
    test_wire_serialization();
    test_closed_loop_packet_loss_recovery();
    printf("\n");
    benchmark_packing();

    printf("\n=== ALL TESTS PASSED SUCCESSFULLY ===\n");
    return 0;
}
