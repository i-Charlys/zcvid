/**
 * @file ternary_rans.h
 * @brief 4-way interleaved 12-bit static rANS entropy coder for tern_vid.
 *
 * Designed for 1080p360 real-time transport:
 *  - Zero dynamic allocation, zero mutable global state, zero encoder/decoder drift.
 *  - 4 independent uint32_t rANS states interleaved per packet (breaks serial
 *    data dependencies and isolates every UDP packet).
 *  - Encodes zero-run symbols (0..13 = run + non-zero trit, 14 = 14 zeros,
 *    15 = 128 zeros) using 4 compile-time static CDF profiles (chosen per packet),
 *    with 1-bit signs packed separately at 1.00 bit/sign.
 */

#ifndef ZCRUDP_TERNARY_RANS_H
#define ZCRUDP_TERNARY_RANS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ZCRUDP_RANS_BITS  12U
#define ZCRUDP_RANS_SCALE (1U << ZCRUDP_RANS_BITS) /* 4096 */
#define ZCRUDP_RANS_L     (1U << 23)
#define ZCRUDP_RANS_PROFILES 4U
#define ZCRUDP_RANS_SYMS     16U

/*
 * 4 static cumulative frequency tables (each summing to 4096, every sym >= 1):
 *   Profile 0: very sparse (~1-3% active, heavy sym 14/15 runs)
 *   Profile 1: sparse (~4-10% active)
 *   Profile 2: medium (~12-25% active)
 *   Profile 3: dense (~25-50% active, concentrated on runs 0..3)
 */
static const uint16_t ZCRUDP_RANS_CUM[ZCRUDP_RANS_PROFILES][ZCRUDP_RANS_SYMS + 1U] = {
    /* Profile 0 (~2% active) */
    { 0, 720, 1080, 1340, 1540, 1700, 1830, 1940, 2030, 2110, 2180, 2240, 2295, 2345, 2390, 3420, 4096 },
    /* Profile 1 (~7% active) */
    { 0, 960, 1440, 1800, 2080, 2300, 2480, 2630, 2760, 2875, 2975, 3065, 3145, 3215, 3280, 3940, 4096 },
    /* Profile 2 (~18% active) */
    { 0, 1350, 2020, 2490, 2830, 3090, 3290, 3450, 3580, 3690, 3780, 3855, 3915, 3965, 4005, 4084, 4096 },
    /* Profile 3 (~35% active) */
    { 0, 1950, 2780, 3260, 3550, 3730, 3850, 3930, 3985, 4020, 4045, 4062, 4074, 4082, 4088, 4094, 4096 }
};

static inline uint8_t zcrudp_rans_pick_profile(size_t num_nz, size_t total_trits) {
    if (total_trits == 0U) return 0U;
    /* activity in per-mille */
    size_t permille = (num_nz * 1000U) / total_trits;
    if (permille <= 35U)  return 0U;
    if (permille <= 110U) return 1U;
    if (permille <= 240U) return 2U;
    return 3U;
}

static inline uint8_t zcrudp_rans_find_sym(uint32_t cf, const uint16_t cum[17]) {
    uint32_t s = 0U;
    if (cf >= cum[s + 8U]) s += 8U;
    if (cf >= cum[s + 4U]) s += 4U;
    if (cf >= cum[s + 2U]) s += 2U;
    if (cf >= cum[s + 1U]) s += 1U;
    return (uint8_t)s;
}

/**
 * @brief Encodes a sequence of run-symbols (0..15) using 4-way interleaved rANS.
 *        Writes [1B profile][16B 4x uint32_t states][reversed byte stream] into out.
 *        Returns total bytes written, or 0 if out_cap is insufficient.
 */
static inline size_t zcrudp_rans_encode_syms(const uint8_t *syms,
                                             size_t num_syms,
                                             uint8_t profile,
                                             uint8_t *out,
                                             size_t out_cap,
                                             uint8_t *scratch,
                                             size_t scratch_cap) {
    if (profile >= ZCRUDP_RANS_PROFILES || out_cap < 17U || scratch_cap < 16U) {
        return 0U;
    }
    const uint16_t *cum = ZCRUDP_RANS_CUM[profile];
    uint32_t st[4] = { ZCRUDP_RANS_L, ZCRUDP_RANS_L, ZCRUDP_RANS_L, ZCRUDP_RANS_L };
    uint8_t *ptr = scratch + scratch_cap;

    for (size_t i = num_syms; i > 0U; i--) {
        size_t idx = i - 1U;
        uint8_t s = syms[idx] & 15U;
        uint32_t start = cum[s];
        uint32_t freq  = cum[s + 1U] - start;
        uint32_t x = st[idx & 3U];
        uint32_t x_max = ((ZCRUDP_RANS_L >> ZCRUDP_RANS_BITS) << 8) * freq;
        while (x >= x_max) {
            if (ptr <= scratch) {
                return 0U;
            }
            *--ptr = (uint8_t)(x & 0xFFU);
            x >>= 8;
        }
        st[idx & 3U] = ((x / freq) << ZCRUDP_RANS_BITS) + (x % freq) + start;
    }

    size_t stream_len = (size_t)((scratch + scratch_cap) - ptr);
    size_t total = 17U + stream_len;
    if (total > out_cap) {
        return 0U;
    }
    out[0] = profile;
    for (size_t lane = 0; lane < 4U; lane++) {
        uint32_t v = st[lane];
        out[1U + lane * 4U + 0U] = (uint8_t)(v & 0xFFU);
        out[1U + lane * 4U + 1U] = (uint8_t)((v >> 8) & 0xFFU);
        out[1U + lane * 4U + 2U] = (uint8_t)((v >> 16) & 0xFFU);
        out[1U + lane * 4U + 3U] = (uint8_t)((v >> 24) & 0xFFU);
    }
    memcpy(out + 17U, ptr, stream_len);
    return total;
}

/**
 * @brief Decodes num_syms run-symbols (0..15) from a 4-way interleaved rANS stream.
 */
static inline bool zcrudp_rans_decode_syms(const uint8_t *in,
                                           size_t in_len,
                                           uint8_t *syms_out,
                                           size_t num_syms) {
    if (in_len < 17U) {
        return false;
    }
    uint8_t profile = in[0];
    if (profile >= ZCRUDP_RANS_PROFILES) {
        return false;
    }
    const uint16_t *cum = ZCRUDP_RANS_CUM[profile];
    uint32_t st[4];
    for (size_t lane = 0; lane < 4U; lane++) {
        size_t o = 1U + lane * 4U;
        st[lane] = (uint32_t)in[o + 0U] |
                   ((uint32_t)in[o + 1U] << 8) |
                   ((uint32_t)in[o + 2U] << 16) |
                   ((uint32_t)in[o + 3U] << 24);
        if (st[lane] < ZCRUDP_RANS_L) {
            return false;
        }
    }
    const uint8_t *ptr = in + 17U;
    const uint8_t *end = in + in_len;

    /* 4-way unrolled decode breaks serial dependency across lanes */
    size_t i = 0;
    for (; i + 4U <= num_syms; i += 4U) {
        for (size_t k = 0; k < 4U; k++) {
            uint32_t x = st[k];
            uint32_t cf = x & (ZCRUDP_RANS_SCALE - 1U);
            uint8_t s = zcrudp_rans_find_sym(cf, cum);
            uint32_t start = cum[s];
            uint32_t freq = cum[s + 1U] - start;
            x = freq * (x >> ZCRUDP_RANS_BITS) + (cf - start);
            while (x < ZCRUDP_RANS_L) {
                if (ptr >= end) return false;
                x = (x << 8) | (uint32_t)(*ptr++);
            }
            st[k] = x;
            syms_out[i + k] = s;
        }
    }
    for (; i < num_syms; i++) {
        uint32_t x = st[i & 3U];
        uint32_t cf = x & (ZCRUDP_RANS_SCALE - 1U);
        uint8_t s = zcrudp_rans_find_sym(cf, cum);
        uint32_t start = cum[s];
        uint32_t freq = cum[s + 1U] - start;
        x = freq * (x >> ZCRUDP_RANS_BITS) + (cf - start);
        while (x < ZCRUDP_RANS_L) {
            if (ptr >= end) return false;
            x = (x << 8) | (uint32_t)(*ptr++);
        }
        st[i & 3U] = x;
        syms_out[i] = s;
    }
    return true;
}

#ifdef __cplusplus
}
#endif

#endif /* ZCRUDP_TERNARY_RANS_H */
