/*
 * zcvid_latent_hybrid.h - Experimental benchmark harness comparing:
 *   - TQ4           : 4x4 H.264 integer transform + scalar quant + CABAC-style EC
 *   - LRF1 / LRF2   : Ultra-low-bit rank-1 / rank-2 ternary latent factorization on 8x8
 *   - HYBRID_RDO    : Per-8x8 RDO between {SKIP, TQ4, LRF1, LRF2}
 *   - HYBRID_CASCADE: Per-8x8 RDO adding LRF1 + TQ4 residual cascade
 *   - HYBRID_FULL   : Per-8x8 RDO adding spectral-domain LRF on 8x8 HEVC integer DCT
 *
 * Strictly portable C11 (uses zcvid_entropy.h & zcvid_gpu_cpu_dec.h), bit-exact
 * closed-loop integer reconstruction on every mode.
 */

#ifndef ZCVID_LATENT_HYBRID_H
#define ZCVID_LATENT_HYBRID_H

#include <math.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#include <intrin.h>
#if defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define ZCVID_LH_ALIGNAS64 _Alignas(64)
#elif defined(_MSC_VER)
#define ZCVID_LH_ALIGNAS64 __declspec(align(64))
#else
#define ZCVID_LH_ALIGNAS64 __attribute__((aligned(64)))
#endif

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <pthread.h>
#endif

#include "zcvid_entropy.h"
#include "zcvid_gpu_cpu_dec.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ZCVID_LH_TQ4            = 0,
    ZCVID_LH_LRF1           = 1,
    ZCVID_LH_LRF2           = 2,
    ZCVID_LH_HYBRID_RDO     = 3,
    ZCVID_LH_HYBRID_CASCADE = 4,
    ZCVID_LH_HYBRID_FULL    = 5,
    ZCVID_LH_HYBRID_FAST    = 6,
    ZCVID_LH_HYBRID_ESR     = 7,
    ZCVID_LH_CODEC_COUNT    = 8
} zcvid_lh_codec_t;

typedef enum {
    ZCVID_LH_BLK_SKIP     = 0,
    ZCVID_LH_BLK_TQ4      = 1,
    ZCVID_LH_BLK_LRF1     = 2,
    ZCVID_LH_BLK_LRF2     = 3,
    ZCVID_LH_BLK_CASCADE  = 4,
    ZCVID_LH_BLK_SPEC_LRF = 5,
    ZCVID_LH_BLK_POLY_ESR = 6
} zcvid_lh_blk_mode_t;

typedef struct {
    uint64_t count[7]; /* indexed by zcvid_lh_blk_mode_t */
} zcvid_lh_stats_t;

/* =========================================================================
 * Fast bit-cost Q8 estimator from live range-coder probability (Q15)
 * ========================================================================= */

static inline uint32_t zcvid_lh_cost_q8(const zcrudp_rc_prob_t *p, uint32_t bit) {
    /* cost(bit) = -log2(P(bit)) in Q8. Quantize P(bit) to 64 bins [1..63]. */
    static const uint16_t k_neglog2_q8[65] = {
        2560, 1536, 1280, 1130, 1024,  942,  874,  817,
         768,  725,  686,  650,  618,  588,  560,  534,
         512,  488,  468,  448,  429,  412,  395,  378,
         363,  348,  334,  320,  307,  294,  281,  269,
         256,  245,  234,  223,  213,  202,  192,  183,
         173,  164,  155,  147,  138,  130,  122,  114,
         106,   98,   91,   84,   77,   70,   63,   56,
          49,   43,   37,   30,   24,   18,   12,    6,
           0
    };
    uint32_t p0 = p->p0;
    uint32_t prob = bit ? (32768U - p0) : p0;
    uint32_t bin = prob >> 9;
    if (bin > 64U) bin = 64U;
    return k_neglog2_q8[bin];
}

static inline uint32_t zcvid_lh_egk_cost_q8(const zcrudp_rc_prob_t *ctx, uint32_t nctx,
                                            uint32_t max_len, uint64_t v) {
    if (max_len > 63U) max_len = 63U;
    if (v == 0ULL) {
        return (max_len > 0U) ? zcvid_lh_cost_q8(&ctx[0], 0U) : 0U;
    }
    uint64_t x = v + 1ULL;
    uint32_t n = 0U;
    while ((x >> (n + 1U)) != 0ULL) n++;
    if (n > max_len) n = max_len;
    uint32_t bits_q8 = 0U;
    for (uint32_t i = 0U; i < n; ++i) {
        bits_q8 += zcvid_lh_cost_q8(&ctx[(i < nctx) ? i : (nctx - 1U)], 1U);
    }
    if (n < max_len) {
        bits_q8 += zcvid_lh_cost_q8(&ctx[(n < nctx) ? n : (nctx - 1U)], 0U);
    }
    return bits_q8 + (n << 8);
}

/* =========================================================================
 * Entropy Context Model for All Modes
 * ========================================================================= */

typedef struct {
    zcrudp_rc_prob_t skip[4];       /* ctx: (left!=0) + 2*(up!=0) */
    zcrudp_rc_prob_t mode_b[5];     /* binary decisions for block mode */
    zcrudp_rc_prob_t tq_cbp[4];     /* per-quadrant non-zero flag */
    zcrudp_rc_prob_t tq_has_ac;
    zcrudp_rc_prob_t tq_last[14];
    zcrudp_rc_prob_t tq_sig[15];
    zcrudp_rc_prob_t tq_gt1[4];
    zcrudp_rc_prob_t tq_gt2[4];
    zcrudp_rc_prob_t tq_mag[4][7];

    zcrudp_rc_prob_t lrf_nz_u[8];   /* u sparsity mask bit per pos */
    zcrudp_rc_prob_t lrf_nz_v[8];   /* v sparsity mask bit per pos */
    zcrudp_rc_prob_t lrf_sign_eq;   /* sign matches previous non-zero element */
    zcrudp_rc_prob_t lrf_alpha_gt1;
    zcrudp_rc_prob_t lrf_alpha_gt2;
    zcrudp_rc_prob_t lrf_alpha_mag[7];

    /* Polynomial Prediction + Exact Spectral Repair (POLY_ESR) contexts */
    zcrudp_rc_prob_t esr_poly_only;
    zcrudp_rc_prob_t esr_poly_has_ac;
    zcrudp_rc_prob_t esr_poly_last[4];
    zcrudp_rc_prob_t esr_poly_sig[5];
    zcrudp_rc_prob_t esr_csbf[4];
    zcrudp_rc_prob_t esr_has_ac[4];
    zcrudp_rc_prob_t esr_last[4][14];
    zcrudp_rc_prob_t esr_sig[4][15];
    zcrudp_rc_prob_t esr_gt1[4];
    zcrudp_rc_prob_t esr_gt2[4];
    zcrudp_rc_prob_t esr_mag[4][7];
} zcvid_lh_model_t;

static inline void zcvid_lh_model_init(zcvid_lh_model_t *m) {
    static const uint16_t skip_p[4] = { 26000U, 16384U, 16384U, 10000U };
    for (int i = 0; i < 4; ++i) zcrudp_rc_prob_init(&m->skip[i], skip_p[i]);
    for (int i = 0; i < 5; ++i) zcrudp_rc_prob_init(&m->mode_b[i], 16384U);
    for (int i = 0; i < 4; ++i) zcrudp_rc_prob_init(&m->tq_cbp[i], 16384U);
    zcrudp_rc_prob_init(&m->tq_has_ac, 14000U);
    for (int i = 0; i < 14; ++i) zcrudp_rc_prob_init(&m->tq_last[i], 16384U);
    for (int k = 0; k < 15; ++k) {
        uint16_t p0 = (k == 0) ? 9000U : (k < 4) ? 15000U : 21000U;
        zcrudp_rc_prob_init(&m->tq_sig[k], p0);
    }
    for (int kb = 0; kb < 4; ++kb) {
        zcrudp_rc_prob_init(&m->tq_gt1[kb], (uint16_t)(19000 + kb * 2500));
        zcrudp_rc_prob_init(&m->tq_gt2[kb], (uint16_t)(21000 + kb * 2000));
        for (int c = 0; c < 7; ++c) {
            int32_t p0 = 14000 + kb * 3000 + c * 2000;
            if (p0 > 28000) p0 = 28000;
            zcrudp_rc_prob_init(&m->tq_mag[kb][c], (uint16_t)p0);
        }
    }
    for (int i = 0; i < 8; ++i) {
        zcrudp_rc_prob_init(&m->lrf_nz_u[i], 14000U);
        zcrudp_rc_prob_init(&m->lrf_nz_v[i], 14000U);
    }
    zcrudp_rc_prob_init(&m->lrf_sign_eq, 8000U); /* bit 0 = sign equals prev (frequent) */
    zcrudp_rc_prob_init(&m->lrf_alpha_gt1, 14000U);
    zcrudp_rc_prob_init(&m->lrf_alpha_gt2, 18000U);
    for (int c = 0; c < 7; ++c) {
        zcrudp_rc_prob_init(&m->lrf_alpha_mag[c], (uint16_t)(14000 + c * 2000));
    }

    /* POLY_ESR priors: degree-2 polynomial frequent; LL active, HF CSBF sparse */
    zcrudp_rc_prob_init(&m->esr_poly_only, 15000U);
    zcrudp_rc_prob_init(&m->esr_poly_has_ac, 12000U);
    for (int i = 0; i < 4; ++i) zcrudp_rc_prob_init(&m->esr_poly_last[i], 16384U);
    for (int k = 0; k < 5; ++k) {
        zcrudp_rc_prob_init(&m->esr_poly_sig[k], (k == 0) ? 8000U : 15000U);
    }
    static const uint16_t csbf_p[4] = { 4000U, 24000U, 24000U, 26000U };
    for (int sb = 0; sb < 4; ++sb) {
        zcrudp_rc_prob_init(&m->esr_csbf[sb], csbf_p[sb]);
        zcrudp_rc_prob_init(&m->esr_has_ac[sb], (sb == 0) ? 10000U : 16384U);
        for (int i = 0; i < 14; ++i) zcrudp_rc_prob_init(&m->esr_last[sb][i], 16384U);
        for (int k = 0; k < 15; ++k) {
            uint16_t p0 = (sb == 0 && k == 0) ? 8000U : (k < 4) ? 16000U : 22000U;
            zcrudp_rc_prob_init(&m->esr_sig[sb][k], p0);
        }
    }
    for (int kb = 0; kb < 4; ++kb) {
        zcrudp_rc_prob_init(&m->esr_gt1[kb], (uint16_t)(17000 + kb * 3000));
        zcrudp_rc_prob_init(&m->esr_gt2[kb], (uint16_t)(20000 + kb * 2500));
        for (int c = 0; c < 7; ++c) {
            int32_t p0 = 14000 + kb * 3000 + c * 2000;
            if (p0 > 28000) p0 = 28000;
            zcrudp_rc_prob_init(&m->esr_mag[kb][c], (uint16_t)p0);
        }
    }
}

/* =========================================================================
 * Exact 4x4 H.264 Integer Transform & Quantizer
 * ========================================================================= */

static inline void zcvid_lh_tq4_forward(const int32_t res[16], int32_t out_w[16]) {
    int32_t t[16];
    for (uint32_t i = 0U; i < 4U; ++i) {
        int32_t s0 = res[i * 4U + 0U], s1 = res[i * 4U + 1U];
        int32_t s2 = res[i * 4U + 2U], s3 = res[i * 4U + 3U];
        int32_t p0 = s0 + s3, p1 = s1 + s2;
        int32_t p2 = s1 - s2, p3 = s0 - s3;
        t[i * 4U + 0U] = p0 + p1;
        t[i * 4U + 1U] = (p3 * 2) + p2;
        t[i * 4U + 2U] = p0 - p1;
        t[i * 4U + 3U] = p3 - (p2 * 2);
    }
    for (uint32_t j = 0U; j < 4U; ++j) {
        int32_t s0 = t[0U + j], s1 = t[4U + j];
        int32_t s2 = t[8U + j], s3 = t[12U + j];
        int32_t p0 = s0 + s3, p1 = s1 + s2;
        int32_t p2 = s1 - s2, p3 = s0 - s3;
        out_w[0U + j]  = p0 + p1;
        out_w[4U + j]  = (p3 * 2) + p2;
        out_w[8U + j]  = p0 - p1;
        out_w[12U + j] = p3 - (p2 * 2);
    }
}

static inline void zcvid_lh_tq4_quant(const int32_t w[16], uint32_t qp, bool is_inter, int16_t lv[16]) {
    static const int32_t TQ_MF[18] = {
        13107, 5243, 8066,  11916, 4660, 7490,  10082, 4194, 6554,
         9362, 3647, 5825,   8192, 3355, 5243,   7282, 2893, 4559
    };
    static const uint32_t TQ_ZZ[16] = {
        0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15
    };
    const uint32_t qbits = 15U + (qp / 6U);
    const int64_t dz = (int64_t)((1ULL << qbits) / (is_inter ? 6ULL : 3ULL));
    const uint32_t qp_mod = qp % 6U;
    for (uint32_t k = 0U; k < 16U; ++k) {
        uint32_t r = TQ_ZZ[k];
        int32_t c = w[r];
        int32_t ac = (c < 0) ? -c : c;
        int32_t mf = TQ_MF[qp_mod * 3U + zcvid_dec_tq_class(r)];
        int32_t mag = (int32_t)(((int64_t)ac * (int64_t)mf + dz) >> qbits);
        if (mag > 511) mag = 511;
        lv[k] = (int16_t)((c < 0) ? -mag : mag);
    }
}

static inline void zcvid_lh_tq4_recon(const int16_t lv[16], uint32_t qp, int32_t out_r[16]) {
    static const uint32_t TQ_ZZ[16] = {
        0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15
    };
    int32_t c[16];
    memset(c, 0, sizeof(c));
    for (uint32_t k = 0U; k < 16U; ++k) {
        uint32_t r = TQ_ZZ[k];
        c[r] = zcvid_dec_tq_dequant((int32_t)lv[k], r, qp);
    }
    zcvid_dec_tq_inverse(c);
    memcpy(out_r, c, sizeof(c));
}

static inline uint32_t zcvid_lh_tq_kb(uint32_t k) {
    static const uint8_t lut[16] = { 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3 };
    return lut[k & 15U];
}

static uint32_t zcvid_lh_tq4_rate_q8(const zcvid_lh_model_t *m, const int16_t lv[16]) {
    uint32_t sig_mask = 0U, last_k = 0U;
    for (uint32_t k = 0U; k < 16U; ++k) {
        if (lv[k] != 0) {
            sig_mask |= (1U << k);
            last_k = k;
        }
    }
    if (sig_mask == 0U) return 0U;
    uint32_t bits_q8 = 0U;
    uint32_t has_ac = (last_k > 0U) ? 1U : 0U;
    bits_q8 += zcvid_lh_cost_q8(&m->tq_has_ac, has_ac);
    if (has_ac) {
        uint32_t v = last_k - 1U;
        uint32_t b3 = (v >> 3) & 1U; bits_q8 += zcvid_lh_cost_q8(&m->tq_last[0], b3);
        uint32_t n2 = 2U | b3;
        uint32_t b2 = (v >> 2) & 1U; bits_q8 += zcvid_lh_cost_q8(&m->tq_last[n2 - 1U], b2);
        uint32_t n1 = (n2 << 1) | b2;
        uint32_t b1 = (v >> 1) & 1U; bits_q8 += zcvid_lh_cost_q8(&m->tq_last[n1 - 1U], b1);
        uint32_t n0 = (n1 << 1) | b1;
        if (n0 != 15U) bits_q8 += zcvid_lh_cost_q8(&m->tq_last[n0 - 1U], v & 1U);
        for (uint32_t k = 0U; k < last_k; ++k) {
            bits_q8 += zcvid_lh_cost_q8(&m->tq_sig[k], (sig_mask >> k) & 1U);
        }
    }
    for (uint32_t k = 0U; k <= last_k; ++k) {
        if (((sig_mask >> k) & 1U) == 0U) continue;
        int32_t val = lv[k];
        bits_q8 += 256U; /* sign bypass 1.0 bit */
        uint32_t mag = (uint32_t)((val < 0) ? -val : val);
        uint32_t kb = zcvid_lh_tq_kb(k);
        uint32_t gt1 = (mag > 1U) ? 1U : 0U;
        bits_q8 += zcvid_lh_cost_q8(&m->tq_gt1[kb], gt1);
        if (gt1) {
            uint32_t gt2 = (mag > 2U) ? 1U : 0U;
            bits_q8 += zcvid_lh_cost_q8(&m->tq_gt2[kb], gt2);
            if (gt2) {
                bits_q8 += zcvid_lh_egk_cost_q8(m->tq_mag[kb], 7U, 9U, (uint64_t)(mag - 3U));
            }
        }
    }
    return bits_q8;
}

static void zcvid_lh_tq4_encode_quad(zcrudp_rc_enc_t *rc, zcvid_lh_model_t *m, const int16_t lv[16]) {
    uint32_t sig_mask = 0U, last_k = 0U;
    for (uint32_t k = 0U; k < 16U; ++k) {
        if (lv[k] != 0) {
            sig_mask |= (1U << k);
            last_k = k;
        }
    }
    uint32_t has_ac = (last_k > 0U) ? 1U : 0U;
    zcrudp_rc_enc_bit(rc, &m->tq_has_ac, has_ac);
    if (has_ac) {
        uint32_t v = last_k - 1U;
        zcrudp_rc_prob_t *lk = m->tq_last;
        uint32_t b3 = (v >> 3) & 1U; zcrudp_rc_enc_bit(rc, &lk[0], b3);
        uint32_t n2 = 2U | b3;
        uint32_t b2 = (v >> 2) & 1U; zcrudp_rc_enc_bit(rc, &lk[n2 - 1U], b2);
        uint32_t n1 = (n2 << 1) | b2;
        uint32_t b1 = (v >> 1) & 1U; zcrudp_rc_enc_bit(rc, &lk[n1 - 1U], b1);
        uint32_t n0 = (n1 << 1) | b1;
        if (n0 != 15U) zcrudp_rc_enc_bit(rc, &lk[n0 - 1U], v & 1U);
        for (uint32_t k = 0U; k < last_k; ++k) {
            zcrudp_rc_enc_bit(rc, &m->tq_sig[k], (sig_mask >> k) & 1U);
        }
    }
    for (uint32_t k = 0U; k <= last_k; ++k) {
        if (((sig_mask >> k) & 1U) == 0U) continue;
        int32_t val = lv[k];
        uint32_t neg = (val < 0) ? 1U : 0U;
        zcrudp_rc_enc_bit_bypass(rc, neg);
        uint32_t mag = (uint32_t)(neg ? -val : val);
        uint32_t kb = zcvid_lh_tq_kb(k);
        uint32_t gt1 = (mag > 1U) ? 1U : 0U;
        zcrudp_rc_enc_bit(rc, &m->tq_gt1[kb], gt1);
        if (gt1) {
            uint32_t gt2 = (mag > 2U) ? 1U : 0U;
            zcrudp_rc_enc_bit(rc, &m->tq_gt2[kb], gt2);
            if (gt2) {
                zcrudp_rc_enc_egk(rc, m->tq_mag[kb], 7U, 9U, (uint64_t)(mag - 3U));
            }
        }
    }
}

static bool zcvid_lh_tq4_decode_quad(zcrudp_rc_dec_t *rc, zcvid_lh_model_t *m, int16_t lv[16]) {
    memset(lv, 0, sizeof(int16_t) * 16U);
    uint32_t sig_mask = 1U;
    if (zcrudp_rc_dec_bit(rc, &m->tq_has_ac) != 0U) {
        zcrudp_rc_prob_t *lk = m->tq_last;
        uint32_t node = 2U | zcrudp_rc_dec_bit(rc, &lk[0]);
        node = (node << 1) | zcrudp_rc_dec_bit(rc, &lk[node - 1U]);
        node = (node << 1) | zcrudp_rc_dec_bit(rc, &lk[node - 1U]);
        node = (node == 15U) ? 30U : ((node << 1) | zcrudp_rc_dec_bit(rc, &lk[node - 1U]));
        uint32_t last_k = node - 15U;
        sig_mask = 1U << last_k;
        for (uint32_t k = 0U; k < last_k; ++k) {
            sig_mask |= zcrudp_rc_dec_bit(rc, &m->tq_sig[k]) << k;
        }
    }
    for (uint32_t k = 0U; k < 16U; ++k) {
        if (((sig_mask >> k) & 1U) == 0U) continue;
        uint32_t neg = zcrudp_rc_dec_bit_bypass(rc);
        uint32_t kb = zcvid_lh_tq_kb(k);
        int32_t mag = 1;
        if (zcrudp_rc_dec_bit(rc, &m->tq_gt1[kb]) != 0U) {
            mag = 2;
            if (zcrudp_rc_dec_bit(rc, &m->tq_gt2[kb]) != 0U) {
                mag = (int32_t)zcrudp_rc_dec_egk(rc, m->tq_mag[kb], 7U, 9U) + 3;
                if (mag > 511) return false;
            }
        }
        lv[k] = (int16_t)(neg ? -mag : mag);
    }
    return true;
}

/* =========================================================================
 * Exact HEVC 8x8 Integer DCT & IDCT — Factored Linear Circuits (Lean 4 Proved)
 * Forward: row >> 2, col >> 9.  Inverse: row >> 7, col >> 12.
 *
 * Scalar & AVX2 paths: Unified 12-mul exact Negacyclic-Karatsuba (mod X^4+1)
 *   + Gauss 3-mul rotation for BOTH forward and inverse via M_odd^T = M_odd
 *   duality (40% fewer muls than 20-mul butterfly, 5.3x fewer than 64-MAC ref),
 *   plus bit-exact sparse/CSBF/poly_only IDCT8 and 64-bit overflow guards.
 * Formal proofs: zcvid/proofs/lean4/ZcvidDCT8Circuit.lean (0 sorry, 0 axioms).
 * ========================================================================= */

static const int32_t k_hevc_dct8[8][8] = {
    { 64,  64,  64,  64,  64,  64,  64,  64 },
    { 89,  75,  50,  18, -18, -50, -75, -89 },
    { 83,  36, -36, -83, -83, -36,  36,  83 },
    { 75, -18, -89, -50,  50,  89,  18, -75 },
    { 64, -64, -64,  64,  64, -64, -64,  64 },
    { 50, -89,  18,  75, -75, -18,  89, -50 },
    { 36, -83,  83, -36, -36,  83, -83,  36 },
    { 18, -50,  75, -89,  89, -75,  50, -18 }
};

static inline void zcvid_lh_dct8_fwd_ref(const int32_t in[64], int32_t out[64]) {
    int64_t tmp[64];
    for (uint32_t y = 0U; y < 8U; ++y) {
        for (uint32_t u = 0U; u < 8U; ++u) {
            int64_t s = 0;
            for (uint32_t x = 0U; x < 8U; ++x) {
                s += (int64_t)in[y * 8U + x] * (int64_t)k_hevc_dct8[u][x];
            }
            tmp[y * 8U + u] = (s + 2) >> 2;
        }
    }
    for (uint32_t v = 0U; v < 8U; ++v) {
        for (uint32_t u = 0U; u < 8U; ++u) {
            int64_t s = 0;
            for (uint32_t y = 0U; y < 8U; ++y) {
                s += tmp[y * 8U + u] * (int64_t)k_hevc_dct8[v][y];
            }
            out[v * 8U + u] = (int32_t)((s + 256) >> 9);
        }
    }
}

static inline void zcvid_lh_dct8_inv_ref(const int32_t in[64], int32_t out[64]) {
    int64_t tmp[64];
    for (uint32_t v = 0U; v < 8U; ++v) {
        for (uint32_t x = 0U; x < 8U; ++x) {
            int64_t s = 0;
            for (uint32_t u = 0U; u < 8U; ++u) {
                s += (int64_t)in[v * 8U + u] * (int64_t)k_hevc_dct8[u][x];
            }
            tmp[v * 8U + x] = (s + 64) >> 7;
        }
    }
    for (uint32_t y = 0U; y < 8U; ++y) {
        for (uint32_t x = 0U; x < 8U; ++x) {
            int64_t s = 0;
            for (uint32_t v = 0U; v < 8U; ++v) {
                s += tmp[v * 8U + x] * (int64_t)k_hevc_dct8[v][y];
            }
            out[y * 8U + x] = (int32_t)((s + 2048) >> 12);
        }
    }
}

/* Exact 12-mul scalar 1D forward HEVC DCT8 (Negacyclic-Karatsuba + Gauss rotation;
 * falls back to 64-bit ref if |in| > 4095, zero int32 overflow proved in Lean 4). */
static inline void zcvid_lh_dct8_fwd_scalar(const int32_t in[64], int32_t out[64]) {
    for (uint32_t i = 0U; i < 64U; ++i) {
        if (in[i] > 4095 || in[i] < -4095) {
            zcvid_lh_dct8_fwd_ref(in, out);
            return;
        }
    }
    int32_t tmp[64];
    for (uint32_t y = 0U; y < 8U; ++y) {
        const int32_t *r = in + y * 8U;
        int32_t e0 = r[0] + r[7], e1 = r[1] + r[6], e2 = r[2] + r[5], e3 = r[3] + r[4];
        int32_t o0 = r[0] - r[7], o1 = r[1] - r[6], o2 = r[2] - r[5], o3 = r[3] - r[4];
        int32_t ee0 = e0 + e3, ee1 = e1 + e2;
        int32_t eo0 = e0 - e3, eo1 = e1 - e2;
        int32_t pev = 36 * (eo0 + eo1);
        /* 9-mul Negacyclic-Karatsuba mod (X^4 + 1) on (o0, o2, o3, o1) */
        int32_t s0 = o0 + o3, s1 = o2 + o1;
        int32_t m0 = 89 * o0, m1 = -75 * o2, m2 = 14 * (o0 + o2);
        int32_t m3 = -18 * o3, m4 = -50 * o1, m5 = -68 * (o3 + o1);
        int32_t m6 = 71 * s0, m7 = -125 * s1, m8 = -54 * (s0 + s1);
        int32_t u0 = m0 + m3, d0 = m0 - m3;
        int32_t u1 = m1 + m4, d1 = m1 - m4;
        int32_t u2 = m2 + m5, d2 = m2 - m5;
        int32_t k0 = m6 - u0, k1 = m7 - u1;
        tmp[y * 8U + 0U] = (64 * (ee0 + ee1) + 2) >> 2;
        tmp[y * 8U + 4U] = (64 * (ee0 - ee1) + 2) >> 2;
        tmp[y * 8U + 2U] = (pev + 47 * eo0 + 2) >> 2;
        tmp[y * 8U + 6U] = (pev - 119 * eo1 + 2) >> 2;
        tmp[y * 8U + 1U] = (d0 - k1 + 2) >> 2;
        tmp[y * 8U + 3U] = (d0 + d1 - d2 + 2) >> 2;
        tmp[y * 8U + 5U] = (k0 + k1 + u2 - m8 + 2) >> 2;
        tmp[y * 8U + 7U] = (-k0 - d1 + 2) >> 2;
    }
    for (uint32_t u = 0U; u < 8U; ++u) {
        int32_t e0 = tmp[0U * 8U + u] + tmp[7U * 8U + u];
        int32_t e1 = tmp[1U * 8U + u] + tmp[6U * 8U + u];
        int32_t e2 = tmp[2U * 8U + u] + tmp[5U * 8U + u];
        int32_t e3 = tmp[3U * 8U + u] + tmp[4U * 8U + u];
        int32_t o0 = tmp[0U * 8U + u] - tmp[7U * 8U + u];
        int32_t o1 = tmp[1U * 8U + u] - tmp[6U * 8U + u];
        int32_t o2 = tmp[2U * 8U + u] - tmp[5U * 8U + u];
        int32_t o3 = tmp[3U * 8U + u] - tmp[4U * 8U + u];
        int32_t ee0 = e0 + e3, ee1 = e1 + e2;
        int32_t eo0 = e0 - e3, eo1 = e1 - e2;
        int32_t pev = 36 * (eo0 + eo1);
        int32_t s0 = o0 + o3, s1 = o2 + o1;
        int32_t m0 = 89 * o0, m1 = -75 * o2, m2 = 14 * (o0 + o2);
        int32_t m3 = -18 * o3, m4 = -50 * o1, m5 = -68 * (o3 + o1);
        int32_t m6 = 71 * s0, m7 = -125 * s1, m8 = -54 * (s0 + s1);
        int32_t u0 = m0 + m3, d0 = m0 - m3;
        int32_t u1 = m1 + m4, d1 = m1 - m4;
        int32_t u2 = m2 + m5, d2 = m2 - m5;
        int32_t k0 = m6 - u0, k1 = m7 - u1;
        out[0U * 8U + u] = (64 * (ee0 + ee1) + 256) >> 9;
        out[4U * 8U + u] = (64 * (ee0 - ee1) + 256) >> 9;
        out[2U * 8U + u] = (pev + 47 * eo0 + 256) >> 9;
        out[6U * 8U + u] = (pev - 119 * eo1 + 256) >> 9;
        out[1U * 8U + u] = (d0 - k1 + 256) >> 9;
        out[3U * 8U + u] = (d0 + d1 - d2 + 256) >> 9;
        out[5U * 8U + u] = (k0 + k1 + u2 - m8 + 256) >> 9;
        out[7U * 8U + u] = (-k0 - d1 + 256) >> 9;
    }
}

/* Exact 12-mul scalar 2D inverse HEVC IDCT8 (reuses identical Negacyclic-Karatsuba
 * circuit by M_odd^T = M_odd duality; falls back to 64-bit ref if |in| > 65535). */
static inline void zcvid_lh_dct8_inv_scalar(const int32_t in[64], int32_t out[64]) {
    for (uint32_t i = 0U; i < 64U; ++i) {
        if (in[i] > 65535 || in[i] < -65535) {
            zcvid_lh_dct8_inv_ref(in, out);
            return;
        }
    }
    int32_t tmp[64];
    for (uint32_t v = 0U; v < 8U; ++v) {
        const int32_t *c = in + v * 8U;
        if ((c[0] | c[1] | c[2] | c[3] | c[4] | c[5] | c[6] | c[7]) == 0) {
            memset(tmp + v * 8U, 0, sizeof(int32_t) * 8U);
            continue;
        }
        int32_t e0 = 64 * (c[0] + c[4]);
        int32_t e1 = 64 * (c[0] - c[4]);
        int32_t pev = 36 * (c[2] + c[6]);
        int32_t g0 = pev + 47 * c[2];
        int32_t g1 = pev - 119 * c[6];
        int32_t E0 = e0 + g0, E1 = e1 + g1, E2 = e1 - g1, E3 = e0 - g0;
        int32_t o0 = c[1], o1 = c[3], o2 = c[5], o3 = c[7];
        int32_t s0 = o0 + o3, s1 = o2 + o1;
        int32_t m0 = 89 * o0, m1 = -75 * o2, m2 = 14 * (o0 + o2);
        int32_t m3 = -18 * o3, m4 = -50 * o1, m5 = -68 * (o3 + o1);
        int32_t m6 = 71 * s0, m7 = -125 * s1, m8 = -54 * (s0 + s1);
        int32_t u0 = m0 + m3, d0 = m0 - m3;
        int32_t u1 = m1 + m4, d1 = m1 - m4;
        int32_t u2 = m2 + m5, d2 = m2 - m5;
        int32_t k0 = m6 - u0, k1 = m7 - u1;
        int32_t O0 = d0 - k1, O1 = d0 + d1 - d2, O2 = k0 + k1 + u2 - m8, O3 = -k0 - d1;
        tmp[v * 8U + 0U] = (E0 + O0 + 64) >> 7;
        tmp[v * 8U + 1U] = (E1 + O1 + 64) >> 7;
        tmp[v * 8U + 2U] = (E2 + O2 + 64) >> 7;
        tmp[v * 8U + 3U] = (E3 + O3 + 64) >> 7;
        tmp[v * 8U + 4U] = (E3 - O3 + 64) >> 7;
        tmp[v * 8U + 5U] = (E2 - O2 + 64) >> 7;
        tmp[v * 8U + 6U] = (E1 - O1 + 64) >> 7;
        tmp[v * 8U + 7U] = (E0 - O0 + 64) >> 7;
    }
    for (uint32_t x = 0U; x < 8U; ++x) {
        int32_t c0 = tmp[0U * 8U + x], c1 = tmp[1U * 8U + x];
        int32_t c2 = tmp[2U * 8U + x], c3 = tmp[3U * 8U + x];
        int32_t c4 = tmp[4U * 8U + x], c5 = tmp[5U * 8U + x];
        int32_t c6 = tmp[6U * 8U + x], c7 = tmp[7U * 8U + x];
        int32_t e0 = 64 * (c0 + c4);
        int32_t e1 = 64 * (c0 - c4);
        int32_t pev = 36 * (c2 + c6);
        int32_t g0 = pev + 47 * c2;
        int32_t g1 = pev - 119 * c6;
        int32_t E0 = e0 + g0, E1 = e1 + g1, E2 = e1 - g1, E3 = e0 - g0;
        int32_t s0 = c1 + c7, s1 = c5 + c3;
        int32_t m0 = 89 * c1, m1 = -75 * c5, m2 = 14 * (c1 + c5);
        int32_t m3 = -18 * c7, m4 = -50 * c3, m5 = -68 * (c7 + c3);
        int32_t m6 = 71 * s0, m7 = -125 * s1, m8 = -54 * (s0 + s1);
        int32_t u0 = m0 + m3, d0 = m0 - m3;
        int32_t u1 = m1 + m4, d1 = m1 - m4;
        int32_t u2 = m2 + m5, d2 = m2 - m5;
        int32_t k0 = m6 - u0, k1 = m7 - u1;
        int32_t O0 = d0 - k1, O1 = d0 + d1 - d2, O2 = k0 + k1 + u2 - m8, O3 = -k0 - d1;
        out[0U * 8U + x] = (E0 + O0 + 2048) >> 12;
        out[1U * 8U + x] = (E1 + O1 + 2048) >> 12;
        out[2U * 8U + x] = (E2 + O2 + 2048) >> 12;
        out[3U * 8U + x] = (E3 + O3 + 2048) >> 12;
        out[4U * 8U + x] = (E3 - O3 + 2048) >> 12;
        out[5U * 8U + x] = (E2 - O2 + 2048) >> 12;
        out[6U * 8U + x] = (E1 - O1 + 2048) >> 12;
        out[7U * 8U + x] = (E0 - O0 + 2048) >> 12;
    }
}

#if defined(__AVX2__)
static inline void zcvid_lh_transpose8x8_avx2(__m256i r[8]) {
    __m256i t0 = _mm256_unpacklo_epi32(r[0], r[1]);
    __m256i t1 = _mm256_unpackhi_epi32(r[0], r[1]);
    __m256i t2 = _mm256_unpacklo_epi32(r[2], r[3]);
    __m256i t3 = _mm256_unpackhi_epi32(r[2], r[3]);
    __m256i t4 = _mm256_unpacklo_epi32(r[4], r[5]);
    __m256i t5 = _mm256_unpackhi_epi32(r[4], r[5]);
    __m256i t6 = _mm256_unpacklo_epi32(r[6], r[7]);
    __m256i t7 = _mm256_unpackhi_epi32(r[6], r[7]);

    __m256i s0 = _mm256_unpacklo_epi64(t0, t2);
    __m256i s1 = _mm256_unpackhi_epi64(t0, t2);
    __m256i s2 = _mm256_unpacklo_epi64(t1, t3);
    __m256i s3 = _mm256_unpackhi_epi64(t1, t3);
    __m256i s4 = _mm256_unpacklo_epi64(t4, t6);
    __m256i s5 = _mm256_unpackhi_epi64(t4, t6);
    __m256i s6 = _mm256_unpacklo_epi64(t5, t7);
    __m256i s7 = _mm256_unpackhi_epi64(t5, t7);

    r[0] = _mm256_permute2x128_si256(s0, s4, 0x20);
    r[1] = _mm256_permute2x128_si256(s1, s5, 0x20);
    r[2] = _mm256_permute2x128_si256(s2, s6, 0x20);
    r[3] = _mm256_permute2x128_si256(s3, s7, 0x20);
    r[4] = _mm256_permute2x128_si256(s0, s4, 0x31);
    r[5] = _mm256_permute2x128_si256(s1, s5, 0x31);
    r[6] = _mm256_permute2x128_si256(s2, s6, 0x31);
    r[7] = _mm256_permute2x128_si256(s3, s7, 0x31);
}

/* Shared-routing 12-mul Negacyclic-Karatsuba + Gauss odd/even 1D AVX2 (int32). */
static inline void zcvid_lh_odd4_karatsuba_avx2(__m256i o0, __m256i o1, __m256i o2, __m256i o3,
                                                __m256i *y1, __m256i *y3, __m256i *y5, __m256i *y7) {
    __m256i s0 = _mm256_add_epi32(o0, o3);
    __m256i s1 = _mm256_add_epi32(o2, o1);
    __m256i m0 = _mm256_mullo_epi32(o0, _mm256_set1_epi32(89));
    __m256i m1 = _mm256_mullo_epi32(o2, _mm256_set1_epi32(-75));
    __m256i m2 = _mm256_mullo_epi32(_mm256_add_epi32(o0, o2), _mm256_set1_epi32(14));
    __m256i m3 = _mm256_mullo_epi32(o3, _mm256_set1_epi32(-18));
    __m256i m4 = _mm256_mullo_epi32(o1, _mm256_set1_epi32(-50));
    __m256i m5 = _mm256_mullo_epi32(_mm256_add_epi32(o3, o1), _mm256_set1_epi32(-68));
    __m256i m6 = _mm256_mullo_epi32(s0, _mm256_set1_epi32(71));
    __m256i m7 = _mm256_mullo_epi32(s1, _mm256_set1_epi32(-125));
    __m256i m8 = _mm256_mullo_epi32(_mm256_add_epi32(s0, s1), _mm256_set1_epi32(-54));
    __m256i u0 = _mm256_add_epi32(m0, m3), d0 = _mm256_sub_epi32(m0, m3);
    __m256i u1 = _mm256_add_epi32(m1, m4), d1 = _mm256_sub_epi32(m1, m4);
    __m256i u2 = _mm256_add_epi32(m2, m5), d2 = _mm256_sub_epi32(m2, m5);
    __m256i k0 = _mm256_sub_epi32(m6, u0);
    __m256i k1 = _mm256_sub_epi32(m7, u1);
    *y1 = _mm256_sub_epi32(d0, k1);
    *y3 = _mm256_sub_epi32(_mm256_add_epi32(d0, d1), d2);
    *y5 = _mm256_add_epi32(_mm256_add_epi32(k0, k1), _mm256_sub_epi32(u2, m8));
    *y7 = _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_add_epi32(k0, d1));
}

static inline void zcvid_lh_dct8_1d_avx2(__m256i s[8], int32_t rnd_val, int shift) {
    const __m256i rnd = _mm256_set1_epi32(rnd_val);
    __m256i e0 = _mm256_add_epi32(s[0], s[7]);
    __m256i e1 = _mm256_add_epi32(s[1], s[6]);
    __m256i e2 = _mm256_add_epi32(s[2], s[5]);
    __m256i e3 = _mm256_add_epi32(s[3], s[4]);
    __m256i o0 = _mm256_sub_epi32(s[0], s[7]);
    __m256i o1 = _mm256_sub_epi32(s[1], s[6]);
    __m256i o2 = _mm256_sub_epi32(s[2], s[5]);
    __m256i o3 = _mm256_sub_epi32(s[3], s[4]);

    __m256i ee0 = _mm256_add_epi32(e0, e3);
    __m256i ee1 = _mm256_add_epi32(e1, e2);
    __m256i eo0 = _mm256_sub_epi32(e0, e3);
    __m256i eo1 = _mm256_sub_epi32(e1, e2);

    __m256i y0 = _mm256_slli_epi32(_mm256_add_epi32(ee0, ee1), 6);
    __m256i y4 = _mm256_slli_epi32(_mm256_sub_epi32(ee0, ee1), 6);
    __m256i pev = _mm256_mullo_epi32(_mm256_add_epi32(eo0, eo1), _mm256_set1_epi32(36));
    __m256i y2 = _mm256_add_epi32(pev, _mm256_mullo_epi32(eo0, _mm256_set1_epi32(47)));
    __m256i y6 = _mm256_sub_epi32(pev, _mm256_mullo_epi32(eo1, _mm256_set1_epi32(119)));

    __m256i y1, y3, y5, y7;
    zcvid_lh_odd4_karatsuba_avx2(o0, o1, o2, o3, &y1, &y3, &y5, &y7);

    if (shift == 2) {
        s[0] = _mm256_srai_epi32(_mm256_add_epi32(y0, rnd), 2);
        s[1] = _mm256_srai_epi32(_mm256_add_epi32(y1, rnd), 2);
        s[2] = _mm256_srai_epi32(_mm256_add_epi32(y2, rnd), 2);
        s[3] = _mm256_srai_epi32(_mm256_add_epi32(y3, rnd), 2);
        s[4] = _mm256_srai_epi32(_mm256_add_epi32(y4, rnd), 2);
        s[5] = _mm256_srai_epi32(_mm256_add_epi32(y5, rnd), 2);
        s[6] = _mm256_srai_epi32(_mm256_add_epi32(y6, rnd), 2);
        s[7] = _mm256_srai_epi32(_mm256_add_epi32(y7, rnd), 2);
    } else {
        s[0] = _mm256_srai_epi32(_mm256_add_epi32(y0, rnd), 9);
        s[1] = _mm256_srai_epi32(_mm256_add_epi32(y1, rnd), 9);
        s[2] = _mm256_srai_epi32(_mm256_add_epi32(y2, rnd), 9);
        s[3] = _mm256_srai_epi32(_mm256_add_epi32(y3, rnd), 9);
        s[4] = _mm256_srai_epi32(_mm256_add_epi32(y4, rnd), 9);
        s[5] = _mm256_srai_epi32(_mm256_add_epi32(y5, rnd), 9);
        s[6] = _mm256_srai_epi32(_mm256_add_epi32(y6, rnd), 9);
        s[7] = _mm256_srai_epi32(_mm256_add_epi32(y7, rnd), 9);
    }
}

/* Native 32-bit AVX2 12-mul Negacyclic-Karatsuba + Gauss forward 8x8 DCT
 * (exact for |in| <= 4095, Lean-4 proved; INT32_MIN-safe range guard -> 64-bit ref). */
static inline void zcvid_lh_dct8_fwd(const int32_t in[64], int32_t out[64]) {
    __m256i v[8];
    const __m256i lo = _mm256_set1_epi32(-4095);
    const __m256i hi = _mm256_set1_epi32(4095);
    __m256i bad = _mm256_setzero_si256();
    for (int i = 0; i < 8; ++i) {
        v[i] = _mm256_loadu_si256((const __m256i *)(in + i * 8));
        bad = _mm256_or_si256(bad, _mm256_or_si256(_mm256_cmpgt_epi32(v[i], hi),
                                                   _mm256_cmpgt_epi32(lo, v[i])));
    }
    if (!_mm256_testz_si256(bad, bad)) {
        zcvid_lh_dct8_fwd_ref(in, out);
        return;
    }
    zcvid_lh_transpose8x8_avx2(v);
    zcvid_lh_dct8_1d_avx2(v, 2, 2);
    zcvid_lh_transpose8x8_avx2(v);
    zcvid_lh_dct8_1d_avx2(v, 256, 9);
    for (int i = 0; i < 8; ++i) _mm256_storeu_si256((__m256i *)(out + i * 8), v[i]);
}

/* 12-mul Negacyclic-Karatsuba + Gauss 1D inverse AVX2 (int32). */
static inline void zcvid_lh_dct8_inv_1d_avx2(__m256i c[8], int32_t rnd_val, int shift) {
    const __m256i rnd = _mm256_set1_epi32(rnd_val);
    __m256i e0 = _mm256_slli_epi32(_mm256_add_epi32(c[0], c[4]), 6);
    __m256i e1 = _mm256_slli_epi32(_mm256_sub_epi32(c[0], c[4]), 6);
    __m256i pev = _mm256_mullo_epi32(_mm256_add_epi32(c[2], c[6]), _mm256_set1_epi32(36));
    __m256i g0 = _mm256_add_epi32(pev, _mm256_mullo_epi32(c[2], _mm256_set1_epi32(47)));
    __m256i g1 = _mm256_sub_epi32(pev, _mm256_mullo_epi32(c[6], _mm256_set1_epi32(119)));
    __m256i E0 = _mm256_add_epi32(e0, g0);
    __m256i E1 = _mm256_add_epi32(e1, g1);
    __m256i E2 = _mm256_sub_epi32(e1, g1);
    __m256i E3 = _mm256_sub_epi32(e0, g0);

    __m256i O0, O1, O2, O3;
    zcvid_lh_odd4_karatsuba_avx2(c[1], c[3], c[5], c[7], &O0, &O1, &O2, &O3);

    if (shift == 7) {
        c[0] = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(E0, O0), rnd), 7);
        c[1] = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(E1, O1), rnd), 7);
        c[2] = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(E2, O2), rnd), 7);
        c[3] = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(E3, O3), rnd), 7);
        c[4] = _mm256_srai_epi32(_mm256_add_epi32(_mm256_sub_epi32(E3, O3), rnd), 7);
        c[5] = _mm256_srai_epi32(_mm256_add_epi32(_mm256_sub_epi32(E2, O2), rnd), 7);
        c[6] = _mm256_srai_epi32(_mm256_add_epi32(_mm256_sub_epi32(E1, O1), rnd), 7);
        c[7] = _mm256_srai_epi32(_mm256_add_epi32(_mm256_sub_epi32(E0, O0), rnd), 7);
    } else {
        c[0] = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(E0, O0), rnd), 12);
        c[1] = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(E1, O1), rnd), 12);
        c[2] = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(E2, O2), rnd), 12);
        c[3] = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(E3, O3), rnd), 12);
        c[4] = _mm256_srai_epi32(_mm256_add_epi32(_mm256_sub_epi32(E3, O3), rnd), 12);
        c[5] = _mm256_srai_epi32(_mm256_add_epi32(_mm256_sub_epi32(E2, O2), rnd), 12);
        c[6] = _mm256_srai_epi32(_mm256_add_epi32(_mm256_sub_epi32(E1, O1), rnd), 12);
        c[7] = _mm256_srai_epi32(_mm256_add_epi32(_mm256_sub_epi32(E0, O0), rnd), 12);
    }
}

static inline void zcvid_lh_dct8_inv(const int32_t in[64], int32_t out[64]) {
    __m256i v[8];
    const __m256i lo = _mm256_set1_epi32(-65535);
    const __m256i hi = _mm256_set1_epi32(65535);
    __m256i bad = _mm256_setzero_si256();
    for (int i = 0; i < 8; ++i) {
        v[i] = _mm256_loadu_si256((const __m256i *)(in + i * 8));
        bad = _mm256_or_si256(bad, _mm256_or_si256(_mm256_cmpgt_epi32(v[i], hi),
                                                   _mm256_cmpgt_epi32(lo, v[i])));
    }
    if (!_mm256_testz_si256(bad, bad)) {
        zcvid_lh_dct8_inv_ref(in, out);
        return;
    }
    zcvid_lh_transpose8x8_avx2(v);
    zcvid_lh_dct8_inv_1d_avx2(v, 64, 7);
    zcvid_lh_transpose8x8_avx2(v);
    zcvid_lh_dct8_inv_1d_avx2(v, 2048, 12);
    for (int i = 0; i < 8; ++i) _mm256_storeu_si256((__m256i *)(out + i * 8), v[i]);
}
#else
static inline void zcvid_lh_dct8_fwd(const int32_t in[64], int32_t out[64]) {
    zcvid_lh_dct8_fwd_scalar(in, out);
}

static inline void zcvid_lh_dct8_inv(const int32_t in[64], int32_t out[64]) {
    zcvid_lh_dct8_inv_scalar(in, out);
}
#endif

/* Bit-exact sparse/CSBF-aware 8x8 IDCT8 for POLY_ESR (Lean 4 proved in Theorem 6a/6b/6c/6d):
 *  - poly_only (3x3 degree-2 polynomial): skips rows/cols 3..7 completely (~17 AVX2 ops!)
 *  - csbf == 1 (LL 4x4 sub-block only): skips rows/cols 4..7 completely
 *  - general csbf: full factored 12-mul AVX2 / scalar IDCT8 (with 64-bit overflow guard) */
static inline void zcvid_lh_dct8_inv_sparse(const int32_t in[64], uint32_t csbf, bool poly_only,
                                            int32_t out[64]) {
    if (poly_only) {
        /* INT32_MIN-safe overflow check for extreme adversarial QP=51 decoded coefficients */
        if (in[0] > 65535 || in[0] < -65535 || in[1] > 65535 || in[1] < -65535 ||
            in[2] > 65535 || in[2] < -65535 || in[8] > 65535 || in[8] < -65535 ||
            in[9] > 65535 || in[9] < -65535 || in[16] > 65535 || in[16] < -65535) {
            int32_t full_w[64];
            memset(full_w, 0, sizeof(full_w));
            full_w[0] = in[0]; full_w[1] = in[1]; full_w[2] = in[2];
            full_w[8] = in[8]; full_w[9] = in[9]; full_w[16] = in[16];
            zcvid_lh_dct8_inv_ref(full_w, out);
            return;
        }
        /* Exact DC-only 2-op collapse (Theorem 6c) */
        if ((in[1] | in[2] | in[8] | in[9] | in[16]) == 0) {
            int32_t t0 = (64 * in[0] + 64) >> 7;
            int32_t v = (64 * t0 + 2048) >> 12;
            for (uint32_t i = 0U; i < 64U; ++i) out[i] = v;
            return;
        }
#if defined(__AVX2__)
        const __m256i k1 = _mm256_loadu_si256((const __m256i *)k_hevc_dct8[1]);
        const __m256i k2 = _mm256_loadu_si256((const __m256i *)k_hevc_dct8[2]);
        const __m256i c64 = _mm256_set1_epi32(64);
        const __m256i c2048 = _mm256_set1_epi32(2048);

        /* Row 0: (0,0), (0,1), (0,2); Row 1: (1,0), (1,1); Row 2: (2,0) */
        __m256i t0 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(
            _mm256_set1_epi32(64 * in[0]),
            _mm256_mullo_epi32(_mm256_set1_epi32(in[1]), k1)),
            _mm256_add_epi32(_mm256_mullo_epi32(_mm256_set1_epi32(in[2]), k2), c64)), 7);
        __m256i t1 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(
            _mm256_set1_epi32(64 * in[8]),
            _mm256_mullo_epi32(_mm256_set1_epi32(in[9]), k1)), c64), 7);
        __m256i t2 = _mm256_set1_epi32((64 * in[16] + 64) >> 7);

        __m256i b0 = _mm256_slli_epi32(t0, 6);
        static const int32_t k1y[4] = { 89, 75, 50, 18 };
        static const int32_t k2y[4] = { 83, 36, -36, -83 };
        for (uint32_t y = 0U; y < 4U; ++y) {
            __m256i ev = _mm256_add_epi32(b0, _mm256_mullo_epi32(t2, _mm256_set1_epi32(k2y[y])));
            __m256i od = _mm256_mullo_epi32(t1, _mm256_set1_epi32(k1y[y]));
            _mm256_storeu_si256((__m256i *)(out + y * 8U),
                                _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(ev, od), c2048), 12));
            _mm256_storeu_si256((__m256i *)(out + (7U - y) * 8U),
                                _mm256_srai_epi32(_mm256_add_epi32(_mm256_sub_epi32(ev, od), c2048), 12));
        }
#else
        int32_t t0[8], t1[8];
        int32_t t2 = (64 * in[16] + 64) >> 7;
        for (uint32_t x = 0U; x < 8U; ++x) {
            t0[x] = (64 * in[0] + k_hevc_dct8[1][x] * in[1] + k_hevc_dct8[2][x] * in[2] + 64) >> 7;
            t1[x] = (64 * in[8] + k_hevc_dct8[1][x] * in[9] + 64) >> 7;
        }
        static const int32_t k1y[4] = { 89, 75, 50, 18 };
        static const int32_t k2y[4] = { 83, 36, -36, -83 };
        for (uint32_t y = 0U; y < 4U; ++y) {
            int32_t c2t = k2y[y] * t2;
            for (uint32_t x = 0U; x < 8U; ++x) {
                int32_t ev = 64 * t0[x] + c2t;
                int32_t od = k1y[y] * t1[x];
                out[y * 8U + x] = (ev + od + 2048) >> 12;
                out[(7U - y) * 8U + x] = (ev - od + 2048) >> 12;
            }
        }
#endif
        return;
    }

    if (csbf == 1U) {
        /* LL-only 4x4 sub-block: only rows 0..3 and cols 0..3 are non-zero (Theorem 6a/6d) */
        for (uint32_t v = 0U; v < 4U; ++v) {
            for (uint32_t u = 0U; u < 4U; ++u) {
                if (in[v * 8U + u] > 65535 || in[v * 8U + u] < -65535) {
                    int32_t full_w[64];
                    memset(full_w, 0, sizeof(full_w));
                    for (uint32_t y = 0U; y < 4U; ++y) {
                        for (uint32_t x = 0U; x < 4U; ++x) full_w[y * 8U + x] = in[y * 8U + x];
                    }
                    zcvid_lh_dct8_inv_ref(full_w, out);
                    return;
                }
            }
        }
#if defined(__AVX2__)
        const __m256i k1 = _mm256_loadu_si256((const __m256i *)k_hevc_dct8[1]);
        const __m256i k2 = _mm256_loadu_si256((const __m256i *)k_hevc_dct8[2]);
        const __m256i k3 = _mm256_loadu_si256((const __m256i *)k_hevc_dct8[3]);
        const __m256i c64 = _mm256_set1_epi32(64);
        const __m256i c2048 = _mm256_set1_epi32(2048);
        __m256i t[4];
        for (uint32_t v = 0U; v < 4U; ++v) {
            const int32_t *row = in + v * 8U;
            __m256i s = _mm256_add_epi32(
                _mm256_add_epi32(_mm256_set1_epi32(64 * row[0]),
                                 _mm256_mullo_epi32(_mm256_set1_epi32(row[1]), k1)),
                _mm256_add_epi32(_mm256_mullo_epi32(_mm256_set1_epi32(row[2]), k2),
                                 _mm256_mullo_epi32(_mm256_set1_epi32(row[3]), k3)));
            t[v] = _mm256_srai_epi32(_mm256_add_epi32(s, c64), 7);
        }
        __m256i b0 = _mm256_slli_epi32(t[0], 6);
        static const int32_t k1y[4] = { 89, 75, 50, 18 };
        static const int32_t k2y[4] = { 83, 36, -36, -83 };
        static const int32_t k3y[4] = { 75, -18, -89, -50 };
        for (uint32_t y = 0U; y < 4U; ++y) {
            __m256i ev = _mm256_add_epi32(b0, _mm256_mullo_epi32(t[2], _mm256_set1_epi32(k2y[y])));
            __m256i od = _mm256_add_epi32(
                _mm256_mullo_epi32(t[1], _mm256_set1_epi32(k1y[y])),
                _mm256_mullo_epi32(t[3], _mm256_set1_epi32(k3y[y])));
            _mm256_storeu_si256((__m256i *)(out + y * 8U),
                                _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(ev, od), c2048), 12));
            _mm256_storeu_si256((__m256i *)(out + (7U - y) * 8U),
                                _mm256_srai_epi32(_mm256_add_epi32(_mm256_sub_epi32(ev, od), c2048), 12));
        }
#else
        int32_t t[4][8];
        for (uint32_t v = 0U; v < 4U; ++v) {
            const int32_t *row = in + v * 8U;
            for (uint32_t x = 0U; x < 8U; ++x) {
                t[v][x] = (64 * row[0] + k_hevc_dct8[1][x] * row[1] +
                           k_hevc_dct8[2][x] * row[2] + k_hevc_dct8[3][x] * row[3] + 64) >> 7;
            }
        }
        static const int32_t k1y[4] = { 89, 75, 50, 18 };
        static const int32_t k2y[4] = { 83, 36, -36, -83 };
        static const int32_t k3y[4] = { 75, -18, -89, -50 };
        for (uint32_t y = 0U; y < 4U; ++y) {
            for (uint32_t x = 0U; x < 8U; ++x) {
                int32_t ev = 64 * t[0][x] + k2y[y] * t[2][x];
                int32_t od = k1y[y] * t[1][x] + k3y[y] * t[3][x];
                out[y * 8U + x] = (ev + od + 2048) >> 12;
                out[(7U - y) * 8U + x] = (ev - od + 2048) >> 12;
            }
        }
#endif
        return;
    }

    zcvid_lh_dct8_inv(in, out);
}

/* =========================================================================
 * Ultra-Low-Bit Ternary Latent Factorization on 8x8: R ~= alpha * u * v^T
 * ========================================================================= */

typedef struct {
    int8_t  u[8];   /* {-1, 0, +1}, first non-zero element strictly +1 */
    int8_t  v[8];   /* {-1, 0, +1}, first non-zero element strictly +1 */
    int32_t level;  /* quantized non-zero signed level (in [-4095, 4095]) */
    int32_t step;   /* quantization step >= 1, derived from QP and ||u||*||v|| */
} zcvid_lh_lrf_t;

static inline uint32_t zcvid_lh_qstep_pix(uint32_t qp) {
    /* Matches H.264/zcvid pixel step ~= 0.625 * 2^(qp/6). */
    double qs = 0.625 * pow(2.0, (double)qp / 6.0);
    int32_t qi = (int32_t)llround(qs);
    return (uint32_t)((qi < 1) ? 1 : qi);
}

static inline int32_t zcvid_lh_lrf_step(uint32_t qp, uint32_t nu, uint32_t nv, bool is_spec) {
    /* RDO scale quant step: shared across nu*nv samples so divide by sqrt(nu*nv),
     * modulated down slightly relative to transform DC sensitivity. */
    double qs = (double)zcvid_lh_qstep_pix(qp);
    if (is_spec) qs *= 16.0; /* 2D DCT8 norm is 16x pixel domain */
    double denom = sqrt((double)(nu * nv));
    if (denom < 1.0) denom = 1.0;
    int32_t st = (int32_t)llround((qs * 1.35) / denom);
    if (st < 1) st = 1;
    return st;
}

typedef struct {
    int32_t  step[2][65]; /* [is_spec][nu * nv] */
    int32_t  tq_mf[16];   /* forward quantizer multipliers in zigzag order */
    int32_t  tq_dq[16];   /* inverse dequant scale in zigzag order */
    uint32_t tq_qbits;
    int64_t  tq_dz[2];    /* [is_inter] deadzone */
    /* Uniform HEVC 8x8 DCT quantizer/dequantizer for POLY_ESR */
    int32_t  dct8_mf;
    uint32_t dct8_qbits;
    int64_t  dct8_dz[2];  /* [is_inter] deadzone */
    int32_t  dct8_dq_step;
} zcvid_lh_step_lut_t;

static inline void zcvid_lh_step_lut_init(zcvid_lh_step_lut_t *lut, uint32_t qp) {
    static const int32_t TQ_MF[18] = {
        13107, 5243, 8066,  11916, 4660, 7490,  10082, 4194, 6554,
         9362, 3647, 5825,   8192, 3355, 5243,   7282, 2893, 4559
    };
    static const int32_t TQ_V[18] = {
        10, 16, 13,  11, 18, 14,  13, 20, 16,
        14, 23, 18,  16, 25, 20,  18, 29, 23
    };
    static const uint32_t TQ_ZZ[16] = {
        0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15
    };
    static const int32_t HEVC_MF[6] = { 26214, 23302, 20560, 18396, 16384, 14564 };
    static const int32_t HEVC_V[6]  = { 40, 45, 51, 57, 64, 72 };
    lut->step[0][0] = 1;
    lut->step[1][0] = 1;
    for (uint32_t area = 1U; area <= 64U; ++area) {
        lut->step[0][area] = zcvid_lh_lrf_step(qp, area, 1U, false);
        lut->step[1][area] = zcvid_lh_lrf_step(qp, area, 1U, true);
    }
    const uint32_t qp_mod = qp % 6U;
    const uint32_t qp_div = qp / 6U;
    lut->tq_qbits = 15U + qp_div;
    lut->tq_dz[0] = (int64_t)((1ULL << lut->tq_qbits) / 3ULL);
    lut->tq_dz[1] = (int64_t)((1ULL << lut->tq_qbits) / 6ULL);
    for (uint32_t k = 0U; k < 16U; ++k) {
        uint32_t cls = zcvid_dec_tq_class(TQ_ZZ[k]);
        lut->tq_mf[k] = TQ_MF[qp_mod * 3U + cls];
        lut->tq_dq[k] = TQ_V[qp_mod * 3U + cls] << qp_div;
    }
    lut->dct8_qbits = 18U + qp_div;
    lut->dct8_dz[0] = (int64_t)((1ULL << lut->dct8_qbits) / 3ULL);
    lut->dct8_dz[1] = (int64_t)((1ULL << lut->dct8_qbits) / 6ULL);
    if (qp_div >= 2U) {
        lut->dct8_dq_step = HEVC_V[qp_mod] << (qp_div - 2U);
        lut->dct8_mf = HEVC_MF[qp_mod];
    } else {
        int32_t dq = ((HEVC_V[qp_mod] << qp_div) + 2) >> 2;
        if (dq < 1) dq = 1;
        lut->dct8_dq_step = dq;
        lut->dct8_mf = (int32_t)(((1ULL << lut->dct8_qbits) + (uint64_t)(dq >> 1)) / (uint64_t)dq);
    }
}

static inline uint32_t zcvid_lh_tq4_quant_fast(const int32_t w[16], const zcvid_lh_step_lut_t *lut,
                                               bool is_inter, int16_t lv[16]) {
    static const uint32_t TQ_ZZ[16] = {
        0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15
    };
    const uint32_t qbits = lut->tq_qbits;
    const int32_t dz = (int32_t)lut->tq_dz[is_inter ? 1 : 0];
    uint32_t sig_mask = 0U;
    for (uint32_t k = 0U; k < 16U; ++k) {
        int32_t c = w[TQ_ZZ[k]];
        uint32_t ac = (c < 0) ? (0U - (uint32_t)c) : (uint32_t)c;
        int32_t mag = (int32_t)((ac * (uint32_t)lut->tq_mf[k] + (uint32_t)dz) >> qbits);
        if (mag > 511) mag = 511;
        int16_t v = (int16_t)((c < 0) ? -mag : mag);
        lv[k] = v;
        if (v != 0) sig_mask |= (1U << k);
    }
    return sig_mask;
}

static inline uint32_t zcvid_lh_clz32(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return (uint32_t)__builtin_clz(x);
#elif defined(_MSC_VER)
    unsigned long idx = 0;
    _BitScanReverse(&idx, x);
    return 31U - (uint32_t)idx;
#else
    uint32_t n = 0U;
    while ((x & 0x80000000U) == 0U) { x <<= 1; n++; }
    return n;
#endif
}

static inline uint32_t zcvid_lh_ctz32(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return (uint32_t)__builtin_ctz(x);
#elif defined(_MSC_VER)
    unsigned long idx = 0;
    _BitScanForward(&idx, x);
    return (uint32_t)idx;
#else
    uint32_t n = 0U;
    while ((x & 1U) == 0U) { x >>= 1; n++; }
    return n;
#endif
}

static inline uint32_t zcvid_lh_tq4_rate_q8_mask(const zcvid_lh_model_t *m, const int16_t lv[16],
                                                 uint32_t sig_mask) {
    if (sig_mask == 0U) return 0U;
    uint32_t last_k = 31U - zcvid_lh_clz32(sig_mask);
    uint32_t bits_q8 = 0U;
    uint32_t has_ac = (last_k > 0U) ? 1U : 0U;
    bits_q8 += zcvid_lh_cost_q8(&m->tq_has_ac, has_ac);
    if (has_ac) {
        uint32_t v = last_k - 1U;
        uint32_t b3 = (v >> 3) & 1U; bits_q8 += zcvid_lh_cost_q8(&m->tq_last[0], b3);
        uint32_t n2 = 2U | b3;
        uint32_t b2 = (v >> 2) & 1U; bits_q8 += zcvid_lh_cost_q8(&m->tq_last[n2 - 1U], b2);
        uint32_t n1 = (n2 << 1) | b2;
        uint32_t b1 = (v >> 1) & 1U; bits_q8 += zcvid_lh_cost_q8(&m->tq_last[n1 - 1U], b1);
        uint32_t n0 = (n1 << 1) | b1;
        if (n0 != 15U) bits_q8 += zcvid_lh_cost_q8(&m->tq_last[n0 - 1U], v & 1U);
        for (uint32_t k = 0U; k < last_k; ++k) {
            bits_q8 += zcvid_lh_cost_q8(&m->tq_sig[k], (sig_mask >> k) & 1U);
        }
    }
    uint32_t sm = sig_mask;
    while (sm != 0U) {
        uint32_t k = zcvid_lh_ctz32(sm);
        sm &= (sm - 1U);
        int32_t val = lv[k];
        bits_q8 += 256U;
        uint32_t mag = (uint32_t)((val < 0) ? -val : val);
        uint32_t kb = zcvid_lh_tq_kb(k);
        uint32_t gt1 = (mag > 1U) ? 1U : 0U;
        bits_q8 += zcvid_lh_cost_q8(&m->tq_gt1[kb], gt1);
        if (gt1) {
            uint32_t gt2 = (mag > 2U) ? 1U : 0U;
            bits_q8 += zcvid_lh_cost_q8(&m->tq_gt2[kb], gt2);
            if (gt2) {
                bits_q8 += zcvid_lh_egk_cost_q8(m->tq_mag[kb], 7U, 9U, (uint64_t)(mag - 3U));
            }
        }
    }
    return bits_q8;
}

static inline void zcvid_lh_tq4_recon_fast(const int16_t lv[16], uint32_t sig_mask,
                                           const zcvid_lh_step_lut_t *lut, int32_t out_r[16]) {
    static const uint32_t TQ_ZZ[16] = {
        0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15
    };
    if (sig_mask == 1U) {
        int32_t dc = ((int32_t)lv[0] * lut->tq_dq[0] + 32) >> 6;
        for (uint32_t k = 0U; k < 16U; ++k) out_r[k] = dc;
        return;
    }
    int32_t c[16];
    memset(c, 0, sizeof(c));
    uint32_t m = sig_mask;
    while (m != 0U) {
        uint32_t k = zcvid_lh_ctz32(m);
        m &= (m - 1U);
        c[TQ_ZZ[k]] = (int32_t)lv[k] * lut->tq_dq[k];
    }
    zcvid_dec_tq_inverse(c);
    memcpy(out_r, c, sizeof(c));
}

/* =========================================================================
 * OpenAI-Inspired Polynomial Prediction + Exact Spectral Repair (POLY_ESR)
 *
 * Stage 1 (Polynomial Prediction): When energy lies in the degree-<=2 cosine
 *   polynomial subspace {(0,0),(0,1),(1,0),(2,0),(1,1),(0,2)} (exact prefix
 *   k=0..5 of LL 4x4 zigzag), codes poly_only=1 and reconstructs via exact
 *   3x3-cropped IDCT8 (~17 AVX2 ops, proved in ZcvidDCT8Circuit.lean Thm 6b).
 * Stage 2 (Exact Spectral Repair): When high-frequency spectral spikes exist
 *   (45-deg diagonals, textures, panning edges), codes sparse 4x4 CSBF
 *   sub-blocks (LL, LH, HL, HH) with Parseval-pruned RDO and sparse IDCT8.
 * ========================================================================= */

typedef struct {
    bool     poly_only;
    uint32_t csbf;
    uint32_t sb_mask[4];
    int16_t  sb_lv[4][16];
    uint32_t poly_mask;
    int16_t  poly_lv[6];
} zcvid_lh_esr_t;

static inline uint32_t zcvid_lh_esr_sb_rate_q8(const zcvid_lh_model_t *m, uint32_t sb,
                                               const int16_t lv[16], uint32_t sig_mask) {
    if (sig_mask == 0U) return 0U;
    uint32_t last_k = 31U - zcvid_lh_clz32(sig_mask);
    uint32_t bits_q8 = 0U;
    uint32_t has_ac = (last_k > 0U) ? 1U : 0U;
    bits_q8 += zcvid_lh_cost_q8(&m->esr_has_ac[sb], has_ac);
    if (has_ac) {
        uint32_t v = last_k - 1U;
        uint32_t b3 = (v >> 3) & 1U; bits_q8 += zcvid_lh_cost_q8(&m->esr_last[sb][0], b3);
        uint32_t n2 = 2U | b3;
        uint32_t b2 = (v >> 2) & 1U; bits_q8 += zcvid_lh_cost_q8(&m->esr_last[sb][n2 - 1U], b2);
        uint32_t n1 = (n2 << 1) | b2;
        uint32_t b1 = (v >> 1) & 1U; bits_q8 += zcvid_lh_cost_q8(&m->esr_last[sb][n1 - 1U], b1);
        uint32_t n0 = (n1 << 1) | b1;
        if (n0 != 15U) bits_q8 += zcvid_lh_cost_q8(&m->esr_last[sb][n0 - 1U], v & 1U);
        for (uint32_t k = 0U; k < last_k; ++k) {
            bits_q8 += zcvid_lh_cost_q8(&m->esr_sig[sb][k], (sig_mask >> k) & 1U);
        }
    }
    uint32_t sm = sig_mask;
    while (sm != 0U) {
        uint32_t k = zcvid_lh_ctz32(sm);
        sm &= (sm - 1U);
        int32_t val = lv[k];
        bits_q8 += 256U;
        uint32_t mag = (uint32_t)((val < 0) ? -val : val);
        uint32_t kb = (sb == 0U) ? zcvid_lh_tq_kb(k) : 3U;
        uint32_t gt1 = (mag > 1U) ? 1U : 0U;
        bits_q8 += zcvid_lh_cost_q8(&m->esr_gt1[kb], gt1);
        if (gt1) {
            uint32_t gt2 = (mag > 2U) ? 1U : 0U;
            bits_q8 += zcvid_lh_cost_q8(&m->esr_gt2[kb], gt2);
            if (gt2) {
                bits_q8 += zcvid_lh_egk_cost_q8(m->esr_mag[kb], 7U, 12U, (uint64_t)(mag - 3U));
            }
        }
    }
    return bits_q8;
}

static inline uint32_t zcvid_lh_esr_rate_q8(const zcvid_lh_model_t *m, const zcvid_lh_esr_t *e) {
    if (e->poly_only) {
        if (e->poly_mask == 0U) return 0U;
        uint32_t bits_q8 = zcvid_lh_cost_q8(&m->esr_poly_only, 1U);
        uint32_t last_k = 31U - zcvid_lh_clz32(e->poly_mask);
        uint32_t has_ac = (last_k > 0U) ? 1U : 0U;
        bits_q8 += zcvid_lh_cost_q8(&m->esr_poly_has_ac, has_ac);
        if (has_ac) {
            uint32_t v = last_k - 1U;
            uint32_t ge2 = (v >= 2U) ? 1U : 0U;
            bits_q8 += zcvid_lh_cost_q8(&m->esr_poly_last[0], ge2);
            if (!ge2) {
                bits_q8 += zcvid_lh_cost_q8(&m->esr_poly_last[1], v & 1U);
            } else {
                uint32_t is4 = (v == 4U) ? 1U : 0U;
                bits_q8 += zcvid_lh_cost_q8(&m->esr_poly_last[2], is4);
                if (!is4) {
                    bits_q8 += zcvid_lh_cost_q8(&m->esr_poly_last[3], v - 2U);
                }
            }
            for (uint32_t k = 0U; k < last_k; ++k) {
                bits_q8 += zcvid_lh_cost_q8(&m->esr_poly_sig[k], (e->poly_mask >> k) & 1U);
            }
        }
        uint32_t sm = e->poly_mask;
        while (sm != 0U) {
            uint32_t k = zcvid_lh_ctz32(sm);
            sm &= (sm - 1U);
            int32_t val = e->poly_lv[k];
            bits_q8 += 256U;
            uint32_t mag = (uint32_t)((val < 0) ? -val : val);
            uint32_t kb = (k == 0U) ? 0U : ((k <= 2U) ? 1U : 2U);
            uint32_t gt1 = (mag > 1U) ? 1U : 0U;
            bits_q8 += zcvid_lh_cost_q8(&m->esr_gt1[kb], gt1);
            if (gt1) {
                uint32_t gt2 = (mag > 2U) ? 1U : 0U;
                bits_q8 += zcvid_lh_cost_q8(&m->esr_gt2[kb], gt2);
                if (gt2) {
                    bits_q8 += zcvid_lh_egk_cost_q8(m->esr_mag[kb], 7U, 12U, (uint64_t)(mag - 3U));
                }
            }
        }
        return bits_q8;
    }

    uint32_t bits_q8 = zcvid_lh_cost_q8(&m->esr_poly_only, 0U);
    for (uint32_t sb = 0U; sb < 4U; ++sb) {
        uint32_t act = (e->csbf >> sb) & 1U;
        bits_q8 += zcvid_lh_cost_q8(&m->esr_csbf[sb], act);
        if (act) {
            bits_q8 += zcvid_lh_esr_sb_rate_q8(m, sb, e->sb_lv[sb], e->sb_mask[sb]);
        }
    }
    return bits_q8;
}

static bool zcvid_lh_esr_quant_and_prune(const int32_t wmat[64], const zcvid_lh_step_lut_t *lut,
                                         const zcvid_lh_model_t *m, uint64_t lambda_q8,
                                         bool is_inter, zcvid_lh_esr_t *e) {
    static const uint32_t TQ_ZZ[16] = {
        0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15
    };
    const uint32_t mf = (uint32_t)lut->dct8_mf;
    const uint32_t qbits = lut->dct8_qbits;
    const uint64_t dz = (uint64_t)lut->dct8_dz[is_inter ? 1 : 0];
    const int32_t dq = lut->dct8_dq_step;

    e->poly_only = false;
    e->csbf = 0U;
    e->poly_mask = 0U;

    for (uint32_t sb = 0U; sb < 4U; ++sb) {
        uint32_t v0 = (sb >> 1) * 4U;
        uint32_t u0 = (sb & 1U) * 4U;
        uint32_t sm = 0U;
        for (uint32_t k = 0U; k < 16U; ++k) {
            uint32_t r = TQ_ZZ[k];
            uint32_t pos = (v0 + (r >> 2)) * 8U + (u0 + (r & 3U));
            int32_t c = wmat[pos];
            uint32_t ac = (c < 0) ? (0U - (uint32_t)c) : (uint32_t)c;
            int32_t mag = (int32_t)(((uint64_t)ac * (uint64_t)mf + dz) >> qbits);
            if (mag > 4095) mag = 4095;
            int16_t lv = (int16_t)((c < 0) ? -mag : mag);
            e->sb_lv[sb][k] = lv;
            if (lv != 0) sm |= (1U << k);
        }
        /* Trailing-1 Parseval RDO pruning */
        while (sm != 0U) {
            uint32_t lk = 31U - zcvid_lh_clz32(sm);
            int32_t lv = e->sb_lv[sb][lk];
            if (lv != 1 && lv != -1) break;
            uint32_t r = TQ_ZZ[lk];
            uint32_t pos = (v0 + (r >> 2)) * 8U + (u0 + (r & 3U));
            int64_t wv = (int64_t)wmat[pos];
            int64_t diff = wv - (int64_t)lv * (int64_t)dq;
            int64_t d_pix = (wv * wv - diff * diff) >> 8;
            if (d_pix <= 0 || ((uint64_t)d_pix << 16) < lambda_q8 * 768ULL) {
                e->sb_lv[sb][lk] = 0;
                sm &= ~(1U << lk);
            } else {
                break;
            }
        }
        e->sb_mask[sb] = sm;
        if (sm != 0U) e->csbf |= (1U << sb);
    }

    if (e->csbf == 0U) return false;

    /* Per-subblock Parseval RDO pruning for high-frequency repair blocks (LH, HL, HH) */
    for (uint32_t sb = 1U; sb < 4U; ++sb) {
        if (((e->csbf >> sb) & 1U) == 0U) continue;
        uint32_t v0 = (sb >> 1) * 4U;
        uint32_t u0 = (sb & 1U) * 4U;
        int64_t d_acc = 0;
        uint32_t sm = e->sb_mask[sb];
        while (sm != 0U) {
            uint32_t k = zcvid_lh_ctz32(sm);
            sm &= (sm - 1U);
            uint32_t r = TQ_ZZ[k];
            uint32_t pos = (v0 + (r >> 2)) * 8U + (u0 + (r & 3U));
            int64_t wv = (int64_t)wmat[pos];
            int64_t diff = wv - (int64_t)e->sb_lv[sb][k] * (int64_t)dq;
            d_acc += wv * wv - diff * diff;
        }
        int64_t d_pix = d_acc >> 8;
        uint32_t r_sb = zcvid_lh_esr_sb_rate_q8(m, sb, e->sb_lv[sb], e->sb_mask[sb])
                      + zcvid_lh_cost_q8(&m->esr_csbf[sb], 1U);
        uint32_t r_off = zcvid_lh_cost_q8(&m->esr_csbf[sb], 0U);
        uint32_t net_r = (r_sb > r_off) ? (r_sb - r_off) : 0U;
        if (d_pix <= 0 || ((uint64_t)d_pix << 16) <= lambda_q8 * (uint64_t)net_r) {
            e->csbf &= ~(1U << sb);
            e->sb_mask[sb] = 0U;
        }
    }

    if (e->csbf == 0U) return false;

    /* Auto-promote to degree-2 Polynomial Prediction mode if all active coefs lie in k=0..5 of LL */
    if (e->csbf == 1U && (e->sb_mask[0] & ~0x3FU) == 0U) {
        e->poly_only = true;
        e->poly_mask = e->sb_mask[0];
        for (uint32_t k = 0U; k < 6U; ++k) e->poly_lv[k] = e->sb_lv[0][k];
    }
    return true;
}

static inline void zcvid_lh_esr_recon(const zcvid_lh_esr_t *e, const zcvid_lh_step_lut_t *lut,
                                      int32_t out_delta[64]) {
    static const uint32_t TQ_ZZ[16] = {
        0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15
    };
    static const uint32_t POLY_POS[6] = { 0U, 1U, 8U, 16U, 9U, 2U };
    const int32_t dq = lut->dct8_dq_step;
    int32_t w_deq[64];

    if (e->poly_only) {
        w_deq[0] = 0; w_deq[1] = 0; w_deq[2] = 0;
        w_deq[8] = 0; w_deq[9] = 0; w_deq[16] = 0;
        uint32_t sm = e->poly_mask;
        while (sm != 0U) {
            uint32_t k = zcvid_lh_ctz32(sm);
            sm &= (sm - 1U);
            w_deq[POLY_POS[k]] = (int32_t)e->poly_lv[k] * dq;
        }
        zcvid_lh_dct8_inv_sparse(w_deq, 1U, true, out_delta);
        return;
    }

    memset(w_deq, 0, sizeof(w_deq));
    for (uint32_t sb = 0U; sb < 4U; ++sb) {
        if (((e->csbf >> sb) & 1U) == 0U) continue;
        uint32_t v0 = (sb >> 1) * 4U;
        uint32_t u0 = (sb & 1U) * 4U;
        uint32_t sm = e->sb_mask[sb];
        while (sm != 0U) {
            uint32_t k = zcvid_lh_ctz32(sm);
            sm &= (sm - 1U);
            uint32_t r = TQ_ZZ[k];
            w_deq[(v0 + (r >> 2)) * 8U + (u0 + (r & 3U))] = (int32_t)e->sb_lv[sb][k] * dq;
        }
    }
    zcvid_lh_dct8_inv_sparse(w_deq, e->csbf, false, out_delta);
}

static void zcvid_lh_esr_encode(zcrudp_rc_enc_t *rc, zcvid_lh_model_t *m, const zcvid_lh_esr_t *e) {
    if (e->poly_only) {
        if (e->poly_mask == 0U) return;
        zcrudp_rc_enc_bit(rc, &m->esr_poly_only, 1U);
        uint32_t last_k = 31U - zcvid_lh_clz32(e->poly_mask);
        uint32_t has_ac = (last_k > 0U) ? 1U : 0U;
        zcrudp_rc_enc_bit(rc, &m->esr_poly_has_ac, has_ac);
        if (has_ac) {
            uint32_t v = last_k - 1U;
            uint32_t ge2 = (v >= 2U) ? 1U : 0U;
            zcrudp_rc_enc_bit(rc, &m->esr_poly_last[0], ge2);
            if (!ge2) {
                zcrudp_rc_enc_bit(rc, &m->esr_poly_last[1], v & 1U);
            } else {
                uint32_t is4 = (v == 4U) ? 1U : 0U;
                zcrudp_rc_enc_bit(rc, &m->esr_poly_last[2], is4);
                if (!is4) {
                    zcrudp_rc_enc_bit(rc, &m->esr_poly_last[3], v - 2U);
                }
            }
            for (uint32_t k = 0U; k < last_k; ++k) {
                zcrudp_rc_enc_bit(rc, &m->esr_poly_sig[k], (e->poly_mask >> k) & 1U);
            }
        }
        for (uint32_t k = 0U; k <= last_k; ++k) {
            if (((e->poly_mask >> k) & 1U) == 0U) continue;
            int32_t val = e->poly_lv[k];
            uint32_t neg = (val < 0) ? 1U : 0U;
            zcrudp_rc_enc_bit_bypass(rc, neg);
            uint32_t mag = (uint32_t)(neg ? -val : val);
            uint32_t kb = (k == 0U) ? 0U : ((k <= 2U) ? 1U : 2U);
            uint32_t gt1 = (mag > 1U) ? 1U : 0U;
            zcrudp_rc_enc_bit(rc, &m->esr_gt1[kb], gt1);
            if (gt1) {
                uint32_t gt2 = (mag > 2U) ? 1U : 0U;
                zcrudp_rc_enc_bit(rc, &m->esr_gt2[kb], gt2);
                if (gt2) {
                    zcrudp_rc_enc_egk(rc, m->esr_mag[kb], 7U, 12U, (uint64_t)(mag - 3U));
                }
            }
        }
        return;
    }

    zcrudp_rc_enc_bit(rc, &m->esr_poly_only, 0U);
    for (uint32_t sb = 0U; sb < 4U; ++sb) {
        uint32_t act = (e->csbf >> sb) & 1U;
        zcrudp_rc_enc_bit(rc, &m->esr_csbf[sb], act);
        if (!act) continue;
        uint32_t sm = e->sb_mask[sb];
        uint32_t last_k = 31U - zcvid_lh_clz32(sm);
        uint32_t has_ac = (last_k > 0U) ? 1U : 0U;
        zcrudp_rc_enc_bit(rc, &m->esr_has_ac[sb], has_ac);
        if (has_ac) {
            uint32_t v = last_k - 1U;
            zcrudp_rc_prob_t *lk = m->esr_last[sb];
            uint32_t b3 = (v >> 3) & 1U; zcrudp_rc_enc_bit(rc, &lk[0], b3);
            uint32_t n2 = 2U | b3;
            uint32_t b2 = (v >> 2) & 1U; zcrudp_rc_enc_bit(rc, &lk[n2 - 1U], b2);
            uint32_t n1 = (n2 << 1) | b2;
            uint32_t b1 = (v >> 1) & 1U; zcrudp_rc_enc_bit(rc, &lk[n1 - 1U], b1);
            uint32_t n0 = (n1 << 1) | b1;
            if (n0 != 15U) zcrudp_rc_enc_bit(rc, &lk[n0 - 1U], v & 1U);
            for (uint32_t k = 0U; k < last_k; ++k) {
                zcrudp_rc_enc_bit(rc, &m->esr_sig[sb][k], (sm >> k) & 1U);
            }
        }
        for (uint32_t k = 0U; k <= last_k; ++k) {
            if (((sm >> k) & 1U) == 0U) continue;
            int32_t val = e->sb_lv[sb][k];
            uint32_t neg = (val < 0) ? 1U : 0U;
            zcrudp_rc_enc_bit_bypass(rc, neg);
            uint32_t mag = (uint32_t)(neg ? -val : val);
            uint32_t kb = (sb == 0U) ? zcvid_lh_tq_kb(k) : 3U;
            uint32_t gt1 = (mag > 1U) ? 1U : 0U;
            zcrudp_rc_enc_bit(rc, &m->esr_gt1[kb], gt1);
            if (gt1) {
                uint32_t gt2 = (mag > 2U) ? 1U : 0U;
                zcrudp_rc_enc_bit(rc, &m->esr_gt2[kb], gt2);
                if (gt2) {
                    zcrudp_rc_enc_egk(rc, m->esr_mag[kb], 7U, 12U, (uint64_t)(mag - 3U));
                }
            }
        }
    }
}

static bool zcvid_lh_esr_decode(zcrudp_rc_dec_t *rc, zcvid_lh_model_t *m, zcvid_lh_esr_t *e) {
    e->poly_only = (zcrudp_rc_dec_bit(rc, &m->esr_poly_only) != 0U);
    if (e->poly_only) {
        e->csbf = 1U;
        uint32_t pm = 1U;
        uint32_t last_k = 0U;
        if (zcrudp_rc_dec_bit(rc, &m->esr_poly_has_ac) != 0U) {
            uint32_t v;
            if (zcrudp_rc_dec_bit(rc, &m->esr_poly_last[0]) == 0U) {
                v = zcrudp_rc_dec_bit(rc, &m->esr_poly_last[1]);
            } else if (zcrudp_rc_dec_bit(rc, &m->esr_poly_last[2]) != 0U) {
                v = 4U;
            } else {
                v = 2U + zcrudp_rc_dec_bit(rc, &m->esr_poly_last[3]);
            }
            last_k = v + 1U;
            pm = 1U << last_k;
            for (uint32_t k = 0U; k < last_k; ++k) {
                pm |= zcrudp_rc_dec_bit(rc, &m->esr_poly_sig[k]) << k;
            }
        }
        e->poly_mask = pm;
        for (uint32_t k = 0U; k <= last_k; ++k) {
            if (((pm >> k) & 1U) == 0U) { e->poly_lv[k] = 0; continue; }
            uint32_t neg = zcrudp_rc_dec_bit_bypass(rc);
            uint32_t kb = (k == 0U) ? 0U : ((k <= 2U) ? 1U : 2U);
            int32_t mag = 1;
            if (zcrudp_rc_dec_bit(rc, &m->esr_gt1[kb]) != 0U) {
                mag = 2;
                if (zcrudp_rc_dec_bit(rc, &m->esr_gt2[kb]) != 0U) {
                    mag = (int32_t)zcrudp_rc_dec_egk(rc, m->esr_mag[kb], 7U, 12U) + 3;
                    if (mag > 4095) return false;
                }
            }
            e->poly_lv[k] = (int16_t)(neg ? -mag : mag);
        }
        return true;
    }

    e->csbf = 0U;
    e->poly_mask = 0U;
    for (uint32_t sb = 0U; sb < 4U; ++sb) {
        e->sb_mask[sb] = 0U;
        if (zcrudp_rc_dec_bit(rc, &m->esr_csbf[sb]) == 0U) continue;
        e->csbf |= (1U << sb);
        uint32_t sm = 1U;
        uint32_t last_k = 0U;
        if (zcrudp_rc_dec_bit(rc, &m->esr_has_ac[sb]) != 0U) {
            zcrudp_rc_prob_t *lk = m->esr_last[sb];
            uint32_t node = 2U | zcrudp_rc_dec_bit(rc, &lk[0]);
            node = (node << 1) | zcrudp_rc_dec_bit(rc, &lk[node - 1U]);
            node = (node << 1) | zcrudp_rc_dec_bit(rc, &lk[node - 1U]);
            node = (node == 15U) ? 30U : ((node << 1) | zcrudp_rc_dec_bit(rc, &lk[node - 1U]));
            last_k = node - 15U;
            sm = 1U << last_k;
            for (uint32_t k = 0U; k < last_k; ++k) {
                sm |= zcrudp_rc_dec_bit(rc, &m->esr_sig[sb][k]) << k;
            }
        }
        e->sb_mask[sb] = sm;
        for (uint32_t k = 0U; k <= last_k; ++k) {
            if (((sm >> k) & 1U) == 0U) { e->sb_lv[sb][k] = 0; continue; }
            uint32_t neg = zcrudp_rc_dec_bit_bypass(rc);
            uint32_t kb = (sb == 0U) ? zcvid_lh_tq_kb(k) : 3U;
            int32_t mag = 1;
            if (zcrudp_rc_dec_bit(rc, &m->esr_gt1[kb]) != 0U) {
                mag = 2;
                if (zcrudp_rc_dec_bit(rc, &m->esr_gt2[kb]) != 0U) {
                    mag = (int32_t)zcrudp_rc_dec_egk(rc, m->esr_mag[kb], 7U, 12U) + 3;
                    if (mag > 4095) return false;
                }
            }
            e->sb_lv[sb][k] = (int16_t)(neg ? -mag : mag);
        }
    }
    return (e->csbf != 0U);
}

/* Bit-exact separable rank-1 8x8 inverse DCT for SPEC_LRF:
 * Replaces 1024-MAC 2D IDCT8 with <= 16 vector adds + 8 SIMD row ops. */
static inline void zcvid_lh_spec_lrf_recon(const zcvid_lh_lrf_t *f, int32_t out[64]) {
    if (f->level == 0) {
        memset(out, 0, sizeof(int32_t) * 64U);
        return;
    }
    const int64_t alpha64 = (int64_t)f->level * (int64_t)f->step;
    if (alpha64 > 65535 || alpha64 < -65535) {
        /* Overflow-safe 64-bit path for extreme/malformed decoded factors (never hit on valid 8-bit frames). */
        int64_t b[8] = {0};
        for (uint32_t u = 0U; u < 8U; ++u) {
            int8_t vu = f->v[u];
            if (vu == 0) continue;
            for (uint32_t x = 0U; x < 8U; ++x) b[x] += (vu > 0) ? k_hevc_dct8[u][x] : -k_hevc_dct8[u][x];
        }
        int64_t t_pos[8], t_neg[8];
        for (uint32_t x = 0U; x < 8U; ++x) {
            int64_t ab = alpha64 * b[x];
            t_pos[x] = (ab + 64) >> 7;
            t_neg[x] = (-ab + 64) >> 7;
        }
        int64_t c_pos[8] = {0}, c_neg[8] = {0};
        for (uint32_t v = 0U; v < 8U; ++v) {
            int8_t uv = f->u[v];
            if (uv > 0) {
                for (uint32_t y = 0U; y < 8U; ++y) c_pos[y] += k_hevc_dct8[v][y];
            } else if (uv < 0) {
                for (uint32_t y = 0U; y < 8U; ++y) c_neg[y] += k_hevc_dct8[v][y];
            }
        }
        for (uint32_t y = 0U; y < 8U; ++y) {
            int64_t cp = c_pos[y], cn = c_neg[y];
            for (uint32_t x = 0U; x < 8U; ++x) {
                out[y * 8U + x] = (int32_t)((t_pos[x] * cp + t_neg[x] * cn + 2048) >> 12);
            }
        }
        return;
    }
    const int32_t alpha = (int32_t)alpha64;
#if defined(__AVX2__)
    __m256i b_vec = _mm256_setzero_si256();
    for (uint32_t u = 0U; u < 8U; ++u) {
        int8_t vu = f->v[u];
        if (vu == 0) continue;
        __m256i row = _mm256_loadu_si256((const __m256i *)k_hevc_dct8[u]);
        b_vec = (vu > 0) ? _mm256_add_epi32(b_vec, row) : _mm256_sub_epi32(b_vec, row);
    }
    __m256i ab = _mm256_mullo_epi32(b_vec, _mm256_set1_epi32(alpha));
    __m256i c64 = _mm256_set1_epi32(64);
    __m256i t_pos = _mm256_srai_epi32(_mm256_add_epi32(ab, c64), 7);
    __m256i t_neg = _mm256_srai_epi32(_mm256_sub_epi32(c64, ab), 7);

    __m256i cp_vec = _mm256_setzero_si256();
    __m256i cn_vec = _mm256_setzero_si256();
    for (uint32_t v = 0U; v < 8U; ++v) {
        int8_t uv = f->u[v];
        if (uv > 0) cp_vec = _mm256_add_epi32(cp_vec, _mm256_loadu_si256((const __m256i *)k_hevc_dct8[v]));
        else if (uv < 0) cn_vec = _mm256_add_epi32(cn_vec, _mm256_loadu_si256((const __m256i *)k_hevc_dct8[v]));
    }
    int32_t c_pos[8], c_neg[8];
    _mm256_storeu_si256((__m256i *)c_pos, cp_vec);
    _mm256_storeu_si256((__m256i *)c_neg, cn_vec);

    const __m256i c2048 = _mm256_set1_epi32(2048);
    for (uint32_t y = 0U; y < 8U; ++y) {
        __m256i s = _mm256_add_epi32(
            _mm256_mullo_epi32(t_pos, _mm256_set1_epi32(c_pos[y])),
            _mm256_mullo_epi32(t_neg, _mm256_set1_epi32(c_neg[y])));
        _mm256_storeu_si256((__m256i *)(out + y * 8U),
                            _mm256_srai_epi32(_mm256_add_epi32(s, c2048), 12));
    }
#else
    int32_t b[8] = {0};
    for (uint32_t u = 0U; u < 8U; ++u) {
        int8_t vu = f->v[u];
        if (vu == 0) continue;
        for (uint32_t x = 0U; x < 8U; ++x) b[x] += (vu > 0) ? k_hevc_dct8[u][x] : -k_hevc_dct8[u][x];
    }
    int32_t t_pos[8], t_neg[8];
    for (uint32_t x = 0U; x < 8U; ++x) {
        int32_t ab = alpha * b[x];
        t_pos[x] = (ab + 64) >> 7;
        t_neg[x] = (-ab + 64) >> 7;
    }
    int32_t c_pos[8] = {0}, c_neg[8] = {0};
    for (uint32_t v = 0U; v < 8U; ++v) {
        int8_t uv = f->u[v];
        if (uv > 0) {
            for (uint32_t y = 0U; y < 8U; ++y) c_pos[y] += k_hevc_dct8[v][y];
        } else if (uv < 0) {
            for (uint32_t y = 0U; y < 8U; ++y) c_neg[y] += k_hevc_dct8[v][y];
        }
    }
    for (uint32_t y = 0U; y < 8U; ++y) {
        int32_t cp = c_pos[y], cn = c_neg[y];
        for (uint32_t x = 0U; x < 8U; ++x) {
            out[y * 8U + x] = (t_pos[x] * cp + t_neg[x] * cn + 2048) >> 12;
        }
    }
#endif
}

/* Exact globally optimal 1D ternary projection on p[0..7] minimizing
 * sum_i (p_i - a * u_i)^2 via descending sort of |p_i|. */
static inline void zcvid_lh_opt_ternary_1d(const double p[8], int8_t out_u[8]) {
    uint32_t ord[8] = { 0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U };
    double ap[8];
    for (uint32_t i = 0U; i < 8U; ++i) ap[i] = fabs(p[i]);
    for (uint32_t i = 1U; i < 8U; ++i) {
        uint32_t key = ord[i];
        double ak = ap[key];
        uint32_t j = i;
        while (j > 0U && ap[ord[j - 1U]] < ak) {
            ord[j] = ord[j - 1U];
            j--;
        }
        ord[j] = key;
    }
    double best_score = -1.0;
    uint32_t best_k = 1U;
    double prefix = 0.0;
    for (uint32_t k = 1U; k <= 8U; ++k) {
        prefix += ap[ord[k - 1U]];
        double score = (prefix * prefix) / (double)k;
        if (score > best_score) {
            best_score = score;
            best_k = k;
        }
    }
    memset(out_u, 0, 8U);
    if (ap[ord[0]] <= 1e-9) {
        out_u[0] = 1;
        return;
    }
    for (uint32_t k = 0U; k < best_k; ++k) {
        uint32_t idx = ord[k];
        if (ap[idx] > 1e-9) {
            out_u[idx] = (p[idx] >= 0.0) ? (int8_t)1 : (int8_t)(-1);
        }
    }
}

static inline void zcvid_lh_canonicalize(int8_t w[8], int32_t *sign_flip) {
    for (uint32_t i = 0U; i < 8U; ++i) {
        if (w[i] != 0) {
            if (w[i] < 0) {
                for (uint32_t j = i; j < 8U; ++j) w[j] = (int8_t)(-w[j]);
                *sign_flip = -(*sign_flip);
            }
            return;
        }
    }
}

static void zcvid_lh_fit_lrf(const int32_t r[64], uint32_t qp, bool is_spec, zcvid_lh_lrf_t *out) {
    memset(out, 0, sizeof(*out));
    /* Pick initial row/column with largest L2 energy */
    uint32_t best_row = 0U;
    double best_e = -1.0;
    for (uint32_t y = 0U; y < 8U; ++y) {
        double e = 0.0;
        for (uint32_t x = 0U; x < 8U; ++x) {
            double rv = (double)r[y * 8U + x];
            e += rv * rv;
        }
        if (e > best_e) {
            best_e = e;
            best_row = y;
        }
    }
    if (best_e <= 1e-6) return;

    double init_v[8];
    for (uint32_t x = 0U; x < 8U; ++x) init_v[x] = (double)r[best_row * 8U + x];
    int8_t u[8], v[8];
    zcvid_lh_opt_ternary_1d(init_v, v);

    /* 3 alternating iterations of exact 1D optimal ternary projections */
    for (int it = 0; it < 3; ++it) {
        uint32_t nv = 0U;
        for (uint32_t x = 0U; x < 8U; ++x) nv += (uint32_t)(v[x] != 0);
        if (nv == 0U) break;
        double pu[8];
        for (uint32_t y = 0U; y < 8U; ++y) {
            double s = 0.0;
            for (uint32_t x = 0U; x < 8U; ++x) s += (double)r[y * 8U + x] * (double)v[x];
            pu[y] = s / (double)nv;
        }
        zcvid_lh_opt_ternary_1d(pu, u);

        uint32_t nu = 0U;
        for (uint32_t y = 0U; y < 8U; ++y) nu += (uint32_t)(u[y] != 0);
        if (nu == 0U) break;
        double pv[8];
        for (uint32_t x = 0U; x < 8U; ++x) {
            double s = 0.0;
            for (uint32_t y = 0U; y < 8U; ++y) s += (double)r[y * 8U + x] * (double)u[y];
            pv[x] = s / (double)nu;
        }
        zcvid_lh_opt_ternary_1d(pv, v);
    }

    uint32_t nu = 0U, nv = 0U;
    double dot = 0.0;
    for (uint32_t y = 0U; y < 8U; ++y) nu += (uint32_t)(u[y] != 0);
    for (uint32_t x = 0U; x < 8U; ++x) nv += (uint32_t)(v[x] != 0);
    if (nu == 0U || nv == 0U) return;
    for (uint32_t y = 0U; y < 8U; ++y) {
        if (u[y] == 0) continue;
        for (uint32_t x = 0U; x < 8U; ++x) {
            dot += (double)r[y * 8U + x] * (double)u[y] * (double)v[x];
        }
    }
    int32_t sflip = 1;
    zcvid_lh_canonicalize(u, &sflip);
    zcvid_lh_canonicalize(v, &sflip);
    double alpha = (dot * (double)sflip) / (double)(nu * nv);

    int32_t step = zcvid_lh_lrf_step(qp, nu, nv, is_spec);
    int32_t lvl = (int32_t)llround(alpha / (double)step);
    if (lvl > 4095) lvl = 4095;
    if (lvl < -4095) lvl = -4095;

    memcpy(out->u, u, 8U);
    memcpy(out->v, v, 8U);
    out->level = lvl;
    out->step = step;
}

/* Pure-integer exact 1D optimal ternary projection (zero floats, zero divisions) */
static inline void zcvid_lh_opt_ternary_1d_i32(const int32_t p[8], int8_t out_u[8]) {
    uint32_t ord[8] = { 0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U };
    int32_t ap[8];
    for (uint32_t i = 0U; i < 8U; ++i) ap[i] = (p[i] < 0) ? -p[i] : p[i];
    for (uint32_t i = 1U; i < 8U; ++i) {
        uint32_t key = ord[i];
        int32_t ak = ap[key];
        uint32_t j = i;
        while (j > 0U && ap[ord[j - 1U]] < ak) {
            ord[j] = ord[j - 1U];
            j--;
        }
        ord[j] = key;
    }
    memset(out_u, 0, 8U);
    if (ap[ord[0]] == 0) {
        out_u[0] = 1;
        return;
    }
    int64_t best_p2 = (int64_t)ap[ord[0]] * (int64_t)ap[ord[0]];
    uint32_t best_k = 1U;
    int64_t prefix = (int64_t)ap[ord[0]];
    for (uint32_t k = 2U; k <= 8U; ++k) {
        int32_t a = ap[ord[k - 1U]];
        if (a == 0) break;
        prefix += (int64_t)a;
        int64_t p2 = prefix * prefix;
        if (p2 * (int64_t)best_k > best_p2 * (int64_t)k) {
            best_p2 = p2;
            best_k = k;
        }
    }
    for (uint32_t k = 0U; k < best_k; ++k) {
        uint32_t idx = ord[k];
        out_u[idx] = (p[idx] >= 0) ? (int8_t)1 : (int8_t)(-1);
    }
}

#if defined(__AVX2__)
static inline int32_t zcvid_lh_hsum_epi32_avx2(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}
#endif

static void zcvid_lh_fit_lrf_fast(const int32_t r[64], const zcvid_lh_step_lut_t *lut,
                                  bool is_spec, zcvid_lh_lrf_t *out) {
    memset(out, 0, sizeof(*out));
    uint32_t best_row = 0U;
    int64_t best_e = 0;

#if defined(__AVX2__)
    __m256i rows[8];
    for (uint32_t y = 0U; y < 8U; ++y) {
        rows[y] = _mm256_loadu_si256((const __m256i *)(r + y * 8U));
        __m256i sq = _mm256_mullo_epi32(rows[y], rows[y]);
        int64_t e = (int64_t)(uint32_t)zcvid_lh_hsum_epi32_avx2(sq);
        if (e > best_e) {
            best_e = e;
            best_row = y;
        }
    }
#else
    for (uint32_t y = 0U; y < 8U; ++y) {
        int64_t e = 0;
        for (uint32_t x = 0U; x < 8U; ++x) {
            int64_t rv = (int64_t)r[y * 8U + x];
            e += rv * rv;
        }
        if (e > best_e) {
            best_e = e;
            best_row = y;
        }
    }
#endif
    if (best_e == 0) return;

    int8_t u[8], v[8];
    zcvid_lh_opt_ternary_1d_i32(r + best_row * 8U, v);

    int32_t pv[8] = {0};
    for (int it = 0; it < 2; ++it) {
        int8_t v_prev[8];
        memcpy(v_prev, v, 8U);
        int32_t pu[8];
#if defined(__AVX2__)
        __m256i vv = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)v));
        for (uint32_t y = 0U; y < 8U; ++y) {
            pu[y] = zcvid_lh_hsum_epi32_avx2(_mm256_mullo_epi32(rows[y], vv));
        }
        zcvid_lh_opt_ternary_1d_i32(pu, u);

        __m256i acc = _mm256_setzero_si256();
        for (uint32_t y = 0U; y < 8U; ++y) {
            if (u[y] > 0) acc = _mm256_add_epi32(acc, rows[y]);
            else if (u[y] < 0) acc = _mm256_sub_epi32(acc, rows[y]);
        }
        _mm256_storeu_si256((__m256i *)pv, acc);
#else
        for (uint32_t y = 0U; y < 8U; ++y) {
            int32_t s = 0;
            for (uint32_t x = 0U; x < 8U; ++x) s += r[y * 8U + x] * (int32_t)v[x];
            pu[y] = s;
        }
        zcvid_lh_opt_ternary_1d_i32(pu, u);

        memset(pv, 0, sizeof(pv));
        for (uint32_t y = 0U; y < 8U; ++y) {
            if (u[y] > 0) {
                for (uint32_t x = 0U; x < 8U; ++x) pv[x] += r[y * 8U + x];
            } else if (u[y] < 0) {
                for (uint32_t x = 0U; x < 8U; ++x) pv[x] -= r[y * 8U + x];
            }
        }
#endif
        zcvid_lh_opt_ternary_1d_i32(pv, v);
        if (memcmp(v, v_prev, 8U) == 0) break;
    }

    uint32_t nu = 0U, nv = 0U;
    int64_t dot = 0;
    for (uint32_t y = 0U; y < 8U; ++y) nu += (uint32_t)(u[y] != 0);
    for (uint32_t x = 0U; x < 8U; ++x) {
        if (v[x] != 0) {
            nv++;
            dot += (int64_t)pv[x] * (int64_t)v[x];
        }
    }
    if (nu == 0U || nv == 0U || dot == 0) return;

    int32_t sflip = 1;
    zcvid_lh_canonicalize(u, &sflip);
    zcvid_lh_canonicalize(v, &sflip);
    int64_t dot_s = dot * (int64_t)sflip;

    int32_t step = lut->step[is_spec ? 1 : 0][nu * nv];
    int64_t denom = (int64_t)(nu * nv) * (int64_t)step;
    int32_t lvl = (int32_t)((dot_s >= 0) ? ((dot_s + (denom >> 1)) / denom)
                                         : ((dot_s - (denom >> 1)) / denom));
    if (lvl > 4095) lvl = 4095;
    if (lvl < -4095) lvl = -4095;
    if (lvl == 0) return;

    memcpy(out->u, u, 8U);
    memcpy(out->v, v, 8U);
    out->level = lvl;
    out->step = step;
}

static inline void zcvid_lh_recon_lrf_add(const zcvid_lh_lrf_t *f, int32_t dst[64]) {
    if (f->level == 0) return;
    int32_t alpha = f->level * f->step;
    for (uint32_t y = 0U; y < 8U; ++y) {
        int32_t uy = (int32_t)f->u[y] * alpha;
        if (uy == 0) continue;
        for (uint32_t x = 0U; x < 8U; ++x) {
            dst[y * 8U + x] += uy * (int32_t)f->v[x];
        }
    }
}

static uint32_t zcvid_lh_lrf_rate_q8(const zcvid_lh_model_t *m, const zcvid_lh_lrf_t *f) {
    if (f->level == 0) return 0U;
    uint32_t bits_q8 = 0U;
    /* Canonical u vector (first nz = +1) */
    bool seen_first = false;
    int32_t prev_s = 1;
    for (uint32_t i = 0U; i < 8U; ++i) {
        uint32_t nz = (f->u[i] != 0) ? 1U : 0U;
        bits_q8 += zcvid_lh_cost_q8(&m->lrf_nz_u[i], nz);
        if (nz) {
            if (!seen_first) {
                seen_first = true;
                prev_s = 1;
            } else {
                uint32_t diff = (f->u[i] != prev_s) ? 1U : 0U;
                bits_q8 += zcvid_lh_cost_q8(&m->lrf_sign_eq, diff);
                prev_s = f->u[i];
            }
        }
    }
    /* Canonical v vector (first nz = +1) */
    seen_first = false;
    prev_s = 1;
    for (uint32_t i = 0U; i < 8U; ++i) {
        uint32_t nz = (f->v[i] != 0) ? 1U : 0U;
        bits_q8 += zcvid_lh_cost_q8(&m->lrf_nz_v[i], nz);
        if (nz) {
            if (!seen_first) {
                seen_first = true;
                prev_s = 1;
            } else {
                uint32_t diff = (f->v[i] != prev_s) ? 1U : 0U;
                bits_q8 += zcvid_lh_cost_q8(&m->lrf_sign_eq, diff);
                prev_s = f->v[i];
            }
        }
    }
    /* Signed level: sign + mag-1 via gt1/gt2/egk */
    bits_q8 += 256U;
    uint32_t mag = (uint32_t)((f->level < 0) ? -f->level : f->level);
    uint32_t gt1 = (mag > 1U) ? 1U : 0U;
    bits_q8 += zcvid_lh_cost_q8(&m->lrf_alpha_gt1, gt1);
    if (gt1) {
        uint32_t gt2 = (mag > 2U) ? 1U : 0U;
        bits_q8 += zcvid_lh_cost_q8(&m->lrf_alpha_gt2, gt2);
        if (gt2) {
            bits_q8 += zcvid_lh_egk_cost_q8(m->lrf_alpha_mag, 7U, 12U, (uint64_t)(mag - 3U));
        }
    }
    return bits_q8;
}

static void zcvid_lh_lrf_encode(zcrudp_rc_enc_t *rc, zcvid_lh_model_t *m, const zcvid_lh_lrf_t *f) {
    bool seen_first = false;
    int32_t prev_s = 1;
    for (uint32_t i = 0U; i < 8U; ++i) {
        uint32_t nz = (f->u[i] != 0) ? 1U : 0U;
        zcrudp_rc_enc_bit(rc, &m->lrf_nz_u[i], nz);
        if (nz) {
            if (!seen_first) {
                seen_first = true;
                prev_s = 1;
            } else {
                uint32_t diff = (f->u[i] != prev_s) ? 1U : 0U;
                zcrudp_rc_enc_bit(rc, &m->lrf_sign_eq, diff);
                prev_s = f->u[i];
            }
        }
    }
    seen_first = false;
    prev_s = 1;
    for (uint32_t i = 0U; i < 8U; ++i) {
        uint32_t nz = (f->v[i] != 0) ? 1U : 0U;
        zcrudp_rc_enc_bit(rc, &m->lrf_nz_v[i], nz);
        if (nz) {
            if (!seen_first) {
                seen_first = true;
                prev_s = 1;
            } else {
                uint32_t diff = (f->v[i] != prev_s) ? 1U : 0U;
                zcrudp_rc_enc_bit(rc, &m->lrf_sign_eq, diff);
                prev_s = f->v[i];
            }
        }
    }
    uint32_t neg = (f->level < 0) ? 1U : 0U;
    zcrudp_rc_enc_bit_bypass(rc, neg);
    uint32_t mag = (uint32_t)(neg ? -f->level : f->level);
    uint32_t gt1 = (mag > 1U) ? 1U : 0U;
    zcrudp_rc_enc_bit(rc, &m->lrf_alpha_gt1, gt1);
    if (gt1) {
        uint32_t gt2 = (mag > 2U) ? 1U : 0U;
        zcrudp_rc_enc_bit(rc, &m->lrf_alpha_gt2, gt2);
        if (gt2) {
            zcrudp_rc_enc_egk(rc, m->lrf_alpha_mag, 7U, 12U, (uint64_t)(mag - 3U));
        }
    }
}

static bool zcvid_lh_lrf_decode_lut(zcrudp_rc_dec_t *rc, zcvid_lh_model_t *m,
                                    const zcvid_lh_step_lut_t *lut, bool is_spec,
                                    zcvid_lh_lrf_t *f) {
    memset(f, 0, sizeof(*f));
    uint32_t nu = 0U, nv = 0U;
    bool seen_first = false;
    int32_t prev_s = 1;
    for (uint32_t i = 0U; i < 8U; ++i) {
        if (zcrudp_rc_dec_bit(rc, &m->lrf_nz_u[i]) != 0U) {
            nu++;
            if (!seen_first) {
                seen_first = true;
                prev_s = 1;
                f->u[i] = 1;
            } else {
                if (zcrudp_rc_dec_bit(rc, &m->lrf_sign_eq) != 0U) prev_s = -prev_s;
                f->u[i] = (int8_t)prev_s;
            }
        }
    }
    seen_first = false;
    prev_s = 1;
    for (uint32_t i = 0U; i < 8U; ++i) {
        if (zcrudp_rc_dec_bit(rc, &m->lrf_nz_v[i]) != 0U) {
            nv++;
            if (!seen_first) {
                seen_first = true;
                prev_s = 1;
                f->v[i] = 1;
            } else {
                if (zcrudp_rc_dec_bit(rc, &m->lrf_sign_eq) != 0U) prev_s = -prev_s;
                f->v[i] = (int8_t)prev_s;
            }
        }
    }
    if (nu == 0U || nv == 0U) return false;
    uint32_t neg = zcrudp_rc_dec_bit_bypass(rc);
    int32_t mag = 1;
    if (zcrudp_rc_dec_bit(rc, &m->lrf_alpha_gt1) != 0U) {
        mag = 2;
        if (zcrudp_rc_dec_bit(rc, &m->lrf_alpha_gt2) != 0U) {
            mag = (int32_t)zcrudp_rc_dec_egk(rc, m->lrf_alpha_mag, 7U, 12U) + 3;
            if (mag > 4095) return false;
        }
    }
    f->level = neg ? -mag : mag;
    f->step = lut->step[is_spec ? 1 : 0][nu * nv];
    return true;
}

static inline bool zcvid_lh_lrf_decode(zcrudp_rc_dec_t *rc, zcvid_lh_model_t *m,
                                       uint32_t qp, bool is_spec, zcvid_lh_lrf_t *f) {
    zcvid_lh_step_lut_t lut;
    zcvid_lh_step_lut_init(&lut, qp);
    return zcvid_lh_lrf_decode_lut(rc, m, &lut, is_spec, f);
}

/* =========================================================================
 * Full Frame Encode & Decode (bit-exact parity across all 6 modes)
 * ========================================================================= */

static inline uint32_t zcvid_lh_quad_sse_clamp(const uint8_t *src, const uint8_t *pred,
                                               uint32_t stride, uint32_t qx, uint32_t qy,
                                               const int32_t rec_delta[16],
                                               uint8_t *opt_rec) {
    uint32_t sse = 0U;
    for (uint32_t y = 0U; y < 4U; ++y) {
        for (uint32_t x = 0U; x < 4U; ++x) {
            size_t idx = (size_t)(qy + y) * stride + (qx + x);
            int32_t rv = rec_delta ? zcvid_dec_clamp_i32((int32_t)pred[idx] + rec_delta[y * 4U + x], 0, 255)
                                   : (int32_t)pred[idx];
            if (opt_rec) opt_rec[y * 4U + x] = (uint8_t)rv;
            int32_t d = (int32_t)src[idx] - rv;
            sse += (uint32_t)(d * d);
        }
    }
    return sse;
}

static inline uint32_t zcvid_lh_sb_sse_clamp(const uint8_t *src, const uint8_t *pred,
                                             uint32_t stride, uint32_t bx, uint32_t by,
                                             const int32_t rec_delta[64],
                                             uint8_t *opt_rec) {
    uint32_t sse = 0U;
    for (uint32_t y = 0U; y < 8U; ++y) {
        for (uint32_t x = 0U; x < 8U; ++x) {
            size_t idx = (size_t)(by + y) * stride + (bx + x);
            int32_t rv = rec_delta ? zcvid_dec_clamp_i32((int32_t)pred[idx] + rec_delta[y * 8U + x], 0, 255)
                                   : (int32_t)pred[idx];
            if (opt_rec) opt_rec[y * 8U + x] = (uint8_t)rv;
            int32_t d = (int32_t)src[idx] - rv;
            sse += (uint32_t)(d * d);
        }
    }
    return sse;
}

static uint32_t zcvid_lh_mode_rate_q8(const zcvid_lh_model_t *m, zcvid_lh_codec_t codec,
                                      uint32_t skip_ctx, zcvid_lh_blk_mode_t blk_mode) {
    if (blk_mode == ZCVID_LH_BLK_SKIP) {
        return zcvid_lh_cost_q8(&m->skip[skip_ctx], 0U);
    }
    uint32_t c = zcvid_lh_cost_q8(&m->skip[skip_ctx], 1U);
    if (codec == ZCVID_LH_TQ4 || codec == ZCVID_LH_LRF1) return c;
    if (codec == ZCVID_LH_LRF2) {
        return c + zcvid_lh_cost_q8(&m->mode_b[0], (blk_mode == ZCVID_LH_BLK_LRF2) ? 1U : 0U);
    }
    if (codec == ZCVID_LH_HYBRID_RDO) {
        uint32_t is_lrf = (blk_mode != ZCVID_LH_BLK_TQ4) ? 1U : 0U;
        c += zcvid_lh_cost_q8(&m->mode_b[0], is_lrf);
        if (is_lrf) {
            c += zcvid_lh_cost_q8(&m->mode_b[1], (blk_mode == ZCVID_LH_BLK_LRF2) ? 1U : 0U);
        }
        return c;
    }
    if (codec == ZCVID_LH_HYBRID_CASCADE) {
        uint32_t is_pure_tq = (blk_mode == ZCVID_LH_BLK_TQ4) ? 1U : 0U;
        c += zcvid_lh_cost_q8(&m->mode_b[0], is_pure_tq ? 0U : 1U);
        if (!is_pure_tq) {
            uint32_t is_r1 = (blk_mode == ZCVID_LH_BLK_LRF1) ? 1U : 0U;
            c += zcvid_lh_cost_q8(&m->mode_b[1], is_r1 ? 0U : 1U);
            if (!is_r1) {
                c += zcvid_lh_cost_q8(&m->mode_b[2], (blk_mode == ZCVID_LH_BLK_CASCADE) ? 1U : 0U);
            }
        }
        return c;
    }
    if (codec == ZCVID_LH_HYBRID_FAST) {
        /* {TQ4, LRF1, CASCADE, SPEC_LRF} (no LRF2) */
        uint32_t has_lrf = (blk_mode != ZCVID_LH_BLK_TQ4) ? 1U : 0U;
        c += zcvid_lh_cost_q8(&m->mode_b[0], has_lrf);
        if (has_lrf) {
            uint32_t is_spec = (blk_mode == ZCVID_LH_BLK_SPEC_LRF) ? 1U : 0U;
            c += zcvid_lh_cost_q8(&m->mode_b[3], is_spec);
            if (!is_spec) {
                c += zcvid_lh_cost_q8(&m->mode_b[1], (blk_mode == ZCVID_LH_BLK_CASCADE) ? 1U : 0U);
            }
        }
        return c;
    }
    if (codec == ZCVID_LH_HYBRID_ESR) {
        /* 1-bit fast spectral path: 0 -> POLY_ESR, 1 -> Spatial {TQ4, LRF1, CASCADE} */
        uint32_t is_spatial = (blk_mode != ZCVID_LH_BLK_POLY_ESR) ? 1U : 0U;
        c += zcvid_lh_cost_q8(&m->mode_b[0], is_spatial);
        if (is_spatial) {
            uint32_t is_lrf = (blk_mode != ZCVID_LH_BLK_TQ4) ? 1U : 0U;
            c += zcvid_lh_cost_q8(&m->mode_b[1], is_lrf);
            if (is_lrf) {
                c += zcvid_lh_cost_q8(&m->mode_b[2], (blk_mode == ZCVID_LH_BLK_CASCADE) ? 1U : 0U);
            }
        }
        return c;
    }
    /* ZCVID_LH_HYBRID_FULL: {TQ4, LRF1, LRF2, CASCADE, SPEC_LRF} */
    uint32_t has_lrf = (blk_mode != ZCVID_LH_BLK_TQ4) ? 1U : 0U;
    c += zcvid_lh_cost_q8(&m->mode_b[0], has_lrf);
    if (has_lrf) {
        uint32_t is_spec = (blk_mode == ZCVID_LH_BLK_SPEC_LRF) ? 1U : 0U;
        c += zcvid_lh_cost_q8(&m->mode_b[3], is_spec);
        if (!is_spec) {
            uint32_t is_r1 = (blk_mode == ZCVID_LH_BLK_LRF1) ? 1U : 0U;
            c += zcvid_lh_cost_q8(&m->mode_b[1], is_r1 ? 0U : 1U);
            if (!is_r1) {
                c += zcvid_lh_cost_q8(&m->mode_b[2], (blk_mode == ZCVID_LH_BLK_CASCADE) ? 1U : 0U);
            }
        }
    }
    return c;
}

static void zcvid_lh_encode_mode(zcrudp_rc_enc_t *rc, zcvid_lh_model_t *m, zcvid_lh_codec_t codec,
                                 uint32_t skip_ctx, zcvid_lh_blk_mode_t blk_mode) {
    if (blk_mode == ZCVID_LH_BLK_SKIP) {
        zcrudp_rc_enc_bit(rc, &m->skip[skip_ctx], 0U);
        return;
    }
    zcrudp_rc_enc_bit(rc, &m->skip[skip_ctx], 1U);
    if (codec == ZCVID_LH_TQ4 || codec == ZCVID_LH_LRF1) return;
    if (codec == ZCVID_LH_LRF2) {
        zcrudp_rc_enc_bit(rc, &m->mode_b[0], (blk_mode == ZCVID_LH_BLK_LRF2) ? 1U : 0U);
        return;
    }
    if (codec == ZCVID_LH_HYBRID_RDO) {
        uint32_t is_lrf = (blk_mode != ZCVID_LH_BLK_TQ4) ? 1U : 0U;
        zcrudp_rc_enc_bit(rc, &m->mode_b[0], is_lrf);
        if (is_lrf) {
            zcrudp_rc_enc_bit(rc, &m->mode_b[1], (blk_mode == ZCVID_LH_BLK_LRF2) ? 1U : 0U);
        }
        return;
    }
    if (codec == ZCVID_LH_HYBRID_CASCADE) {
        uint32_t has_lrf = (blk_mode != ZCVID_LH_BLK_TQ4) ? 1U : 0U;
        zcrudp_rc_enc_bit(rc, &m->mode_b[0], has_lrf);
        if (has_lrf) {
            uint32_t is_r1 = (blk_mode == ZCVID_LH_BLK_LRF1) ? 1U : 0U;
            zcrudp_rc_enc_bit(rc, &m->mode_b[1], is_r1 ? 0U : 1U);
            if (!is_r1) {
                zcrudp_rc_enc_bit(rc, &m->mode_b[2], (blk_mode == ZCVID_LH_BLK_CASCADE) ? 1U : 0U);
            }
        }
        return;
    }
    if (codec == ZCVID_LH_HYBRID_FAST) {
        uint32_t has_lrf = (blk_mode != ZCVID_LH_BLK_TQ4) ? 1U : 0U;
        zcrudp_rc_enc_bit(rc, &m->mode_b[0], has_lrf);
        if (has_lrf) {
            uint32_t is_spec = (blk_mode == ZCVID_LH_BLK_SPEC_LRF) ? 1U : 0U;
            zcrudp_rc_enc_bit(rc, &m->mode_b[3], is_spec);
            if (!is_spec) {
                zcrudp_rc_enc_bit(rc, &m->mode_b[1], (blk_mode == ZCVID_LH_BLK_CASCADE) ? 1U : 0U);
            }
        }
        return;
    }
    if (codec == ZCVID_LH_HYBRID_ESR) {
        uint32_t is_spatial = (blk_mode != ZCVID_LH_BLK_POLY_ESR) ? 1U : 0U;
        zcrudp_rc_enc_bit(rc, &m->mode_b[0], is_spatial);
        if (is_spatial) {
            uint32_t is_lrf = (blk_mode != ZCVID_LH_BLK_TQ4) ? 1U : 0U;
            zcrudp_rc_enc_bit(rc, &m->mode_b[1], is_lrf);
            if (is_lrf) {
                zcrudp_rc_enc_bit(rc, &m->mode_b[2], (blk_mode == ZCVID_LH_BLK_CASCADE) ? 1U : 0U);
            }
        }
        return;
    }
    /* FULL */
    uint32_t has_lrf = (blk_mode != ZCVID_LH_BLK_TQ4) ? 1U : 0U;
    zcrudp_rc_enc_bit(rc, &m->mode_b[0], has_lrf);
    if (has_lrf) {
        uint32_t is_spec = (blk_mode == ZCVID_LH_BLK_SPEC_LRF) ? 1U : 0U;
        zcrudp_rc_enc_bit(rc, &m->mode_b[3], is_spec);
        if (!is_spec) {
            uint32_t is_r1 = (blk_mode == ZCVID_LH_BLK_LRF1) ? 1U : 0U;
            zcrudp_rc_enc_bit(rc, &m->mode_b[1], is_r1 ? 0U : 1U);
            if (!is_r1) {
                zcrudp_rc_enc_bit(rc, &m->mode_b[2], (blk_mode == ZCVID_LH_BLK_CASCADE) ? 1U : 0U);
            }
        }
    }
}

static zcvid_lh_blk_mode_t zcvid_lh_decode_mode(zcrudp_rc_dec_t *rc, zcvid_lh_model_t *m,
                                                zcvid_lh_codec_t codec, uint32_t skip_ctx) {
    if (zcrudp_rc_dec_bit(rc, &m->skip[skip_ctx]) == 0U) return ZCVID_LH_BLK_SKIP;
    if (codec == ZCVID_LH_TQ4) return ZCVID_LH_BLK_TQ4;
    if (codec == ZCVID_LH_LRF1) return ZCVID_LH_BLK_LRF1;
    if (codec == ZCVID_LH_LRF2) {
        return zcrudp_rc_dec_bit(rc, &m->mode_b[0]) ? ZCVID_LH_BLK_LRF2 : ZCVID_LH_BLK_LRF1;
    }
    if (codec == ZCVID_LH_HYBRID_RDO) {
        if (zcrudp_rc_dec_bit(rc, &m->mode_b[0]) == 0U) return ZCVID_LH_BLK_TQ4;
        return zcrudp_rc_dec_bit(rc, &m->mode_b[1]) ? ZCVID_LH_BLK_LRF2 : ZCVID_LH_BLK_LRF1;
    }
    if (codec == ZCVID_LH_HYBRID_CASCADE) {
        if (zcrudp_rc_dec_bit(rc, &m->mode_b[0]) == 0U) return ZCVID_LH_BLK_TQ4;
        if (zcrudp_rc_dec_bit(rc, &m->mode_b[1]) == 0U) return ZCVID_LH_BLK_LRF1;
        return zcrudp_rc_dec_bit(rc, &m->mode_b[2]) ? ZCVID_LH_BLK_CASCADE : ZCVID_LH_BLK_LRF2;
    }
    if (codec == ZCVID_LH_HYBRID_FAST) {
        if (zcrudp_rc_dec_bit(rc, &m->mode_b[0]) == 0U) return ZCVID_LH_BLK_TQ4;
        if (zcrudp_rc_dec_bit(rc, &m->mode_b[3]) != 0U) return ZCVID_LH_BLK_SPEC_LRF;
        return zcrudp_rc_dec_bit(rc, &m->mode_b[1]) ? ZCVID_LH_BLK_CASCADE : ZCVID_LH_BLK_LRF1;
    }
    if (codec == ZCVID_LH_HYBRID_ESR) {
        if (zcrudp_rc_dec_bit(rc, &m->mode_b[0]) == 0U) return ZCVID_LH_BLK_POLY_ESR;
        if (zcrudp_rc_dec_bit(rc, &m->mode_b[1]) == 0U) return ZCVID_LH_BLK_TQ4;
        return zcrudp_rc_dec_bit(rc, &m->mode_b[2]) ? ZCVID_LH_BLK_CASCADE : ZCVID_LH_BLK_LRF1;
    }
    /* FULL */
    if (zcrudp_rc_dec_bit(rc, &m->mode_b[0]) == 0U) return ZCVID_LH_BLK_TQ4;
    if (zcrudp_rc_dec_bit(rc, &m->mode_b[3]) != 0U) return ZCVID_LH_BLK_SPEC_LRF;
    if (zcrudp_rc_dec_bit(rc, &m->mode_b[1]) == 0U) return ZCVID_LH_BLK_LRF1;
    return zcrudp_rc_dec_bit(rc, &m->mode_b[2]) ? ZCVID_LH_BLK_CASCADE : ZCVID_LH_BLK_LRF2;
}

/* =========================================================================
 * Fast AVX2/Scalar Block Helpers & Boosted Slice Encoder/Decoder
 * ========================================================================= */

static inline uint32_t zcvid_lh_load_sb_fast(const uint8_t *src, const uint8_t *pred,
                                             uint32_t stride, uint32_t bx, uint32_t by,
                                             int32_t src_i32[64], int32_t pred_i32[64],
                                             int32_t res[64]) {
#if defined(__AVX2__)
    __m256i sse_acc = _mm256_setzero_si256();
    for (uint32_t y = 0U; y < 8U; ++y) {
        size_t off = (size_t)(by + y) * stride + bx;
        __m256i sv = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(src + off)));
        __m256i pv = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(pred + off)));
        __m256i rv = _mm256_sub_epi32(sv, pv);
        _mm256_storeu_si256((__m256i *)(src_i32 + y * 8U), sv);
        _mm256_storeu_si256((__m256i *)(pred_i32 + y * 8U), pv);
        _mm256_storeu_si256((__m256i *)(res + y * 8U), rv);
        sse_acc = _mm256_add_epi32(sse_acc, _mm256_mullo_epi32(rv, rv));
    }
    return (uint32_t)zcvid_lh_hsum_epi32_avx2(sse_acc);
#else
    uint32_t sse = 0U;
    for (uint32_t y = 0U; y < 8U; ++y) {
        size_t off = (size_t)(by + y) * stride + bx;
        for (uint32_t x = 0U; x < 8U; ++x) {
            int32_t s = (int32_t)src[off + x];
            int32_t p = (int32_t)pred[off + x];
            int32_t d = s - p;
            src_i32[y * 8U + x] = s;
            pred_i32[y * 8U + x] = p;
            res[y * 8U + x] = d;
            sse += (uint32_t)(d * d);
        }
    }
    return sse;
#endif
}

static inline uint32_t zcvid_lh_sb_sse_clamp_i32(const int32_t src_i32[64],
                                                 const int32_t pred_i32[64],
                                                 const int32_t delta[64]) {
#if defined(__AVX2__)
    const __m256i zero = _mm256_setzero_si256();
    const __m256i c255 = _mm256_set1_epi32(255);
    __m256i acc = _mm256_setzero_si256();
    for (uint32_t y = 0U; y < 8U; ++y) {
        __m256i s = _mm256_loadu_si256((const __m256i *)(src_i32 + y * 8U));
        __m256i p = _mm256_loadu_si256((const __m256i *)(pred_i32 + y * 8U));
        __m256i d = _mm256_loadu_si256((const __m256i *)(delta + y * 8U));
        __m256i r = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(p, d), zero), c255);
        __m256i e = _mm256_sub_epi32(s, r);
        acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(e, e));
    }
    return (uint32_t)zcvid_lh_hsum_epi32_avx2(acc);
#else
    uint32_t sse = 0U;
    for (uint32_t i = 0U; i < 64U; ++i) {
        int32_t r = zcvid_dec_clamp_i32(pred_i32[i] + delta[i], 0, 255);
        int32_t e = src_i32[i] - r;
        sse += (uint32_t)(e * e);
    }
    return sse;
#endif
}

static inline void zcvid_lh_store_sb_clamp_u8(const uint8_t *pred, uint32_t stride,
                                              uint32_t bx, uint32_t by,
                                              const int32_t *delta, uint8_t *out_rec) {
    if (!delta) {
        for (uint32_t y = 0U; y < 8U; ++y) {
            memcpy(out_rec + (size_t)(by + y) * stride + bx,
                   pred + (size_t)(by + y) * stride + bx, 8U);
        }
        return;
    }
#if defined(__AVX2__)
    const __m256i zero = _mm256_setzero_si256();
    const __m256i c255 = _mm256_set1_epi32(255);
    for (uint32_t y = 0U; y < 8U; ++y) {
        size_t off = (size_t)(by + y) * stride + bx;
        __m256i pv = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(pred + off)));
        __m256i dv = _mm256_loadu_si256((const __m256i *)(delta + y * 8U));
        __m256i rv = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(pv, dv), zero), c255);
        __m128i lo = _mm256_castsi256_si128(rv);
        __m128i hi = _mm256_extracti128_si256(rv, 1);
        __m128i p16 = _mm_packus_epi32(lo, hi);
        __m128i p8 = _mm_packus_epi16(p16, p16);
        _mm_storel_epi64((__m128i *)(out_rec + off), p8);
    }
#else
    for (uint32_t y = 0U; y < 8U; ++y) {
        size_t off = (size_t)(by + y) * stride + bx;
        for (uint32_t x = 0U; x < 8U; ++x) {
            out_rec[off + x] = (uint8_t)zcvid_dec_clamp_i32((int32_t)pred[off + x] + delta[y * 8U + x], 0, 255);
        }
    }
#endif
}

static inline uint32_t zcvid_lh_quad_sse_clamp_i32(const int32_t src_i32[64],
                                                   const int32_t pred_i32[64],
                                                   uint32_t q, const int32_t qdelta[16]) {
    uint32_t qx0 = (q & 1U) * 4U;
    uint32_t qy0 = (q >> 1) * 4U;
    uint32_t sse = 0U;
    for (uint32_t y = 0U; y < 4U; ++y) {
        uint32_t base = (qy0 + y) * 8U + qx0;
        for (uint32_t x = 0U; x < 4U; ++x) {
            int32_t r = zcvid_dec_clamp_i32(pred_i32[base + x] + qdelta[y * 4U + x], 0, 255);
            int32_t e = src_i32[base + x] - r;
            sse += (uint32_t)(e * e);
        }
    }
    return sse;
}

/* Boosted single-slice encoder supporting ZCVID_LH_HYBRID_FAST and ZCVID_LH_TQ4.
 * Pure integer RDO, exact SKIP early-out, AVX2 DCT8 + integer LRF + rank-1 IDCT8,
 * and energy-gated CASCADE (no LRF2 dead weight). */
static size_t zcvid_lh_encode_slice_fast(zcvid_lh_codec_t codec, const uint8_t *src,
                                         const uint8_t *pred, uint32_t w,
                                         uint32_t sby_begin, uint32_t sby_end,
                                         uint32_t qp, bool is_inter,
                                         const zcvid_lh_step_lut_t *step_lut,
                                         uint64_t lambda_q8,
                                         uint8_t *up_row,
                                         uint8_t *out_bs, size_t out_cap,
                                         uint8_t *out_rec,
                                         zcvid_lh_stats_t *out_stats) {
    (void)qp;
    const uint32_t sbw = w >> 3;
    memset(up_row, 0, sbw);
    if (out_stats) memset(out_stats, 0, sizeof(*out_stats));

    zcrudp_rc_enc_t rc;
    zcrudp_rc_enc_init(&rc, out_bs, out_cap);
    zcvid_lh_model_t m;
    zcvid_lh_model_init(&m);

    const bool is_hybrid = (codec == ZCVID_LH_HYBRID_FAST || codec == ZCVID_LH_HYBRID_ESR);
    const bool use_esr   = (codec == ZCVID_LH_HYBRID_ESR);

    for (uint32_t sby = sby_begin; sby < sby_end; ++sby) {
        uint8_t left_nz = 0U;
        for (uint32_t sbx = 0U; sbx < sbw; ++sbx) {
            uint32_t bx = sbx * 8U, by = sby * 8U;
            uint32_t skip_ctx = (uint32_t)left_nz + ((uint32_t)up_row[sbx] << 1);

            int32_t src_i32[64], pred_i32[64], res[64];
            uint32_t skip_sse = zcvid_lh_load_sb_fast(src, pred, w, bx, by, src_i32, pred_i32, res);

            uint64_t skip_0_q8 = (uint64_t)zcvid_lh_cost_q8(&m.skip[skip_ctx], 0U);
            uint64_t skip_1_q8 = (uint64_t)zcvid_lh_cost_q8(&m.skip[skip_ctx], 1U);
            uint64_t skip_j = ((uint64_t)skip_sse << 16) + lambda_q8 * skip_0_q8;

            /* Exact SKIP early-out: any non-skip mode needs skip=1 + >=2 bits of mode/payload */
            if (skip_sse == 0U || skip_j <= lambda_q8 * (skip_1_q8 + 512ULL)) {
                zcrudp_rc_enc_bit(&rc, &m.skip[skip_ctx], 0U);
                if (out_stats) out_stats->count[(unsigned)ZCVID_LH_BLK_SKIP]++;
                if (out_rec) zcvid_lh_store_sb_clamp_u8(pred, w, bx, by, NULL, out_rec);
                left_nz = 0U;
                up_row[sbx] = 0U;
                if (rc.overflow) return 0U;
                continue;
            }

            zcvid_lh_blk_mode_t best_mode = ZCVID_LH_BLK_SKIP;
            uint64_t best_j = skip_j;

            /* Candidate A: Shared-Routing 8x8 Spectral Stage (POLY_ESR + SPEC_LRF)
             * Single forward DCT8 (zero-vpmulld AVX2 / 12-mul Negacyclic-Karatsuba)
             * shared between Polynomial+Exact-Spectral-Repair and Rank-1 SPEC_LRF. */
            zcvid_lh_lrf_t spec_lrf;
            int32_t spec_delta[64];
            spec_lrf.level = 0;
            uint32_t spec_sse = 0xFFFFFFFFU;
            zcvid_lh_esr_t esr;
            int32_t esr_delta[64];
            uint32_t esr_sse = 0xFFFFFFFFU;
            uint32_t esr_rate_q8 = 0xFFFFFFFFU;
            bool skip_rest = false;
            if (is_hybrid) {
                int32_t wmat[64];
                zcvid_lh_dct8_fwd(res, wmat);

                if (use_esr) {
                    if (zcvid_lh_esr_quant_and_prune(wmat, step_lut, &m, lambda_q8, is_inter, &esr)) {
                        zcvid_lh_esr_recon(&esr, step_lut, esr_delta);
                        esr_sse = zcvid_lh_sb_sse_clamp_i32(src_i32, pred_i32, esr_delta);
                        esr_rate_q8 = zcvid_lh_mode_rate_q8(&m, codec, skip_ctx, ZCVID_LH_BLK_POLY_ESR)
                                    + zcvid_lh_esr_rate_q8(&m, &esr);
                        uint64_t je = ((uint64_t)esr_sse << 16) + lambda_q8 * (uint64_t)esr_rate_q8;
                        if (je < best_j) {
                            best_j = je;
                            best_mode = ZCVID_LH_BLK_POLY_ESR;
                            /* Early-accept ONLY when POLY_ESR is both accurate AND compact
                             * (degree-2 polynomial or low-bit LL block <= 20 bits). High-bit
                             * HF blocks (checkerboards, crisp UI edges) still test LRF1/CASCADE! */
                            if (esr_sse == 0U && esr_rate_q8 <= (20U << 8)) {
                                skip_rest = true;
                            } else if (esr.poly_only && esr_sse <= (skip_sse >> 4) &&
                                       ((uint64_t)esr_sse << 16) <= lambda_q8 * 1024ULL) {
                                skip_rest = true;
                            } else if (esr.csbf == 1U && esr_rate_q8 <= (24U << 8) &&
                                       esr_sse <= (skip_sse >> 4) &&
                                       ((uint64_t)esr_sse << 16) <= lambda_q8 * 1536ULL) {
                                skip_rest = true;
                            }
                        }
                    }
                } else {
                    zcvid_lh_fit_lrf_fast(wmat, step_lut, true, &spec_lrf);
                    if (spec_lrf.level != 0) {
                        zcvid_lh_spec_lrf_recon(&spec_lrf, spec_delta);
                        spec_sse = zcvid_lh_sb_sse_clamp_i32(src_i32, pred_i32, spec_delta);
                        uint32_t rs_q8 = zcvid_lh_mode_rate_q8(&m, codec, skip_ctx, ZCVID_LH_BLK_SPEC_LRF)
                                       + zcvid_lh_lrf_rate_q8(&m, &spec_lrf);
                        uint64_t js = ((uint64_t)spec_sse << 16) + lambda_q8 * (uint64_t)rs_q8;
                        if (js < best_j) {
                            best_j = js;
                            best_mode = ZCVID_LH_BLK_SPEC_LRF;
                            if (spec_sse == 0U ||
                                (spec_sse <= (skip_sse >> 4) && ((uint64_t)spec_sse << 16) <= lambda_q8 * 1024ULL)) {
                                skip_rest = true;
                            }
                        }
                    }
                }
            }

            /* Candidate B: TQ4 (with per-quadrant SKIP, LUT quant/dequant & DC-only fast path)
             * In HYBRID_ESR, skip 4x TQ4 only when 8x8 POLY_ESR already solved the block
             * compactly in the LL sub-block (csbf==1, <=24 bits, >87.5% energy); allow TQ4
             * to compete on high-rate / multi-subblock HF & localized 1-quadrant blocks! */
            int16_t tq_lv[4][16];
            uint32_t tq_cbp = 0U;
            int32_t tq_delta[64];
            const bool try_tq4 = !skip_rest &&
                !(use_esr && best_mode == ZCVID_LH_BLK_POLY_ESR &&
                  esr.csbf == 1U && esr_rate_q8 <= (24U << 8) && esr_sse <= (skip_sse >> 3));
            if (try_tq4) {
                memset(tq_delta, 0, sizeof(tq_delta));
                uint32_t tq_sse = 0U;
                uint32_t tq_rate_q8 = zcvid_lh_mode_rate_q8(&m, codec, skip_ctx, ZCVID_LH_BLK_TQ4);
                for (uint32_t q = 0U; q < 4U; ++q) {
                    uint32_t qx0 = (q & 1U) * 4U;
                    uint32_t qy0 = (q >> 1) * 4U;
                    int32_t qr[16];
                    uint32_t sse0 = 0U;
                    for (uint32_t y = 0U; y < 4U; ++y) {
                        uint32_t base = (qy0 + y) * 8U + qx0;
                        for (uint32_t x = 0U; x < 4U; ++x) {
                            int32_t v = res[base + x];
                            qr[y * 4U + x] = v;
                            sse0 += (uint32_t)(v * v);
                        }
                    }
                    uint32_t cbp0_q8 = zcvid_lh_cost_q8(&m.tq_cbp[q], 0U);
                    uint32_t cbp1_q8 = zcvid_lh_cost_q8(&m.tq_cbp[q], 1U);
                    uint64_t j0 = ((uint64_t)sse0 << 16) + lambda_q8 * (uint64_t)cbp0_q8;
                    if (sse0 == 0U || j0 <= lambda_q8 * ((uint64_t)cbp1_q8 + 512ULL)) {
                        tq_sse += sse0;
                        tq_rate_q8 += cbp0_q8;
                        continue;
                    }
                    int32_t qw[16];
                    int16_t qlv[16];
                    zcvid_lh_tq4_forward(qr, qw);
                    uint32_t sig_mask = zcvid_lh_tq4_quant_fast(qw, step_lut, is_inter, qlv);
                    if (sig_mask == 0U) {
                        tq_sse += sse0;
                        tq_rate_q8 += cbp0_q8;
                        continue;
                    }
                    uint32_t rq8 = zcvid_lh_tq4_rate_q8_mask(&m, qlv, sig_mask);
                    int32_t qrec[16];
                    zcvid_lh_tq4_recon_fast(qlv, sig_mask, step_lut, qrec);
                    uint32_t sse1 = zcvid_lh_quad_sse_clamp_i32(src_i32, pred_i32, q, qrec);
                    uint64_t j1 = ((uint64_t)sse1 << 16) + lambda_q8 * (uint64_t)(cbp1_q8 + rq8);
                    if (j1 < j0) {
                        tq_cbp |= (1U << q);
                        memcpy(tq_lv[q], qlv, sizeof(qlv));
                        tq_sse += sse1;
                        tq_rate_q8 += cbp1_q8 + rq8;
                        for (uint32_t y = 0U; y < 4U; ++y) {
                            uint32_t base = (qy0 + y) * 8U + qx0;
                            for (uint32_t x = 0U; x < 4U; ++x) {
                                tq_delta[base + x] = qrec[y * 4U + x];
                            }
                        }
                    } else {
                        tq_sse += sse0;
                        tq_rate_q8 += cbp0_q8;
                    }
                }
                if (tq_cbp != 0U) {
                    uint64_t jt = ((uint64_t)tq_sse << 16) + lambda_q8 * (uint64_t)tq_rate_q8;
                    if (jt < best_j) {
                        best_j = jt;
                        best_mode = ZCVID_LH_BLK_TQ4;
                    }
                }
            }

            /* Candidate C: LRF1 & Energy-Gated CASCADE
             * Skip spatial LRF1/CASCADE when current best J is already cheaper than
             * the ~10-bit floor of an LRF factor, or SPEC_LRF already captured >87.5% energy. */
            zcvid_lh_lrf_t lrf1;
            int32_t lrf1_delta[64];
            int16_t cas_lv[4][16];
            uint32_t cas_cbp = 0U;
            int32_t cas_delta[64];
            lrf1.level = 0;

            const bool try_spatial_lrf = is_hybrid && !skip_rest
                && best_j > lambda_q8 * (10ULL << 8)
                && !(best_mode == ZCVID_LH_BLK_SPEC_LRF && spec_sse <= (skip_sse >> 3))
                && !(best_mode == ZCVID_LH_BLK_POLY_ESR && esr_rate_q8 <= (24U << 8) && esr_sse <= (skip_sse >> 3));

            if (try_spatial_lrf) {
                zcvid_lh_fit_lrf_fast(res, step_lut, false, &lrf1);
                if (lrf1.level != 0) {
                    memset(lrf1_delta, 0, sizeof(lrf1_delta));
                    zcvid_lh_recon_lrf_add(&lrf1, lrf1_delta);
                    uint32_t sse1 = zcvid_lh_sb_sse_clamp_i32(src_i32, pred_i32, lrf1_delta);
                    uint32_t lrf1_rq8 = zcvid_lh_lrf_rate_q8(&m, &lrf1);
                    uint32_t r1_q8 = zcvid_lh_mode_rate_q8(&m, codec, skip_ctx, ZCVID_LH_BLK_LRF1)
                                   + lrf1_rq8;
                    uint64_t j1 = ((uint64_t)sse1 << 16) + lambda_q8 * (uint64_t)r1_q8;
                    if (j1 < best_j) {
                        best_j = j1;
                        best_mode = ZCVID_LH_BLK_LRF1;
                    }

                    /* Gate CASCADE: only refine with 4x TQ4 if LRF1 captured >=25% energy
                     * AND left enough residual energy to justify >=2 bits of TQ4. */
                    if (sse1 < (skip_sse * 3U) / 4U && ((uint64_t)sse1 << 16) > lambda_q8 * 512ULL) {
                        memcpy(cas_delta, lrf1_delta, sizeof(lrf1_delta));
                        uint32_t cas_sse = 0U;
                        uint32_t cas_rate_q8 = zcvid_lh_mode_rate_q8(&m, codec, skip_ctx, ZCVID_LH_BLK_CASCADE)
                                             + lrf1_rq8;
                        for (uint32_t q = 0U; q < 4U; ++q) {
                            uint32_t qx0 = (q & 1U) * 4U;
                            uint32_t qy0 = (q >> 1) * 4U;
                            int32_t qr[16], qbase[16];
                            for (uint32_t y = 0U; y < 4U; ++y) {
                                uint32_t base = (qy0 + y) * 8U + qx0;
                                for (uint32_t x = 0U; x < 4U; ++x) {
                                    int32_t b = lrf1_delta[base + x];
                                    qbase[y * 4U + x] = b;
                                    qr[y * 4U + x] = res[base + x] - b;
                                }
                            }
                            uint32_t sse0 = zcvid_lh_quad_sse_clamp_i32(src_i32, pred_i32, q, qbase);
                            uint32_t cbp0_q8 = zcvid_lh_cost_q8(&m.tq_cbp[q], 0U);
                            uint32_t cbp1_q8 = zcvid_lh_cost_q8(&m.tq_cbp[q], 1U);
                            uint64_t jq0 = ((uint64_t)sse0 << 16) + lambda_q8 * (uint64_t)cbp0_q8;
                            if (sse0 == 0U || jq0 <= lambda_q8 * ((uint64_t)cbp1_q8 + 512ULL)) {
                                cas_sse += sse0;
                                cas_rate_q8 += cbp0_q8;
                                continue;
                            }
                            int32_t qw[16];
                            int16_t qlv[16];
                            zcvid_lh_tq4_forward(qr, qw);
                            uint32_t sig_mask = zcvid_lh_tq4_quant_fast(qw, step_lut, is_inter, qlv);
                            if (sig_mask == 0U) {
                                cas_sse += sse0;
                                cas_rate_q8 += cbp0_q8;
                                continue;
                            }
                            uint32_t rq8 = zcvid_lh_tq4_rate_q8_mask(&m, qlv, sig_mask);
                            int32_t qrec[16], qcomb[16];
                            zcvid_lh_tq4_recon_fast(qlv, sig_mask, step_lut, qrec);
                            for (int k = 0; k < 16; ++k) qcomb[k] = qbase[k] + qrec[k];
                            uint32_t csse1 = zcvid_lh_quad_sse_clamp_i32(src_i32, pred_i32, q, qcomb);
                            uint64_t jq1 = ((uint64_t)csse1 << 16) + lambda_q8 * (uint64_t)(cbp1_q8 + rq8);
                            if (jq1 < jq0) {
                                cas_cbp |= (1U << q);
                                memcpy(cas_lv[q], qlv, sizeof(qlv));
                                cas_sse += csse1;
                                cas_rate_q8 += cbp1_q8 + rq8;
                                for (uint32_t y = 0U; y < 4U; ++y) {
                                    uint32_t base = (qy0 + y) * 8U + qx0;
                                    for (uint32_t x = 0U; x < 4U; ++x) {
                                        cas_delta[base + x] = qcomb[y * 4U + x];
                                    }
                                }
                            } else {
                                cas_sse += sse0;
                                cas_rate_q8 += cbp0_q8;
                            }
                        }
                        if (cas_cbp != 0U) {
                            uint64_t jc = ((uint64_t)cas_sse << 16) + lambda_q8 * (uint64_t)cas_rate_q8;
                            if (jc < best_j) {
                                best_j = jc;
                                best_mode = ZCVID_LH_BLK_CASCADE;
                            }
                        }
                    }
                }
            }

            /* Emit chosen mode & store exact encoder reconstruction */
            zcvid_lh_encode_mode(&rc, &m, codec, skip_ctx, best_mode);
            if (out_stats) out_stats->count[(unsigned)best_mode]++;

            const int32_t *final_delta = NULL;
            if (best_mode == ZCVID_LH_BLK_TQ4) {
                for (uint32_t q = 0U; q < 4U; ++q) {
                    uint32_t act = (tq_cbp >> q) & 1U;
                    zcrudp_rc_enc_bit(&rc, &m.tq_cbp[q], act);
                    if (act) zcvid_lh_tq4_encode_quad(&rc, &m, tq_lv[q]);
                }
                final_delta = tq_delta;
            } else if (best_mode == ZCVID_LH_BLK_LRF1) {
                zcvid_lh_lrf_encode(&rc, &m, &lrf1);
                final_delta = lrf1_delta;
            } else if (best_mode == ZCVID_LH_BLK_CASCADE) {
                zcvid_lh_lrf_encode(&rc, &m, &lrf1);
                for (uint32_t q = 0U; q < 4U; ++q) {
                    uint32_t act = (cas_cbp >> q) & 1U;
                    zcrudp_rc_enc_bit(&rc, &m.tq_cbp[q], act);
                    if (act) zcvid_lh_tq4_encode_quad(&rc, &m, cas_lv[q]);
                }
                final_delta = cas_delta;
            } else if (best_mode == ZCVID_LH_BLK_SPEC_LRF) {
                zcvid_lh_lrf_encode(&rc, &m, &spec_lrf);
                final_delta = spec_delta;
            } else if (best_mode == ZCVID_LH_BLK_POLY_ESR) {
                zcvid_lh_esr_encode(&rc, &m, &esr);
                final_delta = esr_delta;
            }

            if (out_rec) zcvid_lh_store_sb_clamp_u8(pred, w, bx, by, final_delta, out_rec);

            left_nz = (best_mode != ZCVID_LH_BLK_SKIP) ? 1U : 0U;
            up_row[sbx] = left_nz;
            if (rc.overflow) return 0U;
        }
    }

    size_t body = zcrudp_rc_enc_finish(&rc);
    if (rc.overflow) return 0U;
    if (body == 0U) {
        if (out_cap < 1U) return 0U;
        out_bs[0] = 0U;
        return 1U;
    }
    return body;
}

/* Fast slice decoder supporting all codecs (uses LUT + separable rank-1 IDCT8). */
static int zcvid_lh_decode_slice(zcvid_lh_codec_t codec, const uint8_t *bs, size_t len,
                                 const uint8_t *pred, uint32_t w,
                                 uint32_t sby_begin, uint32_t sby_end,
                                 uint32_t qp, const zcvid_lh_step_lut_t *step_lut,
                                 uint8_t *up_row, uint8_t *out_dec) {
    if (sby_begin >= sby_end) return 0;
    if (!bs || len == 0U) return -1;
    (void)qp;
    const uint32_t sbw = w >> 3;
    memset(up_row, 0, sbw);

    zcrudp_rc_dec_t rc;
    zcrudp_rc_dec_init(&rc, bs, len);
    zcvid_lh_model_t m;
    zcvid_lh_model_init(&m);

    for (uint32_t sby = sby_begin; sby < sby_end; ++sby) {
        uint8_t left_nz = 0U;
        for (uint32_t sbx = 0U; sbx < sbw; ++sbx) {
            uint32_t bx = sbx * 8U, by = sby * 8U;
            uint32_t skip_ctx = (uint32_t)left_nz + ((uint32_t)up_row[sbx] << 1);
            zcvid_lh_blk_mode_t bm = zcvid_lh_decode_mode(&rc, &m, codec, skip_ctx);

            if (bm == ZCVID_LH_BLK_SKIP) {
                zcvid_lh_store_sb_clamp_u8(pred, w, bx, by, NULL, out_dec);
                left_nz = 0U;
                up_row[sbx] = 0U;
                continue;
            }

            int32_t delta[64];
            bool ok = true;

            if (bm == ZCVID_LH_BLK_SPEC_LRF) {
                zcvid_lh_lrf_t f;
                ok = zcvid_lh_lrf_decode_lut(&rc, &m, step_lut, true, &f);
                if (ok) zcvid_lh_spec_lrf_recon(&f, delta);
            } else if (bm == ZCVID_LH_BLK_POLY_ESR) {
                zcvid_lh_esr_t esr;
                ok = zcvid_lh_esr_decode(&rc, &m, &esr);
                if (ok) zcvid_lh_esr_recon(&esr, step_lut, delta);
            } else {
                memset(delta, 0, sizeof(delta));
                if (bm == ZCVID_LH_BLK_TQ4 || bm == ZCVID_LH_BLK_CASCADE) {
                    if (bm == ZCVID_LH_BLK_CASCADE) {
                        zcvid_lh_lrf_t f;
                        ok = zcvid_lh_lrf_decode_lut(&rc, &m, step_lut, false, &f);
                        if (ok) zcvid_lh_recon_lrf_add(&f, delta);
                    }
                    for (uint32_t q = 0U; q < 4U && ok; ++q) {
                        if (zcrudp_rc_dec_bit(&rc, &m.tq_cbp[q]) == 0U) continue;
                        int16_t qlv[16];
                        if (!zcvid_lh_tq4_decode_quad(&rc, &m, qlv)) {
                            ok = false;
                            break;
                        }
                        uint32_t sm = 0U;
                        for (uint32_t k = 0U; k < 16U; ++k) if (qlv[k] != 0) sm |= (1U << k);
                        int32_t qrec[16];
                        zcvid_lh_tq4_recon_fast(qlv, sm, step_lut, qrec);
                        for (uint32_t y = 0U; y < 4U; ++y) {
                            uint32_t base = ((q >> 1) * 4U + y) * 8U + (q & 1U) * 4U;
                            for (uint32_t x = 0U; x < 4U; ++x) {
                                delta[base + x] += qrec[y * 4U + x];
                            }
                        }
                    }
                } else if (bm == ZCVID_LH_BLK_LRF1) {
                    zcvid_lh_lrf_t f;
                    ok = zcvid_lh_lrf_decode_lut(&rc, &m, step_lut, false, &f);
                    if (ok) zcvid_lh_recon_lrf_add(&f, delta);
                } else if (bm == ZCVID_LH_BLK_LRF2) {
                    zcvid_lh_lrf_t f1, f2;
                    ok = zcvid_lh_lrf_decode_lut(&rc, &m, step_lut, false, &f1)
                      && zcvid_lh_lrf_decode_lut(&rc, &m, step_lut, false, &f2);
                    if (ok) {
                        zcvid_lh_recon_lrf_add(&f1, delta);
                        zcvid_lh_recon_lrf_add(&f2, delta);
                    }
                }
            }

            if (!ok) return -1;

            zcvid_lh_store_sb_clamp_u8(pred, w, bx, by, delta, out_dec);
            left_nz = 1U;
            up_row[sbx] = 1U;
        }
    }

    return 0;
}

static size_t zcvid_lh_encode_frame(zcvid_lh_codec_t codec, const uint8_t *src, const uint8_t *pred,
                                    uint32_t w, uint32_t h, uint32_t qp, bool is_inter,
                                    uint8_t *out_bs, size_t out_cap, uint8_t *out_rec,
                                    zcvid_lh_stats_t *out_stats) {
    if (!src || !pred || !out_bs || out_cap < 16U ||
        w == 0U || h == 0U || w > 65535U || h > 65535U ||
        (w & 7U) != 0U || (h & 7U) != 0U ||
        (unsigned)codec >= ZCVID_LH_CODEC_COUNT || qp > 51U) return 0U;
    if (out_stats) memset(out_stats, 0, sizeof(*out_stats));

    /* Header: magic 'ZCLH', u16 w, u16 h, u8 codec, u8 qp, u8 inter, u8 reserved */
    out_bs[0] = 'Z'; out_bs[1] = 'C'; out_bs[2] = 'L'; out_bs[3] = 'H';
    out_bs[4] = (uint8_t)(w & 0xFFU); out_bs[5] = (uint8_t)(w >> 8);
    out_bs[6] = (uint8_t)(h & 0xFFU); out_bs[7] = (uint8_t)(h >> 8);
    out_bs[8] = (uint8_t)codec;
    out_bs[9] = (uint8_t)qp;
    out_bs[10] = is_inter ? 1U : 0U;
    out_bs[11] = 0U;

    const uint32_t sbw = w >> 3;
    const uint32_t sbh = h >> 3;
    uint8_t *up_row = (uint8_t *)calloc(sbw, 1U);
    if (!up_row) return 0U;

    /* H.264/JVT Lagrangian multiplier: lambda = 0.85 * 2^((qp - 12)/3) */
    const double lambda = 0.85 * pow(2.0, ((double)qp - 12.0) / 3.0);

    if (codec == ZCVID_LH_HYBRID_FAST || codec == ZCVID_LH_HYBRID_ESR) {
        zcvid_lh_step_lut_t step_lut;
        zcvid_lh_step_lut_init(&step_lut, qp);
        uint64_t lambda_q8 = (uint64_t)llround(lambda * 256.0);
        size_t body = zcvid_lh_encode_slice_fast(codec, src, pred, w, 0U, sbh, qp, is_inter,
                                                 &step_lut, lambda_q8, up_row,
                                                 out_bs + 12U, out_cap - 12U, out_rec, out_stats);
        free(up_row);
        return (body > 0U) ? (12U + body) : 0U;
    }

    zcrudp_rc_enc_t rc;
    zcrudp_rc_enc_init(&rc, out_bs + 12U, out_cap - 12U);
    zcvid_lh_model_t m;
    zcvid_lh_model_init(&m);

    for (uint32_t sby = 0U; sby < sbh; ++sby) {
        uint8_t left_nz = 0U;
        for (uint32_t sbx = 0U; sbx < sbw; ++sbx) {
            uint32_t bx = sbx * 8U, by = sby * 8U;
            uint32_t skip_ctx = (uint32_t)left_nz + ((uint32_t)up_row[sbx] << 1);

            int32_t res[64];
            for (uint32_t y = 0U; y < 8U; ++y) {
                for (uint32_t x = 0U; x < 8U; ++x) {
                    size_t idx = (size_t)(by + y) * w + (bx + x);
                    res[y * 8U + x] = (int32_t)src[idx] - (int32_t)pred[idx];
                }
            }

            /* Candidate 0: SKIP */
            uint32_t skip_sse = zcvid_lh_sb_sse_clamp(src, pred, w, bx, by, NULL, NULL);
            double skip_j = (double)skip_sse + lambda * ((double)zcvid_lh_mode_rate_q8(&m, codec, skip_ctx, ZCVID_LH_BLK_SKIP) / 256.0);

            zcvid_lh_blk_mode_t best_mode = ZCVID_LH_BLK_SKIP;
            double best_j = skip_j;

            /* Candidate 1: TQ4 (independent 4x4 RDO on each quadrant) */
            int16_t tq_lv[4][16];
            uint32_t tq_cbp = 0U;
            int32_t tq_delta[64];
            memset(tq_lv, 0, sizeof(tq_lv));
            memset(tq_delta, 0, sizeof(tq_delta));
            if (codec == ZCVID_LH_TQ4 || codec >= ZCVID_LH_HYBRID_RDO) {
                uint32_t tq_sse = 0U;
                uint32_t tq_rate_q8 = zcvid_lh_mode_rate_q8(&m, codec, skip_ctx, ZCVID_LH_BLK_TQ4);
                for (uint32_t q = 0U; q < 4U; ++q) {
                    uint32_t qx = bx + (q & 1U) * 4U;
                    uint32_t qy = by + (q >> 1) * 4U;
                    int32_t qr[16], qw[16], qrec[16];
                    int16_t qlv[16];
                    for (uint32_t y = 0U; y < 4U; ++y) {
                        for (uint32_t x = 0U; x < 4U; ++x) {
                            qr[y * 4U + x] = res[((q >> 1) * 4U + y) * 8U + (q & 1U) * 4U + x];
                        }
                    }
                    zcvid_lh_tq4_forward(qr, qw);
                    zcvid_lh_tq4_quant(qw, qp, is_inter, qlv);
                    zcvid_lh_tq4_recon(qlv, qp, qrec);

                    uint32_t sse0 = zcvid_lh_quad_sse_clamp(src, pred, w, qx, qy, NULL, NULL);
                    uint32_t sse1 = zcvid_lh_quad_sse_clamp(src, pred, w, qx, qy, qrec, NULL);
                    uint32_t rq8 = zcvid_lh_tq4_rate_q8(&m, qlv);
                    double j0 = (double)sse0 + lambda * ((double)zcvid_lh_cost_q8(&m.tq_cbp[q], 0U) / 256.0);
                    double j1 = (double)sse1 + lambda * ((double)(zcvid_lh_cost_q8(&m.tq_cbp[q], 1U) + rq8) / 256.0);
                    if (rq8 > 0U && j1 < j0) {
                        tq_cbp |= (1U << q);
                        memcpy(tq_lv[q], qlv, sizeof(qlv));
                        tq_sse += sse1;
                        tq_rate_q8 += zcvid_lh_cost_q8(&m.tq_cbp[q], 1U) + rq8;
                        for (uint32_t y = 0U; y < 4U; ++y) {
                            for (uint32_t x = 0U; x < 4U; ++x) {
                                tq_delta[((q >> 1) * 4U + y) * 8U + (q & 1U) * 4U + x] = qrec[y * 4U + x];
                            }
                        }
                    } else {
                        tq_sse += sse0;
                        tq_rate_q8 += zcvid_lh_cost_q8(&m.tq_cbp[q], 0U);
                    }
                }
                if (tq_cbp != 0U) {
                    double j = (double)tq_sse + lambda * ((double)tq_rate_q8 / 256.0);
                    if (j < best_j) {
                        best_j = j;
                        best_mode = ZCVID_LH_BLK_TQ4;
                    }
                }
            }

            /* Candidate 2 & 3: LRF1 & LRF2 */
            zcvid_lh_lrf_t lrf1, lrf2;
            int32_t lrf1_delta[64], lrf2_delta[64];
            memset(&lrf1, 0, sizeof(lrf1));
            memset(&lrf2, 0, sizeof(lrf2));
            memset(lrf1_delta, 0, sizeof(lrf1_delta));
            memset(lrf2_delta, 0, sizeof(lrf2_delta));

            if (codec >= ZCVID_LH_LRF1) {
                zcvid_lh_fit_lrf(res, qp, false, &lrf1);
                if (lrf1.level != 0) {
                    zcvid_lh_recon_lrf_add(&lrf1, lrf1_delta);
                    uint32_t sse1 = zcvid_lh_sb_sse_clamp(src, pred, w, bx, by, lrf1_delta, NULL);
                    uint32_t r1_q8 = zcvid_lh_mode_rate_q8(&m, codec, skip_ctx, ZCVID_LH_BLK_LRF1)
                                   + zcvid_lh_lrf_rate_q8(&m, &lrf1);
                    double j1 = (double)sse1 + lambda * ((double)r1_q8 / 256.0);
                    if (j1 < best_j) {
                        best_j = j1;
                        best_mode = ZCVID_LH_BLK_LRF1;
                    }

                    if (codec >= ZCVID_LH_LRF2) {
                        int32_t res2[64];
                        for (int k = 0; k < 64; ++k) res2[k] = res[k] - lrf1_delta[k];
                        zcvid_lh_fit_lrf(res2, qp, false, &lrf2);
                        if (lrf2.level != 0) {
                            memcpy(lrf2_delta, lrf1_delta, sizeof(lrf1_delta));
                            zcvid_lh_recon_lrf_add(&lrf2, lrf2_delta);
                            uint32_t sse2 = zcvid_lh_sb_sse_clamp(src, pred, w, bx, by, lrf2_delta, NULL);
                            uint32_t r2_q8 = zcvid_lh_mode_rate_q8(&m, codec, skip_ctx, ZCVID_LH_BLK_LRF2)
                                           + zcvid_lh_lrf_rate_q8(&m, &lrf1)
                                           + zcvid_lh_lrf_rate_q8(&m, &lrf2);
                            double j2 = (double)sse2 + lambda * ((double)r2_q8 / 256.0);
                            if (j2 < best_j) {
                                best_j = j2;
                                best_mode = ZCVID_LH_BLK_LRF2;
                            }
                        }
                    }
                }
            }

            /* Candidate 4: CASCADE (LRF1 + TQ4 on remaining residual) */
            int16_t cas_lv[4][16];
            uint32_t cas_cbp = 0U;
            int32_t cas_delta[64];
            memset(cas_lv, 0, sizeof(cas_lv));
            memset(cas_delta, 0, sizeof(cas_delta));
            if (codec >= ZCVID_LH_HYBRID_CASCADE && lrf1.level != 0) {
                memcpy(cas_delta, lrf1_delta, sizeof(lrf1_delta));
                uint32_t cas_sse = 0U;
                uint32_t cas_rate_q8 = zcvid_lh_mode_rate_q8(&m, codec, skip_ctx, ZCVID_LH_BLK_CASCADE)
                                     + zcvid_lh_lrf_rate_q8(&m, &lrf1);
                for (uint32_t q = 0U; q < 4U; ++q) {
                    uint32_t qx = bx + (q & 1U) * 4U;
                    uint32_t qy = by + (q >> 1) * 4U;
                    int32_t qr[16], qw[16], qrec[16], qbase[16], qcomb[16];
                    int16_t qlv[16];
                    for (uint32_t y = 0U; y < 4U; ++y) {
                        for (uint32_t x = 0U; x < 4U; ++x) {
                            uint32_t si = ((q >> 1) * 4U + y) * 8U + (q & 1U) * 4U + x;
                            qbase[y * 4U + x] = lrf1_delta[si];
                            qr[y * 4U + x] = res[si] - lrf1_delta[si];
                        }
                    }
                    zcvid_lh_tq4_forward(qr, qw);
                    zcvid_lh_tq4_quant(qw, qp, is_inter, qlv);
                    zcvid_lh_tq4_recon(qlv, qp, qrec);
                    for (int k = 0; k < 16; ++k) qcomb[k] = qbase[k] + qrec[k];

                    uint32_t sse0 = zcvid_lh_quad_sse_clamp(src, pred, w, qx, qy, qbase, NULL);
                    uint32_t sse1 = zcvid_lh_quad_sse_clamp(src, pred, w, qx, qy, qcomb, NULL);
                    uint32_t rq8 = zcvid_lh_tq4_rate_q8(&m, qlv);
                    double j0 = (double)sse0 + lambda * ((double)zcvid_lh_cost_q8(&m.tq_cbp[q], 0U) / 256.0);
                    double j1 = (double)sse1 + lambda * ((double)(zcvid_lh_cost_q8(&m.tq_cbp[q], 1U) + rq8) / 256.0);
                    if (rq8 > 0U && j1 < j0) {
                        cas_cbp |= (1U << q);
                        memcpy(cas_lv[q], qlv, sizeof(qlv));
                        cas_sse += sse1;
                        cas_rate_q8 += zcvid_lh_cost_q8(&m.tq_cbp[q], 1U) + rq8;
                        for (uint32_t y = 0U; y < 4U; ++y) {
                            for (uint32_t x = 0U; x < 4U; ++x) {
                                cas_delta[((q >> 1) * 4U + y) * 8U + (q & 1U) * 4U + x] = qcomb[y * 4U + x];
                            }
                        }
                    } else {
                        cas_sse += sse0;
                        cas_rate_q8 += zcvid_lh_cost_q8(&m.tq_cbp[q], 0U);
                    }
                }
                if (cas_cbp != 0U) {
                    double jc = (double)cas_sse + lambda * ((double)cas_rate_q8 / 256.0);
                    if (jc < best_j) {
                        best_j = jc;
                        best_mode = ZCVID_LH_BLK_CASCADE;
                    }
                }
            }

            /* Candidate 5: SPEC_LRF (Rank-1 ternary factorization in 8x8 DCT domain) */
            zcvid_lh_lrf_t spec_lrf;
            int32_t spec_delta[64];
            memset(&spec_lrf, 0, sizeof(spec_lrf));
            memset(spec_delta, 0, sizeof(spec_delta));
            if (codec == ZCVID_LH_HYBRID_FULL) {
                int32_t wmat[64];
                zcvid_lh_dct8_fwd(res, wmat);
                zcvid_lh_fit_lrf(wmat, qp, true, &spec_lrf);
                if (spec_lrf.level != 0) {
                    zcvid_lh_spec_lrf_recon(&spec_lrf, spec_delta);
                    uint32_t sses = zcvid_lh_sb_sse_clamp(src, pred, w, bx, by, spec_delta, NULL);
                    uint32_t rs_q8 = zcvid_lh_mode_rate_q8(&m, codec, skip_ctx, ZCVID_LH_BLK_SPEC_LRF)
                                   + zcvid_lh_lrf_rate_q8(&m, &spec_lrf);
                    double js = (double)sses + lambda * ((double)rs_q8 / 256.0);
                    if (js < best_j) {
                        best_j = js;
                        best_mode = ZCVID_LH_BLK_SPEC_LRF;
                    }
                }
            }

            /* Emit chosen mode & store exact encoder reconstruction */
            zcvid_lh_encode_mode(&rc, &m, codec, skip_ctx, best_mode);
            if (out_stats) out_stats->count[(unsigned)best_mode]++;

            const int32_t *final_delta = NULL;
            if (best_mode == ZCVID_LH_BLK_TQ4) {
                for (uint32_t q = 0U; q < 4U; ++q) {
                    uint32_t act = (tq_cbp >> q) & 1U;
                    zcrudp_rc_enc_bit(&rc, &m.tq_cbp[q], act);
                    if (act) zcvid_lh_tq4_encode_quad(&rc, &m, tq_lv[q]);
                }
                final_delta = tq_delta;
            } else if (best_mode == ZCVID_LH_BLK_LRF1) {
                zcvid_lh_lrf_encode(&rc, &m, &lrf1);
                final_delta = lrf1_delta;
            } else if (best_mode == ZCVID_LH_BLK_LRF2) {
                zcvid_lh_lrf_encode(&rc, &m, &lrf1);
                zcvid_lh_lrf_encode(&rc, &m, &lrf2);
                final_delta = lrf2_delta;
            } else if (best_mode == ZCVID_LH_BLK_CASCADE) {
                zcvid_lh_lrf_encode(&rc, &m, &lrf1);
                for (uint32_t q = 0U; q < 4U; ++q) {
                    uint32_t act = (cas_cbp >> q) & 1U;
                    zcrudp_rc_enc_bit(&rc, &m.tq_cbp[q], act);
                    if (act) zcvid_lh_tq4_encode_quad(&rc, &m, cas_lv[q]);
                }
                final_delta = cas_delta;
            } else if (best_mode == ZCVID_LH_BLK_SPEC_LRF) {
                zcvid_lh_lrf_encode(&rc, &m, &spec_lrf);
                final_delta = spec_delta;
            }

            if (out_rec) {
                uint8_t blk_rec[64];
                zcvid_lh_sb_sse_clamp(src, pred, w, bx, by, final_delta, blk_rec);
                for (uint32_t y = 0U; y < 8U; ++y) {
                    memcpy(out_rec + (size_t)(by + y) * w + bx, blk_rec + y * 8U, 8U);
                }
            }

            left_nz = (best_mode != ZCVID_LH_BLK_SKIP) ? 1U : 0U;
            up_row[sbx] = left_nz;
            if (rc.overflow) {
                free(up_row);
                return 0U;
            }
        }
    }

    free(up_row);
    size_t body = zcrudp_rc_enc_finish(&rc);
    if (rc.overflow) return 0U;
    if (body == 0U) {
        out_bs[12] = 0U;
        body = 1U;
    }
    return 12U + body;
}

static int zcvid_lh_decode_frame(const uint8_t *bs, size_t len, const uint8_t *pred,
                                 uint32_t w, uint32_t h, uint8_t *out_dec) {
    if (!bs || !pred || !out_dec || len < 13U ||
        w == 0U || h == 0U || (w & 7U) != 0U || (h & 7U) != 0U) return -1;
    if (memcmp(bs, "ZCLH", 4) != 0) return -1;
    uint32_t bw_hdr = (uint32_t)bs[4] | ((uint32_t)bs[5] << 8);
    uint32_t bh_hdr = (uint32_t)bs[6] | ((uint32_t)bs[7] << 8);
    if (bw_hdr != w || bh_hdr != h) return -1;
    zcvid_lh_codec_t codec = (zcvid_lh_codec_t)bs[8];
    uint32_t qp = (uint32_t)bs[9];
    if ((unsigned)codec >= ZCVID_LH_CODEC_COUNT || qp > 51U) return -1;

    const uint32_t sbw = w >> 3;
    const uint32_t sbh = h >> 3;
    uint8_t *up_row = (uint8_t *)malloc(sbw);
    if (!up_row) return -1;

    zcvid_lh_step_lut_t step_lut;
    zcvid_lh_step_lut_init(&step_lut, qp);
    int rc = zcvid_lh_decode_slice(codec, bs + 12U, len - 12U, pred, w,
                                   0U, sbh, qp, &step_lut, up_row, out_dec);
    free(up_row);
    return rc;
}

/* =========================================================================
 * Persistent Zero-Alloc Multi-Threaded Slice Engine ("ZCLS" Wire Format)
 * ========================================================================= */

#define ZCVID_LH_MT_MAX_THREADS 24U

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#define ZCVID_LH_CPU_RELAX() _mm_pause()
#elif defined(__x86_64__) || defined(__i386__)
#define ZCVID_LH_CPU_RELAX() __builtin_ia32_pause()
#else
#define ZCVID_LH_CPU_RELAX() ((void)0)
#endif

typedef enum {
    ZCVID_LH_MT_OP_NONE   = 0,
    ZCVID_LH_MT_OP_ENCODE = 1,
    ZCVID_LH_MT_OP_DECODE = 2,
    ZCVID_LH_MT_OP_QUIT   = 3
} zcvid_lh_mt_op_t;

typedef struct zcvid_lh_mt_pool zcvid_lh_mt_pool_t;

typedef struct {
    ZCVID_LH_ALIGNAS64 zcvid_lh_mt_pool_t *pool;
    uint32_t tid;
    uint8_t *up_row;
    size_t   up_cap;
    uint8_t *slice_buf;
    size_t   slice_cap;
    size_t   slice_len;
    int      dec_rc;
    zcvid_lh_stats_t stats;
#if defined(_WIN32)
    HANDLE handle;
#else
    pthread_t handle;
#endif
} zcvid_lh_mt_worker_t;

struct zcvid_lh_mt_pool {
    uint32_t num_threads;
    zcvid_lh_mt_worker_t workers[ZCVID_LH_MT_MAX_THREADS];

    ZCVID_LH_ALIGNAS64 atomic_uint_fast32_t gen;
    ZCVID_LH_ALIGNAS64 atomic_uint_fast32_t active_workers;
    atomic_int op;

#if defined(_WIN32)
    SRWLOCK            work_lock;
    CONDITION_VARIABLE work_cv;
#else
    pthread_mutex_t    work_lock;
    pthread_cond_t     work_cv;
#endif

    /* Per-dispatch job descriptor */
    zcvid_lh_codec_t    codec;
    const uint8_t      *src;
    const uint8_t      *pred;
    uint8_t            *out_rec;
    uint32_t            w;
    uint32_t            h;
    uint32_t            qp;
    bool                is_inter;
    bool                want_stats;
    uint32_t            num_slices;
    zcvid_lh_step_lut_t step_lut;
    uint64_t            lambda_q8;

    /* Per-slice decode pointers */
    const uint8_t      *dec_slice_ptr[ZCVID_LH_MT_MAX_THREADS];
    size_t              dec_slice_len[ZCVID_LH_MT_MAX_THREADS];
};

static inline void zcvid_lh_mt_slice_bounds(uint32_t sbh, uint32_t num_slices, uint32_t tid,
                                            uint32_t *out_y0, uint32_t *out_y1) {
    uint32_t base = sbh / num_slices;
    uint32_t rem = sbh % num_slices;
    uint32_t y0 = tid * base + ((tid < rem) ? tid : rem);
    uint32_t cnt = base + ((tid < rem) ? 1U : 0U);
    *out_y0 = y0;
    *out_y1 = y0 + cnt;
}

static inline void zcvid_lh_mt_run_task(zcvid_lh_mt_pool_t *p, uint32_t tid) {
    int op = atomic_load_explicit(&p->op, memory_order_relaxed);
    zcvid_lh_mt_worker_t *w = &p->workers[tid];
    const uint32_t sbh = p->h >> 3;

    if (op == ZCVID_LH_MT_OP_ENCODE) {
        w->slice_len = 0U;
        if (p->want_stats) memset(&w->stats, 0, sizeof(w->stats));
        if (tid < p->num_slices) {
            uint32_t y0 = 0U, y1 = 0U;
            zcvid_lh_mt_slice_bounds(sbh, p->num_slices, tid, &y0, &y1);
            if (y0 < y1) {
                w->slice_len = zcvid_lh_encode_slice_fast(
                    p->codec, p->src, p->pred, p->w, y0, y1, p->qp, p->is_inter,
                    &p->step_lut, p->lambda_q8, w->up_row,
                    w->slice_buf, w->slice_cap, p->out_rec,
                    p->want_stats ? &w->stats : NULL);
            }
        }
    } else if (op == ZCVID_LH_MT_OP_DECODE) {
        w->dec_rc = 0;
        for (uint32_t s = tid; s < p->num_slices; s += p->num_threads) {
            uint32_t y0 = 0U, y1 = 0U;
            zcvid_lh_mt_slice_bounds(sbh, p->num_slices, s, &y0, &y1);
            if (y0 < y1) {
                int rc = zcvid_lh_decode_slice(
                    p->codec, p->dec_slice_ptr[s], p->dec_slice_len[s],
                    p->pred, p->w, y0, y1, p->qp, &p->step_lut,
                    w->up_row, p->out_rec);
                if (rc != 0) {
                    w->dec_rc = rc;
                    break;
                }
            }
        }
    }
}

#if defined(_WIN32)
static DWORD WINAPI zcvid_lh_mt_worker_entry(LPVOID arg) {
#else
static void *zcvid_lh_mt_worker_entry(void *arg) {
#endif
    zcvid_lh_mt_worker_t *w = (zcvid_lh_mt_worker_t *)arg;
    zcvid_lh_mt_pool_t *p = w->pool;
    uint32_t seen_gen = 0U;

    for (;;) {
        uint32_t cur_gen;
        uint32_t spins = 0U;
        while ((cur_gen = (uint32_t)atomic_load_explicit(&p->gen, memory_order_acquire)) == seen_gen) {
            if (++spins < 512U) {
                ZCVID_LH_CPU_RELAX();
            } else {
#if defined(_WIN32)
                AcquireSRWLockExclusive(&p->work_lock);
                while ((cur_gen = (uint32_t)atomic_load_explicit(&p->gen, memory_order_acquire)) == seen_gen) {
                    SleepConditionVariableSRW(&p->work_cv, &p->work_lock, INFINITE, 0);
                }
                ReleaseSRWLockExclusive(&p->work_lock);
#else
                pthread_mutex_lock(&p->work_lock);
                while ((cur_gen = (uint32_t)atomic_load_explicit(&p->gen, memory_order_acquire)) == seen_gen) {
                    pthread_cond_wait(&p->work_cv, &p->work_lock);
                }
                pthread_mutex_unlock(&p->work_lock);
#endif
                break;
            }
        }
        seen_gen = cur_gen;
        if (atomic_load_explicit(&p->op, memory_order_relaxed) == ZCVID_LH_MT_OP_QUIT) break;
        zcvid_lh_mt_run_task(p, w->tid);
        atomic_fetch_sub_explicit(&p->active_workers, 1U, memory_order_release);
    }
#if defined(_WIN32)
    return 0;
#else
    return NULL;
#endif
}

static inline void zcvid_lh_mt_dispatch(zcvid_lh_mt_pool_t *p, zcvid_lh_mt_op_t op) {
    uint32_t n = p->num_threads;
    atomic_store_explicit(&p->op, (int)op, memory_order_relaxed);
    if (n > 1U) {
        atomic_store_explicit(&p->active_workers, n - 1U, memory_order_relaxed);
        atomic_fetch_add_explicit(&p->gen, 1U, memory_order_release);
#if defined(_WIN32)
        AcquireSRWLockExclusive(&p->work_lock);
        ReleaseSRWLockExclusive(&p->work_lock);
        WakeAllConditionVariable(&p->work_cv);
#else
        pthread_mutex_lock(&p->work_lock);
        pthread_cond_broadcast(&p->work_cv);
        pthread_mutex_unlock(&p->work_lock);
#endif
    }
    if (op != ZCVID_LH_MT_OP_QUIT) {
        zcvid_lh_mt_run_task(p, 0U);
        while (n > 1U && atomic_load_explicit(&p->active_workers, memory_order_acquire) != 0U) {
            ZCVID_LH_CPU_RELAX();
        }
    }
}

static bool zcvid_lh_mt_ensure_scratch(zcvid_lh_mt_pool_t *p, uint32_t w, uint32_t h,
                                       bool need_slice_buf) {
    if (p->num_threads == 0U) return false;
    const uint32_t sbw = w >> 3;
    const uint32_t sbh = h >> 3;
    const uint32_t rows_per = (sbh + p->num_threads - 1U) / p->num_threads + 1U;
    const size_t need_slice = need_slice_buf ? ((size_t)rows_per * (size_t)sbw * 256U + 4096U) : 0U;
    for (uint32_t t = 0U; t < p->num_threads; ++t) {
        zcvid_lh_mt_worker_t *wk = &p->workers[t];
        if (wk->up_cap < sbw) {
            uint8_t *nb = (uint8_t *)realloc(wk->up_row, sbw);
            if (!nb) return false;
            wk->up_row = nb;
            wk->up_cap = sbw;
        }
        if (need_slice_buf && wk->slice_cap < need_slice) {
            uint8_t *nb = (uint8_t *)realloc(wk->slice_buf, need_slice);
            if (!nb) return false;
            wk->slice_buf = nb;
            wk->slice_cap = need_slice;
        }
    }
    return true;
}

static bool zcvid_lh_mt_pool_init(zcvid_lh_mt_pool_t *p, uint32_t num_threads,
                                  uint32_t max_w, uint32_t max_h) {
    if (!p) return false;
    memset(p, 0, sizeof(*p));
    if (num_threads < 1U) num_threads = 1U;
    if (num_threads > ZCVID_LH_MT_MAX_THREADS) num_threads = ZCVID_LH_MT_MAX_THREADS;
    p->num_threads = num_threads;
    atomic_init(&p->gen, 0U);
    atomic_init(&p->active_workers, 0U);
    atomic_init(&p->op, (int)ZCVID_LH_MT_OP_NONE);

#if defined(_WIN32)
    InitializeSRWLock(&p->work_lock);
    InitializeConditionVariable(&p->work_cv);
#else
    pthread_mutex_init(&p->work_lock, NULL);
    pthread_cond_init(&p->work_cv, NULL);
#endif

    for (uint32_t t = 0U; t < num_threads; ++t) {
        p->workers[t].pool = p;
        p->workers[t].tid = t;
    }
    if (max_w >= 8U && max_h >= 8U) {
        if (!zcvid_lh_mt_ensure_scratch(p, max_w, max_h, true)) {
            for (uint32_t t = 0U; t < num_threads; ++t) {
                free(p->workers[t].up_row);
                free(p->workers[t].slice_buf);
                p->workers[t].up_row = NULL;
                p->workers[t].slice_buf = NULL;
            }
#if !defined(_WIN32)
            pthread_mutex_destroy(&p->work_lock);
            pthread_cond_destroy(&p->work_cv);
#endif
            p->num_threads = 0U;
            return false;
        }
    }

    for (uint32_t t = 1U; t < num_threads; ++t) {
#if defined(_WIN32)
        p->workers[t].handle = CreateThread(NULL, 0, zcvid_lh_mt_worker_entry, &p->workers[t], 0, NULL);
        if (!p->workers[t].handle) {
            p->num_threads = t;
            break;
        }
#else
        if (pthread_create(&p->workers[t].handle, NULL, zcvid_lh_mt_worker_entry, &p->workers[t]) != 0) {
            p->num_threads = t;
            break;
        }
#endif
    }
    return true;
}

static void zcvid_lh_mt_pool_destroy(zcvid_lh_mt_pool_t *p) {
    if (!p || p->num_threads == 0U) return;
    if (p->num_threads > 1U) {
        zcvid_lh_mt_dispatch(p, ZCVID_LH_MT_OP_QUIT);
        for (uint32_t t = 1U; t < p->num_threads; ++t) {
#if defined(_WIN32)
            if (p->workers[t].handle) {
                WaitForSingleObject(p->workers[t].handle, INFINITE);
                CloseHandle(p->workers[t].handle);
                p->workers[t].handle = NULL;
            }
#else
            pthread_join(p->workers[t].handle, NULL);
#endif
        }
    }
#if !defined(_WIN32)
    pthread_mutex_destroy(&p->work_lock);
    pthread_cond_destroy(&p->work_cv);
#endif
    for (uint32_t t = 0U; t < ZCVID_LH_MT_MAX_THREADS; ++t) {
        free(p->workers[t].up_row);
        free(p->workers[t].slice_buf);
        p->workers[t].up_row = NULL;
        p->workers[t].slice_buf = NULL;
    }
    p->num_threads = 0U;
}

static size_t zcvid_lh_mt_encode_frame(zcvid_lh_mt_pool_t *p, zcvid_lh_codec_t codec,
                                       const uint8_t *src, const uint8_t *pred,
                                       uint32_t w, uint32_t h, uint32_t qp, bool is_inter,
                                       uint8_t *out_bs, size_t out_cap, uint8_t *out_rec,
                                       zcvid_lh_stats_t *out_stats) {
    if (!p || p->num_threads == 0U || !src || !pred || !out_bs ||
        w == 0U || h == 0U || w > 65535U || h > 65535U ||
        (w & 7U) != 0U || (h & 7U) != 0U || qp > 51U) return 0U;
    if (codec != ZCVID_LH_HYBRID_FAST && codec != ZCVID_LH_HYBRID_ESR && codec != ZCVID_LH_TQ4) return 0U;
    if (out_stats) memset(out_stats, 0, sizeof(*out_stats));

    const uint32_t sbh = h >> 3;
    /* Adaptive slice count: keep >= 8 SB rows (64 px) per slice to avoid range-coder
     * warmup bloat on small frames while keeping full 16-way parallelism at 1080p (135/8=16). */
    uint32_t max_useful = (sbh >= 8U) ? (sbh / 8U) : 1U;
    uint32_t nslices = p->num_threads;
    if (nslices > max_useful) nslices = max_useful;
    if (nslices < 1U) nslices = 1U;

    const size_t hdr_sz = 12U + (size_t)nslices * 4U;
    if (out_cap < hdr_sz + nslices) return 0U;
    if (!zcvid_lh_mt_ensure_scratch(p, w, h, true)) return 0U;

    p->codec = codec;
    p->src = src;
    p->pred = pred;
    p->out_rec = out_rec;
    p->w = w;
    p->h = h;
    p->qp = qp;
    p->is_inter = is_inter;
    p->want_stats = (out_stats != NULL);
    p->num_slices = nslices;
    zcvid_lh_step_lut_init(&p->step_lut, qp);
    const double lambda = 0.85 * pow(2.0, ((double)qp - 12.0) / 3.0);
    p->lambda_q8 = (uint64_t)llround(lambda * 256.0);

    zcvid_lh_mt_dispatch(p, ZCVID_LH_MT_OP_ENCODE);

    /* Assemble header: magic 'ZCLS', u16 w, u16 h, u8 codec, u8 qp, u8 inter, u8 nslices */
    out_bs[0] = 'Z'; out_bs[1] = 'C'; out_bs[2] = 'L'; out_bs[3] = 'S';
    out_bs[4] = (uint8_t)(w & 0xFFU); out_bs[5] = (uint8_t)(w >> 8);
    out_bs[6] = (uint8_t)(h & 0xFFU); out_bs[7] = (uint8_t)(h >> 8);
    out_bs[8] = (uint8_t)codec;
    out_bs[9] = (uint8_t)qp;
    out_bs[10] = is_inter ? 1U : 0U;
    out_bs[11] = (uint8_t)nslices;

    size_t pos = hdr_sz;
    for (uint32_t s = 0U; s < nslices; ++s) {
        size_t slen = p->workers[s].slice_len;
        if (slen == 0U || slen > 0xFFFFFFFFU || slen > out_cap - pos) return 0U;
        uint32_t sl32 = (uint32_t)slen;
        size_t toff = 12U + (size_t)s * 4U;
        out_bs[toff + 0U] = (uint8_t)(sl32 & 0xFFU);
        out_bs[toff + 1U] = (uint8_t)((sl32 >> 8) & 0xFFU);
        out_bs[toff + 2U] = (uint8_t)((sl32 >> 16) & 0xFFU);
        out_bs[toff + 3U] = (uint8_t)((sl32 >> 24) & 0xFFU);
        memcpy(out_bs + pos, p->workers[s].slice_buf, slen);
        pos += slen;
        if (out_stats) {
            for (int k = 0; k < 7; ++k) out_stats->count[k] += p->workers[s].stats.count[k];
        }
    }
    return pos;
}

static int zcvid_lh_mt_decode_frame(zcvid_lh_mt_pool_t *p, const uint8_t *bs, size_t len,
                                    const uint8_t *pred, uint32_t w, uint32_t h,
                                    uint8_t *out_dec) {
    if (!p || p->num_threads == 0U || !bs || !pred || !out_dec || len < 16U ||
        w == 0U || h == 0U || (w & 7U) != 0U || (h & 7U) != 0U) return -1;
    if (memcmp(bs, "ZCLS", 4) != 0) return -1;
    uint32_t bw_hdr = (uint32_t)bs[4] | ((uint32_t)bs[5] << 8);
    uint32_t bh_hdr = (uint32_t)bs[6] | ((uint32_t)bs[7] << 8);
    if (bw_hdr != w || bh_hdr != h) return -1;
    zcvid_lh_codec_t codec = (zcvid_lh_codec_t)bs[8];
    uint32_t qp = (uint32_t)bs[9];
    uint32_t nslices = (uint32_t)bs[11];
    const uint32_t sbh = h >> 3;
    if ((unsigned)codec >= ZCVID_LH_CODEC_COUNT || qp > 51U ||
        nslices < 1U || nslices > ZCVID_LH_MT_MAX_THREADS || nslices > sbh) return -1;

    const size_t hdr_sz = 12U + (size_t)nslices * 4U;
    if (len < hdr_sz) return -1;

    size_t pos = hdr_sz;
    for (uint32_t s = 0U; s < nslices; ++s) {
        size_t toff = 12U + (size_t)s * 4U;
        uint32_t slen = (uint32_t)bs[toff + 0U]
                      | ((uint32_t)bs[toff + 1U] << 8)
                      | ((uint32_t)bs[toff + 2U] << 16)
                      | ((uint32_t)bs[toff + 3U] << 24);
        if (slen == 0U || (size_t)slen > len - pos) return -1;
        p->dec_slice_ptr[s] = bs + pos;
        p->dec_slice_len[s] = (size_t)slen;
        pos += (size_t)slen;
    }
    if (pos != len) return -1;
    if (!zcvid_lh_mt_ensure_scratch(p, w, h, false)) return -1;

    p->codec = codec;
    p->pred = pred;
    p->out_rec = out_dec;
    p->w = w;
    p->h = h;
    p->qp = qp;
    p->num_slices = nslices;
    zcvid_lh_step_lut_init(&p->step_lut, qp);

    /* Always decode in parallel across all available pool threads (even if nslices > num_threads). */
    zcvid_lh_mt_dispatch(p, ZCVID_LH_MT_OP_DECODE);
    uint32_t active = (nslices < p->num_threads) ? nslices : p->num_threads;
    for (uint32_t t = 0U; t < active; ++t) {
        if (p->workers[t].dec_rc != 0) return -1;
    }
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* ZCVID_LATENT_HYBRID_H */
