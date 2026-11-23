/**
 * @file zcvid_codec.h
 * @brief High-throughput, malloc-free C11 Zcvid Video Codec (NV12 & RGBA)
 *
 * Implements:
 *  - Compact MTU-framed slice packets with in-band RLE zero-block skipping
 *    (tokens 0..59048 = 10 trits; tokens 59049..65535 = skip 1..6487 zero blocks)
 *  - Dual pixel format support: GRAY8, NV12 (YUV420), and RGBA (RGB channels)
 *  - Optional Jayant Adaptive Delta Modulation (ADM) with 0-bit wire overhead
 *  - Optional per-packet entropy coding (cfg.entropy): same pictures, fewer
 *    bytes, each packet still decodable on its own (see zcvid_entropy.h)
 *  - AVX2 SIMD fast-path (#ifdef __AVX2__) with strict C11 scalar fallback
 */

#ifndef ZCRUDP_ZCVID_CODEC_H
#define ZCRUDP_ZCVID_CODEC_H

#include "protocol_zcvid.h"
#include "zcvid_entropy.h"
#include "zcvid_rans.h"
#include <assert.h>
#include <string.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Hot helpers called with constant geometry flags: inlining lets the compiler
 * drop the 2x branches from the 1x path. */
#if defined(__GNUC__) || defined(__clang__)
#define ZCRUDP_ZCVID_INLINE static inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define ZCRUDP_ZCVID_INLINE static __forceinline
#else
#define ZCRUDP_ZCVID_INLINE static inline
#endif

/* =========================================================================
 * Codec Constants & Pixel Formats
 * ========================================================================= */

#define ZCRUDP_ZCVID_PKT_MAGIC   0x5456U  /**< 'TV' magic for zcvid video packet */
#define ZCRUDP_ZCVID_PKT_HDR_SZ  12U      /**< Wire size of zcrudp_zcvid_pkt_hdr_t */
#define ZCRUDP_ZCVID_MAX_SKIP    (65536U - ZCRUDP_TRIT_MAX_10P) /**< 6487 blocks */

/** Packet sub-types packed into upper nibble of hdr.pixfmt (lower nibble = pixfmt) */
#define ZCRUDP_ZCVID_PKT_DELTA    0U  /**< Standard 10-trit delta + RLE skip */
#define ZCRUDP_ZCVID_PKT_INTRA    1U  /**< Raw 8-bit anchor slice (zero-drift Intra-Refresh) */
#define ZCRUDP_ZCVID_PKT_DELTA_2X 2U  /**< 2x horizontal subsampled 10-trit delta (0.8 bpp) */
#define ZCRUDP_ZCVID_PKT_TILE_MV  3U  /**< Per-tile (dx, dy) int8 motion vector pairs */
#define ZCRUDP_ZCVID_PKT_DELTA_EC    4U /**< Range-coded DELTA; num_tokens = coded blocks */
#define ZCRUDP_ZCVID_PKT_DELTA_2X_EC 5U /**< Range-coded DELTA_2X; num_tokens = coded blocks */
#define ZCRUDP_ZCVID_PKT_INTRA_EC    6U /**< Range-coded lossless anchor; num_tokens = samples */
#define ZCRUDP_ZCVID_PKT_INTRA_Q_EC  7U /**< Quantized range-coded spatial anchor; num_tokens = samples */
#define ZCRUDP_ZCVID_PKT_DELTA_RANS  8U /**< 4-way interleaved static rANS DELTA */
#define ZCRUDP_ZCVID_PKT_DELTA_RV2   9U /**< Realloc v2 (dual-res RLE + W1/W2 fused tokens) */
#define ZCRUDP_ZCVID_PKT_TILE_MV_DPCM 10U /**< DPCM+RLE compressed per-tile motion vectors */

/** Bits of zcrudp_zcvid_config_t.entropy (0 = fixed-size tokens and raw anchors) */
#define ZCRUDP_ZCVID_EC_DELTA     0x01U /**< Range-code DELTA / DELTA_2X packets */
#define ZCRUDP_ZCVID_EC_INTRA     0x02U /**< Range-code INTRA packets (lossless) */
#define ZCRUDP_ZCVID_EC_ALL       0x03U
#define ZCRUDP_ZCVID_EC_RANS      0x04U /**< 4-way interleaved static rANS for DELTA */
#define ZCRUDP_ZCVID_EC_REALLOC_V2 0x08U /**< Realloc v2 token encoding (W1/W2 fusion) */

#define ZCRUDP_ZCVID_HDR_FMT(h)   ((uint8_t)((h)->pixfmt & 0x0FU))
#define ZCRUDP_ZCVID_HDR_TYPE(h)  ((uint8_t)(((h)->pixfmt >> 4) & 0x0FU))
#define ZCRUDP_ZCVID_MAKE_PF(fmt, ptype) ((uint8_t)(((uint8_t)(fmt) & 0x0FU) | (((uint8_t)(ptype) & 0x0FU) << 4)))

typedef enum {
    ZCRUDP_ZCVID_FMT_GRAY8 = 0, /**< 8-bit Luma only (W * H bytes) */
    ZCRUDP_ZCVID_FMT_NV12  = 1, /**< Y plane (W * H) + interleaved UV plane (W * H / 2) */
    ZCRUDP_ZCVID_FMT_RGBA  = 2  /**< 32-bit RGBA (4 * W * H bytes, encodes R, G, B) */
} zcrudp_zcvid_pixfmt_t;

typedef struct {
    uint16_t width;
    uint16_t height;
    zcrudp_zcvid_pixfmt_t pixfmt;
    uint8_t  delta_min;   /**< Minimum step (or fixed step if delta_min == delta_max) */
    uint8_t  delta_max;   /**< Maximum step for Jayant ADM */
    uint8_t  threshold;   /**< Quantization deadzone threshold */
    uint8_t  gamma_q8;    /**< Leaky decay factor in Q0.8 (e.g. 248 ~ 0.969) */
    uint8_t  neutral_y;   /**< Neutral reference (default 128) */
    uint16_t mtu_payload; /**< Max UDP payload bytes per packet (e.g. 1380) */
    uint8_t  entropy;     /**< ZCRUDP_ZCVID_EC_* bits; encoder side only, the
                               decoder reads the packet type from the header */
    uint32_t ec_max_blocks; /**< CPU cap for range-coded deltas: after this many
                               coded blocks in one plane-encode call, the rest
                               of the call is sent as tokens. 0 = no cap. A
                               block costs 60 to 75 ns to code and nearly as much to
                               decode, so the cap bounds both ends. */
    uint8_t  intra_qstep; /**< Quantization step for INTRA_EC anchors (0 or 1 = lossless
                               INTRA_EC; >= 2 = near-lossless INTRA_Q_EC when
                               ZCRUDP_ZCVID_EC_INTRA is enabled). */
    uint8_t  jnd_boost;   /**< Closed-loop edge-preserving JND deadzone boost (0 = off). */
    uint8_t  no_decay;    /**< Non-zero: true zero-leak mode (no decay toward neutral_y). */
} zcrudp_zcvid_config_t;

typedef struct {
    uint16_t magic;       /**< ZCRUDP_ZCVID_PKT_MAGIC */
    uint8_t  pixfmt;      /**< zcrudp_zcvid_pixfmt_t */
    uint8_t  plane;       /**< Plane index (0=Y/R, 1=UV/G, 2=B) */
    uint16_t frame_id;    /**< Wrapping frame sequence ID */
    uint32_t start_block; /**< Starting 10-pixel block index in plane */
    uint16_t num_tokens;  /**< Number of uint16_t tokens following header */
} zcrudp_zcvid_pkt_hdr_t;

/** Callback invoked by encoder for each MTU-bounded packet produced. */
typedef void (*zcrudp_zcvid_pkt_cb)(const uint8_t *pkt_data,
                                      size_t pkt_len,
                                      void *user_ctx);

/* =========================================================================
 * Packet Header Wire Serialization (Little-Endian)
 * ========================================================================= */

static inline void zcrudp_zcvid_pkt_hdr_write_le(uint8_t dst[ZCRUDP_ZCVID_PKT_HDR_SZ],
                                                   const zcrudp_zcvid_pkt_hdr_t *h) {
    dst[0]  = (uint8_t)(h->magic & 0xFFU);
    dst[1]  = (uint8_t)(h->magic >> 8);
    dst[2]  = h->pixfmt;
    dst[3]  = h->plane;
    dst[4]  = (uint8_t)(h->frame_id & 0xFFU);
    dst[5]  = (uint8_t)(h->frame_id >> 8);
    dst[6]  = (uint8_t)(h->start_block & 0xFFU);
    dst[7]  = (uint8_t)((h->start_block >> 8) & 0xFFU);
    dst[8]  = (uint8_t)((h->start_block >> 16) & 0xFFU);
    dst[9]  = (uint8_t)((h->start_block >> 24) & 0xFFU);
    dst[10] = (uint8_t)(h->num_tokens & 0xFFU);
    dst[11] = (uint8_t)(h->num_tokens >> 8);
}

static inline bool zcrudp_zcvid_pkt_hdr_read_le(const uint8_t src[ZCRUDP_ZCVID_PKT_HDR_SZ],
                                                  zcrudp_zcvid_pkt_hdr_t *h) {
    h->magic = (uint16_t)((uint16_t)src[0] | ((uint16_t)src[1] << 8));
    if (h->magic != ZCRUDP_ZCVID_PKT_MAGIC) {
        return false;
    }
    h->pixfmt = src[2];
    h->plane = src[3];
    h->frame_id = (uint16_t)((uint16_t)src[4] | ((uint16_t)src[5] << 8));
    h->start_block = (uint32_t)src[6] |
                     ((uint32_t)src[7] << 8) |
                     ((uint32_t)src[8] << 16) |
                     ((uint32_t)src[9] << 24);
    h->num_tokens = (uint16_t)((uint16_t)src[10] | ((uint16_t)src[11] << 8));
    return true;
}

static inline void zcrudp_zcvid_write_u16_le(uint8_t *dst, uint16_t val) {
    dst[0] = (uint8_t)(val & 0xFFU);
    dst[1] = (uint8_t)(val >> 8);
}

static inline uint16_t zcrudp_zcvid_read_u16_le(const uint8_t *src) {
    return (uint16_t)((uint16_t)src[0] | ((uint16_t)src[1] << 8));
}

/* =========================================================================
 * Jayant ADM Step Update Helper (0-bit Wire Overhead)
 * ========================================================================= */

static inline uint8_t zcrudp_adm_update_delta(uint8_t cur_delta,
                                              zcrudp_trit_t trit,
                                              uint8_t delta_min,
                                              uint8_t delta_max) {
    if (delta_min >= delta_max) {
        return delta_min;
    }
    if (trit == 0) {
        uint16_t half = (uint16_t)cur_delta >> 1;
        return (half < delta_min) ? delta_min : (uint8_t)half;
    }
    uint16_t dbl = (uint16_t)cur_delta << 1;
    return (dbl > delta_max) ? delta_max : (uint8_t)dbl;
}

/* =========================================================================
 * Fast Plane Decay (AVX2 Accelerated + C11 Scalar Fallback)
 * ========================================================================= */

/**
 * @brief Decays an entire plane toward neutral_y by gamma_q8.
 *        Called once per frame on the receiver before applying received packets.
 */
static inline void zcrudp_zcvid_decay_plane(uint8_t * ZCRUDP_RESTRICT canvas,
                                              size_t len,
                                              uint8_t gamma_q8,
                                              uint8_t neutral_y) {
    size_t i = 0;

#ifdef __AVX2__
    if (len >= 16) {
        /* |y - neutral| * gamma fits an unsigned 16-bit lane for any neutral_y
         * (255 * 255 = 65025); the sign is put back after the shift, which is
         * the truncation toward zero of the scalar path. */
        const __m256i v_neutral = _mm256_set1_epi16((int16_t)neutral_y);
        const __m256i v_gamma   = _mm256_set1_epi16((int16_t)gamma_q8);

        for (; i + 32 <= len; i += 32) {
            __m256i y0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(canvas + i)));
            __m256i y1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(canvas + i + 16)));
            __m256i d0 = _mm256_sub_epi16(y0, v_neutral);
            __m256i d1 = _mm256_sub_epi16(y1, v_neutral);
            __m256i q0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_abs_epi16(d0), v_gamma), 8);
            __m256i q1 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_abs_epi16(d1), v_gamma), 8);
            __m256i r0 = _mm256_add_epi16(v_neutral, _mm256_sign_epi16(q0, d0));
            __m256i r1 = _mm256_add_epi16(v_neutral, _mm256_sign_epi16(q1, d1));
            __m256i pk = _mm256_packus_epi16(r0, r1);
            pk = _mm256_permute4x64_epi64(pk, 0xD8);
            _mm256_storeu_si256((__m256i *)(canvas + i), pk);
        }

        for (; i + 16 <= len; i += 16) {
            __m256i y16 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(canvas + i)));
            __m256i d = _mm256_sub_epi16(y16, v_neutral);
            __m256i q = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_abs_epi16(d), v_gamma), 8);
            __m256i r = _mm256_add_epi16(v_neutral, _mm256_sign_epi16(q, d));
            __m256i p = _mm256_packus_epi16(r, r);
            p = _mm256_permute4x64_epi64(p, 0xD8);
            _mm_storeu_si128((__m128i *)(canvas + i), _mm256_castsi256_si128(p));
        }
    }
#endif

    for (; i < len; i++) {
        canvas[i] = zcrudp_leaky_integrator_step(canvas[i], 0, 0, gamma_q8, neutral_y);
    }
}

/** @brief Config-aware plane decay (no-op when cfg->no_decay is non-zero). */
static inline void zcrudp_zcvid_decay_plane_cfg(uint8_t * ZCRUDP_RESTRICT canvas,
                                                  size_t len,
                                                  const zcrudp_zcvid_config_t *cfg) {
    if (cfg->no_decay) {
        return;
    }
    zcrudp_zcvid_decay_plane(canvas, len, cfg->gamma_q8, cfg->neutral_y);
}

/* =========================================================================
 * DELTA_2X Sample Geometry
 * ========================================================================= */

/**
 * @brief True when a plane stores two interleaved components (NV12 plane 1 is
 *        U0 V0 U1 V1 ...). In DELTA_2X mode a trit then drives two samples of
 *        the SAME component (U0+U1, V0+V1), never a U/V pair.
 */
static inline bool zcrudp_zcvid_2x_interleaved(const zcrudp_zcvid_config_t *cfg,
                                                 uint8_t plane_id) {
    return cfg->pixfmt == ZCRUDP_ZCVID_FMT_NV12 && plane_id == 1U;
}

/** Bytes covered by one 10-trit DELTA_2X block (10 trits x 2 samples). */
#define ZCRUDP_ZCVID_2X_SPAN 20U

/**
 * Index of the sample coded by trit j of a DELTA_2X block starting at base.
 * Planar: 0 2 4 ... 18. Interleaved: U0 V0 U2 V2 ... (0 1 4 5 8 9 ...).
 */
static inline size_t zcrudp_zcvid_2x_idx(size_t base, size_t j, bool interleaved) {
    return interleaved ? (base + 4U * (j >> 1) + (j & 1U)) : (base + 2U * j);
}

/** Distance from a coded sample to the sample that follows it in DELTA_2X. */
static inline size_t zcrudp_zcvid_2x_pair(bool interleaved) {
    return interleaved ? 2U : 1U;
}

/* =========================================================================
 * Encoder Sample Quantiser
 * ========================================================================= */

/**
 * @brief Picks the trit of one sample and advances the closed-loop reference.
 *
 * A step is only taken when it brings the reference closer to the source,
 * that is when the error exceeds both the threshold and half the step. With
 * ADM the step doubles on every non-zero trit, so stepping regardless of its
 * size makes the reference swing around the target with a growing step.
 * Declining the step codes a 0, which halves it for the next frame.
 *
 * This is an encoder decision only: the decoder applies whatever trits it
 * receives, so the bitstream and the decoder are unchanged.
 *
 * @param shrink set to true when the sample is off target but its step is too
 *               coarse and can still be halved: the block then has to be sent
 *               even if all its trits are 0. Never cleared here.
 */
ZCRUDP_ZCVID_INLINE zcrudp_trit_t zcrudp_zcvid_encode_sample(uint8_t cur,
                                                                 uint8_t *ref,
                                                                 uint8_t step,
                                                                 bool adm,
                                                                 const zcrudp_zcvid_config_t *cfg,
                                                                 bool *shrink) {
    uint16_t gamma = cfg->no_decay ? 256U : (uint16_t)cfg->gamma_q8;
    uint8_t dec = zcrudp_leaky_integrator_step_ex(*ref, 0, 0, gamma, cfg->neutral_y);
    int32_t err = (int32_t)cur - (int32_t)dec;
    int32_t mag = (err < 0) ? -err : err;
    zcrudp_trit_t t = 0;
    if (mag > (int32_t)cfg->threshold) {
        if (2 * mag > (int32_t)step) {
            t = (err > 0) ? (zcrudp_trit_t)1 : (zcrudp_trit_t)-1;
        } else if (adm && step > cfg->delta_min) {
            *shrink = true;
        }
    }
    int32_t nv = (int32_t)dec + ((int32_t)t * (int32_t)step);
    *ref = (uint8_t)(nv < 0 ? 0 : (nv > 255 ? 255 : nv));
    return t;
}

#ifdef __AVX2__
#define ZCRUDP_ALIGNAS32 _Alignas(32)

/** Writes the low 10 bytes of @p v to @p dst, and nothing else. */
static inline void zcrudp_zcvid_store10(uint8_t *dst, __m128i v) {
    uint16_t tail = (uint16_t)_mm_extract_epi16(v, 4);
    _mm_storel_epi64((__m128i *)dst, v);
    memcpy(dst + 8, &tail, 2);
}

/**
 * @brief Quantises one full 10-sample block: trits, closed-loop reference and
 *        ADM steps, bit for bit what the scalar path computes.
 *
 * @param src,ref,delta_buf block start; 16 bytes must be readable, 10 are written
 * @param trits receives the 10 trits (16 bytes are stored)
 * @return true when the block has to be sent (a non-zero trit, or with ADM a
 *         step that must be halved)
 */
static inline bool zcrudp_zcvid_encode_block_avx2(const uint8_t *src,
                                                    uint8_t *ref,
                                                    uint8_t *delta_buf,
                                                    const zcrudp_zcvid_config_t *cfg,
                                                    int8_t trits[16]) {
    const __m128i lane10 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0);
    const __m128i zero = _mm_setzero_si128();
    const __m128i v_thr = _mm_set1_epi8((char)cfg->threshold);
    const __m128i v_dmin = _mm_set1_epi8((char)cfg->delta_min);

    const __m128i s = _mm_loadu_si128((const __m128i *)src);
    const __m128i r = _mm_loadu_si128((const __m128i *)ref);
    const __m128i d = (delta_buf != NULL) ? _mm_loadu_si128((const __m128i *)delta_buf) : v_dmin;

    /* Post-decay reference (PDQ): quantize against decayed r_dec */
    __m256i dec16;
    __m128i r_dec;
    if (cfg->no_decay) {
        dec16 = _mm256_cvtepu8_epi16(r);
        r_dec = r;
    } else {
        const __m256i v_neutral = _mm256_set1_epi16((int16_t)cfg->neutral_y);
        const __m256i v_gamma = _mm256_set1_epi16((int16_t)cfg->gamma_q8);
        const __m256i df = _mm256_sub_epi16(_mm256_cvtepu8_epi16(r), v_neutral);
        const __m256i q = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_abs_epi16(df), v_gamma), 8);
        dec16 = _mm256_add_epi16(v_neutral, _mm256_sign_epi16(q, df));
        const __m256i pk_dec = _mm256_permute4x64_epi64(_mm256_packus_epi16(dec16, dec16), 0xD8);
        r_dec = _mm256_castsi256_si128(pk_dec);
    }

    /* |src - r_dec| and its direction, unsigned 8-bit */
    const __m128i up = _mm_subs_epu8(s, r_dec);
    const __m128i dn = _mm_subs_epu8(r_dec, s);
    const __m128i mag = _mm_or_si128(up, dn);

    /* step taken when mag > threshold and 2 * mag > step, i.e. mag > step >> 1 */
    const __m128i half = _mm_and_si128(_mm_srli_epi16(d, 1), _mm_set1_epi8(0x7F));
    const __m128i lim = _mm_max_epu8(v_thr, half);
    const __m128i nz = _mm_andnot_si128(_mm_cmpeq_epi8(_mm_subs_epu8(mag, lim), zero), lane10);
    const __m128i neg = _mm_andnot_si128(_mm_cmpeq_epi8(dn, zero), nz); /* r_dec above src */

    /* trit = +1, -1 or 0 */
    const __m128i t8 = _mm_or_si128(_mm_and_si128(nz, _mm_set1_epi8(1)), neg);
    _mm_storeu_si128((__m128i *)trits, t8);

    bool coded = !_mm_testz_si128(nz, nz);
    if (delta_buf != NULL) {
        /* off target, step too coarse and still above its floor: ask for a shrink */
        const __m128i over = _mm_andnot_si128(_mm_cmpeq_epi8(_mm_subs_epu8(mag, v_thr), zero), lane10);
        const __m128i can_shrink = _mm_andnot_si128(_mm_cmpeq_epi8(_mm_subs_epu8(d, v_dmin), zero), over);
        const __m128i shrink = _mm_andnot_si128(nz, can_shrink);
        coded = coded || !_mm_testz_si128(shrink, shrink);
    }

    /* closed-loop update: clamp(dec16 + trit * step, 0, 255) */
    const __m256i move = _mm256_sign_epi16(_mm256_cvtepu8_epi16(d), _mm256_cvtepi8_epi16(t8));
    const __m256i nv = _mm256_add_epi16(dec16, move);
    const __m256i pk = _mm256_permute4x64_epi64(_mm256_packus_epi16(nv, nv), 0xD8);
    zcrudp_zcvid_store10(ref, _mm256_castsi256_si128(pk));

    if (coded && delta_buf != NULL) {
        __m128i nd;
        if (cfg->delta_min >= cfg->delta_max) {
            nd = v_dmin;
        } else {
            const __m128i dbl = _mm_min_epu8(_mm_adds_epu8(d, d), _mm_set1_epi8((char)cfg->delta_max));
            const __m128i hlv = _mm_max_epu8(half, v_dmin);
            nd = _mm_blendv_epi8(hlv, dbl, nz);
        }
        zcrudp_zcvid_store10(delta_buf, nd);
    }
    return coded;
}

/**
 * @brief Unpacks a 10-trit token into 10 signed bytes (16 are stored), same
 *        result as zcrudp_unpack_10trits(). The five base-3 digits of each
 *        half come from one multiply-high by 1/3, 1/9, 1/27, 1/81 (exact for
 *        values below 243) instead of a chain of divisions.
 */
static inline void zcrudp_zcvid_unpack10_avx2(uint16_t tok, int8_t trits[16]) {
    const uint32_t w  = tok;
    const uint32_t hi = (w * 69043U) >> 24;
    const uint32_t lo = w - hi * ZCRUDP_TRIT_MAX_5P;
    /* Place lo in lower 128-bit lanes (0..4) and hi in upper 128-bit lanes (8..12)
     * so _mm256_slli_si256(q, 2) shifts both halves in-register without stack spill. */
    const __m256i v = _mm256_setr_epi16((short)lo, (short)lo, (short)lo, (short)lo, (short)lo, 0, 0, 0,
                                        (short)hi, (short)hi, (short)hi, (short)hi, (short)hi, 0, 0, 0);
    const __m256i recip = _mm256_setr_epi16(21846, 7282, 2428, 810, 0, 0, 0, 0,
                                            21846, 7282, 2428, 810, 0, 0, 0, 0);
    const __m256i q = _mm256_mulhi_epu16(v, recip);
    const __m256i sel = _mm256_setr_epi16(-1, 0, 0, 0, 0, 0, 0, 0,
                                          -1, 0, 0, 0, 0, 0, 0, 0);
    const __m256i qprev = _mm256_slli_si256(q, 2);
    const __m256i x = _mm256_or_si256(_mm256_and_si256(sel, v), qprev);
    const __m256i digit = _mm256_sub_epi16(x, _mm256_add_epi16(q, _mm256_add_epi16(q, q)));
    const __m256i t16 = _mm256_sub_epi16(digit, _mm256_set1_epi16(1));
    const __m256i pk = _mm256_permute4x64_epi64(_mm256_packs_epi16(t16, t16), 0xD8);
    const __m128i shuf = _mm_setr_epi8(0, 1, 2, 3, 4, 8, 9, 10, 11, 12, -1, -1, -1, -1, -1, -1);
    _mm_storeu_si128((__m128i *)trits, _mm_shuffle_epi8(_mm256_castsi256_si128(pk), shuf));
}

/**
 * @brief Unpacks 16 10-trit tokens in parallel (160 trits) using Granlund-Montgomery
 *        16-bit division by 243 (69043 = 2^16 + 3507) without scalar division.
 *        trits_out[k][0..9] receives the 10 signed trits of toks[k].
 */
static inline void zcrudp_zcvid_unpack10_x16_avx2(const uint16_t toks[16],
                                                    int8_t trits_out[16][16]) {
    const __m256i w = _mm256_loadu_si256((const __m256i *)toks);
    const __m256i gm_c = _mm256_set1_epi16(3507);
    const __m256i v_243 = _mm256_set1_epi16(243);
    const __m256i hi = _mm256_srli_epi16(_mm256_add_epi16(w, _mm256_mulhi_epu16(w, gm_c)), 8);
    const __m256i lo = _mm256_sub_epi16(w, _mm256_mullo_epi16(hi, v_243));

    const __m256i r1 = _mm256_set1_epi16(21846);
    const __m256i r2 = _mm256_set1_epi16(7282);
    const __m256i r3 = _mm256_set1_epi16(2428);
    const __m256i r4 = _mm256_set1_epi16(810);
    const __m256i v3 = _mm256_set1_epi16(3);
    const __m256i v1 = _mm256_set1_epi16(1);

    const __m256i lq1 = _mm256_mulhi_epu16(lo, r1);
    const __m256i lq2 = _mm256_mulhi_epu16(lo, r2);
    const __m256i lq3 = _mm256_mulhi_epu16(lo, r3);
    const __m256i lq4 = _mm256_mulhi_epu16(lo, r4);

    const __m256i hq1 = _mm256_mulhi_epu16(hi, r1);
    const __m256i hq2 = _mm256_mulhi_epu16(hi, r2);
    const __m256i hq3 = _mm256_mulhi_epu16(hi, r3);
    const __m256i hq4 = _mm256_mulhi_epu16(hi, r4);

    const __m256i t0 = _mm256_sub_epi16(_mm256_sub_epi16(lo,  _mm256_mullo_epi16(lq1, v3)), v1);
    const __m256i t1 = _mm256_sub_epi16(_mm256_sub_epi16(lq1, _mm256_mullo_epi16(lq2, v3)), v1);
    const __m256i t2 = _mm256_sub_epi16(_mm256_sub_epi16(lq2, _mm256_mullo_epi16(lq3, v3)), v1);
    const __m256i t3 = _mm256_sub_epi16(_mm256_sub_epi16(lq3, _mm256_mullo_epi16(lq4, v3)), v1);
    const __m256i t4 = _mm256_sub_epi16(lq4, v1);

    const __m256i t5 = _mm256_sub_epi16(_mm256_sub_epi16(hi,  _mm256_mullo_epi16(hq1, v3)), v1);
    const __m256i t6 = _mm256_sub_epi16(_mm256_sub_epi16(hq1, _mm256_mullo_epi16(hq2, v3)), v1);
    const __m256i t7 = _mm256_sub_epi16(_mm256_sub_epi16(hq2, _mm256_mullo_epi16(hq3, v3)), v1);
    const __m256i t8 = _mm256_sub_epi16(_mm256_sub_epi16(hq3, _mm256_mullo_epi16(hq4, v3)), v1);
    const __m256i t9 = _mm256_sub_epi16(hq4, v1);

    ZCRUDP_ALIGNAS32 int16_t rows[10][16];
    _mm256_store_si256((__m256i *)rows[0], t0);
    _mm256_store_si256((__m256i *)rows[1], t1);
    _mm256_store_si256((__m256i *)rows[2], t2);
    _mm256_store_si256((__m256i *)rows[3], t3);
    _mm256_store_si256((__m256i *)rows[4], t4);
    _mm256_store_si256((__m256i *)rows[5], t5);
    _mm256_store_si256((__m256i *)rows[6], t6);
    _mm256_store_si256((__m256i *)rows[7], t7);
    _mm256_store_si256((__m256i *)rows[8], t8);
    _mm256_store_si256((__m256i *)rows[9], t9);

    for (size_t k = 0; k < 16U; k++) {
        for (size_t j = 0; j < 10U; j++) {
            trits_out[k][j] = (int8_t)rows[j][k];
        }
    }
}

/**
 * @brief Applies the 10 trits of a full 1x block: canvas and ADM steps, bit
 *        for bit what zcrudp_zcvid_apply_block() computes. 16 bytes must be
 *        readable at canvas / delta_buf / trits; 10 are written.
 */
static inline void zcrudp_zcvid_apply_block_avx2(uint8_t *canvas,
                                                   uint8_t *delta_buf,
                                                   const zcrudp_zcvid_config_t *cfg,
                                                   const int8_t trits[16]) {
    const __m128i lane10 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0);
    const __m128i zero = _mm_setzero_si128();
    const __m128i v_dmin = _mm_set1_epi8((char)cfg->delta_min);
    const __m128i t = _mm_and_si128(_mm_loadu_si128((const __m128i *)trits), lane10);
    const __m128i c = _mm_loadu_si128((const __m128i *)canvas);
    const __m128i d = (delta_buf != NULL) ? _mm_loadu_si128((const __m128i *)delta_buf) : v_dmin;
    const __m128i pos = _mm_cmpgt_epi8(t, zero);
    const __m128i neg = _mm_cmpgt_epi8(zero, t);

    __m128i nc = _mm_blendv_epi8(c, _mm_adds_epu8(c, d), pos);
    nc = _mm_blendv_epi8(nc, _mm_subs_epu8(c, d), neg);
    zcrudp_zcvid_store10(canvas, nc);

    if (delta_buf != NULL) {
        __m128i nd;
        if (cfg->delta_min >= cfg->delta_max) {
            nd = v_dmin;
        } else {
            const __m128i half = _mm_and_si128(_mm_srli_epi16(d, 1), _mm_set1_epi8(0x7F));
            const __m128i dbl = _mm_min_epu8(_mm_adds_epu8(d, d), _mm_set1_epi8((char)cfg->delta_max));
            nd = _mm_blendv_epi8(_mm_max_epu8(half, v_dmin), dbl, _mm_or_si128(pos, neg));
        }
        zcrudp_zcvid_store10(delta_buf, nd);
    }
}
#endif

/* =========================================================================
 * Entropy-Coded Block Geometry (shared by encoder and decoder)
 * ========================================================================= */

/** Distance between two consecutive coded samples of the same component. */
static inline size_t zcrudp_zcvid_ec_lstep(const zcrudp_zcvid_config_t *cfg,
                                             uint8_t plane_id, bool is_2x) {
    return (zcrudp_zcvid_2x_interleaved(cfg, plane_id) ? 2U : 1U) * (is_2x ? 2U : 1U);
}

/**
 * @brief Row-above part of the contexts of a whole block, read before any
 *        trit of the block is coded: upz[j] goes into the zero-flag context,
 *        ups[j] (0, 1 = plus, 2 = minus) into the sign context.
 *
 * Doing it once per block keeps the per-trit loop short, and for the common
 * case (1x block, ring not wrapping) it is three 16-byte loads.
 */
ZCRUDP_ZCVID_INLINE void zcrudp_zcvid_ec_up_contexts(const zcrudp_tec_model_t *m,
                                                         size_t base,
                                                         bool is_2x,
                                                         bool il,
                                                         uint8_t upz[16],
                                                         uint8_t ups[16]) {
    const uint8_t *hist = m->hist;
    const size_t lstep = m->lstep;
    const size_t stride = m->stride;

    if (stride == 0U) {
        memset(upz, 0, 16);
        memset(ups, 0, 16);
        return;
    }
#ifdef __AVX2__
    if (!is_2x) {
        const size_t pos = (base - stride - lstep) & ZCRUDP_TEC_HIST_MASK;
        if (pos + 2U * lstep + 16U <= ZCRUDP_TEC_HIST_SZ) {
            const __m128i one = _mm_set1_epi8(1);
            const __m128i zero = _mm_setzero_si128();
            const __m128i ul = _mm_loadu_si128((const __m128i *)(hist + pos));
            const __m128i u  = _mm_loadu_si128((const __m128i *)(hist + pos + lstep));
            const __m128i ur = _mm_loadu_si128((const __m128i *)(hist + pos + 2U * lstep));
            /* codes are 0..2, so min(code, 1) is "non-zero" and the sums stay in a byte */
            const __m128i side = _mm_add_epi8(_mm_min_epu8(ul, one), _mm_min_epu8(ur, one));
            const __m128i z = _mm_add_epi8(_mm_slli_epi16(_mm_min_epu8(u, one), 2),
                                           _mm_slli_epi16(side, 3));
            const __m128i sg = _mm_blendv_epi8(u, _mm_blendv_epi8(ul, ur, _mm_cmpeq_epi8(ul, zero)),
                                               _mm_cmpeq_epi8(u, zero));
            _mm_storeu_si128((__m128i *)upz, z);
            _mm_storeu_si128((__m128i *)ups, sg);
            return;
        }
    }
#endif
    for (size_t j = 0; j < 10U; j++) {
        const size_t idx = is_2x ? zcrudp_zcvid_2x_idx(base, j, il) : (base + j);
        const uint32_t u  = hist[(idx - stride) & ZCRUDP_TEC_HIST_MASK];
        const uint32_t ul = hist[(idx - stride - lstep) & ZCRUDP_TEC_HIST_MASK];
        const uint32_t ur = hist[(idx - stride + lstep) & ZCRUDP_TEC_HIST_MASK];
        upz[j] = (uint8_t)(((uint32_t)(u != 0U) << 2) |
                           (((uint32_t)(ul != 0U) + (uint32_t)(ur != 0U)) << 3));
        ups[j] = (uint8_t)(u ? u : (ul ? ul : ur));
    }
}

/**
 * @brief Range-codes the 10 trits of a block. Samples past the end of the
 *        plane are not coded. A block may be all 0: with ADM that is how the
 *        encoder asks for the steps of the block to be halved.
 */
ZCRUDP_ZCVID_INLINE void zcrudp_zcvid_ec_put_block(zcrudp_rc_enc_t *rc_io,
                                               zcrudp_tec_model_t *m,
                                               size_t base,
                                               size_t plane_len,
                                               bool is_2x,
                                               bool il,
                                               const zcrudp_trit_t trits[10]) {
    const size_t pair = zcrudp_zcvid_2x_pair(il);
    /* Coder state and model geometry live in locals for the whole block. */
    zcrudp_rc_enc_t rc = *rc_io;
    uint8_t *hist = m->hist;
    const size_t lstep = m->lstep;
    uint8_t upz[16], ups[16];
    zcrudp_zcvid_ec_up_contexts(m, base, is_2x, il, upz, ups);

    for (size_t j = 0; j < 10U; j++) {
        size_t idx = is_2x ? zcrudp_zcvid_2x_idx(base, j, il) : (base + j);
        if (idx >= plane_len) {
            break;
        }
        const uint32_t l1 = hist[(idx - lstep) & ZCRUDP_TEC_HIST_MASK];
        const uint32_t l2 = hist[(idx - 2U * lstep) & ZCRUDP_TEC_HIST_MASK];
        const uint32_t zctx = (uint32_t)(l1 != 0U) | ((uint32_t)(l2 != 0U) << 1) | upz[j];

        zcrudp_trit_t t = trits[j];
        zcrudp_rc_enc_bit(&rc, &m->zero[zctx], (t != 0) ? 1U : 0U);
        uint8_t code = 0;
        if (t != 0) {
            zcrudp_rc_enc_bit(&rc, &m->sign[l1 * 3U + ups[j]], (t < 0) ? 1U : 0U);
            code = (t < 0) ? 2U : 1U;
        }
        hist[idx & ZCRUDP_TEC_HIST_MASK] = code;
        if (is_2x) {
            hist[(idx + pair) & ZCRUDP_TEC_HIST_MASK] = code;
        }
    }
    *rc_io = rc;
}

ZCRUDP_ZCVID_INLINE void zcrudp_zcvid_ec_get_block(zcrudp_rc_dec_t *rc_io,
                                               zcrudp_tec_model_t *m,
                                               size_t base,
                                               size_t plane_len,
                                               bool is_2x,
                                               bool il,
                                               zcrudp_trit_t trits[10]) {
    const size_t pair = zcrudp_zcvid_2x_pair(il);
    zcrudp_rc_dec_t rc = *rc_io;
    uint8_t *hist = m->hist;
    const size_t lstep = m->lstep;
    uint8_t upz[16], ups[16];
    zcrudp_zcvid_ec_up_contexts(m, base, is_2x, il, upz, ups);

    for (size_t j = 0; j < 10U; j++) {
        trits[j] = 0;
    }
    for (size_t j = 0; j < 10U; j++) {
        size_t idx = is_2x ? zcrudp_zcvid_2x_idx(base, j, il) : (base + j);
        if (idx >= plane_len) {
            break;
        }
        const uint32_t l1 = hist[(idx - lstep) & ZCRUDP_TEC_HIST_MASK];
        const uint32_t l2 = hist[(idx - 2U * lstep) & ZCRUDP_TEC_HIST_MASK];
        const uint32_t zctx = (uint32_t)(l1 != 0U) | ((uint32_t)(l2 != 0U) << 1) | upz[j];

        uint8_t code = 0;
        if (zcrudp_rc_dec_bit(&rc, &m->zero[zctx]) != 0U) {
            bool neg = zcrudp_rc_dec_bit(&rc, &m->sign[l1 * 3U + ups[j]]) != 0U;
            trits[j] = neg ? (zcrudp_trit_t)-1 : (zcrudp_trit_t)1;
            code = neg ? 2U : 1U;
        }
        hist[idx & ZCRUDP_TEC_HIST_MASK] = code;
        if (is_2x) {
            hist[(idx + pair) & ZCRUDP_TEC_HIST_MASK] = code;
        }
    }
    *rc_io = rc;
}

/* =========================================================================
 * Block Emitter: packs non-zero blocks into MTU-sized packets, either as
 * fixed 16-bit tokens with RLE skips or as a range-coded stream.
 * ========================================================================= */

#define ZCRUDP_RANS_MAX_BLOCKS 2048U
#define ZCRUDP_RANS_MAX_SYMS   8192U

typedef struct {
    const zcrudp_zcvid_config_t *cfg;
    uint8_t *pkt_buf;
    zcrudp_zcvid_pkt_cb cb;
    void *user_ctx;
    size_t total_bytes;
    size_t plane_len;
    size_t span;            /**< samples covered by one block (10 or 20) */
    size_t max_tokens;      /**< token mode: tokens per packet */
    size_t max_payload;     /**< bytes available after the header */
    size_t next_blk;        /**< block that follows the last one emitted */
    uint32_t pkt_start_block;
    uint32_t pkt_count;     /**< token mode: tokens, range-coded/rANS: blocks */
    uint16_t frame_id;
    uint8_t plane_id;
    bool is_2x;
    bool il;
    bool ec;
    bool rans;
    bool rv2;
    size_t ec_blocks_left;  /**< range-coded blocks still allowed (CPU cap) */
    zcrudp_rc_enc_t rc;
    zcrudp_tec_model_t model; /**< only touched when ec is set */
    /* Direct rANS accumulator (used when rans is set) */
    uint16_t rans_num_nz;
    uint16_t rans_num_syms;
    uint16_t rans_skip_len;
    uint16_t rans_rem_est;
    uint16_t rans_last_hdr;
    uint32_t rans_worst_bits;
    uint32_t rans_last_bits;
    uint8_t  rans_last_prof;
    uint32_t rans_raw_toks;
    uint32_t rans_zrun;
    uint32_t rans_dropped_blocks; /**< Non-zero only if unreachable rANS drop branch triggers */
    uint8_t  rans_syms[ZCRUDP_RANS_MAX_SYMS];
    uint8_t  rans_sign_buf[(ZCRUDP_RANS_MAX_BLOCKS * 10U + 7U) / 8U + 4U];
    uint8_t  rans_skip_mask[ZCRUDP_RANS_MAX_BLOCKS / 8U];
    uint8_t  rans_skip_vals[ZCRUDP_RANS_MAX_BLOCKS * 3U];
} zcrudp_zcvid_emit_t;

static inline void zcrudp_zcvid_emit_init(zcrudp_zcvid_emit_t *e,
                                            const zcrudp_zcvid_config_t *cfg,
                                            size_t plane_len,
                                            uint8_t plane_id,
                                            uint16_t frame_id,
                                            bool is_2x,
                                            uint8_t *pkt_buf,
                                            zcrudp_zcvid_pkt_cb cb,
                                            void *user_ctx) {
    e->cfg = cfg;
    e->pkt_buf = pkt_buf;
    e->cb = cb;
    e->user_ctx = user_ctx;
    e->total_bytes = 0;
    e->plane_len = plane_len;
    e->span = is_2x ? ZCRUDP_ZCVID_2X_SPAN : 10U;
    e->max_payload = (cfg->mtu_payload > ZCRUDP_ZCVID_PKT_HDR_SZ)
        ? (size_t)(cfg->mtu_payload - ZCRUDP_ZCVID_PKT_HDR_SZ) : 0U;
    e->max_tokens = (e->max_payload >= 2U) ? (e->max_payload / 2U) : 0U;
    e->next_blk = 0;
    e->pkt_start_block = 0;
    e->pkt_count = 0;
    e->frame_id = frame_id;
    e->plane_id = plane_id;
    e->is_2x = is_2x;
    e->il = is_2x && zcrudp_zcvid_2x_interleaved(cfg, plane_id);
    /* A packet too small to hold a worst-case block falls back to tokens. */
    e->ec = (cfg->entropy & ZCRUDP_ZCVID_EC_DELTA) != 0U &&
            e->max_payload >= 2U * ZCRUDP_TEC_BLOCK_MARGIN;
    e->rans = !e->ec && !is_2x &&
              (cfg->entropy & ZCRUDP_ZCVID_EC_RANS) != 0U &&
              e->max_payload >= 64U;
    e->rv2 = !e->rans && !is_2x &&
             (cfg->entropy & ZCRUDP_ZCVID_EC_REALLOC_V2) != 0U;
    e->ec_blocks_left = (cfg->ec_max_blocks != 0U) ? (size_t)cfg->ec_max_blocks : SIZE_MAX;
    e->rans_num_nz = 0U;
    e->rans_num_syms = 0U;
    e->rans_skip_len = 0U;
    e->rans_rem_est = 0U;
    e->rans_last_hdr = 0U;
    e->rans_worst_bits = 0U;
    e->rans_last_bits = 0U;
    e->rans_last_prof = 0xFFU;
    e->rans_raw_toks = 0U;
    e->rans_zrun = 0U;
    e->rans_dropped_blocks = 0U;
}

static inline size_t zcrudp_zcvid_try_rans_pack(uint8_t *payload_io,
                                                  size_t tok_count,
                                                  size_t max_payload,
                                                  uint16_t *out_num_blocks) {
    if (tok_count == 0U || tok_count > 1024U) {
        return 0U;
    }
    const size_t tok_bytes = tok_count * 2U;
    uint8_t syms[4096];
    /* 10 sign bits per token at most, plus the byte the memset below adds:
     * 1281 bytes for 1024 tokens. */
    uint8_t sign_buf[(1024U * 10U + 7U) / 8U + 4U];
    uint8_t skip_mask[128];
    uint8_t skip_vals[2048];
    uint8_t rans_out[2048];
    uint8_t rans_scratch[2048];

    memset(sign_buf, 0, (tok_count * 10U + 7U) / 8U + 1U);
    memset(skip_mask, 0, (tok_count + 7U) / 8U);

    size_t k = 0U;
    size_t num_nz = 0U;
    size_t num_syms = 0U;
    size_t zrun = 0U;
    size_t skip_len = 0U;
    uint32_t pending_skip = 0U;

    for (size_t i = 0U; i < tok_count; i++) {
        uint16_t tok = zcrudp_zcvid_read_u16_le(payload_io + i * 2U);
        if (tok >= ZCRUDP_TRIT_MAX_10P) {
            pending_skip += (uint32_t)(tok - ZCRUDP_TRIT_MAX_10P) + 1U;
            continue;
        }
        if (k > 0U && pending_skip > 0U) {
            skip_mask[k >> 3] |= (uint8_t)(1U << (k & 7U));
            if (pending_skip < 255U) {
                if (skip_len + 1U > sizeof(skip_vals)) return 0U;
                skip_vals[skip_len++] = (uint8_t)pending_skip;
            } else if (pending_skip <= 65535U) {
                if (skip_len + 3U > sizeof(skip_vals)) return 0U;
                skip_vals[skip_len++] = 255U;
                skip_vals[skip_len++] = (uint8_t)(pending_skip & 0xFFU);
                skip_vals[skip_len++] = (uint8_t)((pending_skip >> 8) & 0xFFU);
            } else {
                return 0U;
            }
        }
        pending_skip = 0U;

        zcrudp_trit_t trits[10];
        zcrudp_unpack_10trits(tok, trits);
        for (size_t j = 0U; j < 10U; j++) {
            zcrudp_trit_t t = trits[j];
            if (t == 0) {
                zrun++;
            } else {
                while (zrun >= 128U) {
                    if (num_syms >= sizeof(syms)) return 0U;
                    syms[num_syms++] = 15U;
                    zrun -= 128U;
                }
                while (zrun >= 14U) {
                    if (num_syms >= sizeof(syms)) return 0U;
                    syms[num_syms++] = 14U;
                    zrun -= 14U;
                }
                if (num_syms >= sizeof(syms)) return 0U;
                syms[num_syms++] = (uint8_t)zrun;
                zrun = 0U;
                if (t < 0) {
                    sign_buf[num_nz >> 3] |= (uint8_t)(1U << (num_nz & 7U));
                }
                num_nz++;
            }
        }
        k++;
    }
    if (k == 0U || num_nz > 65535U || num_syms > 65535U || skip_len > 65535U) {
        return 0U;
    }

    const size_t mask_len = (k + 7U) / 8U;
    const size_t sign_len = (num_nz + 7U) / 8U;
    const size_t hdr_len  = 6U + mask_len + skip_len + sign_len;
    if (hdr_len + 17U >= tok_bytes || hdr_len + 17U > max_payload) {
        return 0U;
    }

    uint8_t profile = zcrudp_rans_pick_profile(num_nz, k * 10U);
    size_t rans_len = zcrudp_rans_encode_syms(
        syms, num_syms, profile, rans_out, sizeof(rans_out), rans_scratch, sizeof(rans_scratch));
    if (rans_len == 0U || hdr_len + rans_len >= tok_bytes || hdr_len + rans_len > max_payload) {
        return 0U;
    }

    payload_io[0] = (uint8_t)(num_nz & 0xFFU);
    payload_io[1] = (uint8_t)((num_nz >> 8) & 0xFFU);
    payload_io[2] = (uint8_t)(num_syms & 0xFFU);
    payload_io[3] = (uint8_t)((num_syms >> 8) & 0xFFU);
    payload_io[4] = (uint8_t)(skip_len & 0xFFU);
    payload_io[5] = (uint8_t)((skip_len >> 8) & 0xFFU);
    memcpy(payload_io + 6U, skip_mask, mask_len);
    if (skip_len > 0U) memcpy(payload_io + 6U + mask_len, skip_vals, skip_len);
    if (sign_len > 0U) memcpy(payload_io + 6U + mask_len + skip_len, sign_buf, sign_len);
    memcpy(payload_io + hdr_len, rans_out, rans_len);
    *out_num_blocks = (uint16_t)k;
    return hdr_len + rans_len;
}

static inline size_t zcrudp_zcvid_rans_cap(const zcrudp_zcvid_emit_t *e) {
    return (e->max_payload > 2048U) ? 2048U : e->max_payload;
}

static inline size_t zcrudp_zcvid_emit_rans_pack(zcrudp_zcvid_emit_t *e,
                                                   uint8_t *payload) {
    const size_t rcap = zcrudp_zcvid_rans_cap(e);
    const size_t k = (size_t)e->pkt_count;
    const size_t mask_len = (k + 7U) / 8U;
    const size_t sign_len = ((size_t)e->rans_num_nz + 7U) / 8U;
    const size_t hdr_len  = 6U + mask_len + (size_t)e->rans_skip_len + sign_len;
    if (hdr_len + 17U > rcap) {
        return 0U;
    }
    uint8_t rans_scratch[2048];
    uint8_t profile = zcrudp_rans_pick_profile(e->rans_num_nz, k * 10U);
    size_t rans_len = zcrudp_rans_encode_syms(
        e->rans_syms, e->rans_num_syms, profile,
        payload + hdr_len, rcap - hdr_len,
        rans_scratch, sizeof(rans_scratch));
    if (rans_len == 0U) {
        return 0U;
    }
    payload[0] = (uint8_t)(e->rans_num_nz & 0xFFU);
    payload[1] = (uint8_t)((e->rans_num_nz >> 8) & 0xFFU);
    payload[2] = (uint8_t)(e->rans_num_syms & 0xFFU);
    payload[3] = (uint8_t)((e->rans_num_syms >> 8) & 0xFFU);
    payload[4] = (uint8_t)(e->rans_skip_len & 0xFFU);
    payload[5] = (uint8_t)((e->rans_skip_len >> 8) & 0xFFU);
    memcpy(payload + 6U, e->rans_skip_mask, mask_len);
    if (e->rans_skip_len > 0U) memcpy(payload + 6U + mask_len, e->rans_skip_vals, e->rans_skip_len);
    if (sign_len > 0U) memcpy(payload + 6U + mask_len + e->rans_skip_len, e->rans_sign_buf, sign_len);
    return hdr_len + rans_len;
}

static inline void zcrudp_zcvid_emit_send(zcrudp_zcvid_emit_t *e,
                                            uint8_t ptype,
                                            size_t payload_len,
                                            uint32_t out_count) {
    zcrudp_zcvid_pkt_hdr_t hdr = {
        .magic = ZCRUDP_ZCVID_PKT_MAGIC,
        .pixfmt = ZCRUDP_ZCVID_MAKE_PF(e->cfg->pixfmt, ptype),
        .plane = e->plane_id,
        .frame_id = e->frame_id,
        .start_block = e->pkt_start_block,
        .num_tokens = (uint16_t)out_count
    };
    zcrudp_zcvid_pkt_hdr_write_le(e->pkt_buf, &hdr);
    size_t psz = ZCRUDP_ZCVID_PKT_HDR_SZ + payload_len;
    e->cb(e->pkt_buf, psz, e->user_ctx);
    e->total_bytes += psz;
    e->pkt_count = 0;
    e->rans_num_nz = 0U;
    e->rans_num_syms = 0U;
    e->rans_skip_len = 0U;
    e->rans_rem_est = 0U;
    e->rans_last_hdr = 0U;
    e->rans_worst_bits = 0U;
    e->rans_last_bits = 0U;
    e->rans_last_prof = 0xFFU;
    e->rans_raw_toks = 0U;
    e->rans_zrun = 0U;
}

static inline void zcrudp_zcvid_emit_flush(zcrudp_zcvid_emit_t *e) {
    if (e->pkt_count == 0U) {
        return;
    }
    if (e->rans) {
        if (e->rans_raw_toks > 0U && e->rans_raw_toks <= e->max_tokens) {
            uint8_t tmp_rans[2048];
            size_t rlen = zcrudp_zcvid_emit_rans_pack(e, tmp_rans);
            if (rlen > 0U && rlen < (size_t)e->rans_raw_toks * 2U) {
                memcpy(e->pkt_buf + ZCRUDP_ZCVID_PKT_HDR_SZ, tmp_rans, rlen);
                zcrudp_zcvid_emit_send(e, ZCRUDP_ZCVID_PKT_DELTA_RANS, rlen, e->pkt_count);
                return;
            }
            zcrudp_zcvid_emit_send(e, ZCRUDP_ZCVID_PKT_DELTA,
                                     (size_t)e->rans_raw_toks * 2U, e->rans_raw_toks);
            return;
        }
        size_t rlen = zcrudp_zcvid_emit_rans_pack(e, e->pkt_buf + ZCRUDP_ZCVID_PKT_HDR_SZ);
        if (rlen > 0U) {
            zcrudp_zcvid_emit_send(e, ZCRUDP_ZCVID_PKT_DELTA_RANS, rlen, e->pkt_count);
            return;
        }
        /* Unreachable under the headroom invariant: count dropped blocks and assert in debug builds. */
        e->rans_dropped_blocks += e->pkt_count;
        assert(0 && "zcrudp_zcvid_emit_flush: unreachable rANS drop branch hit");
        e->pkt_count = 0;
        e->rans_num_nz = 0U;
        e->rans_num_syms = 0U;
        e->rans_skip_len = 0U;
        e->rans_rem_est = 0U;
        e->rans_last_hdr = 0U;
        e->rans_worst_bits = 0U;
        e->rans_last_bits = 0U;
        e->rans_last_prof = 0xFFU;
        e->rans_raw_toks = 0U;
        e->rans_zrun = 0U;
        return;
    }
    uint8_t ptype = e->ec
        ? (e->is_2x ? ZCRUDP_ZCVID_PKT_DELTA_2X_EC : ZCRUDP_ZCVID_PKT_DELTA_EC)
        : (e->rv2 ? ZCRUDP_ZCVID_PKT_DELTA_RV2
                  : (e->is_2x ? ZCRUDP_ZCVID_PKT_DELTA_2X : ZCRUDP_ZCVID_PKT_DELTA));
    size_t payload = e->ec ? zcrudp_rc_enc_finish(&e->rc) : (size_t)e->pkt_count * 2U;
    zcrudp_zcvid_emit_send(e, ptype, payload, e->pkt_count);
}

/** Emits one coded block. Blocks must be passed in increasing order. */
static inline void zcrudp_zcvid_emit_block(zcrudp_zcvid_emit_t *e,
                                             size_t blk,
                                             const zcrudp_trit_t trits[10]) {
    if (e->max_payload < 2U) {
        return;
    }
    uint8_t *payload = e->pkt_buf + ZCRUDP_ZCVID_PKT_HDR_SZ;

    if (e->ec && e->ec_blocks_left == 0U) {
        /* CPU cap reached: close the range-coded packet, continue with tokens.
         * The decoder follows on its own, the type is in each packet header. */
        zcrudp_zcvid_emit_flush(e);
        e->ec = false;
    }
    if (e->ec) {
        e->ec_blocks_left--;
        if (e->pkt_count > 0U &&
            (zcrudp_rc_enc_pending(&e->rc) + ZCRUDP_TEC_BLOCK_MARGIN > e->max_payload ||
             e->pkt_count >= 65535U)) {
            zcrudp_zcvid_emit_flush(e);
        }
        if (e->pkt_count == 0U) {
            zcrudp_rc_enc_init(&e->rc, payload, e->max_payload);
            zcrudp_tec_model_reset(&e->model, e->cfg->width,
                                   zcrudp_zcvid_ec_lstep(e->cfg, e->plane_id, e->is_2x));
            e->pkt_start_block = (uint32_t)blk;
        } else {
            zcrudp_tec_enc_run(&e->rc, &e->model, (uint64_t)(blk - e->next_blk));
            zcrudp_tec_hist_clear(&e->model, e->next_blk * e->span, blk * e->span);
        }
        zcrudp_zcvid_ec_put_block(&e->rc, &e->model, blk * e->span, e->plane_len,
                                    e->is_2x, e->il, trits);
        e->pkt_count++;
        e->next_blk = blk + 1U;
        return;
    }

    if (e->rans) {
    retry_rans:
        if (e->pkt_count > 0U) {
            size_t gap = blk - e->next_blk;
            if (gap > 65535U ||
                e->pkt_count >= ZCRUDP_RANS_MAX_BLOCKS ||
                (size_t)e->rans_skip_len + 3U > sizeof(e->rans_skip_vals)) {
                zcrudp_zcvid_emit_flush(e);
            }
        }
        const uint32_t ck_k    = e->pkt_count;
        const uint16_t ck_nz   = e->rans_num_nz;
        const uint16_t ck_syms = e->rans_num_syms;
        const uint16_t ck_skip = e->rans_skip_len;
        const uint32_t ck_raw  = e->rans_raw_toks;
        const uint32_t ck_zrun = e->rans_zrun;
        const uint32_t ck_bits = e->rans_worst_bits;
        const uint8_t  ck_mask = e->rans_skip_mask[ck_k >> 3];
        const uint8_t  ck_sign = e->rans_sign_buf[ck_nz >> 3];

        if (ck_k == 0U) {
            e->pkt_start_block = (uint32_t)blk;
            e->rans_skip_mask[0] = 0U;
            e->rans_sign_buf[0] = 0U;
            e->rans_raw_toks = 0U;
        } else {
            if ((ck_k & 7U) == 0U) {
                e->rans_skip_mask[ck_k >> 3] = 0U;
            }
            size_t gap = blk - e->next_blk;
            if (gap > 0U) {
                e->rans_skip_mask[ck_k >> 3] |= (uint8_t)(1U << (ck_k & 7U));
                if (gap < 255U) {
                    e->rans_skip_vals[e->rans_skip_len++] = (uint8_t)gap;
                } else {
                    e->rans_skip_vals[e->rans_skip_len++] = 255U;
                    e->rans_skip_vals[e->rans_skip_len++] = (uint8_t)(gap & 0xFFU);
                    e->rans_skip_vals[e->rans_skip_len++] = (uint8_t)((gap >> 8) & 0xFFU);
                }
                if (e->rans_raw_toks <= e->max_tokens) {
                    size_t nskip = (gap + ZCRUDP_ZCVID_MAX_SKIP - 1U) / ZCRUDP_ZCVID_MAX_SKIP;
                    if ((size_t)e->rans_raw_toks + nskip + 1U <= e->max_tokens) {
                        size_t g = gap;
                        while (g > 0U) {
                            size_t run = (g > ZCRUDP_ZCVID_MAX_SKIP) ? ZCRUDP_ZCVID_MAX_SKIP : g;
                            zcrudp_zcvid_write_u16_le(
                                payload + (size_t)e->rans_raw_toks * 2U,
                                (uint16_t)(ZCRUDP_TRIT_MAX_10P + (run - 1U)));
                            e->rans_raw_toks++;
                            g -= run;
                        }
                    } else {
                        e->rans_raw_toks = (uint32_t)e->max_tokens + 1U;
                    }
                }
            }
        }
        if (e->rans_raw_toks < e->max_tokens) {
            zcrudp_zcvid_write_u16_le(
                payload + (size_t)e->rans_raw_toks * 2U,
                zcrudp_pack_10trits(trits));
            e->rans_raw_toks++;
        } else {
            e->rans_raw_toks = (uint32_t)e->max_tokens + 1U;
        }

        bool overflow = false;
        for (size_t j = 0U; j < 10U; j++) {
            zcrudp_trit_t t = trits[j];
            if (t == 0) {
                e->rans_zrun++;
            } else {
                while (e->rans_zrun >= 128U) {
                    if (e->rans_num_syms >= ZCRUDP_RANS_MAX_SYMS) { overflow = true; break; }
                    e->rans_syms[e->rans_num_syms++] = 15U;
                    e->rans_worst_bits += 11U;
                    e->rans_zrun -= 128U;
                }
                while (!overflow && e->rans_zrun >= 14U) {
                    if (e->rans_num_syms >= ZCRUDP_RANS_MAX_SYMS) { overflow = true; break; }
                    e->rans_syms[e->rans_num_syms++] = 14U;
                    e->rans_worst_bits += 11U;
                    e->rans_zrun -= 14U;
                }
                if (overflow || e->rans_num_syms >= ZCRUDP_RANS_MAX_SYMS) {
                    overflow = true;
                    break;
                }
                uint8_t rsym = (uint8_t)e->rans_zrun;
                e->rans_syms[e->rans_num_syms++] = rsym;
                e->rans_worst_bits += (rsym == 0U) ? 3U : (rsym <= 2U) ? 4U : (rsym == 3U) ? 5U : 11U;
                e->rans_zrun = 0U;
                if ((e->rans_num_nz & 7U) == 0U) {
                    e->rans_sign_buf[e->rans_num_nz >> 3] = 0U;
                }
                if (t < 0) {
                    e->rans_sign_buf[e->rans_num_nz >> 3] |= (uint8_t)(1U << (e->rans_num_nz & 7U));
                }
                e->rans_num_nz++;
            }
        }
        if (overflow) {
            e->pkt_count = ck_k;
            e->rans_num_nz = ck_nz;
            e->rans_num_syms = ck_syms;
            e->rans_skip_len = ck_skip;
            e->rans_raw_toks = ck_raw;
            e->rans_zrun = ck_zrun;
            e->rans_worst_bits = ck_bits;
            e->rans_skip_mask[ck_k >> 3] = ck_mask;
            e->rans_sign_buf[ck_nz >> 3] = ck_sign;
            zcrudp_zcvid_emit_flush(e);
            goto retry_rans;
        }
        e->pkt_count++;

        const size_t rcap = zcrudp_zcvid_rans_cap(e);
        const size_t cur_hdr = 6U + (((size_t)e->pkt_count + 7U) / 8U) +
                               (size_t)e->rans_skip_len + (((size_t)e->rans_num_nz + 7U) / 8U);
        /* Unconditional upper bound for any profile using per-symbol max bit costs:
         * skip calling emit_rans_pack entirely while the worst-case total still has headroom. */
        const size_t abs_worst = cur_hdr + 21U + (((size_t)e->rans_worst_bits + 7U) / 8U);
        bool need_check = false;
        uint8_t cur_prof = e->rans_last_prof;
        if (abs_worst + 16U >= rcap) {
            cur_prof = zcrudp_rans_pick_profile(e->rans_num_nz, (size_t)e->pkt_count * 10U);
            if (e->rans_last_prof == 0xFFU || cur_prof != e->rans_last_prof) {
                need_check = true;
            } else {
                const size_t hdr_delta = (cur_hdr > (size_t)e->rans_last_hdr)
                    ? (cur_hdr - (size_t)e->rans_last_hdr) : 0U;
                const size_t bits_delta = (e->rans_worst_bits > e->rans_last_bits)
                    ? ((size_t)e->rans_worst_bits - (size_t)e->rans_last_bits) : 0U;
                const size_t worst_add = hdr_delta + ((bits_delta + 7U) / 8U) + 4U;
                if (worst_add + 16U >= (size_t)e->rans_rem_est) {
                    need_check = true;
                }
            }
        }

        if (need_check) {
            /* Pack into temporary buffer when raw tokens are still buffered so we don't clobber them */
            uint8_t tmp_rans[2048];
            uint8_t *dst_buf = (e->rans_raw_toks <= e->max_tokens) ? tmp_rans : payload;
            size_t rlen = zcrudp_zcvid_emit_rans_pack(e, dst_buf);
            if (rlen == 0U) {
                if (ck_k > 0U) {
                    e->pkt_count = ck_k;
                    e->rans_num_nz = ck_nz;
                    e->rans_num_syms = ck_syms;
                    e->rans_skip_len = ck_skip;
                    e->rans_raw_toks = ck_raw;
                    e->rans_zrun = ck_zrun;
                    e->rans_worst_bits = ck_bits;
                    e->rans_skip_mask[ck_k >> 3] = ck_mask;
                    e->rans_sign_buf[ck_nz >> 3] = ck_sign;
                    zcrudp_zcvid_emit_flush(e);
                    goto retry_rans;
                }
                /* Single-block packet couldn't fit rANS: send as raw token. */
                zcrudp_zcvid_emit_send(e, ZCRUDP_ZCVID_PKT_DELTA,
                                         (size_t)e->rans_raw_toks * 2U, e->rans_raw_toks);
                e->next_blk = blk + 1U;
                return;
            } else if (rlen + 16U >= rcap) {
                if (dst_buf == tmp_rans) {
                    if (e->rans_raw_toks <= e->max_tokens && rlen >= (size_t)e->rans_raw_toks * 2U) {
                        zcrudp_zcvid_emit_send(e, ZCRUDP_ZCVID_PKT_DELTA,
                                                 (size_t)e->rans_raw_toks * 2U, e->rans_raw_toks);
                        e->next_blk = blk + 1U;
                        return;
                    }
                    memcpy(payload, tmp_rans, rlen);
                }
                zcrudp_zcvid_emit_send(e, ZCRUDP_ZCVID_PKT_DELTA_RANS, rlen, e->pkt_count);
                e->next_blk = blk + 1U;
                return;
            } else {
                e->rans_rem_est   = (uint16_t)(rcap - rlen);
                e->rans_last_hdr  = (uint16_t)cur_hdr;
                e->rans_last_bits = e->rans_worst_bits;
                e->rans_last_prof = cur_prof;
            }
        }
        e->next_blk = blk + 1U;
        return;
    }

    if (e->pkt_count > 0U) {
        size_t gap = blk - e->next_blk;
        if (e->rv2) {
            /* Realloc v2: fuse skip + W1 (1..160) or W2 (1..16) into a single 16-bit token */
            size_t nz = 0U, p0 = 0U, p1 = 0U;
            for (size_t j = 0U; j < 10U; j++) {
                if (trits[j] != 0) {
                    if (nz == 0U) p0 = j;
                    else if (nz == 1U) p1 = j;
                    nz++;
                }
            }
            if (nz == 1U && gap >= 1U && gap <= 160U) {
                uint32_t pat = (uint32_t)(p0 * 2U + ((trits[p0] < 0) ? 1U : 0U));
                uint16_t tok = (gap <= 128U)
                    ? (uint16_t)(59440U + (pat << 7) + (gap - 1U))
                    : (uint16_t)(62000U + (pat << 5) + (gap - 129U));
                zcrudp_zcvid_write_u16_le(payload + (size_t)e->pkt_count * 2U, tok);
                e->pkt_count++;
                e->next_blk = blk + 1U;
                if ((size_t)e->pkt_count >= e->max_tokens) zcrudp_zcvid_emit_flush(e);
                return;
            }
            if (nz == 2U && gap >= 1U && gap <= 16U) {
                uint32_t pair = (uint32_t)((p0 * (19U - p0)) / 2U + (p1 - p0 - 1U));
                uint32_t pat  = pair * 4U + ((trits[p0] < 0) ? 2U : 0U) + ((trits[p1] < 0) ? 1U : 0U);
                uint16_t tok  = (uint16_t)(62640U + (pat << 4) + (gap - 1U));
                zcrudp_zcvid_write_u16_le(payload + (size_t)e->pkt_count * 2U, tok);
                e->pkt_count++;
                e->next_blk = blk + 1U;
                if ((size_t)e->pkt_count >= e->max_tokens) zcrudp_zcvid_emit_flush(e);
                return;
            }
            /* Count dual-resolution RLE tokens needed for gap */
            size_t tmp_gap = gap, nskip = 0U;
            while (tmp_gap >= 512U) {
                uint32_t p2 = 6U;
                while ((512U << p2) > tmp_gap) p2--;
                tmp_gap -= (size_t)(512U << p2);
                nskip++;
            }
            nskip += (tmp_gap + 383U) / 384U;
            if ((size_t)e->pkt_count + nskip + 1U > e->max_tokens) {
                zcrudp_zcvid_emit_flush(e);
            } else {
                while (gap >= 512U) {
                    uint32_t p2 = 6U;
                    while ((512U << p2) > gap) p2--;
                    zcrudp_zcvid_write_u16_le(payload + (size_t)e->pkt_count * 2U,
                                                (uint16_t)(59433U + p2));
                    e->pkt_count++;
                    gap -= (size_t)(512U << p2);
                }
                while (gap > 0U) {
                    size_t run = (gap > 384U) ? 384U : gap;
                    zcrudp_zcvid_write_u16_le(payload + (size_t)e->pkt_count * 2U,
                                                (uint16_t)(59049U + (run - 1U)));
                    e->pkt_count++;
                    gap -= run;
                }
            }
        } else {
            size_t nskip = (gap + ZCRUDP_ZCVID_MAX_SKIP - 1U) / ZCRUDP_ZCVID_MAX_SKIP;
            if ((size_t)e->pkt_count + nskip + 1U > e->max_tokens) {
                /* The next packet restarts at blk: the skip needs no token. */
                zcrudp_zcvid_emit_flush(e);
            } else {
                while (gap > 0U) {
                    size_t run = (gap > ZCRUDP_ZCVID_MAX_SKIP) ? ZCRUDP_ZCVID_MAX_SKIP : gap;
                    zcrudp_zcvid_write_u16_le(payload + (size_t)e->pkt_count * 2U,
                                                (uint16_t)(ZCRUDP_TRIT_MAX_10P + (run - 1U)));
                    e->pkt_count++;
                    gap -= run;
                }
            }
        }
    }
    if (e->pkt_count == 0U) {
        e->pkt_start_block = (uint32_t)blk;
    }
    zcrudp_zcvid_write_u16_le(payload + (size_t)e->pkt_count * 2U,
                                zcrudp_pack_10trits(trits));
    e->pkt_count++;
    e->next_blk = blk + 1U;
    if ((size_t)e->pkt_count >= e->max_tokens) {
        zcrudp_zcvid_emit_flush(e);
    }
}

static inline size_t zcrudp_zcvid_emit_finish(zcrudp_zcvid_emit_t *e) {
    zcrudp_zcvid_emit_flush(e);
    return e->total_bytes;
}

/* =========================================================================
 * Closed-Loop Edge-Preserving JND Threshold (Encoder-Only, Zero Wire Bits)
 * ========================================================================= */

/**
 * @brief Computes a per-block deadzone threshold from the read-only src buffer
 *        (zero MT data race across worker slices, zero wire overhead).
 *        Distinguishes aligned UI/text edges and 1-2px thin UI lines (keeps
 *        base_thr) from disorderly high-frequency textures (high energy with
 *        low coherence -> raises threshold up to base_thr + jnd_boost).
 */
static inline uint8_t zcrudp_zcvid_block_jnd_threshold(const uint8_t *src,
                                                         size_t base,
                                                         size_t len,
                                                         uint16_t width,
                                                         size_t lstep,
                                                         uint8_t base_thr,
                                                         uint8_t jnd_boost) {
    if (jnd_boost == 0U || base + 10U > len) {
        return base_thr;
    }
    if (width > 0U && (base % (size_t)width) + 10U > (size_t)width) {
        return base_thr;
    }
    int32_t gx_sum = 0, gy_sum = 0;
    uint32_t gx_abs = 0, gy_abs = 0;
    uint32_t nz_gx = 0, nz_gy = 0;
    if (lstep == 0U) lstep = 1U;
    for (size_t j = 0U; j + lstep < 10U; j++) {
        int32_t g = (int32_t)src[base + j + lstep] - (int32_t)src[base + j];
        uint32_t ag = (uint32_t)(g < 0 ? -g : g);
        gx_sum += g;
        gx_abs += ag;
        if (ag > (uint32_t)base_thr) nz_gx++;
    }
    if (width > 0U && base >= (size_t)width) {
        for (size_t j = 0U; j < 10U; j++) {
            int32_t g = (int32_t)src[base + j] - (int32_t)src[base + j - width];
            uint32_t ag = (uint32_t)(g < 0 ? -g : g);
            gy_sum += g;
            gy_abs += ag;
            if (ag > (uint32_t)base_thr) nz_gy++;
        }
    }
    uint32_t energy = gx_abs + gy_abs;
    uint32_t coh = (uint32_t)(gx_sum < 0 ? -gx_sum : gx_sum) +
                   (uint32_t)(gy_sum < 0 ? -gy_sum : gy_sum);
    /* Protect 1-2px thin UI lines (sparse high-contrast transitions) and coherent edges. */
    if ((nz_gx + nz_gy) < 8U || energy < 80U || energy <= 2U * coh) {
        return base_thr;
    }
    uint32_t disorder = energy - 2U * coh;
    uint32_t add = disorder >> 6;
    if (add > (uint32_t)jnd_boost) add = (uint32_t)jnd_boost;
    uint32_t eff = (uint32_t)base_thr + add;
    return (eff > 255U) ? 255U : (uint8_t)eff;
}

/* =========================================================================
 * Plane Encoder (MTU-Packetized, Zero Blocks Skipped)
 * ========================================================================= */

/**
 * @brief Encodes a sub-range of 10-pixel blocks [blk_begin, blk_end) of a plane
 *        into MTU-sized UDP packets. Ideal for lock-free multi-threaded slicing.
 *        pkt_buf must hold cfg->mtu_payload bytes.
 */
static inline size_t zcrudp_zcvid_encode_plane_range(const uint8_t * ZCRUDP_RESTRICT src,
                                                       uint8_t * ZCRUDP_RESTRICT ref,
                                                       uint8_t * ZCRUDP_RESTRICT delta_buf,
                                                       size_t len,
                                                       size_t blk_begin,
                                                       size_t blk_end,
                                                       uint8_t plane_id,
                                                       uint16_t frame_id,
                                                       const zcrudp_zcvid_config_t *cfg,
                                                       uint8_t *pkt_buf,
                                                       zcrudp_zcvid_pkt_cb cb,
                                                       void *user_ctx) {
    const size_t total_blocks = (len + 9U) / 10U;
    if (blk_end > total_blocks) {
        blk_end = total_blocks;
    }

    zcrudp_zcvid_emit_t em;
    zcrudp_zcvid_emit_init(&em, cfg, len, plane_id, frame_id, false, pkt_buf, cb, user_ctx);

#ifdef __AVX2__
    const uint32_t max_dist = (cfg->neutral_y > (uint8_t)(255U - cfg->neutral_y))
        ? (uint32_t)cfg->neutral_y : (uint32_t)(255U - cfg->neutral_y);
    const uint32_t max_decay_shift = cfg->no_decay
        ? 0U : ((max_dist * (256U - (uint32_t)cfg->gamma_q8) + 255U) >> 8);
    const bool can_fast_skip = ((uint32_t)cfg->threshold >= max_decay_shift);
    const uint8_t fast_skip_thr = can_fast_skip
        ? (uint8_t)((uint32_t)cfg->threshold - max_decay_shift) : 0U;
#endif

    for (size_t blk = blk_begin; blk < blk_end; blk++) {
        size_t base = blk * 10U;

#ifdef __AVX2__
        /* Stationary-region skip: 80 pixels (8 blocks) at a time. A skipped
         * block only decays, with or without ADM (its steps are kept).
         * With PDQ, |src - dec| <= |src - ref| + max_decay_shift <= threshold
         * whenever |src - ref| <= fast_skip_thr, preserving 100% scalar parity. */
        if (can_fast_skip && base + 80U <= len && (blk + 8U) <= blk_end) {
            const __m256i vt = _mm256_set1_epi8((char)fast_skip_thr);

            __m256i vs0 = _mm256_loadu_si256((const __m256i *)(src + base));
            __m256i vr0 = _mm256_loadu_si256((const __m256i *)(ref + base));
            __m256i ad0 = _mm256_or_si256(_mm256_subs_epu8(vs0, vr0), _mm256_subs_epu8(vr0, vs0));

            __m256i vs1 = _mm256_loadu_si256((const __m256i *)(src + base + 32U));
            __m256i vr1 = _mm256_loadu_si256((const __m256i *)(ref + base + 32U));
            __m256i ad1 = _mm256_or_si256(_mm256_subs_epu8(vs1, vr1), _mm256_subs_epu8(vr1, vs1));

            __m128i vs2 = _mm_loadu_si128((const __m128i *)(src + base + 64U));
            __m128i vr2 = _mm_loadu_si128((const __m128i *)(ref + base + 64U));
            __m128i ad2 = _mm_or_si128(_mm_subs_epu8(vs2, vr2), _mm_subs_epu8(vr2, vs2));

            __m256i over01 = _mm256_or_si256(_mm256_subs_epu8(ad0, vt),
                                             _mm256_subs_epu8(ad1, vt));
            __m128i over2  = _mm_subs_epu8(ad2, _mm256_castsi256_si128(vt));

            if (_mm256_testz_si256(over01, over01) &&
                _mm_testz_si128(over2, over2)) {
                if (!cfg->no_decay) {
                    zcrudp_zcvid_decay_plane(ref + base, 80U, cfg->gamma_q8, cfg->neutral_y);
                }
                blk += 7U; /* loop increments +1 -> total +8 */
                continue;
            }
        }
#endif

        zcrudp_zcvid_config_t bcfg = *cfg;
        if (cfg->jnd_boost > 0U) {
            size_t jnd_lstep = zcrudp_zcvid_2x_interleaved(cfg, plane_id) ? 2U : 1U;
            bcfg.threshold = zcrudp_zcvid_block_jnd_threshold(
                src, base, len, cfg->width, jnd_lstep, cfg->threshold, cfg->jnd_boost);
        }

#ifdef __AVX2__
        if ((blk + 1U) < blk_end && base + 16U <= len) {
            /* One whole block per iteration, same arithmetic as
             * zcrudp_zcvid_encode_sample() and zcrudp_adm_update_delta().
             * 16 bytes are loaded, only the 10 of this block are written back
             * (the next 6 may belong to another worker's slice). */
            ZCRUDP_ALIGNAS32 int8_t block_trits[16];
            if (zcrudp_zcvid_encode_block_avx2(src + base, ref + base,
                                                 delta_buf ? delta_buf + base : NULL,
                                                 &bcfg, block_trits)) {
                zcrudp_zcvid_emit_block(&em, blk, block_trits);
            }
            continue;
        }
#endif

        zcrudp_trit_t trits[10] = { 0 };
        bool coded = false;

        for (size_t j = 0; j < 10U; j++) {
            size_t idx = base + j;
            if (idx >= len) {
                break;
            }
            uint8_t d = (delta_buf != NULL) ? delta_buf[idx] : bcfg.delta_min;
            zcrudp_trit_t t = zcrudp_zcvid_encode_sample(src[idx], &ref[idx], d,
                                                           delta_buf != NULL, &bcfg, &coded);
            trits[j] = t;
            if (t != 0) {
                coded = true;
            }
        }

        if (!coded) {
            continue;
        }
        if (delta_buf != NULL) {
            for (size_t j = 0; j < 10U && (base + j) < len; j++) {
                size_t idx = base + j;
                delta_buf[idx] = zcrudp_adm_update_delta(
                    delta_buf[idx], trits[j], cfg->delta_min, cfg->delta_max);
            }
        }
        zcrudp_zcvid_emit_block(&em, blk, trits);
    }

    return zcrudp_zcvid_emit_finish(&em);
}

/**
 * @brief Encodes an entire contiguous 8-bit plane into MTU-sized UDP packets.
 */
static inline size_t zcrudp_zcvid_encode_plane(const uint8_t * ZCRUDP_RESTRICT src,
                                                 uint8_t * ZCRUDP_RESTRICT ref,
                                                 uint8_t * ZCRUDP_RESTRICT delta_buf,
                                                 size_t len,
                                                 uint8_t plane_id,
                                                 uint16_t frame_id,
                                                 const zcrudp_zcvid_config_t *cfg,
                                                 uint8_t *pkt_buf,
                                                 zcrudp_zcvid_pkt_cb cb,
                                                 void *user_ctx) {
    size_t total_blocks = (len + 9U) / 10U;
    return zcrudp_zcvid_encode_plane_range(src, ref, delta_buf, len,
                                             0, total_blocks,
                                             plane_id, frame_id, cfg,
                                             pkt_buf, cb, user_ctx);
}

/* =========================================================================
 * Packet Decoder (Applies Deltas onto Pre-Decayed Canvas)
 * ========================================================================= */

/** Applies the 10 trits of one coded block to the canvas and the ADM steps. */
ZCRUDP_ZCVID_INLINE void zcrudp_zcvid_apply_block(uint8_t * ZCRUDP_RESTRICT canvas,
                                              uint8_t * ZCRUDP_RESTRICT delta_buf,
                                              size_t plane_len,
                                              const zcrudp_zcvid_config_t *cfg,
                                              size_t base,
                                              bool is_2x,
                                              bool il,
                                              const zcrudp_trit_t trits[10]) {
    const size_t pair = zcrudp_zcvid_2x_pair(il);

    for (size_t j = 0; j < 10U; j++) {
        size_t idx = is_2x ? zcrudp_zcvid_2x_idx(base, j, il) : (base + j);
        if (idx >= plane_len) {
            break;
        }
        if (trits[j] != 0) {
            uint8_t d = (delta_buf != NULL) ? delta_buf[idx] : cfg->delta_min;
            int32_t val = (int32_t)canvas[idx] + ((int32_t)trits[j] * (int32_t)d);
            if (val < 0) val = 0;
            if (val > 255) val = 255;
            canvas[idx] = (uint8_t)val;
            if (delta_buf != NULL) {
                delta_buf[idx] = zcrudp_adm_update_delta(
                    d, trits[j], cfg->delta_min, cfg->delta_max);
            }
            if (is_2x && (idx + pair) < plane_len) {
                canvas[idx + pair] = canvas[idx];
                if (delta_buf != NULL) {
                    delta_buf[idx + pair] = delta_buf[idx];
                }
            }
        } else if (delta_buf != NULL) {
            delta_buf[idx] = zcrudp_adm_update_delta(
                delta_buf[idx], 0, cfg->delta_min, cfg->delta_max);
            if (is_2x && (idx + pair) < plane_len) {
                delta_buf[idx + pair] = zcrudp_adm_update_delta(
                    delta_buf[idx + pair], 0, cfg->delta_min, cfg->delta_max);
            }
        }
    }
}

/** Fixed-token payload: 10-trit words and RLE skips. */
ZCRUDP_ZCVID_INLINE bool zcrudp_zcvid_decode_tokens(const uint8_t *payload,
                                                        size_t payload_len,
                                                        const zcrudp_zcvid_pkt_hdr_t *hdr,
                                                        uint8_t * ZCRUDP_RESTRICT canvas,
                                                        uint8_t * ZCRUDP_RESTRICT delta_buf,
                                                        size_t plane_len,
                                                        const zcrudp_zcvid_config_t *cfg,
                                                        bool is_2x,
                                                        bool il) {
    if (payload_len < (size_t)hdr->num_tokens * 2U) {
        return false;
    }
    const size_t span = is_2x ? ZCRUDP_ZCVID_2X_SPAN : 10U;
    const size_t total_blocks = (plane_len + span - 1U) / span;
    size_t blk = hdr->start_block;
    if (blk > total_blocks) {
        return false;
    }

    for (uint16_t t = 0; t < hdr->num_tokens; t++) {
#ifdef __AVX2__
        /* Batch-unpack 16 consecutive non-RLE 1x tokens in parallel via Granlund-Montgomery 243 */
        if (!is_2x && (size_t)t + 16U < (size_t)hdr->num_tokens &&
            blk + 16U <= total_blocks && (blk + 16U) * 10U + 6U <= plane_len) {
            uint16_t batch[16];
            bool all_data = true;
            for (size_t k = 0U; k < 16U; k++) {
                uint16_t w = zcrudp_zcvid_read_u16_le(payload + ((size_t)t + k) * 2U);
                if (w >= ZCRUDP_TRIT_MAX_10P) {
                    all_data = false;
                    break;
                }
                batch[k] = w;
            }
            if (all_data) {
                ZCRUDP_ALIGNAS32 int8_t trits16[16][16];
                zcrudp_zcvid_unpack10_x16_avx2(batch, trits16);
                for (size_t k = 0U; k < 16U; k++) {
                    zcrudp_zcvid_apply_block_avx2(canvas + blk * 10U,
                                                    delta_buf ? delta_buf + blk * 10U : NULL,
                                                    cfg, trits16[k]);
                    blk++;
                }
                t += 15U; /* loop increments +1 -> total +16 */
                continue;
            }
        }
#endif
        uint16_t tok = zcrudp_zcvid_read_u16_le(payload + (size_t)t * 2U);
        if (tok >= ZCRUDP_TRIT_MAX_10P) {
            size_t skip = (size_t)(tok - ZCRUDP_TRIT_MAX_10P) + 1U;
            if (skip > total_blocks - blk) {
                return false;
            }
            blk += skip;
            continue;
        }
        if (blk >= total_blocks) {
            return false;
        }
#ifdef __AVX2__
        if (!is_2x && (t + 1U) < hdr->num_tokens && blk * 10U + 16U <= plane_len) {
            ZCRUDP_ALIGNAS32 int8_t wide[16];
            zcrudp_zcvid_unpack10_avx2(tok, wide);
            zcrudp_zcvid_apply_block_avx2(canvas + blk * 10U,
                                            delta_buf ? delta_buf + blk * 10U : NULL, cfg, wide);
            blk++;
            continue;
        }
#endif
        zcrudp_trit_t trits[10];
        zcrudp_unpack_10trits(tok, trits);
        zcrudp_zcvid_apply_block(canvas, delta_buf, plane_len, cfg,
                                   blk * span, is_2x, il, trits);
        blk++;
    }
    return true;
}

/** Realloc v2 payload: dual-resolution RLE + fused W1/W2 tokens (zero division). */
ZCRUDP_ZCVID_INLINE bool zcrudp_zcvid_decode_rv2(const uint8_t *payload,
                                                     size_t payload_len,
                                                     const zcrudp_zcvid_pkt_hdr_t *hdr,
                                                     uint8_t * ZCRUDP_RESTRICT canvas,
                                                     uint8_t * ZCRUDP_RESTRICT delta_buf,
                                                     size_t plane_len,
                                                     const zcrudp_zcvid_config_t *cfg) {
    static const uint8_t ZCRUDP_RV2_W2_PAIR[45] = {
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09,
        0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19,
        0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29,
        0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
        0x45, 0x46, 0x47, 0x48, 0x49,
        0x56, 0x57, 0x58, 0x59,
        0x67, 0x68, 0x69,
        0x78, 0x79,
        0x89
    };
    if (payload_len < (size_t)hdr->num_tokens * 2U) {
        return false;
    }
    const size_t total_blocks = (plane_len + 9U) / 10U;
    size_t blk = hdr->start_block;
    if (blk > total_blocks) return false;

    for (uint16_t t = 0; t < hdr->num_tokens; t++) {
        uint16_t tok = zcrudp_zcvid_read_u16_le(payload + (size_t)t * 2U);
        if (tok < ZCRUDP_TRIT_MAX_10P) {
            if (blk >= total_blocks) return false;
#ifdef __AVX2__
            if ((t + 1U) < hdr->num_tokens && blk * 10U + 16U <= plane_len) {
                ZCRUDP_ALIGNAS32 int8_t wide[16];
                zcrudp_zcvid_unpack10_avx2(tok, wide);
                zcrudp_zcvid_apply_block_avx2(canvas + blk * 10U,
                                                delta_buf ? delta_buf + blk * 10U : NULL, cfg, wide);
                blk++;
                continue;
            }
#endif
            zcrudp_trit_t trits[10];
            zcrudp_unpack_10trits(tok, trits);
            zcrudp_zcvid_apply_block(canvas, delta_buf, plane_len, cfg,
                                       blk * 10U, false, false, trits);
            blk++;
            continue;
        }
        if (tok <= 59432U) {
            size_t skip = (size_t)(tok - 59049U) + 1U;
            if (skip > total_blocks - blk) return false;
            blk += skip;
            continue;
        }
        if (tok <= 59439U) {
            size_t skip = (size_t)(512U << (tok - 59433U));
            if (skip > total_blocks - blk) return false;
            blk += skip;
            continue;
        }
        zcrudp_trit_t trits[10] = { 0 };
        if (tok <= 61999U) {
            uint32_t off = (uint32_t)(tok - 59440U);
            uint32_t pat = off >> 7;
            size_t skip = (size_t)((off & 127U) + 1U);
            if (skip >= total_blocks - blk) return false;
            blk += skip;
            trits[pat >> 1] = (pat & 1U) ? (zcrudp_trit_t)-1 : (zcrudp_trit_t)1;
        } else if (tok <= 62639U) {
            uint32_t off = (uint32_t)(tok - 62000U);
            uint32_t pat = off >> 5;
            size_t skip = (size_t)(129U + (off & 31U));
            if (skip >= total_blocks - blk) return false;
            blk += skip;
            trits[pat >> 1] = (pat & 1U) ? (zcrudp_trit_t)-1 : (zcrudp_trit_t)1;
        } else if (tok <= 65519U) {
            uint32_t off = (uint32_t)(tok - 62640U);
            uint32_t pat = off >> 4;
            size_t skip = (size_t)((off & 15U) + 1U);
            if (skip >= total_blocks - blk) return false;
            blk += skip;
            uint8_t pair = ZCRUDP_RV2_W2_PAIR[pat >> 2];
            trits[pair >> 4]  = (pat & 2U) ? (zcrudp_trit_t)-1 : (zcrudp_trit_t)1;
            trits[pair & 15U] = (pat & 1U) ? (zcrudp_trit_t)-1 : (zcrudp_trit_t)1;
        } else {
            return false;
        }
        zcrudp_zcvid_apply_block(canvas, delta_buf, plane_len, cfg,
                                   blk * 10U, false, false, trits);
        blk++;
    }
    return true;
}

/** Range-coded payload: hdr->num_tokens non-zero blocks separated by zero runs. */
ZCRUDP_ZCVID_INLINE bool zcrudp_zcvid_decode_ec(const uint8_t *payload,
                                                    size_t payload_len,
                                                    const zcrudp_zcvid_pkt_hdr_t *hdr,
                                                    uint8_t * ZCRUDP_RESTRICT canvas,
                                                    uint8_t * ZCRUDP_RESTRICT delta_buf,
                                                    size_t plane_len,
                                                    const zcrudp_zcvid_config_t *cfg,
                                                    bool is_2x,
                                                    bool il) {
    const size_t span = is_2x ? ZCRUDP_ZCVID_2X_SPAN : 10U;
    const size_t total_blocks = (plane_len + span - 1U) / span;
    zcrudp_rc_dec_t rc;
    zcrudp_tec_model_t model;
    zcrudp_rc_dec_init(&rc, payload, payload_len);
    zcrudp_tec_model_reset(&model, cfg->width,
                           zcrudp_zcvid_ec_lstep(cfg, hdr->plane, is_2x));

    size_t blk = hdr->start_block;
    if (blk > total_blocks) {
        return false;
    }
    for (uint16_t t = 0; t < hdr->num_tokens; t++) {
        if (t > 0U) {
            uint64_t run = zcrudp_tec_dec_run(&rc, &model);
            if (run >= (uint64_t)(total_blocks - blk)) {
                return false;
            }
            zcrudp_tec_hist_clear(&model, blk * span, (blk + (size_t)run) * span);
            blk += (size_t)run;
        }
        if (blk >= total_blocks) {
            return false;
        }
#ifdef __AVX2__
        ZCRUDP_ALIGNAS32 zcrudp_trit_t trits[16];
#else
        zcrudp_trit_t trits[10];
#endif
        zcrudp_zcvid_ec_get_block(&rc, &model, blk * span, plane_len, is_2x, il, trits);
#ifdef __AVX2__
        if (!is_2x && (t + 1U) < hdr->num_tokens && blk * 10U + 16U <= plane_len) {
            zcrudp_zcvid_apply_block_avx2(canvas + blk * 10U,
                                            delta_buf ? delta_buf + blk * 10U : NULL, cfg, trits);
            blk++;
            continue;
        }
#endif
        zcrudp_zcvid_apply_block(canvas, delta_buf, plane_len, cfg,
                                   blk * span, is_2x, il, trits);
        blk++;
    }
    return true;
}

/** 4-way interleaved static rANS payload: [num_nz][num_syms][skip_len][skip_mask][skips][signs][rANS]. */
ZCRUDP_ZCVID_INLINE bool zcrudp_zcvid_decode_rans(const uint8_t *payload,
                                                      size_t payload_len,
                                                      const zcrudp_zcvid_pkt_hdr_t *hdr,
                                                      uint8_t * ZCRUDP_RESTRICT canvas,
                                                      uint8_t * ZCRUDP_RESTRICT delta_buf,
                                                      size_t plane_len,
                                                      const zcrudp_zcvid_config_t *cfg) {
    if (payload_len < 6U + 17U) {
        return false;
    }
    const size_t k = (size_t)hdr->num_tokens;
    if (k == 0U || k > ZCRUDP_RANS_MAX_BLOCKS) {
        return false;
    }
    const size_t num_nz   = (size_t)zcrudp_zcvid_read_u16_le(payload + 0U);
    const size_t num_syms = (size_t)zcrudp_zcvid_read_u16_le(payload + 2U);
    const size_t skip_len = (size_t)zcrudp_zcvid_read_u16_le(payload + 4U);
    const size_t mask_len = (k + 7U) / 8U;
    const size_t sign_len = (num_nz + 7U) / 8U;
    const size_t hdr_len  = 6U + mask_len + skip_len + sign_len;

    if (num_syms > ZCRUDP_RANS_MAX_SYMS || num_nz > k * 10U || hdr_len + 17U > payload_len) {
        return false;
    }

    const uint8_t *skip_mask = payload + 6U;
    const uint8_t *skip_vals = skip_mask + mask_len;
    const uint8_t *sign_buf  = skip_vals + skip_len;
    const uint8_t *rans_in   = payload + hdr_len;
    const size_t   rans_len  = payload_len - hdr_len;

    uint8_t syms[ZCRUDP_RANS_MAX_SYMS];
    if (!zcrudp_rans_decode_syms(rans_in, rans_len, syms, num_syms)) {
        return false;
    }

#ifdef __AVX2__
    ZCRUDP_ALIGNAS32 zcrudp_trit_t trits[ZCRUDP_RANS_MAX_BLOCKS * 10U + 16U];
#else
    zcrudp_trit_t trits[ZCRUDP_RANS_MAX_BLOCKS * 10U + 16U];
#endif
    const size_t total_trits = k * 10U;
    memset(trits, 0, total_trits);

    size_t pos = 0U;
    size_t nz_idx = 0U;
    for (size_t i = 0U; i < num_syms; i++) {
        uint8_t s = syms[i];
        if (s == 15U) {
            pos += 128U;
        } else if (s == 14U) {
            pos += 14U;
        } else {
            pos += (size_t)s;
            if (pos >= total_trits || nz_idx >= num_nz) {
                return false;
            }
            uint8_t neg = (uint8_t)((sign_buf[nz_idx >> 3] >> (nz_idx & 7U)) & 1U);
            trits[pos] = neg ? (zcrudp_trit_t)-1 : (zcrudp_trit_t)1;
            pos++;
            nz_idx++;
        }
    }
    if (nz_idx != num_nz) {
        return false;
    }

    const size_t total_blocks = (plane_len + 9U) / 10U;
    size_t blk = hdr->start_block;
    if (blk > total_blocks) return false;
    size_t s_off = 0U;

    for (size_t bi = 0U; bi < k; bi++) {
        if (bi > 0U) {
            if ((skip_mask[bi >> 3] & (1U << (bi & 7U))) != 0U) {
                if (s_off >= skip_len) return false;
                uint32_t sk = skip_vals[s_off++];
                if (sk == 255U) {
                    if (s_off + 2U > skip_len) return false;
                    sk = (uint32_t)zcrudp_zcvid_read_u16_le(skip_vals + s_off);
                    s_off += 2U;
                }
                if ((size_t)sk >= total_blocks - blk) {
                    return false;
                }
                blk += (size_t)sk;
            }
        }
        if (blk >= total_blocks) {
            return false;
        }
#ifdef __AVX2__
        if ((bi + 1U) < k && blk * 10U + 16U <= plane_len) {
            zcrudp_zcvid_apply_block_avx2(canvas + blk * 10U,
                                            delta_buf ? delta_buf + blk * 10U : NULL,
                                            cfg, trits + bi * 10U);
            blk++;
            continue;
        }
#endif
        zcrudp_zcvid_apply_block(canvas, delta_buf, plane_len, cfg,
                                   blk * 10U, false, false, trits + bi * 10U);
        blk++;
    }
    return s_off == skip_len;
}

/**
 * @brief Decodes a single zcvid video packet onto a plane canvas.
 *        Assumes zcrudp_zcvid_decay_plane() was already called for this frame
 *        (so skipped blocks already have their leaky decay applied).
 *
 *        Handles DELTA, DELTA_2X, INTRA and their range-coded variants. Other
 *        types (TILE_MV, unknown) are rejected and leave the canvas untouched.
 *        Every write stays inside [canvas, canvas + plane_len).
 *
 * @return true if packet was valid and decoded, false if malformed/out-of-bounds.
 */
static inline bool zcrudp_zcvid_decode_packet(const uint8_t *pkt_data,
                                                size_t pkt_len,
                                                uint8_t * ZCRUDP_RESTRICT canvas,
                                                uint8_t * ZCRUDP_RESTRICT delta_buf,
                                                size_t plane_len,
                                                const zcrudp_zcvid_config_t *cfg,
                                                zcrudp_zcvid_pkt_hdr_t *out_hdr) {
    if (pkt_len < ZCRUDP_ZCVID_PKT_HDR_SZ) {
        return false;
    }
    zcrudp_zcvid_pkt_hdr_t hdr;
    if (!zcrudp_zcvid_pkt_hdr_read_le(pkt_data, &hdr)) {
        return false;
    }
    uint8_t ptype = ZCRUDP_ZCVID_HDR_TYPE(&hdr);
    if (out_hdr != NULL) {
        *out_hdr = hdr;
    }
    const uint8_t *payload = pkt_data + ZCRUDP_ZCVID_PKT_HDR_SZ;
    const size_t payload_len = pkt_len - ZCRUDP_ZCVID_PKT_HDR_SZ;

    if (ptype == ZCRUDP_ZCVID_PKT_INTRA) {
        /* num_tokens = byte length of raw anchor slice; start_block = byte offset */
        size_t byte_off = (size_t)hdr.start_block;
        size_t byte_len = (size_t)hdr.num_tokens;
        if (payload_len < byte_len || byte_off >= plane_len) {
            return false;
        }
        if (byte_len > plane_len - byte_off) {
            byte_len = plane_len - byte_off;
        }
        memcpy(canvas + byte_off, payload, byte_len);
        if (delta_buf != NULL) {
            memset(delta_buf + byte_off, cfg->delta_min, byte_len);
        }
        return true;
    }

    if (ptype == ZCRUDP_ZCVID_PKT_INTRA_EC || ptype == ZCRUDP_ZCVID_PKT_INTRA_Q_EC) {
        /* num_tokens = sample count; start_block = byte offset. Prediction only
         * uses samples rebuilt from this very packet. */
        size_t off = (size_t)hdr.start_block;
        size_t n = (size_t)hdr.num_tokens;
        if (off >= plane_len) {
            return false;
        }
        if (n > plane_len - off) {
            n = plane_len - off;
        }
        uint8_t qstep = 1U;
        const uint8_t *rc_in = payload;
        size_t rc_len = payload_len;
        if (ptype == ZCRUDP_ZCVID_PKT_INTRA_Q_EC) {
            /* The step alone is a whole packet: the encoder drops trailing
             * zero bytes of the range-coded stream, and a slice equal to its
             * prediction codes to nothing but zeros. */
            if (payload_len < 1U) {
                return false;
            }
            qstep = payload[0];
            if (qstep < 2U || qstep > 64U) {
                return false;
            }
            rc_in = payload + 1U;
            rc_len = payload_len - 1U;
        }
        const size_t lstep = zcrudp_zcvid_2x_interleaved(cfg, hdr.plane) ? 2U : 1U;
        const size_t stride = cfg->width;
        zcrudp_rc_dec_t rc;
        zcrudp_iec_model_t model;
        zcrudp_rc_dec_init(&rc, rc_in, rc_len);
        zcrudp_iec_model_reset(&model);
        int32_t prev_res[2] = { 0, 0 };

        for (size_t i = 0; i < n; i++) {
            size_t idx = off + i;
            int32_t pr = prev_res[i & (lstep - 1U)];
            uint32_t act = (uint32_t)(pr < 0 ? -pr : pr) * 2U;
            uint8_t pred = (i >= lstep) ? canvas[idx - lstep] : (uint8_t)128;
            if (stride != 0U && i >= stride + lstep) {
                uint8_t a = pred;
                uint8_t b = canvas[idx - stride];
                uint8_t c = canvas[idx - stride - lstep];
                pred = zcrudp_iec_med(a, b, c);
                act = (uint32_t)(a > c ? a - c : c - a) + (uint32_t)(b > c ? b - c : c - b);
            }
            uint32_t sctx = (pr == 0) ? 0U : (pr > 0) ? 1U : 2U;
            int32_t res = zcrudp_iec_dec_residual(&rc, &model, zcrudp_iec_bucket(act), sctx);
            if (qstep == 1U) {
                canvas[idx] = (uint8_t)((int32_t)pred + res);
                prev_res[i & (lstep - 1U)] = (int32_t)(int8_t)(uint8_t)res;
            } else {
                int32_t rec = (int32_t)pred + res * (int32_t)qstep;
                if (rec < 0) rec = 0;
                if (rec > 255) rec = 255;
                canvas[idx] = (uint8_t)rec;
                prev_res[i & (lstep - 1U)] = res;
            }
        }
        if (delta_buf != NULL) {
            memset(delta_buf + off, cfg->delta_min, n);
        }
        return true;
    }

    const bool il = zcrudp_zcvid_2x_interleaved(cfg, hdr.plane);
    switch (ptype) {
        case ZCRUDP_ZCVID_PKT_DELTA:
            return zcrudp_zcvid_decode_tokens(payload, payload_len, &hdr, canvas, delta_buf,
                                                plane_len, cfg, false, false);
        case ZCRUDP_ZCVID_PKT_DELTA_2X:
            return zcrudp_zcvid_decode_tokens(payload, payload_len, &hdr, canvas, delta_buf,
                                                plane_len, cfg, true, il);
        case ZCRUDP_ZCVID_PKT_DELTA_EC:
            return zcrudp_zcvid_decode_ec(payload, payload_len, &hdr, canvas, delta_buf,
                                            plane_len, cfg, false, false);
        case ZCRUDP_ZCVID_PKT_DELTA_2X_EC:
            return zcrudp_zcvid_decode_ec(payload, payload_len, &hdr, canvas, delta_buf,
                                            plane_len, cfg, true, il);
        case ZCRUDP_ZCVID_PKT_DELTA_RANS:
            return zcrudp_zcvid_decode_rans(payload, payload_len, &hdr, canvas, delta_buf,
                                              plane_len, cfg);
        case ZCRUDP_ZCVID_PKT_DELTA_RV2:
            return zcrudp_zcvid_decode_rv2(payload, payload_len, &hdr, canvas, delta_buf,
                                             plane_len, cfg);
        default:
            return false;
    }
}

/**
 * @brief Single-pass fused plane decoder: decays skipped gaps once as packets
 *        advance instead of running a full-plane decay pass before decoding.
 *        Packets for a plane must be in ascending start_block order for deltas,
 *        followed by any INTRA anchors. Bit-exact with decay_plane + decode_packet.
 */
static inline bool zcrudp_zcvid_decode_packets_fused(const uint8_t * const *pkts,
                                                         const size_t *lens,
                                                         size_t count,
                                                         uint8_t * ZCRUDP_RESTRICT canvas,
                                                         uint8_t * ZCRUDP_RESTRICT delta_buf,
                                                         size_t plane_len,
                                                         const zcrudp_zcvid_config_t *cfg) {
    size_t decayed_until = cfg->no_decay ? plane_len : 0U;

    for (size_t i = 0U; i < count; i++) {
        zcrudp_zcvid_pkt_hdr_t hdr;
        if (lens[i] < ZCRUDP_ZCVID_PKT_HDR_SZ ||
            !zcrudp_zcvid_pkt_hdr_read_le(pkts[i], &hdr)) {
            return false;
        }
        uint8_t ptype = ZCRUDP_ZCVID_HDR_TYPE(&hdr);
        bool is_anchor = (ptype == ZCRUDP_ZCVID_PKT_INTRA ||
                          ptype == ZCRUDP_ZCVID_PKT_INTRA_EC ||
                          ptype == ZCRUDP_ZCVID_PKT_INTRA_Q_EC);
        if (is_anchor) {
            if (decayed_until < plane_len) {
                zcrudp_zcvid_decay_plane(canvas + decayed_until, plane_len - decayed_until,
                                           cfg->gamma_q8, cfg->neutral_y);
                decayed_until = plane_len;
            }
            if (!zcrudp_zcvid_decode_packet(pkts[i], lens[i], canvas, delta_buf,
                                              plane_len, cfg, NULL)) {
                return false;
            }
            continue;
        }

        size_t span = (ptype == ZCRUDP_ZCVID_PKT_DELTA_2X ||
                       ptype == ZCRUDP_ZCVID_PKT_DELTA_2X_EC)
            ? ZCRUDP_ZCVID_2X_SPAN : 10U;
        size_t start_px = (size_t)hdr.start_block * span;
        if (start_px > plane_len) {
            return false;
        }

        /* Decay any un-decayed gap up to start_px, then decay through this packet's
         * span only once before applying its deltas. For the last packet or full
         * frames, this fuses the decay into the active working set in L1/L2 cache. */
        size_t next_start = plane_len;
        if (i + 1U < count) {
            zcrudp_zcvid_pkt_hdr_t nh;
            if (lens[i + 1U] >= ZCRUDP_ZCVID_PKT_HDR_SZ &&
                zcrudp_zcvid_pkt_hdr_read_le(pkts[i + 1U], &nh)) {
                uint8_t ntype = ZCRUDP_ZCVID_HDR_TYPE(&nh);
                if (ntype != ZCRUDP_ZCVID_PKT_INTRA &&
                    ntype != ZCRUDP_ZCVID_PKT_INTRA_EC &&
                    ntype != ZCRUDP_ZCVID_PKT_INTRA_Q_EC) {
                    size_t nspan = (ntype == ZCRUDP_ZCVID_PKT_DELTA_2X ||
                                    ntype == ZCRUDP_ZCVID_PKT_DELTA_2X_EC)
                        ? ZCRUDP_ZCVID_2X_SPAN : 10U;
                    size_t np = (size_t)nh.start_block * nspan;
                    if (np <= plane_len && np >= decayed_until) {
                        next_start = np;
                    }
                }
            }
        }
        if (next_start > decayed_until) {
            zcrudp_zcvid_decay_plane(canvas + decayed_until, next_start - decayed_until,
                                       cfg->gamma_q8, cfg->neutral_y);
            decayed_until = next_start;
        }
        if (!zcrudp_zcvid_decode_packet(pkts[i], lens[i], canvas, delta_buf,
                                          plane_len, cfg, NULL)) {
            return false;
        }
    }
    if (decayed_until < plane_len) {
        zcrudp_zcvid_decay_plane(canvas + decayed_until, plane_len - decayed_until,
                                   cfg->gamma_q8, cfg->neutral_y);
    }
    return true;
}

/* =========================================================================
 * Rolling Intra-Refresh (Zero-Drift Anchor Slices) & 2x High-Motion Mode
 * ========================================================================= */

/**
 * @brief Emits anchor packets for byte range [byte_begin, byte_end) of a plane,
 *        resetting encoder ref & delta_buf to exact source values.
 *        Eliminates HUD/text wash-out and bounds worst-case drift to N frames.
 *
 *        With ZCRUDP_ZCVID_EC_INTRA the samples are predicted from their
 *        neighbours inside the same packet and the error is range-coded. The
 *        result is still exact; a packet that would not shrink is sent raw.
 *
 *        Frame order: emit the anchors of a frame AFTER its delta pass. The
 *        receiver decays the plane once, before any packet, while the delta
 *        pass of the encoder decays every sample it visits. An anchor written
 *        before the delta pass is therefore decayed on the encoder only, and
 *        the two references drift apart unless gamma_q8 is 255.
 */
static inline size_t zcrudp_zcvid_encode_intra_range(const uint8_t * ZCRUDP_RESTRICT src,
                                                       uint8_t * ZCRUDP_RESTRICT ref,
                                                       uint8_t * ZCRUDP_RESTRICT delta_buf,
                                                       size_t byte_begin,
                                                       size_t byte_end,
                                                       size_t plane_len,
                                                       uint8_t plane_id,
                                                       uint16_t frame_id,
                                                       const zcrudp_zcvid_config_t *cfg,
                                                       uint8_t *pkt_buf,
                                                       zcrudp_zcvid_pkt_cb cb,
                                                       void *user_ctx) {
    if (byte_end > plane_len) byte_end = plane_len;
    if (byte_begin >= byte_end || cfg->mtu_payload <= ZCRUDP_ZCVID_PKT_HDR_SZ + 1U) return 0;

    size_t max_payload = (size_t)(cfg->mtu_payload - ZCRUDP_ZCVID_PKT_HDR_SZ);
    if (max_payload > 65535U) max_payload = 65535U;

    const uint8_t qstep = (cfg->intra_qstep > 1U)
        ? ((cfg->intra_qstep <= 64U) ? cfg->intra_qstep : 64U)
        : 1U;
    const size_t hdr_extra = (qstep > 1U) ? 1U : 0U;
    const bool ec = (cfg->entropy & ZCRUDP_ZCVID_EC_INTRA) != 0U &&
                    cfg->mtu_payload > ZCRUDP_ZCVID_PKT_HDR_SZ &&
                    max_payload >= hdr_extra + 2U * ZCRUDP_IEC_SAMPLE_MARGIN;
    const size_t lstep = zcrudp_zcvid_2x_interleaved(cfg, plane_id) ? 2U : 1U;
    const size_t stride = cfg->width;
    uint8_t *payload = pkt_buf + ZCRUDP_ZCVID_PKT_HDR_SZ;

    size_t total_bytes = 0;
    size_t off = byte_begin;
    while (off < byte_end) {
        size_t chunk = byte_end - off;
        if (chunk > max_payload) chunk = max_payload;
        uint8_t ptype = ZCRUDP_ZCVID_PKT_INTRA;
        size_t payload_len = chunk;

        if (ec) {
            if (qstep > 1U) {
                payload[0] = qstep;
            }
            zcrudp_rc_enc_t rc;
            zcrudp_iec_model_t model;
            zcrudp_rc_enc_init(&rc, payload + hdr_extra, max_payload - hdr_extra);
            zcrudp_iec_model_reset(&model);
            int32_t prev_res[2] = { 0, 0 };
            size_t limit = byte_end - off;
            if (limit > 65535U) limit = 65535U;
            size_t n = 0;

            for (; n < limit; n++) {
                if (zcrudp_rc_enc_pending(&rc) + ZCRUDP_IEC_SAMPLE_MARGIN > max_payload - hdr_extra) {
                    break;
                }
                size_t idx = off + n;
                int32_t pr = prev_res[n & (lstep - 1U)];
                uint32_t act = (uint32_t)(pr < 0 ? -pr : pr) * 2U;
                uint8_t pred = (n >= lstep) ? ref[idx - lstep] : (uint8_t)128;
                if (stride != 0U && n >= stride + lstep) {
                    uint8_t a = pred;
                    uint8_t b = ref[idx - stride];
                    uint8_t c = ref[idx - stride - lstep];
                    pred = zcrudp_iec_med(a, b, c);
                    act = (uint32_t)(a > c ? a - c : c - a) + (uint32_t)(b > c ? b - c : c - b);
                }
                uint32_t sctx = (pr == 0) ? 0U : (pr > 0) ? 1U : 2U;
                int32_t res;
                uint8_t rec;
                if (qstep == 1U) {
                    res = (int32_t)(int8_t)(uint8_t)(src[idx] - pred);
                    rec = src[idx];
                } else {
                    int32_t err = (int32_t)src[idx] - (int32_t)pred;
                    int32_t q = (int32_t)qstep;
                    res = (err >= 0) ? ((err + (q >> 1)) / q) : -((-err + (q >> 1)) / q);
                    int32_t rval = (int32_t)pred + res * q;
                    if (rval < 0) rval = 0;
                    if (rval > 255) rval = 255;
                    rec = (uint8_t)rval;
                }
                zcrudp_iec_enc_residual(&rc, &model, zcrudp_iec_bucket(act), sctx, res);
                ref[idx] = rec;
                prev_res[n & (lstep - 1U)] = res;
            }
            size_t coded = hdr_extra + zcrudp_rc_enc_finish(&rc);
            if (coded < n) {
                ptype = (qstep > 1U) ? ZCRUDP_ZCVID_PKT_INTRA_Q_EC : ZCRUDP_ZCVID_PKT_INTRA_EC;
                payload_len = coded;
                chunk = n;
            } else if (n < chunk) {
                /* No gain: send the same samples raw (n <= coded <= max_payload). */
                chunk = n;
                payload_len = n;
            }
        }

        if (ptype == ZCRUDP_ZCVID_PKT_INTRA) {
            memcpy(payload, src + off, chunk);
            memcpy(ref + off, src + off, chunk);
        }
        if (delta_buf != NULL) {
            memset(delta_buf + off, cfg->delta_min, chunk);
        }

        zcrudp_zcvid_pkt_hdr_t hdr = {
            .magic = ZCRUDP_ZCVID_PKT_MAGIC,
            .pixfmt = ZCRUDP_ZCVID_MAKE_PF(cfg->pixfmt, ptype),
            .plane = plane_id,
            .frame_id = frame_id,
            .start_block = (uint32_t)off,
            .num_tokens = (uint16_t)chunk
        };
        zcrudp_zcvid_pkt_hdr_write_le(pkt_buf, &hdr);
        size_t psz = ZCRUDP_ZCVID_PKT_HDR_SZ + payload_len;
        cb(pkt_buf, psz, user_ctx);
        total_bytes += psz;
        off += chunk;
    }
    return total_bytes;
}

/**
 * @brief Encodes a plane in 2x horizontal subsampled delta mode (0.8 bpp max)
 *        for high-entropy / 3D rotation scenes exceeding the bitrate budget.
 *        On the interleaved NV12 chroma plane (plane_id 1) each trit drives two
 *        samples of the same component, see zcrudp_zcvid_2x_interleaved().
 */
static inline size_t zcrudp_zcvid_encode_plane_2x(const uint8_t * ZCRUDP_RESTRICT src,
                                                    uint8_t * ZCRUDP_RESTRICT ref,
                                                    uint8_t * ZCRUDP_RESTRICT delta_buf,
                                                    size_t len,
                                                    uint8_t plane_id,
                                                    uint16_t frame_id,
                                                    const zcrudp_zcvid_config_t *cfg,
                                                    uint8_t *pkt_buf,
                                                    zcrudp_zcvid_pkt_cb cb,
                                                    void *user_ctx) {
    const bool il = zcrudp_zcvid_2x_interleaved(cfg, plane_id);
    const size_t span = ZCRUDP_ZCVID_2X_SPAN;
    const size_t pair = zcrudp_zcvid_2x_pair(il);
    const size_t total_blocks = (len + span - 1U) / span;

    zcrudp_zcvid_emit_t em;
    zcrudp_zcvid_emit_init(&em, cfg, len, plane_id, frame_id, true, pkt_buf, cb, user_ctx);

    for (size_t blk = 0; blk < total_blocks; blk++) {
        size_t base = blk * span;
        zcrudp_trit_t trits[10] = { 0 };
        bool coded = false;

        for (size_t j = 0; j < 10U; j++) {
            size_t idx = zcrudp_zcvid_2x_idx(base, j, il);
            if (idx >= len) break;

            uint8_t d = (delta_buf != NULL) ? delta_buf[idx] : cfg->delta_min;
            zcrudp_trit_t t = zcrudp_zcvid_encode_sample(src[idx], &ref[idx], d,
                                                           delta_buf != NULL, cfg, &coded);
            if (idx + pair < len) {
                if (t != 0) {
                    ref[idx + pair] = ref[idx];
                } else if (!cfg->no_decay) {
                    ref[idx + pair] = zcrudp_leaky_integrator_step(
                        ref[idx + pair], 0, 0, cfg->gamma_q8, cfg->neutral_y);
                }
            }
            trits[j] = t;
            if (t != 0) coded = true;
        }

        if (!coded) {
            continue;
        }
        if (delta_buf != NULL) {
            for (size_t j = 0; j < 10U; j++) {
                size_t idx = zcrudp_zcvid_2x_idx(base, j, il);
                if (idx >= len) break;
                delta_buf[idx] = zcrudp_adm_update_delta(
                    delta_buf[idx], trits[j], cfg->delta_min, cfg->delta_max);
                if (idx + pair < len) {
                    if (trits[j] != 0) {
                        delta_buf[idx + pair] = delta_buf[idx];
                    } else {
                        delta_buf[idx + pair] = zcrudp_adm_update_delta(
                            delta_buf[idx + pair], 0, cfg->delta_min, cfg->delta_max);
                    }
                }
            }
        }
        zcrudp_zcvid_emit_block(&em, blk, trits);
    }

    return zcrudp_zcvid_emit_finish(&em);
}

/* =========================================================================
 * Dual-Format Frame Helpers (NV12 & RGBA)
 * ========================================================================= */

/**
 * @brief Returns total planar buffer size in bytes for a given config.
 *        NV12: 1.5 * W * H, RGBA: 3 * W * H (R, G, B planes), GRAY8: W * H.
 */
static inline size_t zcrudp_zcvid_planar_size(const zcrudp_zcvid_config_t *cfg) {
    size_t y_sz = (size_t)cfg->width * (size_t)cfg->height;
    switch (cfg->pixfmt) {
        case ZCRUDP_ZCVID_FMT_NV12:  return y_sz + (y_sz / 2U);
        case ZCRUDP_ZCVID_FMT_RGBA:  return y_sz * 3U;
        case ZCRUDP_ZCVID_FMT_GRAY8:
        default:                       return y_sz;
    }
}

/**
 * @brief Splits packed RGBA (4 bytes/px) into 3 planar channels (R, G, B).
 *        (Zero-allocation helper for ZCRUDP_ZCVID_FMT_RGBA).
 */
static inline void zcrudp_zcvid_rgba_to_planar(const uint8_t * ZCRUDP_RESTRICT rgba,
                                                 uint8_t * ZCRUDP_RESTRICT planar_rgb,
                                                 size_t num_pixels) {
    uint8_t * ZCRUDP_RESTRICT r_dst = planar_rgb;
    uint8_t * ZCRUDP_RESTRICT g_dst = planar_rgb + num_pixels;
    uint8_t * ZCRUDP_RESTRICT b_dst = planar_rgb + num_pixels * 2U;

    for (size_t i = 0; i < num_pixels; i++) {
        r_dst[i] = rgba[i * 4U + 0U];
        g_dst[i] = rgba[i * 4U + 1U];
        b_dst[i] = rgba[i * 4U + 2U];
    }
}

/**
 * @brief Recombines 3 planar channels (R, G, B) back into packed RGBA (alpha=255).
 */
static inline void zcrudp_zcvid_planar_to_rgba(const uint8_t * ZCRUDP_RESTRICT planar_rgb,
                                                 uint8_t * ZCRUDP_RESTRICT rgba,
                                                 size_t num_pixels) {
    const uint8_t * ZCRUDP_RESTRICT r_src = planar_rgb;
    const uint8_t * ZCRUDP_RESTRICT g_src = planar_rgb + num_pixels;
    const uint8_t * ZCRUDP_RESTRICT b_src = planar_rgb + num_pixels * 2U;

    for (size_t i = 0; i < num_pixels; i++) {
        rgba[i * 4U + 0U] = r_src[i];
        rgba[i * 4U + 1U] = g_src[i];
        rgba[i * 4U + 2U] = b_src[i];
        rgba[i * 4U + 3U] = 255U;
    }
}

/**
 * @brief Encodes a full planar frame (GRAY8, NV12, or planar RGB from RGBA).
 */
static inline size_t zcrudp_zcvid_encode_frame(const uint8_t * ZCRUDP_RESTRICT src_planar,
                                                 uint8_t * ZCRUDP_RESTRICT ref_planar,
                                                 uint8_t * ZCRUDP_RESTRICT delta_planar,
                                                 uint16_t frame_id,
                                                 const zcrudp_zcvid_config_t *cfg,
                                                 uint8_t *pkt_buf,
                                                 zcrudp_zcvid_pkt_cb cb,
                                                 void *user_ctx) {
    size_t y_sz = (size_t)cfg->width * (size_t)cfg->height;
    size_t bytes = 0;

    if (cfg->pixfmt == ZCRUDP_ZCVID_FMT_GRAY8) {
        bytes += zcrudp_zcvid_encode_plane(
            src_planar, ref_planar, delta_planar,
            y_sz, 0, frame_id, cfg, pkt_buf, cb, user_ctx);
    } else if (cfg->pixfmt == ZCRUDP_ZCVID_FMT_NV12) {
        bytes += zcrudp_zcvid_encode_plane(
            src_planar, ref_planar, delta_planar,
            y_sz, 0, frame_id, cfg, pkt_buf, cb, user_ctx);
        bytes += zcrudp_zcvid_encode_plane(
            src_planar + y_sz, ref_planar + y_sz,
            delta_planar ? (delta_planar + y_sz) : NULL,
            y_sz / 2U, 1, frame_id, cfg, pkt_buf, cb, user_ctx);
    } else if (cfg->pixfmt == ZCRUDP_ZCVID_FMT_RGBA) {
        for (uint8_t p = 0; p < 3U; p++) {
            size_t off = (size_t)p * y_sz;
            bytes += zcrudp_zcvid_encode_plane(
                src_planar + off, ref_planar + off,
                delta_planar ? (delta_planar + off) : NULL,
                y_sz, p, frame_id, cfg, pkt_buf, cb, user_ctx);
        }
    }
    return bytes;
}

/* =========================================================================
 * Global Motion Compensation (GMC) for Panoramic / Camera-Pan Scenes
 * ========================================================================= */

/**
 * @brief Estimates global integer translation (dx, dy) in [-search_r, +search_r]
 *        using a sparse subsampled grid (~256 sample points, < 0.02 ms at 1080p).
 */
static inline void zcrudp_zcvid_estimate_gmc(const uint8_t * ZCRUDP_RESTRICT src,
                                               const uint8_t * ZCRUDP_RESTRICT ref,
                                               uint16_t width,
                                               uint16_t height,
                                               int8_t search_r,
                                               int8_t *out_dx,
                                               int8_t *out_dy) {
    const int step_x = (width > 128) ? (width / 16) : 8;
    const int step_y = (height > 128) ? (height / 16) : 8;
    const int margin = (int)search_r + 2;

    uint32_t best_sad = UINT32_MAX;
    int8_t best_dx = 0;
    int8_t best_dy = 0;

    /* Planes too small to hold a single sample row/column inside the search
     * margin would compare nothing and report the first candidate. */
    if (search_r < 0 || (int)width <= 2 * margin || (int)height <= 2 * margin) {
        *out_dx = 0;
        *out_dy = 0;
        return;
    }

    /* Coarse step-2 search; (0, 0) gets a slight bias to avoid needless shifts */
    for (int dy = -search_r; dy <= search_r; dy += 2) {
        for (int dx = -search_r; dx <= search_r; dx += 2) {
            uint32_t sad = 0;
            for (int y = margin; y < (int)height - margin; y += step_y) {
                const uint8_t *sp = src + (size_t)y * width;
                const uint8_t *rp = ref + (size_t)(y - dy) * width - dx;
                for (int x = margin; x < (int)width - margin; x += step_x) {
                    int diff = (int)sp[x] - (int)rp[x];
                    sad += (uint32_t)(diff < 0 ? -diff : diff);
                }
            }
            if (dx == 0 && dy == 0) {
                sad = (sad > 64U) ? (sad - 64U) : 0U;
            }
            if (sad < best_sad) {
                best_sad = sad;
                best_dx = (int8_t)dx;
                best_dy = (int8_t)dy;
            }
        }
    }

    /* Fine +-1 refinement around coarse best */
    int8_t cx = best_dx;
    int8_t cy = best_dy;
    for (int dy = cy - 1; dy <= cy + 1; dy++) {
        if (dy < -search_r || dy > search_r) continue;
        for (int dx = cx - 1; dx <= cx + 1; dx++) {
            if (dx < -search_r || dx > search_r) continue;
            uint32_t sad = 0;
            for (int y = margin; y < (int)height - margin; y += step_y) {
                const uint8_t *sp = src + (size_t)y * width;
                const uint8_t *rp = ref + (size_t)(y - dy) * width - dx;
                for (int x = margin; x < (int)width - margin; x += step_x) {
                    int diff = (int)sp[x] - (int)rp[x];
                    sad += (uint32_t)(diff < 0 ? -diff : diff);
                }
            }
            if (sad < best_sad) {
                best_sad = sad;
                best_dx = (int8_t)dx;
                best_dy = (int8_t)dy;
            }
        }
    }

    *out_dx = best_dx;
    *out_dy = best_dy;
}

/* =========================================================================
 * Per-Tile Motion Vectors (64x64) & Luma-First Dynamic Rate Controller
 * ========================================================================= */

/**
 * @brief Estimates per-tile (dx, dy) motion vectors for a plane.
 *        Static tiles (HUD/UI) stay at (0,0) via zero-motion bias.
 *        out_mvs must hold 2 * num_tiles_x * num_tiles_y int8_t entries.
 */
static inline uint32_t zcrudp_zcvid_tile_sad(const uint8_t * ZCRUDP_RESTRICT src,
                                                 const uint8_t * ZCRUDP_RESTRICT ref,
                                                 uint16_t width,
                                                 uint16_t height,
                                                 int x0, int y0, int x1, int y1,
                                                 int dx, int dy) {
#ifdef __AVX2__
    if (x0 - dx >= 0 && x1 - dx <= (int)width &&
        y0 - dy >= 0 && y1 - dy <= (int)height &&
        (x1 - x0) >= 32) {
        uint64_t sad = 0U;
        uint32_t cnt = (uint32_t)((x1 - x0) * (y1 - y0));
        __m256i acc = _mm256_setzero_si256();
        for (int y = y0; y < y1; y++) {
            const uint8_t *sp = src + (size_t)y * width + (size_t)x0;
            const uint8_t *rp = ref + (size_t)(y - dy) * width + (size_t)(x0 - dx);
            int x = x0;
            for (; x + 32 <= x1; x += 32, sp += 32, rp += 32) {
                __m256i vs = _mm256_loadu_si256((const __m256i *)sp);
                __m256i vr = _mm256_loadu_si256((const __m256i *)rp);
                acc = _mm256_add_epi64(acc, _mm256_sad_epu8(vs, vr));
            }
            for (; x < x1; x++, sp++, rp++) {
                int d = (int)*sp - (int)*rp;
                sad += (uint64_t)(d < 0 ? -d : d);
            }
        }
        uint64_t lanes[4];
        _mm256_storeu_si256((__m256i *)lanes, acc);
        sad += lanes[0] + lanes[1] + lanes[2] + lanes[3];
        return (cnt > 0U) ? (uint32_t)((sad * 256ULL) / cnt) : UINT32_MAX;
    }
#endif
    uint32_t sad = 0;
    uint32_t cnt = 0;
    for (int y = y0; y < y1; y += 2) {
        int sy = y - dy;
        if (sy < 0 || sy >= (int)height) continue;
        const uint8_t *sp = src + (size_t)y * width;
        const uint8_t *rp = ref + (size_t)sy * width;
        for (int x = x0; x < x1; x += 2) {
            int sx = x - dx;
            if (sx < 0 || sx >= (int)width) continue;
            int d = (int)sp[x] - (int)rp[sx];
            sad += (uint32_t)(d < 0 ? -d : d);
            cnt++;
        }
    }
    return (cnt > 0U) ? ((sad * 256U) / cnt) : UINT32_MAX;
}

static inline uint32_t zcrudp_zcvid_tile_sad_pyramid(const uint8_t * ZCRUDP_RESTRICT src,
                                                         const uint8_t * ZCRUDP_RESTRICT ref,
                                                         uint16_t width,
                                                         uint16_t height,
                                                         int x0, int y0, int x1, int y1,
                                                         int dx, int dy,
                                                         int level) {
#ifdef __AVX2__
    if (x0 - dx >= 0 && x1 - dx <= (int)width &&
        y0 - dy >= 0 && y1 - dy <= (int)height &&
        (x1 - x0) >= 32) {
        if (level >= 4 && (y1 - y0) >= 4) {
            uint64_t sad = 0U;
            uint32_t cnt = 0U;
            const __m256i ones8  = _mm256_set1_epi8(1);
            const __m256i ones16 = _mm256_set1_epi16(1);
            __m256i acc = _mm256_setzero_si256();
            for (int y = y0; y + 4 <= y1; y += 4) {
                const uint8_t *sp = src + (size_t)y * width + (size_t)x0;
                const uint8_t *rp = ref + (size_t)(y - dy) * width + (size_t)(x0 - dx);
                for (int x = x0; x + 32 <= x1; x += 32, sp += 32, rp += 32) {
                    __m256i s01 = _mm256_avg_epu8(_mm256_loadu_si256((const __m256i *)sp),
                                                  _mm256_loadu_si256((const __m256i *)(sp + width)));
                    __m256i s23 = _mm256_avg_epu8(_mm256_loadu_si256((const __m256i *)(sp + (size_t)width * 2U)),
                                                  _mm256_loadu_si256((const __m256i *)(sp + (size_t)width * 3U)));
                    __m256i s4  = _mm256_avg_epu8(s01, s23);
                    __m256i r01 = _mm256_avg_epu8(_mm256_loadu_si256((const __m256i *)rp),
                                                  _mm256_loadu_si256((const __m256i *)(rp + width)));
                    __m256i r23 = _mm256_avg_epu8(_mm256_loadu_si256((const __m256i *)(rp + (size_t)width * 2U)),
                                                  _mm256_loadu_si256((const __m256i *)(rp + (size_t)width * 3U)));
                    __m256i r4  = _mm256_avg_epu8(r01, r23);
                    __m256i sh2 = _mm256_srli_epi16(_mm256_maddubs_epi16(s4, ones8), 1);
                    __m256i rh2 = _mm256_srli_epi16(_mm256_maddubs_epi16(r4, ones8), 1);
                    __m256i sh4 = _mm256_srli_epi32(_mm256_madd_epi16(sh2, ones16), 1);
                    __m256i rh4 = _mm256_srli_epi32(_mm256_madd_epi16(rh2, ones16), 1);
                    acc = _mm256_add_epi32(acc, _mm256_abs_epi32(_mm256_sub_epi32(sh4, rh4)));
                    cnt += 8U;
                }
            }
            if (cnt > 0U) {
                uint32_t lanes[8];
                _mm256_storeu_si256((__m256i *)lanes, acc);
                for (int l = 0; l < 8; l++) sad += lanes[l];
                return (uint32_t)((sad * 256ULL) / cnt);
            }
        } else if (level >= 2 && (y1 - y0) >= 2) {
            uint64_t sad = 0U;
            uint32_t cnt = 0U;
            const __m256i ones8 = _mm256_set1_epi8(1);
            __m256i acc = _mm256_setzero_si256();
            for (int y = y0; y + 2 <= y1; y += 2) {
                const uint8_t *sp = src + (size_t)y * width + (size_t)x0;
                const uint8_t *rp = ref + (size_t)(y - dy) * width + (size_t)(x0 - dx);
                for (int x = x0; x + 32 <= x1; x += 32, sp += 32, rp += 32) {
                    __m256i s2  = _mm256_avg_epu8(_mm256_loadu_si256((const __m256i *)sp),
                                                  _mm256_loadu_si256((const __m256i *)(sp + width)));
                    __m256i r2  = _mm256_avg_epu8(_mm256_loadu_si256((const __m256i *)rp),
                                                  _mm256_loadu_si256((const __m256i *)(rp + width)));
                    __m256i sh2 = _mm256_srli_epi16(_mm256_maddubs_epi16(s2, ones8), 1);
                    __m256i rh2 = _mm256_srli_epi16(_mm256_maddubs_epi16(r2, ones8), 1);
                    __m256i df  = _mm256_abs_epi16(_mm256_sub_epi16(sh2, rh2));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(df, _mm256_set1_epi16(1)));
                    cnt += 16U;
                }
            }
            if (cnt > 0U) {
                uint32_t lanes[8];
                _mm256_storeu_si256((__m256i *)lanes, acc);
                for (int l = 0; l < 8; l++) sad += lanes[l];
                return (uint32_t)((sad * 256ULL) / cnt);
            }
        }
    }
#else
    (void)level;
#endif
    return zcrudp_zcvid_tile_sad(src, ref, width, height, x0, y0, x1, y1, dx, dy);
}

static inline void zcrudp_zcvid_estimate_tile_mv_ex(const uint8_t * ZCRUDP_RESTRICT src,
                                                      const uint8_t * ZCRUDP_RESTRICT ref,
                                                      uint16_t width,
                                                      uint16_t height,
                                                      uint16_t tile_sz,
                                                      int8_t search_r,
                                                      const int8_t *prev_mvs,
                                                      int8_t *out_mvs) {
    if (tile_sz == 0U || search_r < 0) {
        return;
    }
    const uint16_t tx_cnt = (uint16_t)((width + tile_sz - 1U) / tile_sz);
    const uint16_t ty_cnt = (uint16_t)((height + tile_sz - 1U) / tile_sz);

    for (uint16_t ty = 0; ty < ty_cnt; ty++) {
        int y0 = (int)ty * (int)tile_sz;
        int y1 = y0 + (int)tile_sz;
        if (y1 > (int)height) y1 = (int)height;

        for (uint16_t tx = 0; tx < tx_cnt; tx++) {
            int x0 = (int)tx * (int)tile_sz;
            int x1 = x0 + (int)tile_sz;
            if (x1 > (int)width) x1 = (int)width;

            uint32_t best_sad = zcrudp_zcvid_tile_sad(src, ref, width, height, x0, y0, x1, y1, 0, 0);
            int8_t best_dx = 0;
            int8_t best_dy = 0;
            size_t tidx = ((size_t)ty * tx_cnt + tx) * 2U;

            if (best_sad <= 32U) {
                out_mvs[tidx + 0U] = 0;
                out_mvs[tidx + 1U] = 0;
                continue;
            }

            /* Temporal predictor from previous frame's MV at the same tile */
            if (prev_mvs != NULL) {
                int pdx = prev_mvs[tidx + 0U];
                int pdy = prev_mvs[tidx + 1U];
                if ((pdx != 0 || pdy != 0) &&
                    pdx >= -search_r && pdx <= search_r &&
                    pdy >= -search_r && pdy <= search_r) {
                    uint32_t psad = zcrudp_zcvid_tile_sad(src, ref, width, height, x0, y0, x1, y1, pdx, pdy);
                    if (psad + 2U < best_sad) {
                        best_sad = psad;
                        best_dx = (int8_t)pdx;
                        best_dy = (int8_t)pdy;
                    }
                }
            }

            /* Spatial predictors from already-estimated left / top neighbor tiles */
            if (tx > 0U) {
                int pdx = out_mvs[tidx - 2U];
                int pdy = out_mvs[tidx - 1U];
                if ((pdx != 0 || pdy != 0) && (pdx != best_dx || pdy != best_dy) &&
                    pdx >= -search_r && pdx <= search_r &&
                    pdy >= -search_r && pdy <= search_r) {
                    uint32_t psad = zcrudp_zcvid_tile_sad(src, ref, width, height, x0, y0, x1, y1, pdx, pdy);
                    if (psad + 2U < best_sad) {
                        best_sad = psad;
                        best_dx = (int8_t)pdx;
                        best_dy = (int8_t)pdy;
                    }
                }
            }
            if (ty > 0U) {
                size_t uidx = tidx - (size_t)tx_cnt * 2U;
                int pdx = out_mvs[uidx + 0U];
                int pdy = out_mvs[uidx + 1U];
                if ((pdx != 0 || pdy != 0) && (pdx != best_dx || pdy != best_dy) &&
                    pdx >= -search_r && pdx <= search_r &&
                    pdy >= -search_r && pdy <= search_r) {
                    uint32_t psad = zcrudp_zcvid_tile_sad(src, ref, width, height, x0, y0, x1, y1, pdx, pdy);
                    if (psad + 2U < best_sad) {
                        best_sad = psad;
                        best_dx = (int8_t)pdx;
                        best_dy = (int8_t)pdy;
                    }
                }
            }

            /* Coarse-to-fine box-filtered pyramid search unless a predictor already locked on */
            if (best_sad > 1024U) {
                int step = 1;
                while (step * 2 <= (int)search_r) {
                    step *= 2;
                }
                for (; step >= 2; step >>= 1) {
                    int pyr_lvl = (step >= 4) ? 4 : 2;
                    int cx = best_dx;
                    int cy = best_dy;
                    uint32_t cur_pyr_sad = zcrudp_zcvid_tile_sad_pyramid(
                        src, ref, width, height, x0, y0, x1, y1, cx, cy, pyr_lvl);
                    for (int oy = -1; oy <= 1; oy++) {
                        for (int ox = -1; ox <= 1; ox++) {
                            if (ox == 0 && oy == 0) continue;
                            int dx = cx + ox * step;
                            int dy = cy + oy * step;
                            if (dx < -search_r || dx > search_r || dy < -search_r || dy > search_r) {
                                continue;
                            }
                            uint32_t norm_sad = zcrudp_zcvid_tile_sad_pyramid(
                                src, ref, width, height, x0, y0, x1, y1, dx, dy, pyr_lvl);
                            if (norm_sad != UINT32_MAX && norm_sad + 2U < cur_pyr_sad) {
                                cur_pyr_sad = norm_sad;
                                best_dx = (int8_t)dx;
                                best_dy = (int8_t)dy;
                            }
                        }
                    }
                }
                best_sad = zcrudp_zcvid_tile_sad(
                    src, ref, width, height, x0, y0, x1, y1, best_dx, best_dy);
            }

            /* Fine +-1 refinement */
            int8_t cx = best_dx;
            int8_t cy = best_dy;
            for (int dy = cy - 1; dy <= cy + 1; dy++) {
                if (dy < -search_r || dy > search_r) continue;
                for (int dx = cx - 1; dx <= cx + 1; dx++) {
                    if (dx < -search_r || dx > search_r) continue;
                    if (dx == cx && dy == cy) continue;
                    uint32_t norm_sad = zcrudp_zcvid_tile_sad(
                        src, ref, width, height, x0, y0, x1, y1, dx, dy);
                    if (norm_sad < best_sad) {
                        best_sad = norm_sad;
                        best_dx = (int8_t)dx;
                        best_dy = (int8_t)dy;
                    }
                }
            }

            out_mvs[tidx + 0U] = best_dx;
            out_mvs[tidx + 1U] = best_dy;
        }
    }
}

static inline void zcrudp_zcvid_estimate_tile_mv(const uint8_t * ZCRUDP_RESTRICT src,
                                                   const uint8_t * ZCRUDP_RESTRICT ref,
                                                   uint16_t width,
                                                   uint16_t height,
                                                   uint16_t tile_sz,
                                                   int8_t search_r,
                                                   int8_t *out_mvs) {
    zcrudp_zcvid_estimate_tile_mv_ex(src, ref, width, height, tile_sz, search_r, NULL, out_mvs);
}

/**
 * @brief Applies per-tile (dx, dy) motion compensation from src_prev into dst_mc,
 *        and optionally shifts delta_prev into delta_mc in lockstep.
 *        src_prev/dst_mc and delta_prev/delta_mc must not alias.
 */
static inline void zcrudp_zcvid_apply_tile_mv_state(const uint8_t * ZCRUDP_RESTRICT src_prev,
                                                      uint8_t * ZCRUDP_RESTRICT dst_mc,
                                                      const uint8_t * ZCRUDP_RESTRICT delta_prev,
                                                      uint8_t * ZCRUDP_RESTRICT delta_mc,
                                                      uint16_t width,
                                                      uint16_t height,
                                                      uint16_t tile_sz,
                                                      const int8_t *mvs) {
    if (tile_sz == 0U) {
        return;
    }
    const uint16_t tx_cnt = (uint16_t)((width + tile_sz - 1U) / tile_sz);
    const uint16_t ty_cnt = (uint16_t)((height + tile_sz - 1U) / tile_sz);
    const bool copy_delta = (delta_prev != NULL && delta_mc != NULL);

    for (uint16_t ty = 0; ty < ty_cnt; ty++) {
        int y0 = (int)ty * (int)tile_sz;
        int y1 = y0 + (int)tile_sz;
        if (y1 > (int)height) y1 = (int)height;

        for (uint16_t tx = 0; tx < tx_cnt; tx++) {
            int x0 = (int)tx * (int)tile_sz;
            int x1 = x0 + (int)tile_sz;
            if (x1 > (int)width) x1 = (int)width;
            size_t cw = (size_t)(x1 - x0);

            size_t tidx = ((size_t)ty * tx_cnt + tx) * 2U;
            int dx = mvs[tidx + 0U];
            int dy = mvs[tidx + 1U];

            for (int y = y0; y < y1; y++) {
                int sy = y - dy;
                if (sy < 0) sy = 0;
                if (sy >= (int)height) sy = (int)height - 1;
                size_t src_off = (size_t)sy * width;
                size_t dst_off = (size_t)y * width + (size_t)x0;

                int sx = x0 - dx;
                if (sx >= 0 && sx + (int)cw <= (int)width) {
                    memcpy(dst_mc + dst_off, src_prev + src_off + (size_t)sx, cw);
                    if (copy_delta) {
                        memcpy(delta_mc + dst_off, delta_prev + src_off + (size_t)sx, cw);
                    }
                } else {
                    for (size_t j = 0; j < cw; j++) {
                        int cx = x0 + (int)j - dx;
                        if (cx < 0) cx = 0;
                        if (cx >= (int)width) cx = (int)width - 1;
                        dst_mc[dst_off + j] = src_prev[src_off + (size_t)cx];
                        if (copy_delta) {
                            delta_mc[dst_off + j] = delta_prev[src_off + (size_t)cx];
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief Applies per-tile (dx, dy) motion compensation from src_prev into dst_mc.
 *        src_prev and dst_mc must not alias.
 */
static inline void zcrudp_zcvid_apply_tile_mv(const uint8_t * ZCRUDP_RESTRICT src_prev,
                                                uint8_t * ZCRUDP_RESTRICT dst_mc,
                                                uint16_t width,
                                                uint16_t height,
                                                uint16_t tile_sz,
                                                const int8_t *mvs) {
    zcrudp_zcvid_apply_tile_mv_state(src_prev, dst_mc, NULL, NULL, width, height, tile_sz, mvs);
}

/**
 * @brief Decodes a ZCRUDP_ZCVID_PKT_TILE_MV packet into out_mvs[num_tiles * 2].
 */
static inline bool zcrudp_zcvid_decode_tile_mv_pkt(const uint8_t *pkt_data,
                                                     size_t pkt_len,
                                                     int8_t *out_mvs,
                                                     size_t num_tiles) {
    if (pkt_len < ZCRUDP_ZCVID_PKT_HDR_SZ || out_mvs == NULL) {
        return false;
    }
    zcrudp_zcvid_pkt_hdr_t hdr;
    if (!zcrudp_zcvid_pkt_hdr_read_le(pkt_data, &hdr)) {
        return false;
    }
    if (ZCRUDP_ZCVID_HDR_TYPE(&hdr) != ZCRUDP_ZCVID_PKT_TILE_MV) {
        return false;
    }
    size_t off = (size_t)hdr.start_block;
    size_t cnt = (size_t)hdr.num_tokens;
    if (off > num_tiles || cnt > num_tiles - off) {
        return false;
    }
    if (pkt_len - ZCRUDP_ZCVID_PKT_HDR_SZ < cnt * 2U) {
        return false;
    }
    memcpy(out_mvs + off * 2U, pkt_data + ZCRUDP_ZCVID_PKT_HDR_SZ, cnt * 2U);
    return true;
}

/**
 * @brief Applies Y-plane tile MVs to an NV12 interleaved UV plane (U0 V0 U1 V1...),
 *        dividing dx by 2 in sample-pairs (byte shift = 2 * (dx/2), strictly even
 *        so U and V components never swap) and dy by 2.
 */
static inline void zcrudp_zcvid_apply_tile_mv_nv12_state(const uint8_t * ZCRUDP_RESTRICT uv_prev,
                                                           uint8_t * ZCRUDP_RESTRICT uv_mc,
                                                           const uint8_t * ZCRUDP_RESTRICT uv_delta_prev,
                                                           uint8_t * ZCRUDP_RESTRICT uv_delta_mc,
                                                           uint16_t width,
                                                           uint16_t height,
                                                           uint16_t tile_sz,
                                                           const int8_t *y_mvs) {
    if (tile_sz < 2U || width < 2U || height < 2U) {
        return;
    }
    const uint16_t uv_h = height / 2U;
    const uint16_t uv_tile_h = tile_sz / 2U;
    const uint16_t tx_cnt = (uint16_t)((width + tile_sz - 1U) / tile_sz);
    const uint16_t ty_cnt = (uint16_t)((height + tile_sz - 1U) / tile_sz);
    const bool copy_delta = (uv_delta_prev != NULL && uv_delta_mc != NULL);

    for (uint16_t ty = 0; ty < ty_cnt; ty++) {
        int y0 = (int)ty * (int)uv_tile_h;
        int y1 = y0 + (int)uv_tile_h;
        if (y1 > (int)uv_h) y1 = (int)uv_h;

        for (uint16_t tx = 0; tx < tx_cnt; tx++) {
            int x0 = (int)tx * (int)tile_sz;
            int x1 = x0 + (int)tile_sz;
            if (x1 > (int)width) x1 = (int)width;
            size_t cw = (size_t)(x1 - x0);

            size_t tidx = ((size_t)ty * tx_cnt + tx) * 2U;
            /* Strict even byte shift: 2 * (dx_y / 2) preserves U0,V0 alignment */
            int dx_bytes = (y_mvs[tidx + 0U] / 2) * 2;
            int dy_uv    = y_mvs[tidx + 1U] / 2;

            for (int y = y0; y < y1; y++) {
                int sy = y - dy_uv;
                if (sy < 0) sy = 0;
                if (sy >= (int)uv_h) sy = (int)uv_h - 1;
                size_t src_off = (size_t)sy * width;
                size_t dst_off = (size_t)y * width + (size_t)x0;

                int sx = x0 - dx_bytes;
                if (sx >= 0 && sx + (int)cw <= (int)width) {
                    memcpy(uv_mc + dst_off, uv_prev + src_off + (size_t)sx, cw);
                    if (copy_delta) {
                        memcpy(uv_delta_mc + dst_off, uv_delta_prev + src_off + (size_t)sx, cw);
                    }
                } else {
                    for (size_t j = 0; j < cw; j += 2U) {
                        int cx = x0 + (int)j - dx_bytes;
                        if (cx < 0) cx = 0;
                        if (j + 1U < cw) {
                            if (cx + 1 >= (int)width) cx = (int)width - 2;
                            uv_mc[dst_off + j] = uv_prev[src_off + (size_t)cx];
                            uv_mc[dst_off + j + 1U] = uv_prev[src_off + (size_t)cx + 1U];
                            if (copy_delta) {
                                uv_delta_mc[dst_off + j] = uv_delta_prev[src_off + (size_t)cx];
                                uv_delta_mc[dst_off + j + 1U] = uv_delta_prev[src_off + (size_t)cx + 1U];
                            }
                        } else {
                            if (cx >= (int)width) cx = (int)width - 1;
                            uv_mc[dst_off + j] = uv_prev[src_off + (size_t)cx];
                            if (copy_delta) {
                                uv_delta_mc[dst_off + j] = uv_delta_prev[src_off + (size_t)cx];
                            }
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief Compresses per-tile MVs using spatial left-neighbor DPCM + zero-run RLE.
 *        Format per chunk:
 *          0x00..0x7F : run of 1..128 tiles with (ddx, ddy) == (0, 0)
 *          0x80..0xBF : packed 1-byte (ddx in [-3..3], ddy in [-3..3])
 *          0xFF       : 3-byte literal (0xFF, dx, dy)
 *        Falls back to raw TILE_MV if DPCM is not smaller.
 */
static inline size_t zcrudp_zcvid_encode_tile_mv_dpcm(const int8_t *mvs,
                                                         size_t num_tiles,
                                                         uint16_t tiles_x,
                                                         uint8_t plane_id,
                                                         uint16_t frame_id,
                                                         const zcrudp_zcvid_config_t *cfg,
                                                         uint8_t *pkt_buf,
                                                         zcrudp_zcvid_pkt_cb cb,
                                                         void *user_ctx) {
    if (cfg->mtu_payload <= ZCRUDP_ZCVID_PKT_HDR_SZ + 1U) {
        return 0U;
    }
    const size_t max_payload = (size_t)(cfg->mtu_payload - ZCRUDP_ZCVID_PKT_HDR_SZ);
    uint8_t *payload = pkt_buf + ZCRUDP_ZCVID_PKT_HDR_SZ;
    size_t total_bytes = 0U;
    size_t off = 0U;

    while (off < num_tiles) {
        size_t plen = 0U;
        size_t cnt = 0U;
        size_t zero_run = 0U;
        int8_t pdx = 0, pdy = 0;

        while (off + cnt < num_tiles && cnt < 65535U) {
            size_t idx = off + cnt;
            if (tiles_x > 0U && (idx % tiles_x) == 0U) {
                pdx = 0;
                pdy = 0;
            }
            int8_t dx = mvs[idx * 2U + 0U];
            int8_t dy = mvs[idx * 2U + 1U];
            int ddx = (int)dx - (int)pdx;
            int ddy = (int)dy - (int)pdy;
            pdx = dx;
            pdy = dy;

            if (ddx == 0 && ddy == 0) {
                zero_run++;
                if (zero_run == 128U) {
                    if (plen + 1U > max_payload) {
                        zero_run--;
                        break;
                    }
                    payload[plen++] = 0x7FU;
                    zero_run = 0U;
                }
                cnt++;
                continue;
            }

            size_t need = ((zero_run > 0U) ? 1U : 0U) +
                          ((ddx >= -3 && ddx <= 3 && ddy >= -3 && ddy <= 3) ? 1U : 3U);
            if (plen + need > max_payload) {
                break;
            }
            if (zero_run > 0U) {
                payload[plen++] = (uint8_t)(zero_run - 1U);
                zero_run = 0U;
            }
            if (ddx >= -3 && ddx <= 3 && ddy >= -3 && ddy <= 3) {
                payload[plen++] = (uint8_t)(0x80U | (((uint32_t)(ddx + 3) & 7U) << 3) |
                                            ((uint32_t)(ddy + 3) & 7U));
            } else {
                payload[plen++] = 0xFFU;
                payload[plen++] = (uint8_t)dx;
                payload[plen++] = (uint8_t)dy;
            }
            cnt++;
        }
        if (zero_run > 0U) {
            if (plen + 1U <= max_payload) {
                payload[plen++] = (uint8_t)(zero_run - 1U);
            } else {
                cnt -= zero_run;
            }
        }
        uint8_t ptype = ZCRUDP_ZCVID_PKT_TILE_MV_DPCM;
        if (cnt == 0U) {
            /* max_payload == 2 and first tile needed a 3-byte literal: emit raw */
            cnt = 1U;
            plen = 2U;
            ptype = ZCRUDP_ZCVID_PKT_TILE_MV;
            memcpy(payload, mvs + off * 2U, 2U);
        } else if (plen >= cnt * 2U && cnt * 2U <= max_payload) {
            ptype = ZCRUDP_ZCVID_PKT_TILE_MV;
            plen = cnt * 2U;
            memcpy(payload, mvs + off * 2U, plen);
        }
        zcrudp_zcvid_pkt_hdr_t hdr = {
            .magic = ZCRUDP_ZCVID_PKT_MAGIC,
            .pixfmt = ZCRUDP_ZCVID_MAKE_PF(cfg->pixfmt, ptype),
            .plane = plane_id,
            .frame_id = frame_id,
            .start_block = (uint32_t)off,
            .num_tokens = (uint16_t)cnt
        };
        zcrudp_zcvid_pkt_hdr_write_le(pkt_buf, &hdr);
        size_t psz = ZCRUDP_ZCVID_PKT_HDR_SZ + plen;
        cb(pkt_buf, psz, user_ctx);
        total_bytes += psz;
        off += cnt;
    }
    return total_bytes;
}

/**
 * @brief Decodes either ZCRUDP_ZCVID_PKT_TILE_MV or ZCRUDP_ZCVID_PKT_TILE_MV_DPCM.
 */
static inline bool zcrudp_zcvid_decode_tile_mv_dpcm(const uint8_t *pkt_data,
                                                      size_t pkt_len,
                                                      uint16_t tiles_x,
                                                      int8_t *out_mvs,
                                                      size_t num_tiles) {
    if (pkt_len < ZCRUDP_ZCVID_PKT_HDR_SZ || out_mvs == NULL) {
        return false;
    }
    zcrudp_zcvid_pkt_hdr_t hdr;
    if (!zcrudp_zcvid_pkt_hdr_read_le(pkt_data, &hdr)) {
        return false;
    }
    uint8_t ptype = ZCRUDP_ZCVID_HDR_TYPE(&hdr);
    if (ptype == ZCRUDP_ZCVID_PKT_TILE_MV) {
        return zcrudp_zcvid_decode_tile_mv_pkt(pkt_data, pkt_len, out_mvs, num_tiles);
    }
    if (ptype != ZCRUDP_ZCVID_PKT_TILE_MV_DPCM) {
        return false;
    }
    size_t off = (size_t)hdr.start_block;
    size_t cnt = (size_t)hdr.num_tokens;
    if (off > num_tiles || cnt > num_tiles - off) {
        return false;
    }
    const uint8_t *payload = pkt_data + ZCRUDP_ZCVID_PKT_HDR_SZ;
    const size_t payload_len = pkt_len - ZCRUDP_ZCVID_PKT_HDR_SZ;
    size_t p = 0U, done = 0U;
    int8_t pdx = 0, pdy = 0;

    while (done < cnt) {
        if (p >= payload_len) return false;
        uint8_t tag = payload[p++];
        if (tag < 0x80U) {
            size_t run = (size_t)tag + 1U;
            if (done + run > cnt) return false;
            for (size_t r = 0U; r < run; r++) {
                size_t idx = off + done;
                if (tiles_x > 0U && (idx % tiles_x) == 0U) {
                    pdx = 0;
                    pdy = 0;
                }
                out_mvs[idx * 2U + 0U] = pdx;
                out_mvs[idx * 2U + 1U] = pdy;
                done++;
            }
        } else if (tag < 0xC0U) {
            size_t idx = off + done;
            if (tiles_x > 0U && (idx % tiles_x) == 0U) {
                pdx = 0;
                pdy = 0;
            }
            int ddx = (int)((tag >> 3) & 7U) - 3;
            int ddy = (int)(tag & 7U) - 3;
            pdx = (int8_t)((int)pdx + ddx);
            pdy = (int8_t)((int)pdy + ddy);
            out_mvs[idx * 2U + 0U] = pdx;
            out_mvs[idx * 2U + 1U] = pdy;
            done++;
        } else if (tag == 0xFFU) {
            if (p + 2U > payload_len) return false;
            size_t idx = off + done;
            pdx = (int8_t)payload[p++];
            pdy = (int8_t)payload[p++];
            out_mvs[idx * 2U + 0U] = pdx;
            out_mvs[idx * 2U + 1U] = pdy;
            done++;
        } else {
            return false;
        }
    }
    return true;
}

/**
 * @brief Serializes per-tile motion vectors into MTU-sized ZCRUDP_ZCVID_PKT_TILE_MV packets.
 */
static inline size_t zcrudp_zcvid_encode_tile_mv_pkts(const int8_t *mvs,
                                                        size_t num_tiles,
                                                        uint8_t plane_id,
                                                        uint16_t frame_id,
                                                        const zcrudp_zcvid_config_t *cfg,
                                                        uint8_t *pkt_buf,
                                                        zcrudp_zcvid_pkt_cb cb,
                                                        void *user_ctx) {
    if (cfg->mtu_payload <= ZCRUDP_ZCVID_PKT_HDR_SZ + 1U) {
        return 0U;
    }
    const size_t max_tiles_per_pkt = (size_t)(cfg->mtu_payload - ZCRUDP_ZCVID_PKT_HDR_SZ) / 2U;
    size_t total_bytes = 0;

    for (size_t off = 0; off < num_tiles; off += max_tiles_per_pkt) {
        size_t chunk = num_tiles - off;
        if (chunk > max_tiles_per_pkt) chunk = max_tiles_per_pkt;

        zcrudp_zcvid_pkt_hdr_t hdr = {
            .magic = ZCRUDP_ZCVID_PKT_MAGIC,
            .pixfmt = ZCRUDP_ZCVID_MAKE_PF(cfg->pixfmt, ZCRUDP_ZCVID_PKT_TILE_MV),
            .plane = plane_id,
            .frame_id = frame_id,
            .start_block = (uint32_t)off,
            .num_tokens = (uint16_t)chunk
        };
        zcrudp_zcvid_pkt_hdr_write_le(pkt_buf, &hdr);
        memcpy(pkt_buf + ZCRUDP_ZCVID_PKT_HDR_SZ, mvs + off * 2U, chunk * 2U);
        size_t psz = ZCRUDP_ZCVID_PKT_HDR_SZ + chunk * 2U;
        cb(pkt_buf, psz, user_ctx);
        total_bytes += psz;
    }
    return total_bytes;
}

/**
 * @brief Luma-first dynamic rate-controlled frame encoder (NV12).
 *        Keeps Y at full 1x DELTA while compressing UV at 2x (or raising threshold)
 *        whenever the frame budget would be exceeded, plus rolling Intra-Refresh.
 */
static inline size_t zcrudp_zcvid_encode_nv12_rc(const uint8_t * ZCRUDP_RESTRICT src,
                                                   uint8_t * ZCRUDP_RESTRICT ref,
                                                   uint8_t * ZCRUDP_RESTRICT delta_buf,
                                                   uint16_t frame_id,
                                                   size_t target_bytes_per_frame,
                                                   uint16_t intra_cycle_frames,
                                                   const zcrudp_zcvid_config_t *cfg,
                                                   uint8_t *pkt_buf,
                                                   zcrudp_zcvid_pkt_cb cb,
                                                   void *user_ctx) {
    size_t y_sz = (size_t)cfg->width * (size_t)cfg->height;
    size_t uv_sz = y_sz / 2U;
    size_t bytes = 0;
    size_t total_y_blks = (y_sz + 9U) / 10U;
    size_t i0 = 0, i1 = 0, blk0 = total_y_blks, blk1 = total_y_blks;

    if (intra_cycle_frames > 0U && y_sz > 0U) {
        size_t slice_sz = (y_sz + intra_cycle_frames - 1U) / intra_cycle_frames;
        size_t slot = (size_t)(frame_id % intra_cycle_frames);
        i0 = slot * slice_sz;
        i1 = i0 + slice_sz;
        if (i0 > y_sz) i0 = y_sz;
        if (i1 > y_sz) i1 = y_sz;
        blk0 = (i0 + 9U) / 10U;
        blk1 = i1 / 10U;
        if (blk0 > total_y_blks) blk0 = total_y_blks;
        if (blk1 > total_y_blks) blk1 = total_y_blks;
    }

    /* 1. Full-resolution Luma (Y) first, skipping blocks wholly inside the rolling Intra slice */
    if (intra_cycle_frames > 0U && blk0 < blk1) {
        if (blk0 > 0U) {
            bytes += zcrudp_zcvid_encode_plane_range(src, ref, delta_buf, y_sz, 0U, blk0,
                                                       0, frame_id, cfg, pkt_buf, cb, user_ctx);
        }
        if (blk1 < total_y_blks) {
            bytes += zcrudp_zcvid_encode_plane_range(src, ref, delta_buf, y_sz, blk1, total_y_blks,
                                                       0, frame_id, cfg, pkt_buf, cb, user_ctx);
        }
    } else {
        bytes += zcrudp_zcvid_encode_plane_range(src, ref, delta_buf, y_sz, 0U, total_y_blks,
                                                   0, frame_id, cfg, pkt_buf, cb, user_ctx);
    }

    /* 2. Chroma (UV): use 2x subsampled mode if Y already consumed > 60% of budget */
    uint8_t *uv_delta = delta_buf ? (delta_buf + y_sz) : NULL;
    if (target_bytes_per_frame > 0 && bytes > (target_bytes_per_frame * 6U) / 10U) {
        zcrudp_zcvid_config_t uv_cfg = *cfg;
        uv_cfg.threshold = (cfg->threshold <= 253U) ? (uint8_t)(cfg->threshold + 2U) : 255U;
        bytes += zcrudp_zcvid_encode_plane_2x(src + y_sz, ref + y_sz, uv_delta,
                                                uv_sz, 1, frame_id, &uv_cfg,
                                                pkt_buf, cb, user_ctx);
    } else {
        bytes += zcrudp_zcvid_encode_plane(src + y_sz, ref + y_sz, uv_delta,
                                             uv_sz, 1, frame_id, cfg,
                                             pkt_buf, cb, user_ctx);
    }

    /* 3. Rolling Intra-Refresh slice on Y (prevents HUD/text wash-out) */
    if (intra_cycle_frames > 0U && i0 < i1) {
        bytes += zcrudp_zcvid_encode_intra_range(src, ref, delta_buf, i0, i1,
                                                   y_sz, 0, frame_id, cfg,
                                                   pkt_buf, cb, user_ctx);
    }
    return bytes;
}

/**
 * @brief Disocclusion-aware Intra repair: scans a plane in 64-byte chunks and
 *        emits ZCRUDP_ZCVID_PKT_INTRA for any chunk where >= 2 pixels have
 *        |src - ref| > err_thresh (e.g. uncovered background or scene cut).
 *        Eliminates multi-frame ADM ramp-up ghosting, pushing 3D PSNR to 35-42+ dB.
 *
 *        Call it after the delta pass of the frame (see the frame order note
 *        of zcrudp_zcvid_encode_intra_range): it then anchors exactly what
 *        the deltas could not follow, and stays bit-exact for any gamma_q8.
 *        After a global or per-tile motion shift this is what restores the
 *        uncovered borders and the static overlays the shift dragged along.
 */
static inline size_t zcrudp_zcvid_repair_disocclusions(const uint8_t * ZCRUDP_RESTRICT src,
                                                         uint8_t * ZCRUDP_RESTRICT ref,
                                                         uint8_t * ZCRUDP_RESTRICT delta_buf,
                                                         size_t plane_len,
                                                         uint8_t err_thresh,
                                                         uint8_t plane_id,
                                                         uint16_t frame_id,
                                                         const zcrudp_zcvid_config_t *cfg,
                                                         uint8_t *pkt_buf,
                                                         zcrudp_zcvid_pkt_cb cb,
                                                         void *user_ctx) {
    size_t bytes = 0;
    const size_t CHUNK = 64U;
    size_t run_start = 0;
    bool in_run = false;

    for (size_t off = 0; off < plane_len; off += CHUNK) {
        size_t end = off + CHUNK;
        if (end > plane_len) end = plane_len;

        uint32_t bad = 0;
        uint32_t max_err = 0;
        for (size_t i = off; i < end; i++) {
            int d = (int)src[i] - (int)ref[i];
            if (d < 0) d = -d;
            if ((uint32_t)d > max_err) {
                max_err = (uint32_t)d;
            }
            if (d > (int)err_thresh) {
                bad++;
            }
        }

        if (bad >= 2U || max_err > (uint32_t)err_thresh * 2U) {
            if (!in_run) {
                run_start = off;
                in_run = true;
            }
        } else if (in_run) {
            bytes += zcrudp_zcvid_encode_intra_range(
                src, ref, delta_buf, run_start, off, plane_len,
                plane_id, frame_id, cfg, pkt_buf, cb, user_ctx);
            in_run = false;
        }
    }
    if (in_run) {
        bytes += zcrudp_zcvid_encode_intra_range(
            src, ref, delta_buf, run_start, plane_len, plane_len,
            plane_id, frame_id, cfg, pkt_buf, cb, user_ctx);
    }
    return bytes;
}

/**
 * @brief Shifts a plane in-place by (dx, dy) using a scratch row/plane or memmove.
 *        Both encoder and decoder apply the identical shift before delta coding.
 */
static inline void zcrudp_zcvid_shift_plane(uint8_t *plane,
                                              uint16_t width,
                                              uint16_t height,
                                              int8_t dx,
                                              int8_t dy,
                                              uint8_t fill_val) {
    if (dx == 0 && dy == 0) {
        return;
    }
    const int w = (int)width;
    const int h = (int)height;

    if (dy >= 0) {
        for (int y = h - 1; y >= 0; y--) {
            uint8_t *dst_row = plane + (size_t)y * (size_t)w;
            int src_y = y - dy;
            if (src_y < 0) {
                memset(dst_row, fill_val, (size_t)w);
            } else {
                const uint8_t *src_row = plane + (size_t)src_y * (size_t)w;
                if (dx >= 0) {
                    if (dx < w) {
                        memmove(dst_row + dx, src_row, (size_t)(w - dx));
                        memset(dst_row, fill_val, (size_t)dx);
                    } else {
                        memset(dst_row, fill_val, (size_t)w);
                    }
                } else {
                    int adx = -dx;
                    if (adx < w) {
                        memmove(dst_row, src_row + adx, (size_t)(w - adx));
                        memset(dst_row + (w - adx), fill_val, (size_t)adx);
                    } else {
                        memset(dst_row, fill_val, (size_t)w);
                    }
                }
            }
        }
    } else {
        for (int y = 0; y < h; y++) {
            uint8_t *dst_row = plane + (size_t)y * (size_t)w;
            int src_y = y - dy;
            if (src_y >= h) {
                memset(dst_row, fill_val, (size_t)w);
            } else {
                const uint8_t *src_row = plane + (size_t)src_y * (size_t)w;
                if (dx >= 0) {
                    if (dx < w) {
                        memmove(dst_row + dx, src_row, (size_t)(w - dx));
                        memset(dst_row, fill_val, (size_t)dx);
                    } else {
                        memset(dst_row, fill_val, (size_t)w);
                    }
                } else {
                    int adx = -dx;
                    if (adx < w) {
                        memmove(dst_row, src_row + adx, (size_t)(w - adx));
                        memset(dst_row + (w - adx), fill_val, (size_t)adx);
                    } else {
                        memset(dst_row, fill_val, (size_t)w);
                    }
                }
            }
        }
    }
}

#ifdef __cplusplus
}
#endif

#endif /* ZCRUDP_ZCVID_CODEC_H */
