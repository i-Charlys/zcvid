/**
 * @file protocol_ternary.h
 * @brief zcrudp Ternary Video Transport Profile C11 Header
 *
 * Strict C11, zero-dynamic-allocation (malloc-free), pure integer arithmetic,
 * fully unrolled/vectorizable 5-trit (8-bit) and 10-trit (16-bit) packing/unpacking primitives
 * for zcrudp ultra-low-latency UDP transport.
 */

#ifndef ZCRUDP_PROTOCOL_TERNARY_H
#define ZCRUDP_PROTOCOL_TERNARY_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifndef ZCRUDP_RESTRICT
#ifdef __cplusplus
#if defined(__GNUC__) || defined(__clang__)
#define ZCRUDP_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define ZCRUDP_RESTRICT __restrict
#else
#define ZCRUDP_RESTRICT
#endif
#else
#define ZCRUDP_RESTRICT restrict
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Constants and Protocol Types
 * ========================================================================= */

/** TFV Record Types for Ternary Video Profile (Track D) */
#define ZCRUDP_TFV_TYPE_TERNARY_HEADER     0xD0  /**< Header record for slice/block metadata */
#define ZCRUDP_TFV_TYPE_TERNARY_PACK10     0xD1  /**< 10-trit packed uint16_t data record */
#define ZCRUDP_TFV_TYPE_TERNARY_PACK5_PAIR 0xD2  /**< Dual 5-trit packed uint16_t data record */
#define ZCRUDP_TFV_TYPE_TERNARY_SKIP       0xD3  /**< RLE skip record: value = count of zero 10-trit blocks */

/** Maximum combinatorial bounds for base-3 representations */
#define ZCRUDP_TRIT_MAX_5P   243U   /**< 3^5  = 243 <= 256 (8 bits) */
#define ZCRUDP_TRIT_MAX_10P  59049U /**< 3^10 = 59049 <= 65536 (16 bits) */

/** Precomputed packed values for all-zero trits (t_i = 0 => v_i = 1) */
#define ZCRUDP_TRIT_ALL_ZERO_5P   121U   /**< sum_{i=0..4} 3^i = (243 - 1) / 2 */
#define ZCRUDP_TRIT_ALL_ZERO_10P  29524U /**< sum_{i=0..9} 3^i = (59049 - 1) / 2 */

/** Signed Trit values {-1, 0, +1} */
typedef int8_t zcrudp_trit_t;

/** Unsigned mapped Trit values {0, 1, 2} where v = t + 1 */
typedef uint8_t zcrudp_utrit_t;

/* =========================================================================
 * Structure Definitions (Strictly Static / No Malloc)
 * ========================================================================= */

/**
 * @brief Atomic TFV (Type 8 / Flags 8 / Value 16) payload record format.
 */
typedef struct {
    uint8_t  type;    /**< Record type identifier (e.g. ZCRUDP_TFV_TYPE_TERNARY_PACK10) */
    uint8_t  flags;   /**< Sequence offset or sub-stream flags */
    uint16_t value;   /**< 16-bit value carrying 10-trits packed in base-3 */
} zcrudp_tfv_record_t;

/**
 * @brief Header structure for a slice/block of ternary video deltas.
 */
typedef struct {
    uint16_t slice_index;  /**< Slice sequence number / index */
    uint16_t start_pixel;  /**< Starting pixel offset in frame canvas */
    uint16_t count;        /**< Number of packed records in slice */
    uint8_t  flags;        /**< Additional metadata flags (e.g., end-of-frame) */
    uint8_t  reserved;     /**< Alignment padding */
} zcrudp_ternary_slice_header_t;

_Static_assert(sizeof(zcrudp_tfv_record_t) == 4, "zcrudp_tfv_record_t must be 4 bytes");
_Static_assert(sizeof(zcrudp_ternary_slice_header_t) == 8, "zcrudp_ternary_slice_header_t must be 8 bytes");

/* =========================================================================
 * Powers of 3 Static LookUp Table for Compile-Time Unrolling
 * ========================================================================= */

static const uint16_t ZCRUDP_POW3[11] = {
    1U, 3U, 9U, 27U, 81U, 243U, 729U, 2187U, 6561U, 19683U, 59049U
};

/* =========================================================================
 * Inline Packing / Unpacking Functions (5-Trit & 10-Trit)
 * ========================================================================= */

/**
 * @brief Maps a signed trit (-1, 0, +1) to an unsigned trit (0, 1, 2).
 */
static inline zcrudp_utrit_t zcrudp_trit_to_utrit(zcrudp_trit_t t) {
    return (zcrudp_utrit_t)(t + 1);
}

/**
 * @brief Maps an unsigned trit (0, 1, 2) back to a signed trit (-1, 0, +1).
 */
static inline zcrudp_trit_t zcrudp_utrit_to_trit(zcrudp_utrit_t u) {
    return (zcrudp_trit_t)u - 1;
}

/**
 * @brief Packs 5 signed trits into a single uint8_t byte (base-3).
 *
 * @param trits Array of 5 signed trits in {-1, 0, +1}.
 * @return uint8_t Encoded byte in range [0, 242].
 */
static inline uint8_t zcrudp_pack_5trits(const zcrudp_trit_t trits[5]) {
    uint32_t val = (uint32_t)(trits[0] + 1) +
                   (uint32_t)(trits[1] + 1) * 3U +
                   (uint32_t)(trits[2] + 1) * 9U +
                   (uint32_t)(trits[3] + 1) * 27U +
                   (uint32_t)(trits[4] + 1) * 81U;
    return (uint8_t)val;
}

/**
 * @brief Unpacks 5 signed trits from a uint8_t byte (base-3).
 *
 * @param byte Encoded byte in range [0, 242].
 * @param trits Output array for 5 signed trits {-1, 0, +1}.
 */
static inline void zcrudp_unpack_5trits(uint8_t byte, zcrudp_trit_t trits[5]) {
    uint32_t v  = byte;
    uint32_t q1 = (v * 21846U) >> 16;
    uint32_t q2 = (v * 7282U)  >> 16;
    uint32_t q3 = (v * 2428U)  >> 16;
    uint32_t q4 = (v * 810U)   >> 16;
    trits[0] = (zcrudp_trit_t)(v  - q1 * 3U) - 1;
    trits[1] = (zcrudp_trit_t)(q1 - q2 * 3U) - 1;
    trits[2] = (zcrudp_trit_t)(q2 - q3 * 3U) - 1;
    trits[3] = (zcrudp_trit_t)(q3 - q4 * 3U) - 1;
    trits[4] = (zcrudp_trit_t)q4 - 1;
}

/**
 * @brief Unpacks 5 signed trits from a uint8_t byte (base-3) with range validation.
 *
 * @param byte Encoded byte; must be in [0, 242].
 * @param trits Output array for 5 signed trits {-1, 0, +1}.
 * @return true if byte < 243, false if corrupted/out-of-range.
 */
static inline bool zcrudp_unpack_5trits_checked(uint8_t byte, zcrudp_trit_t trits[5]) {
    if (byte >= ZCRUDP_TRIT_MAX_5P) {
        return false;
    }
    zcrudp_unpack_5trits(byte, trits);
    return true;
}

/**
 * @brief Packs 10 signed trits into a single uint16_t word (base-3).
 *
 * @param trits Array of 10 signed trits in {-1, 0, +1}.
 * @return uint16_t Encoded 16-bit word in range [0, 59048].
 */
static inline uint16_t zcrudp_pack_10trits(const zcrudp_trit_t trits[10]) {
    uint32_t val = (uint32_t)(trits[0] + 1) +
                   (uint32_t)(trits[1] + 1) * 3U +
                   (uint32_t)(trits[2] + 1) * 9U +
                   (uint32_t)(trits[3] + 1) * 27U +
                   (uint32_t)(trits[4] + 1) * 81U +
                   (uint32_t)(trits[5] + 1) * 243U +
                   (uint32_t)(trits[6] + 1) * 729U +
                   (uint32_t)(trits[7] + 1) * 2187U +
                   (uint32_t)(trits[8] + 1) * 6561U +
                   (uint32_t)(trits[9] + 1) * 19683U;
    return (uint16_t)val;
}

/**
 * @brief Unpacks 10 signed trits from a uint16_t word (base-3).
 *
 * Splits the 10-trit word into two independent 5-trit halves (lo = word % 243,
 * hi = word / 243) so the two 5-trit extractions execute in parallel without
 * a 10-step serial division chain.
 *
 * @param word Encoded 16-bit word in range [0, 59048].
 * @param trits Output array for 10 signed trits {-1, 0, +1}.
 */
static inline void zcrudp_unpack_10trits(uint16_t word, zcrudp_trit_t trits[10]) {
    uint32_t w  = word;
    uint32_t hi = (w * 69043U) >> 24;
    uint32_t lo = w - hi * ZCRUDP_TRIT_MAX_5P;
    zcrudp_unpack_5trits((uint8_t)lo, &trits[0]);
    zcrudp_unpack_5trits((uint8_t)hi, &trits[5]);
}

/**
 * @brief Unpacks 10 signed trits from a uint16_t word (base-3) with range validation.
 *
 * @param word Encoded 16-bit word; must be in [0, 59048].
 * @param trits Output array for 10 signed trits {-1, 0, +1}.
 * @return true if word < 59049, false if corrupted/out-of-range.
 */
static inline bool zcrudp_unpack_10trits_checked(uint16_t word, zcrudp_trit_t trits[10]) {
    if (word >= ZCRUDP_TRIT_MAX_10P) {
        return false;
    }
    zcrudp_unpack_10trits(word, trits);
    return true;
}

/**
 * @brief Packs 10 signed trits as two independent 5-trit bytes in a uint16_t
 *        (for ZCRUDP_TFV_TYPE_TERNARY_PACK5_PAIR).
 *
 * @param trits Array of 10 signed trits in {-1, 0, +1}.
 * @return uint16_t Low byte = trits[0..4], high byte = trits[5..9].
 */
static inline uint16_t zcrudp_pack_5trits_pair(const zcrudp_trit_t trits[10]) {
    uint8_t b0 = zcrudp_pack_5trits(&trits[0]);
    uint8_t b1 = zcrudp_pack_5trits(&trits[5]);
    return (uint16_t)((uint16_t)b0 | ((uint16_t)b1 << 8));
}

/**
 * @brief Unpacks two 5-trit bytes from a uint16_t word with range validation.
 *
 * @param word Packed dual 5-trit word.
 * @param trits Output array for 10 signed trits {-1, 0, +1}.
 * @return true if both bytes are < 243, false if corrupted/out-of-range.
 */
static inline bool zcrudp_unpack_5trits_pair_checked(uint16_t word, zcrudp_trit_t trits[10]) {
    uint8_t b0 = (uint8_t)(word & 0xFFU);
    uint8_t b1 = (uint8_t)(word >> 8);
    if (b0 >= ZCRUDP_TRIT_MAX_5P || b1 >= ZCRUDP_TRIT_MAX_5P) {
        return false;
    }
    zcrudp_unpack_5trits(b0, &trits[0]);
    zcrudp_unpack_5trits(b1, &trits[5]);
    return true;
}

/* =========================================================================
 * Endian-Safe Wire Serialization Helpers (Little-Endian Wire Format)
 * ========================================================================= */

/**
 * @brief Serializes a TFV record into a 4-byte little-endian wire buffer.
 */
static inline void zcrudp_tfv_write_le(uint8_t dst[4], const zcrudp_tfv_record_t *rec) {
    dst[0] = rec->type;
    dst[1] = rec->flags;
    dst[2] = (uint8_t)(rec->value & 0xFFU);
    dst[3] = (uint8_t)(rec->value >> 8);
}

/**
 * @brief Deserializes a TFV record from a 4-byte little-endian wire buffer.
 */
static inline void zcrudp_tfv_read_le(const uint8_t src[4], zcrudp_tfv_record_t *rec) {
    rec->type = src[0];
    rec->flags = src[1];
    rec->value = (uint16_t)((uint16_t)src[2] | ((uint16_t)src[3] << 8));
}

/**
 * @brief Serializes a slice header into an 8-byte little-endian wire buffer.
 */
static inline void zcrudp_slice_header_write_le(uint8_t dst[8],
                                                const zcrudp_ternary_slice_header_t *hdr) {
    dst[0] = (uint8_t)(hdr->slice_index & 0xFFU);
    dst[1] = (uint8_t)(hdr->slice_index >> 8);
    dst[2] = (uint8_t)(hdr->start_pixel & 0xFFU);
    dst[3] = (uint8_t)(hdr->start_pixel >> 8);
    dst[4] = (uint8_t)(hdr->count & 0xFFU);
    dst[5] = (uint8_t)(hdr->count >> 8);
    dst[6] = hdr->flags;
    dst[7] = hdr->reserved;
}

/**
 * @brief Deserializes a slice header from an 8-byte little-endian wire buffer.
 */
static inline void zcrudp_slice_header_read_le(const uint8_t src[8],
                                               zcrudp_ternary_slice_header_t *hdr) {
    hdr->slice_index = (uint16_t)((uint16_t)src[0] | ((uint16_t)src[1] << 8));
    hdr->start_pixel = (uint16_t)((uint16_t)src[2] | ((uint16_t)src[3] << 8));
    hdr->count = (uint16_t)((uint16_t)src[4] | ((uint16_t)src[5] << 8));
    hdr->flags = src[6];
    hdr->reserved = src[7];
}

/* =========================================================================
 * Leaky Integrator Reconstruction Primitive
 * ========================================================================= */

/**
 * @brief Applies leaky integration update step to a single pixel.
 *
 * Y(t) = clamp_0_255( Y_neutral + gamma * (Y(t-1) - Y_neutral) + delta * trit )
 *
 * @param prev_y Previous absolute pixel value (0..255).
 * @param trit Signed delta trit (-1, 0, +1).
 * @param delta Step magnitude (e.g. 8).
 * @param gamma_q8 Decay factor in Q0.8 fixed-point (e.g., 243 for ~0.95).
 * @param neutral_y Neutral gray background reference (default 128).
 * @return uint8_t Reconstructed pixel value clamped to [0, 255].
 */
static inline uint8_t zcrudp_leaky_integrator_step(uint8_t prev_y,
                                                   zcrudp_trit_t trit,
                                                   uint8_t delta,
                                                   uint8_t gamma_q8,
                                                   uint8_t neutral_y) {
    int32_t diff = (int32_t)prev_y - (int32_t)neutral_y;
    int32_t decayed_diff = (diff * (int32_t)gamma_q8) / 256;
    int32_t new_val = (int32_t)neutral_y + decayed_diff + ((int32_t)trit * (int32_t)delta);

    if (new_val < 0) return 0;
    if (new_val > 255) return 255;
    return (uint8_t)new_val;
}

/**
 * @brief Applies leaky integration in-place across a contiguous pixel slice.
 *
 * Uses restrict pointers so C11 compilers can auto-vectorize the loop.
 */
static inline void zcrudp_leaky_integrator_batch(uint8_t * ZCRUDP_RESTRICT canvas,
                                                 const zcrudp_trit_t * ZCRUDP_RESTRICT trits,
                                                 size_t count,
                                                 uint8_t delta,
                                                 uint8_t gamma_q8,
                                                 uint8_t neutral_y) {
    for (size_t i = 0; i < count; i++) {
        canvas[i] = zcrudp_leaky_integrator_step(canvas[i], trits[i],
                                                delta, gamma_q8, neutral_y);
    }
}

/* =========================================================================
 * Closed-Loop Encoder Primitives (Quantization + Reference Sync)
 * ========================================================================= */

/**
 * @brief Quantizes the difference between current pixel and reference into a trit.
 */
static inline zcrudp_trit_t zcrudp_quantize_trit(uint8_t current_y,
                                                uint8_t ref_y,
                                                uint8_t threshold) {
    int32_t err = (int32_t)current_y - (int32_t)ref_y;
    if (err > (int32_t)threshold) return 1;
    if (err < -(int32_t)threshold) return -1;
    return 0;
}

/**
 * @brief Closed-loop encoder step: quantizes trit and updates encoder reference
 *        using the exact same leaky integrator as the decoder to prevent drift.
 */
static inline zcrudp_trit_t zcrudp_encode_step(uint8_t current_y,
                                               uint8_t *ref_y,
                                               uint8_t threshold,
                                               uint8_t delta,
                                               uint8_t gamma_q8,
                                               uint8_t neutral_y) {
    zcrudp_trit_t trit = zcrudp_quantize_trit(current_y, *ref_y, threshold);
    *ref_y = zcrudp_leaky_integrator_step(*ref_y, trit, delta, gamma_q8, neutral_y);
    return trit;
}

#ifdef __cplusplus
}
#endif

#endif /* ZCRUDP_PROTOCOL_TERNARY_H */
