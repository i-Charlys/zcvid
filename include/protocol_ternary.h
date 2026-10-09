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

/** Maximum combinatorial bounds for base-3 representations */
#define ZCRUDP_TRIT_MAX_5P   243U   /**< 3^5  = 243 <= 256 (8 bits) */
#define ZCRUDP_TRIT_MAX_10P  59049U /**< 3^10 = 59049 <= 65536 (16 bits) */

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
    uint32_t v = byte;
    trits[0] = (zcrudp_trit_t)(v % 3U) - 1; v /= 3U;
    trits[1] = (zcrudp_trit_t)(v % 3U) - 1; v /= 3U;
    trits[2] = (zcrudp_trit_t)(v % 3U) - 1; v /= 3U;
    trits[3] = (zcrudp_trit_t)(v % 3U) - 1; v /= 3U;
    trits[4] = (zcrudp_trit_t)(v % 3U) - 1;
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
 * @param word Encoded 16-bit word in range [0, 59048].
 * @param trits Output array for 10 signed trits {-1, 0, +1}.
 */
static inline void zcrudp_unpack_10trits(uint16_t word, zcrudp_trit_t trits[10]) {
    uint32_t v = word;
    trits[0] = (zcrudp_trit_t)(v % 3U) - 1; v /= 3U;
    trits[1] = (zcrudp_trit_t)(v % 3U) - 1; v /= 3U;
    trits[2] = (zcrudp_trit_t)(v % 3U) - 1; v /= 3U;
    trits[3] = (zcrudp_trit_t)(v % 3U) - 1; v /= 3U;
    trits[4] = (zcrudp_trit_t)(v % 3U) - 1; v /= 3U;
    trits[5] = (zcrudp_trit_t)(v % 3U) - 1; v /= 3U;
    trits[6] = (zcrudp_trit_t)(v % 3U) - 1; v /= 3U;
    trits[7] = (zcrudp_trit_t)(v % 3U) - 1; v /= 3U;
    trits[8] = (zcrudp_trit_t)(v % 3U) - 1; v /= 3U;
    trits[9] = (zcrudp_trit_t)(v % 3U) - 1;
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
    int32_t decayed_diff = (diff * (int32_t)gamma_q8) >> 8;
    int32_t new_val = (int32_t)neutral_y + decayed_diff + ((int32_t)trit * (int32_t)delta);

    if (new_val < 0) return 0;
    if (new_val > 255) return 255;
    return (uint8_t)new_val;
}

#ifdef __cplusplus
}
#endif

#endif /* ZCRUDP_PROTOCOL_TERNARY_H */
