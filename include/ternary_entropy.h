/**
 * @file ternary_entropy.h
 * @brief Adaptive binary range coder and context models for the ternary codec.
 *
 * Fixed base-3 packing spends 1.6 bit per trit whatever the picture. Trits are
 * far from uniform (most are 0, and the others cluster along moving edges), so
 * their entropy is well under log2(3). This coder spends what the trits are
 * actually worth.
 *
 * Every packet is coded on its own: the coder and all probabilities restart at
 * the packet boundary, and contexts only look at trits carried by the same
 * packet. Nothing depends on decoder state (canvas, ADM steps, earlier
 * packets), so a lost packet cannot desynchronise the ones that follow.
 *
 * Strict C11, no allocation. The decoder accepts arbitrary bytes: it never
 * reads past the payload (missing bytes read as 0) and yields bounded values.
 */

#ifndef ZCRUDP_TERNARY_ENTROPY_H
#define ZCRUDP_TERNARY_ENTROPY_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Adaptive bit model
 *
 * One 16-bit estimate per context, moved 1/32 of the way toward each coded
 * bit. Measured against a fast+slow pair of estimators, this single rate gave
 * the same or smaller packets on every test scene, so the simpler form stays.
 *
 * Speed, for whoever is tempted to tune this coder. A coded trit costs 5 to
 * 6 ns. Alone in a loop, a branch-free interval update is 30 % faster than the
 * branching one below (2.9 against 4.1 ns per bit). Inside the block coder it
 * is not: the zero flag still decides whether a sign follows, so the same
 * misprediction comes back one line later; coding the sign through a mask for
 * every trit, into a second stream so both coders advance in parallel, removes
 * the branch but adds a full coder step to the 60 % of trits that are 0 and
 * was 20 % slower end to end. Two coders in lockstep gain nothing either
 * while branches dominate. The cost is the number of coded decisions; cap it
 * with cfg.ec_max_blocks rather than by rewriting this.
 * ========================================================================= */

#define ZCRUDP_RC_PROB_BITS  12U
#define ZCRUDP_RC_TOP        (1U << 24)
#define ZCRUDP_RC_RATE_SHIFT 5U

/** Upper bound, in bits, on the cost of one modelled binary decision. */
#define ZCRUDP_RC_MAX_BITS_PER_DECISION 11U

typedef struct {
    uint16_t p0; /**< P(bit == 0) in Q15 */
} zcrudp_rc_prob_t;

/** @param p0_q15 initial P(bit == 0) in Q15, within [32, 32736]. */
static inline void zcrudp_rc_prob_init(zcrudp_rc_prob_t *p, uint16_t p0_q15) {
    p->p0 = p0_q15;
}

/** P(bit == 0) in Q12. Always within [3, 4092], so no symbol has zero width. */
static inline uint32_t zcrudp_rc_prob_p12(const zcrudp_rc_prob_t *p) {
    return (uint32_t)p->p0 >> 3;
}

static inline void zcrudp_rc_prob_update(zcrudp_rc_prob_t *p, uint32_t bit) {
    if (bit == 0U) {
        p->p0 = (uint16_t)(p->p0 + ((32768U - p->p0) >> ZCRUDP_RC_RATE_SHIFT));
    } else {
        p->p0 = (uint16_t)(p->p0 - (p->p0 >> ZCRUDP_RC_RATE_SHIFT));
    }
}

/* =========================================================================
 * Range encoder (32-bit range, carry propagated through a cache byte)
 * ========================================================================= */

typedef struct {
    uint64_t low;
    uint32_t range;
    uint8_t  cache;
    size_t   cache_size; /**< pending bytes: cache + following 0xFF run */
    bool     first;      /**< the first byte is always 0 and is not sent */
    uint8_t *out;
    size_t   pos;
    size_t   cap;
    bool     overflow;
} zcrudp_rc_enc_t;

static inline void zcrudp_rc_enc_init(zcrudp_rc_enc_t *e, uint8_t *out, size_t cap) {
    e->low = 0;
    e->range = 0xFFFFFFFFU;
    e->cache = 0;
    e->cache_size = 1;
    e->first = true;
    e->out = out;
    e->pos = 0;
    e->cap = cap;
    e->overflow = false;
}

static inline void zcrudp_rc_enc_put(zcrudp_rc_enc_t *e, uint8_t b) {
    if (e->first) {
        e->first = false;
        return;
    }
    if (e->pos < e->cap) {
        e->out[e->pos++] = b;
    } else {
        e->overflow = true;
    }
}

static inline void zcrudp_rc_enc_shift_low(zcrudp_rc_enc_t *e) {
    if ((uint32_t)e->low < 0xFF000000U || (e->low >> 32) != 0U) {
        uint8_t carry = (uint8_t)(e->low >> 32);
        uint8_t temp = e->cache;
        do {
            zcrudp_rc_enc_put(e, (uint8_t)(temp + carry));
            temp = 0xFFU;
        } while (--e->cache_size != 0U);
        e->cache = (uint8_t)(e->low >> 24);
    }
    e->cache_size++;
    e->low = (e->low & 0x00FFFFFFU) << 8;
}

/** Upper bound on the payload size if the stream were closed right now. */
static inline size_t zcrudp_rc_enc_pending(const zcrudp_rc_enc_t *e) {
    return e->pos + e->cache_size + 4U;
}

static inline void zcrudp_rc_enc_bit(zcrudp_rc_enc_t *e, zcrudp_rc_prob_t *p, uint32_t bit) {
    uint32_t bound = (e->range >> ZCRUDP_RC_PROB_BITS) * zcrudp_rc_prob_p12(p);
    if (bit == 0U) {
        e->range = bound;
    } else {
        e->low += bound;
        e->range -= bound;
    }
    zcrudp_rc_prob_update(p, bit);
    while (e->range < ZCRUDP_RC_TOP) {
        e->range <<= 8;
        zcrudp_rc_enc_shift_low(e);
    }
}

/** Codes the low @p nbits of @p val, most significant first, at 1 bit each. */
static inline void zcrudp_rc_enc_direct(zcrudp_rc_enc_t *e, uint64_t val, uint32_t nbits) {
    while (nbits-- > 0U) {
        e->range >>= 1;
        if ((val >> nbits) & 1U) {
            e->low += e->range;
        }
        while (e->range < ZCRUDP_RC_TOP) {
            e->range <<= 8;
            zcrudp_rc_enc_shift_low(e);
        }
    }
}

/**
 * @brief Closes the stream and returns the payload size in bytes.
 *
 * Any value inside the final interval decodes correctly and the decoder reads
 * missing bytes as 0, so the value with the most trailing zero bytes is picked
 * and those bytes are not sent.
 */
static inline size_t zcrudp_rc_enc_finish(zcrudp_rc_enc_t *e) {
    uint64_t hi = e->low + (uint64_t)e->range - 1U;
    for (uint32_t shift = 24U; shift > 0U; shift -= 8U) {
        uint64_t mask = ((uint64_t)1 << shift) - 1U;
        uint64_t v = (e->low + mask) & ~mask;
        if (v <= hi) {
            e->low = v;
            break;
        }
    }
    for (int i = 0; i < 5; i++) {
        zcrudp_rc_enc_shift_low(e);
    }
    while (e->pos > 0U && e->out[e->pos - 1U] == 0U) {
        e->pos--;
    }
    return e->pos;
}

/* =========================================================================
 * Range decoder
 * ========================================================================= */

typedef struct {
    uint32_t range;
    uint32_t code;
    const uint8_t *in;
    size_t pos;
    size_t len;
} zcrudp_rc_dec_t;

static inline uint8_t zcrudp_rc_dec_byte(zcrudp_rc_dec_t *d) {
    return (d->pos < d->len) ? d->in[d->pos++] : (uint8_t)0;
}

static inline void zcrudp_rc_dec_init(zcrudp_rc_dec_t *d, const uint8_t *in, size_t len) {
    d->range = 0xFFFFFFFFU;
    d->code = 0;
    d->in = in;
    d->pos = 0;
    d->len = len;
    for (int i = 0; i < 4; i++) {
        d->code = (d->code << 8) | zcrudp_rc_dec_byte(d);
    }
}

static inline uint32_t zcrudp_rc_dec_bit(zcrudp_rc_dec_t *d, zcrudp_rc_prob_t *p) {
    uint32_t bound = (d->range >> ZCRUDP_RC_PROB_BITS) * zcrudp_rc_prob_p12(p);
    uint32_t bit;
    if (d->code < bound) {
        d->range = bound;
        bit = 0U;
    } else {
        d->code -= bound;
        d->range -= bound;
        bit = 1U;
    }
    zcrudp_rc_prob_update(p, bit);
    while (d->range < ZCRUDP_RC_TOP) {
        d->range <<= 8;
        d->code = (d->code << 8) | zcrudp_rc_dec_byte(d);
    }
    return bit;
}

static inline uint64_t zcrudp_rc_dec_direct(zcrudp_rc_dec_t *d, uint32_t nbits) {
    uint64_t val = 0;
    while (nbits-- > 0U) {
        d->range >>= 1;
        uint32_t bit = (d->code >= d->range) ? 1U : 0U;
        if (bit) {
            d->code -= d->range;
        }
        val = (val << 1) | bit;
        while (d->range < ZCRUDP_RC_TOP) {
            d->range <<= 8;
            d->code = (d->code << 8) | zcrudp_rc_dec_byte(d);
        }
    }
    return val;
}

/* =========================================================================
 * Adaptive Exp-Golomb: unary length under context, mantissa at 1 bit each
 * ========================================================================= */

/** Codes v >= 0. @p ctx must hold @p nctx models; lengths past it share the last. */
static inline void zcrudp_rc_enc_egk(zcrudp_rc_enc_t *e, zcrudp_rc_prob_t *ctx,
                                     uint32_t nctx, uint32_t max_len, uint64_t v) {
    uint64_t x = v + 1U;
    uint32_t n = 0;
    while ((x >> (n + 1U)) != 0U) {
        n++;
    }
    if (n > max_len) {
        n = max_len;
        x = ((uint64_t)1 << (n + 1U)) - 1U;
    }
    for (uint32_t i = 0; i < n; i++) {
        zcrudp_rc_enc_bit(e, &ctx[(i < nctx) ? i : (nctx - 1U)], 1U);
    }
    if (n < max_len) {
        zcrudp_rc_enc_bit(e, &ctx[(n < nctx) ? n : (nctx - 1U)], 0U);
    }
    zcrudp_rc_enc_direct(e, x, n);
}

static inline uint64_t zcrudp_rc_dec_egk(zcrudp_rc_dec_t *d, zcrudp_rc_prob_t *ctx,
                                         uint32_t nctx, uint32_t max_len) {
    uint32_t n = 0;
    while (n < max_len &&
           zcrudp_rc_dec_bit(d, &ctx[(n < nctx) ? n : (nctx - 1U)]) != 0U) {
        n++;
    }
    return (((uint64_t)1 << n) | zcrudp_rc_dec_direct(d, n)) - 1U;
}

/* =========================================================================
 * Trit model (DELTA_EC / DELTA_2X_EC)
 *
 * hist[] holds the trits already coded in this packet, indexed by sample
 * position. It supplies the left neighbours and, once the packet spans more
 * than one row, the row above.
 *
 * Neighbours are read without range checks: a position that precedes the
 * packet (or the plane, the index then wraps inside the ring) was cleared by
 * zcrudp_tec_model_reset() and reads as 0, "nothing known".
 *
 * Contexts (built in ternary_codec.h, which knows the block geometry):
 *   zero flag: (left != 0) | (left2 != 0) << 1 | (up != 0) << 2 |
 *              ((up-left != 0) + (up-right != 0)) << 3          -> 24 contexts
 *   sign:      left * 3 + (up, else up-left, else up-right)     ->  9 contexts
 * The row above is read once per block, before the block is coded.
 * ========================================================================= */

#define ZCRUDP_TEC_HIST_SZ   8192U
#define ZCRUDP_TEC_HIST_MASK (ZCRUDP_TEC_HIST_SZ - 1U)
#define ZCRUDP_TEC_ZCTX      24U
#define ZCRUDP_TEC_SCTX      9U
#define ZCRUDP_TEC_RUN_SETS  3U
#define ZCRUDP_TEC_RUN_CTX   20U
#define ZCRUDP_TEC_RUN_MAXLEN 32U

/** Worst-case bytes one coded block can add (run + 10 trits), plus the flush. */
#define ZCRUDP_TEC_BLOCK_MARGIN 96U

typedef struct {
    zcrudp_rc_prob_t zero[ZCRUDP_TEC_ZCTX];
    zcrudp_rc_prob_t sign[ZCRUDP_TEC_SCTX];
    zcrudp_rc_prob_t run[ZCRUDP_TEC_RUN_SETS][ZCRUDP_TEC_RUN_CTX];
    uint32_t run_set;  /**< context set picked from the previous run length */
    size_t   stride;   /**< distance to the sample above, 0 when unusable */
    size_t   lstep;    /**< distance to the previous sample of the same kind */
    uint8_t  hist[ZCRUDP_TEC_HIST_SZ]; /**< 0 = zero, 1 = plus, 2 = minus */
} zcrudp_tec_model_t;

static inline void zcrudp_tec_model_reset(zcrudp_tec_model_t *m, size_t stride, size_t lstep) {
    /* P(zero) priors: a trit with quiet neighbours is most likely 0. */
    for (uint32_t c = 0; c < ZCRUDP_TEC_ZCTX; c++) {
        uint32_t busy = (c & 1U) + ((c >> 1) & 1U) + ((c >> 2) & 1U) + (c >> 3);
        static const uint16_t prior[6] = { 26000U, 18000U, 13000U, 10000U, 8000U, 6500U };
        zcrudp_rc_prob_init(&m->zero[c], prior[busy]);
    }
    for (uint32_t c = 0; c < ZCRUDP_TEC_SCTX; c++) {
        /* c = left * 3 + up, 1 = plus, 2 = minus; bit 1 means minus. */
        uint32_t l = c / 3U, u = c % 3U;
        int lean = ((l == 1U) ? 1 : (l == 2U) ? -1 : 0) + ((u == 1U) ? 1 : (u == 2U) ? -1 : 0);
        zcrudp_rc_prob_init(&m->sign[c], (uint16_t)(16384 + lean * 5000));
    }
    for (uint32_t s = 0; s < ZCRUDP_TEC_RUN_SETS; s++) {
        for (uint32_t c = 0; c < ZCRUDP_TEC_RUN_CTX; c++) {
            zcrudp_rc_prob_init(&m->run[s][c], 16384U);
        }
    }
    m->run_set = 0;
    m->lstep = lstep;
    /* The row above must lie outside the block being coded (20 samples at most). */
    m->stride = (stride >= 32U && stride + 4U * lstep + 8U <= ZCRUDP_TEC_HIST_SZ) ? stride : 0U;
    memset(m->hist, 0, sizeof(m->hist));
}

/** Marks samples [from, to) as zero (blocks skipped inside the packet). */
static inline void zcrudp_tec_hist_clear(zcrudp_tec_model_t *m, size_t from, size_t to) {
    if (to <= from) {
        return;
    }
    if (to - from >= ZCRUDP_TEC_HIST_SZ) {
        memset(m->hist, 0, sizeof(m->hist));
        return;
    }
    size_t a = from & ZCRUDP_TEC_HIST_MASK;
    size_t n = to - from;
    size_t first = ZCRUDP_TEC_HIST_SZ - a;
    if (first > n) {
        first = n;
    }
    memset(m->hist + a, 0, first);
    memset(m->hist, 0, n - first);
}

static inline void zcrudp_tec_enc_run(zcrudp_rc_enc_t *e, zcrudp_tec_model_t *m, uint64_t run) {
    zcrudp_rc_enc_egk(e, m->run[m->run_set], ZCRUDP_TEC_RUN_CTX, ZCRUDP_TEC_RUN_MAXLEN, run);
    m->run_set = (run == 0U) ? 0U : (run < 4U) ? 1U : 2U;
}

static inline uint64_t zcrudp_tec_dec_run(zcrudp_rc_dec_t *d, zcrudp_tec_model_t *m) {
    uint64_t run = zcrudp_rc_dec_egk(d, m->run[m->run_set], ZCRUDP_TEC_RUN_CTX,
                                     ZCRUDP_TEC_RUN_MAXLEN);
    m->run_set = (run == 0U) ? 0U : (run < 4U) ? 1U : 2U;
    return run;
}

/* =========================================================================
 * Residual model (INTRA_EC): lossless prediction error of 8-bit samples
 * ========================================================================= */

#define ZCRUDP_IEC_BUCKETS 6U
#define ZCRUDP_IEC_MAG_CTX 8U
#define ZCRUDP_IEC_MAG_MAXLEN 7U

/** Worst-case bytes one coded sample can add, plus the flush. */
#define ZCRUDP_IEC_SAMPLE_MARGIN 32U

typedef struct {
    zcrudp_rc_prob_t zero[ZCRUDP_IEC_BUCKETS];
    zcrudp_rc_prob_t sign[3];
    zcrudp_rc_prob_t mag[ZCRUDP_IEC_BUCKETS][ZCRUDP_IEC_MAG_CTX];
} zcrudp_iec_model_t;

static inline void zcrudp_iec_model_reset(zcrudp_iec_model_t *m) {
    static const uint16_t zprior[ZCRUDP_IEC_BUCKETS] = {
        26000U, 16000U, 9000U, 5000U, 3000U, 2000U
    };
    for (uint32_t b = 0; b < ZCRUDP_IEC_BUCKETS; b++) {
        zcrudp_rc_prob_init(&m->zero[b], zprior[b]);
        for (uint32_t c = 0; c < ZCRUDP_IEC_MAG_CTX; c++) {
            /* Unary length: early stop likely in calm buckets, later in busy ones. */
            int32_t p = 24000 - (int32_t)b * 3500 + (int32_t)c * 3000;
            if (p < 4000) p = 4000;
            if (p > 28000) p = 28000;
            zcrudp_rc_prob_init(&m->mag[b][c], (uint16_t)p);
        }
    }
    for (uint32_t c = 0; c < 3U; c++) {
        zcrudp_rc_prob_init(&m->sign[c], 16384U);
    }
}

/** Activity bucket from the magnitudes of the neighbouring residuals. */
static inline uint32_t zcrudp_iec_bucket(uint32_t act) {
    if (act == 0U) return 0U;
    if (act <= 2U) return 1U;
    if (act <= 5U) return 2U;
    if (act <= 11U) return 3U;
    if (act <= 24U) return 4U;
    return 5U;
}

/** Median edge detector of JPEG-LS: left a, above b, above-left c. */
static inline uint8_t zcrudp_iec_med(uint8_t a, uint8_t b, uint8_t c) {
    uint8_t mn = (a < b) ? a : b;
    uint8_t mx = (a < b) ? b : a;
    if (c >= mx) return mn;
    if (c <= mn) return mx;
    return (uint8_t)((int32_t)a + (int32_t)b - (int32_t)c);
}

static inline void zcrudp_iec_enc_residual(zcrudp_rc_enc_t *e, zcrudp_iec_model_t *m,
                                           uint32_t bucket, uint32_t sign_ctx, int32_t res) {
    if (res == 0) {
        zcrudp_rc_enc_bit(e, &m->zero[bucket], 0U);
        return;
    }
    zcrudp_rc_enc_bit(e, &m->zero[bucket], 1U);
    zcrudp_rc_enc_bit(e, &m->sign[sign_ctx], (res < 0) ? 1U : 0U);
    uint32_t mag = (uint32_t)((res < 0) ? -res : res) - 1U;
    zcrudp_rc_enc_egk(e, m->mag[bucket], ZCRUDP_IEC_MAG_CTX, ZCRUDP_IEC_MAG_MAXLEN, mag);
}

/** @return residual within [-255, 255]; the caller wraps it modulo 256. */
static inline int32_t zcrudp_iec_dec_residual(zcrudp_rc_dec_t *d, zcrudp_iec_model_t *m,
                                              uint32_t bucket, uint32_t sign_ctx) {
    if (zcrudp_rc_dec_bit(d, &m->zero[bucket]) == 0U) {
        return 0;
    }
    uint32_t neg = zcrudp_rc_dec_bit(d, &m->sign[sign_ctx]);
    int32_t mag = (int32_t)zcrudp_rc_dec_egk(d, m->mag[bucket], ZCRUDP_IEC_MAG_CTX,
                                             ZCRUDP_IEC_MAG_MAXLEN) + 1;
    return neg ? -mag : mag;
}

#ifdef __cplusplus
}
#endif

#endif /* ZCRUDP_TERNARY_ENTROPY_H */
