/*
 * zcvid_gpu_cpu_dec.h - Portable C11 CPU Decoder for zcvid GPU-encoded streams.
 *
 * Bit-exact integer implementation of zcvid/shaders/zcvid_decode.hlsl:
 *   - Pass 1: Motion-compensated reference (quarter-pel 4-tap HEVC filter) +
 *             leaky decay + adaptive step halving.
 *   - Pass 2: Delta 1x/2x base-3 trit records, 2-pixel 4:2:2 8-bit intra,
 *             4-pixel 6-bit intra (exact neutral chroma: q6 << 2), and
 *             4x4 H.264 integer transform coefficients.
 *   - Pass 2b: 4x4 inverse integer transform (intra or motion-predicted inter).
 *   - Pass 3: Optional display-only deblocking + YCbCr -> RGBA8 conversion.
 *
 * Strictly portable C11: no OS headers, no SIMD intrinsics, no D3D11/Vulkan/Metal.
 * Works identically on Linux, macOS, Windows, x86_64, ARM64 (Apple Silicon), and RISC-V.
 */
#ifndef ZCVID_GPU_CPU_DEC_H
#define ZCVID_GPU_CPU_DEC_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int32_t x;
  int32_t y;
} zcvid_mv_t;

typedef struct {
  uint32_t width;
  uint32_t height;
  uint32_t delta_min;
  uint32_t delta_max;
  uint32_t gamma_q8;
  uint32_t neutral_y;
  uint32_t mv_qpel;  /* 1: tile_mvs are in quarter-pels, 0: whole-pels */
  uint32_t post_thr; /* display-only deblocking threshold (0 = off) */
} zcvid_gpu_dec_params_t;

typedef struct {
  uint32_t width;
  uint32_t height;
  uint32_t *prev_yuvd; /* holds last decoded frame after decode_frame() */
  uint32_t *cur_yuvd;  /* scratch working buffer, swapped with prev_yuvd */
  int32_t *coef;       /* 48 dequantized coefficients per 4x4 block */
  uint8_t *tflag;      /* 0: none, 1: intra transform, 2: inter transform */
  uint32_t nblocks;
} zcvid_gpu_dec_state_t;

static inline int32_t zcvid_dec_clamp_i32(int32_t v, int32_t lo, int32_t hi) {
  return (v < lo) ? lo : ((v > hi) ? hi : v);
}

static inline uint32_t zcvid_dec_tq_class(uint32_t r) {
  uint32_t i = r >> 2;
  uint32_t j = r & 3u;
  if (((i | j) & 1u) == 0u) return 0u;
  if ((i & j & 1u) != 0u) return 1u;
  return 2u;
}

static inline int32_t zcvid_dec_tq_dequant(int32_t level, uint32_t r, uint32_t qp) {
  static const int32_t TQ_V[18] = {
    10, 16, 13,  11, 18, 14,  13, 20, 16,
    14, 23, 18,  16, 25, 20,  18, 29, 23
  };
  int32_t scaled = level * TQ_V[(qp % 6u) * 3u + zcvid_dec_tq_class(r)];
  uint32_t sh = qp / 6u;
  /* Signed left shift avoided via two's-complement unsigned shift. */
  return (int32_t)((uint32_t)scaled << sh);
}

static inline void zcvid_dec_tq_inverse(int32_t c[16]) {
  for (uint32_t i = 0; i < 4u; ++i) {
    int32_t d0 = c[i * 4u], d1 = c[i * 4u + 1u], d2 = c[i * 4u + 2u], d3 = c[i * 4u + 3u];
    int32_t e0 = d0 + d2, e1 = d0 - d2;
    int32_t e2 = (d1 >> 1) - d3, e3 = d1 + (d3 >> 1);
    c[i * 4u]      = e0 + e3;
    c[i * 4u + 1u] = e1 + e2;
    c[i * 4u + 2u] = e1 - e2;
    c[i * 4u + 3u] = e0 - e3;
  }
  for (uint32_t j = 0; j < 4u; ++j) {
    int32_t d0 = c[j], d1 = c[4u + j], d2 = c[8u + j], d3 = c[12u + j];
    int32_t e0 = d0 + d2, e1 = d0 - d2;
    int32_t e2 = (d1 >> 1) - d3, e3 = d1 + (d3 >> 1);
    c[j]      = e0 + e3;
    c[4u + j] = e1 + e2;
    c[8u + j] = e1 - e2;
    c[12u + j] = e0 - e3;
  }
  for (uint32_t k = 0; k < 16u; ++k) {
    c[k] = (c[k] + 32) >> 6;
  }
}

static inline int32_t zcvid_dec_tq_sample8x8(const int32_t c[16], uint32_t lx, uint32_t ly) {
  uint32_t u = lx >> 1;
  uint32_t v = ly >> 1;
  uint32_t un = ((lx & 1u) == 0u) ? ((u > 0u) ? (u - 1u) : 0u) : ((u < 3u) ? (u + 1u) : 3u);
  uint32_t vn = ((ly & 1u) == 0u) ? ((v > 0u) ? (v - 1u) : 0u) : ((v < 3u) ? (v + 1u) : 3u);
  return (9 * c[v * 4u + u] + 3 * c[v * 4u + un] + 3 * c[vn * 4u + u] + c[vn * 4u + un] + 8) >> 4;
}

/* Exact 8x8 HEVC DCT8 dequantizer (proved overflow-free in ZcvidDCT8Circuit.lean Thm 7a). */
static inline int32_t zcvid_dec_tq8_dequant(int32_t level, uint32_t qp) {
  static const int32_t TQ8_V[6] = { 40, 45, 51, 57, 64, 72 };
  uint32_t qp_mod = qp % 6u;
  uint32_t qp_div = qp / 6u;
  int32_t dq_step = (qp_div >= 2u)
                        ? (TQ8_V[qp_mod] << (qp_div - 2u))
                        : (((TQ8_V[qp_mod] << qp_div) + 2) >> 2);
  int32_t v = level * dq_step;
  return zcvid_dec_clamp_i32(v, -65535, 65535);
}

/* Exact 12-mul 1D HEVC IDCT8 (Negacyclic-Karatsuba 9-mul odd + Gauss 3-mul even).
 * Shared bit-for-bit with zcvid_encode.hlsl, zcvid_decode.hlsl, and ZcvidDCT8Circuit.lean. */
static inline void zcvid_dec_dct8_inv_1d(int32_t c0, int32_t c1, int32_t c2, int32_t c3,
                                         int32_t c4, int32_t c5, int32_t c6, int32_t c7,
                                         int32_t rnd, int sh,
                                         int32_t *out, uint32_t stride) {
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
  int32_t O0 = d0 - k1;
  int32_t O1 = d0 + d1 - d2;
  int32_t O2 = k0 + k1 + u2 - m8;
  int32_t O3 = -k0 - d1;

  out[0u * stride] = (E0 + O0 + rnd) >> sh;
  out[1u * stride] = (E1 + O1 + rnd) >> sh;
  out[2u * stride] = (E2 + O2 + rnd) >> sh;
  out[3u * stride] = (E3 + O3 + rnd) >> sh;
  out[4u * stride] = (E3 - O3 + rnd) >> sh;
  out[5u * stride] = (E2 - O2 + rnd) >> sh;
  out[6u * stride] = (E1 - O1 + rnd) >> sh;
  out[7u * stride] = (E0 - O0 + rnd) >> sh;
}

/* Exact 2D 8x8 HEVC IDCT8 with Lean-4-proved sparse fast-paths (DC-only, LL-only, general). */
static inline void zcvid_dec_tq8_inverse(const int32_t in[64], uint32_t csbf, int32_t out[64]) {
  if (csbf == 0u) {
    memset(out, 0, sizeof(int32_t) * 64u);
    return;
  }
  if (csbf == 1u) {
    int32_t ac_or = in[1] | in[2] | in[3] |
                    in[8] | in[9] | in[10] | in[11] |
                    in[16] | in[17] | in[18] | in[19] |
                    in[24] | in[25] | in[26] | in[27];
    if (ac_or == 0) {
      int32_t t0 = (64 * in[0] + 64) >> 7;
      int32_t v = (64 * t0 + 2048) >> 12;
      for (uint32_t i = 0u; i < 64u; ++i) out[i] = v;
      return;
    }
    static const int32_t k1[8] = { 89, 75, 50, 18, -18, -50, -75, -89 };
    static const int32_t k2[8] = { 83, 36, -36, -83, -83, -36, 36, 83 };
    static const int32_t k3[8] = { 75, -18, -89, -50, 50, 89, 18, -75 };
    int32_t t[4][8];
    for (uint32_t v = 0u; v < 4u; ++v) {
      const int32_t *row = in + v * 8u;
      for (uint32_t x = 0u; x < 8u; ++x) {
        t[v][x] = (64 * row[0] + k1[x] * row[1] + k2[x] * row[2] + k3[x] * row[3] + 64) >> 7;
      }
    }
    for (uint32_t y = 0u; y < 4u; ++y) {
      for (uint32_t x = 0u; x < 8u; ++x) {
        int32_t ev = 64 * t[0][x] + k2[y] * t[2][x];
        int32_t od = k1[y] * t[1][x] + k3[y] * t[3][x];
        out[y * 8u + x] = (ev + od + 2048) >> 12;
        out[(7u - y) * 8u + x] = (ev - od + 2048) >> 12;
      }
    }
    return;
  }
  int32_t tmp[64];
  uint32_t v_end = ((csbf & 12u) != 0u) ? 8u : 4u;
  for (uint32_t v = 0u; v < v_end; ++v) {
    const int32_t *c = in + v * 8u;
    if ((c[0] | c[1] | c[2] | c[3] | c[4] | c[5] | c[6] | c[7]) == 0) {
      memset(tmp + v * 8u, 0, sizeof(int32_t) * 8u);
      continue;
    }
    zcvid_dec_dct8_inv_1d(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                          64, 7, tmp + v * 8u, 1u);
  }
  if (v_end < 8u) {
    memset(tmp + 32u, 0, sizeof(int32_t) * 32u);
  }
  for (uint32_t x = 0u; x < 8u; ++x) {
    zcvid_dec_dct8_inv_1d(tmp[0u * 8u + x], tmp[1u * 8u + x], tmp[2u * 8u + x], tmp[3u * 8u + x],
                          tmp[4u * 8u + x], tmp[5u * 8u + x], tmp[6u * 8u + x], tmp[7u * 8u + x],
                          2048, 12, out + x, 8u);
  }
}

static inline int32_t zcvid_dec_leaky(int32_t prev_val, int32_t neutral, uint32_t gamma_q8) {
  int32_t diff = prev_val - neutral;
  return zcvid_dec_clamp_i32(neutral + (diff * (int32_t)gamma_q8) / 256, 0, 255);
}

static inline uint32_t zcvid_dec_ref_word_at(const zcvid_gpu_dec_state_t *st,
                                             int32_t x, int32_t y) {
  x = zcvid_dec_clamp_i32(x, 0, (int32_t)st->width - 1);
  y = zcvid_dec_clamp_i32(y, 0, (int32_t)st->height - 1);
  return st->prev_yuvd[(uint32_t)y * st->width + (uint32_t)x];
}

static inline void zcvid_dec_taps_q(int32_t f, int32_t t[4]) {
  if (f == 1)      { t[0] = -4; t[1] = 54; t[2] = 16; t[3] = -2; }
  else if (f == 2) { t[0] = -4; t[1] = 36; t[2] = 36; t[3] = -4; }
  else if (f == 3) { t[0] = -2; t[1] = 16; t[2] = 54; t[3] = -4; }
  else             { t[0] =  0; t[1] = 64; t[2] =  0; t[3] =  0; }
}

static inline uint32_t zcvid_dec_fetch_ref_q(const zcvid_gpu_dec_state_t *st,
                                             uint32_t px, uint32_t py,
                                             zcvid_mv_t mvq) {
  int32_t sxq = (int32_t)px * 4 - mvq.x;
  int32_t syq = (int32_t)py * 4 - mvq.y;
  /* Arithmetic floor division by 4 (matches HLSL >> 2 on signed int). */
  int32_t ix = (sxq >= 0) ? (sxq >> 2) : -(((-sxq) + 3) >> 2);
  int32_t iy = (syq >= 0) ? (syq >> 2) : -(((-syq) + 3) >> 2);
  int32_t fx = sxq - ix * 4;
  int32_t fy = syq - iy * 4;
  uint32_t w00 = zcvid_dec_ref_word_at(st, ix, iy);
  if ((fx | fy) == 0) return w00;

  int32_t tx[4], ty[4];
  zcvid_dec_taps_q(fx, tx);
  zcvid_dec_taps_q(fy, ty);
  int32_t acc0 = 0, acc1 = 0, acc2 = 0;
  for (int32_t j = 0; j < 4; ++j) {
    int32_t h0 = 0, h1 = 0, h2 = 0;
    for (int32_t i = 0; i < 4; ++i) {
      uint32_t w = zcvid_dec_ref_word_at(st, ix - 1 + i, iy - 1 + j);
      int32_t t = tx[i];
      h0 += t * (int32_t)(w & 0xFFu);
      h1 += t * (int32_t)((w >> 8) & 0xFFu);
      h2 += t * (int32_t)((w >> 16) & 0xFFu);
    }
    acc0 += ty[j] * h0;
    acc1 += ty[j] * h1;
    acc2 += ty[j] * h2;
  }
  return (w00 & 0xFF000000u) |
         (uint32_t)zcvid_dec_clamp_i32((acc0 + 2048) >> 12, 0, 255) |
         ((uint32_t)zcvid_dec_clamp_i32((acc1 + 2048) >> 12, 0, 255) << 8) |
         ((uint32_t)zcvid_dec_clamp_i32((acc2 + 2048) >> 12, 0, 255) << 16);
}

static inline uint32_t zcvid_dec_prev_shifted(const zcvid_gpu_dec_state_t *st,
                                              const zcvid_gpu_dec_params_t *p,
                                              const zcvid_mv_t *tile_mvs,
                                              uint32_t tiles_x, uint32_t idx) {
  uint32_t py = idx / st->width;
  uint32_t px = idx - py * st->width;
  zcvid_mv_t mv = {0, 0};
  if (tile_mvs) {
    mv = tile_mvs[(py >> 5) * tiles_x + (px >> 5)];
  }
  if (!p->mv_qpel) {
    mv.x *= 4;
    mv.y *= 4;
  }
  return zcvid_dec_fetch_ref_q(st, px, py, mv);
}

static inline int32_t zcvid_dec_step_of(uint32_t ref_w, uint32_t delta_min) {
  int32_t s = (int32_t)((ref_w >> 24) & 0xFFu);
  return (s > (int32_t)delta_min) ? s : (int32_t)delta_min;
}

static inline void zcvid_dec_apply_trit(zcvid_gpu_dec_state_t *st,
                                        const zcvid_gpu_dec_params_t *p,
                                        uint32_t idx, int32_t t, int32_t d_used,
                                        uint32_t tag, uint32_t ref_w) {
  if (t == 0 || idx >= st->width * st->height) return;
  uint32_t shift = tag * 8u;
  int32_t neutral = (tag == 0u) ? (int32_t)p->neutral_y : 128;
  int32_t prev = (int32_t)((ref_w >> shift) & 0xFFu);
  int32_t v = zcvid_dec_clamp_i32(
      zcvid_dec_leaky(prev, neutral, p->gamma_q8) + t * d_used, 0, 255);

  uint32_t clear_mask = ~(0xFFu << shift);
  uint32_t set_bits = (uint32_t)v << shift;
  if (tag == 0u) {
    clear_mask &= 0x00FFFFFFu;
    int32_t d_next = d_used << 1;
    if (d_next > (int32_t)p->delta_max) d_next = (int32_t)p->delta_max;
    set_bits |= (uint32_t)d_next << 24;
  }
  st->cur_yuvd[idx] = (st->cur_yuvd[idx] & clear_mask) | set_bits;
}

static inline int zcvid_gpu_dec_init(zcvid_gpu_dec_state_t *st,
                                     uint32_t width, uint32_t height) {
  if (!st || width == 0u || height == 0u) return -1;
  memset(st, 0, sizeof(*st));
  size_t num_px = (size_t)width * (size_t)height;
  uint32_t bw = (width + 3u) >> 2;
  uint32_t bh = (height + 3u) >> 2;
  uint32_t nblocks = bw * bh;

  st->prev_yuvd = (uint32_t *)calloc(num_px, sizeof(uint32_t));
  st->cur_yuvd  = (uint32_t *)calloc(num_px, sizeof(uint32_t));
  st->coef      = (int32_t *)calloc((size_t)nblocks * 48u, sizeof(int32_t));
  st->tflag     = (uint8_t *)calloc(nblocks, sizeof(uint8_t));
  if (!st->prev_yuvd || !st->cur_yuvd || !st->coef || !st->tflag) {
    free(st->prev_yuvd);
    free(st->cur_yuvd);
    free(st->coef);
    free(st->tflag);
    memset(st, 0, sizeof(*st));
    return -1;
  }
  st->width = width;
  st->height = height;
  st->nblocks = nblocks;
  return 0;
}

static inline void zcvid_gpu_dec_reset(zcvid_gpu_dec_state_t *st,
                                       uint32_t neutral_y, uint32_t delta_min) {
  if (!st || !st->prev_yuvd) return;
  uint32_t fill = (neutral_y & 0xFFu) | (128u << 8) | (128u << 16) |
                  ((delta_min & 0xFFu) << 24);
  size_t num_px = (size_t)st->width * (size_t)st->height;
  for (size_t i = 0; i < num_px; ++i) {
    st->prev_yuvd[i] = fill;
    st->cur_yuvd[i]  = fill;
  }
  memset(st->coef, 0, (size_t)st->nblocks * 48u * sizeof(int32_t));
  memset(st->tflag, 0, st->nblocks);
}

static inline void zcvid_gpu_dec_destroy(zcvid_gpu_dec_state_t *st) {
  if (!st) return;
  free(st->prev_yuvd);
  free(st->cur_yuvd);
  free(st->coef);
  free(st->tflag);
  memset(st, 0, sizeof(*st));
}

/*
 * Decodes one frame of zcvid GPU records (2 uint32 words per record) plus
 * optional 32x32 tile motion vectors. After returning 0, st->prev_yuvd
 * contains the reconstructed YCbCr+delta frame (matching zcvid_decode.hlsl).
 */
static inline int zcvid_gpu_dec_decode_frame(zcvid_gpu_dec_state_t *st,
                                             const zcvid_gpu_dec_params_t *p,
                                             const zcvid_mv_t *tile_mvs,
                                             const uint32_t *records,
                                             uint32_t record_words) {
  if (!st || !p || !st->prev_yuvd || st->width != p->width || st->height != p->height) {
    return -1;
  }
  static const uint32_t TQ_ZZ[16] = {
    0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15
  };
  static const uint32_t RECIP[5] = { 21846u, 7282u, 2428u, 810u, 0u };

  uint32_t w = st->width;
  uint32_t h = st->height;
  uint32_t num_px = w * h;
  uint32_t tiles_x = (w + 31u) >> 5;
  uint32_t bw = (w + 3u) >> 2;
  uint32_t bh = (h + 3u) >> 2;

  /* Pass 1: CSMotionDecay */
  for (uint32_t py = 0; py < h; ++py) {
    for (uint32_t px = 0; px < w; ++px) {
      uint32_t idx = py * w + px;
      uint32_t rw = zcvid_dec_prev_shifted(st, p, tile_mvs, tiles_x, idx);
      int32_t y  = zcvid_dec_leaky((int32_t)(rw & 0xFFu), (int32_t)p->neutral_y, p->gamma_q8);
      int32_t cb = zcvid_dec_leaky((int32_t)((rw >> 8) & 0xFFu), 128, p->gamma_q8);
      int32_t cr = zcvid_dec_leaky((int32_t)((rw >> 16) & 0xFFu), 128, p->gamma_q8);
      int32_t d  = zcvid_dec_step_of(rw, p->delta_min) >> 1;
      if (d < (int32_t)p->delta_min) d = (int32_t)p->delta_min;
      st->cur_yuvd[idx] = (uint32_t)y | ((uint32_t)cb << 8) |
                          ((uint32_t)cr << 16) | ((uint32_t)d << 24);
    }
  }

  /* Pass 2: CSApply */
  uint32_t any_transform = 0u;
  for (uint32_t rec_idx = 0; rec_idx + 1u < record_words; rec_idx += 2u) {
    uint32_t w0 = records[rec_idx + 0u];
    uint32_t w1 = records[rec_idx + 1u];
    uint32_t tag   = (w0 >> 30) & 3u;
    uint32_t is_2x = (w0 >> 29) & 1u;
    uint32_t id    = w0 & 0x1FFFFFFFu;

    if (tag == 3u) {
      if (is_2x != 0u && ((w0 >> 28) & 1u) != 0u) {
        uint32_t pid  = w0 & 0x0FFFFFFFu;
        uint32_t b    = pid / 48u;
        uint32_t qp   = (w1 >> 16) & 0xFFu;
        uint32_t pred = (w1 >> 24) & 0xFFu;
        if (b >= st->nblocks || qp > 51u || pred > 5u) continue;
        int32_t level = (int16_t)(w1 & 0xFFFFu);
        if (pred >= 4u) {
          if (level < -2047 || level > 2047) continue;
          uint32_t bx0 = (b % bw) & ~1u;
          uint32_t by0 = (b / bw) & ~1u;
          if (bx0 + 1u >= bw || by0 + 1u >= bh) continue;
          st->coef[pid] = zcvid_dec_tq8_dequant(level, qp);
          st->tflag[by0 * bw + bx0] = (uint8_t)(1u + pred);
        } else {
          if (level < -255 || level > 255) continue;
          uint32_t r = TQ_ZZ[pid % 16u];
          st->coef[pid] = zcvid_dec_tq_dequant(level, r, qp);
          st->tflag[b]  = (uint8_t)(1u + pred);
        }
        any_transform = 1u;
        continue;
      }
      if (is_2x != 0u) {
        uint32_t idx4  = (w0 & 0x3FFFFFu) << 1;
        uint32_t cr_q6 = (w0 >> 22) & 0x3Fu;
        uint32_t y0_q6 = w1 & 0x3Fu;
        uint32_t y1_q6 = (w1 >> 6) & 0x3Fu;
        uint32_t y2_q6 = (w1 >> 12) & 0x3Fu;
        uint32_t y3_q6 = (w1 >> 18) & 0x3Fu;
        uint32_t cb_q6 = (w1 >> 24) & 0x3Fu;
        uint32_t cb = cb_q6 << 2;
        uint32_t cr = cr_q6 << 2;
        uint32_t chroma_d = (cb << 8) | (cr << 16) | (p->delta_min << 24);
        if (idx4 + 0u < num_px) st->cur_yuvd[idx4 + 0u] = ((y0_q6 << 2) | (y0_q6 >> 4)) | chroma_d;
        if (idx4 + 1u < num_px) st->cur_yuvd[idx4 + 1u] = ((y1_q6 << 2) | (y1_q6 >> 4)) | chroma_d;
        if (idx4 + 2u < num_px) st->cur_yuvd[idx4 + 2u] = ((y2_q6 << 2) | (y2_q6 >> 4)) | chroma_d;
        if (idx4 + 3u < num_px) st->cur_yuvd[idx4 + 3u] = ((y3_q6 << 2) | (y3_q6 >> 4)) | chroma_d;
      } else {
        uint32_t y0 = w1 & 0xFFu;
        uint32_t cb = (w1 >> 8) & 0xFFu;
        uint32_t y1 = (w1 >> 16) & 0xFFu;
        uint32_t cr = (w1 >> 24) & 0xFFu;
        uint32_t chroma_d = (cb << 8) | (cr << 16) | (p->delta_min << 24);
        if (id < num_px)      st->cur_yuvd[id]      = y0 | chroma_d;
        if (id + 1u < num_px) st->cur_yuvd[id + 1u] = y1 | chroma_d;
      }
      continue;
    }

    uint32_t px_per_blk = (is_2x != 0u) ? 20u : 10u;
    if (w1 >= 59049u || id >= (num_px + px_per_blk - 1u) / px_per_blk) continue;

    uint32_t base_px = id * px_per_blk;
    uint32_t hi = (w1 * 69043u) >> 24;
    uint32_t lo = w1 - hi * 243u;
    uint32_t step_px = (is_2x != 0u) ? 2u : 1u;
    uint32_t halves[2] = { lo, hi };

    for (uint32_t hh = 0; hh < 2u; ++hh) {
      uint32_t v = halves[hh];
      uint32_t prev_q = v;
      for (uint32_t k = 0; k < 5u; ++k) {
        uint32_t q = (v * RECIP[k]) >> 16;
        int32_t t = (int32_t)(prev_q - q * 3u) - 1;
        prev_q = q;

        uint32_t j = hh * 5u + k;
        uint32_t idx = base_px + j * step_px;
        if (t != 0 && idx < num_px) {
          uint32_t ref_w = zcvid_dec_prev_shifted(st, p, tile_mvs, tiles_x, idx);
          int32_t d = zcvid_dec_step_of(ref_w, p->delta_min);
          zcvid_dec_apply_trit(st, p, idx, t, d, tag, ref_w);
          if (is_2x != 0u && idx + 1u < num_px) {
            uint32_t ref_w1 = zcvid_dec_prev_shifted(st, p, tile_mvs, tiles_x, idx + 1u);
            zcvid_dec_apply_trit(st, p, idx + 1u, t, d, tag, ref_w1);
          }
        }
      }
    }
  }

  /* Pass 2b: CSITrans (4x4 mode 1..2, legacy 8x8 mode 3..4, exact 8x8 DCT8 mode 5..6) */
  if (any_transform) {
    for (uint32_t b = 0; b < bw * bh; ++b) {
      uint32_t mode = st->tflag[b];
      if (mode == 0u) continue;
      st->tflag[b] = 0u;
      uint32_t bx0 = b % bw;
      uint32_t by0 = b / bw;
      uint32_t x0 = bx0 * 4u;
      uint32_t y0 = by0 * 4u;
      int is_inter = (mode == 2u) || (mode == 4u) || (mode == 6u);

      if (mode <= 2u) {
        int inside = (x0 + 4u <= w) && (y0 + 4u <= h);
        uint32_t cur[16];
        for (uint32_t j0 = 0; j0 < 16u; ++j0) {
          cur[j0] = inside ? st->cur_yuvd[(y0 + (j0 >> 2)) * w + x0 + (j0 & 3u)] : 0x00808080u;
        }
        int32_t rec[48];
        for (uint32_t pl = 0; pl < 3u; ++pl) {
          int32_t c[16];
          for (uint32_t k = 0; k < 16u; ++k) {
            uint32_t pid = b * 48u + pl * 16u + k;
            c[TQ_ZZ[k]] = st->coef[pid];
            st->coef[pid] = 0;
          }
          zcvid_dec_tq_inverse(c);
          for (uint32_t q = 0; q < 16u; ++q) {
            int32_t pv = is_inter ? (int32_t)((cur[q] >> (pl * 8u)) & 0xFFu) : 128;
            rec[pl * 16u + q] = zcvid_dec_clamp_i32(pv + c[q], 0, 255);
          }
        }
        if (!inside) continue;
        for (uint32_t j = 0; j < 16u; ++j) {
          uint32_t idx = (y0 + (j >> 2)) * w + x0 + (j & 3u);
          st->cur_yuvd[idx] = (uint32_t)rec[j] |
                              ((uint32_t)rec[16u + j] << 8) |
                              ((uint32_t)rec[32u + j] << 16) |
                              (p->delta_min << 24);
        }
      } else if (mode <= 4u) {
        int inside8 = (x0 + 8u <= w) && (y0 + 8u <= h);
        int32_t c_pl[3][16];
        for (uint32_t pl = 0; pl < 3u; ++pl) {
          for (uint32_t k = 0; k < 16u; ++k) {
            uint32_t pid = b * 48u + pl * 16u + k;
            c_pl[pl][TQ_ZZ[k]] = st->coef[pid];
            st->coef[pid] = 0;
          }
          zcvid_dec_tq_inverse(c_pl[pl]);
        }
        if (!inside8) continue;
        for (uint32_t ly = 0; ly < 8u; ++ly) {
          for (uint32_t lx = 0; lx < 8u; ++lx) {
            uint32_t idx = (y0 + ly) * w + x0 + lx;
            uint32_t pw = is_inter ? st->cur_yuvd[idx] : 0x00808080u;
            int32_t ry = zcvid_dec_clamp_i32((int32_t)(pw & 0xFFu) + zcvid_dec_tq_sample8x8(c_pl[0], lx, ly), 0, 255);
            int32_t rb = zcvid_dec_clamp_i32((int32_t)((pw >> 8) & 0xFFu) + zcvid_dec_tq_sample8x8(c_pl[1], lx, ly), 0, 255);
            int32_t rr = zcvid_dec_clamp_i32((int32_t)((pw >> 16) & 0xFFu) + zcvid_dec_tq_sample8x8(c_pl[2], lx, ly), 0, 255);
            st->cur_yuvd[idx] = (uint32_t)ry | ((uint32_t)rb << 8) | ((uint32_t)rr << 16) | (p->delta_min << 24);
          }
        }
      } else {
        /* Exact 8x8 Factored HEVC DCT8 (mode 5 = intra, mode 6 = inter). */
        int valid_sb = ((bx0 & 1u) == 0u) && ((by0 & 1u) == 0u) && (bx0 + 1u < bw) && (by0 + 1u < bh);
        if (!valid_sb) continue;
        int inside8 = (x0 + 8u <= w) && (y0 + 8u <= h);
        uint32_t bq[4] = { b, b + 1u, b + bw, b + bw + 1u };
        int32_t res_pl[3][64];
        for (uint32_t pl = 0u; pl < 3u; ++pl) {
          int32_t w_in[64];
          uint32_t csbf = 0u;
          for (uint32_t q = 0u; q < 4u; ++q) {
            uint32_t u0 = (q & 1u) * 4u;
            uint32_t v0 = (q >> 1) * 4u;
            uint32_t base_pid = bq[q] * 48u + pl * 16u;
            int32_t nz_or = 0;
            for (uint32_t k = 0u; k < 16u; ++k) {
              int32_t cv = st->coef[base_pid + k];
              st->coef[base_pid + k] = 0;
              uint32_t r = TQ_ZZ[k];
              w_in[(v0 + (r >> 2)) * 8u + (u0 + (r & 3u))] = cv;
              nz_or |= cv;
            }
            if (nz_or != 0) csbf |= (1u << q);
          }
          if (inside8) {
            zcvid_dec_tq8_inverse(w_in, csbf, res_pl[pl]);
          }
        }
        if (!inside8) continue;
        for (uint32_t ly = 0u; ly < 8u; ++ly) {
          for (uint32_t lx = 0u; lx < 8u; ++lx) {
            uint32_t idx = (y0 + ly) * w + x0 + lx;
            uint32_t pw = is_inter ? st->cur_yuvd[idx] : 0x00808080u;
            uint32_t p8 = ly * 8u + lx;
            int32_t ry = zcvid_dec_clamp_i32((int32_t)(pw & 0xFFu) + res_pl[0][p8], 0, 255);
            int32_t rb = zcvid_dec_clamp_i32((int32_t)((pw >> 8) & 0xFFu) + res_pl[1][p8], 0, 255);
            int32_t rr = zcvid_dec_clamp_i32((int32_t)((pw >> 16) & 0xFFu) + res_pl[2][p8], 0, 255);
            st->cur_yuvd[idx] = (uint32_t)ry | ((uint32_t)rb << 8) | ((uint32_t)rr << 16) | (p->delta_min << 24);
          }
        }
      }
    }
  }

  /* Swap prev_yuvd and cur_yuvd so prev_yuvd holds the newly decoded frame. */
  uint32_t *tmp = st->prev_yuvd;
  st->prev_yuvd = st->cur_yuvd;
  st->cur_yuvd = tmp;
  return 0;
}

/*
 * Pass 3: CSYCbCrToRGBA (with optional display-only deblocking).
 * Writes width * height * 4 bytes (R, G, B, 255) to out_rgba.
 */
static inline void zcvid_gpu_dec_to_rgba(const zcvid_gpu_dec_state_t *st,
                                         const zcvid_gpu_dec_params_t *p,
                                         uint8_t *out_rgba) {
  if (!st || !p || !st->prev_yuvd || !out_rgba) return;
  uint32_t w = st->width;
  uint32_t h = st->height;
  for (uint32_t yy = 0; yy < h; ++yy) {
    for (uint32_t x = 0; x < w; ++x) {
      uint32_t idx = yy * w + x;
      uint32_t word = st->prev_yuvd[idx];
      int32_t yc = (int32_t)(word & 0xFFu);
      int32_t cbc = (int32_t)((word >> 8) & 0xFFu);
      int32_t crc = (int32_t)((word >> 16) & 0xFFu);
      if (p->post_thr != 0u) {
        int32_t adj = 0, adj_cb = 0, adj_cr = 0;
        int hi_qp = (p->post_thr > 6u);
        int32_t base_thr = (p->post_thr < 6u) ? (int32_t)p->post_thr : 6;
        int32_t beta = ((int32_t)(p->post_thr >> 1) > 4) ? (int32_t)(p->post_thr >> 1) : 4;
        int32_t c_thr = (int32_t)(p->post_thr >> 1);
        int32_t nx = ((x & 3u) == 0u && x > 0u) ? -1 : (((x & 3u) == 3u && x + 1u < w) ? 1 : 0);
        int32_t ny = ((yy & 3u) == 0u && yy > 0u) ? -(int32_t)w : (((yy & 3u) == 3u && yy + 1u < h) ? (int32_t)w : 0);
        int ok_in[2] = {
          (nx < 0) ? (x + 1u < w && x >= 2u) : (nx > 0 ? (x > 0u && x + 2u < w) : 0),
          (ny < 0) ? (yy + 1u < h && yy >= 2u) : (ny > 0 ? (yy > 0u && yy + 2u < h) : 0)
        };
        int32_t n_off[2] = { nx, ny };
        for (uint32_t k = 0u; k < 2u; ++k) {
          if (n_off[k] != 0) {
            uint32_t wn = st->prev_yuvd[(uint32_t)((int32_t)idx + n_off[k])];
            int32_t yn = (int32_t)(wn & 0xFFu);
            int32_t d = yn - yc;
            int32_t lim = base_thr;
            int smooth_side = 0;
            if (hi_qp && ok_in[k]) {
              int32_t p1 = (int32_t)(st->prev_yuvd[(uint32_t)((int32_t)idx - n_off[k])] & 0xFFu);
              int32_t q1 = (int32_t)(st->prev_yuvd[(uint32_t)((int32_t)idx + 2 * n_off[k])] & 0xFFu);
              int32_t dp = yc - p1; if (dp < 0) dp = -dp;
              int32_t dq = yn - q1; if (dq < 0) dq = -dq;
              if (dp <= beta && dq <= beta) {
                lim = (int32_t)p->post_thr;
                smooth_side = 1;
              }
            }
            if ((d < 0 ? -d : d) <= lim) adj += d / 4;
            if (smooth_side) {
              int32_t dcb = (int32_t)((wn >> 8) & 0xFFu) - cbc;
              int32_t dcr = (int32_t)((wn >> 16) & 0xFFu) - crc;
              if ((dcb < 0 ? -dcb : dcb) <= c_thr) adj_cb += dcb / 4;
              if ((dcr < 0 ? -dcr : dcr) <= c_thr) adj_cr += dcr / 4;
            }
          }
        }
        yc = zcvid_dec_clamp_i32(yc + adj, 0, 255);
        if (hi_qp) {
          cbc = zcvid_dec_clamp_i32(cbc + adj_cb, 0, 255);
          crc = zcvid_dec_clamp_i32(crc + adj_cr, 0, 255);
        }
      }
      float yf = (float)yc;
      float cb = (float)cbc - 128.0f;
      float cr = (float)crc - 128.0f;

      int32_t r = zcvid_dec_clamp_i32((int32_t)(yf + 1.402f * cr + 0.5f), 0, 255);
      int32_t g = zcvid_dec_clamp_i32((int32_t)(yf - 0.344136f * cb - 0.714136f * cr + 0.5f), 0, 255);
      int32_t b = zcvid_dec_clamp_i32((int32_t)(yf + 1.772f * cb + 0.5f), 0, 255);

      uint8_t *dst = out_rgba + (size_t)idx * 4u;
      dst[0] = (uint8_t)r;
      dst[1] = (uint8_t)g;
      dst[2] = (uint8_t)b;
      dst[3] = 255u;
    }
  }
}

#ifdef __cplusplus
}
#endif

#endif /* ZCVID_GPU_CPU_DEC_H */
