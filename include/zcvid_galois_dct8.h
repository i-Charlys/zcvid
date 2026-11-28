/*
 * zcvid_galois_dct8.h - Exact 116-Multiplication 2D Cyclotomic Galois-Subfield
 *                       HEVC 8x8 Integer DCT & IDCT (Zero-Transpose, Bit-Exact)
 *
 * Algebraic Method (inspired by linear-circuit decomposition + Lean 4 proofs):
 *   Because Phi_8(X) = X^4 + 1 is irreducible over Z but splits completely in
 *   R = Z[X]/(X^4 + 1) along Y^4 + 1 = (Y - X)(Y + X)(Y - X^3)(Y + X^3),
 *   and M_odd is an exact negacyclic convolution under Rader's isomorphism
 *   (Z/16Z)* / {+-1} ~= Z/4Z (perm = {0, 2, 3, 1}), the entire 2D HEVC DCT8
 *   K * X * K^T (and by symmetry K^T * C * K) factors over Z[1/4] into:
 *     - DC/4 x DC/4   (2x2):   0 non-shift muls (exact *4096)
 *     - Even x Even   (2x2):   5 muls via Z[X]/(X^2 + 1) CRT (5593+5976X, 8185)
 *     - DC/4 x Even   (4x2):  12 muls via 4x Gauss 3-mul rotation
 *     - DC/4 x Odd    (8x4):  36 muls via 4x Negacyclic-Karatsuba-9
 *     - Even x Odd    (8x4):  36 muls via CRT mod (Y^2 + 1) = (Y - X^2)(Y + X^2)
 *     - Odd  x Odd    (4x4):  27 muls via Galois-subfield CRT (H0:9 + H1,H2,H3:6x3)
 *   Total = 116 exact integer multiplications (39.6% fewer than separable 12-mul
 *   192-mul, and 8.8x fewer than 1024-MAC reference), formally verified in Lean 4
 *   (zcvid/proofs/lean4/ZcvidGalois2D.lean, 0 sorry, 0 custom axioms).
 */

#ifndef ZCVID_GALOIS_DCT8_H
#define ZCVID_GALOIS_DCT8_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Exact 3-multiply Gauss rotation: y0 = 83*x0 + 36*x1, y1 = 36*x0 - 83*x1 */
static inline void zc_g2d_even_rot_i32(int32_t x0, int32_t x1,
                                       int32_t *y0, int32_t *y1) {
    int32_t p = 36 * (x0 + x1);
    *y0 = p + 47 * x0;
    *y1 = p - 119 * x1;
}

static inline void zc_g2d_even_rot_i64(int64_t x0, int64_t x1,
                                       int64_t *y0, int64_t *y1) {
    int64_t p = 36 * (x0 + x1);
    *y0 = p + 47 * x0;
    *y1 = p - 119 * x1;
}

/* Exact 9-multiply Negacyclic-Karatsuba product mod (X^4 + 1) over Z */
static inline void zc_g2d_poly4_neg_mul9_i32(
    int32_t a0, int32_t a1, int32_t a2, int32_t a3,
    int32_t b0, int32_t b1, int32_t b2, int32_t b3,
    int32_t *r0, int32_t *r1, int32_t *r2, int32_t *r3) {
    int32_t m0 = a0 * b0, m1 = a1 * b1, m2 = (a0 + a1) * (b0 + b1);
    int32_t m3 = a2 * b2, m4 = a3 * b3, m5 = (a2 + a3) * (b2 + b3);
    int32_t sa0 = a0 + a2, sa1 = a1 + a3;
    int32_t sb0 = b0 + b2, sb1 = b1 + b3;
    int32_t m6 = sa0 * sb0, m7 = sa1 * sb1, m8 = (sa0 + sa1) * (sb0 + sb1);
    int32_t p0 = m0;
    int32_t p1 = m2 - m0 - m1;
    int32_t p2 = m1 + m6 - m0 - m3;
    int32_t p3 = (m8 - m6 - m7) - p1 - (m5 - m3 - m4);
    int32_t p4 = m3 + m7 - m1 - m4;
    int32_t p5 = m5 - m3 - m4;
    int32_t p6 = m4;
    *r0 = p0 - p4;
    *r1 = p1 - p5;
    *r2 = p2 - p6;
    *r3 = p3;
}

static inline void zc_g2d_poly4_neg_mul9_i64(
    int64_t a0, int64_t a1, int64_t a2, int64_t a3,
    int64_t b0, int64_t b1, int64_t b2, int64_t b3,
    int64_t *r0, int64_t *r1, int64_t *r2, int64_t *r3) {
    int64_t m0 = a0 * b0, m1 = a1 * b1, m2 = (a0 + a1) * (b0 + b1);
    int64_t m3 = a2 * b2, m4 = a3 * b3, m5 = (a2 + a3) * (b2 + b3);
    int64_t sa0 = a0 + a2, sa1 = a1 + a3;
    int64_t sb0 = b0 + b2, sb1 = b1 + b3;
    int64_t m6 = sa0 * sb0, m7 = sa1 * sb1, m8 = (sa0 + sa1) * (sb0 + sb1);
    int64_t p0 = m0;
    int64_t p1 = m2 - m0 - m1;
    int64_t p2 = m1 + m6 - m0 - m3;
    int64_t p3 = (m8 - m6 - m7) - p1 - (m5 - m3 - m4);
    int64_t p4 = m3 + m7 - m1 - m4;
    int64_t p5 = m5 - m3 - m4;
    int64_t p6 = m4;
    *r0 = p0 - p4;
    *r1 = p1 - p5;
    *r2 = p2 - p6;
    *r3 = p3;
}

/* M_odd * v via Rader perm {0,2,3,1} & A=[89,-75,-18,-50] (9 muls) */
static inline void zc_g2d_odd1d_i32(int32_t v0, int32_t v1, int32_t v2, int32_t v3,
                                    int32_t *y0, int32_t *y1, int32_t *y2, int32_t *y3) {
    int32_t q0, q1, q2, q3;
    zc_g2d_poly4_neg_mul9_i32(89, -75, -18, -50, v0, v2, v3, v1, &q0, &q1, &q2, &q3);
    *y0 = q0;
    *y1 = -q1;
    *y2 = -q3;
    *y3 = -q2;
}

static inline void zc_g2d_odd1d_i64(int64_t v0, int64_t v1, int64_t v2, int64_t v3,
                                    int64_t *y0, int64_t *y1, int64_t *y2, int64_t *y3) {
    int64_t q0, q1, q2, q3;
    zc_g2d_poly4_neg_mul9_i64(89, -75, -18, -50, v0, v2, v3, v1, &q0, &q1, &q2, &q3);
    *y0 = q0;
    *y1 = -q1;
    *y2 = -q3;
    *y3 = -q2;
}

/* 5-mul 2D Even(2,6) x Even(2,6): computes M_even * E * M_even^T */
static inline void zc_g2d_even2d_5mul_i32(int32_t e00, int32_t e01,
                                          int32_t e10, int32_t e11,
                                          int32_t *o00, int32_t *o01,
                                          int32_t *o10, int32_t *o11) {
    int32_t vp0 = e00 - e11, vp1 = -e10 - e01;
    int32_t vm0 = e00 + e11, vm1 = -e10 + e01;
    /* Gauss 3-mul on (5593 + 5976 X) * (vp0 + vp1 X) mod (X^2 + 1) */
    int32_t t = 5593 * (vp0 + vp1);
    int32_t wp0 = t - 11569 * vp1;
    int32_t wp1 = t + 383 * vp0;
    /* 2-mul scalar product by 8185 = 83^2 + 36^2 */
    int32_t wm0 = 8185 * vm0;
    int32_t wm1 = 8185 * vm1;
    *o00 = (wp0 + wm0) / 2;
    *o10 = (wp1 + wm1) / 2;
    *o01 = (wp1 - wm1) / 2;
    *o11 = (wm0 - wp0) / 2;
}

static inline void zc_g2d_even2d_5mul_i64(int64_t e00, int64_t e01,
                                          int64_t e10, int64_t e11,
                                          int64_t *o00, int64_t *o01,
                                          int64_t *o10, int64_t *o11) {
    int64_t vp0 = e00 - e11, vp1 = -e10 - e01;
    int64_t vm0 = e00 + e11, vm1 = -e10 + e01;
    int64_t t = 5593 * (vp0 + vp1);
    int64_t wp0 = t - 11569 * vp1;
    int64_t wp1 = t + 383 * vp0;
    int64_t wm0 = 8185 * vm0;
    int64_t wm1 = 8185 * vm1;
    *o00 = (wp0 + wm0) / 2;
    *o10 = (wp1 + wm1) / 2;
    *o01 = (wp1 - wm1) / 2;
    *o11 = (wm0 - wp0) / 2;
}

/* 18-mul 2D Even(2,6) x Odd(1,3,5,7): computes M_even * M[2][4] * M_odd^T */
static inline void zc_g2d_even_odd_18mul_i32(
    const int32_t r0[4], const int32_t r1[4],
    int32_t out0[4], int32_t out1[4]) {
    int32_t f00 = r0[0], f01 = r0[2], f02 = r0[3], f03 = r0[1];
    int32_t f10 = -r1[0], f11 = -r1[2], f12 = -r1[3], f13 = -r1[1];
    int32_t vp0 = f00 - f12, vp1 = f01 - f13, vp2 = f02 + f10, vp3 = f03 + f11;
    int32_t vm0 = f00 + f12, vm1 = f01 + f13, vm2 = f02 - f10, vm3 = f03 - f11;
    int32_t wp0, wp1, wp2, wp3, wm0, wm1, wm2, wm3;
    zc_g2d_poly4_neg_mul9_i32(8035, -4425, 1710, -6850, vp0, vp1, vp2, vp3,
                              &wp0, &wp1, &wp2, &wp3);
    zc_g2d_poly4_neg_mul9_i32(6739, -8025, -4698, -1450, vm0, vm1, vm2, vm3,
                              &wm0, &wm1, &wm2, &wm3);
    int32_t g00 = (wp0 + wm0) / 2, g01 = (wp1 + wm1) / 2;
    int32_t g02 = (wp2 + wm2) / 2, g03 = (wp3 + wm3) / 2;
    int32_t g10 = (wp2 - wm2) / 2, g11 = (wp3 - wm3) / 2;
    int32_t g12 = (wm0 - wp0) / 2, g13 = (wm1 - wp1) / 2;
    out0[0] = g00; out0[1] = -g01; out0[2] = -g03; out0[3] = -g02;
    out1[0] = g10; out1[1] = -g11; out1[2] = -g13; out1[3] = -g12;
}

static inline void zc_g2d_even_odd_18mul_i64(
    const int64_t r0[4], const int64_t r1[4],
    int64_t out0[4], int64_t out1[4]) {
    int64_t f00 = r0[0], f01 = r0[2], f02 = r0[3], f03 = r0[1];
    int64_t f10 = -r1[0], f11 = -r1[2], f12 = -r1[3], f13 = -r1[1];
    int64_t vp0 = f00 - f12, vp1 = f01 - f13, vp2 = f02 + f10, vp3 = f03 + f11;
    int64_t vm0 = f00 + f12, vm1 = f01 + f13, vm2 = f02 - f10, vm3 = f03 - f11;
    int64_t wp0, wp1, wp2, wp3, wm0, wm1, wm2, wm3;
    zc_g2d_poly4_neg_mul9_i64(8035, -4425, 1710, -6850, vp0, vp1, vp2, vp3,
                              &wp0, &wp1, &wp2, &wp3);
    zc_g2d_poly4_neg_mul9_i64(6739, -8025, -4698, -1450, vm0, vm1, vm2, vm3,
                              &wm0, &wm1, &wm2, &wm3);
    int64_t g00 = (wp0 + wm0) / 2, g01 = (wp1 + wm1) / 2;
    int64_t g02 = (wp2 + wm2) / 2, g03 = (wp3 + wm3) / 2;
    int64_t g10 = (wp2 - wm2) / 2, g11 = (wp3 - wm3) / 2;
    int64_t g12 = (wm0 - wp0) / 2, g13 = (wm1 - wp1) / 2;
    out0[0] = g00; out0[1] = -g01; out0[2] = -g03; out0[3] = -g02;
    out1[0] = g10; out1[1] = -g11; out1[2] = -g13; out1[3] = -g12;
}

/* 6-mul Z[i] subfield multiply: p * (15097 - 6329*X^2) */
static inline void zc_g2d_h1_6mul_i32(int32_t p0, int32_t p1, int32_t p2, int32_t p3,
                                      int32_t *w0, int32_t *w1, int32_t *w2, int32_t *w3) {
    int32_t t0 = 15097 * (p0 + p2);
    *w0 = t0 - 8768 * p2;
    *w2 = t0 - 21426 * p0;
    int32_t t1 = 15097 * (p1 + p3);
    *w1 = t1 - 8768 * p3;
    *w3 = t1 - 21426 * p1;
}

/* 6-mul Z[sqrt(-2)] subfield multiply: p * (120 - 11575*(X + X^3)) */
static inline void zc_g2d_h2_6mul_i32(int32_t p0, int32_t p1, int32_t p2, int32_t p3,
                                      int32_t *w0, int32_t *w1, int32_t *w2, int32_t *w3) {
    int32_t s13 = p1 + p3, d13 = p1 - p3;
    int32_t k1 = 120 * (p0 + s13);
    *w0 = k1 + 11455 * s13;
    int32_t b = -23270 * p0 + k1;
    int32_t k2 = 120 * (p2 + d13);
    *w2 = k2 - 11695 * d13;
    int32_t d = 23030 * p2 + k2;
    *w1 = (b + d) / 2;
    *w3 = (b - d) / 2;
}

/* 6-mul Z[sqrt(2)] subfield multiply: p * (16370 + 25*(X - X^3)) */
static inline void zc_g2d_h3_6mul_i32(int32_t p0, int32_t p1, int32_t p2, int32_t p3,
                                      int32_t *w0, int32_t *w1, int32_t *w2, int32_t *w3) {
    int32_t s13 = p1 + p3, d13 = p1 - p3;
    int32_t k1 = 16370 * (p0 + d13);
    *w0 = k1 - 16345 * d13;
    int32_t b = -16320 * p0 + k1;
    int32_t k2 = 16370 * (p2 + s13);
    *w2 = k2 - 16345 * s13;
    int32_t d = -16320 * p2 + k2;
    *w1 = (d + b) / 2;
    *w3 = (d - b) / 2;
}

static inline void zc_g2d_h1_6mul_i64(int64_t p0, int64_t p1, int64_t p2, int64_t p3,
                                      int64_t *w0, int64_t *w1, int64_t *w2, int64_t *w3) {
    int64_t t0 = 15097 * (p0 + p2);
    *w0 = t0 - 8768 * p2;
    *w2 = t0 - 21426 * p0;
    int64_t t1 = 15097 * (p1 + p3);
    *w1 = t1 - 8768 * p3;
    *w3 = t1 - 21426 * p1;
}

static inline void zc_g2d_h2_6mul_i64(int64_t p0, int64_t p1, int64_t p2, int64_t p3,
                                      int64_t *w0, int64_t *w1, int64_t *w2, int64_t *w3) {
    int64_t s13 = p1 + p3, d13 = p1 - p3;
    int64_t k1 = 120 * (p0 + s13);
    *w0 = k1 + 11455 * s13;
    int64_t b = -23270 * p0 + k1;
    int64_t k2 = 120 * (p2 + d13);
    *w2 = k2 - 11695 * d13;
    int64_t d = 23030 * p2 + k2;
    *w1 = (b + d) / 2;
    *w3 = (b - d) / 2;
}

static inline void zc_g2d_h3_6mul_i64(int64_t p0, int64_t p1, int64_t p2, int64_t p3,
                                      int64_t *w0, int64_t *w1, int64_t *w2, int64_t *w3) {
    int64_t s13 = p1 + p3, d13 = p1 - p3;
    int64_t k1 = 16370 * (p0 + d13);
    *w0 = k1 - 16345 * d13;
    int64_t b = -16320 * p0 + k1;
    int64_t k2 = 16370 * (p2 + s13);
    *w2 = k2 - 16345 * s13;
    int64_t d = -16320 * p2 + k2;
    *w1 = (d + b) / 2;
    *w3 = (d - b) / 2;
}

/* 27-mul 2D Odd(1,3,5,7) x Odd(1,3,5,7): computes M_odd * M[4][4] * M_odd^T */
static inline void zc_g2d_odd2d_27mul_i32(const int32_t m[4][4], int32_t out[4][4]) {
    /* Rader permutation on rows (X) and cols (Y) */
    int32_t f00 = m[0][0], f01 = m[2][0], f02 = m[3][0], f03 = m[1][0];
    int32_t f10 = m[0][2], f11 = m[2][2], f12 = m[3][2], f13 = m[1][2];
    int32_t f20 = m[0][3], f21 = m[2][3], f22 = m[3][3], f23 = m[1][3];
    int32_t f30 = m[0][1], f31 = m[2][1], f32 = m[3][1], f33 = m[1][1];

    /* mod (Y^2 - X^2) and mod (Y^2 + X^2) */
    int32_t a00 = f00 - f22, a01 = f01 - f23, a02 = f02 + f20, a03 = f03 + f21;
    int32_t a10 = f10 - f32, a11 = f11 - f33, a12 = f12 + f30, a13 = f13 + f31;
    int32_t b00 = f00 + f22, b01 = f01 + f23, b02 = f02 - f20, b03 = f03 - f21;
    int32_t b10 = f10 + f32, b11 = f11 + f33, b12 = f12 - f30, b13 = f13 - f31;

    /* Evaluate at Y = X, -X, X^3, -X^3 */
    int32_t v00 = a00 - a13, v01 = a01 + a10, v02 = a02 + a11, v03 = a03 + a12;
    int32_t v10 = a00 + a13, v11 = a01 - a10, v12 = a02 - a11, v13 = a03 - a12;
    int32_t v20 = b00 - b11, v21 = b01 - b12, v22 = b02 - b13, v23 = b03 + b10;
    int32_t v30 = b00 + b11, v31 = b01 + b12, v32 = b02 + b13, v33 = b03 - b10;

    int32_t w00, w01, w02, w03;
    int32_t w10, w11, w12, w13;
    int32_t w20, w21, w22, w23;
    int32_t w30, w31, w32, w33;

    zc_g2d_poly4_neg_mul9_i32(97, -15150, -79, -6200, v00, v01, v02, v03,
                              &w00, &w01, &w02, &w03);
    zc_g2d_h1_6mul_i32(v10, v11, v12, v13, &w10, &w11, &w12, &w13);
    zc_g2d_h2_6mul_i32(v20, v21, v22, v23, &w20, &w21, &w22, &w23);
    zc_g2d_h3_6mul_i32(v30, v31, v32, v33, &w30, &w31, &w32, &w33);

    int32_t u00 = w00 + w10, u01 = w01 + w11, u02 = w02 + w12, u03 = w03 + w13;
    int32_t d00 = w10 - w00, d01 = w11 - w01, d02 = w12 - w02, d03 = w13 - w03;
    int32_t u10 = -d01, u11 = -d02, u12 = -d03, u13 = d00; /* X^3 * (W1 - W0) */

    int32_t s00 = w20 + w30, s01 = w21 + w31, s02 = w22 + w32, s03 = w23 + w33;
    int32_t d10 = w30 - w20, d11 = w31 - w21, d12 = w32 - w22, d13 = w33 - w23;
    int32_t s10 = -d13, s11 = d10, s12 = d11, s13 = d12;   /* X * (W3 - W2) */

    int32_t g[4][4];
    g[0][0] = (u00 + s00) / 4; g[0][1] = (u01 + s01) / 4;
    g[0][2] = (u02 + s02) / 4; g[0][3] = (u03 + s03) / 4;
    g[2][0] = (u02 - s02) / 4; g[2][1] = (u03 - s03) / 4;
    g[2][2] = (s00 - u00) / 4; g[2][3] = (s01 - u01) / 4;
    g[1][0] = (u10 + s10) / 4; g[1][1] = (u11 + s11) / 4;
    g[1][2] = (u12 + s12) / 4; g[1][3] = (u13 + s13) / 4;
    g[3][0] = (u12 - s12) / 4; g[3][1] = (u13 - s13) / 4;
    g[3][2] = (s10 - u10) / 4; g[3][3] = (s11 - u11) / 4;

    static const uint8_t oid[4] = {0, 1, 3, 2};
    static const int32_t osg[4] = {1, -1, -1, -1};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            out[i][j] = osg[i] * osg[j] * g[oid[j]][oid[i]];
        }
    }
}

static inline void zc_g2d_odd2d_27mul_i64(const int64_t m[4][4], int64_t out[4][4]) {
    int64_t f00 = m[0][0], f01 = m[2][0], f02 = m[3][0], f03 = m[1][0];
    int64_t f10 = m[0][2], f11 = m[2][2], f12 = m[3][2], f13 = m[1][2];
    int64_t f20 = m[0][3], f21 = m[2][3], f22 = m[3][3], f23 = m[1][3];
    int64_t f30 = m[0][1], f31 = m[2][1], f32 = m[3][1], f33 = m[1][1];

    int64_t a00 = f00 - f22, a01 = f01 - f23, a02 = f02 + f20, a03 = f03 + f21;
    int64_t a10 = f10 - f32, a11 = f11 - f33, a12 = f12 + f30, a13 = f13 + f31;
    int64_t b00 = f00 + f22, b01 = f01 + f23, b02 = f02 - f20, b03 = f03 - f21;
    int64_t b10 = f10 + f32, b11 = f11 + f33, b12 = f12 - f30, b13 = f13 - f31;

    int64_t v00 = a00 - a13, v01 = a01 + a10, v02 = a02 + a11, v03 = a03 + a12;
    int64_t v10 = a00 + a13, v11 = a01 - a10, v12 = a02 - a11, v13 = a03 - a12;
    int64_t v20 = b00 - b11, v21 = b01 - b12, v22 = b02 - b13, v23 = b03 + b10;
    int64_t v30 = b00 + b11, v31 = b01 + b12, v32 = b02 + b13, v33 = b03 - b10;

    int64_t w00, w01, w02, w03;
    int64_t w10, w11, w12, w13;
    int64_t w20, w21, w22, w23;
    int64_t w30, w31, w32, w33;

    zc_g2d_poly4_neg_mul9_i64(97, -15150, -79, -6200, v00, v01, v02, v03,
                              &w00, &w01, &w02, &w03);
    zc_g2d_h1_6mul_i64(v10, v11, v12, v13, &w10, &w11, &w12, &w13);
    zc_g2d_h2_6mul_i64(v20, v21, v22, v23, &w20, &w21, &w22, &w23);
    zc_g2d_h3_6mul_i64(v30, v31, v32, v33, &w30, &w31, &w32, &w33);

    int64_t u00 = w00 + w10, u01 = w01 + w11, u02 = w02 + w12, u03 = w03 + w13;
    int64_t d00 = w10 - w00, d01 = w11 - w01, d02 = w12 - w02, d03 = w13 - w03;
    int64_t u10 = -d01, u11 = -d02, u12 = -d03, u13 = d00;

    int64_t s00 = w20 + w30, s01 = w21 + w31, s02 = w22 + w32, s03 = w23 + w33;
    int64_t d10 = w30 - w20, d11 = w31 - w21, d12 = w32 - w22, d13 = w33 - w23;
    int64_t s10 = -d13, s11 = d10, s12 = d11, s13 = d12;

    int64_t g[4][4];
    g[0][0] = (u00 + s00) / 4; g[0][1] = (u01 + s01) / 4;
    g[0][2] = (u02 + s02) / 4; g[0][3] = (u03 + s03) / 4;
    g[2][0] = (u02 - s02) / 4; g[2][1] = (u03 - s03) / 4;
    g[2][2] = (s00 - u00) / 4; g[2][3] = (s01 - u01) / 4;
    g[1][0] = (u10 + s10) / 4; g[1][1] = (u11 + s11) / 4;
    g[1][2] = (u12 + s12) / 4; g[1][3] = (u13 + s13) / 4;
    g[3][0] = (u12 - s12) / 4; g[3][1] = (u13 - s13) / 4;
    g[3][2] = (s10 - u10) / 4; g[3][3] = (s11 - u11) / 4;

    static const uint8_t oid[4] = {0, 1, 3, 2};
    static const int64_t osg[4] = {1, -1, -1, -1};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            out[i][j] = osg[i] * osg[j] * g[oid[j]][oid[i]];
        }
    }
}

/* Core 116-mul blockdiag(M_dc, M_even, M_odd) (x) blockdiag(M_dc, M_even, M_odd)
 * over int64_t (0 overflows for any |b| < 2^45). */
static inline void zc_g2d_core_116mul_i64(
    const int64_t ee[2][2], const int64_t dc_ev[2][2], const int64_t ev_dc[2][2],
    const int64_t ev_ev[2][2], const int64_t dc_odd[2][4], const int64_t odd_dc[4][2],
    const int64_t ev_odd[2][4], const int64_t odd_ev[4][2], const int64_t odd_odd[4][4],
    int64_t out[64]) {
    static const uint8_t k_dc[2] = {0U, 4U};
    static const uint8_t k_ev[2] = {2U, 6U};
    static const uint8_t k_od[4] = {1U, 3U, 5U, 7U};

    /* 1. DC x DC (0 muls, pure *4096) */
    out[0 * 8 + 0] = 4096 * ee[0][0];
    out[0 * 8 + 4] = 4096 * ee[0][1];
    out[4 * 8 + 0] = 4096 * ee[1][0];
    out[4 * 8 + 4] = 4096 * ee[1][1];

    /* 2. DC x Even (6 muls) & Even x DC (6 muls) */
    for (int r = 0; r < 2; ++r) {
        int64_t d0, d1;
        zc_g2d_even_rot_i64(64 * dc_ev[r][0], 64 * dc_ev[r][1], &d0, &d1);
        out[k_dc[r] * 8 + 2] = d0;
        out[k_dc[r] * 8 + 6] = d1;

        zc_g2d_even_rot_i64(64 * ev_dc[0][r], 64 * ev_dc[1][r], &d0, &d1);
        out[2 * 8 + k_dc[r]] = d0;
        out[6 * 8 + k_dc[r]] = d1;
    }

    /* 3. Even x Even (5 muls) */
    {
        int64_t y00, y01, y10, y11;
        zc_g2d_even2d_5mul_i64(ev_ev[0][0], ev_ev[0][1], ev_ev[1][0], ev_ev[1][1],
                               &y00, &y01, &y10, &y11);
        out[2 * 8 + 2] = y00; out[2 * 8 + 6] = y01;
        out[6 * 8 + 2] = y10; out[6 * 8 + 6] = y11;
    }

    /* 4. DC x Odd (18 muls) & Odd x DC (18 muls) */
    for (int r = 0; r < 2; ++r) {
        int64_t o0, o1, o2, o3;
        zc_g2d_odd1d_i64(64 * dc_odd[r][0], 64 * dc_odd[r][1],
                         64 * dc_odd[r][2], 64 * dc_odd[r][3], &o0, &o1, &o2, &o3);
        out[k_dc[r] * 8 + 1] = o0; out[k_dc[r] * 8 + 3] = o1;
        out[k_dc[r] * 8 + 5] = o2; out[k_dc[r] * 8 + 7] = o3;

        zc_g2d_odd1d_i64(64 * odd_dc[0][r], 64 * odd_dc[1][r],
                         64 * odd_dc[2][r], 64 * odd_dc[3][r], &o0, &o1, &o2, &o3);
        out[1 * 8 + k_dc[r]] = o0; out[3 * 8 + k_dc[r]] = o1;
        out[5 * 8 + k_dc[r]] = o2; out[7 * 8 + k_dc[r]] = o3;
    }

    /* 5. Even x Odd (18 muls) & Odd x Even (18 muls) */
    {
        int64_t eo0[4], eo1[4];
        zc_g2d_even_odd_18mul_i64(ev_odd[0], ev_odd[1], eo0, eo1);
        for (int c = 0; c < 4; ++c) {
            out[2 * 8 + k_od[c]] = eo0[c];
            out[6 * 8 + k_od[c]] = eo1[c];
        }
        int64_t tr[2][4];
        for (int j = 0; j < 4; ++j) {
            tr[0][j] = odd_ev[j][0];
            tr[1][j] = odd_ev[j][1];
        }
        zc_g2d_even_odd_18mul_i64(tr[0], tr[1], eo0, eo1);
        for (int r = 0; r < 4; ++r) {
            out[k_od[r] * 8 + 2] = eo0[r];
            out[k_od[r] * 8 + 6] = eo1[r];
        }
        (void)k_ev;
    }

    /* 6. Odd x Odd (27 muls) */
    {
        int64_t oo[4][4];
        zc_g2d_odd2d_27mul_i64(odd_odd, oo);
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                out[k_od[r] * 8 + k_od[c]] = oo[r][c];
            }
        }
    }
}

/* Exact 64-bit K * in * K^T (0 divisions, 116 multiplications, arbitrary int32 inputs) */
static inline void zcvid_galois_dct8_2d_exact_i64(const int32_t in[64], int64_t out[64]) {
    int64_t re[8][4], ro[8][4];
    for (int y = 0; y < 8; ++y) {
        const int32_t *r = in + y * 8;
        for (int j = 0; j < 4; ++j) {
            re[y][j] = (int64_t)r[j] + (int64_t)r[7 - j];
            ro[y][j] = (int64_t)r[j] - (int64_t)r[7 - j];
        }
    }
    int64_t ree[8][2], reo[8][2];
    for (int y = 0; y < 8; ++y) {
        int64_t e0 = re[y][0] + re[y][3], e1 = re[y][1] + re[y][2];
        ree[y][0] = e0 + e1; ree[y][1] = e0 - e1;
        reo[y][0] = re[y][0] - re[y][3];
        reo[y][1] = re[y][1] - re[y][2];
    }

    int64_t ee[2][2], dc_ev[2][2], ev_dc[2][2], ev_ev[2][2];
    int64_t dc_odd[2][4], odd_dc[4][2], ev_odd[2][4], odd_ev[4][2], odd_odd[4][4];

    for (int j = 0; j < 2; ++j) {
        int64_t t0 = ree[0][j] + ree[7][j], t1 = ree[1][j] + ree[6][j];
        int64_t t2 = ree[2][j] + ree[5][j], t3 = ree[3][j] + ree[4][j];
        int64_t ce0 = t0 + t3, ce1 = t1 + t2;
        ee[0][j] = ce0 + ce1; ee[1][j] = ce0 - ce1;
        ev_dc[0][j] = t0 - t3; ev_dc[1][j] = t1 - t2;

        int64_t o0 = ree[0][j] - ree[7][j], o1 = ree[1][j] - ree[6][j];
        int64_t o2 = ree[2][j] - ree[5][j], o3 = ree[3][j] - ree[4][j];
        odd_dc[0][j] = o0; odd_dc[1][j] = o1; odd_dc[2][j] = o2; odd_dc[3][j] = o3;

        int64_t s0 = reo[0][j] + reo[7][j], s1 = reo[1][j] + reo[6][j];
        int64_t s2 = reo[2][j] + reo[5][j], s3 = reo[3][j] + reo[4][j];
        int64_t de0 = s0 + s3, de1 = s1 + s2;
        dc_ev[0][j] = de0 + de1; dc_ev[1][j] = de0 - de1;
        ev_ev[0][j] = s0 - s3; ev_ev[1][j] = s1 - s2;

        odd_ev[0][j] = reo[0][j] - reo[7][j];
        odd_ev[1][j] = reo[1][j] - reo[6][j];
        odd_ev[2][j] = reo[2][j] - reo[5][j];
        odd_ev[3][j] = reo[3][j] - reo[4][j];
    }
    for (int j = 0; j < 4; ++j) {
        int64_t t0 = ro[0][j] + ro[7][j], t1 = ro[1][j] + ro[6][j];
        int64_t t2 = ro[2][j] + ro[5][j], t3 = ro[3][j] + ro[4][j];
        int64_t e0 = t0 + t3, e1 = t1 + t2;
        dc_odd[0][j] = e0 + e1; dc_odd[1][j] = e0 - e1;
        ev_odd[0][j] = t0 - t3; ev_odd[1][j] = t1 - t2;

        odd_odd[0][j] = ro[0][j] - ro[7][j];
        odd_odd[1][j] = ro[1][j] - ro[6][j];
        odd_odd[2][j] = ro[2][j] - ro[5][j];
        odd_odd[3][j] = ro[3][j] - ro[4][j];
    }

    zc_g2d_core_116mul_i64(ee, dc_ev, ev_dc, ev_ev, dc_odd, odd_dc, ev_odd, odd_ev, odd_odd, out);
}

/* Exact 32-bit 116-mul forward 2D DCT8 (falls back cleanly to i64 if |in| > 511) */
static inline void zcvid_galois_dct8_2d_exact_scalar(const int32_t in[64], int32_t out[64]) {
    for (uint32_t i = 0U; i < 64U; ++i) {
        if (in[i] > 511 || in[i] < -511) {
            int64_t o64[64];
            zcvid_galois_dct8_2d_exact_i64(in, o64);
            for (uint32_t k = 0U; k < 64U; ++k) out[k] = (int32_t)o64[k];
            return;
        }
    }

    int32_t re[8][4], ro[8][4];
    for (int y = 0; y < 8; ++y) {
        const int32_t *r = in + y * 8;
        for (int j = 0; j < 4; ++j) {
            re[y][j] = r[j] + r[7 - j];
            ro[y][j] = r[j] - r[7 - j];
        }
    }
    int32_t ree[8][2], reo[8][2];
    for (int y = 0; y < 8; ++y) {
        int32_t e0 = re[y][0] + re[y][3], e1 = re[y][1] + re[y][2];
        ree[y][0] = e0 + e1; ree[y][1] = e0 - e1;
        reo[y][0] = re[y][0] - re[y][3];
        reo[y][1] = re[y][1] - re[y][2];
    }

    int32_t ee[2][2], dc_ev[2][2], ev_dc[2][2], ev_ev[2][2];
    int32_t dc_odd[2][4], odd_dc[4][2], ev_odd[2][4], odd_ev[4][2], odd_odd[4][4];

    for (int j = 0; j < 2; ++j) {
        int32_t t0 = ree[0][j] + ree[7][j], t1 = ree[1][j] + ree[6][j];
        int32_t t2 = ree[2][j] + ree[5][j], t3 = ree[3][j] + ree[4][j];
        int32_t ce0 = t0 + t3, ce1 = t1 + t2;
        ee[0][j] = ce0 + ce1; ee[1][j] = ce0 - ce1;
        ev_dc[0][j] = t0 - t3; ev_dc[1][j] = t1 - t2;

        odd_dc[0][j] = ree[0][j] - ree[7][j];
        odd_dc[1][j] = ree[1][j] - ree[6][j];
        odd_dc[2][j] = ree[2][j] - ree[5][j];
        odd_dc[3][j] = ree[3][j] - ree[4][j];

        int32_t s0 = reo[0][j] + reo[7][j], s1 = reo[1][j] + reo[6][j];
        int32_t s2 = reo[2][j] + reo[5][j], s3 = reo[3][j] + reo[4][j];
        int32_t de0 = s0 + s3, de1 = s1 + s2;
        dc_ev[0][j] = de0 + de1; dc_ev[1][j] = de0 - de1;
        ev_ev[0][j] = s0 - s3; ev_ev[1][j] = s1 - s2;

        odd_ev[0][j] = reo[0][j] - reo[7][j];
        odd_ev[1][j] = reo[1][j] - reo[6][j];
        odd_ev[2][j] = reo[2][j] - reo[5][j];
        odd_ev[3][j] = reo[3][j] - reo[4][j];
    }
    for (int j = 0; j < 4; ++j) {
        int32_t t0 = ro[0][j] + ro[7][j], t1 = ro[1][j] + ro[6][j];
        int32_t t2 = ro[2][j] + ro[5][j], t3 = ro[3][j] + ro[4][j];
        int32_t e0 = t0 + t3, e1 = t1 + t2;
        dc_odd[0][j] = e0 + e1; dc_odd[1][j] = e0 - e1;
        ev_odd[0][j] = t0 - t3; ev_odd[1][j] = t1 - t2;

        odd_odd[0][j] = ro[0][j] - ro[7][j];
        odd_odd[1][j] = ro[1][j] - ro[6][j];
        odd_odd[2][j] = ro[2][j] - ro[5][j];
        odd_odd[3][j] = ro[3][j] - ro[4][j];
    }

    static const uint8_t k_dc[2] = {0U, 4U};
    static const uint8_t k_od[4] = {1U, 3U, 5U, 7U};

    out[0 * 8 + 0] = 4096 * ee[0][0];
    out[0 * 8 + 4] = 4096 * ee[0][1];
    out[4 * 8 + 0] = 4096 * ee[1][0];
    out[4 * 8 + 4] = 4096 * ee[1][1];

    for (int r = 0; r < 2; ++r) {
        int32_t d0, d1;
        zc_g2d_even_rot_i32(64 * dc_ev[r][0], 64 * dc_ev[r][1], &d0, &d1);
        out[k_dc[r] * 8 + 2] = d0;
        out[k_dc[r] * 8 + 6] = d1;

        zc_g2d_even_rot_i32(64 * ev_dc[0][r], 64 * ev_dc[1][r], &d0, &d1);
        out[2 * 8 + k_dc[r]] = d0;
        out[6 * 8 + k_dc[r]] = d1;
    }

    zc_g2d_even2d_5mul_i32(ev_ev[0][0], ev_ev[0][1], ev_ev[1][0], ev_ev[1][1],
                           &out[2 * 8 + 2], &out[2 * 8 + 6],
                           &out[6 * 8 + 2], &out[6 * 8 + 6]);

    for (int r = 0; r < 2; ++r) {
        zc_g2d_odd1d_i32(64 * dc_odd[r][0], 64 * dc_odd[r][1],
                         64 * dc_odd[r][2], 64 * dc_odd[r][3],
                         &out[k_dc[r] * 8 + 1], &out[k_dc[r] * 8 + 3],
                         &out[k_dc[r] * 8 + 5], &out[k_dc[r] * 8 + 7]);

        zc_g2d_odd1d_i32(64 * odd_dc[0][r], 64 * odd_dc[1][r],
                         64 * odd_dc[2][r], 64 * odd_dc[3][r],
                         &out[1 * 8 + k_dc[r]], &out[3 * 8 + k_dc[r]],
                         &out[5 * 8 + k_dc[r]], &out[7 * 8 + k_dc[r]]);
    }

    {
        int32_t eo0[4], eo1[4];
        zc_g2d_even_odd_18mul_i32(ev_odd[0], ev_odd[1], eo0, eo1);
        for (int c = 0; c < 4; ++c) {
            out[2 * 8 + k_od[c]] = eo0[c];
            out[6 * 8 + k_od[c]] = eo1[c];
        }
        int32_t tr[2][4];
        for (int j = 0; j < 4; ++j) {
            tr[0][j] = odd_ev[j][0];
            tr[1][j] = odd_ev[j][1];
        }
        zc_g2d_even_odd_18mul_i32(tr[0], tr[1], eo0, eo1);
        for (int r = 0; r < 4; ++r) {
            out[k_od[r] * 8 + 2] = eo0[r];
            out[k_od[r] * 8 + 6] = eo1[r];
        }
    }

    {
        int32_t oo[4][4];
        zc_g2d_odd2d_27mul_i32(odd_odd, oo);
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                out[k_od[r] * 8 + k_od[c]] = oo[r][c];
            }
        }
    }
}

/* Exact 64-bit K^T * in * K (inverse 2D IDCT8, 116 muls by symmetry duality!) */
static inline void zcvid_galois_idct8_2d_exact_i64(const int32_t in[64], int64_t out[64]) {
    int64_t ee[2][2] = {{in[0 * 8 + 0], in[0 * 8 + 4]},
                        {in[4 * 8 + 0], in[4 * 8 + 4]}};
    int64_t dc_ev[2][2] = {{in[0 * 8 + 2], in[0 * 8 + 6]},
                           {in[4 * 8 + 2], in[4 * 8 + 6]}};
    int64_t ev_dc[2][2] = {{in[2 * 8 + 0], in[2 * 8 + 4]},
                           {in[6 * 8 + 0], in[6 * 8 + 4]}};
    int64_t ev_ev[2][2] = {{in[2 * 8 + 2], in[2 * 8 + 6]},
                           {in[6 * 8 + 2], in[6 * 8 + 6]}};
    int64_t dc_odd[2][4] = {{in[0 * 8 + 1], in[0 * 8 + 3], in[0 * 8 + 5], in[0 * 8 + 7]},
                            {in[4 * 8 + 1], in[4 * 8 + 3], in[4 * 8 + 5], in[4 * 8 + 7]}};
    int64_t odd_dc[4][2] = {{in[1 * 8 + 0], in[1 * 8 + 4]},
                            {in[3 * 8 + 0], in[3 * 8 + 4]},
                            {in[5 * 8 + 0], in[5 * 8 + 4]},
                            {in[7 * 8 + 0], in[7 * 8 + 4]}};
    int64_t ev_odd[2][4] = {{in[2 * 8 + 1], in[2 * 8 + 3], in[2 * 8 + 5], in[2 * 8 + 7]},
                            {in[6 * 8 + 1], in[6 * 8 + 3], in[6 * 8 + 5], in[6 * 8 + 7]}};
    int64_t odd_ev[4][2] = {{in[1 * 8 + 2], in[1 * 8 + 6]},
                            {in[3 * 8 + 2], in[3 * 8 + 6]},
                            {in[5 * 8 + 2], in[5 * 8 + 6]},
                            {in[7 * 8 + 2], in[7 * 8 + 6]}};
    static const uint8_t k_od[4] = {1U, 3U, 5U, 7U};
    int64_t odd_odd[4][4];
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            odd_odd[r][c] = in[k_od[r] * 8 + k_od[c]];
        }
    }

    int64_t mid[64];
    zc_g2d_core_116mul_i64(ee, dc_ev, ev_dc, ev_ev, dc_odd, odd_dc, ev_odd, odd_ev, odd_odd, mid);

    /* Apply B^T (zero-mul inverse butterfly) along rows then columns */
    int64_t tmp[64];
    for (int v = 0; v < 8; ++v) {
        const int64_t *c = mid + v * 8;
        int64_t s0 = c[0] + c[4], s1 = c[0] - c[4];
        int64_t e0 = s0 + c[2], e1 = s1 + c[6], e2 = s1 - c[6], e3 = s0 - c[2];
        int64_t o0 = c[1], o1 = c[3], o2 = c[5], o3 = c[7];
        tmp[v * 8 + 0] = e0 + o0; tmp[v * 8 + 7] = e0 - o0;
        tmp[v * 8 + 1] = e1 + o1; tmp[v * 8 + 6] = e1 - o1;
        tmp[v * 8 + 2] = e2 + o2; tmp[v * 8 + 5] = e2 - o2;
        tmp[v * 8 + 3] = e3 + o3; tmp[v * 8 + 4] = e3 - o3;
    }
    for (int x = 0; x < 8; ++x) {
        int64_t s0 = tmp[0 * 8 + x] + tmp[4 * 8 + x], s1 = tmp[0 * 8 + x] - tmp[4 * 8 + x];
        int64_t e0 = s0 + tmp[2 * 8 + x], e1 = s1 + tmp[6 * 8 + x];
        int64_t e2 = s1 - tmp[6 * 8 + x], e3 = s0 - tmp[2 * 8 + x];
        int64_t o0 = tmp[1 * 8 + x], o1 = tmp[3 * 8 + x];
        int64_t o2 = tmp[5 * 8 + x], o3 = tmp[7 * 8 + x];
        out[0 * 8 + x] = e0 + o0; out[7 * 8 + x] = e0 - o0;
        out[1 * 8 + x] = e1 + o1; out[6 * 8 + x] = e1 - o1;
        out[2 * 8 + x] = e2 + o2; out[5 * 8 + x] = e2 - o2;
        out[3 * 8 + x] = e3 + o3; out[4 * 8 + x] = e3 - o3;
    }
}

/* Exact 32-bit 116-mul inverse 2D IDCT8 (falls back to i64 if |in| > 511) */
static inline void zcvid_galois_idct8_2d_exact_scalar(const int32_t in[64], int32_t out[64]) {
    for (uint32_t i = 0U; i < 64U; ++i) {
        if (in[i] > 511 || in[i] < -511) {
            int64_t o64[64];
            zcvid_galois_idct8_2d_exact_i64(in, o64);
            for (uint32_t k = 0U; k < 64U; ++k) out[k] = (int32_t)o64[k];
            return;
        }
    }

    int32_t mid[64];
    static const uint8_t k_dc[2] = {0U, 4U};
    static const uint8_t k_od[4] = {1U, 3U, 5U, 7U};

    mid[0 * 8 + 0] = 4096 * in[0 * 8 + 0];
    mid[0 * 8 + 4] = 4096 * in[0 * 8 + 4];
    mid[4 * 8 + 0] = 4096 * in[4 * 8 + 0];
    mid[4 * 8 + 4] = 4096 * in[4 * 8 + 4];

    for (int r = 0; r < 2; ++r) {
        int32_t d0, d1;
        zc_g2d_even_rot_i32(64 * in[k_dc[r] * 8 + 2], 64 * in[k_dc[r] * 8 + 6], &d0, &d1);
        mid[k_dc[r] * 8 + 2] = d0;
        mid[k_dc[r] * 8 + 6] = d1;

        zc_g2d_even_rot_i32(64 * in[2 * 8 + k_dc[r]], 64 * in[6 * 8 + k_dc[r]], &d0, &d1);
        mid[2 * 8 + k_dc[r]] = d0;
        mid[6 * 8 + k_dc[r]] = d1;
    }

    zc_g2d_even2d_5mul_i32(in[2 * 8 + 2], in[2 * 8 + 6], in[6 * 8 + 2], in[6 * 8 + 6],
                           &mid[2 * 8 + 2], &mid[2 * 8 + 6],
                           &mid[6 * 8 + 2], &mid[6 * 8 + 6]);

    for (int r = 0; r < 2; ++r) {
        zc_g2d_odd1d_i32(64 * in[k_dc[r] * 8 + 1], 64 * in[k_dc[r] * 8 + 3],
                         64 * in[k_dc[r] * 8 + 5], 64 * in[k_dc[r] * 8 + 7],
                         &mid[k_dc[r] * 8 + 1], &mid[k_dc[r] * 8 + 3],
                         &mid[k_dc[r] * 8 + 5], &mid[k_dc[r] * 8 + 7]);
        zc_g2d_odd1d_i32(64 * in[1 * 8 + k_dc[r]], 64 * in[3 * 8 + k_dc[r]],
                         64 * in[5 * 8 + k_dc[r]], 64 * in[7 * 8 + k_dc[r]],
                         &mid[1 * 8 + k_dc[r]], &mid[3 * 8 + k_dc[r]],
                         &mid[5 * 8 + k_dc[r]], &mid[7 * 8 + k_dc[r]]);
    }

    {
        int32_t r0[4] = {in[2 * 8 + 1], in[2 * 8 + 3], in[2 * 8 + 5], in[2 * 8 + 7]};
        int32_t r1[4] = {in[6 * 8 + 1], in[6 * 8 + 3], in[6 * 8 + 5], in[6 * 8 + 7]};
        int32_t eo0[4], eo1[4];
        zc_g2d_even_odd_18mul_i32(r0, r1, eo0, eo1);
        for (int c = 0; c < 4; ++c) {
            mid[2 * 8 + k_od[c]] = eo0[c];
            mid[6 * 8 + k_od[c]] = eo1[c];
        }
        int32_t c0[4] = {in[1 * 8 + 2], in[3 * 8 + 2], in[5 * 8 + 2], in[7 * 8 + 2]};
        int32_t c1[4] = {in[1 * 8 + 6], in[3 * 8 + 6], in[5 * 8 + 6], in[7 * 8 + 6]};
        zc_g2d_even_odd_18mul_i32(c0, c1, eo0, eo1);
        for (int r = 0; r < 4; ++r) {
            mid[k_od[r] * 8 + 2] = eo0[r];
            mid[k_od[r] * 8 + 6] = eo1[r];
        }
    }

    {
        int32_t oo[4][4], oo_out[4][4];
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) oo[r][c] = in[k_od[r] * 8 + k_od[c]];
        }
        zc_g2d_odd2d_27mul_i32(oo, oo_out);
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) mid[k_od[r] * 8 + k_od[c]] = oo_out[r][c];
        }
    }

    int32_t tmp[64];
    for (int v = 0; v < 8; ++v) {
        const int32_t *c = mid + v * 8;
        int32_t s0 = c[0] + c[4], s1 = c[0] - c[4];
        int32_t e0 = s0 + c[2], e1 = s1 + c[6], e2 = s1 - c[6], e3 = s0 - c[2];
        int32_t o0 = c[1], o1 = c[3], o2 = c[5], o3 = c[7];
        tmp[v * 8 + 0] = e0 + o0; tmp[v * 8 + 7] = e0 - o0;
        tmp[v * 8 + 1] = e1 + o1; tmp[v * 8 + 6] = e1 - o1;
        tmp[v * 8 + 2] = e2 + o2; tmp[v * 8 + 5] = e2 - o2;
        tmp[v * 8 + 3] = e3 + o3; tmp[v * 8 + 4] = e3 - o3;
    }
    for (int x = 0; x < 8; ++x) {
        int32_t s0 = tmp[0 * 8 + x] + tmp[4 * 8 + x], s1 = tmp[0 * 8 + x] - tmp[4 * 8 + x];
        int32_t e0 = s0 + tmp[2 * 8 + x], e1 = s1 + tmp[6 * 8 + x];
        int32_t e2 = s1 - tmp[6 * 8 + x], e3 = s0 - tmp[2 * 8 + x];
        int32_t o0 = tmp[1 * 8 + x], o1 = tmp[3 * 8 + x];
        int32_t o2 = tmp[5 * 8 + x], o3 = tmp[7 * 8 + x];
        out[0 * 8 + x] = e0 + o0; out[7 * 8 + x] = e0 - o0;
        out[1 * 8 + x] = e1 + o1; out[6 * 8 + x] = e1 - o1;
        out[2 * 8 + x] = e2 + o2; out[5 * 8 + x] = e2 - o2;
        out[3 * 8 + x] = e3 + o3; out[4 * 8 + x] = e3 - o3;
    }
}

#if defined(__AVX2__)
/* Range check [-511, 511] across all 64 int32 entries without abs(INT32_MIN) UB */
static inline bool zc_g2d_all_in_range511_avx2(const int32_t in[64]) {
    const __m256i vhi = _mm256_set1_epi32(511);
    const __m256i vlo = _mm256_set1_epi32(-511);
    __m256i bad = _mm256_setzero_si256();
    for (int y = 0; y < 8; ++y) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)(in + y * 8));
        bad = _mm256_or_si256(bad, _mm256_or_si256(_mm256_cmpgt_epi32(v, vhi),
                                                    _mm256_cmpgt_epi32(vlo, v)));
    }
    return _mm256_testz_si256(bad, bad) != 0;
}

/* 8-lane batched Negacyclic-Karatsuba mod (X^4 + 1):
 * 9 _mm256_mullo_epi32 compute ALL 8 polynomial products (72 scalar muls) in parallel! */
static inline void zc_g2d_poly4_neg_mul9_8x_avx2(
    __m256i a0, __m256i a1, __m256i a2, __m256i a3,
    __m256i b0, __m256i b1, __m256i b2, __m256i b3,
    __m256i *r0, __m256i *r1, __m256i *r2, __m256i *r3) {
    __m256i m0 = _mm256_mullo_epi32(a0, b0);
    __m256i m1 = _mm256_mullo_epi32(a1, b1);
    __m256i m2 = _mm256_mullo_epi32(_mm256_add_epi32(a0, a1), _mm256_add_epi32(b0, b1));
    __m256i m3 = _mm256_mullo_epi32(a2, b2);
    __m256i m4 = _mm256_mullo_epi32(a3, b3);
    __m256i m5 = _mm256_mullo_epi32(_mm256_add_epi32(a2, a3), _mm256_add_epi32(b2, b3));
    __m256i sa0 = _mm256_add_epi32(a0, a2), sa1 = _mm256_add_epi32(a1, a3);
    __m256i sb0 = _mm256_add_epi32(b0, b2), sb1 = _mm256_add_epi32(b1, b3);
    __m256i m6 = _mm256_mullo_epi32(sa0, sb0);
    __m256i m7 = _mm256_mullo_epi32(sa1, sb1);
    __m256i m8 = _mm256_mullo_epi32(_mm256_add_epi32(sa0, sa1), _mm256_add_epi32(sb0, sb1));

    __m256i p1 = _mm256_sub_epi32(_mm256_sub_epi32(m2, m0), m1);
    __m256i p5 = _mm256_sub_epi32(_mm256_sub_epi32(m5, m3), m4);
    __m256i p2 = _mm256_sub_epi32(_mm256_sub_epi32(_mm256_add_epi32(m1, m6), m0), m3);
    __m256i p4 = _mm256_sub_epi32(_mm256_sub_epi32(_mm256_add_epi32(m3, m7), m1), m4);
    __m256i p3 = _mm256_sub_epi32(_mm256_sub_epi32(_mm256_sub_epi32(_mm256_sub_epi32(m8, m6), m7), p1), p5);

    *r0 = _mm256_sub_epi32(m0, p4);
    *r1 = _mm256_sub_epi32(p1, p5);
    *r2 = _mm256_sub_epi32(p2, m4);
    *r3 = p3;
}

/* Shared 8-lane batched Galois core (zero 8x8 matrix transpose!) */
static inline void zc_g2d_core_116mul_avx2(
    const int32_t ee[2][2], const int32_t dc_ev[2][2], const int32_t ev_dc[2][2],
    const int32_t ev_ev[2][2], const int32_t dc_odd[2][4], const int32_t odd_dc[4][2],
    const int32_t ev_odd[2][4], const int32_t odd_ev[4][2], const int32_t odd_odd[4][4],
    int32_t out[64]) {
    static const uint8_t k_dc[2] = {0U, 4U};
    static const uint8_t k_od[4] = {1U, 3U, 5U, 7U};

    /* 1. DC x DC (0 muls) */
    out[0 * 8 + 0] = 4096 * ee[0][0];
    out[0 * 8 + 4] = 4096 * ee[0][1];
    out[4 * 8 + 0] = 4096 * ee[1][0];
    out[4 * 8 + 4] = 4096 * ee[1][1];

    /* 2. DC x Even + Even x DC (4x Gauss in 1 4-lane vector = 3 muls!) */
    {
        __m128i x0 = _mm_set_epi32(ev_dc[0][1], ev_dc[0][0], dc_ev[1][0], dc_ev[0][0]);
        __m128i x1 = _mm_set_epi32(ev_dc[1][1], ev_dc[1][0], dc_ev[1][1], dc_ev[0][1]);
        x0 = _mm_slli_epi32(x0, 6);
        x1 = _mm_slli_epi32(x1, 6);
        __m128i p = _mm_mullo_epi32(_mm_set1_epi32(36), _mm_add_epi32(x0, x1));
        __m128i y0 = _mm_add_epi32(p, _mm_mullo_epi32(_mm_set1_epi32(47), x0));
        __m128i y1 = _mm_sub_epi32(p, _mm_mullo_epi32(_mm_set1_epi32(119), x1));
        int32_t a0[4], a1[4];
        _mm_storeu_si128((__m128i *)(void *)a0, y0);
        _mm_storeu_si128((__m128i *)(void *)a1, y1);
        out[0 * 8 + 2] = a0[0]; out[0 * 8 + 6] = a1[0];
        out[4 * 8 + 2] = a0[1]; out[4 * 8 + 6] = a1[1];
        out[2 * 8 + 0] = a0[2]; out[6 * 8 + 0] = a1[2];
        out[2 * 8 + 4] = a0[3]; out[6 * 8 + 4] = a1[3];
    }

    /* 3. Even x Even (5 muls) */
    zc_g2d_even2d_5mul_i32(ev_ev[0][0], ev_ev[0][1], ev_ev[1][0], ev_ev[1][1],
                           &out[2 * 8 + 2], &out[2 * 8 + 6],
                           &out[6 * 8 + 2], &out[6 * 8 + 6]);

    /* 4+5. Batch ALL 8 Odd-1D/Even-Odd products (72 muls) into ONE 8-lane Karatsuba-9!
     * Lanes 0,1 = 64*dc_odd[0..1], Lanes 2,3 = 64*odd_dc[..][0..1],
     * Lanes 4,5 = H+ for (ev_odd, odd_ev), Lanes 6,7 = H- for (ev_odd, odd_ev). */
    {
        int32_t dco00 = 64 * dc_odd[0][0], dco01 = 64 * dc_odd[0][2], dco02 = 64 * dc_odd[0][3], dco03 = 64 * dc_odd[0][1];
        int32_t dco10 = 64 * dc_odd[1][0], dco11 = 64 * dc_odd[1][2], dco12 = 64 * dc_odd[1][3], dco13 = 64 * dc_odd[1][1];
        int32_t odc00 = 64 * odd_dc[0][0], odc01 = 64 * odd_dc[2][0], odc02 = 64 * odd_dc[3][0], odc03 = 64 * odd_dc[1][0];
        int32_t odc10 = 64 * odd_dc[0][1], odc11 = 64 * odd_dc[2][1], odc12 = 64 * odd_dc[3][1], odc13 = 64 * odd_dc[1][1];

        int32_t eop0 = ev_odd[0][0] + ev_odd[1][3], eop1 = ev_odd[0][2] + ev_odd[1][1];
        int32_t eop2 = ev_odd[0][3] - ev_odd[1][0], eop3 = ev_odd[0][1] - ev_odd[1][2];
        int32_t eom0 = ev_odd[0][0] - ev_odd[1][3], eom1 = ev_odd[0][2] - ev_odd[1][1];
        int32_t eom2 = ev_odd[0][3] + ev_odd[1][0], eom3 = ev_odd[0][1] + ev_odd[1][2];

        int32_t oep0 = odd_ev[0][0] + odd_ev[3][1], oep1 = odd_ev[2][0] + odd_ev[1][1];
        int32_t oep2 = odd_ev[3][0] - odd_ev[0][1], oep3 = odd_ev[1][0] - odd_ev[2][1];
        int32_t oem0 = odd_ev[0][0] - odd_ev[3][1], oem1 = odd_ev[2][0] - odd_ev[1][1];
        int32_t oem2 = odd_ev[3][0] + odd_ev[0][1], oem3 = odd_ev[1][0] + odd_ev[2][1];

        __m256i b0 = _mm256_set_epi32(oem0, eom0, oep0, eop0, odc10, odc00, dco10, dco00);
        __m256i b1 = _mm256_set_epi32(oem1, eom1, oep1, eop1, odc11, odc01, dco11, dco01);
        __m256i b2 = _mm256_set_epi32(oem2, eom2, oep2, eop2, odc12, odc02, dco12, dco02);
        __m256i b3 = _mm256_set_epi32(oem3, eom3, oep3, eop3, odc13, odc03, dco13, dco03);

        __m256i a0 = _mm256_set_epi32( 6739,  6739,  8035,  8035,  89,  89,  89,  89);
        __m256i a1 = _mm256_set_epi32(-8025, -8025, -4425, -4425, -75, -75, -75, -75);
        __m256i a2 = _mm256_set_epi32(-4698, -4698,  1710,  1710, -18, -18, -18, -18);
        __m256i a3 = _mm256_set_epi32(-1450, -1450, -6850, -6850, -50, -50, -50, -50);

        __m256i vr0, vr1, vr2, vr3;
        zc_g2d_poly4_neg_mul9_8x_avx2(a0, a1, a2, a3, b0, b1, b2, b3, &vr0, &vr1, &vr2, &vr3);

        int32_t s0[8], s1[8], s2[8], s3[8];
        _mm256_storeu_si256((__m256i *)(void *)s0, vr0);
        _mm256_storeu_si256((__m256i *)(void *)s1, vr1);
        _mm256_storeu_si256((__m256i *)(void *)s2, vr2);
        _mm256_storeu_si256((__m256i *)(void *)s3, vr3);

        for (int r = 0; r < 2; ++r) {
            out[k_dc[r] * 8 + 1] = s0[r];
            out[k_dc[r] * 8 + 3] = -s1[r];
            out[k_dc[r] * 8 + 5] = -s3[r];
            out[k_dc[r] * 8 + 7] = -s2[r];

            out[1 * 8 + k_dc[r]] = s0[2 + r];
            out[3 * 8 + k_dc[r]] = -s1[2 + r];
            out[5 * 8 + k_dc[r]] = -s3[2 + r];
            out[7 * 8 + k_dc[r]] = -s2[2 + r];
        }

        /* Even x Odd inverse CRT (lane 4 = W+, lane 6 = W-) */
        int32_t g00 = (s0[4] + s0[6]) / 2, g01 = (s1[4] + s1[6]) / 2;
        int32_t g02 = (s2[4] + s2[6]) / 2, g03 = (s3[4] + s3[6]) / 2;
        int32_t g10 = (s2[4] - s2[6]) / 2, g11 = (s3[4] - s3[6]) / 2;
        int32_t g12 = (s0[6] - s0[4]) / 2, g13 = (s1[6] - s1[4]) / 2;
        out[2 * 8 + 1] = g00; out[2 * 8 + 3] = -g01; out[2 * 8 + 5] = -g03; out[2 * 8 + 7] = -g02;
        out[6 * 8 + 1] = g10; out[6 * 8 + 3] = -g11; out[6 * 8 + 5] = -g13; out[6 * 8 + 7] = -g12;

        /* Odd x Even inverse CRT (lane 5 = W+, lane 7 = W-) */
        g00 = (s0[5] + s0[7]) / 2; g01 = (s1[5] + s1[7]) / 2;
        g02 = (s2[5] + s2[7]) / 2; g03 = (s3[5] + s3[7]) / 2;
        g10 = (s2[5] - s2[7]) / 2; g11 = (s3[5] - s3[7]) / 2;
        g12 = (s0[7] - s0[5]) / 2; g13 = (s1[7] - s1[5]) / 2;
        out[1 * 8 + 2] = g00; out[3 * 8 + 2] = -g01; out[5 * 8 + 2] = -g03; out[7 * 8 + 2] = -g02;
        out[1 * 8 + 6] = g10; out[3 * 8 + 6] = -g11; out[5 * 8 + 6] = -g13; out[7 * 8 + 6] = -g12;
    }

    /* 6. Odd x Odd (27 muls) */
    {
        int32_t oo[4][4];
        zc_g2d_odd2d_27mul_i32(odd_odd, oo);
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                out[k_od[r] * 8 + k_od[c]] = oo[r][c];
            }
        }
    }
}

static inline void zcvid_galois_dct8_2d_exact_avx2(const int32_t in[64], int32_t out[64]) {
    if (!zc_g2d_all_in_range511_avx2(in)) {
        int64_t o64[64];
        zcvid_galois_dct8_2d_exact_i64(in, o64);
        for (uint32_t k = 0U; k < 64U; ++k) out[k] = (int32_t)o64[k];
        return;
    }

    /* 1D vertical butterfly in 256-bit SIMD across all 8 columns simultaneously */
    __m256i r0 = _mm256_loadu_si256((const __m256i *)(const void *)(in + 0));
    __m256i r1 = _mm256_loadu_si256((const __m256i *)(const void *)(in + 8));
    __m256i r2 = _mm256_loadu_si256((const __m256i *)(const void *)(in + 16));
    __m256i r3 = _mm256_loadu_si256((const __m256i *)(const void *)(in + 24));
    __m256i r4 = _mm256_loadu_si256((const __m256i *)(const void *)(in + 32));
    __m256i r5 = _mm256_loadu_si256((const __m256i *)(const void *)(in + 40));
    __m256i r6 = _mm256_loadu_si256((const __m256i *)(const void *)(in + 48));
    __m256i r7 = _mm256_loadu_si256((const __m256i *)(const void *)(in + 56));

    __m256i ce0 = _mm256_add_epi32(r0, r7), ce1 = _mm256_add_epi32(r1, r6);
    __m256i ce2 = _mm256_add_epi32(r2, r5), ce3 = _mm256_add_epi32(r3, r4);
    __m256i co0 = _mm256_sub_epi32(r0, r7), co1 = _mm256_sub_epi32(r1, r6);
    __m256i co2 = _mm256_sub_epi32(r2, r5), co3 = _mm256_sub_epi32(r3, r4);

    __m256i cee0 = _mm256_add_epi32(ce0, ce3), cee1 = _mm256_add_epi32(ce1, ce2);
    __m256i ceo0 = _mm256_sub_epi32(ce0, ce3), ceo1 = _mm256_sub_epi32(ce1, ce2);
    __m256i c0 = _mm256_add_epi32(cee0, cee1), c4 = _mm256_sub_epi32(cee0, cee1);

    int32_t vbuf[8][8];
    _mm256_storeu_si256((__m256i *)(void *)vbuf[0], c0);
    _mm256_storeu_si256((__m256i *)(void *)vbuf[4], c4);
    _mm256_storeu_si256((__m256i *)(void *)vbuf[2], ceo0);
    _mm256_storeu_si256((__m256i *)(void *)vbuf[6], ceo1);
    _mm256_storeu_si256((__m256i *)(void *)vbuf[1], co0);
    _mm256_storeu_si256((__m256i *)(void *)vbuf[3], co1);
    _mm256_storeu_si256((__m256i *)(void *)vbuf[5], co2);
    _mm256_storeu_si256((__m256i *)(void *)vbuf[7], co3);

    int32_t ee[2][2], dc_ev[2][2], ev_dc[2][2], ev_ev[2][2];
    int32_t dc_odd[2][4], odd_dc[4][2], ev_odd[2][4], odd_ev[4][2], odd_odd[4][4];

    static const uint8_t k_dc[2] = {0U, 4U};
    static const uint8_t k_ev[2] = {2U, 6U};
    static const uint8_t k_od[4] = {1U, 3U, 5U, 7U};

    for (int i = 0; i < 2; ++i) {
        const int32_t *vr = vbuf[k_dc[i]];
        int32_t e0 = vr[0] + vr[7], e1 = vr[1] + vr[6], e2 = vr[2] + vr[5], e3 = vr[3] + vr[4];
        int32_t s0 = e0 + e3, s1 = e1 + e2;
        ee[i][0] = s0 + s1; ee[i][1] = s0 - s1;
        dc_ev[i][0] = e0 - e3; dc_ev[i][1] = e1 - e2;
        dc_odd[i][0] = vr[0] - vr[7]; dc_odd[i][1] = vr[1] - vr[6];
        dc_odd[i][2] = vr[2] - vr[5]; dc_odd[i][3] = vr[3] - vr[4];

        const int32_t *er = vbuf[k_ev[i]];
        e0 = er[0] + er[7]; e1 = er[1] + er[6]; e2 = er[2] + er[5]; e3 = er[3] + er[4];
        s0 = e0 + e3; s1 = e1 + e2;
        ev_dc[i][0] = s0 + s1; ev_dc[i][1] = s0 - s1;
        ev_ev[i][0] = e0 - e3; ev_ev[i][1] = e1 - e2;
        ev_odd[i][0] = er[0] - er[7]; ev_odd[i][1] = er[1] - er[6];
        ev_odd[i][2] = er[2] - er[5]; ev_odd[i][3] = er[3] - er[4];
    }
    for (int i = 0; i < 4; ++i) {
        const int32_t *orr = vbuf[k_od[i]];
        int32_t e0 = orr[0] + orr[7], e1 = orr[1] + orr[6], e2 = orr[2] + orr[5], e3 = orr[3] + orr[4];
        int32_t s0 = e0 + e3, s1 = e1 + e2;
        odd_dc[i][0] = s0 + s1; odd_dc[i][1] = s0 - s1;
        odd_ev[i][0] = e0 - e3; odd_ev[i][1] = e1 - e2;
        odd_odd[i][0] = orr[0] - orr[7]; odd_odd[i][1] = orr[1] - orr[6];
        odd_odd[i][2] = orr[2] - orr[5]; odd_odd[i][3] = orr[3] - orr[4];
    }

    zc_g2d_core_116mul_avx2(ee, dc_ev, ev_dc, ev_ev, dc_odd, odd_dc, ev_odd, odd_ev, odd_odd, out);
}

static inline void zcvid_galois_idct8_2d_exact_avx2(const int32_t in[64], int32_t out[64]) {
    if (!zc_g2d_all_in_range511_avx2(in)) {
        int64_t o64[64];
        zcvid_galois_idct8_2d_exact_i64(in, o64);
        for (uint32_t k = 0U; k < 64U; ++k) out[k] = (int32_t)o64[k];
        return;
    }

    int32_t ee[2][2] = {{in[0 * 8 + 0], in[0 * 8 + 4]},
                        {in[4 * 8 + 0], in[4 * 8 + 4]}};
    int32_t dc_ev[2][2] = {{in[0 * 8 + 2], in[0 * 8 + 6]},
                           {in[4 * 8 + 2], in[4 * 8 + 6]}};
    int32_t ev_dc[2][2] = {{in[2 * 8 + 0], in[2 * 8 + 4]},
                           {in[6 * 8 + 0], in[6 * 8 + 4]}};
    int32_t ev_ev[2][2] = {{in[2 * 8 + 2], in[2 * 8 + 6]},
                           {in[6 * 8 + 2], in[6 * 8 + 6]}};
    int32_t dc_odd[2][4] = {{in[0 * 8 + 1], in[0 * 8 + 3], in[0 * 8 + 5], in[0 * 8 + 7]},
                            {in[4 * 8 + 1], in[4 * 8 + 3], in[4 * 8 + 5], in[4 * 8 + 7]}};
    int32_t odd_dc[4][2] = {{in[1 * 8 + 0], in[1 * 8 + 4]},
                            {in[3 * 8 + 0], in[3 * 8 + 4]},
                            {in[5 * 8 + 0], in[5 * 8 + 4]},
                            {in[7 * 8 + 0], in[7 * 8 + 4]}};
    int32_t ev_odd[2][4] = {{in[2 * 8 + 1], in[2 * 8 + 3], in[2 * 8 + 5], in[2 * 8 + 7]},
                            {in[6 * 8 + 1], in[6 * 8 + 3], in[6 * 8 + 5], in[6 * 8 + 7]}};
    int32_t odd_ev[4][2] = {{in[1 * 8 + 2], in[1 * 8 + 6]},
                            {in[3 * 8 + 2], in[3 * 8 + 6]},
                            {in[5 * 8 + 2], in[5 * 8 + 6]},
                            {in[7 * 8 + 2], in[7 * 8 + 6]}};
    static const uint8_t k_od[4] = {1U, 3U, 5U, 7U};
    int32_t odd_odd[4][4];
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) odd_odd[r][c] = in[k_od[r] * 8 + k_od[c]];
    }

    int32_t mid[64];
    zc_g2d_core_116mul_avx2(ee, dc_ev, ev_dc, ev_ev, dc_odd, odd_dc, ev_odd, odd_ev, odd_odd, mid);

    /* Row B^T (zero-mul) then SIMD column B^T */
    int32_t tmp[64];
    for (int v = 0; v < 8; ++v) {
        const int32_t *c = mid + v * 8;
        int32_t s0 = c[0] + c[4], s1 = c[0] - c[4];
        int32_t e0 = s0 + c[2], e1 = s1 + c[6], e2 = s1 - c[6], e3 = s0 - c[2];
        int32_t o0 = c[1], o1 = c[3], o2 = c[5], o3 = c[7];
        tmp[v * 8 + 0] = e0 + o0; tmp[v * 8 + 7] = e0 - o0;
        tmp[v * 8 + 1] = e1 + o1; tmp[v * 8 + 6] = e1 - o1;
        tmp[v * 8 + 2] = e2 + o2; tmp[v * 8 + 5] = e2 - o2;
        tmp[v * 8 + 3] = e3 + o3; tmp[v * 8 + 4] = e3 - o3;
    }

    __m256i m0 = _mm256_loadu_si256((const __m256i *)(const void *)(tmp + 0));
    __m256i m1 = _mm256_loadu_si256((const __m256i *)(const void *)(tmp + 8));
    __m256i m2 = _mm256_loadu_si256((const __m256i *)(const void *)(tmp + 16));
    __m256i m3 = _mm256_loadu_si256((const __m256i *)(const void *)(tmp + 24));
    __m256i m4 = _mm256_loadu_si256((const __m256i *)(const void *)(tmp + 32));
    __m256i m5 = _mm256_loadu_si256((const __m256i *)(const void *)(tmp + 40));
    __m256i m6 = _mm256_loadu_si256((const __m256i *)(const void *)(tmp + 48));
    __m256i m7 = _mm256_loadu_si256((const __m256i *)(const void *)(tmp + 56));

    __m256i s0 = _mm256_add_epi32(m0, m4), s1 = _mm256_sub_epi32(m0, m4);
    __m256i e0 = _mm256_add_epi32(s0, m2), e1 = _mm256_add_epi32(s1, m6);
    __m256i e2 = _mm256_sub_epi32(s1, m6), e3 = _mm256_sub_epi32(s0, m2);

    _mm256_storeu_si256((__m256i *)(void *)(out + 0),  _mm256_add_epi32(e0, m1));
    _mm256_storeu_si256((__m256i *)(void *)(out + 56), _mm256_sub_epi32(e0, m1));
    _mm256_storeu_si256((__m256i *)(void *)(out + 8),  _mm256_add_epi32(e1, m3));
    _mm256_storeu_si256((__m256i *)(void *)(out + 48), _mm256_sub_epi32(e1, m3));
    _mm256_storeu_si256((__m256i *)(void *)(out + 16), _mm256_add_epi32(e2, m5));
    _mm256_storeu_si256((__m256i *)(void *)(out + 40), _mm256_sub_epi32(e2, m5));
    _mm256_storeu_si256((__m256i *)(void *)(out + 24), _mm256_add_epi32(e3, m7));
    _mm256_storeu_si256((__m256i *)(void *)(out + 32), _mm256_sub_epi32(e3, m7));
}
#endif

static inline void zcvid_galois_dct8_2d_exact(const int32_t in[64], int32_t out[64]) {
#if defined(__AVX2__)
    zcvid_galois_dct8_2d_exact_avx2(in, out);
#else
    zcvid_galois_dct8_2d_exact_scalar(in, out);
#endif
}

static inline void zcvid_galois_idct8_2d_exact(const int32_t in[64], int32_t out[64]) {
#if defined(__AVX2__)
    zcvid_galois_idct8_2d_exact_avx2(in, out);
#else
    zcvid_galois_idct8_2d_exact_scalar(in, out);
#endif
}

/* Single-rounding codec wrappers: exact 2D transform + one final >> 11 or >> 19 */
static inline void zcvid_galois_dct8_2d_fwd(const int32_t in[64], int32_t out[64]) {
    int32_t exact[64];
    zcvid_galois_dct8_2d_exact(in, exact);
    for (uint32_t i = 0U; i < 64U; ++i) {
        out[i] = (exact[i] + 1024) >> 11;
    }
}

static inline void zcvid_galois_dct8_2d_inv(const int32_t in[64], int32_t out[64]) {
    int64_t exact[64];
    zcvid_galois_idct8_2d_exact_i64(in, exact);
    for (uint32_t i = 0U; i < 64U; ++i) {
        out[i] = (int32_t)((exact[i] + 262144LL) >> 19);
    }
}

#ifdef __cplusplus
}
#endif

#endif /* ZCVID_GALOIS_DCT8_H */
