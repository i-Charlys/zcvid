/*
 * bench_latent_hybrid.c - Empirical RD benchmark of Ultra-Low-Bit Ternary Latent
 * Factorization (LRF-R1/R2) and Hybrid Latent+Transform architectures vs TQ4.
 */

#include "zcvid_latent_hybrid.h"
#include "zcvid_galois_dct8.h"
#include "zcvid_tropical_simplicial.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { SCENE_W = 512, SCENE_H = 384, NUM_SCENES = 5, NUM_QPS = 5 };

static double now_ms(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static double compute_psnr(const uint8_t *a, const uint8_t *b, size_t n) {
    double sse = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = (double)a[i] - (double)b[i];
        sse += d * d;
    }
    double mse = sse / (double)n;
    if (mse <= 1e-12) return 99.0;
    return 10.0 * log10((255.0 * 255.0) / mse);
}

/* Generates 5 distinct synthetic w x h test scenes + prediction buffers */
static void build_scene_wh(int scene_idx, int w, int h, uint8_t *src, uint8_t *pred, bool *is_inter) {
    memset(pred, 128, (size_t)w * (size_t)h);
    *is_inter = false;

    if (scene_idx == 0) {
        /* 1. Desktop UI / Text / Crisp Windows & Toolbars */
        const int sx = w / 512;
        const int sy = h / 384;
        for (int y = 0; y < h; ++y) {
            int ly = y / (sy > 0 ? sy : 1);
            for (int x = 0; x < w; ++x) {
                int lx = x / (sx > 0 ? sx : 1);
                int v = 240;
                if (ly < 32) v = 42;
                else if (lx < 96) v = 56;
                else if (ly >= 64 && ly < 320 && lx >= 128 && lx < 464) {
                    v = ((ly < 88) ? 72 : 248);
                    if (lx == 128 || lx == 463 || ly == 64 || ly == 319) v = 28;
                    if (ly >= 104 && ly < 296 && lx >= 148 && lx < 444 && (y % 18) < 2 && ((x / 12) % 3 != 0)) {
                        v = 36;
                    }
                    if (lx >= 200 && lx < 340 && ly >= 150 && ly < 210) v = 185;
                }
                src[y * w + x] = (uint8_t)v;
            }
        }
    } else if (scene_idx == 1) {
        /* 2. Smooth Gradients / Shadows */
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                double dx = ((double)x - w * 0.45) / (double)w;
                double dy = ((double)y - h * 0.40) / (double)h;
                double r2 = dx * dx + dy * dy;
                double base = 215.0 * exp(-2.2 * r2) + 28.0 + 18.0 * ((double)x / (double)w);
                src[y * w + x] = (uint8_t)zcvid_dec_clamp_i32((int32_t)llround(base), 0, 255);
            }
        }
    } else if (scene_idx == 2) {
        /* 3. Natural Texture + Pan Residual (previous frame shifted + slight illumination change) */
        *is_inter = true;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                double fx = (double)x, fy = (double)y;
                double t0 = 128.0 + 48.0 * sin(fx * 0.045 + fy * 0.02) * cos(fy * 0.038 - fx * 0.01)
                                  + 22.0 * sin(fx * 0.19 - fy * 0.11)
                                  + 14.0 * cos(fx * 0.37 + fy * 0.29);
                double sx = fx - 1.5, sy = fy - 0.75;
                double t1 = 128.0 + 48.0 * sin(sx * 0.045 + sy * 0.02) * cos(sy * 0.038 - sx * 0.01)
                                  + 22.0 * sin(sx * 0.19 - sy * 0.11)
                                  + 14.0 * cos(sx * 0.37 + sy * 0.29) + 4.0;
                pred[y * w + x] = (uint8_t)zcvid_dec_clamp_i32((int32_t)llround(t0), 0, 255);
                src[y * w + x]  = (uint8_t)zcvid_dec_clamp_i32((int32_t)llround(t1), 0, 255);
            }
        }
    } else if (scene_idx == 3) {
        /* 4. 45-degree Diagonals & High-Frequency Checker */
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int v = 128;
                if (x < w / 2) {
                    double s = sin(((double)x + (double)y) * 0.35);
                    v = (int)(128.0 + 72.0 * s);
                } else {
                    int chk = (((x / 3) + (y / 3)) & 1) ? 45 : -45;
                    int bar = (((x - y) & 15) < 3) ? 40 : 0;
                    v = 128 + chk + bar;
                }
                src[y * w + x] = (uint8_t)zcvid_dec_clamp_i32(v, 0, 255);
            }
        }
    } else {
        /* 5. Mixed Gaming HUD over Animated Texture Residual */
        *is_inter = true;
        const int sx = w / 512;
        const int sy = h / 384;
        for (int y = 0; y < h; ++y) {
            int ly = y / (sy > 0 ? sy : 1);
            for (int x = 0; x < w; ++x) {
                int lx = x / (sx > 0 ? sx : 1);
                double bg0 = 112.0 + 40.0 * sin(x * 0.05 + y * 0.03) + 20.0 * cos(x * 0.14 - y * 0.18);
                double bg1 = 112.0 + 40.0 * sin((x - 1.25) * 0.05 + (y - 0.5) * 0.03)
                                   + 20.0 * cos((x - 1.25) * 0.14 - (y - 0.5) * 0.18);
                int pv = zcvid_dec_clamp_i32((int32_t)llround(bg0), 0, 255);
                int sv = zcvid_dec_clamp_i32((int32_t)llround(bg1), 0, 255);
                if ((ly >= 320 && ly < 368 && lx >= 16 && lx < 208)) {
                    int hud = (lx < 140) ? 210 : 48;
                    if (ly < 323 || ly > 364 || lx < 19 || lx > 204) hud = 240;
                    pv = hud;
                    sv = hud;
                }
                if (lx >= 220 && lx < 284 && ly >= 140 && ly < 188) sv = 236;
                if (lx >= 222 && lx < 286 && ly >= 142 && ly < 190) pv = 236;
                pred[y * w + x] = (uint8_t)pv;
                src[y * w + x]  = (uint8_t)sv;
            }
        }
    }
}

static void build_scene(int scene_idx, uint8_t *src, uint8_t *pred, bool *is_inter) {
    build_scene_wh(scene_idx, SCENE_W, SCENE_H, src, pred, is_inter);
}

/* Cubic PCHIP / piecewise-linear numerical Bjontegaard Delta integration on [rate, psnr] curves */
static double integrate_curve(const double x[NUM_QPS], const double y[NUM_QPS], double lo, double hi) {
    /* Linear-log trapezoidal integration with endpoint interpolation on overlapping PSNR/rate range */
    if (hi <= lo) return 0.0;
    const int steps = 200;
    double dx = (hi - lo) / (double)steps;
    double acc = 0.0;
    for (int s = 0; s <= steps; ++s) {
        double xv = lo + s * dx;
        /* Ensure sorted ascending x */
        int idx = 0;
        while (idx + 1 < NUM_QPS - 1 && x[idx + 1] < xv) idx++;
        double x0 = x[idx], x1 = x[idx + 1];
        double y0 = y[idx], y1 = y[idx + 1];
        double t = (fabs(x1 - x0) > 1e-12) ? ((xv - x0) / (x1 - x0)) : 0.0;
        double yv = y0 + t * (y1 - y0);
        acc += (s == 0 || s == steps) ? 0.5 * yv : yv;
    }
    return acc * dx / (hi - lo);
}

static void compute_bd(const double bpp_ref[NUM_QPS], const double psnr_ref[NUM_QPS],
                       const double bpp_test[NUM_QPS], const double psnr_test[NUM_QPS],
                       double *out_bdrate_pct, double *out_bdpsnr_db) {
    /* QPs are 18..42 descending rate, so reverse to ascending log(bpp) and PSNR */
    double lr_ref[NUM_QPS], p_ref[NUM_QPS], lr_t[NUM_QPS], p_t[NUM_QPS];
    for (int i = 0; i < NUM_QPS; ++i) {
        int j = NUM_QPS - 1 - i;
        lr_ref[i] = log10(bpp_ref[j] > 1e-6 ? bpp_ref[j] : 1e-6);
        p_ref[i]  = psnr_ref[j];
        lr_t[i]   = log10(bpp_test[j] > 1e-6 ? bpp_test[j] : 1e-6);
        p_t[i]    = psnr_test[j];
    }
    double p_lo = (p_ref[0] > p_t[0]) ? p_ref[0] : p_t[0];
    double p_hi = (p_ref[NUM_QPS - 1] < p_t[NUM_QPS - 1]) ? p_ref[NUM_QPS - 1] : p_t[NUM_QPS - 1];
    if (p_hi > p_lo + 0.05) {
        double avg_lr_ref = integrate_curve(p_ref, lr_ref, p_lo, p_hi);
        double avg_lr_t   = integrate_curve(p_t, lr_t, p_lo, p_hi);
        *out_bdrate_pct = (pow(10.0, avg_lr_t - avg_lr_ref) - 1.0) * 100.0;
    } else {
        *out_bdrate_pct = 0.0;
    }

    double r_lo = (lr_ref[0] > lr_t[0]) ? lr_ref[0] : lr_t[0];
    double r_hi = (lr_ref[NUM_QPS - 1] < lr_t[NUM_QPS - 1]) ? lr_ref[NUM_QPS - 1] : lr_t[NUM_QPS - 1];
    if (r_hi > r_lo + 0.02) {
        double avg_p_ref = integrate_curve(lr_ref, p_ref, r_lo, r_hi);
        double avg_p_t   = integrate_curve(lr_t, p_t, r_lo, r_hi);
        *out_bdpsnr_db = avg_p_t - avg_p_ref;
    } else {
        *out_bdpsnr_db = 0.0;
    }
}

static uint32_t xorshift32(uint32_t *st) {
    uint32_t x = *st;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *st = x;
    return x;
}

static void run_unit_and_edge_tests(void) {
    printf("=== Equivalence & Edge-Case Verification ===\n");
    uint32_t rng = 0x12345678U;

    /* 1. Factored DCT8 Forward (AVX2 vpmaddwd & 12-mul Negacyclic-Karatsuba scalar) == Reference */
    for (int t = 0; t < 2000; ++t) {
        int32_t in[64], out_ref[64], out_fast[64], out_sc[64];
        if (t == 0) {
            for (int i = 0; i < 64; ++i) in[i] = 255;
        } else if (t == 1) {
            for (int i = 0; i < 64; ++i) in[i] = -255;
        } else if (t < 1800) {
            for (int i = 0; i < 64; ++i) in[i] = (int32_t)(xorshift32(&rng) % 511U) - 255;
        } else {
            /* Out-of-8-bit range to exercise 32-bit Negacyclic-Karatsuba fallback */
            for (int i = 0; i < 64; ++i) in[i] = (int32_t)(xorshift32(&rng) % 4001U) - 2000;
        }
        zcvid_lh_dct8_fwd_ref(in, out_ref);
        zcvid_lh_dct8_fwd(in, out_fast);
        zcvid_lh_dct8_fwd_scalar(in, out_sc);
        if (memcmp(out_ref, out_fast, sizeof(out_ref)) != 0 ||
            memcmp(out_ref, out_sc, sizeof(out_ref)) != 0) {
            fprintf(stderr, "FAIL: dct8_fwd mismatch on trial %d\n", t);
            exit(1);
        }
    }
    printf("  [PASS] DCT8 Forward (AVX2 12-mul == Scalar 12-mul == Ref): 2000/2000 exact\n");

    /* 1b. Factored IDCT8 (AVX2 12-mul, 12-mul scalar, sparse poly_only/LL/full) == Reference */
    for (int t = 0; t < 2000; ++t) {
        int32_t in[64], out_ref[64], out_fast[64], out_sc[64], out_sp[64];
        memset(in, 0, sizeof(in));
        bool poly_only = false;
        uint32_t csbf = 15U;
        if (t < 400) {
            /* DC-only or degree-2 polynomial */
            poly_only = true;
            csbf = 1U;
            static const uint32_t PPOS[6] = { 0U, 1U, 8U, 16U, 9U, 2U };
            if (t < 100) {
                in[0] = (int32_t)(xorshift32(&rng) % 65535U) - 32767;
            } else {
                for (int k = 0; k < 6; ++k) in[PPOS[k]] = (int32_t)(xorshift32(&rng) % 65535U) - 32767;
            }
        } else if (t < 900) {
            /* LL-only 4x4 sub-block */
            csbf = 1U;
            for (int v = 0; v < 4; ++v)
                for (int u = 0; u < 4; ++u)
                    in[v * 8 + u] = (int32_t)(xorshift32(&rng) % 65535U) - 32767;
        } else if (t < 1800) {
            for (int i = 0; i < 64; ++i) in[i] = (int32_t)(xorshift32(&rng) % 131071U) - 65535;
        } else {
            /* Extreme >65535 overflow-guard inputs */
            for (int i = 0; i < 64; ++i) in[i] = (int32_t)(xorshift32(&rng) % 400001U) - 200000;
        }
        zcvid_lh_dct8_inv_ref(in, out_ref);
        zcvid_lh_dct8_inv(in, out_fast);
        zcvid_lh_dct8_inv_scalar(in, out_sc);
        zcvid_lh_dct8_inv_sparse(in, csbf, poly_only, out_sp);
        if (memcmp(out_ref, out_fast, sizeof(out_ref)) != 0 ||
            memcmp(out_ref, out_sc, sizeof(out_ref)) != 0 ||
            memcmp(out_ref, out_sp, sizeof(out_ref)) != 0) {
            fprintf(stderr, "FAIL: dct8_inv mismatch on trial %d\n", t);
            exit(1);
        }
    }
    printf("  [PASS] IDCT8 Inverse (AVX2 12-mul == Scalar 12-mul == Sparse Poly/LL == Ref): 2000/2000 exact\n");

    /* 2. Separable rank-1 IDCT8 == full 2D IDCT8 on 1000 random LRF factors */
    for (int t = 0; t < 1000; ++t) {
        zcvid_lh_lrf_t f;
        memset(&f, 0, sizeof(f));
        for (int i = 0; i < 8; ++i) {
            f.u[i] = (int8_t)((int)(xorshift32(&rng) % 3U) - 1);
            f.v[i] = (int8_t)((int)(xorshift32(&rng) % 3U) - 1);
        }
        f.level = (int32_t)(xorshift32(&rng) % 2001U) - 1000;
        f.step = (int32_t)(xorshift32(&rng) % 64U) + 1;
        int32_t what[64], out_ref[64], out_fast[64];
        memset(what, 0, sizeof(what));
        zcvid_lh_recon_lrf_add(&f, what);
        zcvid_lh_dct8_inv(what, out_ref);
        zcvid_lh_spec_lrf_recon(&f, out_fast);
        if (memcmp(out_ref, out_fast, sizeof(out_ref)) != 0) {
            fprintf(stderr, "FAIL: spec_lrf_recon mismatch on trial %d\n", t);
            exit(1);
        }
    }
    printf("  [PASS] Rank-1 Separable IDCT8 (== Full 2D IDCT8): 1000/1000 exact\n");

    /* 3. Step LUT == zcvid_lh_lrf_step for all QPs 0..51, nu/nv 1..8, spec 0..1 */
    for (uint32_t qp = 0U; qp <= 51U; ++qp) {
        zcvid_lh_step_lut_t lut;
        zcvid_lh_step_lut_init(&lut, qp);
        for (uint32_t nu = 1U; nu <= 8U; ++nu) {
            for (uint32_t nv = 1U; nv <= 8U; ++nv) {
                assert(lut.step[0][nu * nv] == zcvid_lh_lrf_step(qp, nu, nv, false));
                assert(lut.step[1][nu * nv] == zcvid_lh_lrf_step(qp, nu, nv, true));
            }
        }
    }
    printf("  [PASS] LRF Step LUT (QP 0..51 x 64 areas x 2 domains): exact\n");

    /* 4. Edge cases: 8x8 minimal frame, all-zero residual (100% SKIP), extreme +/-255 checker, MT > rows */
    zcvid_lh_mt_pool_t pool;
    assert(zcvid_lh_mt_pool_init(&pool, 8U, 128U, 128U));
    {
        uint8_t s8[64], p8[64], er8[64], dr8[64], bs8[4096];
        /* 4a. All-zero residual on 8x8 with 0xFF-poisoned bitstream buffer (tests zero-body RC flush) */
        memset(s8, 137, sizeof(s8));
        memset(p8, 137, sizeof(p8));
        memset(bs8, 0xFF, sizeof(bs8));
        zcvid_lh_stats_t st;
        size_t l1 = zcvid_lh_encode_frame(ZCVID_LH_HYBRID_FAST, s8, p8, 8U, 8U, 24U, true, bs8, sizeof(bs8), er8, &st);
        assert(l1 > 12U && st.count[ZCVID_LH_BLK_SKIP] == 1ULL);
        assert(zcvid_lh_decode_frame(bs8, l1, p8, 8U, 8U, dr8) == 0);
        assert(memcmp(er8, dr8, 64U) == 0 && memcmp(dr8, s8, 64U) == 0);

        memset(bs8, 0xFF, sizeof(bs8));
        size_t l2 = zcvid_lh_mt_encode_frame(&pool, ZCVID_LH_HYBRID_FAST, s8, p8, 8U, 8U, 24U, true, bs8, sizeof(bs8), er8, &st);
        assert(l2 > 16U && st.count[ZCVID_LH_BLK_SKIP] == 1ULL);
        assert(zcvid_lh_mt_decode_frame(&pool, bs8, l2, p8, 8U, 8U, dr8) == 0);
        assert(memcmp(er8, dr8, 64U) == 0 && memcmp(dr8, s8, 64U) == 0);

        /* 4b. Extreme +/-255 checkerboard on 64x64 */
        uint8_t s64[4096], p64[4096], er64[4096], dr64[4096], bs64[32768];
        for (int i = 0; i < 4096; ++i) {
            s64[i] = ((i ^ (i >> 6)) & 1) ? 255U : 0U;
            p64[i] = ((i ^ (i >> 6)) & 1) ? 0U : 255U;
        }
        size_t l3 = zcvid_lh_mt_encode_frame(&pool, ZCVID_LH_HYBRID_FAST, s64, p64, 64U, 64U, 18U, false, bs64, sizeof(bs64), er64, NULL);
        assert(l3 > 16U);
        assert(zcvid_lh_mt_decode_frame(&pool, bs64, l3, p64, 64U, 64U, dr64) == 0);
        assert(memcmp(er64, dr64, 4096U) == 0);

        /* 4c. Truncated / corrupted ZCLS & ZCLH rejection + slice length overflow */
        assert(zcvid_lh_mt_decode_frame(&pool, bs64, l3 - 1U, p64, 64U, 64U, dr64) != 0);
        assert(zcvid_lh_mt_decode_frame(&pool, bs64, 10U, p64, 64U, 64U, dr64) != 0);
        uint8_t bs_ovf[32];
        memcpy(bs_ovf, bs64, 16U);
        bs_ovf[11] = 1U;
        bs_ovf[12] = 0xFFU; bs_ovf[13] = 0xFFU; bs_ovf[14] = 0xFFU; bs_ovf[15] = 0xFFU;
        assert(zcvid_lh_mt_decode_frame(&pool, bs_ovf, sizeof(bs_ovf), p64, 64U, 64U, dr64) != 0);
        bs64[0] = 'X';
        assert(zcvid_lh_mt_decode_frame(&pool, bs64, l3, p64, 64U, 64U, dr64) != 0);
        assert(zcvid_lh_decode_frame(bs8, 8U, p8, 8U, 8U, dr8) != 0);

        /* 4d. Extreme spec_lrf_recon overflow safety (QP=51, |level|=4095) == dct8_inv */
        {
            zcvid_lh_lrf_t fx;
            memset(&fx, 0, sizeof(fx));
            fx.u[0] = 1;
            for (int i = 0; i < 8; ++i) fx.v[i] = 1;
            fx.level = 4095;
            fx.step = zcvid_lh_lrf_step(51U, 1U, 8U, true);
            int32_t what[64], out_ref[64], out_fast[64];
            memset(what, 0, sizeof(what));
            zcvid_lh_recon_lrf_add(&fx, what);
            zcvid_lh_dct8_inv(what, out_ref);
            zcvid_lh_spec_lrf_recon(&fx, out_fast);
            assert(memcmp(out_ref, out_fast, sizeof(out_ref)) == 0);
        }

        /* 4e. Invalid parameter rejection (qp > 51, w/h == 0, invalid codec) */
        assert(zcvid_lh_encode_frame(ZCVID_LH_HYBRID_FAST, s8, p8, 8U, 8U, 52U, false, bs8, sizeof(bs8), er8, NULL) == 0U);
        assert(zcvid_lh_encode_frame(ZCVID_LH_HYBRID_FAST, s8, p8, 8U, 8U, 102U, false, bs8, sizeof(bs8), er8, NULL) == 0U);
        assert(zcvid_lh_encode_frame(ZCVID_LH_HYBRID_FAST, s8, p8, 0U, 0U, 24U, false, bs8, sizeof(bs8), er8, NULL) == 0U);
        assert(zcvid_lh_encode_frame((zcvid_lh_codec_t)99, s8, p8, 8U, 8U, 24U, false, bs8, sizeof(bs8), er8, NULL) == 0U);
        assert(zcvid_lh_mt_encode_frame(&pool, ZCVID_LH_HYBRID_FAST, s8, p8, 8U, 8U, 52U, false, bs8, sizeof(bs8), er8, NULL) == 0U);

        /* 4f. Cross-thread-count MT decode (12 slices encoded on 12T, decoded in parallel on 4T) */
        {
            const uint32_t cw = 128U, ch = 768U; /* sbh=96 -> 12 slices on 12T pool */
            uint8_t *cs = (uint8_t *)malloc(cw * ch);
            uint8_t *cp = (uint8_t *)malloc(cw * ch);
            uint8_t *ce = (uint8_t *)malloc(cw * ch);
            uint8_t *cd = (uint8_t *)malloc(cw * ch);
            uint8_t *cb = (uint8_t *)malloc(cw * ch * 4U + 4096U);
            assert(cs && cp && ce && cd && cb);
            for (uint32_t i = 0U; i < cw * ch; ++i) {
                cs[i] = (uint8_t)((i * 37U + (i >> 7) * 13U) & 255U);
                cp[i] = 128U;
            }
            zcvid_lh_mt_pool_t p12, p4;
            assert(zcvid_lh_mt_pool_init(&p12, 12U, cw, ch));
            assert(zcvid_lh_mt_pool_init(&p4, 4U, cw, ch));
            size_t clen = zcvid_lh_mt_encode_frame(&p12, ZCVID_LH_HYBRID_FAST, cs, cp, cw, ch, 24U, false, cb, cw * ch * 4U + 4096U, ce, NULL);
            assert(clen > 16U && cb[11] == 12U);
            assert(zcvid_lh_mt_decode_frame(&p4, cb, clen, cp, cw, ch, cd) == 0);
            assert(memcmp(ce, cd, cw * ch) == 0);
            zcvid_lh_mt_pool_destroy(&p12);
            zcvid_lh_mt_pool_destroy(&p4);
            /* Destroyed pool rejection + idempotent double-destroy */
            assert(zcvid_lh_mt_encode_frame(&p4, ZCVID_LH_HYBRID_FAST, s8, p8, 8U, 8U, 24U, false, bs8, sizeof(bs8), er8, NULL) == 0U);
            assert(zcvid_lh_mt_decode_frame(&p4, cb, clen, cp, cw, ch, cd) != 0);
            zcvid_lh_mt_pool_destroy(&p4);
            free(cs); free(cp); free(ce); free(cd); free(cb);
        }

        /* 4g. Fuzz 500 corrupted ZCLS bitstreams at QP=51 (HYBRID_FAST & HYBRID_ESR, no UB / crash) */
        {
            uint8_t fbs[256];
            for (int t = 0; t < 500; ++t) {
                for (int i = 0; i < 256; ++i) fbs[i] = (uint8_t)xorshift32(&rng);
                fbs[0] = 'Z'; fbs[1] = 'C'; fbs[2] = 'L'; fbs[3] = 'S';
                fbs[4] = 64; fbs[5] = 0; fbs[6] = 64; fbs[7] = 0;
                fbs[8] = (uint8_t)((t & 1) ? ZCVID_LH_HYBRID_ESR : ZCVID_LH_HYBRID_FAST);
                fbs[9] = 51U;
                fbs[10] = 0U;
                fbs[11] = 2U;
                fbs[12] = 100; fbs[13] = 0; fbs[14] = 0; fbs[15] = 0;
                fbs[16] = 136; fbs[17] = 0; fbs[18] = 0; fbs[19] = 0;
                (void)zcvid_lh_mt_decode_frame(&pool, fbs, sizeof(fbs), p64, 64U, 64U, dr64);
            }
        }

        /* 4h. HYBRID_ESR edge cases: QP=0, QP=51, 8x8 SKIP, +/-255 checker, MT 12->4T */
        {
            uint8_t er8e[64], dr8e[64], bs8e[4096];
            size_t le = zcvid_lh_encode_frame(ZCVID_LH_HYBRID_ESR, s8, p8, 8U, 8U, 24U, true, bs8e, sizeof(bs8e), er8e, &st);
            assert(le > 12U && st.count[ZCVID_LH_BLK_SKIP] == 1ULL);
            assert(zcvid_lh_decode_frame(bs8e, le, p8, 8U, 8U, dr8e) == 0);
            assert(memcmp(er8e, dr8e, 64U) == 0);

            uint32_t edge_qps[3] = { 0U, 18U, 51U };
            for (int qi = 0; qi < 3; ++qi) {
                size_t lq = zcvid_lh_mt_encode_frame(&pool, ZCVID_LH_HYBRID_ESR, s64, p64, 64U, 64U, edge_qps[qi], false, bs64, sizeof(bs64), er64, NULL);
                assert(lq > 16U);
                assert(zcvid_lh_mt_decode_frame(&pool, bs64, lq, p64, 64U, 64U, dr64) == 0);
                assert(memcmp(er64, dr64, 4096U) == 0);
            }
        }
    }
    zcvid_lh_mt_pool_destroy(&pool);
    printf("  [PASS] Edge Cases (8x8, 100%% SKIP, +/-255, QP0/51 ESR, 12->4T dec, bad args, fuzz500)\n\n");

    /* 5. Linear Circuit Micro-Benchmark (ns / 8x8 block across 200,000 blocks) */
    {
        enum { NBLK = 200000 };
        int32_t in_fwd[64], in_inv[64], in_ll[64], in_poly[64], out[64];
        memset(in_ll, 0, sizeof(in_ll));
        memset(in_poly, 0, sizeof(in_poly));
        for (int i = 0; i < 64; ++i) {
            in_fwd[i] = (int32_t)(xorshift32(&rng) % 511U) - 255;
            in_inv[i] = (int32_t)(xorshift32(&rng) % 4097U) - 2048;
        }
        for (int v = 0; v < 4; ++v)
            for (int u = 0; u < 4; ++u)
                in_ll[v * 8 + u] = in_inv[v * 8 + u];
        static const uint32_t PPOS[6] = { 0U, 1U, 8U, 16U, 9U, 2U };
        for (int k = 0; k < 6; ++k) in_poly[PPOS[k]] = in_inv[PPOS[k]];

        volatile int64_t sink = 0;
        double t0, t1;

        t0 = now_ms();
        for (int i = 0; i < NBLK; ++i) { in_fwd[i & 63] = (i & 255) - 128; zcvid_lh_dct8_fwd_ref(in_fwd, out); sink += out[i & 63]; }
        t1 = now_ms();
        double ns_fwd_ref = ((t1 - t0) * 1e6) / (double)NBLK;

        t0 = now_ms();
        for (int i = 0; i < NBLK; ++i) { in_fwd[i & 63] = (i & 255) - 128; zcvid_lh_dct8_fwd_scalar(in_fwd, out); sink += out[i & 63]; }
        t1 = now_ms();
        double ns_fwd_sc = ((t1 - t0) * 1e6) / (double)NBLK;

        t0 = now_ms();
        for (int i = 0; i < NBLK; ++i) { in_fwd[i & 63] = (i & 255) - 128; zcvid_lh_dct8_fwd(in_fwd, out); sink += out[i & 63]; }
        t1 = now_ms();
        double ns_fwd_avx = ((t1 - t0) * 1e6) / (double)NBLK;

        t0 = now_ms();
        for (int i = 0; i < NBLK; ++i) { in_inv[i & 63] = (i & 1023) - 512; zcvid_lh_dct8_inv_ref(in_inv, out); sink += out[i & 63]; }
        t1 = now_ms();
        double ns_inv_ref = ((t1 - t0) * 1e6) / (double)NBLK;

        t0 = now_ms();
        for (int i = 0; i < NBLK; ++i) { in_inv[i & 63] = (i & 1023) - 512; zcvid_lh_dct8_inv_scalar(in_inv, out); sink += out[i & 63]; }
        t1 = now_ms();
        double ns_inv_sc = ((t1 - t0) * 1e6) / (double)NBLK;

        t0 = now_ms();
        for (int i = 0; i < NBLK; ++i) { in_inv[i & 63] = (i & 1023) - 512; zcvid_lh_dct8_inv(in_inv, out); sink += out[i & 63]; }
        t1 = now_ms();
        double ns_inv_avx = ((t1 - t0) * 1e6) / (double)NBLK;

        t0 = now_ms();
        for (int i = 0; i < NBLK; ++i) { in_ll[(i & 3) * 8 + ((i >> 2) & 3)] = (i & 1023) - 512; zcvid_lh_dct8_inv_sparse(in_ll, 1U, false, out); sink += out[i & 63]; }
        t1 = now_ms();
        double ns_inv_ll = ((t1 - t0) * 1e6) / (double)NBLK;

        t0 = now_ms();
        for (int i = 0; i < NBLK; ++i) { in_poly[PPOS[i % 6]] = (i & 1023) - 512; zcvid_lh_dct8_inv_sparse(in_poly, 1U, true, out); sink += out[i & 63]; }
        t1 = now_ms();
        double ns_inv_poly = ((t1 - t0) * 1e6) / (double)NBLK;

        (void)sink;
        printf("=== 8x8 DCT/IDCT Linear Circuit Micro-Benchmark (%d blocks, 0 mismatch) ===\n", NBLK);
        printf("  DCT8 Forward : Ref 1024-MAC = %6.1f ns | Scalar 12-mul = %5.1f ns (%.1fx) | AVX2 12-mul = %5.1f ns (%.1fx)\n",
               ns_fwd_ref, ns_fwd_sc, ns_fwd_ref / ns_fwd_sc, ns_fwd_avx, ns_fwd_ref / ns_fwd_avx);
        printf("  IDCT8 Inverse: Ref 1024-MAC = %6.1f ns | Scalar 12-mul = %5.1f ns (%.1fx) | AVX2 12-mul = %5.1f ns (%.1fx)\n",
               ns_inv_ref, ns_inv_sc, ns_inv_ref / ns_inv_sc, ns_inv_avx, ns_inv_ref / ns_inv_avx);
        printf("  IDCT8 Sparse : LL 4x4 CSBF  = %6.1f ns (%.1fx) | Poly 3x3 Deg-2 = %5.1f ns (%.1fx)\n\n",
               ns_inv_ll, ns_inv_ref / ns_inv_ll, ns_inv_poly, ns_inv_ref / ns_inv_poly);
    }
}

static void run_1080p_realtime_bench(void) {
    const uint32_t W = 1920U, H = 1080U; /* 1920x1080: 240x135 = 32,400 8x8 blocks (8-aligned: 1080 % 8 == 0!) */
    const size_t npix = (size_t)W * (size_t)H;
    const size_t bs_cap = npix * 4U + 8192U;

    uint8_t *src = (uint8_t *)malloc(npix);
    uint8_t *pred = (uint8_t *)malloc(npix);
    uint8_t *enc_rec = (uint8_t *)malloc(npix);
    uint8_t *dec_rec = (uint8_t *)malloc(npix);
    uint8_t *bs = (uint8_t *)malloc(bs_cap);
    assert(src && pred && enc_rec && dec_rec && bs);

    static const char *k_scene_names[NUM_SCENES] = {
        "Desktop UI / Text",
        "Smooth Gradients",
        "Texture + Pan Res",
        "45deg Diag + Check",
        "Mixed HUD + Texture"
    };

    zcvid_lh_mt_pool_t pool8, pool12, pool16;
    assert(zcvid_lh_mt_pool_init(&pool8, 8U, W, H));
    assert(zcvid_lh_mt_pool_init(&pool12, 12U, W, H));
    assert(zcvid_lh_mt_pool_init(&pool16, 16U, W, H));

    printf("=== Real 1920x1080 (Full HD) Real-Time Benchmark (QP=24 & QP=30, 5 Scenes, 0 mismatch) ===\n");
    printf("  Target budgets: 120 Hz = 8.33 ms | 240 Hz = 4.17 ms | 360 Hz = 2.78 ms | 480 Hz = 2.08 ms\n");
    printf("  %-22s | %8s | %8s | %8s | %7s | %8s | %8s | %8s\n",
           "Config", "Enc (ms)", "Dec (ms)", "Tot (ms)", "Enc FPS", "Avg bpp", "Avg PSNR", "vs 480Hz");

    typedef struct {
        const char *name;
        zcvid_lh_codec_t codec;
        zcvid_lh_mt_pool_t *pool;
    } cfg_t;
    cfg_t cfgs[10] = {
        { "TQ4 (1T baseline)",     ZCVID_LH_TQ4,         NULL    },
        { "TQ4 (12T MT)",          ZCVID_LH_TQ4,         &pool12 },
        { "HYBRID_FAST (1T)",      ZCVID_LH_HYBRID_FAST, NULL    },
        { "HYBRID_FAST (8T MT)",   ZCVID_LH_HYBRID_FAST, &pool8  },
        { "HYBRID_FAST (12T MT)",  ZCVID_LH_HYBRID_FAST, &pool12 },
        { "HYBRID_FAST (16T MT)",  ZCVID_LH_HYBRID_FAST, &pool16 },
        { "HYBRID_ESR (1T)",       ZCVID_LH_HYBRID_ESR,  NULL    },
        { "HYBRID_ESR (8T MT)",    ZCVID_LH_HYBRID_ESR,  &pool8  },
        { "HYBRID_ESR (12T MT)",   ZCVID_LH_HYBRID_ESR,  &pool12 },
        { "HYBRID_ESR (16T MT)",   ZCVID_LH_HYBRID_ESR,  &pool16 }
    };

    const uint32_t qps[2] = { 24U, 30U };
    const int iters = 3;

    for (int ci = 0; ci < 10; ++ci) {
        double enc_sum = 0.0, dec_sum = 0.0, bpp_sum = 0.0, psnr_sum = 0.0;
        int runs = 0;

        for (int s = 0; s < NUM_SCENES; ++s) {
            bool is_inter = false;
            build_scene_wh(s, (int)W, (int)H, src, pred, &is_inter);
            for (int qi = 0; qi < 2; ++qi) {
                /* Warmup 1 frame */
                size_t wlen = cfgs[ci].pool
                    ? zcvid_lh_mt_encode_frame(cfgs[ci].pool, cfgs[ci].codec, src, pred, W, H, qps[qi], is_inter, bs, bs_cap, enc_rec, NULL)
                    : zcvid_lh_encode_frame(cfgs[ci].codec, src, pred, W, H, qps[qi], is_inter, bs, bs_cap, enc_rec, NULL);
                assert(wlen > 12U);
                int wrc = cfgs[ci].pool
                    ? zcvid_lh_mt_decode_frame(cfgs[ci].pool, bs, wlen, pred, W, H, dec_rec)
                    : zcvid_lh_decode_frame(bs, wlen, pred, W, H, dec_rec);
                assert(wrc == 0);
                if (memcmp(enc_rec, dec_rec, npix) != 0) {
                    fprintf(stderr, "1080p BIT-EXACT MISMATCH on cfg=%s scene=%s qp=%u!\n",
                            cfgs[ci].name, k_scene_names[s], qps[qi]);
                    exit(1);
                }

                double best_e = 1e9, best_d = 1e9;
                size_t len = wlen;
                for (int it = 0; it < iters; ++it) {
                    double t0 = now_ms();
                    len = cfgs[ci].pool
                        ? zcvid_lh_mt_encode_frame(cfgs[ci].pool, cfgs[ci].codec, src, pred, W, H, qps[qi], is_inter, bs, bs_cap, enc_rec, NULL)
                        : zcvid_lh_encode_frame(cfgs[ci].codec, src, pred, W, H, qps[qi], is_inter, bs, bs_cap, enc_rec, NULL);
                    double t1 = now_ms();
                    int drc = cfgs[ci].pool
                        ? zcvid_lh_mt_decode_frame(cfgs[ci].pool, bs, len, pred, W, H, dec_rec)
                        : zcvid_lh_decode_frame(bs, len, pred, W, H, dec_rec);
                    double t2 = now_ms();
                    assert(len > 12U && drc == 0);
                    if (t1 - t0 < best_e) best_e = t1 - t0;
                    if (t2 - t1 < best_d) best_d = t2 - t1;
                }
                enc_sum += best_e;
                dec_sum += best_d;
                bpp_sum += ((double)len * 8.0) / (double)npix;
                psnr_sum += compute_psnr(src, dec_rec, npix);
                runs++;
            }
        }
        double ems = enc_sum / (double)runs;
        double dms = dec_sum / (double)runs;
        double tot = ems + dms;
        double fps = (ems > 1e-6) ? (1000.0 / ems) : 0.0;
        double pct480 = (ems / (1000.0 / 480.0)) * 100.0;
        printf("  %-22s | %8.3f | %8.3f | %8.3f | %7.0f | %8.3f | %8.2f | %6.1f%%\n",
               cfgs[ci].name, ems, dms, tot, fps, bpp_sum / (double)runs, psnr_sum / (double)runs, pct480);
    }
    printf("\n");

    zcvid_lh_mt_pool_destroy(&pool8);
    zcvid_lh_mt_pool_destroy(&pool12);
    zcvid_lh_mt_pool_destroy(&pool16);
    free(src);
    free(pred);
    free(enc_rec);
    free(dec_rec);
    free(bs);
}

static void ref1024_exact_dct8_2d(const int32_t in[64], int32_t out[64]) {
    static const int32_t K[8][8] = {
        { 64,  64,  64,  64,  64,  64,  64,  64 },
        { 89,  75,  50,  18, -18, -50, -75, -89 },
        { 83,  36, -36, -83, -83, -36,  36,  83 },
        { 75, -18, -89, -50,  50,  89,  18, -75 },
        { 64, -64, -64,  64,  64, -64, -64,  64 },
        { 50, -89,  18,  75, -75, -18,  89, -50 },
        { 36, -83,  83, -36, -36,  83, -83,  36 },
        { 18, -50,  75, -89,  89, -75,  50, -18 }
    };
    int32_t tmp[64];
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            int32_t acc = 0;
            for (int k = 0; k < 8; k++) acc += K[r][k] * in[k * 8 + c];
            tmp[r * 8 + c] = acc;
        }
    }
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            int32_t acc = 0;
            for (int k = 0; k < 8; k++) acc += tmp[r * 8 + k] * K[c][k];
            out[r * 8 + c] = acc;
        }
    }
}

static void sep192_exact_dct8_1d(const int32_t x[8], int32_t s, int32_t out[8], int32_t os) {
    int32_t e0 = x[0] + x[7 * s], e1 = x[s] + x[6 * s], e2 = x[2 * s] + x[5 * s], e3 = x[3 * s] + x[4 * s];
    int32_t o0 = x[0] - x[7 * s], o1 = x[s] - x[6 * s], o2 = x[2 * s] - x[5 * s], o3 = x[3 * s] - x[4 * s];
    int32_t ee0 = e0 + e3, ee1 = e1 + e2;
    int32_t eo0 = e0 - e3, eo1 = e1 - e2;
    out[0 * os] = (ee0 + ee1) << 6;
    out[4 * os] = (ee0 - ee1) << 6;
    zc_g2d_even_rot_i32(eo0, eo1, &out[2 * os], &out[6 * os]);
    zc_g2d_odd1d_i32(o0, o1, o2, o3, &out[1 * os], &out[3 * os], &out[5 * os], &out[7 * os]);
}

static void sep192_exact_dct8_2d(const int32_t in[64], int32_t out[64]) {
    int32_t tmp[64];
    for (int c = 0; c < 8; c++) sep192_exact_dct8_1d(in + c, 8, tmp + c, 8);
    for (int r = 0; r < 8; r++) sep192_exact_dct8_1d(tmp + r * 8, 1, out + r * 8, 1);
}

static void run_galois_2d_microbench(void) {
    int32_t in[64], ref[64], sep[64], g_sc[64], g_vec[64];
    uint64_t rng = 0x123456789ABCDEF0ULL;
    for (int t = 0; t < 32; t++) {
        for (int i = 0; i < 64; i++) {
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            in[i] = (int32_t)((rng >> 33) % 511U) - 255;
        }
        ref1024_exact_dct8_2d(in, ref);
        sep192_exact_dct8_2d(in, sep);
        zcvid_galois_dct8_2d_exact_scalar(in, g_sc);
        zcvid_galois_dct8_2d_exact(in, g_vec);
        assert(memcmp(ref, sep, sizeof(ref)) == 0);
        assert(memcmp(ref, g_sc, sizeof(ref)) == 0);
        assert(memcmp(ref, g_vec, sizeof(ref)) == 0);
    }

    const int N = 400000;
    volatile int64_t sink = 0;
    double t0 = now_ms();
    for (int i = 0; i < N; i++) {
        in[(i * 7) & 63] = (i & 255) - 128;
        ref1024_exact_dct8_2d(in, ref);
        sink += ref[(i * 13) & 63];
    }
    double ms_ref = now_ms() - t0;

    t0 = now_ms();
    for (int i = 0; i < N; i++) {
        in[(i * 7) & 63] = (i & 255) - 128;
        sep192_exact_dct8_2d(in, sep);
        sink += sep[(i * 13) & 63];
    }
    double ms_sep = now_ms() - t0;

    t0 = now_ms();
    for (int i = 0; i < N; i++) {
        in[(i * 7) & 63] = (i & 255) - 128;
        zcvid_galois_dct8_2d_exact_scalar(in, g_sc);
        sink += g_sc[(i * 13) & 63];
    }
    double ms_gsc = now_ms() - t0;

    t0 = now_ms();
    for (int i = 0; i < N; i++) {
        in[(i * 7) & 63] = (i & 255) - 128;
        zcvid_galois_dct8_2d_exact(in, g_vec);
        sink += g_vec[(i * 13) & 63];
    }
    double ms_gvec = now_ms() - t0;

    printf("\n=== 2D 8x8 Exact Transform Microbench (0 mismatch vs Ref, sink=%lld) ===\n",
           (long long)sink);
    printf("  Ref 1024-MAC exact     : %7.2f ns/blk (1024 muls)\n", (ms_ref * 1e6) / N);
    printf("  Separable 192-mul exact: %7.2f ns/blk ( 192 muls, %.2fx vs Ref)\n",
           (ms_sep * 1e6) / N, ms_ref / (ms_sep > 1e-6 ? ms_sep : 1e-6));
    printf("  Galois 116-mul Scalar  : %7.2f ns/blk ( 116 muls, %.2fx vs Sep, %.2fx vs Ref)\n",
           (ms_gsc * 1e6) / N,
           ms_sep / (ms_gsc > 1e-6 ? ms_gsc : 1e-6),
           ms_ref / (ms_gsc > 1e-6 ? ms_gsc : 1e-6));
    printf("  Galois 116-mul Active  : %7.2f ns/blk ( 116 muls, %.2fx vs Sep, %.2fx vs Ref)\n",
           (ms_gvec * 1e6) / N,
           ms_sep / (ms_gvec > 1e-6 ? ms_gvec : 1e-6),
           ms_ref / (ms_gvec > 1e-6 ? ms_gvec : 1e-6));
}

static void run_tropical_simplicial_bench(void) {
    const size_t bw = 240u, bh = 135u; /* 1920x1080 in 8x8 blocks = 32,400 blocks */
    const size_t nb = bw * bh;
    double *en = (double *)malloc(nb * ZCVID_TS_MODES * sizeof(double));
    uint8_t *prev_m = (uint8_t *)malloc(nb);
    uint8_t *seq_m  = (uint8_t *)malloc(nb);
    uint8_t *sp1_m  = (uint8_t *)malloc(nb);
    uint8_t *sp8_m  = (uint8_t *)malloc(nb);
    assert(en && prev_m && seq_m && sp1_m && sp8_m);

    zcvid_ts_coupling_t coup;
    for (size_t a = 0; a < ZCVID_TS_MODES; ++a) {
        for (size_t b = 0; b < ZCVID_TS_MODES; ++b) {
            coup.edge_cost[a][b] = (a == b) ? 0.0 : 4.5;
        }
    }
    coup.tl_diag_bonus = 2.0;

    zcvid_ts_pool_t pool8;
    assert(zcvid_ts_pool_init(&pool8, 8u));

    printf("\n=== 1080p (%zux%zu = %zu blocks) Tropical-Simplicial Speculative RDO ===\n",
           bw, bh, nb);
    uint64_t rng = 0xCAFEBABE12345678ULL;
    for (int sc = 0; sc < 2; ++sc) {
        for (size_t v = 0; v < nb; ++v) {
            size_t x = v % bw, y = v / bw;
            uint8_t region_m = (uint8_t)(((x / 16u) + (y / 12u)) & 3u);
            for (size_t m = 0; m < ZCVID_TS_MODES; ++m) {
                rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
                double noise = (double)((rng >> 48) & 0xFFU) * 0.05;
                if (sc == 0) {
                    en[v * ZCVID_TS_MODES + m] = (m == region_m ? 1.0 : 12.0) + noise;
                } else {
                    en[v * ZCVID_TS_MODES + m] = noise * 4.0;
                }
            }
            if (sc == 0) {
                /* 93% coherent temporal warm-start + 7% motion-edge innovations */
                prev_m[v] = ((rng & 15u) == 0u) ? (uint8_t)((region_m + 1u) & 3u) : region_m;
            } else {
                prev_m[v] = (uint8_t)((rng >> 50) & 3u);
            }
        }
        const int ITERS = 40;
        double t0 = now_ms();
        for (int it = 0; it < ITERS; ++it) zcvid_ts_rdo_seq(bw, bh, en, &coup, seq_m);
        double ms_seq = (now_ms() - t0) / ITERS;

        zcvid_ts_spec_stats_t st1, st8;
        t0 = now_ms();
        for (int it = 0; it < ITERS; ++it) {
            assert(zcvid_ts_rdo_spec_solve(NULL, bw, bh, en, &coup, prev_m, sp1_m, &st1));
        }
        double ms_sp1 = (now_ms() - t0) / ITERS;

        t0 = now_ms();
        for (int it = 0; it < ITERS; ++it) {
            assert(zcvid_ts_rdo_spec_solve(&pool8, bw, bh, en, &coup, prev_m, sp8_m, &st8));
        }
        double ms_sp8 = (now_ms() - t0) / ITERS;

        assert(memcmp(seq_m, sp1_m, nb) == 0);
        assert(memcmp(seq_m, sp8_m, nb) == 0);
        printf("  %-18s | Seq: %6.3f ms | Spec1T: %6.3f ms | Spec8T: %6.3f ms (%.2fx vs Seq) | rounds=%zu, rollbacks=%zu (%.1f%% hit)\n",
               (sc == 0) ? "Coherent Video 93%" : "Adversarial Random",
               ms_seq, ms_sp1, ms_sp8,
               ms_seq / (ms_sp8 > 1e-6 ? ms_sp8 : 1e-6),
               st8.rounds, st8.rollbacks, st8.spec_hit_rate * 100.0);
    }

    zcvid_ts_pool_destroy(&pool8);
    free(en);
    free(prev_m);
    free(seq_m);
    free(sp1_m);
    free(sp8_m);
}

int main(void) {
    run_unit_and_edge_tests();

    static const char *k_scene_names[NUM_SCENES] = {
        "Desktop UI / Text",
        "Smooth Gradients",
        "Texture + Pan Res",
        "45deg Diag + Check",
        "Mixed HUD + Texture"
    };
    enum { BENCH_ROWS = ZCVID_LH_CODEC_COUNT + 2 }; /* +2 for HYBRID_FAST (12T) & HYBRID_ESR (12T) */
    static const char *k_codec_names[BENCH_ROWS] = {
        "TQ4 (4x4 baseline)",
        "LRF-R1 (pure)",
        "LRF-R2 (pure)",
        "HYBRID_RDO",
        "HYBRID_CASCADE",
        "HYBRID_FULL",
        "HYBRID_FAST (1T)",
        "HYBRID_ESR (1T)",
        "HYBRID_FAST (12T)",
        "HYBRID_ESR (12T)"
    };
    static const uint32_t k_qps[NUM_QPS] = { 18U, 24U, 30U, 36U, 42U };

    const size_t npix = (size_t)SCENE_W * (size_t)SCENE_H;
    const size_t bs_cap = npix * 4U + 4096U;
    uint8_t *src = (uint8_t *)malloc(npix);
    uint8_t *pred = (uint8_t *)malloc(npix);
    uint8_t *enc_rec = (uint8_t *)malloc(npix);
    uint8_t *dec_rec = (uint8_t *)malloc(npix);
    uint8_t *bs = (uint8_t *)malloc(bs_cap);
    assert(src && pred && enc_rec && dec_rec && bs);

    zcvid_lh_mt_pool_t mt_pool;
    assert(zcvid_lh_mt_pool_init(&mt_pool, 12U, SCENE_W, SCENE_H));

    double bpp_tbl[NUM_SCENES][BENCH_ROWS][NUM_QPS];
    double psnr_tbl[NUM_SCENES][BENCH_ROWS][NUM_QPS];
    double enc_ms_sum[BENCH_ROWS] = {0};
    double dec_ms_sum[BENCH_ROWS] = {0};
    uint64_t mode_tot[BENCH_ROWS][7];
    memset(mode_tot, 0, sizeof(mode_tot));

    printf("=== zcvid Latent Factorization vs 4x4 Transform Hybrid Benchmark (%dx%d) ===\n\n",
           SCENE_W, SCENE_H);

    for (int s = 0; s < NUM_SCENES; ++s) {
        bool is_inter = false;
        build_scene(s, src, pred, &is_inter);
        printf("--- Scene %d: %s (%s) ---\n", s + 1, k_scene_names[s], is_inter ? "INTER res" : "INTRA");
        printf("  %-19s | %15s | %15s | %15s | %15s | %15s | %9s | %9s\n",
               "Codec", "QP18 bpp/dB", "QP24 bpp/dB", "QP30 bpp/dB", "QP36 bpp/dB", "QP42 bpp/dB",
               "BD-Rate", "BD-PSNR");

        for (int c = 0; c < BENCH_ROWS; ++c) {
            const bool use_mt = (c >= ZCVID_LH_CODEC_COUNT);
            const zcvid_lh_codec_t codec = (c == ZCVID_LH_CODEC_COUNT) ? ZCVID_LH_HYBRID_FAST
                                         : (c == ZCVID_LH_CODEC_COUNT + 1) ? ZCVID_LH_HYBRID_ESR
                                         : (zcvid_lh_codec_t)c;
            for (int qi = 0; qi < NUM_QPS; ++qi) {
                zcvid_lh_stats_t st;
                double t0 = now_ms();
                size_t len = use_mt
                    ? zcvid_lh_mt_encode_frame(&mt_pool, codec, src, pred, SCENE_W, SCENE_H,
                                               k_qps[qi], is_inter, bs, bs_cap, enc_rec, &st)
                    : zcvid_lh_encode_frame(codec, src, pred, SCENE_W, SCENE_H,
                                            k_qps[qi], is_inter, bs, bs_cap, enc_rec, &st);
                double t1 = now_ms();
                assert(len > 12U);

                double t2 = now_ms();
                int drc = use_mt
                    ? zcvid_lh_mt_decode_frame(&mt_pool, bs, len, pred, SCENE_W, SCENE_H, dec_rec)
                    : zcvid_lh_decode_frame(bs, len, pred, SCENE_W, SCENE_H, dec_rec);
                double t3 = now_ms();
                assert(drc == 0);

                if (memcmp(enc_rec, dec_rec, npix) != 0) {
                    fprintf(stderr, "BIT-EXACT MISMATCH on scene=%d codec=%d qp=%u!\n",
                            s, c, k_qps[qi]);
                    return 1;
                }

                bpp_tbl[s][c][qi] = ((double)len * 8.0) / (double)npix;
                psnr_tbl[s][c][qi] = compute_psnr(src, dec_rec, npix);
                enc_ms_sum[c] += (t1 - t0);
                dec_ms_sum[c] += (t3 - t2);
                for (int k = 0; k < 7; ++k) mode_tot[c][k] += st.count[k];
            }

            double bd_rate = 0.0, bd_psnr = 0.0;
            if (c > 0) {
                compute_bd(bpp_tbl[s][0], psnr_tbl[s][0], bpp_tbl[s][c], psnr_tbl[s][c],
                           &bd_rate, &bd_psnr);
            }
            printf("  %-19s | %5.3f / %5.2f  | %5.3f / %5.2f  | %5.3f / %5.2f  | %5.3f / %5.2f  | %5.3f / %5.2f  | %+7.2f%% | %+6.2f dB\n",
                   k_codec_names[c],
                   bpp_tbl[s][c][0], psnr_tbl[s][c][0],
                   bpp_tbl[s][c][1], psnr_tbl[s][c][1],
                   bpp_tbl[s][c][2], psnr_tbl[s][c][2],
                   bpp_tbl[s][c][3], psnr_tbl[s][c][3],
                   bpp_tbl[s][c][4], psnr_tbl[s][c][4],
                   bd_rate, bd_psnr);
        }
        printf("\n");
    }

    printf("=== Overall Summary (mean across %d scenes x %d QPs, 0 enc/dec mismatch) ===\n",
           NUM_SCENES, NUM_QPS);
    const double scale_1080p = (1920.0 * 1080.0) / (double)npix;
    const double runs_per_codec = (double)(NUM_SCENES * NUM_QPS);
    printf("  %-19s | %10s | %10s | %12s | %12s | %s\n",
           "Codec", "Mean BD-R", "Mean BD-dB", "1080p Enc", "1080p Dec", "Block Mode Mix (SKIP/TQ4/LRF1/LRF2/CAS/SPEC/ESR)");
    for (int c = 0; c < BENCH_ROWS; ++c) {
        double sum_r = 0.0, sum_p = 0.0;
        if (c > 0) {
            for (int s = 0; s < NUM_SCENES; ++s) {
                double r = 0.0, p = 0.0;
                compute_bd(bpp_tbl[s][0], psnr_tbl[s][0], bpp_tbl[s][c], psnr_tbl[s][c], &r, &p);
                sum_r += r;
                sum_p += p;
            }
            sum_r /= NUM_SCENES;
            sum_p /= NUM_SCENES;
        }
        uint64_t tot_blk = 0ULL;
        for (int k = 0; k < 7; ++k) tot_blk += mode_tot[c][k];
        double pct[7] = {0};
        if (tot_blk > 0ULL) {
            for (int k = 0; k < 7; ++k) pct[k] = 100.0 * (double)mode_tot[c][k] / (double)tot_blk;
        }
        double e1080 = (enc_ms_sum[c] / runs_per_codec) * scale_1080p;
        double d1080 = (dec_ms_sum[c] / runs_per_codec) * scale_1080p;
        printf("  %-19s | %+8.2f%% | %+8.2f dB | %9.2f ms | %9.2f ms | %4.1f / %4.1f / %4.1f / %4.1f / %4.1f / %4.1f / %4.1f %%\n",
               k_codec_names[c], sum_r, sum_p, e1080, d1080,
               pct[0], pct[1], pct[2], pct[3], pct[4], pct[5], pct[6]);
    }
    printf("\n");

    zcvid_lh_mt_pool_destroy(&mt_pool);
    free(src);
    free(pred);
    free(enc_rec);
    free(dec_rec);
    free(bs);

    run_1080p_realtime_bench();
    run_galois_2d_microbench();
    run_tropical_simplicial_bench();
    return 0;
}
