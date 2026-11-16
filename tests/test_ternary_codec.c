/**
 * @file test_ternary_codec.c
 * @brief Unit tests & 1080p120 latency/bandwidth benchmark for ternary_codec.h
 *        (NV12 & RGBA, Jayant ADM, RLE zero-skip, entropy coding, AVX2/Scalar).
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>

#include "ternary_codec.h"
#include "ternary_mt.h"
#include "ternary_zcrudp_bridge.h"

static inline uint64_t read_ns(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

typedef struct {
    uint8_t *pkts;
    size_t  *lens;
    size_t   count;
    size_t   max_pkts;
    size_t   mtu;
    size_t   total_bytes;
} pkt_sink_t;

static void sink_cb(const uint8_t *data, size_t len, void *ctx) {
    pkt_sink_t *s = (pkt_sink_t *)ctx;
    if (s->count < s->max_pkts && len <= s->mtu) {
        memcpy(s->pkts + s->count * s->mtu, data, len);
        s->lens[s->count] = len;
        s->count++;
    }
    s->total_bytes += len;
}

static void test_nv12_and_rgba_roundtrip(void) {
    printf("[TEST] Running NV12 & RGBA Jayant ADM Roundtrip Test... ");

    const zcrudp_ternary_pixfmt_t fmts[2] = {
        ZCRUDP_TERNARY_FMT_NV12,
        ZCRUDP_TERNARY_FMT_RGBA
    };

    for (int f_idx = 0; f_idx < 2; f_idx++) {
        zcrudp_ternary_config_t cfg = {
            .width = 160,
            .height = 90,
            .pixfmt = fmts[f_idx],
            .delta_min = 4,
            .delta_max = 32,
            .threshold = 3,
            .gamma_q8 = 248,
            .neutral_y = 128,
            .mtu_payload = 1380
        };

        size_t psz = zcrudp_ternary_planar_size(&cfg);
        uint8_t *src = (uint8_t *)malloc(psz);
        uint8_t *enc_ref = (uint8_t *)malloc(psz);
        uint8_t *enc_delta = (uint8_t *)malloc(psz);
        uint8_t *dec_canvas = (uint8_t *)malloc(psz);
        uint8_t *dec_delta = (uint8_t *)malloc(psz);
        uint8_t pkt_buf[1400];

        assert(src && enc_ref && enc_delta && dec_canvas && dec_delta);
        memset(enc_ref, cfg.neutral_y, psz);
        memset(dec_canvas, cfg.neutral_y, psz);
        memset(enc_delta, cfg.delta_min, psz);
        memset(dec_delta, cfg.delta_min, psz);

        uint8_t *pkt_store = (uint8_t *)malloc(256 * 1400);
        size_t pkt_lens[256];
        pkt_sink_t sink = {
            .pkts = pkt_store,
            .lens = pkt_lens,
            .count = 0,
            .max_pkts = 256,
            .mtu = 1400,
            .total_bytes = 0
        };

        size_t y_sz = (size_t)cfg.width * (size_t)cfg.height;

        for (uint16_t fid = 0; fid < 20; fid++) {
            for (size_t i = 0; i < psz; i++) {
                src[i] = (uint8_t)((i * 7U + fid * 13U) & 0xFFU);
            }
            sink.count = 0;
            sink.total_bytes = 0;

            zcrudp_ternary_encode_frame(src, enc_ref, enc_delta, fid, &cfg,
                                        pkt_buf, sink_cb, &sink);

            /* Decoder: decay all planes once, then apply received packets */
            if (cfg.pixfmt == ZCRUDP_TERNARY_FMT_NV12) {
                zcrudp_ternary_decay_plane(dec_canvas, y_sz, cfg.gamma_q8, cfg.neutral_y);
                zcrudp_ternary_decay_plane(dec_canvas + y_sz, y_sz / 2U, cfg.gamma_q8, cfg.neutral_y);
            } else {
                for (size_t p = 0; p < 3U; p++) {
                    zcrudp_ternary_decay_plane(dec_canvas + p * y_sz, y_sz, cfg.gamma_q8, cfg.neutral_y);
                }
            }

            for (size_t p = 0; p < sink.count; p++) {
                const uint8_t *pd = sink.pkts + p * sink.mtu;
                zcrudp_ternary_pkt_hdr_t hdr;
                bool ok = zcrudp_ternary_pkt_hdr_read_le(pd, &hdr);
                assert(ok);
                size_t plane_off = (hdr.plane == 0) ? 0 : (hdr.plane * y_sz);
                size_t plane_len = (cfg.pixfmt == ZCRUDP_TERNARY_FMT_NV12 && hdr.plane == 1)
                    ? (y_sz / 2U) : y_sz;
                ok = zcrudp_ternary_decode_packet(
                    pd, sink.lens[p],
                    dec_canvas + plane_off,
                    dec_delta + plane_off,
                    plane_len, &cfg, NULL);
                assert(ok);
            }

            /* Bit-exact match between encoder closed-loop reference & decoder canvas */
            assert(memcmp(enc_ref, dec_canvas, psz) == 0);
        }

        free(pkt_store);
        free(src);
        free(enc_ref);
        free(enc_delta);
        free(dec_canvas);
        free(dec_delta);
    }

    printf("PASSED (NV12 & RGBA bit-exact)\n");
}

static void bench_1080p120_format(zcrudp_ternary_pixfmt_t fmt, const char *name, uint8_t entropy) {
    zcrudp_ternary_config_t cfg = {
        .width = 1920,
        .height = 1080,
        .pixfmt = fmt,
        .delta_min = 4,
        .delta_max = 32,
        .threshold = 6,
        .gamma_q8 = 250,
        .neutral_y = 128,
        .mtu_payload = 1380,
        .entropy = entropy
    };

    size_t psz = zcrudp_ternary_planar_size(&cfg);
    size_t y_sz = (size_t)cfg.width * (size_t)cfg.height;
    uint8_t *src = (uint8_t *)malloc(psz);
    uint8_t *enc_ref = (uint8_t *)malloc(psz);
    uint8_t *dec_canvas = (uint8_t *)malloc(psz);
    uint8_t *pkt_store = (uint8_t *)malloc(1024 * 1400);
    size_t pkt_lens[1024];
    uint8_t pkt_buf[1400];

    memset(src, 128, psz);
    memset(enc_ref, 128, psz);
    memset(dec_canvas, 128, psz);

    /* Simulate ~5% active region (moving 420x250 sprite on 1080p) */
    for (size_t y = 200; y < 450; y++) {
        for (size_t x = 400; x < 820; x++) {
            src[y * 1920U + x] = 200;
        }
    }

    pkt_sink_t sink = {
        .pkts = pkt_store,
        .lens = pkt_lens,
        .count = 0,
        .max_pkts = 1024,
        .mtu = 1400,
        .total_bytes = 0
    };

    const int ITERS = 30;
    uint64_t enc_ns_total = 0;
    uint64_t dec_ns_total = 0;
    size_t last_bytes = 0;
    size_t last_pkts = 0;

    for (int it = 0; it < ITERS; it++) {
        /* Shift sprite by 2 pixels each frame */
        memset(src + 200 * 1920U, 128, 250 * 1920U);
        size_t sx = 400 + (size_t)(it * 2);
        for (size_t y = 200; y < 450; y++) {
            memset(src + y * 1920U + sx, 200, 420);
        }

        sink.count = 0;
        sink.total_bytes = 0;

        uint64_t t0 = read_ns();
        zcrudp_ternary_encode_frame(src, enc_ref, NULL, (uint16_t)it, &cfg,
                                    pkt_buf, sink_cb, &sink);
        uint64_t t1 = read_ns();

        uint64_t t2 = read_ns();
        if (fmt == ZCRUDP_TERNARY_FMT_NV12) {
            zcrudp_ternary_decay_plane(dec_canvas, y_sz, cfg.gamma_q8, cfg.neutral_y);
            zcrudp_ternary_decay_plane(dec_canvas + y_sz, y_sz / 2U, cfg.gamma_q8, cfg.neutral_y);
        } else {
            for (size_t p = 0; p < 3U; p++) {
                zcrudp_ternary_decay_plane(dec_canvas + p * y_sz, y_sz, cfg.gamma_q8, cfg.neutral_y);
            }
        }
        for (size_t p = 0; p < sink.count; p++) {
            const uint8_t *pd = sink.pkts + p * sink.mtu;
            zcrudp_ternary_pkt_hdr_t hdr;
            zcrudp_ternary_pkt_hdr_read_le(pd, &hdr);
            size_t plane_off = (hdr.plane == 0) ? 0 : (hdr.plane * y_sz);
            size_t plane_len = (fmt == ZCRUDP_TERNARY_FMT_NV12 && hdr.plane == 1)
                ? (y_sz / 2U) : y_sz;
            zcrudp_ternary_decode_packet(pd, sink.lens[p],
                                         dec_canvas + plane_off, NULL,
                                         plane_len, &cfg, NULL);
        }
        uint64_t t3 = read_ns();

        if (it >= 5) { /* skip warmup */
            enc_ns_total += (t1 - t0);
            dec_ns_total += (t3 - t2);
            last_bytes = sink.total_bytes;
            last_pkts = sink.count;
        }
    }

    double enc_ms = (double)enc_ns_total / ((ITERS - 5) * 1e6);
    double dec_ms = (double)dec_ns_total / ((ITERS - 5) * 1e6);
    double mbps_120 = ((double)last_bytes * 8.0 * 120.0) / 1e6;

    printf("  -> %-5s 1080p120 (~5%% active, %s): Enc %.3f ms | Dec %.3f ms | %zu pkts/f | %.2f Mbps\n",
           name, entropy ? "entropy" : "tokens ", enc_ms, dec_ms, last_pkts, mbps_120);

    free(pkt_store);
    free(src);
    free(enc_ref);
    free(dec_canvas);
}

static void bench_1080p120_mt(zcrudp_ternary_pixfmt_t fmt, const char *name, uint32_t threads, uint8_t entropy) {
    zcrudp_ternary_config_t cfg = {
        .width = 1920,
        .height = 1080,
        .pixfmt = fmt,
        .delta_min = 4,
        .delta_max = 32,
        .threshold = 6,
        .gamma_q8 = 250,
        .neutral_y = 128,
        .mtu_payload = 1380,
        .entropy = entropy
    };

    size_t psz = zcrudp_ternary_planar_size(&cfg);
    size_t y_sz = (size_t)cfg.width * (size_t)cfg.height;
    uint8_t *src = (uint8_t *)malloc(psz);
    uint8_t *enc_ref = (uint8_t *)malloc(psz);
    uint8_t *dec_canvas = (uint8_t *)malloc(psz);

    const size_t MAX_PKTS_PER_T = 256;
    uint8_t *slice_pkts = (uint8_t *)malloc(threads * MAX_PKTS_PER_T * 1400);
    size_t  *slice_lens = (size_t *)malloc(threads * MAX_PKTS_PER_T * sizeof(size_t));
    uint8_t *flat_pkts  = (uint8_t *)malloc(threads * MAX_PKTS_PER_T * 1400);
    size_t  *flat_lens  = (size_t *)malloc(threads * MAX_PKTS_PER_T * sizeof(size_t));

    memset(src, 128, psz);
    memset(enc_ref, 128, psz);
    memset(dec_canvas, 128, psz);

    zcrudp_ternary_mt_pool_t pool;
    zcrudp_ternary_mt_init(&pool, threads, slice_pkts, slice_lens, MAX_PKTS_PER_T, 1400);

    const int ITERS = 40;
    uint64_t enc_ns_total = 0;
    uint64_t dec_ns_total = 0;
    size_t last_bytes = 0;
    size_t last_pkts = 0;

    for (int it = 0; it < ITERS; it++) {
        memset(src + 200 * 1920U, 128, 250 * 1920U);
        size_t sx = 400 + (size_t)(it * 2);
        for (size_t y = 200; y < 450; y++) {
            memset(src + y * 1920U + sx, 200, 420);
        }

        size_t total_bytes = 0;
        size_t total_pkts = 0;

        uint64_t t0 = read_ns();
        if (fmt == ZCRUDP_TERNARY_FMT_NV12) {
            size_t pk0 = 0, pk1 = 0;
            total_bytes += zcrudp_ternary_mt_encode_plane(
                &pool, src, enc_ref, NULL, y_sz, 0, (uint16_t)it, &cfg, &pk0);
            /* Gather packets for decode simulation */
            for (uint32_t t = 0; t < pool.num_threads; t++) {
                size_t c = pool.workers[t].sink.count;
                if (c > 0) {
                    memcpy(flat_pkts + total_pkts * 1400, pool.workers[t].sink.pkts_base, c * 1400);
                    memcpy(flat_lens + total_pkts, pool.workers[t].sink.lens_base, c * sizeof(size_t));
                    total_pkts += c;
                }
            }
            total_bytes += zcrudp_ternary_mt_encode_plane(
                &pool, src + y_sz, enc_ref + y_sz, NULL, y_sz / 2U, 1, (uint16_t)it, &cfg, &pk1);
            for (uint32_t t = 0; t < pool.num_threads; t++) {
                size_t c = pool.workers[t].sink.count;
                if (c > 0) {
                    memcpy(flat_pkts + total_pkts * 1400, pool.workers[t].sink.pkts_base, c * 1400);
                    memcpy(flat_lens + total_pkts, pool.workers[t].sink.lens_base, c * sizeof(size_t));
                    total_pkts += c;
                }
            }
        } else {
            for (uint8_t pl = 0; pl < 3U; pl++) {
                size_t pk = 0;
                total_bytes += zcrudp_ternary_mt_encode_plane(
                    &pool, src + pl * y_sz, enc_ref + pl * y_sz, NULL,
                    y_sz, pl, (uint16_t)it, &cfg, &pk);
                for (uint32_t t = 0; t < pool.num_threads; t++) {
                    size_t c = pool.workers[t].sink.count;
                    if (c > 0) {
                        memcpy(flat_pkts + total_pkts * 1400, pool.workers[t].sink.pkts_base, c * 1400);
                        memcpy(flat_lens + total_pkts, pool.workers[t].sink.lens_base, c * sizeof(size_t));
                        total_pkts += c;
                    }
                }
            }
        }
        uint64_t t1 = read_ns();

        uint64_t t2 = read_ns();
        if (fmt == ZCRUDP_TERNARY_FMT_NV12) {
            zcrudp_ternary_mt_decay_plane(&pool, dec_canvas, y_sz, &cfg);
            zcrudp_ternary_mt_decay_plane(&pool, dec_canvas + y_sz, y_sz / 2U, &cfg);
        } else {
            for (uint8_t pl = 0; pl < 3U; pl++) {
                zcrudp_ternary_mt_decay_plane(&pool, dec_canvas + pl * y_sz, y_sz, &cfg);
            }
        }
        zcrudp_ternary_mt_decode_packets(&pool, dec_canvas, NULL, y_sz, &cfg,
                                         flat_pkts, flat_lens, 1400, total_pkts);
        uint64_t t3 = read_ns();

        /* Verify bit-exactness on last frame */
        if (it == ITERS - 1) {
            assert(memcmp(enc_ref, dec_canvas, psz) == 0);
        }

        if (it >= 10) {
            enc_ns_total += (t1 - t0);
            dec_ns_total += (t3 - t2);
            last_bytes = total_bytes;
            last_pkts = total_pkts;
        }
    }

    zcrudp_ternary_mt_destroy(&pool);

    double enc_ms = (double)enc_ns_total / ((ITERS - 10) * 1e6);
    double dec_ms = (double)dec_ns_total / ((ITERS - 10) * 1e6);
    double total_ms = enc_ms + dec_ms;
    double mbps_120 = ((double)last_bytes * 8.0 * 120.0) / 1e6;
    double mbps_360 = ((double)last_bytes * 8.0 * 360.0) / 1e6;
    double budget_360_pct = (total_ms / (1000.0 / 360.0)) * 100.0;

    const char *mode_str = (entropy == ZCRUDP_TERNARY_EC_RANS) ? "rans   "
                         : (entropy == ZCRUDP_TERNARY_EC_REALLOC_V2) ? "rv2    "
                         : (entropy != 0U) ? "entropy" : "tokens ";
    printf("  -> %-5s 1080p (%uT MT, %s) : Enc %.3f ms | Dec %.3f ms | Total %.3f ms | %zu pkts/f | %.2f Mbps @120Hz | %.2f Mbps @360Hz (%.1f%% of 2.78ms budget)\n",
           name, threads, mode_str, enc_ms, dec_ms, total_ms, last_pkts, mbps_120, mbps_360, budget_360_pct);

    free(slice_pkts);
    free(slice_lens);
    free(flat_pkts);
    free(flat_lens);
    free(src);
    free(enc_ref);
    free(dec_canvas);
}

static void test_gmc_panoramic_savings(void) {
    printf("[TEST] Running Global Motion Compensation (GMC) Panoramic Test... ");

    const uint16_t W = 640, H = 360;
    size_t sz = (size_t)W * H;
    uint8_t *prev = (uint8_t *)malloc(sz);
    uint8_t *curr = (uint8_t *)malloc(sz);
    assert(prev && curr);

    /* Smooth non-periodic 2D textured background */
    for (uint16_t y = 0; y < H; y++) {
        for (uint16_t x = 0; x < W; x++) {
            uint32_t v = ((uint32_t)x * 3U + (uint32_t)y * 5U +
                          ((uint32_t)x * x + (uint32_t)y * y) / 64U) & 0xFFU;
            prev[y * W + x] = (uint8_t)v;
        }
    }
    /* Pan camera by (+5, -3) */
    memcpy(curr, prev, sz);
    zcrudp_ternary_shift_plane(curr, W, H, 5, -3, 128);

    int8_t dx = 0, dy = 0;
    uint64_t t0 = read_ns();
    zcrudp_ternary_estimate_gmc(curr, prev, W, H, 8, &dx, &dy);
    uint64_t t1 = read_ns();

    assert(dx == 5 && dy == -3);
    zcrudp_ternary_shift_plane(prev, W, H, dx, dy, 128);
    assert(memcmp(prev, curr, sz) == 0);

    printf("PASSED (detected dx=%d, dy=%d in %.3f ms, 0 residual error)\n",
           dx, dy, (double)(t1 - t0) / 1e6);

    free(prev);
    free(curr);
}

static void test_intra_refresh_and_2x_mode(void) {
    printf("[TEST] Running Rolling Intra-Refresh (HUD Crispness) & 2x Mode Test... ");

    zcrudp_ternary_config_t cfg = {
        .width = 320,
        .height = 180,
        .pixfmt = ZCRUDP_TERNARY_FMT_GRAY8,
        .delta_min = 3,
        .delta_max = 32,
        .threshold = 3,
        .gamma_q8 = 254, /* gentle leak + rolling intra-refresh */
        .neutral_y = 128,
        .mtu_payload = 1380
    };

    size_t len = (size_t)cfg.width * cfg.height;
    uint8_t *src = (uint8_t *)malloc(len);
    uint8_t *enc_ref = (uint8_t *)malloc(len);
    uint8_t *enc_delta = (uint8_t *)malloc(len);
    uint8_t *dec_canvas = (uint8_t *)malloc(len);
    uint8_t *dec_delta = (uint8_t *)malloc(len);
    uint8_t *pkt_store = (uint8_t *)malloc(256 * 1400);
    size_t pkt_lens[256];
    uint8_t pkt_buf[1400];

    memset(enc_ref, cfg.neutral_y, len);
    memset(dec_canvas, cfg.neutral_y, len);
    memset(enc_delta, cfg.delta_min, len);
    memset(dec_delta, cfg.delta_min, len);

    /* Sharp HUD pattern: black background (0) + white text/crosshair (255) */
    memset(src, 0, len);
    for (size_t y = 80; y < 100; y++) {
        memset(src + y * cfg.width + 120, 255, 80);
    }

    pkt_sink_t sink = {
        .pkts = pkt_store, .lens = pkt_lens, .count = 0,
        .max_pkts = 256, .mtu = 1400, .total_bytes = 0
    };

    /* Rolling Intra-Refresh over 12 frames (1/12th of frame per tick) */
    const size_t N_CYCLE = 12;
    size_t slice_bytes = (len + N_CYCLE - 1U) / N_CYCLE;

    for (uint16_t f = 0; f < 12; f++) {
        sink.count = 0;
        sink.total_bytes = 0;

        /* Encode delta first, then overwrite rolling Intra-Refresh slice at end of frame */
        if (f & 1U) {
            zcrudp_ternary_encode_plane_2x(src, enc_ref, enc_delta, len,
                                           0, f, &cfg, pkt_buf, sink_cb, &sink);
        } else {
            zcrudp_ternary_encode_plane(src, enc_ref, enc_delta, len,
                                        0, f, &cfg, pkt_buf, sink_cb, &sink);
        }
        size_t i0 = (size_t)f * slice_bytes;
        size_t i1 = i0 + slice_bytes;
        zcrudp_ternary_encode_intra_range(src, enc_ref, enc_delta, i0, i1, len,
                                          0, f, &cfg, pkt_buf, sink_cb, &sink);

        zcrudp_ternary_decay_plane(dec_canvas, len, cfg.gamma_q8, cfg.neutral_y);
        for (size_t p = 0; p < sink.count; p++) {
            bool ok = zcrudp_ternary_decode_packet(
                sink.pkts + p * sink.mtu, sink.lens[p],
                dec_canvas, dec_delta, len, &cfg, NULL);
            assert(ok);
        }
        assert(memcmp(enc_ref, dec_canvas, len) == 0);
    }

    /* After 1 full Intra-Refresh cycle, sharp HUD white & black must be crisp */
    assert(dec_canvas[90 * cfg.width + 150] >= 250);
    assert(dec_canvas[10 * cfg.width + 10] <= 5);

    free(pkt_store);
    free(src);
    free(enc_ref);
    free(enc_delta);
    free(dec_canvas);
    free(dec_delta);
    printf("PASSED (HUD crisp, 0 wash-out)\n");
}

static void test_3d_complex_scene_psnr(uint8_t entropy) {
    printf("[TEST] Running Complex 3D-like Textured + Pan + HUD PSNR Test (%s)... ",
           entropy ? "entropy" : "tokens");

    zcrudp_ternary_config_t cfg = {
        .width = 640,
        .height = 360,
        .pixfmt = ZCRUDP_TERNARY_FMT_GRAY8,
        .delta_min = 3,
        .delta_max = 32,
        .threshold = 4,
        .gamma_q8 = 254,
        .neutral_y = 128,
        .mtu_payload = 1380,
        .entropy = entropy
    };

    size_t len = (size_t)cfg.width * cfg.height;
    uint8_t *src = (uint8_t *)malloc(len);
    uint8_t *enc_ref = (uint8_t *)malloc(len);
    uint8_t *enc_delta = (uint8_t *)malloc(len);
    uint8_t *dec_canvas = (uint8_t *)malloc(len);
    uint8_t *dec_delta = (uint8_t *)malloc(len);
    uint8_t *pkt_store = (uint8_t *)malloc(512 * 1400);
    size_t pkt_lens[512];
    uint8_t pkt_buf[1400];

    memset(enc_ref, cfg.neutral_y, len);
    memset(dec_canvas, cfg.neutral_y, len);
    memset(enc_delta, cfg.delta_min, len);
    memset(dec_delta, cfg.delta_min, len);

    pkt_sink_t sink = {
        .pkts = pkt_store, .lens = pkt_lens, .count = 0,
        .max_pkts = 512, .mtu = 1400, .total_bytes = 0
    };

    double sum_psnr = 0.0;
    size_t sum_bytes = 0;
    const int FRAMES = 24;
    const size_t N_CYCLE = 24;
    size_t slice_bytes = (len + N_CYCLE - 1U) / N_CYCLE;

    for (int f = 0; f < FRAMES; f++) {
        /* Complex textured 3D-like scene (continuous triangle-wave lighting) + static HUD */
        for (uint16_t y = 0; y < cfg.height; y++) {
            for (uint16_t x = 0; x < cfg.width; x++) {
                uint32_t wx = (uint32_t)x + (uint32_t)(f * 3);
                uint32_t wy = (uint32_t)y + (uint32_t)f;
                uint32_t raw = (wx * 3U + wy * 5U) % 510U;
                uint32_t tri = (raw > 255U) ? (510U - raw) : raw;
                src[y * cfg.width + x] = (uint8_t)tri;
            }
        }
        /* Static sharp HUD bar */
        for (uint16_t y = 16; y < 36; y++) {
            memset(src + y * cfg.width + 20, 240, 140);
        }

        sink.count = 0;
        sink.total_bytes = 0;

        if (f == 0) {
            /* Initial Intra keyframe */
            zcrudp_ternary_encode_intra_range(src, enc_ref, enc_delta, 0, len,
                                              len, 0, (uint16_t)f, &cfg, pkt_buf, sink_cb, &sink);
        } else {
            /* GMC: estimate & shift both refs */
            int8_t dx = 0, dy = 0;
            zcrudp_ternary_estimate_gmc(src, enc_ref, cfg.width, cfg.height, 8, &dx, &dy);
            zcrudp_ternary_shift_plane(enc_ref, cfg.width, cfg.height, dx, dy, cfg.neutral_y);
            zcrudp_ternary_shift_plane(dec_canvas, cfg.width, cfg.height, dx, dy, cfg.neutral_y);

            /* 1x Delta, then anchors where the deltas could not follow (borders
             * uncovered by the shift, HUD dragged by it), then the rolling slice */
            size_t i0 = (size_t)f * slice_bytes;
            zcrudp_ternary_encode_plane(src, enc_ref, enc_delta, len,
                                        0, (uint16_t)f, &cfg, pkt_buf, sink_cb, &sink);
            zcrudp_ternary_repair_disocclusions(src, enc_ref, enc_delta, len, 20U,
                                                0, (uint16_t)f, &cfg, pkt_buf, sink_cb, &sink);
            zcrudp_ternary_encode_intra_range(src, enc_ref, enc_delta, i0, i0 + slice_bytes,
                                              len, 0, (uint16_t)f, &cfg, pkt_buf, sink_cb, &sink);
        }

        zcrudp_ternary_decay_plane(dec_canvas, len, cfg.gamma_q8, cfg.neutral_y);
        assert(sink.count < sink.max_pkts);
        for (size_t p = 0; p < sink.count; p++) {
            zcrudp_ternary_decode_packet(sink.pkts + p * sink.mtu, sink.lens[p],
                                         dec_canvas, dec_delta, len, &cfg, NULL);
        }
        assert(memcmp(enc_ref, dec_canvas, len) == 0);

        double mse = 0.0;
        for (size_t i = 0; i < len; i++) {
            double d = (double)src[i] - (double)dec_canvas[i];
            mse += d * d;
        }
        mse /= (double)len;
        double psnr = (mse > 1e-6) ? (10.0 * log10((255.0 * 255.0) / mse)) : 99.0;
        if (f >= 4) {
            sum_psnr += psnr;
            sum_bytes += sink.total_bytes;
        }
    }

    double avg_psnr = sum_psnr / (FRAMES - 4);
    double avg_bpp = ((double)sum_bytes / (FRAMES - 4) * 8.0) / (double)len;
    printf("(avg PSNR = %.2f dB, avg %.2f bpp) ", avg_psnr, avg_bpp);
    fflush(stdout);
    assert(avg_psnr > 35.0);

    printf("PASSED\n");

    free(pkt_store);
    free(src);
    free(enc_ref);
    free(enc_delta);
    free(dec_canvas);
    free(dec_delta);
}

static void test_tile_mv_3d_parallax_35db(uint8_t entropy) {
    printf("[TEST] Running 32x32 Tile MV + Luma-First RC + Bridge (35+ dB target, %s)... ",
           entropy ? "entropy" : "tokens");

    zcrudp_ternary_config_t cfg = {
        .width = 640,
        .height = 360,
        .pixfmt = ZCRUDP_TERNARY_FMT_NV12,
        .delta_min = 4,
        .delta_max = 48,
        .threshold = 2,
        .gamma_q8 = 255,
        .neutral_y = 128,
        .mtu_payload = 1380,
        .entropy = entropy
    };

    size_t psz = zcrudp_ternary_planar_size(&cfg);
    size_t y_sz = (size_t)cfg.width * cfg.height;
    size_t uv_sz = y_sz / 2U;
    const uint16_t TILE = 32;
    size_t tx_cnt = (cfg.width + TILE - 1U) / TILE;
    size_t ty_cnt = (cfg.height + TILE - 1U) / TILE;
    size_t num_tiles = tx_cnt * ty_cnt;

    uint8_t *src = (uint8_t *)malloc(psz);
    uint8_t *enc_ref = (uint8_t *)malloc(psz);
    uint8_t *enc_mc = (uint8_t *)malloc(psz);
    uint8_t *enc_delta = (uint8_t *)malloc(psz);
    uint8_t *dec_canvas = (uint8_t *)malloc(psz);
    uint8_t *dec_mc = (uint8_t *)malloc(psz);
    uint8_t *dec_delta = (uint8_t *)malloc(psz);
    int8_t  *mvs = (int8_t *)malloc(num_tiles * 2U);
    uint8_t *pkt_store = (uint8_t *)malloc(512 * 1400);
    size_t pkt_lens[512];
    uint8_t pkt_buf[1400];

    memset(enc_ref, cfg.neutral_y, psz);
    memset(dec_canvas, cfg.neutral_y, psz);
    memset(enc_delta, cfg.delta_min, psz);
    memset(dec_delta, cfg.delta_min, psz);

    pkt_sink_t sink = {
        .pkts = pkt_store, .lens = pkt_lens, .count = 0,
        .max_pkts = 512, .mtu = 1400, .total_bytes = 0
    };
    zcrudp_ternary_bridge_router_t router = {
        .tx_fn = NULL, .bridge_ctx = NULL,
        .reliable_bytes = 0, .unreliable_bytes = 0,
        .reliable_pkts = 0, .unreliable_pkts = 0
    };

    double sum_psnr_y = 0.0;
    size_t sum_bytes = 0;
    const int FRAMES = 20;

    for (int f = 0; f < FRAMES; f++) {
        /* 1. Panning textured 3D background (+3, +1) */
        for (uint16_t y = 0; y < cfg.height; y++) {
            for (uint16_t x = 0; x < cfg.width; x++) {
                uint32_t wx = (uint32_t)x + (uint32_t)(f * 3);
                uint32_t wy = (uint32_t)y + (uint32_t)f;
                uint32_t raw = (wx * 3U + wy * 5U) % 510U;
                src[y * cfg.width + x] = (uint8_t)((raw > 255U) ? (510U - raw) : raw);
            }
        }
        /* 2. Independent foreground object moving (-4, +2) */
        int fx0 = 320 - f * 4;
        int fy0 = 140 + f * 2;
        for (int y = fy0; y < fy0 + 64; y++) {
            if (y >= 0 && y < cfg.height) {
                for (int x = fx0; x < fx0 + 64; x++) {
                    if (x >= 0 && x < cfg.width) {
                        src[(size_t)y * cfg.width + (size_t)x] = (uint8_t)(200 - ((x - fx0) + (y - fy0)) / 2);
                    }
                }
            }
        }
        /* 3. Static sharp HUD overlay (0, 0) */
        for (uint16_t y = 8; y < 40; y++) {
            memset(src + y * cfg.width + 16, 250, 120);
        }
        /* Smooth UV chroma */
        memset(src + y_sz, 128, uv_sz);

        sink.count = 0;
        sink.total_bytes = 0;

        if (f == 0) {
            zcrudp_ternary_encode_intra_range(src, enc_ref, enc_delta, 0, y_sz,
                                              y_sz, 0, (uint16_t)f, &cfg, pkt_buf, sink_cb, &sink);
            zcrudp_ternary_encode_intra_range(src + y_sz, enc_ref + y_sz, enc_delta + y_sz,
                                              0, uv_sz, uv_sz, 1, (uint16_t)f, &cfg, pkt_buf, sink_cb, &sink);
        } else {
            zcrudp_ternary_estimate_tile_mv(src, enc_ref, cfg.width, cfg.height, TILE, 8, mvs);
            zcrudp_ternary_apply_tile_mv(enc_ref, enc_mc, cfg.width, cfg.height, TILE, mvs);
            memcpy(enc_ref, enc_mc, y_sz);

            zcrudp_ternary_encode_tile_mv_pkts(mvs, num_tiles, 0, (uint16_t)f, &cfg,
                                               pkt_buf, sink_cb, &sink);
            zcrudp_ternary_encode_nv12_rc(src, enc_ref, enc_delta, (uint16_t)f,
                                          25000U, 20U, &cfg, pkt_buf, sink_cb, &sink);
            zcrudp_ternary_repair_disocclusions(src, enc_ref, enc_delta, y_sz, 20U,
                                                0, (uint16_t)f, &cfg, pkt_buf, sink_cb, &sink);
        }

        /* Route & decode packets in exact encoder order: TILE_MV -> decay -> DELTA/INTRA */
        for (size_t p = 0; p < sink.count; p++) {
            const uint8_t *pd = sink.pkts + p * sink.mtu;
            zcrudp_ternary_bridge_pkt_cb(pd, sink.lens[p], &router);
            zcrudp_ternary_pkt_hdr_t hdr;
            assert(zcrudp_ternary_pkt_hdr_read_le(pd, &hdr));
            if (ZCRUDP_TERNARY_HDR_TYPE(&hdr) == ZCRUDP_TERNARY_PKT_TILE_MV) {
                const int8_t *rx_mvs = (const int8_t *)(pd + ZCRUDP_TERNARY_PKT_HDR_SZ);
                zcrudp_ternary_apply_tile_mv(dec_canvas, dec_mc, cfg.width, cfg.height, TILE, rx_mvs);
                memcpy(dec_canvas, dec_mc, y_sz);
            }
        }

        zcrudp_ternary_decay_plane(dec_canvas, y_sz, cfg.gamma_q8, cfg.neutral_y);
        zcrudp_ternary_decay_plane(dec_canvas + y_sz, uv_sz, cfg.gamma_q8, cfg.neutral_y);

        for (size_t p = 0; p < sink.count; p++) {
            const uint8_t *pd = sink.pkts + p * sink.mtu;
            zcrudp_ternary_pkt_hdr_t hdr;
            assert(zcrudp_ternary_pkt_hdr_read_le(pd, &hdr));
            if (ZCRUDP_TERNARY_HDR_TYPE(&hdr) != ZCRUDP_TERNARY_PKT_TILE_MV) {
                size_t plane_off = (hdr.plane == 0) ? 0 : y_sz;
                size_t plane_len = (hdr.plane == 0) ? y_sz : uv_sz;
                bool ok = zcrudp_ternary_decode_packet(
                    pd, sink.lens[p], dec_canvas + plane_off, dec_delta + plane_off,
                    plane_len, &cfg, NULL);
                assert(ok);
            }
        }
        assert(sink.count < sink.max_pkts);
        assert(memcmp(enc_ref, dec_canvas, psz) == 0);

        double mse = 0.0;
        for (size_t i = 0; i < y_sz; i++) {
            double d = (double)src[i] - (double)dec_canvas[i];
            mse += d * d;
        }
        mse /= (double)y_sz;
        double psnr = (mse > 1e-6) ? (10.0 * log10((255.0 * 255.0) / mse)) : 99.0;
        if (f >= 3) {
            sum_psnr_y += psnr;
            sum_bytes += sink.total_bytes;
        }
    }

    double avg_psnr = sum_psnr_y / (FRAMES - 3);
    double avg_bpp = ((double)sum_bytes / (FRAMES - 3) * 8.0) / (double)y_sz;
    assert(router.reliable_pkts > 0 && router.unreliable_pkts > 0);
    printf("(avg Y-PSNR = %.2f dB, %.2f bpp, rel=%llu / unrel=%llu pkts) ",
           avg_psnr, avg_bpp,
           (unsigned long long)router.reliable_pkts,
           (unsigned long long)router.unreliable_pkts);
    fflush(stdout);
    assert(avg_psnr >= 35.0);
    printf("PASSED\n");

    free(pkt_store);
    free(mvs);
    free(src);
    free(enc_ref);
    free(enc_mc);
    free(enc_delta);
    free(dec_canvas);
    free(dec_mc);
    free(dec_delta);
}
/*
 * NV12 chroma is interleaved (U0 V0 U1 V1 ...). DELTA_2X on that plane must
 * pair samples of the same component: pairing byte neighbours would copy U
 * into V. U and V are given very different values so such a mix-up shows.
 */
static void test_nv12_chroma_2x_keeps_components(void) {
    printf("[TEST] Running NV12 Interleaved Chroma DELTA_2X Component Test... ");

    zcrudp_ternary_config_t cfg = {
        .width = 320,
        .height = 180,
        .pixfmt = ZCRUDP_TERNARY_FMT_NV12,
        .delta_min = 4,
        .delta_max = 32,
        .threshold = 2,
        .gamma_q8 = 255,
        .neutral_y = 128,
        .mtu_payload = 1380
    };
    size_t y_sz = (size_t)cfg.width * cfg.height;
    size_t uv_sz = y_sz / 2U;

    uint8_t *src = (uint8_t *)malloc(uv_sz);
    uint8_t *enc_ref = (uint8_t *)malloc(uv_sz);
    uint8_t *enc_delta = (uint8_t *)malloc(uv_sz);
    uint8_t *dec_canvas = (uint8_t *)malloc(uv_sz);
    uint8_t *dec_delta = (uint8_t *)malloc(uv_sz);
    uint8_t *pkt_store = (uint8_t *)malloc(128 * 1400);
    size_t pkt_lens[128];
    uint8_t pkt_buf[1400];
    assert(src && enc_ref && enc_delta && dec_canvas && dec_delta && pkt_store);

    memset(enc_ref, cfg.neutral_y, uv_sz);
    memset(dec_canvas, cfg.neutral_y, uv_sz);
    memset(enc_delta, cfg.delta_min, uv_sz);
    memset(dec_delta, cfg.delta_min, uv_sz);

    /* Smooth horizontal ramps: U around 200, V around 60. */
    for (size_t i = 0; i + 1U < uv_sz; i += 2U) {
        uint8_t ramp = (uint8_t)(((i / 2U) % (cfg.width / 2U)) / 8U);
        src[i] = (uint8_t)(200U + ramp);
        src[i + 1U] = (uint8_t)(60U - ramp);
    }

    pkt_sink_t sink = {
        .pkts = pkt_store, .lens = pkt_lens, .count = 0,
        .max_pkts = 128, .mtu = 1400, .total_bytes = 0
    };

    for (int f = 0; f < 40; f++) {
        sink.count = 0;
        sink.total_bytes = 0;
        zcrudp_ternary_encode_plane_2x(src, enc_ref, enc_delta, uv_sz, 1,
                                       (uint16_t)f, &cfg, pkt_buf, sink_cb, &sink);
        assert(sink.count < sink.max_pkts);

        zcrudp_ternary_decay_plane(dec_canvas, uv_sz, cfg.gamma_q8, cfg.neutral_y);
        for (size_t p = 0; p < sink.count; p++) {
            bool ok = zcrudp_ternary_decode_packet(
                sink.pkts + p * sink.mtu, sink.lens[p], dec_canvas, dec_delta,
                uv_sz, &cfg, NULL);
            assert(ok);
        }
        assert(memcmp(enc_ref, dec_canvas, uv_sz) == 0);
        assert(memcmp(enc_delta, dec_delta, uv_sz) == 0);
    }

    double err_u = 0.0, err_v = 0.0;
    for (size_t i = 0; i + 1U < uv_sz; i += 2U) {
        err_u += fabs((double)src[i] - (double)dec_canvas[i]);
        err_v += fabs((double)src[i + 1U] - (double)dec_canvas[i + 1U]);
    }
    err_u /= (double)(uv_sz / 2U);
    err_v /= (double)(uv_sz / 2U);
    printf("(mean abs err U = %.2f, V = %.2f) ", err_u, err_v);
    fflush(stdout);
    assert(err_u < 4.0 && err_v < 4.0);
    printf("PASSED\n");

    free(pkt_store);
    free(dec_delta);
    free(dec_canvas);
    free(enc_delta);
    free(enc_ref);
    free(src);
}

/* =========================================================================
 * Entropy coding, quantiser and robustness tests
 * ========================================================================= */

static uint32_t tex_hash(uint32_t x, uint32_t y) {
    uint32_t h = x * 374761393U + y * 668265263U;
    h = (h ^ (h >> 13)) * 1274126177U;
    return h ^ (h >> 16);
}

/* Three octaves of bilinear value noise: smooth, non-periodic texture. */
static uint8_t tex_at(uint32_t x, uint32_t y) {
    uint32_t acc = 0;
    for (uint32_t o = 0; o < 3U; o++) {
        uint32_t sh = 6U - 2U * o;
        uint32_t cell = 1U << sh;
        uint32_t gx = x >> sh, gy = y >> sh;
        uint32_t fx = x & (cell - 1U), fy = y & (cell - 1U);
        uint32_t a = tex_hash(gx, gy + o * 977U) & 255U;
        uint32_t b = tex_hash(gx + 1U, gy + o * 977U) & 255U;
        uint32_t c = tex_hash(gx, gy + 1U + o * 977U) & 255U;
        uint32_t d = tex_hash(gx + 1U, gy + 1U + o * 977U) & 255U;
        uint32_t top = a * (cell - fx) + b * fx;
        uint32_t bot = c * (cell - fx) + d * fx;
        acc += ((top * (cell - fy) + bot * fy) >> (2U * sh)) >> o;
    }
    return (uint8_t)((acc * 255U) / 445U);
}

/*
 * Test scene: textured background panning 1 px per frame on the top two
 * thirds, a still bottom third, a moving block and a sharp static HUD.
 * Fills every plane of the configured format.
 */
static void make_scene(uint8_t *src, const zcrudp_ternary_config_t *cfg, int f) {
    const uint32_t W = cfg->width, H = cfg->height;
    const size_t y_sz = (size_t)W * H;
    const uint32_t planes = (cfg->pixfmt == ZCRUDP_TERNARY_FMT_RGBA) ? 3U : 1U;

    for (uint32_t p = 0; p < planes; p++) {
        uint8_t *pl = src + (size_t)p * y_sz;
        for (uint32_t y = 0; y < H; y++) {
            uint32_t pan = (y < (2U * H) / 3U) ? (uint32_t)f : 0U;
            for (uint32_t x = 0; x < W; x++) {
                pl[(size_t)y * W + x] = tex_at(x + pan + p * 37U, y + p * 91U);
            }
        }
        uint32_t ox = (W / 8U + (uint32_t)f * 3U) % (W > 40U ? W - 40U : 1U);
        for (uint32_t y = H / 3U; y < H / 3U + 24U && y < H; y++) {
            for (uint32_t x = ox; x < ox + 32U && x < W; x++) {
                pl[(size_t)y * W + x] = (uint8_t)(200U - (x - ox) - (y - H / 3U));
            }
        }
        for (uint32_t y = 4U; y < 14U && y < H; y++) {
            for (uint32_t x = 6U; x < 70U && x < W; x++) {
                pl[(size_t)y * W + x] = ((x / 5U) & 1U) ? 245U : 15U;
            }
        }
    }
    if (cfg->pixfmt == ZCRUDP_TERNARY_FMT_NV12) {
        uint8_t *uv = src + y_sz;
        for (uint32_t y = 0; y < H / 2U; y++) {
            for (uint32_t x = 0; x + 1U < W; x += 2U) {
                uv[(size_t)y * W + x] = (uint8_t)(96U + tex_at(x / 2U + (uint32_t)f, y + 300U) / 4U);
                uv[(size_t)y * W + x + 1U] = (uint8_t)(170U - tex_at(x / 2U + (uint32_t)f, y + 700U) / 4U);
            }
        }
    }
}

static uint32_t cfg_num_planes(const zcrudp_ternary_config_t *cfg) {
    return (cfg->pixfmt == ZCRUDP_TERNARY_FMT_RGBA) ? 3U
         : (cfg->pixfmt == ZCRUDP_TERNARY_FMT_NV12) ? 2U : 1U;
}

static size_t cfg_plane_len(const zcrudp_ternary_config_t *cfg, uint32_t plane) {
    size_t y_sz = (size_t)cfg->width * cfg->height;
    return (cfg->pixfmt == ZCRUDP_TERNARY_FMT_NV12 && plane == 1U) ? (y_sz / 2U) : y_sz;
}

/* One encoder + decoder pair with its own state, for side-by-side runs. */
typedef struct {
    zcrudp_ternary_config_t cfg;
    uint8_t *ref, *canvas, *enc_delta, *dec_delta;
    pkt_sink_t sink;
    size_t delta_bytes, intra_bytes, trit_tokens;
} codec_pipe_t;

static void pipe_init(codec_pipe_t *c, const zcrudp_ternary_config_t *cfg, bool adm, size_t max_pkts) {
    size_t psz = zcrudp_ternary_planar_size(cfg);
    c->cfg = *cfg;
    c->ref = (uint8_t *)malloc(psz);
    c->canvas = (uint8_t *)malloc(psz);
    c->enc_delta = adm ? (uint8_t *)malloc(psz) : NULL;
    c->dec_delta = adm ? (uint8_t *)malloc(psz) : NULL;
    c->sink.pkts = (uint8_t *)malloc(max_pkts * 1400U);
    c->sink.lens = (size_t *)malloc(max_pkts * sizeof(size_t));
    c->sink.max_pkts = max_pkts;
    c->sink.mtu = 1400;
    c->sink.count = 0;
    c->sink.total_bytes = 0;
    c->delta_bytes = c->intra_bytes = c->trit_tokens = 0;
    assert(c->ref && c->canvas && c->sink.pkts && c->sink.lens);
    memset(c->ref, cfg->neutral_y, psz);
    memset(c->canvas, cfg->neutral_y, psz);
    if (adm) {
        assert(c->enc_delta && c->dec_delta);
        memset(c->enc_delta, cfg->delta_min, psz);
        memset(c->dec_delta, cfg->delta_min, psz);
    }
}

static void pipe_free(codec_pipe_t *c) {
    free(c->ref); free(c->canvas); free(c->enc_delta); free(c->dec_delta);
    free(c->sink.pkts); free(c->sink.lens);
}

/* Encodes one frame: keyframe on f == 0, else delta (+ optional chroma 2x) and a rolling anchor slice. */
static void pipe_encode(codec_pipe_t *c, const uint8_t *src, int f, bool chroma_2x, uint32_t intra_cycle) {
    const zcrudp_ternary_config_t *cfg = &c->cfg;
    const size_t y_sz = (size_t)cfg->width * cfg->height;
    uint8_t pkt_buf[1400];
    c->sink.count = 0;

    for (uint32_t p = 0; p < cfg_num_planes(cfg); p++) {
        size_t off = (size_t)p * y_sz;
        size_t len = cfg_plane_len(cfg, p);
        uint8_t *ed = c->enc_delta ? c->enc_delta + off : NULL;
        if (f == 0) {
            zcrudp_ternary_encode_intra_range(src + off, c->ref + off, ed, 0, len, len,
                                              (uint8_t)p, 0, cfg, pkt_buf, sink_cb, &c->sink);
            continue;
        }
        if (chroma_2x && p == 1U) {
            zcrudp_ternary_encode_plane_2x(src + off, c->ref + off, ed, len, (uint8_t)p,
                                           (uint16_t)f, cfg, pkt_buf, sink_cb, &c->sink);
        } else {
            zcrudp_ternary_encode_plane(src + off, c->ref + off, ed, len, (uint8_t)p,
                                        (uint16_t)f, cfg, pkt_buf, sink_cb, &c->sink);
        }
        if (intra_cycle > 0U) {
            size_t slice = (len + intra_cycle - 1U) / intra_cycle;
            size_t i0 = (size_t)((uint32_t)f % intra_cycle) * slice;
            zcrudp_ternary_encode_intra_range(src + off, c->ref + off, ed, i0, i0 + slice, len,
                                              (uint8_t)p, (uint16_t)f, cfg, pkt_buf, sink_cb, &c->sink);
        }
    }
    assert(c->sink.count < c->sink.max_pkts);

    for (size_t i = 0; i < c->sink.count; i++) {
        const uint8_t *pd = c->sink.pkts + i * c->sink.mtu;
        zcrudp_ternary_pkt_hdr_t hdr;
        assert(zcrudp_ternary_pkt_hdr_read_le(pd, &hdr));
        uint8_t type = ZCRUDP_TERNARY_HDR_TYPE(&hdr);
        assert(c->sink.lens[i] <= cfg->mtu_payload);
        if (type == ZCRUDP_TERNARY_PKT_INTRA || type == ZCRUDP_TERNARY_PKT_INTRA_EC) {
            c->intra_bytes += c->sink.lens[i];
        } else {
            c->delta_bytes += c->sink.lens[i];
        }
        if (type == ZCRUDP_TERNARY_PKT_DELTA || type == ZCRUDP_TERNARY_PKT_DELTA_2X) {
            for (uint16_t t = 0; t < hdr.num_tokens; t++) {
                uint16_t tok = zcrudp_ternary_read_u16_le(pd + ZCRUDP_TERNARY_PKT_HDR_SZ + (size_t)t * 2U);
                if (tok < ZCRUDP_TRIT_MAX_10P) c->trit_tokens++;
            }
        }
    }
}

/* Decays, then applies every packet of the frame except index skip_pkt (SIZE_MAX = none). */
static void pipe_decode(codec_pipe_t *c, size_t skip_pkt) {
    const zcrudp_ternary_config_t *cfg = &c->cfg;
    const size_t y_sz = (size_t)cfg->width * cfg->height;
    for (uint32_t p = 0; p < cfg_num_planes(cfg); p++) {
        zcrudp_ternary_decay_plane(c->canvas + (size_t)p * y_sz, cfg_plane_len(cfg, p),
                                   cfg->gamma_q8, cfg->neutral_y);
    }
    for (size_t i = 0; i < c->sink.count; i++) {
        if (i == skip_pkt) continue;
        const uint8_t *pd = c->sink.pkts + i * c->sink.mtu;
        zcrudp_ternary_pkt_hdr_t hdr;
        assert(zcrudp_ternary_pkt_hdr_read_le(pd, &hdr));
        assert(hdr.plane < cfg_num_planes(cfg));
        size_t off = (size_t)hdr.plane * y_sz;
        bool ok = zcrudp_ternary_decode_packet(pd, c->sink.lens[i], c->canvas + off,
                                               c->dec_delta ? c->dec_delta + off : NULL,
                                               cfg_plane_len(cfg, hdr.plane), cfg, NULL);
        assert(ok);
        (void)ok;
    }
}

#ifdef __AVX2__
/* Every one of the 59049 tokens must unpack to the trits the scalar path gives. */
static void test_avx2_unpack_matches_scalar(void) {
    printf("[TEST] Running AVX2 Token Unpack vs Scalar (all 59049 tokens)... ");
    for (uint32_t tok = 0; tok < ZCRUDP_TRIT_MAX_10P; tok++) {
        zcrudp_trit_t want[10];
        int8_t got[16];
        zcrudp_unpack_10trits((uint16_t)tok, want);
        zcrudp_ternary_unpack10_avx2((uint16_t)tok, got);
        assert(memcmp(want, got, 10) == 0);
    }
    printf("PASSED\n");
}

/*
 * The vector block quantiser and the vector trit application against the
 * scalar code, on random blocks and settings (ADM on and off, extreme steps,
 * thresholds, leak factors and neutral values). The 6 bytes after a block
 * belong to its neighbour and must come back untouched.
 */
static void test_avx2_block_matches_scalar(void) {
    printf("[TEST] Running AVX2 Block Quantiser / Apply vs Scalar (random)... ");
    uint32_t rng = 0xC0FFEEU;
    #define RND() (rng ^= rng << 13, rng ^= rng >> 17, rng ^= rng << 5, rng)
    const int ROUNDS = 200000;
    int coded_blocks = 0;

    for (int r = 0; r < ROUNDS; r++) {
        uint8_t dmin = (uint8_t)(1U + RND() % 8U);
        zcrudp_ternary_config_t cfg = {
            .width = 16, .height = 1, .pixfmt = ZCRUDP_TERNARY_FMT_GRAY8,
            .delta_min = dmin, .threshold = (uint8_t)(RND() % 12U), .mtu_payload = 1380
        };
        uint32_t pick = RND();
        cfg.delta_max = (pick & 1U) ? (uint8_t)(dmin + RND() % 250U) : dmin;
        if (cfg.delta_max < dmin) cfg.delta_max = 255;
        cfg.gamma_q8 = (pick & 2U) ? 255 : (uint8_t)(RND() % 256U);
        cfg.neutral_y = (pick & 4U) ? 128 : (uint8_t)(RND() % 256U);
        bool adm = (pick & 8U) != 0U;
        /* small differences most of the time: that is where the decisions flip */
        uint32_t spread = (pick & 16U) ? 256U : 24U;

        uint8_t src[16], ref_s[16], ref_v[16], d_s[16], d_v[16];
        for (int i = 0; i < 16; i++) {
            ref_s[i] = (uint8_t)RND();
            int v = (int)ref_s[i] + (int)(RND() % spread) - (int)(spread / 2U);
            src[i] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
            d_s[i] = (pick & 32U) ? (uint8_t)RND() : (uint8_t)(dmin << (RND() % 5U));
        }
        memcpy(ref_v, ref_s, 16);
        memcpy(d_v, d_s, 16);

        /* scalar block, as zcrudp_ternary_encode_plane_range() does it */
        zcrudp_trit_t t_s[10];
        bool coded_s = false;
        for (int j = 0; j < 10; j++) {
            uint8_t d = adm ? d_s[j] : cfg.delta_min;
            t_s[j] = zcrudp_ternary_encode_sample(src[j], &ref_s[j], d, adm, &cfg, &coded_s);
            if (t_s[j] != 0) coded_s = true;
        }
        if (coded_s && adm) {
            for (int j = 0; j < 10; j++) {
                d_s[j] = zcrudp_adm_update_delta(d_s[j], t_s[j], cfg.delta_min, cfg.delta_max);
            }
        }

        int8_t t_v[16];
        bool coded_v = zcrudp_ternary_encode_block_avx2(src, ref_v, adm ? d_v : NULL, &cfg, t_v);
        assert(coded_v == coded_s);
        assert(memcmp(t_v, t_s, 10) == 0);
        assert(memcmp(ref_v, ref_s, 16) == 0);
        assert(memcmp(d_v, d_s, 16) == 0);
        coded_blocks += coded_s;

        /* decoder side: same trits on a random canvas */
        uint8_t can_s[16], can_v[16], dd_s[16], dd_v[16];
        for (int i = 0; i < 16; i++) {
            can_s[i] = can_v[i] = (uint8_t)RND();
            dd_s[i] = dd_v[i] = (uint8_t)RND();
        }
        zcrudp_ternary_apply_block(can_s, adm ? dd_s : NULL, 16, &cfg, 0, false, false, t_s);
        zcrudp_ternary_apply_block_avx2(can_v, adm ? dd_v : NULL, &cfg, t_v);
        assert(memcmp(can_v, can_s, 16) == 0);
        assert(memcmp(dd_v, dd_s, 16) == 0);
    }
    #undef RND
    assert(coded_blocks > ROUNDS / 4 && coded_blocks < ROUNDS);
    printf("PASSED (%d blocks, %d coded)\n", ROUNDS, coded_blocks);
}
#endif

/*
 * The vectorised decay must match the scalar integrator for any neutral value,
 * not only 128 (a signed 16-bit product overflows as soon as |y - neutral|
 * times gamma exceeds 32767).
 */
static void test_decay_matches_scalar_any_neutral(void) {
    printf("[TEST] Running Plane Decay vs Scalar Integrator (all neutral values)... ");
    static const uint8_t neutrals[6] = { 0, 16, 127, 128, 235, 255 };
    static const uint8_t gammas[5] = { 0, 1, 200, 250, 255 };
    uint8_t buf[256 + 77];

    for (size_t n = 0; n < 6U; n++) {
        for (size_t g = 0; g < 5U; g++) {
            for (size_t i = 0; i < sizeof(buf); i++) buf[i] = (uint8_t)(i * 7U);
            zcrudp_ternary_decay_plane(buf, sizeof(buf), gammas[g], neutrals[n]);
            for (size_t i = 0; i < sizeof(buf); i++) {
                uint8_t want = zcrudp_leaky_integrator_step((uint8_t)(i * 7U), 0, 0,
                                                            gammas[g], neutrals[n]);
                assert(buf[i] == want);
            }
        }
    }
    printf("PASSED\n");
}

/*
 * Range-coded packets must rebuild exactly the picture the token packets do,
 * for every format and mode, and cost fewer bytes.
 */
static void test_entropy_matches_tokens(void) {
    printf("[TEST] Running Entropy-Coded vs Token Bitstream Equivalence...\n");

    typedef struct {
        const char *name;
        zcrudp_ternary_pixfmt_t fmt;
        uint16_t w, h, mtu;
        bool adm, chroma_2x;
        uint8_t gamma;
        uint32_t intra_cycle;
    } variant_t;
    static const variant_t variants[] = {
        { "GRAY8 ADM",                 ZCRUDP_TERNARY_FMT_GRAY8, 320, 180, 1380, true,  false, 254, 12 },
        { "NV12 ADM + chroma 2x",      ZCRUDP_TERNARY_FMT_NV12,  320, 180, 1380, true,  true,  254, 12 },
        { "RGBA fixed step",           ZCRUDP_TERNARY_FMT_RGBA,  320, 180, 1380, false, false, 250, 0  },
        { "NV12 ADM, 300-byte MTU",    ZCRUDP_TERNARY_FMT_NV12,  322, 90,  300,  true,  true,  255, 6  },
        { "GRAY8 ADM, 8200 px wide",   ZCRUDP_TERNARY_FMT_GRAY8, 8200, 6,  1380, true,  false, 254, 4  },
    };

    for (size_t v = 0; v < sizeof(variants) / sizeof(variants[0]); v++) {
        const variant_t *va = &variants[v];
        zcrudp_ternary_config_t cfg = {
            .width = va->w, .height = va->h, .pixfmt = va->fmt,
            .delta_min = 2, .delta_max = 32, .threshold = 2,
            .gamma_q8 = va->gamma, .neutral_y = 128, .mtu_payload = va->mtu
        };
        size_t psz = zcrudp_ternary_planar_size(&cfg);
        uint8_t *src = (uint8_t *)malloc(psz);
        assert(src);

        codec_pipe_t tok, ec;
        pipe_init(&tok, &cfg, va->adm, 4096);
        cfg.entropy = ZCRUDP_TERNARY_EC_ALL;
        pipe_init(&ec, &cfg, va->adm, 4096);

        for (int f = 0; f < 12; f++) {
            make_scene(src, &cfg, f);
            pipe_encode(&tok, src, f, va->chroma_2x, va->intra_cycle);
            pipe_encode(&ec, src, f, va->chroma_2x, va->intra_cycle);
            pipe_decode(&tok, SIZE_MAX);
            pipe_decode(&ec, SIZE_MAX);

            assert(memcmp(tok.ref, tok.canvas, psz) == 0);
            assert(memcmp(ec.ref, ec.canvas, psz) == 0);
            assert(memcmp(tok.canvas, ec.canvas, psz) == 0);
            if (va->adm) {
                assert(memcmp(ec.enc_delta, ec.dec_delta, psz) == 0);
                assert(memcmp(tok.dec_delta, ec.dec_delta, psz) == 0);
            }
        }

        assert(ec.delta_bytes < tok.delta_bytes);
        assert(ec.intra_bytes < tok.intra_bytes);
        double bits_per_trit = (double)ec.delta_bytes * 8.0 / ((double)tok.trit_tokens * 10.0);
        printf("       %-26s delta %7zu -> %7zu B (x%.2f, %.2f bit/coded trit vs 1.60) | anchors %7zu -> %7zu B (x%.2f)\n",
               va->name, tok.delta_bytes, ec.delta_bytes,
               (double)tok.delta_bytes / (double)ec.delta_bytes, bits_per_trit,
               tok.intra_bytes, ec.intra_bytes,
               (double)tok.intra_bytes / (double)ec.intra_bytes);

        pipe_free(&tok);
        pipe_free(&ec);
        free(src);
    }
    printf("       PASSED (bit-exact with tokens in every mode)\n");
}

/*
 * Each range-coded packet restarts the coder and its statistics, so dropping
 * one must leave every sample covered by the other packets exactly as the
 * encoder expects; the lost span only keeps its decayed previous value.
 */
static void test_entropy_packet_loss_is_isolated(void) {
    printf("[TEST] Running Entropy-Coded Packet Loss Isolation Test... ");

    zcrudp_ternary_config_t cfg = {
        .width = 320, .height = 180, .pixfmt = ZCRUDP_TERNARY_FMT_GRAY8,
        .delta_min = 3, .delta_max = 3, .threshold = 2,
        .gamma_q8 = 252, .neutral_y = 128, .mtu_payload = 400,
        .entropy = ZCRUDP_TERNARY_EC_ALL
    };
    size_t len = (size_t)cfg.width * cfg.height;
    uint8_t *src = (uint8_t *)malloc(len);
    uint8_t *before = (uint8_t *)malloc(len);
    uint8_t *synced = (uint8_t *)malloc(len);
    assert(src && before && synced);

    codec_pipe_t c;
    pipe_init(&c, &cfg, false, 1024);
    size_t drops = 0;

    for (int f = 0; f < 6; f++) {
        make_scene(src, &cfg, f);
        memcpy(synced, c.canvas, len);
        pipe_encode(&c, src, f, false, 0);
        if (f == 0) {
            pipe_decode(&c, SIZE_MAX);
            continue;
        }
        assert(c.sink.count >= 4U);

        /* Try several loss positions from the same synchronised state. */
        for (size_t drop = 0; drop < c.sink.count; drop += 1U + c.sink.count / 5U) {
            zcrudp_ternary_pkt_hdr_t hdr, next;
            assert(zcrudp_ternary_pkt_hdr_read_le(c.sink.pkts + drop * c.sink.mtu, &hdr));
            assert(ZCRUDP_TERNARY_HDR_TYPE(&hdr) == ZCRUDP_TERNARY_PKT_DELTA_EC);
            size_t lost_begin = (size_t)hdr.start_block * 10U;
            size_t lost_end = len;
            if (drop + 1U < c.sink.count) {
                assert(zcrudp_ternary_pkt_hdr_read_le(c.sink.pkts + (drop + 1U) * c.sink.mtu, &next));
                lost_end = (size_t)next.start_block * 10U;
            }

            memcpy(c.canvas, synced, len);
            memcpy(before, synced, len);
            zcrudp_ternary_decay_plane(before, len, cfg.gamma_q8, cfg.neutral_y);
            pipe_decode(&c, drop);

            for (size_t i = 0; i < len; i++) {
                uint8_t want = (i >= lost_begin && i < lost_end) ? before[i] : c.ref[i];
                assert(c.canvas[i] == want);
            }
            drops++;
        }

        memcpy(c.canvas, synced, len);
        pipe_decode(&c, SIZE_MAX);
        assert(memcmp(c.ref, c.canvas, len) == 0);
    }

    pipe_free(&c);
    free(src);
    free(before);
    free(synced);
    printf("PASSED (%zu single-packet losses, no spill outside the lost span)\n", drops);
}

static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/*
 * Packets come off the network: whatever their content, decoding must stay
 * inside the plane. The planes sit between guard bands that are checked after
 * every packet (and the sanitizer build catches out-of-bounds reads).
 */
static void test_malformed_packets_stay_in_bounds(void) {
    printf("[TEST] Running Malformed / Hostile Packet Bounds Test... ");

    zcrudp_ternary_config_t cfg = {
        .width = 64, .height = 18, .pixfmt = ZCRUDP_TERNARY_FMT_NV12,
        .delta_min = 2, .delta_max = 32, .threshold = 2,
        .gamma_q8 = 254, .neutral_y = 128, .mtu_payload = 256,
        .entropy = ZCRUDP_TERNARY_EC_ALL
    };
    enum { GUARD = 64 };
    size_t y_sz = (size_t)cfg.width * cfg.height;
    size_t psz = zcrudp_ternary_planar_size(&cfg);
    uint8_t *canvas_buf = (uint8_t *)malloc(psz + 2U * GUARD);
    uint8_t *delta_buf = (uint8_t *)malloc(psz + 2U * GUARD);
    uint8_t *src = (uint8_t *)malloc(psz);
    assert(canvas_buf && delta_buf && src);
    memset(canvas_buf, 0xA5, psz + 2U * GUARD);
    memset(delta_buf, 0xA5, psz + 2U * GUARD);
    uint8_t *canvas = canvas_buf + GUARD;
    uint8_t *delta = delta_buf + GUARD;
    memset(canvas, 128, psz);
    memset(delta, cfg.delta_min, psz);

    /* Real packets of every type to mutate. */
    codec_pipe_t c;
    pipe_init(&c, &cfg, true, 1024);
    size_t valid_count = 0;
    static uint8_t valid_pkts[512][256];
    static size_t valid_lens[512];
    for (int f = 0; f < 4; f++) {
        c.cfg.entropy = (f & 1) ? 0U : ZCRUDP_TERNARY_EC_ALL;
        make_scene(src, &cfg, f);
        pipe_encode(&c, src, f, (f & 2) != 0, 3);
        pipe_decode(&c, SIZE_MAX);
        for (size_t i = 0; i < c.sink.count && valid_count < 512U; i++) {
            memcpy(valid_pkts[valid_count], c.sink.pkts + i * c.sink.mtu, c.sink.lens[i]);
            valid_lens[valid_count++] = c.sink.lens[i];
        }
    }
    assert(valid_count > 16U);

    uint32_t rng = 0x1234567U;
    size_t rejected = 0;
    const size_t ROUNDS = 60000;
    for (size_t r = 0; r < ROUNDS; r++) {
        uint8_t pkt[300];
        size_t plen;
        if (r & 1U) {
            /* Mutate a real packet: flip bytes, maybe truncate. */
            size_t pick = xorshift32(&rng) % valid_count;
            plen = valid_lens[pick];
            memcpy(pkt, valid_pkts[pick], plen);
            uint32_t flips = 1U + xorshift32(&rng) % 4U;
            for (uint32_t k = 0; k < flips; k++) {
                pkt[2U + xorshift32(&rng) % (plen - 2U)] ^= (uint8_t)(1U << (xorshift32(&rng) % 8U));
            }
            if ((xorshift32(&rng) % 4U) == 0U) {
                plen = xorshift32(&rng) % (plen + 1U);
            }
        } else {
            /* Random payload behind a valid magic, any type nibble. */
            plen = xorshift32(&rng) % sizeof(pkt);
            for (size_t i = 0; i < plen; i++) pkt[i] = (uint8_t)xorshift32(&rng);
            if (plen >= ZCRUDP_TERNARY_PKT_HDR_SZ) {
                zcrudp_ternary_pkt_hdr_t hdr = {
                    .magic = ZCRUDP_TERNARY_PKT_MAGIC,
                    .pixfmt = (uint8_t)xorshift32(&rng),
                    .plane = (uint8_t)(xorshift32(&rng) % 2U),
                    .frame_id = (uint16_t)xorshift32(&rng),
                    .start_block = (xorshift32(&rng) % 3U) ? (xorshift32(&rng) % 200U) : xorshift32(&rng),
                    .num_tokens = (uint16_t)((xorshift32(&rng) % 3U) ? (xorshift32(&rng) % 150U) : xorshift32(&rng))
                };
                zcrudp_ternary_pkt_hdr_write_le(pkt, &hdr);
            }
        }

        uint32_t plane = (plen > 3U) ? (pkt[3] & 1U) : 0U;
        size_t off = plane ? y_sz : 0U;
        size_t plane_len = plane ? (y_sz / 2U) : y_sz;
        /* A heap copy sized to the packet lets the sanitizer see over-reads. */
        uint8_t *exact = (uint8_t *)malloc(plen ? plen : 1U);
        assert(exact);
        memcpy(exact, pkt, plen);
        bool ok = zcrudp_ternary_decode_packet(exact, plen, canvas + off, delta + off,
                                               plane_len, &cfg, NULL);
        free(exact);
        if (!ok) rejected++;

        for (size_t i = 0; i < GUARD; i++) {
            assert(canvas_buf[i] == 0xA5 && canvas_buf[GUARD + psz + i] == 0xA5);
            assert(delta_buf[i] == 0xA5 && delta_buf[GUARD + psz + i] == 0xA5);
        }
    }

    /* A TILE_MV packet is not picture data: refused, canvas left as it was. */
    uint8_t mv_pkt[ZCRUDP_TERNARY_PKT_HDR_SZ + 8];
    zcrudp_ternary_pkt_hdr_t mv_hdr = {
        .magic = ZCRUDP_TERNARY_PKT_MAGIC,
        .pixfmt = ZCRUDP_TERNARY_MAKE_PF(cfg.pixfmt, ZCRUDP_TERNARY_PKT_TILE_MV),
        .plane = 0, .frame_id = 1, .start_block = 0, .num_tokens = 4
    };
    zcrudp_ternary_pkt_hdr_write_le(mv_pkt, &mv_hdr);
    memset(mv_pkt + ZCRUDP_TERNARY_PKT_HDR_SZ, 3, 8);
    memcpy(src, canvas, psz);
    assert(!zcrudp_ternary_decode_packet(mv_pkt, sizeof(mv_pkt), canvas, delta, y_sz, &cfg, NULL));
    assert(memcmp(src, canvas, psz) == 0);

    pipe_free(&c);
    free(src);
    free(canvas_buf);
    free(delta_buf);
    printf("PASSED (%zu packets, %zu rejected, guard bands intact)\n", ROUNDS, rejected);
}

/*
 * The CPU cap switches a plane from range-coded packets to tokens once the
 * allowed number of blocks is spent. The picture must not change, the stream
 * must really hold both kinds of packets, and the capped block count must hold.
 */
static void test_entropy_cpu_cap(void) {
    printf("[TEST] Running Entropy CPU Cap (range-coded, then tokens) Test... ");

    zcrudp_ternary_config_t cfg = {
        .width = 320, .height = 180, .pixfmt = ZCRUDP_TERNARY_FMT_GRAY8,
        .delta_min = 2, .delta_max = 32, .threshold = 2,
        .gamma_q8 = 254, .neutral_y = 128, .mtu_payload = 600
    };
    const uint32_t CAP = 1500;
    size_t len = (size_t)cfg.width * cfg.height;
    uint8_t *src = (uint8_t *)malloc(len);
    assert(src);

    codec_pipe_t tok, capped, full;
    pipe_init(&tok, &cfg, true, 2048);
    cfg.entropy = ZCRUDP_TERNARY_EC_ALL;
    pipe_init(&full, &cfg, true, 2048);
    cfg.ec_max_blocks = CAP;
    pipe_init(&capped, &cfg, true, 2048);

    size_t ec_pkts = 0, tok_pkts = 0;
    for (int f = 0; f < 8; f++) {
        make_scene(src, &cfg, f);
        pipe_encode(&tok, src, f, false, 0);
        pipe_encode(&full, src, f, false, 0);
        pipe_encode(&capped, src, f, false, 0);

        size_t ec_blocks = 0;
        for (size_t i = 0; i < capped.sink.count; i++) {
            zcrudp_ternary_pkt_hdr_t hdr;
            assert(zcrudp_ternary_pkt_hdr_read_le(capped.sink.pkts + i * capped.sink.mtu, &hdr));
            uint8_t type = ZCRUDP_TERNARY_HDR_TYPE(&hdr);
            if (type == ZCRUDP_TERNARY_PKT_DELTA_EC) {
                ec_pkts++;
                ec_blocks += hdr.num_tokens;
                assert(tok_pkts == 0 || f > 1); /* range-coded packets come first */
            } else if (type == ZCRUDP_TERNARY_PKT_DELTA) {
                tok_pkts++;
            }
        }
        assert(ec_blocks <= CAP);

        pipe_decode(&tok, SIZE_MAX);
        pipe_decode(&full, SIZE_MAX);
        pipe_decode(&capped, SIZE_MAX);
        assert(memcmp(capped.ref, capped.canvas, len) == 0);
        assert(memcmp(capped.canvas, tok.canvas, len) == 0);
        assert(memcmp(capped.dec_delta, tok.dec_delta, len) == 0);
    }
    assert(ec_pkts > 0 && tok_pkts > 0);
    assert(full.delta_bytes < capped.delta_bytes && capped.delta_bytes < tok.delta_bytes);

    printf("PASSED (delta bytes: tokens %zu, capped %zu, uncapped %zu)\n",
           tok.delta_bytes, capped.delta_bytes, full.delta_bytes);
    pipe_free(&tok);
    pipe_free(&full);
    pipe_free(&capped);
    free(src);
}

/*
 * Multi-threaded decode of a frame that mixes deltas and the anchors that
 * overwrite them must give the picture a one-by-one decode gives, whichever
 * thread gets which packet.
 */
static void test_mt_decode_keeps_packet_order(void) {
    printf("[TEST] Running MT Decode Order (deltas + overlapping anchors) Test... ");

    zcrudp_ternary_config_t cfg = {
        .width = 320, .height = 180, .pixfmt = ZCRUDP_TERNARY_FMT_NV12,
        .delta_min = 2, .delta_max = 32, .threshold = 2,
        .gamma_q8 = 254, .neutral_y = 128, .mtu_payload = 200,
        .entropy = ZCRUDP_TERNARY_EC_INTRA
    };
    size_t psz = zcrudp_ternary_planar_size(&cfg);
    size_t y_sz = (size_t)cfg.width * cfg.height;
    uint8_t *src = (uint8_t *)malloc(psz);
    uint8_t *mt_canvas = (uint8_t *)malloc(psz);
    uint8_t *mt_delta = (uint8_t *)malloc(psz);
    assert(src && mt_canvas && mt_delta);
    memset(mt_canvas, cfg.neutral_y, psz);
    memset(mt_delta, cfg.delta_min, psz);

    codec_pipe_t c;
    pipe_init(&c, &cfg, true, 4096);

    const uint32_t THREADS = 8;
    uint8_t *slice_pkts = (uint8_t *)malloc(THREADS * 16U * 1400U);
    size_t *slice_lens = (size_t *)malloc(THREADS * 16U * sizeof(size_t));
    assert(slice_pkts && slice_lens);
    zcrudp_ternary_mt_pool_t pool;
    zcrudp_ternary_mt_init(&pool, THREADS, slice_pkts, slice_lens, 16, 1400);

    size_t mixed_frames = 0;
    for (int f = 0; f < 10; f++) {
        make_scene(src, &cfg, f);
        /* Per plane: deltas, then an anchor slice on top of them (cycle of 3). */
        pipe_encode(&c, src, f, (f & 1) != 0, 3);
        pipe_decode(&c, SIZE_MAX);
        assert(memcmp(c.ref, c.canvas, psz) == 0);
        if (f > 0) mixed_frames++;

        zcrudp_ternary_mt_decay_plane(&pool, mt_canvas, y_sz, &cfg);
        zcrudp_ternary_mt_decay_plane(&pool, mt_canvas + y_sz, y_sz / 2U, &cfg);
        zcrudp_ternary_mt_decode_packets(&pool, mt_canvas, mt_delta, y_sz, &cfg,
                                         c.sink.pkts, c.sink.lens, c.sink.mtu, c.sink.count);
        assert(memcmp(mt_canvas, c.canvas, psz) == 0);
        assert(memcmp(mt_delta, c.dec_delta, psz) == 0);
    }

    zcrudp_ternary_mt_destroy(&pool);
    pipe_free(&c);
    free(slice_pkts);
    free(slice_lens);
    free(src);
    free(mt_canvas);
    free(mt_delta);
    printf("PASSED (%zu mixed frames on %u threads, identical to sequential)\n",
           mixed_frames, THREADS);
}

/*
 * Rate/quality summary on the textured scene: the same encoder settings with
 * tokens and with entropy coding, then entropy coding with a finer threshold
 * that spends part of the saving on quality.
 */
static void test_entropy_rate_quality(void) {
    printf("[TEST] Running Rate / Quality Summary (640x360 NV12, textured pan)...\n");

    typedef struct { const char *name; uint8_t entropy, threshold, delta_min; } point_t;
    static const point_t points[3] = {
        { "tokens,  threshold 4", 0U, 4U, 3U },
        { "entropy, threshold 4", ZCRUDP_TERNARY_EC_ALL, 4U, 3U },
        { "entropy, threshold 2", ZCRUDP_TERNARY_EC_ALL, 2U, 3U },
    };
    double psnr[3], bpp[3];

    for (size_t k = 0; k < 3U; k++) {
        zcrudp_ternary_config_t cfg = {
            .width = 640, .height = 360, .pixfmt = ZCRUDP_TERNARY_FMT_NV12,
            .delta_min = points[k].delta_min, .delta_max = 32, .threshold = points[k].threshold,
            .gamma_q8 = 254, .neutral_y = 128, .mtu_payload = 1380,
            .entropy = points[k].entropy
        };
        size_t psz = zcrudp_ternary_planar_size(&cfg);
        size_t y_sz = (size_t)cfg.width * cfg.height;
        uint8_t *src = (uint8_t *)malloc(psz);
        assert(src);
        codec_pipe_t c;
        pipe_init(&c, &cfg, true, 4096);

        double sum_psnr = 0.0;
        size_t bytes_at_4 = 0;
        const int FRAMES = 24;
        for (int f = 0; f < FRAMES; f++) {
            make_scene(src, &cfg, f);
            pipe_encode(&c, src, f, false, 24);
            pipe_decode(&c, SIZE_MAX);
            assert(memcmp(c.ref, c.canvas, psz) == 0);
            if (f == 3) bytes_at_4 = c.delta_bytes + c.intra_bytes;
            if (f >= 4) {
                double mse = 0.0;
                for (size_t i = 0; i < y_sz; i++) {
                    double d = (double)src[i] - (double)c.canvas[i];
                    mse += d * d;
                }
                mse /= (double)y_sz;
                sum_psnr += (mse > 1e-6) ? (10.0 * log10((255.0 * 255.0) / mse)) : 99.0;
            }
        }
        psnr[k] = sum_psnr / (FRAMES - 4);
        bpp[k] = ((double)(c.delta_bytes + c.intra_bytes - bytes_at_4) / (FRAMES - 4) * 8.0) / (double)y_sz;
        printf("       %-22s Y-PSNR %.2f dB at %.2f bpp\n", points[k].name, psnr[k], bpp[k]);
        pipe_free(&c);
        free(src);
    }

    /* Same settings: same picture, fewer bits. Finer threshold: better picture, still fewer bits. */
    assert(fabs(psnr[1] - psnr[0]) < 1e-9 && bpp[1] < bpp[0]);
    assert(psnr[2] > psnr[0] && bpp[2] < bpp[0]);
    printf("       PASSED\n");
}

static void test_intra_q_ec_roundtrip_and_bounds(void) {
    printf("[TEST] Running Quantized Spatial Anchor (INTRA_Q_EC) & Bounds Test...\n");
    const uint16_t W = 320, H = 180;
    const size_t len = (size_t)W * H;
    uint8_t *src = (uint8_t *)malloc(len);
    uint8_t *ref_q = (uint8_t *)malloc(len), *can_q = (uint8_t *)malloc(len);
    uint8_t *ref_l = (uint8_t *)malloc(len);
    uint8_t *ed = (uint8_t *)malloc(len), *dd = (uint8_t *)malloc(len);
    assert(src && ref_q && can_q && ref_l && ed && dd);

    for (uint16_t y = 0; y < H; y++) {
        for (uint16_t x = 0; x < W; x++) {
            int v = 128 + (int)(45.0 * sin(x * 0.19) * cos(y * 0.17) +
                                30.0 * sin((x + y) * 0.37));
            if (((x / 37) + (y / 41)) % 3 == 0) v += 40;
            src[(size_t)y * W + x] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    }

    zcrudp_ternary_config_t cfg_l = {
        .width = W, .height = H, .pixfmt = ZCRUDP_TERNARY_FMT_GRAY8,
        .delta_min = 3, .delta_max = 32, .threshold = 3,
        .gamma_q8 = 254, .neutral_y = 128, .mtu_payload = 1380,
        .entropy = ZCRUDP_TERNARY_EC_ALL, .intra_qstep = 0
    };
    zcrudp_ternary_config_t cfg_q = cfg_l;
    cfg_q.intra_qstep = 4;

    pkt_sink_t sq = { (uint8_t *)malloc(256U * 1400U), (size_t *)malloc(256U * sizeof(size_t)), 0, 256, 1400, 0 };
    pkt_sink_t sl = { (uint8_t *)malloc(256U * 1400U), (size_t *)malloc(256U * sizeof(size_t)), 0, 256, 1400, 0 };
    assert(sq.pkts && sq.lens && sl.pkts && sl.lens);
    uint8_t pkt_buf[1400];

    memset(ref_l, 128, len);
    memset(ref_q, 128, len);
    memset(can_q, 128, len);
    memset(ed, cfg_q.delta_min, len);
    memset(dd, cfg_q.delta_min, len);

    size_t bytes_l = zcrudp_ternary_encode_intra_range(src, ref_l, NULL, 0, len, len, 0, 0, &cfg_l,
                                                       pkt_buf, sink_cb, &sl);
    size_t bytes_q = zcrudp_ternary_encode_intra_range(src, ref_q, ed, 0, len, len, 0, 0, &cfg_q,
                                                       pkt_buf, sink_cb, &sq);
    assert(bytes_q < bytes_l);
    assert(sq.count > 1U);

    bool saw_q_ec = false;
    for (size_t i = 0; i < sq.count; i++) {
        const uint8_t *p = sq.pkts + i * sq.mtu;
        zcrudp_ternary_pkt_hdr_t h;
        assert(zcrudp_ternary_pkt_hdr_read_le(p, &h));
        if (ZCRUDP_TERNARY_HDR_TYPE(&h) == ZCRUDP_TERNARY_PKT_INTRA_Q_EC) {
            saw_q_ec = true;
        }
        assert(zcrudp_ternary_route_packet(p, sq.lens[i]) == ZCRUDP_TERNARY_CHAN_RELIABLE_FEC);
        assert(zcrudp_ternary_decode_packet(p, sq.lens[i], can_q, dd, len, &cfg_q, NULL));
    }
    assert(saw_q_ec);
    assert(memcmp(ref_q, can_q, len) == 0);
    assert(memcmp(ed, dd, len) == 0);

    double mse = 0.0;
    for (size_t i = 0; i < len; i++) {
        double d = (double)src[i] - (double)can_q[i];
        mse += d * d;
    }
    double psnr = 10.0 * log10((255.0 * 255.0) / (mse / (double)len));
    assert(psnr >= 43.0);

    /* Loss isolation: drop packet 1 and ensure all other packets decode identically */
    uint8_t *can_loss = (uint8_t *)malloc(len);
    assert(can_loss);
    memset(can_loss, 128, len);
    zcrudp_ternary_pkt_hdr_t drop_h;
    assert(zcrudp_ternary_pkt_hdr_read_le(sq.pkts + 1U * sq.mtu, &drop_h));
    for (size_t i = 0; i < sq.count; i++) {
        if (i == 1U) continue;
        assert(zcrudp_ternary_decode_packet(sq.pkts + i * sq.mtu, sq.lens[i], can_loss, NULL, len, &cfg_q, NULL));
    }
    size_t d0 = drop_h.start_block;
    size_t d1 = d0 + drop_h.num_tokens;
    assert(memcmp(can_loss, can_q, d0) == 0);
    assert(memcmp(can_loss + d1, can_q + d1, len - d1) == 0);

    /* Malformed INTRA_Q_EC bounds checks */
    uint8_t bad_pkt[64];
    memcpy(bad_pkt, sq.pkts, sq.lens[0] < sizeof(bad_pkt) ? sq.lens[0] : sizeof(bad_pkt));
    bad_pkt[2] = ZCRUDP_TERNARY_MAKE_PF(ZCRUDP_TERNARY_FMT_GRAY8, ZCRUDP_TERNARY_PKT_INTRA_Q_EC);
    assert(!zcrudp_ternary_decode_packet(bad_pkt, ZCRUDP_TERNARY_PKT_HDR_SZ + 1U, can_loss, NULL, len, &cfg_q, NULL));
    bad_pkt[ZCRUDP_TERNARY_PKT_HDR_SZ] = 0U;
    assert(!zcrudp_ternary_decode_packet(bad_pkt, ZCRUDP_TERNARY_PKT_HDR_SZ + 4U, can_loss, NULL, len, &cfg_q, NULL));
    bad_pkt[ZCRUDP_TERNARY_PKT_HDR_SZ] = 1U;
    assert(!zcrudp_ternary_decode_packet(bad_pkt, ZCRUDP_TERNARY_PKT_HDR_SZ + 4U, can_loss, NULL, len, &cfg_q, NULL));
    bad_pkt[ZCRUDP_TERNARY_PKT_HDR_SZ] = 65U;
    assert(!zcrudp_ternary_decode_packet(bad_pkt, ZCRUDP_TERNARY_PKT_HDR_SZ + 4U, can_loss, NULL, len, &cfg_q, NULL));

    free(sq.pkts); free(sq.lens); free(sl.pkts); free(sl.lens);
    free(can_loss); free(src); free(ref_q); free(can_q); free(ref_l); free(ed); free(dd);
    printf("PASSED (PSNR %.2f dB, %zu -> %zu B, loss isolated)\n", psnr, bytes_l, bytes_q);
}

static void test_detailed_photo_pan_prediction(void) {
    printf("[TEST] Running Detailed Panning Photo Prediction (Baseline vs TileMV+Q_EC)...\n");
    const uint16_t IW = 480, IH = 270, W = 320, H = 180;
    const size_t ilen = (size_t)IW * IH, len = (size_t)W * H;
    const uint16_t tile_sz = 32U;
    const uint16_t tx_cnt = (W + tile_sz - 1U) / tile_sz;
    const uint16_t ty_cnt = (H + tile_sz - 1U) / tile_sz;
    const size_t num_tiles = (size_t)tx_cnt * ty_cnt;

    uint8_t *photo = (uint8_t *)malloc(ilen);
    uint8_t *src = (uint8_t *)malloc(len);
    uint8_t *ref = (uint8_t *)malloc(len), *ref_mc = (uint8_t *)malloc(len);
    uint8_t *can = (uint8_t *)malloc(len), *can_mc = (uint8_t *)malloc(len);
    uint8_t *ed = (uint8_t *)malloc(len), *ed_mc = (uint8_t *)malloc(len);
    uint8_t *dd = (uint8_t *)malloc(len), *dd_mc = (uint8_t *)malloc(len);
    int8_t *enc_mvs = (int8_t *)calloc(num_tiles * 2U, 1);
    int8_t *dec_mvs = (int8_t *)calloc(num_tiles * 2U, 1);
    assert(photo && src && ref && ref_mc && can && can_mc && ed && ed_mc && dd && dd_mc && enc_mvs && dec_mvs);

    uint32_t rng = 135792468U;
    for (uint16_t y = 0; y < IH; y++) {
        for (uint16_t x = 0; x < IW; x++) {
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            int n = (int)(rng % 9U) - 4;
            int v = 128 + (int)(45.0 * sin(x * 0.19) * cos(y * 0.17) + 35.0 * sin((x + y) * 0.43)) + n;
            if (((x / 31) + (y / 37)) % 3 == 0) v += 50;
            photo[(size_t)y * IW + x] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    }

    const int pans[2] = { 1, 3 };
    for (int pi = 0; pi < 2; pi++) {
        int pan = pans[pi];
        zcrudp_ternary_config_t cfg = {
            .width = W, .height = H, .pixfmt = ZCRUDP_TERNARY_FMT_GRAY8,
            .delta_min = 3, .delta_max = 32, .threshold = 3,
            .gamma_q8 = 254, .neutral_y = 128, .mtu_payload = 1380,
            .entropy = ZCRUDP_TERNARY_EC_ALL, .intra_qstep = 4
        };
        pkt_sink_t st = { (uint8_t *)malloc(1024U * 1400U), (size_t *)malloc(1024U * sizeof(size_t)), 0, 1024, 1400, 0 };
        assert(st.pkts && st.lens);
        uint8_t pkt_buf[1400];

        memset(ref, 128, len); memset(can, 128, len);
        memset(ed, cfg.delta_min, len); memset(dd, cfg.delta_min, len);

        double sum_psnr = 0.0;
        const int FRAMES = 16;
        for (int f = 0; f < FRAMES; f++) {
            int ox = f * pan, oy = (f * pan) / 2;
            for (uint16_t y = 0; y < H; y++) {
                memcpy(src + (size_t)y * W, photo + (size_t)(y + oy) * IW + (size_t)ox, W);
            }
            st.count = 0;
            if (f == 0) {
                zcrudp_ternary_encode_intra_range(src, ref, ed, 0, len, len, 0, 0, &cfg,
                                                  pkt_buf, sink_cb, &st);
            } else {
                zcrudp_ternary_estimate_tile_mv(src, ref, W, H, tile_sz, 16, enc_mvs);
                zcrudp_ternary_encode_tile_mv_pkts(enc_mvs, num_tiles, 0, (uint16_t)f, &cfg,
                                                   pkt_buf, sink_cb, &st);
                zcrudp_ternary_apply_tile_mv_state(ref, ref_mc, ed, ed_mc, W, H, tile_sz, enc_mvs);
                memcpy(ref, ref_mc, len);
                memcpy(ed, ed_mc, len);
                zcrudp_ternary_encode_plane(src, ref, ed, len, 0, (uint16_t)f, &cfg,
                                            pkt_buf, sink_cb, &st);
                zcrudp_ternary_repair_disocclusions(src, ref, ed, len, 16, 0, (uint16_t)f, &cfg,
                                                    pkt_buf, sink_cb, &st);
            }

            size_t pidx = 0;
            if (f > 0) {
                while (pidx < st.count) {
                    const uint8_t *p = st.pkts + pidx * st.mtu;
                    zcrudp_ternary_pkt_hdr_t h;
                    if (!zcrudp_ternary_pkt_hdr_read_le(p, &h) ||
                        ZCRUDP_TERNARY_HDR_TYPE(&h) != ZCRUDP_TERNARY_PKT_TILE_MV) {
                        break;
                    }
                    assert(zcrudp_ternary_decode_tile_mv_pkt(p, st.lens[pidx],
                                                             dec_mvs, num_tiles));
                    pidx++;
                }
                zcrudp_ternary_apply_tile_mv_state(can, can_mc, dd, dd_mc, W, H, tile_sz, dec_mvs);
                memcpy(can, can_mc, len);
                memcpy(dd, dd_mc, len);
            }
            zcrudp_ternary_decay_plane(can, len, cfg.gamma_q8, cfg.neutral_y);
            for (; pidx < st.count; pidx++) {
                assert(zcrudp_ternary_decode_packet(st.pkts + pidx * st.mtu, st.lens[pidx],
                                                    can, dd, len, &cfg, NULL));
            }
            assert(memcmp(ref, can, len) == 0);
            assert(memcmp(ed, dd, len) == 0);

            if (f >= 4) {
                double mse = 0.0;
                for (size_t i = 0; i < len; i++) {
                    double d = (double)src[i] - (double)can[i];
                    mse += d * d;
                }
                sum_psnr += 10.0 * log10((255.0 * 255.0) / (mse / (double)len));
            }
        }
        double avg_psnr = sum_psnr / (FRAMES - 4);
        assert(avg_psnr >= 38.0);
        printf("       pan %d px/frame -> avg Y-PSNR %.2f dB\n", pan, avg_psnr);
        free(st.pkts); free(st.lens);
    }

    free(photo); free(src); free(ref); free(ref_mc); free(can); free(can_mc);
    free(ed); free(ed_mc); free(dd); free(dd_mc); free(enc_mvs); free(dec_mvs);
    printf("       PASSED\n");
}

int main(void) {
    printf("=== zcrudp Ternary Codec Test & 1080p120 Benchmark ===\n");
#ifdef __AVX2__
    printf("SIMD: AVX2 ENABLED\n\n");
#else
    printf("SIMD: Scalar C11\n\n");
#endif

    test_decay_matches_scalar_any_neutral();
#ifdef __AVX2__
    test_avx2_unpack_matches_scalar();
    test_avx2_block_matches_scalar();
#endif
    test_nv12_and_rgba_roundtrip();
    test_gmc_panoramic_savings();
    test_intra_refresh_and_2x_mode();
    test_nv12_chroma_2x_keeps_components();
    test_entropy_matches_tokens();
    test_entropy_packet_loss_is_isolated();
    test_malformed_packets_stay_in_bounds();
    test_entropy_cpu_cap();
    test_mt_decode_keeps_packet_order();
    test_intra_q_ec_roundtrip_and_bounds();
    test_detailed_photo_pan_prediction();
    test_3d_complex_scene_psnr(0U);
    test_3d_complex_scene_psnr(ZCRUDP_TERNARY_EC_ALL);
    test_tile_mv_3d_parallax_35db(0U);
    test_tile_mv_3d_parallax_35db(ZCRUDP_TERNARY_EC_ALL);
    test_entropy_rate_quality();
    printf("\n[BENCHMARK] 1920x1080 @ 120 Hz -> 360 Hz (budget 2.778 ms/frame @ 360 Hz):\n");
    bench_1080p120_format(ZCRUDP_TERNARY_FMT_NV12, "NV12", 0U);
    bench_1080p120_format(ZCRUDP_TERNARY_FMT_NV12, "NV12", ZCRUDP_TERNARY_EC_ALL);
    bench_1080p120_format(ZCRUDP_TERNARY_FMT_RGBA, "RGBA", 0U);
    bench_1080p120_format(ZCRUDP_TERNARY_FMT_RGBA, "RGBA", ZCRUDP_TERNARY_EC_ALL);
    bench_1080p120_mt(ZCRUDP_TERNARY_FMT_NV12, "NV12", 8, 0U);
    bench_1080p120_mt(ZCRUDP_TERNARY_FMT_NV12, "NV12", 8, ZCRUDP_TERNARY_EC_REALLOC_V2);
    bench_1080p120_mt(ZCRUDP_TERNARY_FMT_NV12, "NV12", 8, ZCRUDP_TERNARY_EC_ALL);
    bench_1080p120_mt(ZCRUDP_TERNARY_FMT_NV12, "NV12", 8, ZCRUDP_TERNARY_EC_RANS);
    bench_1080p120_mt(ZCRUDP_TERNARY_FMT_RGBA, "RGBA", 8, 0U);
    bench_1080p120_mt(ZCRUDP_TERNARY_FMT_RGBA, "RGBA", 8, ZCRUDP_TERNARY_EC_ALL);
    bench_1080p120_mt(ZCRUDP_TERNARY_FMT_RGBA, "RGBA", 8, ZCRUDP_TERNARY_EC_RANS);

    printf("\n=== ALL CODEC TESTS PASSED ===\n");
    return 0;
}
