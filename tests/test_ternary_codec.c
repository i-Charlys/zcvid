/**
 * @file test_ternary_codec.c
 * @brief Unit tests & 1080p120 latency/bandwidth benchmark for ternary_codec.h
 *        (NV12 & RGBA, Jayant ADM, RLE zero-skip, AVX2/Scalar).
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

static void bench_1080p120_format(zcrudp_ternary_pixfmt_t fmt, const char *name) {
    zcrudp_ternary_config_t cfg = {
        .width = 1920,
        .height = 1080,
        .pixfmt = fmt,
        .delta_min = 4,
        .delta_max = 32,
        .threshold = 6,
        .gamma_q8 = 250,
        .neutral_y = 128,
        .mtu_payload = 1380
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

    printf("  -> %-5s 1080p120 (~5%% active): Enc %.3f ms | Dec %.3f ms | %zu pkts/f | %.2f Mbps\n",
           name, enc_ms, dec_ms, last_pkts, mbps_120);

    free(pkt_store);
    free(src);
    free(enc_ref);
    free(dec_canvas);
}

static void bench_1080p120_mt(zcrudp_ternary_pixfmt_t fmt, const char *name, uint32_t threads) {
    zcrudp_ternary_config_t cfg = {
        .width = 1920,
        .height = 1080,
        .pixfmt = fmt,
        .delta_min = 4,
        .delta_max = 32,
        .threshold = 6,
        .gamma_q8 = 250,
        .neutral_y = 128,
        .mtu_payload = 1380
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

    printf("  -> %-5s 1080p (%uT MT) : Enc %.3f ms | Dec %.3f ms | Total %.3f ms | %zu pkts/f | %.2f Mbps @120Hz | %.2f Mbps @360Hz (%.1f%% of 2.78ms budget)\n",
           name, threads, enc_ms, dec_ms, total_ms, last_pkts, mbps_120, mbps_360, budget_360_pct);

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

static void test_3d_complex_scene_psnr(void) {
    printf("[TEST] Running Complex 3D-like Textured + Pan + HUD PSNR Test... ");

    zcrudp_ternary_config_t cfg = {
        .width = 640,
        .height = 360,
        .pixfmt = ZCRUDP_TERNARY_FMT_GRAY8,
        .delta_min = 3,
        .delta_max = 32,
        .threshold = 4,
        .gamma_q8 = 254,
        .neutral_y = 128,
        .mtu_payload = 1380
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

            /* 1x Delta + Rolling Intra-Refresh 1/24th per frame */
            size_t i0 = (size_t)f * slice_bytes;
            zcrudp_ternary_encode_plane(src, enc_ref, enc_delta, len,
                                        0, (uint16_t)f, &cfg, pkt_buf, sink_cb, &sink);
            zcrudp_ternary_encode_intra_range(src, enc_ref, enc_delta, i0, i0 + slice_bytes,
                                              len, 0, (uint16_t)f, &cfg, pkt_buf, sink_cb, &sink);
        }

        zcrudp_ternary_decay_plane(dec_canvas, len, cfg.gamma_q8, cfg.neutral_y);
        for (size_t p = 0; p < sink.count; p++) {
            zcrudp_ternary_decode_packet(sink.pkts + p * sink.mtu, sink.lens[p],
                                         dec_canvas, dec_delta, len, &cfg, NULL);
        }

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
    assert(avg_psnr > 15.0);

    printf("PASSED\n");

    free(pkt_store);
    free(src);
    free(enc_ref);
    free(enc_delta);
    free(dec_canvas);
    free(dec_delta);
}

static void test_tile_mv_3d_parallax_35db(void) {
    printf("[TEST] Running 64x64 Tile MV + Luma-First RC + Bridge (35+ dB target)... ");

    zcrudp_ternary_config_t cfg = {
        .width = 640,
        .height = 360,
        .pixfmt = ZCRUDP_TERNARY_FMT_NV12,
        .delta_min = 4,
        .delta_max = 48,
        .threshold = 2,
        .gamma_q8 = 255,
        .neutral_y = 128,
        .mtu_payload = 1380
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
            zcrudp_ternary_repair_disocclusions(src, enc_ref, enc_delta, y_sz, 20U,
                                                0, (uint16_t)f, &cfg, pkt_buf, sink_cb, &sink);
            zcrudp_ternary_encode_nv12_rc(src, enc_ref, enc_delta, (uint16_t)f,
                                          25000U, 20U, &cfg, pkt_buf, sink_cb, &sink);
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

int main(void) {
    printf("=== zcrudp Ternary Codec Test & 1080p120 Benchmark ===\n");
#ifdef __AVX2__
    printf("SIMD: AVX2 ENABLED\n\n");
#else
    printf("SIMD: Scalar C11\n\n");
#endif

    test_nv12_and_rgba_roundtrip();
    test_gmc_panoramic_savings();
    test_intra_refresh_and_2x_mode();
    test_nv12_chroma_2x_keeps_components();
    test_3d_complex_scene_psnr();
    test_tile_mv_3d_parallax_35db();
    printf("\n[BENCHMARK] 1920x1080 @ 120 Hz -> 360 Hz (budget 2.778 ms/frame @ 360 Hz):\n");
    bench_1080p120_format(ZCRUDP_TERNARY_FMT_NV12, "NV12");
    bench_1080p120_format(ZCRUDP_TERNARY_FMT_RGBA, "RGBA");
    bench_1080p120_mt(ZCRUDP_TERNARY_FMT_NV12, "NV12", 8);
    bench_1080p120_mt(ZCRUDP_TERNARY_FMT_RGBA, "RGBA", 8);

    printf("\n=== ALL CODEC TESTS PASSED ===\n");
    return 0;
}
