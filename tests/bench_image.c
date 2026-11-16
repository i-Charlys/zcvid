/**
 * @file bench_image.c
 * @brief Token vs entropy-coded size on a real picture.
 *
 * The unit tests use generated scenes. This tool takes a photograph as a
 * binary PGM (P5, 8-bit, at least 1296x736) and plays a 1280x720 window over
 * it, so the numbers reflect camera statistics rather than a smooth texture.
 *
 *   bench_image photo.pgm [pan_px_per_frame=1] [noise_amplitude=1]
 *
 * Any image converts to PGM with e.g. `magick photo.jpg -colorspace Gray photo.pgm`.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "ternary_codec.h"

enum { OUT_W = 1280, OUT_H = 720, FRAMES = 20, WARMUP = 4, INTRA_CYCLE = 24 };

typedef struct {
    uint8_t (*pkts)[1400];
    size_t *lens;
    size_t count, max;
    size_t delta_bytes, anchor_bytes;
} sink_t;

static void sink_cb(const uint8_t *data, size_t len, void *ctx) {
    sink_t *s = (sink_t *)ctx;
    if (s->count >= s->max || len > 1400U) {
        fprintf(stderr, "packet store overflow\n");
        exit(2);
    }
    memcpy(s->pkts[s->count], data, len);
    s->lens[s->count++] = len;
    uint8_t type = (uint8_t)((data[2] >> 4) & 0x0FU);
    if (type == ZCRUDP_TERNARY_PKT_INTRA || type == ZCRUDP_TERNARY_PKT_INTRA_EC ||
        type == ZCRUDP_TERNARY_PKT_INTRA_Q_EC) {
        s->anchor_bytes += len;
    } else {
        s->delta_bytes += len;
    }
}

static double now_ms(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static int pgm_token(FILE *f, int *out) {
    int c = fgetc(f);
    while (c == '#' || c == ' ' || c == '\n' || c == '\r' || c == '\t') {
        if (c == '#') {
            while (c != '\n' && c != EOF) c = fgetc(f);
        } else {
            c = fgetc(f);
        }
    }
    if (c < '0' || c > '9') return 0;
    int v = 0;
    while (c >= '0' && c <= '9') {
        v = v * 10 + (c - '0');
        c = fgetc(f);
    }
    *out = v;
    return 1;
}

static uint8_t *load_pgm(const char *path, int *w, int *h) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    int maxv = 0;
    if (fgetc(f) != 'P' || fgetc(f) != '5' || !pgm_token(f, w) || !pgm_token(f, h) ||
        !pgm_token(f, &maxv) || maxv != 255 || *w <= 0 || *h <= 0) {
        fclose(f);
        return NULL;
    }
    size_t n = (size_t)*w * (size_t)*h;
    uint8_t *px = (uint8_t *)malloc(n);
    if (px && fread(px, 1, n, f) != n) {
        free(px);
        px = NULL;
    }
    fclose(f);
    return px;
}

typedef struct { double delta_bpp, anchor_bpp, mv_bpp, key_bpp, psnr, enc_ms, dec_ms; } result_t;

static result_t run(const uint8_t *img, int iw, int ih, int pan, int noise,
                    uint8_t entropy, uint8_t intra_qstep, bool use_tile_mv,
                    bool use_repair, uint8_t *final_ref) {
    zcrudp_ternary_config_t cfg = {
        .width = OUT_W, .height = OUT_H, .pixfmt = ZCRUDP_TERNARY_FMT_GRAY8,
        .delta_min = 3, .delta_max = 32, .threshold = 3,
        .gamma_q8 = 254, .neutral_y = 128, .mtu_payload = 1380,
        .entropy = entropy, .intra_qstep = intra_qstep
    };
    const size_t len = (size_t)OUT_W * OUT_H;
    const uint16_t tile_sz = 32U;
    const uint16_t tx_cnt = (OUT_W + tile_sz - 1U) / tile_sz;
    const uint16_t ty_cnt = (OUT_H + tile_sz - 1U) / tile_sz;
    const size_t num_tiles = (size_t)tx_cnt * (size_t)ty_cnt;

    uint8_t *src = (uint8_t *)malloc(len), *ref = (uint8_t *)malloc(len);
    uint8_t *ref_mc = (uint8_t *)malloc(len);
    uint8_t *canvas = (uint8_t *)malloc(len), *canvas_mc = (uint8_t *)malloc(len);
    uint8_t *enc_delta = (uint8_t *)malloc(len), *enc_delta_mc = (uint8_t *)malloc(len);
    uint8_t *dec_delta = (uint8_t *)malloc(len), *dec_delta_mc = (uint8_t *)malloc(len);
    int8_t *enc_mvs = (int8_t *)calloc(num_tiles * 2U, 1);
    int8_t *dec_mvs = (int8_t *)calloc(num_tiles * 2U, 1);
    sink_t s = { malloc(4096U * 1400U), malloc(4096U * sizeof(size_t)), 0, 4096, 0, 0 };
    uint8_t pkt_buf[1400];
    if (!src || !ref || !ref_mc || !canvas || !canvas_mc ||
        !enc_delta || !enc_delta_mc || !dec_delta || !dec_delta_mc ||
        !enc_mvs || !dec_mvs || !s.pkts || !s.lens) exit(2);
    memset(ref, 128, len);
    memset(canvas, 128, len);
    memset(enc_delta, cfg.delta_min, len);
    memset(dec_delta, cfg.delta_min, len);

    result_t r = { 0 };
    size_t mv_bytes = 0;
    uint32_t rng = 2463534242U;
    const int max_x = iw - OUT_W, max_y = ih - OUT_H;
    const size_t slice = (len + INTRA_CYCLE - 1U) / INTRA_CYCLE;

    for (int f = 0; f < FRAMES; f++) {
        int ox = (max_x > 0) ? ((f * pan) % (max_x + 1)) : 0;
        int oy = (max_y > 0) ? ((f * pan / 2) % (max_y + 1)) : 0;
        for (int y = 0; y < OUT_H; y++) {
            const uint8_t *row = img + (size_t)(y + oy) * (size_t)iw + (size_t)ox;
            for (int x = 0; x < OUT_W; x++) {
                int v = row[x];
                if (noise > 0) {
                    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                    v += (int)(rng % (uint32_t)(2 * noise + 1)) - noise;
                }
                src[(size_t)y * OUT_W + (size_t)x] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
            }
        }

        s.count = 0;
        size_t d0 = s.delta_bytes, a0 = s.anchor_bytes, m0 = mv_bytes;
        double t0 = now_ms();
        if (f == 0) {
            zcrudp_ternary_encode_intra_range(src, ref, enc_delta, 0, len, len, 0, 0, &cfg,
                                              pkt_buf, sink_cb, &s);
        } else {
            if (use_tile_mv) {
                zcrudp_ternary_estimate_tile_mv(src, ref, OUT_W, OUT_H, tile_sz, 16, enc_mvs);
                mv_bytes += zcrudp_ternary_encode_tile_mv_pkts(enc_mvs, num_tiles, 0, (uint16_t)f,
                                                               &cfg, pkt_buf, sink_cb, &s);
                zcrudp_ternary_apply_tile_mv_state(ref, ref_mc, enc_delta, enc_delta_mc,
                                                   OUT_W, OUT_H, tile_sz, enc_mvs);
                memcpy(ref, ref_mc, len);
                memcpy(enc_delta, enc_delta_mc, len);
            }
            zcrudp_ternary_encode_plane(src, ref, enc_delta, len, 0, (uint16_t)f, &cfg,
                                        pkt_buf, sink_cb, &s);
            if (use_repair) {
                zcrudp_ternary_repair_disocclusions(src, ref, enc_delta, len, 16, 0, (uint16_t)f,
                                                    &cfg, pkt_buf, sink_cb, &s);
            } else {
                size_t i0 = (size_t)(f % INTRA_CYCLE) * slice;
                zcrudp_ternary_encode_intra_range(src, ref, enc_delta, i0, i0 + slice, len, 0,
                                                  (uint16_t)f, &cfg, pkt_buf, sink_cb, &s);
            }
        }
        double t1 = now_ms();

        size_t pkt_idx = 0;
        if (f > 0 && use_tile_mv) {
            while (pkt_idx < s.count) {
                zcrudp_ternary_pkt_hdr_t hdr;
                if (!zcrudp_ternary_pkt_hdr_read_le(s.pkts[pkt_idx], &hdr) ||
                    ZCRUDP_TERNARY_HDR_TYPE(&hdr) != ZCRUDP_TERNARY_PKT_TILE_MV) {
                    break;
                }
                if (!zcrudp_ternary_decode_tile_mv_pkt(s.pkts[pkt_idx], s.lens[pkt_idx],
                                                       dec_mvs, num_tiles)) {
                    fprintf(stderr, "tile mv decode failed\n");
                    exit(2);
                }
                pkt_idx++;
            }
            zcrudp_ternary_apply_tile_mv_state(canvas, canvas_mc, dec_delta, dec_delta_mc,
                                               OUT_W, OUT_H, tile_sz, dec_mvs);
            memcpy(canvas, canvas_mc, len);
            memcpy(dec_delta, dec_delta_mc, len);
        }

        zcrudp_ternary_decay_plane(canvas, len, cfg.gamma_q8, cfg.neutral_y);
        for (; pkt_idx < s.count; pkt_idx++) {
            if (!zcrudp_ternary_decode_packet(s.pkts[pkt_idx], s.lens[pkt_idx],
                                              canvas, dec_delta, len, &cfg, NULL)) {
                fprintf(stderr, "decode failed\n");
                exit(2);
            }
        }
        double t2 = now_ms();
        if (memcmp(ref, canvas, len) != 0) {
            fprintf(stderr, "encoder and decoder references differ at frame %d\n", f);
            exit(2);
        }
        if (f == 0) {
            r.key_bpp = (double)s.anchor_bytes * 8.0 / (double)len;
        } else if (f >= WARMUP) {
            double mse = 0.0;
            for (size_t i = 0; i < len; i++) {
                double d = (double)src[i] - (double)canvas[i];
                mse += d * d;
            }
            mse /= (double)len;
            r.psnr += 10.0 * log10(65025.0 / (mse > 1e-9 ? mse : 1e-9));
            r.delta_bpp += (double)(s.delta_bytes - d0) * 8.0 / (double)len;
            r.anchor_bpp += (double)(s.anchor_bytes - a0) * 8.0 / (double)len;
            r.mv_bpp += (double)(mv_bytes - m0) * 8.0 / (double)len;
            r.enc_ms += t1 - t0;
            r.dec_ms += t2 - t1;
        }
    }
    const double n = (double)(FRAMES - WARMUP);
    r.psnr /= n; r.delta_bpp /= n; r.anchor_bpp /= n; r.mv_bpp /= n;
    r.enc_ms /= n; r.dec_ms /= n;
    if (final_ref) memcpy(final_ref, ref, len);
    free(src); free(ref); free(ref_mc); free(canvas); free(canvas_mc);
    free(enc_delta); free(enc_delta_mc); free(dec_delta); free(dec_delta_mc);
    free(enc_mvs); free(dec_mvs); free(s.pkts); free(s.lens);
    return r;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s photo.pgm [pan_px_per_frame=1] [noise_amplitude=1]\n", argv[0]);
        return 1;
    }
    int pan = (argc > 2) ? atoi(argv[2]) : 1;
    int noise = (argc > 3) ? atoi(argv[3]) : 1;
    int iw = 0, ih = 0;
    uint8_t *img = load_pgm(argv[1], &iw, &ih);
    if (!img || iw < OUT_W + 16 || ih < OUT_H + 16) {
        fprintf(stderr, "%s: need a binary 8-bit PGM of at least %dx%d\n", argv[1], OUT_W + 16, OUT_H + 16);
        return 1;
    }

    const size_t len = (size_t)OUT_W * OUT_H;
    uint8_t *ref_tok = (uint8_t *)malloc(len), *ref_ec = (uint8_t *)malloc(len);
    if (!ref_tok || !ref_ec) return 2;
    result_t tok = run(img, iw, ih, pan, noise, 0U, 0U, false, false, ref_tok);
    result_t ec  = run(img, iw, ih, pan, noise, ZCRUDP_TERNARY_EC_ALL, 0U, false, false, ref_ec);
    if (memcmp(ref_tok, ref_ec, len) != 0) {
        fprintf(stderr, "token and entropy-coded pictures differ\n");
        return 2;
    }
    result_t qrepair = run(img, iw, ih, pan, noise, ZCRUDP_TERNARY_EC_ALL, 4U, false, true, NULL);
    result_t tilemv  = run(img, iw, ih, pan, noise, ZCRUDP_TERNARY_EC_ALL, 4U, true,  true, NULL);
    result_t tilerans = run(img, iw, ih, pan, noise, ZCRUDP_TERNARY_EC_RANS | ZCRUDP_TERNARY_EC_INTRA, 4U, true, true, NULL);

    printf("%s (%dx%d), 1280x720 window, pan %d px/frame, noise +-%d\n",
           argv[1], iw, ih, pan, noise);
    printf("  [1] baseline (no MC, rolling)  : Y-PSNR %.2f dB | key %.3f -> %.3f bpp | deltas %.3f -> %.3f bpp | anchors %.3f -> %.3f bpp | enc/dec %.2f/%.2f ms -> %.2f/%.2f ms\n",
           ec.psnr, tok.key_bpp, ec.key_bpp, tok.delta_bpp, ec.delta_bpp, tok.anchor_bpp, ec.anchor_bpp,
           tok.enc_ms, tok.dec_ms, ec.enc_ms, ec.dec_ms);
    printf("  [2] no-MC + INTRA_Q_EC (q=4)   : Y-PSNR %.2f dB | total %.3f bpp (deltas %.3f + repair %.3f) | enc/dec %.2f/%.2f ms\n",
           qrepair.psnr, qrepair.delta_bpp + qrepair.anchor_bpp, qrepair.delta_bpp, qrepair.anchor_bpp,
           qrepair.enc_ms, qrepair.dec_ms);
    printf("  [3] TileMV + EC + Q_EC (q=4)   : Y-PSNR %.2f dB | total %.3f bpp (mv %.3f + deltas %.3f + repair %.3f) | enc/dec %.2f/%.2f ms\n",
           tilemv.psnr, tilemv.mv_bpp + tilemv.delta_bpp + tilemv.anchor_bpp,
           tilemv.mv_bpp, tilemv.delta_bpp, tilemv.anchor_bpp, tilemv.enc_ms, tilemv.dec_ms);
    printf("  [4] TileMV + rANS + Q_EC (q=4) : Y-PSNR %.2f dB | total %.3f bpp (mv %.3f + deltas %.3f + repair %.3f) | enc/dec %.2f/%.2f ms\n",
           tilerans.psnr, tilerans.mv_bpp + tilerans.delta_bpp + tilerans.anchor_bpp,
           tilerans.mv_bpp, tilerans.delta_bpp, tilerans.anchor_bpp, tilerans.enc_ms, tilerans.dec_ms);
    free(img); free(ref_tok); free(ref_ec);
    return 0;
}
