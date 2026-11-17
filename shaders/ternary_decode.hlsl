// ternary_decode.hlsl - D3D11 Compute Shader (cs_5_0)
// Full-Color (YCbCr) + 32x32 Tile MV + Jayant ADM + Intra-Refresh Decoder.
//
// Pass 1: CSMotionDecay -> Applies Tile MV from g_prev_yuvd + Leaky Decay into g_cur_yuvd
// Pass 2: CSApply       -> Applies received Delta (Y/Cb/Cr) & Intra records
// Pass 3: CSYCbCrToRGBA -> Converts g_cur_yuvd to Swapchain RGBA Texture2D

cbuffer TernaryDecParams : register(b0) {
    uint width;
    uint height;
    uint delta_min;
    uint delta_max;
    uint gamma_q8;
    uint neutral_y;
    uint record_words;   // Total uint32 words in g_in_records (2 words per record)
    uint tiles_x;
    uint mv_qpel;        // 1: tile vectors are in quarter pels
    uint post_thr;       // display deblocking: largest step smoothed (0 = off)
    uint dpad1;
    uint dpad2;
};

StructuredBuffer<int2>    g_tile_mvs   : register(t0);
StructuredBuffer<uint>    g_in_records : register(t1);
StructuredBuffer<uint>    g_prev_yuvd  : register(t2);
RWStructuredBuffer<uint>  g_cur_yuvd   : register(u0);
RWTexture2D<unorm float4> g_out_rgba   : register(u1);
RWStructuredBuffer<uint>  g_coef       : register(u2); // transform intra: 48 dequantized coefficients per 4x4 block
RWStructuredBuffer<uint>  g_tflag      : register(u3); // transform intra: 1 per block with coefficients this frame

// -------------------------------------------------------------------------
// 4x4 integer transform intra (shared, word for word, by ternary_encode.hlsl
// and ternary_decode.hlsl: the reconstruction must be bit-exact on both
// sides, on any GPU, so everything is integer).
//
// The H.264 core transform and its quantizer: forward Cf.X.Cf^T, level =
// (|W| * MF + f) >> (15 + qp / 6); dequantized W' = level * V << (qp / 6);
// inverse with the >>1 butterflies and a final (x + 32) >> 6. Samples are
// coded around 128 (no prediction), so a block is self-contained: a refresh
// repairs whatever the decoder held before.
//
// Record: w0 = 0xF0000000 | (block * 48 + plane * 16 + k)   (k: zigzag index)
//         w1 = (level & 0xFFFF) | (qp << 16)                (level in [-255, 255])
// -------------------------------------------------------------------------
static const uint TQ_ZZ[16] = { 0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15 };
static const int TQ_MF[18] = { 13107, 5243, 8066,  11916, 4660, 7490,  10082, 4194, 6554,
                               9362, 3647, 5825,   8192, 3355, 5243,   7282, 2893, 4559 };
static const int TQ_V[18]  = { 10, 16, 13,  11, 18, 14,  13, 20, 16,
                               14, 23, 18,  16, 25, 20,  18, 29, 23 };

// 0: both coordinates even, 1: both odd, 2: mixed.
uint TqClass(uint r) {
    uint i = r >> 2;
    uint j = r & 3u;
    if (((i | j) & 1u) == 0u) return 0u;
    if ((i & j & 1u) != 0u) return 1u;
    return 2u;
}

int TqDequant(int level, uint r, uint qp) {
    return (level * TQ_V[(qp % 6u) * 3u + TqClass(r)]) << (qp / 6u);
}

// In place, raster order: dequantized coefficients in, residual out.
void TqInverse(inout int c[16]) {
    [unroll] for (uint i = 0; i < 4u; i++) {
        int d0 = c[i * 4u], d1 = c[i * 4u + 1u], d2 = c[i * 4u + 2u], d3 = c[i * 4u + 3u];
        int e0 = d0 + d2, e1 = d0 - d2, e2 = (d1 >> 1) - d3, e3 = d1 + (d3 >> 1);
        c[i * 4u] = e0 + e3; c[i * 4u + 1u] = e1 + e2; c[i * 4u + 2u] = e1 - e2; c[i * 4u + 3u] = e0 - e3;
    }
    [unroll] for (uint j = 0; j < 4u; j++) {
        int d0 = c[j], d1 = c[4u + j], d2 = c[8u + j], d3 = c[12u + j];
        int e0 = d0 + d2, e1 = d0 - d2, e2 = (d1 >> 1) - d3, e3 = d1 + (d3 >> 1);
        c[j] = e0 + e3; c[4u + j] = e1 + e2; c[8u + j] = e1 - e2; c[12u + j] = e0 - e3;
    }
    [unroll] for (uint k = 0; k < 16u; k++) c[k] = (c[k] + 32) >> 6;
}

int LeakyDecay(int prev_val, int neutral) {
    int diff = prev_val - neutral;
    return clamp(neutral + (diff * (int)gamma_q8) / 256, 0, 255);
}


// -------------------------------------------------------------------------
// Motion-compensated reference with a quarter-pel vector (shared, word for
// word, by ternary_encode.hlsl and ternary_decode.hlsl). Y, Cb and Cr are
// interpolated in integers (4-tap, see below), so both sides get the same value on
// any GPU; the adaptive step comes from the integer-pel position. A vector
// that is a multiple of 4 reads exactly one reference pixel, as before.
// -------------------------------------------------------------------------
uint RefWordAt(int x, int y) {
    x = clamp(x, 0, (int)width - 1);
    y = clamp(y, 0, (int)height - 1);
    return g_prev_yuvd[(uint)y * width + (uint)x];
}

// Taps (sum 64) for a quarter-pel phase; phase 0 is the identity.
int4 TapsQ(int f) {
    if (f == 1) return int4(-4, 54, 16, -2);
    if (f == 2) return int4(-4, 36, 36, -4);
    if (f == 3) return int4(-2, 16, 54, -4);
    return int4(0, 64, 0, 0);
}

uint FetchRefQ(uint px, uint py, int2 mvq) {
    int sxq = (int)px * 4 - mvq.x;
    int syq = (int)py * 4 - mvq.y;
    int ix = sxq >> 2;
    int iy = syq >> 2;
    int fx = sxq & 3;
    int fy = syq & 3;
    uint w00 = RefWordAt(ix, iy);
    if ((fx | fy) == 0) return w00;
    // Separable 4-tap filter (the HEVC chroma taps at 1/4, 1/2, 3/4): sharper
    // than bilinear, whose blur would build up frame after frame through the
    // closed loop. Full precision between the passes, one rounding at the end.
    int4 tx = TapsQ(fx);
    int4 ty = TapsQ(fy);
    int acc0 = 0, acc1 = 0, acc2 = 0;
    [loop] for (int j = 0; j < 4; j++) {
        int tv = (j == 0) ? ty.x : (j == 1) ? ty.y : (j == 2) ? ty.z : ty.w;
        int h0 = 0, h1 = 0, h2 = 0;
        [unroll] for (int i = 0; i < 4; i++) {
            uint w = RefWordAt(ix - 1 + i, iy - 1 + j);
            int t = (i == 0) ? tx.x : (i == 1) ? tx.y : (i == 2) ? tx.z : tx.w;
            h0 += t * (int)(w & 0xFFu);
            h1 += t * (int)((w >> 8) & 0xFFu);
            h2 += t * (int)((w >> 16) & 0xFFu);
        }
        acc0 += tv * h0;
        acc1 += tv * h1;
        acc2 += tv * h2;
    }
    return (w00 & 0xFF000000u) |
           (uint)clamp((acc0 + 2048) >> 12, 0, 255) |
           ((uint)clamp((acc1 + 2048) >> 12, 0, 255) << 8) |
           ((uint)clamp((acc2 + 2048) >> 12, 0, 255) << 16);
}

// Motion-compensated reference word of a pixel (previous frame, tile MV applied).
uint PrevShifted(uint idx) {
    uint py = idx / width;
    uint px = idx - py * width;
    int2 mv = g_tile_mvs[(py >> 5) * tiles_x + (px >> 5)];
    return FetchRefQ(px, py, (mv_qpel != 0u) ? mv : mv * 4);
}

int StepOf(uint ref_w) {
    return max((int)delta_min, (int)((ref_w >> 24) & 0xFFu));
}

// =========================================================================
// Pass 1: Motion-Compensated Copy + Leaky Decay (1 thread per pixel)
// This is the "all trits are zero" reconstruction, so the adaptive step is
// halved here; CSApply overrides it where a non-zero luma trit arrives.
// =========================================================================
[numthreads(16, 16, 1)]
void CSMotionDecay(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= width || dtid.y >= height) return;

    uint idx = dtid.y * width + dtid.x;
    uint w = PrevShifted(idx);
    int y  = LeakyDecay((int)(w & 0xFFu), (int)neutral_y);
    int cb = LeakyDecay((int)((w >> 8) & 0xFFu), 128);
    int cr = LeakyDecay((int)((w >> 16) & 0xFFu), 128);
    uint d = (uint)max((int)delta_min, StepOf(w) >> 1);

    g_cur_yuvd[idx] = (uint)y | ((uint)cb << 8) | ((uint)cr << 16) | (d << 24);
}

// =========================================================================
// Pass 2: Apply Delta (Y/Cb/Cr) & Intra Records
// Values are rebuilt from the read-only previous frame, never from
// g_cur_yuvd, so a duplicated record is idempotent and threads working on
// different planes of the same pixel only touch their own bytes.
// =========================================================================
void ApplyTrit(uint idx, int t, int d_used, uint tag, uint ref_w) {
    // No early return: the caller's loop must stay unrollable by fxc.
    if (t != 0 && idx < width * height) {
        uint shift = tag * 8u;
        int neutral = (tag == 0u) ? (int)neutral_y : 128;
        int prev = (int)((ref_w >> shift) & 0xFFu);
        int v = clamp(LeakyDecay(prev, neutral) + t * d_used, 0, 255);

        uint clear_mask = ~(0xFFu << shift);
        uint set_bits = (uint)v << shift;
        if (tag == 0u) {
            clear_mask &= 0x00FFFFFFu;
            set_bits |= (uint)min((int)delta_max, d_used << 1) << 24;
        }
        InterlockedAnd(g_cur_yuvd[idx], clear_mask);
        InterlockedOr(g_cur_yuvd[idx], set_bits);
    }
}

[numthreads(64, 1, 1)]
void CSApply(uint3 dtid : SV_DispatchThreadID) {
    uint rec_idx = dtid.x * 2u;
    if (rec_idx + 1u >= record_words) return;

    uint w0 = g_in_records[rec_idx + 0u];
    uint w1 = g_in_records[rec_idx + 1u];
    uint tag   = (w0 >> 30) & 3u;
    uint is_2x = (w0 >> 29) & 1u;
    uint id    = w0 & 0x1FFFFFFFu;
    uint num_px = width * height;

    if (tag == 3u) {
        if (is_2x != 0u && ((w0 >> 28) & 1u) != 0u) {
            // Transform intra coefficient: stored dequantized, CSITrans rebuilds the block.
            uint pid = w0 & 0x0FFFFFFFu;
            uint b = pid / 48u;
            uint qp = (w1 >> 16) & 0xFFu;
            uint pred = (w1 >> 24) & 0xFFu;
            uint nblocks = ((width + 3u) >> 2) * ((height + 3u) >> 2);
            if (b >= nblocks || qp > 51u || pred > 1u) return;
            int level = (int)(w1 << 16) >> 16;
            if (level < -255 || level > 255) return;
            uint r = TQ_ZZ[pid % 16u];
            g_coef[pid] = (uint)TqDequant(level, r, qp);
            g_tflag[b] = 1u + pred; // every record of a block carries the same pred

            return;
        }
        if (is_2x != 0u) {
            // 4-pixel 6-bit Intra: w0 = [tag=3|is_4px=1|cr_q6(6)|idx/2(22)], w1 = [y0..y3(24)|cb_q6(6)]
            uint idx4 = (w0 & 0x3FFFFFu) << 1;
            uint cr_q6 = (w0 >> 22) & 0x3Fu;
            uint y0_q6 = w1 & 0x3Fu;
            uint y1_q6 = (w1 >> 6) & 0x3Fu;
            uint y2_q6 = (w1 >> 12) & 0x3Fu;
            uint y3_q6 = (w1 >> 18) & 0x3Fu;
            uint cb_q6 = (w1 >> 24) & 0x3Fu;
            uint cb = (cb_q6 << 2) | (cb_q6 >> 4);
            uint cr = (cr_q6 << 2) | (cr_q6 >> 4);
            uint chroma_d = (cb << 8) | (cr << 16) | (delta_min << 24);
            if (idx4 + 0u < num_px) g_cur_yuvd[idx4 + 0u] = ((y0_q6 << 2) | (y0_q6 >> 4)) | chroma_d;
            if (idx4 + 1u < num_px) g_cur_yuvd[idx4 + 1u] = ((y1_q6 << 2) | (y1_q6 >> 4)) | chroma_d;
            if (idx4 + 2u < num_px) g_cur_yuvd[idx4 + 2u] = ((y2_q6 << 2) | (y2_q6 >> 4)) | chroma_d;
            if (idx4 + 3u < num_px) g_cur_yuvd[idx4 + 3u] = ((y3_q6 << 2) | (y3_q6 >> 4)) | chroma_d;
        } else {
            // 2-pixel 4:2:2 Intra pair: [Y0 | Cb<<8 | Y1<<16 | Cr<<24]
            uint y0 = w1 & 0xFFu;
            uint cb = (w1 >> 8) & 0xFFu;
            uint y1 = (w1 >> 16) & 0xFFu;
            uint cr = (w1 >> 24) & 0xFFu;
            uint chroma_d = (cb << 8) | (cr << 16) | (delta_min << 24);
            if (id < num_px) g_cur_yuvd[id] = y0 | chroma_d;
            if (id + 1u < num_px) g_cur_yuvd[id + 1u] = y1 | chroma_d;
        }
        return;
    }

    // Corrupt word (>= 3^10) or block index past the frame: ignore the record.
    uint px_per_blk = (is_2x != 0u) ? 20u : 10u;
    if (w1 >= 59049u || id >= (num_px + px_per_blk - 1u) / px_per_blk) return;

    uint base_px = id * px_per_blk;
    uint hi = (w1 * 69043u) >> 24;
    uint lo = mad(-hi, 243u, w1);
    uint step_px = (is_2x != 0u) ? 2u : 1u;

    static const uint RECIP[5] = { 21846u, 7282u, 2428u, 810u, 0u };
    uint halves[2] = { lo, hi };

    // Real loops: unrolled, the motion compensation filter of PrevShifted
    // would be inlined 20 times and the shader would take long to compile.
    [loop]
    for (uint h = 0; h < 2u; h++) {
        uint v = halves[h];
        uint prev_q = v;
        [loop]
        for (uint k = 0; k < 5u; k++) {
            uint q = (v * RECIP[k]) >> 16;
            int t = (int)(prev_q - q * 3u) - 1;
            prev_q = q;

            uint j = h * 5u + k;
            uint idx = base_px + j * step_px;
            if (t != 0 && idx < num_px) {
                uint ref_w = PrevShifted(idx);
                int d = StepOf(ref_w);
                ApplyTrit(idx, t, d, tag, ref_w);
                if (is_2x != 0u && idx + 1u < num_px) {
                    ApplyTrit(idx + 1u, t, d, tag, PrevShifted(idx + 1u));
                }
            }
        }
    }
}

// =========================================================================
// Pass 3: YCbCr -> RGBA Swapchain Output (1 thread per pixel)
// =========================================================================
[numthreads(16, 16, 1)]
void CSYCbCrToRGBA(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= width || dtid.y >= height) return;

    uint w = g_prev_yuvd[dtid.y * width + dtid.x];
    float y  = (float)(w & 0xFFu);
    // Display-only deblocking: across a 4x4 block edge, a small luma step
    // (at most post_thr, too small to be a real edge) is pulled a quarter of
    // the way towards the other side. Applied after the closed loop: the
    // reference is untouched, so nothing can drift.
    if (post_thr != 0u) {
        int yc = (int)(w & 0xFFu);
        int adj = 0;
        uint x = dtid.x, yy = dtid.y;
        uint idx = yy * width + x;
        if ((x & 3u) == 0u && x > 0u) {
            int d = (int)(g_prev_yuvd[idx - 1u] & 0xFFu) - yc;
            if (abs(d) <= (int)post_thr) adj += d / 4;
        } else if ((x & 3u) == 3u && x + 1u < width) {
            int d = (int)(g_prev_yuvd[idx + 1u] & 0xFFu) - yc;
            if (abs(d) <= (int)post_thr) adj += d / 4;
        }
        if ((yy & 3u) == 0u && yy > 0u) {
            int d = (int)(g_prev_yuvd[idx - width] & 0xFFu) - yc;
            if (abs(d) <= (int)post_thr) adj += d / 4;
        } else if ((yy & 3u) == 3u && yy + 1u < height) {
            int d = (int)(g_prev_yuvd[idx + width] & 0xFFu) - yc;
            if (abs(d) <= (int)post_thr) adj += d / 4;
        }
        y = (float)clamp(yc + adj, 0, 255);
    }
    float cb = (float)((w >> 8) & 0xFFu) - 128.0f;
    float cr = (float)((w >> 16) & 0xFFu) - 128.0f;

    float r = clamp((y + 1.402f * cr) / 255.0f, 0.0f, 1.0f);
    float g = clamp((y - 0.344136f * cb - 0.714136f * cr) / 255.0f, 0.0f, 1.0f);
    float b = clamp((y + 1.772f * cb) / 255.0f, 0.0f, 1.0f);

    g_out_rgba[dtid.xy] = float4(r, g, b, 1.0f);
}

// =========================================================================
// Pass 2b: Transform intra reconstruction (1 thread per 4x4 block)
// Runs after CSApply. A flagged block is rebuilt from its coefficients
// (missing ones are 0), then its coefficients and flag are cleared so the
// buffers are all-zero again for the next frame.
// =========================================================================
[numthreads(64, 1, 1)]
void CSITrans(uint3 dtid : SV_DispatchThreadID) {
    uint bw = (width + 3u) >> 2;
    uint bh = (height + 3u) >> 2;
    uint b = dtid.x;
    uint mode = (b < bw * bh) ? g_tflag[b] : 0u;
    if (mode == 0u) return;
    g_tflag[b] = 0u;
    uint x0 = (b % bw) * 4u;
    uint y0 = (b / bw) * 4u;
    bool inside = (x0 + 4u <= width) && (y0 + 4u <= height);
    // Inter blocks are predicted from what CSMotionDecay wrote there.
    uint cur[16];
    [unroll] for (uint j0 = 0; j0 < 16u; j0++) {
        cur[j0] = inside ? g_cur_yuvd[(y0 + (j0 >> 2)) * width + x0 + (j0 & 3u)] : 0x00808080u;
    }
    int rec[48];
    [unroll] for (uint pl = 0; pl < 3u; pl++) {
        int c[16];
        [unroll] for (uint k = 0; k < 16u; k++) {
            uint pid = b * 48u + pl * 16u + k;
            c[TQ_ZZ[k]] = (int)g_coef[pid];
            g_coef[pid] = 0u;
        }
        TqInverse(c);
        [unroll] for (uint q = 0; q < 16u; q++) {
            int pv = (mode == 2u) ? (int)((cur[q] >> (pl * 8u)) & 0xFFu) : 128;
            rec[pl * 16u + q] = clamp(pv + c[q], 0, 255);
        }
    }
    if (!inside) return; // never coded by the encoder
    [unroll] for (uint j = 0; j < 16u; j++) {
        uint idx = (y0 + (j >> 2)) * width + x0 + (j & 3u);
        g_cur_yuvd[idx] = (uint)rec[j] | ((uint)rec[16u + j] << 8) |
                          ((uint)rec[32u + j] << 16) | (delta_min << 24);
    }
}
