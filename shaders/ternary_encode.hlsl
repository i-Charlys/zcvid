// ternary_encode.hlsl - D3D11 Compute Shader (cs_5_0)
// Full-Color (YCbCr 4:4:4 / 4:2:0-ready) + 32x32 Tile MV + Jayant ADM +
// Rolling Intra-Refresh & Disocclusion Repair, Zero-Atomic-Contention Ref.
//
// Pass 0: CSLumaHalf / CSLumaQuarter -> box-filtered luma of the source at 1/2 and 1/4 resolution
// Pass 1: CSTileMV  -> Estimates (dx, dy) per 32x32 tile into g_tile_mvs
// Pass 2: CSEncode  -> Encodes Y, Cb, Cr 10-trit blocks + Intra blocks
//                      1 uint32 per pixel in g_ref_yuvd: [Y:8 | Cb:8 | Cr:8 | delta:8]

cbuffer TernaryParams : register(b0) {
    uint width;
    uint height;
    uint threshold;
    uint delta_min;
    uint delta_max;
    uint gamma_q8;
    uint neutral_y;
    uint total_blocks;   // ceil(width * height / 10)
    uint frame_id;
    uint intra_cycles;   // e.g. 30 (1/30th of rows refreshed per frame)
    uint intra_err_thr;  // e.g. 24 (disocclusion threshold -> instant Intra)
    uint tiles_x;        // ceil(width / 32)
    uint tiles_y;        // ceil(height / 32)
    uint search_r;       // e.g. 6
    uint max_record_words; // Rate-control cap in uint32 words (0 = unlimited)
    uint mode_flags;     // bit 0 = DELTA_2X mode (20 px per 10-trit block)
                         // bit 1 = rolling refresh in 8-bit 2-px intra
                         // bit 2 = transform intra (CSIntraT) on 4x4 blocks
                         // bit 3 = transform inter: residual vs motion prediction
                         // bit 4 = tile vectors in quarter pels
                         // bit 6 = ordered dither on the 6-bit rolling intra
    uint tq_qp;          // luma qp of the transform intra (chroma: +3)
    uint tq_cqp;         // chroma qp offset of the transform intra
    uint tq_inter_thr;   // mean |luma residual| above which a block is transform-coded
    uint tq_pad2;
};

Texture2D<unorm float4>   g_src_rgba    : register(t0);
StructuredBuffer<uint>    g_prev_yuvd   : register(t1); // Ping-pong previous frame ref [Y|Cb|Cr|delta]
Texture2D<unorm float4>   g_prev_src    : register(t2); // Previous source frame (motion search only)
Texture2D<uint>           g_half_cur    : register(t3); // Half-res luma of the source (CSLumaHalf)
Texture2D<uint>           g_half_prev   : register(t4); // Half-res luma of the previous source
StructuredBuffer<int2>    g_prev_mvs    : register(t5); // Tile vectors of the previous frame
Texture2D<uint>           g_quarter_cur : register(t6); // Quarter-res luma of the source (CSLumaQuarter)
Texture2D<uint>           g_quarter_prev: register(t7); // Quarter-res luma of the previous source
RWStructuredBuffer<uint>  g_cur_yuvd    : register(u0); // Ping-pong current frame ref
RWStructuredBuffer<int2>  g_tile_mvs    : register(u1); // (dx, dy) per 32x32 tile
RWStructuredBuffer<uint>  g_out_records : register(u2); // Compact output records
RWByteAddressBuffer       g_out_counter : register(u3); // [0]=record_words_cnt
RWTexture2D<uint>         g_half_out    : register(u4); // CSLumaHalf / CSLumaQuarter output
RWStructuredBuffer<uint>  g_tflag_out   : register(u5); // CSIntraT: 1 per 4x4 block it coded
StructuredBuffer<uint>    g_tflag       : register(t8); // CSEncode: the same flags, read-only
// Capture hints per tile (desktop duplication): x = 0 none, 1 moved by
// (y, z) whole pixels (a MoveRect), 2 unchanged since the previous frame.
StructuredBuffer<int4>    g_mv_hint     : register(t9);

static const uint POW3[10] = { 1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683 };
static const uint BAYER4[16] = { 0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5 };

// Output records are always 2 uint32 words:
//   Delta: word0 = (plane << 30) | (is_2x << 29) | blk      (plane 0=Y, 1=Cb, 2=Cr)
//          word1 = 10 trits packed in base 3 (0..59048)
//   Intra: word0 = 0xC0000000 | idx                          (idx = first pixel of the pair)
//          word1 = Y0 | (Cb << 8) | (Y1 << 16) | (Cr << 24)  (4:2:2 pair idx, idx + 1)
//
// Closed-loop contract shared with ternary_decode.hlsl (any change must be
// mirrored there, otherwise encoder and decoder references drift apart):
//   - the reference of a pixel is the previous frame at its tile-MV position;
//   - a trit t != 0 adds t * d, where d is the step stored at that reference;
//   - the step halves (floor delta_min) after every frame unless a non-zero
//     luma trit doubles it (cap delta_max);
//   - pixels are coded as (idx, idx + 1) pairs. A pair is either entirely
//     Intra (shared chroma, both trits forced to 0) or entirely Delta;
//   - in DELTA_2X mode one trit drives both pixels of the pair with the step
//     of the first pixel.

void RgbToYCbCr(float3 rgb, out int y, out int cb, out int cr) {
    float r = rgb.r * 255.0f;
    float g = rgb.g * 255.0f;
    float b = rgb.b * 255.0f;
    y  = clamp((int)round( 0.299f * r + 0.587f * g + 0.114f * b), 0, 255);
    cb = clamp((int)round(-0.169f * r - 0.331f * g + 0.500f * b + 128.0f), 0, 255);
    cr = clamp((int)round( 0.500f * r - 0.419f * g - 0.081f * b + 128.0f), 0, 255);
}

int LeakyStep(int prev_val, int trit, int delta_val, int neutral) {
    int diff = prev_val - neutral;
    int decayed = (diff * (int)gamma_q8) / 256;
    return clamp(neutral + decayed + trit * delta_val, 0, 255);
}

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
//         w1 = (level & 0xFFFF) | (qp << 16) | (pred << 24)  (level in [-255, 255])
// pred = 0: samples around 128 (intra). pred = 1: residual against the
// motion-compensated, decayed reference, i.e. exactly the value the
// decoder's CSMotionDecay left in the block (inter).
// -------------------------------------------------------------------------
#define TQ_INTER_QP_MAX 51u
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

// Tile vector in quarter pels, whatever unit this frame uses.
int2 MvQ(int2 mv) {
    return ((mode_flags & 16u) != 0u) ? mv : mv * 4;
}

// =========================================================================
// Pass 1: Per-Tile 32x32 Motion Estimation (1 threadgroup per tile)
// =========================================================================
groupshared uint gs_best_sad;
groupshared int2 gs_best_mv;

// The search matches the source against the previous *source* frame, not
// against the reconstructed reference. The reference carries coding noise of
// up to +-threshold per pixel; on low-contrast tiles that noise outweighs the
// picture, the best match becomes a random vector, and every wrong vector
// smears the tile a bit more (the decoder copies from the wrong place) until
// moving areas dissolve. Source frames have no such noise.
//
// It is a 3-level pyramid: every multiple-of-4 vector within +-search_r is
// tried on the quarter-resolution luma, the winner is refined by +-2 px on
// the half-resolution luma, then by +-1 px at full resolution. The box
// filters matter: on unfiltered pixels a 1 px miss on sharp content scores as
// badly as any wrong vector, so a sparse search returns noise.
//
// On low-texture tiles the coarse stages can still pick a far-off vector, so
// the last stage also tries the vectors this tile and its four neighbours had
// in the previous frame: once textured tiles lock on a pan, it spreads.
int LumaOf(float3 rgb) {
    return (int)round((0.299f * rgb.r + 0.587f * rgb.g + 0.114f * rgb.b) * 255.0f);
}

[numthreads(16, 16, 1)]
void CSLumaHalf(uint3 dtid : SV_DispatchThreadID) {
    uint hw = (width + 1u) >> 1;
    uint hh = (height + 1u) >> 1;
    if (dtid.x >= hw || dtid.y >= hh) return;
    float3 sum = float3(0.0f, 0.0f, 0.0f);
    for (uint j = 0; j < 2u; j++) {
        for (uint i = 0; i < 2u; i++) {
            uint x = min(dtid.x * 2u + i, width - 1u);
            uint y = min(dtid.y * 2u + j, height - 1u);
            sum += g_src_rgba.Load(int3(x, y, 0)).rgb;
        }
    }
    g_half_out[dtid.xy] = (uint)LumaOf(sum * 0.25f);
}

[numthreads(16, 16, 1)]
void CSLumaQuarter(uint3 dtid : SV_DispatchThreadID) {
    uint qw = (width + 3u) >> 2;
    uint qh = (height + 3u) >> 2;
    if (dtid.x >= qw || dtid.y >= qh) return;
    uint hw = (width + 1u) >> 1;
    uint hh = (height + 1u) >> 1;
    uint sum = 0u;
    for (uint j = 0; j < 2u; j++) {
        for (uint i = 0; i < 2u; i++) {
            uint x = min(dtid.x * 2u + i, hw - 1u);
            uint y = min(dtid.y * 2u + j, hh - 1u);
            sum += g_half_cur.Load(int3(x, y, 0));
        }
    }
    g_half_out[dtid.xy] = (sum + 2u) >> 2;
}

// SAD of a tile on a reduced luma plane: 64 samples (8x8 grid of pitch
// `step`, first sample at `first`), vector in pixels of that plane.
uint EvalTileSadLow(Texture2D<uint> cur, Texture2D<uint> prev, int w, int h,
                    uint x0, uint y0, uint first, uint step, int dx, int dy) {
    uint sad = 0;
    for (uint j = 0; j < 8u; j++) {
        int py = min((int)(y0 + first + j * step), h - 1);
        int sy = clamp(py - dy, 0, h - 1);
        for (uint i = 0; i < 8u; i++) {
            int px = min((int)(x0 + first + i * step), w - 1);
            int sx = clamp(px - dx, 0, w - 1);
            int cy = (int)cur.Load(int3(px, py, 0));
            int ry = (int)prev.Load(int3(sx, sy, 0));
            sad += (uint)abs(cy - ry);
        }
    }
    return sad;
}

// SAD of a 32x32 tile at full resolution (64 samples), vector in pixels.
uint EvalTileSad(uint x0, uint y0, int dx, int dy) {
    uint sad = 0;
    for (uint ly = 2; ly < 32; ly += 4) {
        int py = min((int)(y0 + ly), (int)height - 1);
        int sy = clamp(py - dy, 0, (int)height - 1);
        for (uint lx = 2; lx < 32; lx += 4) {
            int px = min((int)(x0 + lx), (int)width - 1);
            int sx = clamp(px - dx, 0, (int)width - 1);

            int cy = LumaOf(g_src_rgba.Load(int3(px, py, 0)).rgb);
            int ry = LumaOf(g_prev_src.Load(int3(sx, sy, 0)).rgb);
            sad += (uint)abs(cy - ry);
        }
    }
    return sad;
}

// Candidates are ranked by InterlockedMin on (sad << 14 | vector bits), the
// vector always in full-resolution pixels (|d| <= 63). The zero vector uses
// bits 0 (no other vector can: dx + 64 >= 1) and gets a light -4u SAD bias
// so static/flat tiles do not jitter on tiny noise fluctuations.
// Vectors in quarter pels (|d| <= 255), 9 bits each; SAD (64 samples) above.
uint MvKey(uint norm_sad, int dx, int dy) {
    uint biased_sad = (dx == 0 && dy == 0) ? (norm_sad > 4u ? norm_sad - 4u : 0u) : norm_sad;
    uint mv_bits = (dx == 0 && dy == 0) ? 0u : (((uint)(dy + 256) << 9) | (uint)(dx + 256));
    return (biased_sad << 18) | mv_bits;
}

int2 MvOfKey(uint key) {
    uint mv_bits = key & 0x3FFFFu;
    if (mv_bits == 0u) return int2(0, 0);
    return int2((int)(mv_bits & 511u) - 256, (int)(mv_bits >> 9) - 256);
}

// SAD of a 32x32 tile (64 samples) against the previous source moved by a
// quarter-pel vector, interpolated like FetchRefQ.
uint EvalTileSadQ(uint x0, uint y0, int dxq, int dyq) {
    uint sad = 0;
    for (uint ly = 2; ly < 32; ly += 4) {
        int py = min((int)(y0 + ly), (int)height - 1);
        int syq = py * 4 - dyq;
        int fy = syq & 3;
        int iy0 = clamp(syq >> 2, 0, (int)height - 1);
        int iy1 = clamp((syq >> 2) + 1, 0, (int)height - 1);
        for (uint lx = 2; lx < 32; lx += 4) {
            int px = min((int)(x0 + lx), (int)width - 1);
            int sxq = px * 4 - dxq;
            int fx = sxq & 3;
            int ix0 = clamp(sxq >> 2, 0, (int)width - 1);
            int ix1 = clamp((sxq >> 2) + 1, 0, (int)width - 1);
            int cy = LumaOf(g_src_rgba.Load(int3(px, py, 0)).rgb);
            int a = LumaOf(g_prev_src.Load(int3(ix0, iy0, 0)).rgb);
            int b = LumaOf(g_prev_src.Load(int3(ix1, iy0, 0)).rgb);
            int c = LumaOf(g_prev_src.Load(int3(ix0, iy1, 0)).rgb);
            int d = LumaOf(g_prev_src.Load(int3(ix1, iy1, 0)).rgb);
            int ry = (a * (4 - fx) * (4 - fy) + b * fx * (4 - fy) + c * (4 - fx) * fy + d * fx * fy + 8) >> 4;
            sad += (uint)abs(cy - ry);
        }
    }
    return sad;
}

[numthreads(16, 1, 1)]
void CSTileMV(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tx = gid.x;
    uint ty = gid.y;
    if (tx >= tiles_x || ty >= tiles_y) return;
    int4 hint = g_mv_hint[ty * tiles_x + tx];
    if (hint.x == 2) {
        // The capture says nothing changed here: no search, zero vector.
        if (gtid.x == 0) g_tile_mvs[ty * tiles_x + tx] = int2(0, 0);
        return;
    }

    if (gtid.x == 0) {
        gs_best_sad = 0xFFFFFFFFu;
        gs_best_mv  = int2(0, 0);
    }
    GroupMemoryBarrierWithGroupSync();

    bool qpel = (mode_flags & 16u) != 0u;
    // Quarter-pel vectors travel as int8: |v| <= 127 qpel.
    int r = min((int)search_r, qpel ? 31 : 60);

    // Stage 1: every multiple-of-4 vector within +-search_r, quarter-res luma.
    int qr = r >> 2;
    int span = 2 * qr + 1;
    int total_cands = span * span;
    int qw = (int)((width + 3u) >> 2);
    int qh = (int)((height + 3u) >> 2);

    for (int c = (int)gtid.x; c < total_cands; c += 16) {
        int qdy = (c / span) - qr;
        int qdx = (c % span) - qr;
        uint norm_sad = EvalTileSadLow(g_quarter_cur, g_quarter_prev, qw, qh,
                                       tx * 8u, ty * 8u, 0u, 1u, qdx, qdy);
        if (norm_sad != 0xFFFFFFFFu) {
            InterlockedMin(gs_best_sad, MvKey(norm_sad, qdx * 16, qdy * 16));
        }
    }
    GroupMemoryBarrierWithGroupSync();

    uint best_q = gs_best_sad;
    GroupMemoryBarrierWithGroupSync();
    if (gtid.x == 0) {
        gs_best_sad = 0xFFFFFFFFu;
    }
    GroupMemoryBarrierWithGroupSync();

    // Stage 2: the quarter-res winner and its 8 neighbours at +-2 px, half-res luma.
    if (gtid.x < 9u && r > 0) {
        int2 qmv = (best_q == 0xFFFFFFFFu) ? int2(0, 0) : MvOfKey(best_q);
        int hdx = qmv.x / 8 + ((int)(gtid.x % 3u) - 1);
        int hdy = qmv.y / 8 + ((int)(gtid.x / 3u) - 1);
        int hw = (int)((width + 1u) >> 1);
        int hh = (int)((height + 1u) >> 1);
        // Stay inside +-r: stage 3 only keeps vectors within r, so a winner
        // at r + 2 here would leave it with nothing but rejected candidates.
        uint norm_sad = (abs(hdx * 2) <= r && abs(hdy * 2) <= r)
            ? EvalTileSadLow(g_half_cur, g_half_prev, hw, hh,
                             tx * 16u, ty * 16u, 1u, 2u, hdx, hdy)
            : 0xFFFFFFFFu;
        if (norm_sad != 0xFFFFFFFFu) {
            InterlockedMin(gs_best_sad, MvKey(norm_sad, hdx * 8, hdy * 8));
        }
    }
    GroupMemoryBarrierWithGroupSync();

    uint best_coarse = gs_best_sad;
    GroupMemoryBarrierWithGroupSync();
    if (gtid.x == 0) {
        gs_best_sad = 0xFFFFFFFFu;
    }
    GroupMemoryBarrierWithGroupSync();

    // Stage 3, full resolution: the half-res winner and its 8 neighbours
    // (threads 0..8), the zero vector (9), and the previous-frame vectors of
    // this tile and of its left/right/up/down neighbours (10..14).
    if (gtid.x < 16u && r > 0) {
        int2 cand = int2(0, 0);
        if (gtid.x < 9u) {
            int2 cmv = (best_coarse == 0xFFFFFFFFu) ? int2(0, 0) : (MvOfKey(best_coarse) / 4);
            cand = cmv + int2((int)(gtid.x % 3u) - 1, (int)(gtid.x / 3u) - 1);
        } else if (gtid.x >= 10u) {
            uint ntx = tx;
            uint nty = ty;
            if (gtid.x == 11u) ntx = (tx > 0u) ? tx - 1u : tx;
            if (gtid.x == 12u) ntx = min(tx + 1u, tiles_x - 1u);
            if (gtid.x == 13u) nty = (ty > 0u) ? ty - 1u : ty;
            if (gtid.x == 14u) nty = min(ty + 1u, tiles_y - 1u);
            cand = g_prev_mvs[nty * tiles_x + ntx];
            if (qpel) cand = (cand + 2) >> 2; // nearest whole pel
        } else if (gtid.x == 15u) {
            // The vector the capture reports for a moved window, if any.
            cand = (hint.x == 1) ? hint.yz : int2(0, 0);
        }
        if (cand.x >= -r && cand.x <= r && cand.y >= -r && cand.y <= r) {
            uint norm_sad = EvalTileSad(tx * 32u, ty * 32u, cand.x, cand.y);
            if (norm_sad != 0xFFFFFFFFu) {
                InterlockedMin(gs_best_sad, MvKey(norm_sad, cand.x * 4, cand.y * 4));
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Stage 4 (quarter-pel mode): the 8 half-pel neighbours of the winner,
    // then the 8 quarter-pel neighbours of that.
    [unroll] for (uint st = 0; st < 2u; st++) {
        if (qpel && r > 0) {
            uint cur_best = gs_best_sad;
            GroupMemoryBarrierWithGroupSync();
            if (gtid.x < 8u && cur_best != 0xFFFFFFFFu) {
                int stepq = (st == 0u) ? 2 : 1;
                uint o = (gtid.x < 4u) ? gtid.x : gtid.x + 1u; // skip the centre
                int2 c0 = MvOfKey(cur_best);
                int2 cq = c0 + int2((int)(o % 3u) - 1, (int)(o / 3u) - 1) * stepq;
                if (abs(cq.x) <= 127 && abs(cq.y) <= 127) {
                    InterlockedMin(gs_best_sad, MvKey(EvalTileSadQ(tx * 32u, ty * 32u, cq.x, cq.y), cq.x, cq.y));
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    if (gtid.x == 0) {
        // No candidate had a valid sample (or search disabled): keep (0,0).
        uint best = gs_best_sad;
        int2 v = (best == 0xFFFFFFFFu) ? int2(0, 0) : MvOfKey(best);
        g_tile_mvs[ty * tiles_x + tx] = qpel ? v : (v / 4);
    }
}

// =========================================================================
// Pass 2: Full-Color Ternary Encode + Jayant ADM + Disocclusion/Intra
// =========================================================================
struct PixState {
    bool valid;       // idx is inside the frame
    bool rolling;     // rolling intra-refresh row
    bool want_intra;  // rolling refresh row or disocclusion
    uint px;          // x coordinate in frame
    int cy, ccb, ccr; // source
    int ry, rcb, rcr; // motion-compensated reference
    int d;            // adaptive step stored at the reference
    int jnd;          // edge-preserving JND deadzone boost from s.ry
};

PixState LoadPix(uint idx, uint num_px, uint intra_y0, uint intra_y1, bool allow_disocc) {
    PixState s;
    s.valid = (idx < num_px);
    // No early return: the caller's loop must stay unrollable by fxc.
    idx = min(idx, num_px - 1u);
    uint py = idx / width;
    uint px = idx - py * width;
    s.px = px;
    int2 mvq = MvQ(g_prev_mvs[(py >> 5) * tiles_x + (px >> 5)]);
    int sx = clamp(((int)px * 4 - mvq.x) >> 2, 0, (int)width - 1);
    int sy = clamp(((int)py * 4 - mvq.y) >> 2, 0, (int)height - 1);
    uint ref_w = FetchRefQ(px, py, mvq);

    s.ry  = (int)(ref_w & 0xFFu);
    s.rcb = (int)((ref_w >> 8) & 0xFFu);
    s.rcr = (int)((ref_w >> 16) & 0xFFu);
    s.d   = max((int)delta_min, (int)((ref_w >> 24) & 0xFFu));

    // Closed-loop edge-preserving JND deadzone boost using only reference luma (s.ry).
    int ry_l = (int)(g_prev_yuvd[(uint)sy * width + (uint)max(sx - 1, 0)] & 0xFFu);
    int ry_r = (int)(g_prev_yuvd[(uint)sy * width + (uint)min(sx + 1, (int)width - 1)] & 0xFFu);
    int ry_u = (int)(g_prev_yuvd[(uint)max(sy - 1, 0) * width + (uint)sx] & 0xFFu);
    int ry_d = (int)(g_prev_yuvd[(uint)min(sy + 1, (int)height - 1) * width + (uint)sx] & 0xFFu);
    int gx = abs(ry_r - ry_l);
    int gy = abs(ry_d - ry_u);
    int act_x = abs(ry_r - s.ry) + abs(s.ry - ry_l);
    int act_y = abs(ry_d - s.ry) + abs(s.ry - ry_u);
    int osc = (act_x - gx) + (act_y - gy);
    int coh = abs(gx - gy) + abs(act_x - act_y);
    int max_jump = max(max(abs(ry_r - s.ry), abs(s.ry - ry_l)),
                       max(abs(ry_d - s.ry), abs(s.ry - ry_u)));
    int tex_energy = max(osc - coh, 0);
    s.jnd = (threshold > 0u && max_jump <= 64) ? clamp((tex_energy - 12) >> 3, 0, (int)threshold + 6) : 0;

    float4 src_c = g_src_rgba.Load(int3(px, py, 0));
    RgbToYCbCr(src_c.rgb, s.cy, s.ccb, s.ccr);

    s.rolling = (py >= intra_y0 && py < intra_y1);
    bool disocc  = allow_disocc && (abs(s.cy - s.ry) > (int)intra_err_thr);
    s.want_intra = s.valid && (s.rolling || disocc);
    return s;
}

// A step is only taken when it brings the reference closer to the source:
// the error must exceed the threshold and half the step d. Stepping with a
// step larger than twice the error overshoots, and since a non-zero luma trit
// doubles the step, the reference then swings around its target. Declining
// codes a 0, which lets the step halve. Encoder-only decision: the decoder
// applies whatever trits it receives.
// 1 when CSIntraT coded the 4x4 block holding this pixel.
uint TFlagAt(uint idx) {
    if ((mode_flags & 4u) == 0u) return 0u;
    uint py = idx / width;
    uint px = idx - py * width;
    return g_tflag[(py >> 2) * ((width + 3u) >> 2) + (px >> 2)];
}

int Quant(int err, int thr, int d) {
    int mag = abs(err);
    if (mag <= thr || 2 * mag <= d) return 0;
    return (err > 0) ? 1 : -1;
}

// Writes the reconstructed reference exactly as the decoder will rebuild it.
void StoreDelta(uint idx, PixState s, int ty, int tcb, int tcr, int d_used, int next_d) {
    int ny  = LeakyStep(s.ry,  ty,  d_used, (int)neutral_y);
    int ncb = LeakyStep(s.rcb, tcb, d_used, 128);
    int ncr = LeakyStep(s.rcr, tcr, d_used, 128);
    g_cur_yuvd[idx] = (uint)ny | ((uint)ncb << 8) | ((uint)ncr << 16) | ((uint)next_d << 24);
}

[numthreads(64, 1, 1)]
void CSEncode(uint3 dtid : SV_DispatchThreadID) {
    uint is_2x = mode_flags & 1u;
    uint eff_blocks = (is_2x != 0u) ? ((total_blocks + 1u) >> 1) : total_blocks;
    if (dtid.x >= eff_blocks) return;
    // Rotate starting block per frame so hard-cap budgets never starve the bottom of the frame.
    uint blk = (dtid.x + ((frame_id * 29713u) % eff_blocks)) % eff_blocks;

    uint num_px = width * height;
    uint cycles = max(intra_cycles, 1u);
    uint rows_per_slot = (height + cycles - 1u) / cycles;
    uint intra_y0 = (frame_id % cycles) * rows_per_slot;
    uint intra_y1 = intra_y0 + rows_per_slot;

    // Hard rate control: g_out_counter counts total desired words (for CPU RC),
    // while g_out_records and g_cur_yuvd only commit records within max_record_words.
    uint cur_words = g_out_counter.Load(0);
    bool budget_full = (max_record_words != 0u) && (cur_words >= max_record_words);
    // Disocclusion intra up to half the budget: newly uncovered areas are
    // repaired in one frame instead of converging over several at delta_min.
    bool allow_disocc = (max_record_words == 0u) || (cur_words < (max_record_words >> 1));
    bool allow_chroma = (max_record_words == 0u) || (cur_words < max_record_words);
    // The rate control has budget to spare: rolling refresh keeps full 8-bit
    // samples and per-pair chroma (2-px intra) instead of 6-bit 4-px records.
    bool intra8 = (mode_flags & 2u) != 0u;

    int y_thr = (int)threshold;
    int c_thr = (int)threshold * 2 + 4;

    uint packed_y  = 0;
    uint packed_cb = 0;
    uint packed_cr = 0;
    bool nz_y  = false;
    bool nz_cb = false;
    bool nz_cr = false;

    // 1x: 5 pairs, 2 trits each (10 px). 2x: 10 pairs, 1 trit each (20 px).
    uint base_px = blk * ((is_2x != 0u) ? 20u : 10u);
    uint num_pairs = (is_2x != 0u) ? 10u : 5u;
    bool skip_pair = false;
    uint intra_mask = 0u;

    // A real loop: unrolled, every iteration would inline the motion
    // compensation filter of LoadPix several times over, and compiling the
    // shader took most of a minute.
    [loop]
    for (uint k = 0; k < 10u; k++) {
        if (k < num_pairs) {
            uint idx_a = base_px + k * 2u;
            uint idx_b = idx_a + 1u;
            PixState a = LoadPix(idx_a, num_px, intra_y0, intra_y1, allow_disocc);
            PixState b = LoadPix(idx_b, num_px, intra_y0, intra_y1, allow_disocc);

            int ty_a = 0, tcb_a = 0, tcr_a = 0;
            int ty_b = 0, tcb_b = 0, tcr_b = 0;
            bool same_row_2 = (a.px + 1u < width);
            bool same_row_4 = (a.px + 3u < width);

            // A pair never straddles two 4x4 blocks (both start on even x).
            bool tcoded = a.valid && (TFlagAt(idx_a) != 0u);
            if (skip_pair) {
                skip_pair = false;
            } else if (tcoded) {
                // CSIntraT already wrote the reference and the records.
                intra_mask |= (3u << (k * 2u));
            } else if (a.valid && same_row_2 && (a.want_intra || b.want_intra)) {
                if (budget_full) {
                    StoreDelta(idx_a, a, 0, 0, 0, a.d, max((int)delta_min, a.d >> 1));
                    if (b.valid) StoreDelta(idx_b, b, 0, 0, 0, b.d, max((int)delta_min, b.d >> 1));
                } else if ((k & 1u) == 0u && (k + 1u) < num_pairs && a.rolling && !intra8 && same_row_4 && (idx_a + 3u < num_px) &&
                           TFlagAt(idx_a + 2u) == 0u) {
                    PixState c = LoadPix(idx_a + 2u, num_px, intra_y0, intra_y1, allow_disocc);
                    PixState d = LoadPix(idx_a + 3u, num_px, intra_y0, intra_y1, allow_disocc);
                    uint slot = 0;
                    g_out_counter.InterlockedAdd(0, 2u, slot);
                    if (max_record_words == 0u || slot + 2u <= max_record_words) {
                        // Mode bit 6: ordered (Bayer 4x4) dither instead of plain
                        // rounding to 6 bits, so a smooth gradient turns into fine
                        // noise rather than 4-level bands. A constant 8 is the plain
                        // rounding. Encoder-only: the records carry the quantized
                        // values whatever the rounding.
                        bool dith = (mode_flags & 64u) != 0u;
                        uint brow = ((idx_a / width) & 3u) * 4u;
                        int y0_q6 = clamp((4 * a.cy + (dith ? (int)BAYER4[brow + ((a.px + 0u) & 3u)] : 8)) >> 4, 0, 63);
                        int y1_q6 = clamp((4 * b.cy + (dith ? (int)BAYER4[brow + ((a.px + 1u) & 3u)] : 8)) >> 4, 0, 63);
                        int y2_q6 = clamp((4 * c.cy + (dith ? (int)BAYER4[brow + ((a.px + 2u) & 3u)] : 8)) >> 4, 0, 63);
                        int y3_q6 = clamp((4 * d.cy + (dith ? (int)BAYER4[brow + ((a.px + 3u) & 3u)] : 8)) >> 4, 0, 63);
                        int cb_q6 = clamp(((a.ccb + b.ccb + c.ccb + d.ccb + 8) >> 4), 0, 63);
                        int cr_q6 = clamp(((a.ccr + b.ccr + c.ccr + d.ccr + 8) >> 4), 0, 63);
                        // idx_a is even (blocks start on even pixels, k is even): the
                        // 22-bit field holds the pixel-pair index, good for 8.3 Mpx (4K).
                        g_out_records[slot + 0u] = 0xE0000000u | ((uint)cr_q6 << 22) | ((idx_a >> 1) & 0x3FFFFFu);
                        g_out_records[slot + 1u] = (uint)y0_q6 | ((uint)y1_q6 << 6) | ((uint)y2_q6 << 12) | ((uint)y3_q6 << 18) | ((uint)cb_q6 << 24);
                        uint cb_rec = (uint)((cb_q6 << 2) | (cb_q6 >> 4));
                        uint cr_rec = (uint)((cr_q6 << 2) | (cr_q6 >> 4));
                        uint chroma_d = (cb_rec << 8) | (cr_rec << 16) | (delta_min << 24);
                        g_cur_yuvd[idx_a + 0u] = (uint)((y0_q6 << 2) | (y0_q6 >> 4)) | chroma_d;
                        g_cur_yuvd[idx_a + 1u] = (uint)((y1_q6 << 2) | (y1_q6 >> 4)) | chroma_d;
                        g_cur_yuvd[idx_a + 2u] = (uint)((y2_q6 << 2) | (y2_q6 >> 4)) | chroma_d;
                        g_cur_yuvd[idx_a + 3u] = (uint)((y3_q6 << 2) | (y3_q6 >> 4)) | chroma_d;
                        intra_mask |= (0xFu << (k * 2u));
                    } else {
                        StoreDelta(idx_a + 0u, a, 0, 0, 0, a.d, max((int)delta_min, a.d >> 1));
                        StoreDelta(idx_a + 1u, b, 0, 0, 0, b.d, max((int)delta_min, b.d >> 1));
                        StoreDelta(idx_a + 2u, c, 0, 0, 0, c.d, max((int)delta_min, c.d >> 1));
                        StoreDelta(idx_a + 3u, d, 0, 0, 0, d.d, max((int)delta_min, d.d >> 1));
                    }
                    skip_pair = true;
                } else {
                    int cy_b = b.valid ? b.cy : a.cy;
                    int cb = b.valid ? ((a.ccb + b.ccb + 1) >> 1) : a.ccb;
                    int cr = b.valid ? ((a.ccr + b.ccr + 1) >> 1) : a.ccr;

                    uint slot = 0;
                    g_out_counter.InterlockedAdd(0, 2u, slot);
                    if (max_record_words == 0u || slot + 2u <= max_record_words) {
                        g_out_records[slot + 0u] = 0xC0000000u | (idx_a & 0x1FFFFFFFu);
                        g_out_records[slot + 1u] = (uint)a.cy | ((uint)cb << 8) | ((uint)cy_b << 16) | ((uint)cr << 24);

                        uint chroma_d = ((uint)cb << 8) | ((uint)cr << 16) | (delta_min << 24);
                        g_cur_yuvd[idx_a] = (uint)a.cy | chroma_d;
                        if (b.valid) g_cur_yuvd[idx_b] = (uint)cy_b | chroma_d;
                        intra_mask |= (3u << (k * 2u));
                    } else {
                        StoreDelta(idx_a, a, 0, 0, 0, a.d, max((int)delta_min, a.d >> 1));
                        if (b.valid) StoreDelta(idx_b, b, 0, 0, 0, b.d, max((int)delta_min, b.d >> 1));
                    }
                }
            } else if (a.valid) {
                if (budget_full) {
                    StoreDelta(idx_a, a, 0, 0, 0, a.d, max((int)delta_min, a.d >> 1));
                    if (b.valid) StoreDelta(idx_b, b, 0, 0, 0, b.d, max((int)delta_min, b.d >> 1));
                } else if (is_2x != 0u) {
                    // Mean error of the pair (encoder-only decision, any rounding is fine).
                    int ey  = b.valid ? (((a.cy  - a.ry)  + (b.cy  - b.ry))  >> 1) : (a.cy  - a.ry);
                    int ecb = b.valid ? (((a.ccb - a.rcb) + (b.ccb - b.rcb)) >> 1) : (a.ccb - a.rcb);
                    int ecr = b.valid ? (((a.ccr - a.rcr) + (b.ccr - b.rcr)) >> 1) : (a.ccr - a.rcr);
                    int jnd_p = b.valid ? min(a.jnd, b.jnd) : a.jnd;
                    ty_a  = Quant(ey, y_thr + jnd_p, a.d);
                    tcb_a = allow_chroma ? Quant(ecb, c_thr + jnd_p * 2, a.d) : 0;
                    tcr_a = allow_chroma ? Quant(ecr, c_thr + jnd_p * 2, a.d) : 0;

                    int up = min((int)delta_max, a.d << 1);
                    StoreDelta(idx_a, a, ty_a, tcb_a, tcr_a, a.d,
                               (ty_a == 0) ? max((int)delta_min, a.d >> 1) : up);
                    if (b.valid) {
                        StoreDelta(idx_b, b, ty_a, tcb_a, tcr_a, a.d,
                                   (ty_a == 0) ? max((int)delta_min, b.d >> 1) : up);
                    }
                } else {
                    ty_a  = Quant(a.cy - a.ry, y_thr + a.jnd, a.d);
                    tcb_a = allow_chroma ? Quant(a.ccb - a.rcb, c_thr + a.jnd * 2, a.d) : 0;
                    tcr_a = allow_chroma ? Quant(a.ccr - a.rcr, c_thr + a.jnd * 2, a.d) : 0;
                    StoreDelta(idx_a, a, ty_a, tcb_a, tcr_a, a.d,
                               (ty_a == 0) ? max((int)delta_min, a.d >> 1)
                                           : min((int)delta_max, a.d << 1));
                    if (b.valid) {
                        ty_b  = Quant(b.cy - b.ry, y_thr + b.jnd, b.d);
                        tcb_b = allow_chroma ? Quant(b.ccb - b.rcb, c_thr + b.jnd * 2, b.d) : 0;
                        tcr_b = allow_chroma ? Quant(b.ccr - b.rcr, c_thr + b.jnd * 2, b.d) : 0;
                        StoreDelta(idx_b, b, ty_b, tcb_b, tcr_b, b.d,
                                   (ty_b == 0) ? max((int)delta_min, b.d >> 1)
                                               : min((int)delta_max, b.d << 1));
                    }
                }
            }

            if (ty_a != 0 || ty_b != 0)   nz_y  = true;
            if (tcb_a != 0 || tcb_b != 0) nz_cb = true;
            if (tcr_a != 0 || tcr_b != 0) nz_cr = true;

            if (is_2x != 0u) {
                packed_y  += (uint)(ty_a  + 1) * POW3[k];
                packed_cb += (uint)(tcb_a + 1) * POW3[k];
                packed_cr += (uint)(tcr_a + 1) * POW3[k];
            } else if (k < 5u) {
                uint j = k * 2u;
                packed_y  += (uint)(ty_a  + 1) * POW3[j] + (uint)(ty_b  + 1) * POW3[j + 1u];
                packed_cb += (uint)(tcb_a + 1) * POW3[j] + (uint)(tcb_b + 1) * POW3[j + 1u];
                packed_cr += (uint)(tcr_a + 1) * POW3[j] + (uint)(tcr_b + 1) * POW3[j + 1u];
            }
        }
    }

    uint hdr_base = (is_2x << 29) | (blk & 0x1FFFFFFFu);
    bool drop_y = false, drop_cb = false, drop_cr = false;
    if (nz_y) {
        uint slot = 0;
        g_out_counter.InterlockedAdd(0, 2u, slot);
        if (max_record_words == 0u || slot + 2u <= max_record_words) {
            g_out_records[slot + 0u] = (0u << 30) | hdr_base;
            g_out_records[slot + 1u] = packed_y;
        } else {
            drop_y = true;
        }
    }
    if (nz_cb) {
        uint slot = 0;
        g_out_counter.InterlockedAdd(0, 2u, slot);
        if (max_record_words == 0u || slot + 2u <= max_record_words) {
            g_out_records[slot + 0u] = (1u << 30) | hdr_base;
            g_out_records[slot + 1u] = packed_cb;
        } else {
            drop_cb = true;
        }
    }
    if (nz_cr) {
        uint slot = 0;
        g_out_counter.InterlockedAdd(0, 2u, slot);
        if (max_record_words == 0u || slot + 2u <= max_record_words) {
            g_out_records[slot + 0u] = (2u << 30) | hdr_base;
            g_out_records[slot + 1u] = packed_cr;
        } else {
            drop_cr = true;
        }
    }

    // Boundary safety: if a delta record exceeded max_record_words, revert its
    // plane(s) in g_cur_yuvd to pure motion-compensated decay so Enc == Dec.
    if (drop_y || drop_cb || drop_cr) {
        uint px_cnt = (is_2x != 0u) ? 20u : 10u;
        for (uint p = 0; p < px_cnt; p++) {
            uint pidx = base_px + p;
            if (pidx < num_px && ((intra_mask & (1u << p)) == 0u)) {
                uint py = pidx / width;
                uint px = pidx - py * width;
                uint ref_w = FetchRefQ(px, py, MvQ(g_prev_mvs[(py >> 5) * tiles_x + (px >> 5)]));
                uint cur_w = g_cur_yuvd[pidx];
                if (drop_y) {
                    int ry = (int)(ref_w & 0xFFu);
                    int ny = LeakyStep(ry, 0, 0, (int)neutral_y);
                    int d  = max((int)delta_min, (int)((ref_w >> 24) & 0xFFu) >> 1);
                    cur_w = (cur_w & 0x00FFFF00u) | (uint)ny | ((uint)d << 24);
                }
                if (drop_cb) {
                    int rcb = (int)((ref_w >> 8) & 0xFFu);
                    int ncb = LeakyStep(rcb, 0, 0, 128);
                    cur_w = (cur_w & 0xFFFF00FFu) | ((uint)ncb << 8);
                }
                if (drop_cr) {
                    int rcr = (int)((ref_w >> 16) & 0xFFu);
                    int ncr = LeakyStep(rcr, 0, 0, 128);
                    cur_w = (cur_w & 0xFF00FFFFu) | ((uint)ncr << 16);
                }
                g_cur_yuvd[pidx] = cur_w;
            }
        }
    }
}

// =========================================================================
// Pass 1b: Transform intra (1 thread per 4x4 block, runs before CSEncode)
//
// Takes the blocks that need intra (rolling refresh rows, disocclusion) and
// codes them with the 4x4 integer transform instead of raw samples: about a
// third of the bits for the same area. Writes the reconstruction into the
// reference and flags the block, so CSEncode leaves its pixels alone.
// =========================================================================
[numthreads(64, 1, 1)]
void CSIntraT(uint3 dtid : SV_DispatchThreadID) {
    uint bw = (width + 3u) >> 2;
    uint bh = (height + 3u) >> 2;
    if (dtid.x >= bw * bh) return;
    // Start at a different block every frame: blocks are taken while the
    // budget lasts, and a fixed order would always starve the same ones.
    uint b = (dtid.x + (frame_id * 40503u) % (bw * bh)) % (bw * bh);
    uint x0 = (b % bw) * 4u;
    uint y0 = (b / bw) * 4u;
    if ((mode_flags & 4u) == 0u || x0 + 4u > width || y0 + 4u > height) {
        g_tflag_out[b] = 0u;
        return;
    }

    uint cycles = max(intra_cycles, 1u);
    uint rows_per_slot = (height + cycles - 1u) / cycles;
    uint iy0 = (frame_id % cycles) * rows_per_slot;
    uint iy1 = iy0 + rows_per_slot;
    bool rolling = (y0 < iy1) && (y0 + 4u > iy0);

    uint cur_words = g_out_counter.Load(0);
    // Same gates as the pixel intra of CSEncode, kept below the hard cap so
    // that deltas always keep part of the budget.
    bool allow_disocc = (max_record_words == 0u) || (cur_words < (max_record_words >> 1));
    bool allow_roll = (max_record_words == 0u) || (cur_words < ((max_record_words * 3u) >> 2));

    int sy[16], sb[16], sr[16];
    int py_[16], pb_[16], pr_[16]; // motion prediction, as the decoder rebuilds it
    uint ndis = 0u;
    int sad = 0;
    [loop] for (uint j = 0; j < 16u; j++) {
        uint px = x0 + (j & 3u);
        uint py = y0 + (j >> 2);
        RgbToYCbCr(g_src_rgba.Load(int3(px, py, 0)).rgb, sy[j], sb[j], sr[j]);
        uint ref_w = FetchRefQ(px, py, MvQ(g_prev_mvs[(py >> 5) * tiles_x + (px >> 5)]));
        int ry = (int)(ref_w & 0xFFu);
        py_[j] = LeakyStep(ry, 0, 0, (int)neutral_y);
        pb_[j] = LeakyStep((int)((ref_w >> 8) & 0xFFu), 0, 0, 128);
        pr_[j] = LeakyStep((int)((ref_w >> 16) & 0xFFu), 0, 0, 128);
        if (abs(sy[j] - ry) > (int)intra_err_thr) ndis++;
        sad += abs(sy[j] - py_[j]);
    }
    bool want_intra = !rolling && allow_disocc && ndis >= 4u;
    // Inter: the motion prediction is off by more than the ternary deltas
    // fix in a frame or two (zoom, rotation, objects moving inside a tile).
    bool allow_inter = (mode_flags & 8u) != 0u &&
                       ((max_record_words == 0u) || (cur_words < ((max_record_words * 3u) >> 2)));
    bool want_inter = !want_intra && !rolling && allow_inter && sad > 16 * (int)tq_inter_thr;
    if (!want_intra && !want_inter) {
        g_tflag_out[b] = 0u;
        return;
    }
    uint pred = want_inter ? 1u : 0u;

    // Inter residuals stay at a usable precision whatever the rate control
    // does: at a coarse qp they quantize to nothing, the block keeps its bad
    // prediction and, skipped by CSEncode, never converges.
    uint qp_y = clamp(tq_qp, 12u, (pred != 0u) ? TQ_INTER_QP_MAX : 51u);
    uint qp_c = min(qp_y + tq_cqp, 51u);
    int lv[48];
    uint n = 0u;
    [unroll] for (uint pl = 0; pl < 3u; pl++) {
        int x[16];
        [unroll] for (uint q = 0; q < 16u; q++) {
            int sv = (pl == 0u) ? sy[q] : (pl == 1u) ? sb[q] : sr[q];
            int pv = (pred == 0u) ? 128 : ((pl == 0u) ? py_[q] : (pl == 1u) ? pb_[q] : pr_[q]);
            x[q] = sv - pv;
        }
        // Forward core transform: rows then columns.
        [unroll] for (uint i = 0; i < 4u; i++) {
            int s0 = x[i * 4u] + x[i * 4u + 3u], s1 = x[i * 4u + 1u] + x[i * 4u + 2u];
            int d0 = x[i * 4u] - x[i * 4u + 3u], d1 = x[i * 4u + 1u] - x[i * 4u + 2u];
            x[i * 4u] = s0 + s1; x[i * 4u + 2u] = s0 - s1;
            x[i * 4u + 1u] = 2 * d0 + d1; x[i * 4u + 3u] = d0 - 2 * d1;
        }
        [unroll] for (uint jc = 0; jc < 4u; jc++) {
            int s0 = x[jc] + x[12u + jc], s1 = x[4u + jc] + x[8u + jc];
            int d0 = x[jc] - x[12u + jc], d1 = x[4u + jc] - x[8u + jc];
            x[jc] = s0 + s1; x[8u + jc] = s0 - s1;
            x[4u + jc] = 2 * d0 + d1; x[12u + jc] = d0 - 2 * d1;
        }
        uint qp = (pl == 0u) ? qp_y : qp_c;
        int qbits = 15 + (int)(qp / 6u);
        int fq = (1 << qbits) / 3;
        [unroll] for (uint k = 0; k < 16u; k++) {
            uint r = TQ_ZZ[k];
            int w = x[r];
            int z = (abs(w) * TQ_MF[(qp % 6u) * 3u + TqClass(r)] + fq) >> qbits;
            z = min(z, 255);
            int l = (w < 0) ? -z : z;
            lv[pl * 16u + k] = l;
            if (l != 0) n++;
        }
    }
    // Reconstruction, exactly as CSITrans of the decoder does it.
    int rec[48];
    [unroll] for (uint pl2 = 0; pl2 < 3u; pl2++) {
        uint qp = (pl2 == 0u) ? qp_y : qp_c;
        int c[16];
        [unroll] for (uint k = 0; k < 16u; k++) {
            uint r = TQ_ZZ[k];
            c[r] = TqDequant(lv[pl2 * 16u + k], r, qp);
        }
        TqInverse(c);
        [unroll] for (uint q = 0; q < 16u; q++) {
            int pv = (pred == 0u) ? 128 : ((pl2 == 0u) ? py_[q] : (pl2 == 1u) ? pb_[q] : pr_[q]);
            rec[pl2 * 16u + q] = clamp(pv + c[q], 0, 255);
        }
    }
    if (pred != 0u) {
        // Inter only when it pays: the block must come out clearly closer to
        // the source than its prediction, otherwise ternary deltas keep it.
        int sad_after = 0;
        [unroll] for (uint q2 = 0; q2 < 16u; q2++) sad_after += abs(sy[q2] - rec[q2]);
        // ... and by at least half a luma level per pixel: on converged content
        // tiny gains would otherwise recode the same blocks frame after frame.
        if (n == 0u || sad_after * 4 > sad * 3 || sad - sad_after < 8) {
            g_tflag_out[b] = 0u;
            return;
        }
    }
    // The luma DC record always goes out: it is what tells the decoder that
    // the block was transform-coded (and resets it to 128 + residual).
    if (lv[0] == 0) n++;

    uint slot = 0u;
    g_out_counter.InterlockedAdd(0, 2u * n, slot);
    if (max_record_words != 0u && slot + 2u * n > max_record_words) {
        // Refused: but the part of the reservation below the cap is inside
        // the frame the CPU reads back, and holds whatever the previous frame
        // left there. Overwrite it with 0xFFFFFFFF, a record nobody applies
        // (its block lies past the frame) and the packers drop.
        [loop] for (uint w = slot; w < max_record_words; w++) g_out_records[w] = 0xFFFFFFFFu;
        g_tflag_out[b] = 0u;
        return;
    }
    [loop] for (uint i2 = 0; i2 < 48u; i2++) {
        if (lv[i2] != 0 || i2 == 0u) {
            uint qp = (i2 < 16u) ? qp_y : qp_c;
            g_out_records[slot]      = 0xF0000000u | (b * 48u + i2);
            g_out_records[slot + 1u] = ((uint)lv[i2] & 0xFFFFu) | (qp << 16) | (pred << 24);
            slot += 2u;
        }
    }

    [unroll] for (uint j2 = 0; j2 < 16u; j2++) {
        uint idx = (y0 + (j2 >> 2)) * width + x0 + (j2 & 3u);
        g_cur_yuvd[idx] = (uint)rec[j2] | ((uint)rec[16u + j2] << 8) |
                          ((uint)rec[32u + j2] << 16) | (delta_min << 24);
    }
    g_tflag_out[b] = 1u;
}
