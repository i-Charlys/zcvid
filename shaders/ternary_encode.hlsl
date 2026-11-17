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

uint TqPositions() {
    return ((((width + 3u) >> 2) * ((height + 3u) >> 2) + 63u) >> 6) << 6;
}
// misc words: +0 total wanted, +1 total v, +2 accepted total (min),
// +3 words used in g_tq_stage
uint TqMisc() {
    uint p = TqPositions();
    return 2u * p + 2u * (p >> 6);
}
uint TqSums() { return 2u * TqPositions(); }
uint TqBases() { uint p = TqPositions(); return 2u * p + (p >> 6); }
uint TqHalfCap() {
    return (max_record_words != 0u) ? (max_record_words >> 1) : 0xFFFFFFFFu;
}
uint TqInterCap() {
    if (max_record_words == 0u) return 0xFFFFFFFFu;
    // a block is at most 48 records: room for the last one below the cap
    return (max_record_words > 96u) ? max_record_words - 96u : 0u;
}



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
// Transform budget, by position q in the frame's rotated block order (P =
// positions = 4x4 blocks rounded up to 64, G = P / 64 groups of them):
// [0, P) words each block wants (bit 31: inter, bit 30: within the first
// half of the cap), [P, 2P) where its records wait in g_tq_stage,
// [2P, 2P + G) per-group sums, [2P + G, 2P + 2G) per-group bases, then the
// words TqMisc() lists. Cleared every frame.
RWStructuredBuffer<uint>  g_tq_scratch  : register(u6);
// Records of every block that wants the transform, in any order (each block
// knows where its own are); the accepted ones are copied to g_out_records.
RWStructuredBuffer<uint>  g_tq_stage    : register(u7);
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
    uint py;          // y coordinate in frame
    int cy, ccb, ccr; // source
    int ry, rcb, rcr; // motion-compensated reference
    int d;            // adaptive step stored at the reference
    int jnd;          // edge-preserving JND deadzone boost from s.ry
};

// The rest of a pixel's state once its reference word and the four
// integer-position luma neighbours of that reference are known.
PixState FinishPix(bool valid, uint px, uint py, uint ref_w, int ry_l, int ry_r, int ry_u, int ry_d,
                   uint intra_y0, uint intra_y1, bool allow_disocc) {
    PixState s;
    s.valid = valid;
    s.px = px;
    s.py = py;
    s.ry  = (int)(ref_w & 0xFFu);
    s.rcb = (int)((ref_w >> 8) & 0xFFu);
    s.rcr = (int)((ref_w >> 16) & 0xFFu);
    s.d   = max((int)delta_min, (int)((ref_w >> 24) & 0xFFu));

    // Closed-loop edge-preserving JND deadzone boost using only reference luma
    // (encoder-only: it moves the deadzone, the decoder never computes it).
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

PixState LoadPix(uint idx, uint num_px, uint intra_y0, uint intra_y1, bool allow_disocc) {
    bool valid = (idx < num_px);
    // No early return: the caller's loop must stay unrollable by fxc.
    idx = min(idx, num_px - 1u);
    uint py = idx / width;
    uint px = idx - py * width;
    int2 mvq = MvQ(g_prev_mvs[(py >> 5) * tiles_x + (px >> 5)]);
    int sx = clamp(((int)px * 4 - mvq.x) >> 2, 0, (int)width - 1);
    int sy = clamp(((int)py * 4 - mvq.y) >> 2, 0, (int)height - 1);
    uint ref_w = FetchRefQ(px, py, mvq);
    int ry_l = (int)(g_prev_yuvd[(uint)sy * width + (uint)max(sx - 1, 0)] & 0xFFu);
    int ry_r = (int)(g_prev_yuvd[(uint)sy * width + (uint)min(sx + 1, (int)width - 1)] & 0xFFu);
    int ry_u = (int)(g_prev_yuvd[(uint)max(sy - 1, 0) * width + (uint)sx] & 0xFFu);
    int ry_d = (int)(g_prev_yuvd[(uint)min(sy + 1, (int)height - 1) * width + (uint)sx] & 0xFFu);
    return FinishPix(valid, px, py, ref_w, ry_l, ry_r, ry_u, ry_d, intra_y0, intra_y1, allow_disocc);
}

// Both pixels of a pair (idx_a even). They share a row and a tile, hence the
// vector and the filter phase: one 5x4 window of the reference serves both
// filters, and its middle words are the integer neighbours the JND needs
// (20 reads instead of 40; 8 instead of 10 for a whole-pel vector). The
// filtered values are the same integers FetchRefQ computes.
// (Returned as a struct: an earlier version with out-parameters and an early
// return compiled into a shader that coded differently.)
struct PixPair {
    PixState a;
    PixState b;
};

PixPair LoadPair(uint idx_a, uint num_px, uint intra_y0, uint intra_y1, bool allow_disocc) {
    PixPair r;
    uint py = min(idx_a, num_px - 1u) / width;
    uint px = min(idx_a, num_px - 1u) - py * width;
    if (idx_a + 1u >= num_px || px + 1u >= width) {
        r.a = LoadPix(idx_a, num_px, intra_y0, intra_y1, allow_disocc);
        r.b = LoadPix(idx_a + 1u, num_px, intra_y0, intra_y1, allow_disocc);
    } else {
    int2 mvq = MvQ(g_prev_mvs[(py >> 5) * tiles_x + (px >> 5)]);
    int sxq = (int)px * 4 - mvq.x;
    int syq = (int)py * 4 - mvq.y;
    int ix = sxq >> 2;
    int iy = syq >> 2;
    int fx = sxq & 3;
    int fy = syq & 3;
    uint wa, wb;
    int r0, r1, r2, r3, u1, u2, d1, d2; // lumas: row iy at ix-1..ix+2, rows iy-1 / iy+1 at ix, ix+1
    if ((fx | fy) == 0) {
        wa = RefWordAt(ix, iy);
        wb = RefWordAt(ix + 1, iy);
        r0 = (int)(RefWordAt(ix - 1, iy) & 0xFFu);
        r1 = (int)(wa & 0xFFu);
        r2 = (int)(wb & 0xFFu);
        r3 = (int)(RefWordAt(ix + 2, iy) & 0xFFu);
        u1 = (int)(RefWordAt(ix, iy - 1) & 0xFFu);
        u2 = (int)(RefWordAt(ix + 1, iy - 1) & 0xFFu);
        d1 = (int)(RefWordAt(ix, iy + 1) & 0xFFu);
        d2 = (int)(RefWordAt(ix + 1, iy + 1) & 0xFFu);
    } else {
        int4 tx = TapsQ(fx);
        int4 ty = TapsQ(fy);
        int a0 = 0, a1 = 0, a2 = 0, b0 = 0, b1 = 0, b2 = 0;
        uint w[20];
        [unroll] for (int j = 0; j < 4; j++) {
            [unroll] for (int i = 0; i < 5; i++) w[j * 5 + i] = RefWordAt(ix - 1 + i, iy - 1 + j);
            int tv = (j == 0) ? ty.x : (j == 1) ? ty.y : (j == 2) ? ty.z : ty.w;
            int ha0 = 0, ha1 = 0, ha2 = 0, hb0 = 0, hb1 = 0, hb2 = 0;
            [unroll] for (int i2 = 0; i2 < 4; i2++) {
                int t = (i2 == 0) ? tx.x : (i2 == 1) ? tx.y : (i2 == 2) ? tx.z : tx.w;
                uint ua = w[j * 5 + i2];
                uint ub = w[j * 5 + i2 + 1];
                ha0 += t * (int)(ua & 0xFFu); ha1 += t * (int)((ua >> 8) & 0xFFu); ha2 += t * (int)((ua >> 16) & 0xFFu);
                hb0 += t * (int)(ub & 0xFFu); hb1 += t * (int)((ub >> 8) & 0xFFu); hb2 += t * (int)((ub >> 16) & 0xFFu);
            }
            a0 += tv * ha0; a1 += tv * ha1; a2 += tv * ha2;
            b0 += tv * hb0; b1 += tv * hb1; b2 += tv * hb2;
        }
        wa = (w[6] & 0xFF000000u) |
             (uint)clamp((a0 + 2048) >> 12, 0, 255) |
             ((uint)clamp((a1 + 2048) >> 12, 0, 255) << 8) |
             ((uint)clamp((a2 + 2048) >> 12, 0, 255) << 16);
        wb = (w[7] & 0xFF000000u) |
             (uint)clamp((b0 + 2048) >> 12, 0, 255) |
             ((uint)clamp((b1 + 2048) >> 12, 0, 255) << 8) |
             ((uint)clamp((b2 + 2048) >> 12, 0, 255) << 16);
        r0 = (int)(w[5] & 0xFFu); r1 = (int)(w[6] & 0xFFu); r2 = (int)(w[7] & 0xFFu); r3 = (int)(w[8] & 0xFFu);
        u1 = (int)(w[1] & 0xFFu); u2 = (int)(w[2] & 0xFFu);
        d1 = (int)(w[11] & 0xFFu); d2 = (int)(w[12] & 0xFFu);
    }
    r.a = FinishPix(true, px, py, wa, r0, r2, u1, d1, intra_y0, intra_y1, allow_disocc);
    r.b = FinishPix(true, px + 1u, py, wb, r1, r3, u2, d2, intra_y0, intra_y1, allow_disocc);
    }
    return r;
}

// A step is only taken when it brings the reference closer to the source:
// the error must exceed the threshold and half the step d. Stepping with a
// step larger than twice the error overshoots, and since a non-zero luma trit
// doubles the step, the reference then swings around its target. Declining
// codes a 0, which lets the step halve. Encoder-only decision: the decoder
// applies whatever trits it receives.
// 1 when CSIntraT coded the 4x4 block holding this pixel.
uint TFlagAtXY(uint px, uint py) {
    if ((mode_flags & 4u) == 0u) return 0u;
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

// One pixel pair (idx_a even, pair k of its block). Writes the pair's
// reference and returns its trits for the block's delta records:
// bits 0-11 = (trit + 1) for Y a, Y b, Cb a, Cb b, Cr a, Cr b (2 bits each),
// bit 12 = transform-coded by CSIntraT (a dropped delta record must not
// revert it), bit 13 = an intra record is pending in rec0 / rec1.
// An intra pair (2-px, or 4-px rolling taking the next pair too) is written
// as plain decay here; CSEncode's second phase reserves its record together
// with the block's delta records and, if the record fits the cap, overwrites
// the decay with the intra samples (IntraRecon).
uint EncodePair(uint idx_a, uint k, uint num_pairs, uint num_px, uint intra_y0, uint intra_y1,
                bool allow_disocc, bool allow_chroma, bool budget_full, bool intra8, uint is_2x,
                out uint rec0, out uint rec1) {
    rec0 = 0u;
    rec1 = 0u;
    uint idx_b = idx_a + 1u;
    int y_thr = (int)threshold;
    int c_thr = (int)threshold * 2 + 4;

    // An odd pair is taken by its even neighbour when that one codes a 4-px
    // rolling intra. The neighbour's choice only depends on geometry, flags
    // and the group's budget, so it is recomputed here rather than waited for.
    if ((k & 1u) != 0u && !budget_full && !intra8) {
        uint il = idx_a - 2u;
        uint ly = il / width;
        uint lx = il - ly * width;
        if (ly >= intra_y0 && ly < intra_y1 && lx + 3u < width && il + 3u < num_px &&
            TFlagAtXY(lx, ly) == 0u && TFlagAtXY(lx + 2u, ly) == 0u) {
            return 0x555u;
        }
    }

    PixPair pr = LoadPair(idx_a, num_px, intra_y0, intra_y1, allow_disocc);
    PixState a = pr.a;
    PixState b = pr.b;

    int ty_a = 0, tcb_a = 0, tcr_a = 0;
    int ty_b = 0, tcb_b = 0, tcr_b = 0;
    uint flags = 0u;
    bool same_row_2 = (a.px + 1u < width);
    bool same_row_4 = (a.px + 3u < width);

    // A pair never straddles two 4x4 blocks (both start on even x).
    bool tcoded = a.valid && (TFlagAtXY(a.px, a.py) != 0u);
    if (tcoded) {
        // CSIntraT already wrote the reference and the records.
        flags = 0x1000u;
    } else if (a.valid && same_row_2 && (a.want_intra || b.want_intra)) {
        StoreDelta(idx_a, a, 0, 0, 0, a.d, max((int)delta_min, a.d >> 1));
        if (b.valid) StoreDelta(idx_b, b, 0, 0, 0, b.d, max((int)delta_min, b.d >> 1));
        if (budget_full) {
            // decay only
        } else if ((k & 1u) == 0u && (k + 1u) < num_pairs && a.rolling && !intra8 && same_row_4 && (idx_a + 3u < num_px) &&
                   TFlagAtXY(a.px + 2u, a.py) == 0u) {
            PixState c = LoadPix(idx_a + 2u, num_px, intra_y0, intra_y1, allow_disocc);
            PixState d = LoadPix(idx_a + 3u, num_px, intra_y0, intra_y1, allow_disocc);
            StoreDelta(idx_a + 2u, c, 0, 0, 0, c.d, max((int)delta_min, c.d >> 1));
            StoreDelta(idx_a + 3u, d, 0, 0, 0, d.d, max((int)delta_min, d.d >> 1));
            // Mode bit 6: ordered (Bayer 4x4) dither instead of plain
            // rounding to 6 bits, so a smooth gradient turns into fine
            // noise rather than 4-level bands. A constant 8 is the plain
            // rounding. Encoder-only: the records carry the quantized
            // values whatever the rounding.
            bool dith = (mode_flags & 64u) != 0u;
            uint brow = (a.py & 3u) * 4u;
            int y0_q6 = clamp((4 * a.cy + (dith ? (int)BAYER4[brow + ((a.px + 0u) & 3u)] : 8)) >> 4, 0, 63);
            int y1_q6 = clamp((4 * b.cy + (dith ? (int)BAYER4[brow + ((a.px + 1u) & 3u)] : 8)) >> 4, 0, 63);
            int y2_q6 = clamp((4 * c.cy + (dith ? (int)BAYER4[brow + ((a.px + 2u) & 3u)] : 8)) >> 4, 0, 63);
            int y3_q6 = clamp((4 * d.cy + (dith ? (int)BAYER4[brow + ((a.px + 3u) & 3u)] : 8)) >> 4, 0, 63);
            int cb_q6 = clamp(((a.ccb + b.ccb + c.ccb + d.ccb + 8) >> 4), 0, 63);
            int cr_q6 = clamp(((a.ccr + b.ccr + c.ccr + d.ccr + 8) >> 4), 0, 63);
            // idx_a is even (blocks start on even pixels, k is even): the
            // 22-bit field holds the pixel-pair index, good for 8.3 Mpx (4K).
            rec0 = 0xE0000000u | ((uint)cr_q6 << 22) | ((idx_a >> 1) & 0x3FFFFFu);
            rec1 = (uint)y0_q6 | ((uint)y1_q6 << 6) | ((uint)y2_q6 << 12) | ((uint)y3_q6 << 18) | ((uint)cb_q6 << 24);
            flags = 0x2000u;
        } else {
            int cy_b = b.valid ? b.cy : a.cy;
            int cb = b.valid ? ((a.ccb + b.ccb + 1) >> 1) : a.ccb;
            int cr = b.valid ? ((a.ccr + b.ccr + 1) >> 1) : a.ccr;
            rec0 = 0xC0000000u | (idx_a & 0x1FFFFFFFu);
            rec1 = (uint)a.cy | ((uint)cb << 8) | ((uint)cy_b << 16) | ((uint)cr << 24);
            flags = 0x2000u;
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
    return (uint)(ty_a + 1) | ((uint)(ty_b + 1) << 2) | ((uint)(tcb_a + 1) << 4) |
           ((uint)(tcb_b + 1) << 6) | ((uint)(tcr_a + 1) << 8) | ((uint)(tcr_b + 1) << 10) | flags;
}

// The reference an accepted intra record leaves, exactly as the decoder
// rebuilds it from the record. Returns the pixel mask (bits from the pair's
// first pixel) it covers.
uint IntraRecon(uint r0, uint r1, uint num_px) {
    uint dm = delta_min << 24;
    if ((r0 >> 29) == 7u) {
        // 4-px rolling intra: 6-bit luma x4, shared 6-bit chroma.
        uint idx_a = (r0 & 0x3FFFFFu) << 1;
        uint cb_q6 = (r1 >> 24) & 63u;
        uint cr_q6 = (r0 >> 22) & 63u;
        uint chroma_d = (((cb_q6 << 2) | (cb_q6 >> 4)) << 8) | (((cr_q6 << 2) | (cr_q6 >> 4)) << 16) | dm;
        [unroll] for (uint i = 0; i < 4u; i++) {
            uint q = (r1 >> (6u * i)) & 63u;
            g_cur_yuvd[idx_a + i] = ((q << 2) | (q >> 4)) | chroma_d;
        }
        return 0xFu;
    }
    // 2-px intra: 8-bit luma x2, shared 8-bit chroma.
    uint idx_a = r0 & 0x1FFFFFFFu;
    uint chroma_d = (r1 & 0x0000FF00u) | (((r1 >> 24) & 0xFFu) << 16) | dm;
    g_cur_yuvd[idx_a] = (r1 & 0xFFu) | chroma_d;
    if (idx_a + 1u < num_px) g_cur_yuvd[idx_a + 1u] = ((r1 >> 16) & 0xFFu) | chroma_d;
    return 3u;
}

// One group = 64 consecutive blocks (in the frame's rotated block order).
// Phase A: one thread per pixel pair, neighbouring threads on neighbouring
// pairs, so reads and writes coalesce; each pair leaves its trits (and any
// intra record) in groupshared. Phase B: one thread per block reserves all
// of the block's records with one atomic, in the block's order (intra
// records, then Y, Cb, Cr deltas), and writes those that fit the cap.
// The budget is read once per group, so every pair of a block (and the even
// pair a 4-px rolling intra takes its neighbour from) sees the same one.
groupshared uint gs_pair[640];
groupshared uint gs_rec0[640];
groupshared uint gs_rec1[640];
groupshared uint gs_words;

[numthreads(64, 1, 1)]
void CSEncode(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint is_2x = mode_flags & 1u;
    uint eff_blocks = (is_2x != 0u) ? ((total_blocks + 1u) >> 1) : total_blocks;
    // Rotate starting block per frame so hard-cap budgets never starve the bottom of the frame.
    uint rot = (frame_id * 29713u) % eff_blocks;
    uint num_px = width * height;
    uint cycles = max(intra_cycles, 1u);
    uint rows_per_slot = (height + cycles - 1u) / cycles;
    uint intra_y0 = (frame_id % cycles) * rows_per_slot;
    uint intra_y1 = intra_y0 + rows_per_slot;
    // 1x: 5 pairs, 2 trits each (10 px). 2x: 10 pairs, 1 trit each (20 px).
    uint px_per_blk = (is_2x != 0u) ? 20u : 10u;
    uint num_pairs = (is_2x != 0u) ? 10u : 5u;

    // Hard rate control: g_out_counter counts total desired words (for CPU RC),
    // while g_out_records and g_cur_yuvd only commit records within max_record_words.
    // The gates read what the transform passes took (CSTqFinal's total),
    // refused demand included, not the live counter: the live value depended on how far
    // other groups had got, so the same frame coded differently run to run.
    // Under overload the deltas give way, as they always did.
    if (gtid.x == 0u) {
        gs_words = g_tq_scratch[TqMisc() + 2u] + g_out_counter.Load(4);
    }
    GroupMemoryBarrierWithGroupSync();
    uint cur_words = gs_words;
    bool budget_full = (max_record_words != 0u) && (cur_words >= max_record_words);
    // Disocclusion intra up to half the budget: newly uncovered areas are
    // repaired in one frame instead of converging over several at delta_min.
    bool allow_disocc = (max_record_words == 0u) || (cur_words < (max_record_words >> 1));
    bool allow_chroma = (max_record_words == 0u) || (cur_words < max_record_words);
    // The rate control has budget to spare: rolling refresh keeps full 8-bit
    // samples and per-pair chroma (2-px intra) instead of 6-bit 4-px records.
    bool intra8 = (mode_flags & 2u) != 0u;

    // A real loop: unrolled, every iteration would inline the motion
    // compensation filter several times over.
    [loop]
    for (uint round = 0; round < num_pairs; round++) {
        uint p = round * 64u + gtid.x;
        uint lb = p / num_pairs;
        uint k = p - lb * num_pairs;
        uint gb = gid.x * 64u + lb;
        uint code = 0x555u;
        uint r0 = 0u, r1 = 0u;
        if (gb < eff_blocks) {
            uint blk = (gb + rot) % eff_blocks;
            code = EncodePair(blk * px_per_blk + k * 2u, k, num_pairs, num_px, intra_y0, intra_y1,
                              allow_disocc, allow_chroma, budget_full, intra8, is_2x, r0, r1);
        }
        gs_pair[p] = code;
        gs_rec0[p] = r0;
        gs_rec1[p] = r1;
    }
    // Phase B overwrites and reads back references other threads wrote.
    AllMemoryBarrierWithGroupSync();

    uint gb = gid.x * 64u + gtid.x;
    if (gb >= eff_blocks) return;
    uint blk = (gb + rot) % eff_blocks;
    uint base_px = blk * px_per_blk;
    uint g0 = gtid.x * num_pairs;

    uint packed_y  = 0;
    uint packed_cb = 0;
    uint packed_cr = 0;
    bool nz_y  = false;
    bool nz_cb = false;
    bool nz_cr = false;
    uint intra_mask = 0u;
    uint n_intra = 0u;
    [unroll] for (uint k = 0; k < 10u; k++) {
        uint c = (k < num_pairs) ? gs_pair[g0 + k] : 0x555u;
        uint ya = c & 3u, yb = (c >> 2) & 3u;
        uint cba = (c >> 4) & 3u, cbb = (c >> 6) & 3u;
        uint cra = (c >> 8) & 3u, crb = (c >> 10) & 3u;
        if ((c & 0x1000u) != 0u) intra_mask |= (3u << (k * 2u));
        if ((c & 0x2000u) != 0u) n_intra++;
        if (ya != 1u || yb != 1u)   nz_y  = true;
        if (cba != 1u || cbb != 1u) nz_cb = true;
        if (cra != 1u || crb != 1u) nz_cr = true;
        if (is_2x != 0u) {
            packed_y  += ya  * POW3[k];
            packed_cb += cba * POW3[k];
            packed_cr += cra * POW3[k];
        } else if (k < 5u) {
            uint j = k * 2u;
            packed_y  += ya  * POW3[j] + yb  * POW3[j + 1u];
            packed_cb += cba * POW3[j] + cbb * POW3[j + 1u];
            packed_cr += cra * POW3[j] + crb * POW3[j + 1u];
        }
    }

    uint n_rec = n_intra + (nz_y ? 1u : 0u) + (nz_cb ? 1u : 0u) + (nz_cr ? 1u : 0u);
    if (n_rec == 0u) return;
    uint slot = 0u;
    g_out_counter.InterlockedAdd(0, 2u * n_rec, slot);

    // Intra records first, in pair order; a record that does not fit leaves
    // the pair at the decay EncodePair already wrote.
    if (n_intra != 0u) {
        [loop] for (uint k2 = 0; k2 < num_pairs; k2++) {
            if ((gs_pair[g0 + k2] & 0x2000u) != 0u) {
                if (max_record_words == 0u || slot + 2u <= max_record_words) {
                    uint r0 = gs_rec0[g0 + k2];
                    uint r1 = gs_rec1[g0 + k2];
                    g_out_records[slot + 0u] = r0;
                    g_out_records[slot + 1u] = r1;
                    intra_mask |= IntraRecon(r0, r1, num_px) << (k2 * 2u);
                }
                slot += 2u;
            }
        }
    }

    uint hdr_base = (is_2x << 29) | (blk & 0x1FFFFFFFu);
    bool drop_y = false, drop_cb = false, drop_cr = false;
    if (nz_y) {
        if (max_record_words == 0u || slot + 2u <= max_record_words) {
            g_out_records[slot + 0u] = (0u << 30) | hdr_base;
            g_out_records[slot + 1u] = packed_y;
        } else {
            drop_y = true;
        }
        slot += 2u;
    }
    if (nz_cb) {
        if (max_record_words == 0u || slot + 2u <= max_record_words) {
            g_out_records[slot + 0u] = (1u << 30) | hdr_base;
            g_out_records[slot + 1u] = packed_cb;
        } else {
            drop_cb = true;
        }
        slot += 2u;
    }
    if (nz_cr) {
        if (max_record_words == 0u || slot + 2u <= max_record_words) {
            g_out_records[slot + 0u] = (2u << 30) | hdr_base;
            g_out_records[slot + 1u] = packed_cr;
        } else {
            drop_cr = true;
        }
        slot += 2u;
    }

    // Boundary safety: if a delta record exceeded max_record_words, revert its
    // plane(s) in g_cur_yuvd to pure motion-compensated decay so Enc == Dec.
    if (drop_y || drop_cb || drop_cr) {
        for (uint p = 0; p < px_per_blk; p++) {
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
// The budget is shared out without any read of the record counter during
// the pass, which made the outcome depend on how fast the GPU ran the
// threads (a faster shader closed the gates earlier, and lost up to 0.3 dB):
// CSIntraTCount writes what each block wants, CSTqScan replays the old rules
// in the frame's rotated block order and gives every accepted block its
// offset, CSIntraT writes the accepted blocks there.
void IntraTBlock(uint q) {
    uint bw = (width + 3u) >> 2;
    uint bh = (height + 3u) >> 2;
    uint nb = bw * bh;
    if (q >= nb) {
        g_tq_scratch[q] = 0u;
        return;
    }
    // Start at a different block every frame: blocks are taken while the
    // budget lasts, and a fixed order would always starve the same ones.
    uint b = (q + (frame_id * 40503u) % nb) % nb;
    uint x0 = (b % bw) * 4u;
    uint y0 = (b / bw) * 4u;
    if ((mode_flags & 4u) == 0u || x0 + 4u > width || y0 + 4u > height) {
        g_tq_scratch[q] = 0u;
        return;
    }

    uint cycles = max(intra_cycles, 1u);
    uint rows_per_slot = (height + cycles - 1u) / cycles;
    uint iy0 = (frame_id % cycles) * rows_per_slot;
    uint iy1 = iy0 + rows_per_slot;
    bool rolling = (y0 < iy1) && (y0 + 4u > iy0);

    // The budget limits are CSTqScan's; here every block says what it wants.
    bool allow_disocc = true;
    bool allow_inter = (mode_flags & 8u) != 0u;
    // Rolling rows are left to CSEncode (pixel intra).
    if (rolling) {
        g_tq_scratch[q] = 0u;
        return;
    }

    // The block lies in one 32x32 tile (4 divides 32): one vector and one
    // filter phase for its 16 pixels, so a single 7x7 window of the reference
    // feeds them all (49 reads instead of 16 x 16). Same integers as FetchRefQ.
    uint refw[16];
    {
        int2 mvq = MvQ(g_prev_mvs[(y0 >> 5) * tiles_x + (x0 >> 5)]);
        int sxq = (int)x0 * 4 - mvq.x;
        int syq = (int)y0 * 4 - mvq.y;
        int bx = sxq >> 2;
        int by = syq >> 2;
        int fx = sxq & 3;
        int fy = syq & 3;
        if ((fx | fy) == 0) {
            [unroll] for (uint j0 = 0; j0 < 16u; j0++) refw[j0] = RefWordAt(bx + (int)(j0 & 3u), by + (int)(j0 >> 2));
        } else {
            int4 tx = TapsQ(fx);
            int4 ty = TapsQ(fy);
            int acc[48];
            [unroll] for (uint z = 0; z < 48u; z++) acc[z] = 0;
            [unroll] for (int r = 0; r < 7; r++) {
                uint w[7];
                [unroll] for (int c = 0; c < 7; c++) w[c] = RefWordAt(bx - 1 + c, by - 1 + r);
                [unroll] for (int ox = 0; ox < 4; ox++) {
                    int h0 = 0, h1 = 0, h2 = 0;
                    [unroll] for (int i = 0; i < 4; i++) {
                        int t = (i == 0) ? tx.x : (i == 1) ? tx.y : (i == 2) ? tx.z : tx.w;
                        uint u = w[ox + i];
                        h0 += t * (int)(u & 0xFFu);
                        h1 += t * (int)((u >> 8) & 0xFFu);
                        h2 += t * (int)((u >> 16) & 0xFFu);
                    }
                    // source row r feeds output rows r - 3 .. r, with tap r - oy
                    [unroll] for (int oy = 0; oy < 4; oy++) {
                        int jj = r - oy;
                        if (jj >= 0 && jj < 4) {
                            int tv = (jj == 0) ? ty.x : (jj == 1) ? ty.y : (jj == 2) ? ty.z : ty.w;
                            acc[(oy * 4 + ox) * 3 + 0] += tv * h0;
                            acc[(oy * 4 + ox) * 3 + 1] += tv * h1;
                            acc[(oy * 4 + ox) * 3 + 2] += tv * h2;
                        }
                    }
                }
            }
            [unroll] for (uint q0 = 0; q0 < 16u; q0++) {
                refw[q0] = (uint)clamp((acc[q0 * 3u] + 2048) >> 12, 0, 255) |
                           ((uint)clamp((acc[q0 * 3u + 1u] + 2048) >> 12, 0, 255) << 8) |
                           ((uint)clamp((acc[q0 * 3u + 2u] + 2048) >> 12, 0, 255) << 16);
            }
        }
    }

    int sy[16], sb[16], sr[16];
    int py_[16], pb_[16], pr_[16]; // motion prediction, as the decoder rebuilds it
    uint ndis = 0u;
    int sad = 0;
    [unroll] for (uint j = 0; j < 16u; j++) {
        uint px = x0 + (j & 3u);
        uint py = y0 + (j >> 2);
        RgbToYCbCr(g_src_rgba.Load(int3(px, py, 0)).rgb, sy[j], sb[j], sr[j]);
        uint ref_w = refw[j];
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
    bool want_inter = !want_intra && !rolling && allow_inter && sad > 16 * (int)tq_inter_thr;
    if (!want_intra && !want_inter) {
        g_tq_scratch[q] = 0u;
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
            g_tq_scratch[q] = 0u;
            return;
        }
    }
    // The luma DC record always goes out: it is what tells the decoder that
    // the block was transform-coded (and resets it to 128 + residual).
    if (lv[0] == 0) n++;

    // Records into the staging buffer, reconstruction straight into the
    // reference: if the budget refuses the block, CSEncode codes its pixels
    // like any others and overwrites it.
    uint slot = 0u;
    uint stage_n = 0u, stride = 0u;
    g_tq_stage.GetDimensions(stage_n, stride);
    InterlockedAdd(g_tq_scratch[TqMisc() + 3u], 2u * n, slot);
    if (slot + 2u * n > stage_n) {
        g_tq_scratch[q] = 0u;
        return;
    }
    g_tq_scratch[q] = (2u * n) | (pred << 31);
    g_tq_scratch[TqPositions() + q] = slot;
    [loop] for (uint i2 = 0; i2 < 48u; i2++) {
        if (lv[i2] != 0 || i2 == 0u) {
            uint qp = (i2 < 16u) ? qp_y : qp_c;
            g_tq_stage[slot]      = 0xF0000000u | (b * 48u + i2);
            g_tq_stage[slot + 1u] = ((uint)lv[i2] & 0xFFFFu) | (qp << 16) | (pred << 24);
            slot += 2u;
        }
    }

    [unroll] for (uint j2 = 0; j2 < 16u; j2++) {
        uint idx = (y0 + (j2 >> 2)) * width + x0 + (j2 & 3u);
        g_cur_yuvd[idx] = (uint)rec[j2] | ((uint)rec[16u + j2] << 8) |
                          ((uint)rec[32u + j2] << 16) | (delta_min << 24);
    }
}

// --- Budget: which blocks get transform-coded, and where their records go.
// The rule is the one the encoder used to apply with counter reads, now in
// the fixed rotated order: every block while what the blocks before it want
// stays under half the cap; after that inter blocks only, while what was
// accepted stays under TqInterCap(). Both sums only grow, so every decision
// is a prefix sum: A(q) = words wanted before q decides the first rule;
// v(q) = the words q contributes if it is accepted under the first rule or
// is inter, and S(q) = sum of v before q is its offset. An inter block past
// the first rule is accepted iff S(q) < TqInterCap() (once S reaches it,
// later S are larger still). 6 dispatches: Count, ScanA, Mid, ScanV,
// CSIntraT, Final; the two scans run over per-group sums only.
groupshared uint gs_tq[64];

// Exclusive prefix sum over the 64 threads of a group; returns the sum
// before this thread, and the group total in gs_tq[63] + own value.
uint GroupExclusive64(uint t, uint v) {
    gs_tq[t] = v;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint o = 1u; o < 64u; o <<= 1) {
        uint add = (t >= o) ? gs_tq[t - o] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gs_tq[t] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    return gs_tq[t] - v;
}

[numthreads(64, 1, 1)]
void CSIntraTCount(uint3 dtid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    IntraTBlock(dtid.x);
    DeviceMemoryBarrierWithGroupSync();
    uint w = g_tq_scratch[dtid.x] & 0x3FFFFFFFu;
    uint ex = GroupExclusive64(gtid.x, w);
    if (gtid.x == 63u) g_tq_scratch[TqSums() + gid.x] = ex + w;
}

// One group over the per-group sums: their exclusive prefix (the group
// bases) and the total. first = true: the sums of wanted words; false: of v.
groupshared uint gs_scan[1024];
void ScanGroups(uint t, bool first) {
    uint p = TqPositions();
    uint ng = p >> 6;
    uint per = (ng + 1023u) >> 10;
    uint g0 = min(t * per, ng);
    uint g1 = min(g0 + per, ng);
    uint sum = 0u;
    for (uint g = g0; g < g1; g++) sum += g_tq_scratch[TqSums() + g];
    gs_scan[t] = sum;
    GroupMemoryBarrierWithGroupSync();
    [loop] for (uint o = 1u; o < 1024u; o <<= 1) {
        uint add = (t >= o) ? gs_scan[t - o] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gs_scan[t] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    uint run = gs_scan[t] - sum;
    for (uint g2 = g0; g2 < g1; g2++) {
        uint v = g_tq_scratch[TqSums() + g2];
        g_tq_scratch[TqBases() + g2] = run;
        run += v;
    }
    if (t == 1023u) {
        uint m = TqMisc();
        if (first) {
            g_tq_scratch[m] = run;
        } else {
            g_tq_scratch[m + 1u] = run;
            g_tq_scratch[m + 2u] = run;
        }
    }
}

[numthreads(1024, 1, 1)]
void CSTqScanA(uint3 gtid : SV_GroupThreadID) {
    ScanGroups(gtid.x, true);
}

[numthreads(1024, 1, 1)]
void CSTqScanV(uint3 gtid : SV_GroupThreadID) {
    ScanGroups(gtid.x, false);
}

// v of a position, from its wanted words and flags.
uint TqV(uint c) {
    uint w = c & 0x3FFFFFFFu;
    bool inter = (c >> 31) != 0u;
    bool first = ((c >> 30) & 1u) != 0u;
    return (first || inter) ? w : 0u;
}

[numthreads(64, 1, 1)]
void CSTqMid(uint3 dtid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint p = TqPositions();
    uint c = g_tq_scratch[dtid.x];
    uint w = c & 0x3FFFFFFFu;
    uint a = g_tq_scratch[TqBases() + gid.x] + GroupExclusive64(gtid.x, w);
    if (w != 0u && a < TqHalfCap()) c |= 0x40000000u;
    g_tq_scratch[dtid.x] = c;
    uint v = TqV(c);
    uint ex = GroupExclusive64(gtid.x, v);
    if (gtid.x == 63u) g_tq_scratch[TqSums() + gid.x] = ex + v;
}

[numthreads(64, 1, 1)]
void CSIntraT(uint3 dtid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint p = TqPositions();
    uint c = g_tq_scratch[dtid.x];
    uint w = c & 0x3FFFFFFFu;
    uint v = TqV(c);
    uint sq = g_tq_scratch[TqBases() + gid.x] + GroupExclusive64(gtid.x, v);
    bool first = ((c >> 30) & 1u) != 0u;
    bool ok = (w != 0u) && (first || ((c >> 31) != 0u && sq < TqInterCap()));
    // Where the inter limit is crossed, the accepted total ends.
    if (ok && !first && sq + w >= TqInterCap()) {
        uint dummy;
        InterlockedMin(g_tq_scratch[TqMisc() + 2u], sq + w, dummy);
    }
    uint nb = ((width + 3u) >> 2) * ((height + 3u) >> 2);
    if (dtid.x >= nb) return;
    uint b = (dtid.x + (frame_id * 40503u) % nb) % nb;
    if (ok) {
        uint from = g_tq_scratch[p + dtid.x];
        [loop] for (uint i = 0u; i < w; i++) g_out_records[sq + i] = g_tq_stage[from + i];
    }
    g_tflag_out[b] = ok ? 1u : 0u;
}

// Starts the record counter where the transform records end, and leaves
// the demand the budget refused for the rate control.
[numthreads(1, 1, 1)]
void CSTqFinal() {
    uint m = TqMisc();
    uint total = g_tq_scratch[m + 2u];
    g_out_counter.Store(0, total);
    g_out_counter.Store(4, (g_tq_scratch[m] - min(total, g_tq_scratch[m])) >> 1);
}
