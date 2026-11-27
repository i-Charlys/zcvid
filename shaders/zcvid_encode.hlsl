// zcvid_encode.hlsl - D3D11 Compute Shader (cs_5_0)
// Full-Color (YCbCr 4:4:4 / 4:2:0-ready) + 32x32 Tile MV + Jayant ADM +
// Rolling Intra-Refresh & Disocclusion Repair, Zero-Atomic-Contention Ref.
//
// Pass 0: CSLumaHalf / CSLumaQuarter -> box-filtered luma of the source at 1/2 and 1/4 resolution
// Pass 1: CSTileMV     -> Estimates (dx, dy) per 32x32 tile into g_tile_mvs (wide quarter-pel)
// Pass 1a: CSMotionPred -> Single-pass motion compensation + leaky decay into g_pred_yuvd & g_cur_yuvd
// Pass 1b: CSIntraT*   -> 2-tier priority 4x4 integer transform (intra/inter)
// Pass 2: CSEncode     -> Encodes Y, Cb, Cr 10-trit blocks + Intra blocks
//                         1 uint32 per pixel in g_ref_yuvd: [Y:8 | Cb:8 | Cr:8 | delta:8]

cbuffer ZcvidParams : register(b0) {
    uint width;
    uint height;
    uint threshold;
    uint delta_min;
    uint delta_max;
    uint gamma_q8;
    uint neutral_y;
    uint total_blocks;   // ceil(width * height / 10)
    uint frame_id;
    uint intra_cycles;   // e.g. 60..480 (1/cycles of rows refreshed per frame)
    uint intra_err_thr;  // e.g. 24 (disocclusion threshold -> instant Intra)
    uint tiles_x;        // ceil(width / 32)
    uint tiles_y;        // ceil(height / 32)
    uint search_r;       // adaptive search radius in whole pixels (24..124)
    uint max_record_words; // Rate-control cap in uint32 words (0 = unlimited)
    uint mode_flags;     // bit 0 = DELTA_2X mode (20 px per 10-trit block)
                         // bit 1 = rolling refresh in 8-bit 2-px intra
                         // bit 2 = transform intra (CSIntraT) on 4x4 blocks
                         // bit 3 = transform inter: residual vs motion prediction
                         // bit 4 = tile vectors in quarter pels
                         // bit 5 = reserved
                         // bit 6 = ordered dither on the 6-bit rolling intra
    uint tq_qp;          // luma qp of the transform (12..51)
    uint tq_cqp;         // chroma qp offset of the transform (3..8)
    uint tq_inter_thr;   // mean |luma residual| above which a block is transform-coded
    uint tq_hi_thr;      // mean |luma residual| for Tier-1 high-priority transform admission
};

uint TqPositions() {
    return ((((width + 3u) >> 2) * ((height + 3u) >> 2) + 63u) >> 6) << 6;
}
// g_tq_scratch layout (P = TqPositions(), G = P / 64):
//   [0, P)                 c: wanted words (bits 0..23) | RDO bin 0..15 (bits 24..27)
//   [P, 2P)                stage offset in g_tq_stage
//   [2P, 2P + G)           per-group sums of Tier 1 (bin > cut_bin)
//   [2P + G, 2P + 2G)      per-group exclusive bases of Tier 1
//   [2P + 2G, 2P + 3G)     per-group sums of Tier 2 (bin == cut_bin)
//   [2P + 3G, 2P + 4G)     per-group exclusive bases of Tier 2
//   [2P + 4G, 2P + 4G + 16) 16-bin RDO word histogram (hist[0..15])
//   [2P + 4G + 16 .. +23)  misc words:
//     +0 total_wanted (all blocks, bins 0..15)
//     +1 stage_alloc  (words allocated in g_tq_stage)
//     +2 hi_total     (Tier-1 total in CSTqScanA; updated to final_total in CSTqFinal)
//     +3 lo_end       (max dst + w for admitted Tier-2 blocks in CSIntraT)
//     +4 cut_bin      (0..15: bin > cut_bin all admitted, bin == cut_bin fills remainder)
uint TqSumsHi()  { return 2u * TqPositions(); }
uint TqBasesHi() { uint p = TqPositions(); return 2u * p + (p >> 6); }
uint TqSumsEq()  { uint p = TqPositions(); return 2u * p + 2u * (p >> 6); }
uint TqBasesEq() { uint p = TqPositions(); return 2u * p + 3u * (p >> 6); }
uint TqHist()    { uint p = TqPositions(); return 2u * p + 4u * (p >> 6); }
uint TqMisc()    { uint p = TqPositions(); return 2u * p + 4u * (p >> 6) + 16u; }

Texture2D<unorm float4>   g_src_rgba    : register(t0);
StructuredBuffer<uint>    g_prev_yuvd   : register(t1); // Ping-pong previous frame ref [Y|Cb|Cr|delta]
Texture2D<unorm float4>   g_prev_src    : register(t2); // Previous source frame (motion search only)
Texture2D<uint>           g_half_cur    : register(t3); // Half-res luma of the source (CSLumaHalf)
Texture2D<uint>           g_half_prev   : register(t4); // Half-res luma of the previous source
StructuredBuffer<int2>    g_prev_mvs    : register(t5); // Tile vectors (prev frame in CSTileMV, cur frame after)
Texture2D<uint>           g_quarter_cur : register(t6); // Quarter-res luma of the source (CSLumaQuarter)
Texture2D<uint>           g_quarter_prev: register(t7); // Quarter-res luma of the previous source
RWStructuredBuffer<uint>  g_cur_yuvd    : register(u0); // Ping-pong current frame ref
RWStructuredBuffer<int2>  g_tile_mvs    : register(u1); // (dx, dy) per 32x32 tile
RWStructuredBuffer<uint>  g_out_records : register(u2); // Compact output records
RWByteAddressBuffer       g_out_counter : register(u3); // [0]=record_words_cnt, [4]=tq_wanted, [8]=cut_tiles, [12]=tq_accepted
RWTexture2D<uint>         g_half_out    : register(u4); // CSLumaHalf / CSLumaQuarter output
RWStructuredBuffer<uint>  g_tflag_out   : register(u5); // CSMotionPred: pred_yuvd; CSIntraT: 1 per coded 4x4 block
StructuredBuffer<uint>    g_tflag       : register(t8); // CSEncode: the same flags, read-only
RWStructuredBuffer<uint>  g_tq_scratch  : register(u6);
// Records of every block that wants the transform, in any order (each block
// knows where its own are); the accepted ones are copied to g_out_records.
RWStructuredBuffer<uint>  g_tq_stage    : register(u7);
// Capture hints per tile (desktop duplication): x = 0 none, 1 moved by
// (y, z) whole pixels (a MoveRect), 2 unchanged since the previous frame.
StructuredBuffer<int4>    g_mv_hint     : register(t9);
// Precomputed motion-compensated decayed prediction [decayed_Y|Cb|Cr|orig_d] (written by CSMotionPred).
StructuredBuffer<uint>    g_pred_yuvd   : register(t10);

bool TqIsSceneCut() {
    if ((mode_flags & 128u) != 0u) return true;
    if ((mode_flags & 32u) == 0u) return false;
    uint nt = tiles_x * tiles_y;
    return (nt > 0u) && (g_out_counter.Load(8) * 3u >= nt);
}

uint TqSceneCutQp() {
    uint nb = ((width + 3u) >> 2) * ((height + 3u) >> 2);
    if (max_record_words == 0u || max_record_words >= nb * 3u) return max(tq_qp, 28u);
    if (max_record_words >= nb) return max(tq_qp, 30u);
    // Mid/tight cuts (4K @ 50M -> QP 36 + smooth AQ; 4K @ 10M -> QP 38 + RDO
    // hybrid 8x8/4x4 so sharp edges stay crisp while all superblocks fit).
    if (max_record_words * 3u >= nb) return max(tq_qp, 36u);
    return max(tq_qp, 38u);
}

// H.264/HEVC-style perceptual chroma QP curve: preserves vivid chroma fidelity
// across low/mid/high QP so Zcvid beats both NVENC and Pyrowave on 4K cinema
// gradients (50M & 10M) and sub-pixel gaming motion alike without starving Y
// at ultra-low bitrates (QP 40..46).
uint TqChromaQp(uint qp_y) {
    if (qp_y <= 20u) return (qp_y > 2u) ? (qp_y - 2u) : qp_y;
    if (qp_y <= 30u) return 18u + ((qp_y - 20u) * 5u) / 10u;
    if (qp_y <= 38u) return 23u + ((qp_y - 30u) * 9u) / 8u;
    return min(32u + ((qp_y - 38u) * 11u) / 10u, 46u);
}

// Variance-Adaptive Quantization (AQ) for scene cuts: smooth/flat blocks emit
// only 1-3 DC/tilt coefficients (which intra DC DPCM codes in ~4 bits), so
// lowering their QP boosts smooth-gradient quality by +8..15 dB at tiny bit cost.
uint AqAdjustCutQp(uint base_qp, uint ac_sad) {
    if (base_qp < 24u) return base_qp;
    if (base_qp >= 38u) {
        // Tight 4K cuts: boost truly flat gradients (Scene 2) while keeping
        // textured blocks at base_qp so 100% of superblocks refresh on Frame 0.
        if (ac_sad < 28u) return base_qp - 12u;
        if (ac_sad < 64u) return base_qp - 4u;
        return base_qp;
    }
    if (ac_sad < 40u) return max(base_qp - 14u, 18u);
    if (ac_sad < 80u) return max(base_qp - 6u, 24u);
    return base_qp;
}

uint TqInterCap() {
    uint stage_n = 0u, stride = 0u;
    g_tq_stage.GetDimensions(stage_n, stride);
    uint max_safe = (stage_n > 96u) ? ((stage_n - 96u) & ~1u) : (stage_n & ~1u);
    if (max_record_words == 0u) return 0x7FFFFFFFu;
    // Unified pure-transform codec: 100% of max_record_words belongs to
    // the transform bitstream (clamped safely to g_tq_stage capacity).
    uint base_cap = min(max_record_words & ~1u, max_safe);
    if (TqIsSceneCut()) {
        // On a GPU-detected abrupt scene cut (or single-frame keyframe), expand
        // the Frame-0 I-frame cap so 100% of superblocks refresh cleanly on
        // Frame 0 (minimum 3 DC records = 6 words/SB = 1.5*nb just for DC).
        // CABAC v4 (mode_flags bit 8: 8x8 DCT8) codes ~35% more coefs per byte,
        // so allow +1*nb record words while keeping encoded frame bytes <= legacy.
        uint nb = ((width + 3u) >> 2) * ((height + 3u) >> 2);
        bool dct8_cut = (mode_flags & 256u) != 0u;
        uint cut_mult = (max_record_words >= nb) ? (dct8_cut ? 5u : 4u)
                                                 : (dct8_cut ? 4u : 3u);
        uint cut_cap = nb * cut_mult;
        if (cut_cap > max_safe) cut_cap = max_safe;
        return max(base_cap, cut_cap);
    }
    return base_cap;
}

static const uint POW3[10] = { 1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683 };
static const uint BAYER4[16] = { 0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5 };

// Output records are always 2 uint32 words:
//   Delta: word0 = (plane << 30) | (is_2x << 29) | blk      (plane 0=Y, 1=Cb, 2=Cr)
//          word1 = 10 trits packed in base 3 (0..59048)
//   Intra: word0 = 0xC0000000 | idx                          (idx = first pixel of the pair)
//          word1 = Y0 | (Cb << 8) | (Y1 << 16) | (Cr << 24)  (4:2:2 pair idx, idx + 1)
//
// Closed-loop contract shared with zcvid_decode.hlsl (any change must be
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
// 4x4 integer transform intra (shared, word for word, by zcvid_encode.hlsl
// and zcvid_decode.hlsl: the reconstruction must be bit-exact on both
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
static const int TQ8_MF[6] = { 26214, 23302, 20560, 18396, 16384, 14564 };
static const int TQ8_V[6]  = { 40, 45, 51, 57, 64, 72 };

int Tq8Dequant(int level, uint qp) {
    uint qp_mod = qp % 6u;
    uint qp_div = qp / 6u;
    int dq_step = (qp_div >= 2u)
                      ? (TQ8_V[qp_mod] << (qp_div - 2u))
                      : (((TQ8_V[qp_mod] << qp_div) + 2) >> 2);
    return clamp(level * dq_step, -65535, 65535);
}

// Exact 12-mul 1D forward HEVC DCT8 (Negacyclic-Karatsuba 9-mul odd + Gauss 3-mul even).
// Proved bit-exact and overflow-free in zcvid/proofs/lean4/ZcvidDCT8Circuit.lean.
void Dct8Fwd1D(inout int x0, inout int x1, inout int x2, inout int x3,
               inout int x4, inout int x5, inout int x6, inout int x7,
               int rnd, int sh) {
    int e0 = x0 + x7, e1 = x1 + x6, e2 = x2 + x5, e3 = x3 + x4;
    int o0 = x0 - x7, o1 = x1 - x6, o2 = x2 - x5, o3 = x3 - x4;
    int ee0 = e0 + e3, ee1 = e1 + e2;
    int eo0 = e0 - e3, eo1 = e1 - e2;
    int pev = 36 * (eo0 + eo1);

    int s0 = o0 + o3, s1 = o2 + o1;
    int m0 = 89 * o0, m1 = -75 * o2, m2 = 14 * (o0 + o2);
    int m3 = -18 * o3, m4 = -50 * o1, m5 = -68 * (o3 + o1);
    int m6 = 71 * s0, m7 = -125 * s1, m8 = -54 * (s0 + s1);
    int u0 = m0 + m3, d0 = m0 - m3;
    int u1 = m1 + m4, d1 = m1 - m4;
    int u2 = m2 + m5, d2 = m2 - m5;
    int k0 = m6 - u0, k1 = m7 - u1;

    x0 = (64 * (ee0 + ee1) + rnd) >> sh;
    x4 = (64 * (ee0 - ee1) + rnd) >> sh;
    x2 = (pev + 47 * eo0 + rnd) >> sh;
    x6 = (pev - 119 * eo1 + rnd) >> sh;
    x1 = (d0 - k1 + rnd) >> sh;
    x3 = (d0 + d1 - d2 + rnd) >> sh;
    x5 = (k0 + k1 + u2 - m8 + rnd) >> sh;
    x7 = (-k0 - d1 + rnd) >> sh;
}

// Exact 12-mul 1D inverse HEVC IDCT8 (reuses identical Negacyclic-Karatsuba circuit
// by M_odd^T = M_odd duality). Shared word-for-word with zcvid_decode.hlsl.
void Dct8Inv1D(inout int x0, inout int x1, inout int x2, inout int x3,
               inout int x4, inout int x5, inout int x6, inout int x7,
               int rnd, int sh) {
    int e0 = 64 * (x0 + x4);
    int e1 = 64 * (x0 - x4);
    int pev = 36 * (x2 + x6);
    int g0 = pev + 47 * x2;
    int g1 = pev - 119 * x6;
    int E0 = e0 + g0, E1 = e1 + g1, E2 = e1 - g1, E3 = e0 - g0;

    int s0 = x1 + x7, s1 = x5 + x3;
    int m0 = 89 * x1, m1 = -75 * x5, m2 = 14 * (x1 + x5);
    int m3 = -18 * x7, m4 = -50 * x3, m5 = -68 * (x7 + x3);
    int m6 = 71 * s0, m7 = -125 * s1, m8 = -54 * (s0 + s1);
    int u0 = m0 + m3, d0 = m0 - m3;
    int u1 = m1 + m4, d1 = m1 - m4;
    int u2 = m2 + m5, d2 = m2 - m5;
    int k0 = m6 - u0, k1 = m7 - u1;
    int O0 = d0 - k1;
    int O1 = d0 + d1 - d2;
    int O2 = k0 + k1 + u2 - m8;
    int O3 = -k0 - d1;

    x0 = (E0 + O0 + rnd) >> sh;
    x1 = (E1 + O1 + rnd) >> sh;
    x2 = (E2 + O2 + rnd) >> sh;
    x3 = (E3 + O3 + rnd) >> sh;
    x4 = (E3 - O3 + rnd) >> sh;
    x5 = (E2 - O2 + rnd) >> sh;
    x6 = (E1 - O1 + rnd) >> sh;
    x7 = (E0 - O0 + rnd) >> sh;
}

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
// word, by zcvid_encode.hlsl and zcvid_decode.hlsl). Y, Cb and Cr are
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

groupshared int  gs_cur_low[64];
groupshared int  gs_cur_full[64];
groupshared int  gs_cur_q[128]; // Stage 4 dense 128-sample quincunx (16 rows x 8 staggered cols)
groupshared uint gs_sad_red[64];
groupshared int  gs_ref[1444]; // Stage 1: up to 38x38 quarter-res window; Stage 4: 35x35 full-res luma patch
groupshared int2 gs_cand_mv[16];  // Wide-range candidate MVs (supports up to +-508 qpel = +-127 px)
groupshared uint gs_cand_raw[16]; // Unpenalized source SAD per candidate slot

// SAD of a tile on a reduced luma plane using cached gs_cur_low[64].
uint EvalTileSadLowCached(Texture2D<uint> prev, int w, int h,
                          uint x0, uint y0, uint first, uint step, int dx, int dy) {
    uint sad = 0;
    [loop] for (uint j = 0; j < 8u; j++) {
        int py = min((int)(y0 + first + j * step), h - 1);
        int sy = clamp(py - dy, 0, h - 1);
        [unroll] for (uint i = 0; i < 8u; i++) {
            int px = min((int)(x0 + first + i * step), w - 1);
            int sx = clamp(px - dx, 0, w - 1);
            int cy = gs_cur_low[j * 8u + i];
            int ry = (int)prev.Load(int3(sx, sy, 0));
            sad += (uint)abs(cy - ry);
        }
    }
    return sad;
}

// SAD of a 32x32 tile at full resolution (64 samples) using cached gs_cur_full[64].
uint EvalTileSadCached(uint x0, uint y0, int dx, int dy) {
    uint sad = 0;
    [loop] for (uint j = 0; j < 8u; j++) {
        int py = min((int)(y0 + 2u + j * 4u), (int)height - 1);
        int sy = clamp(py - dy, 0, (int)height - 1);
        [unroll] for (uint i = 0; i < 8u; i++) {
            int px = min((int)(x0 + 2u + i * 4u), (int)width - 1);
            int sx = clamp(px - dx, 0, (int)width - 1);
            int cy = gs_cur_full[j * 8u + i];
            int ry = LumaOf(g_prev_src.Load(int3(sx, sy, 0)).rgb);
            sad += (uint)abs(cy - ry);
        }
    }
    return sad;
}

// Stage 1 quarter-res candidate key: packs exact 14-bit SAD + 6-bit signed
// quarter-res offsets qdx, qdy in [-31, +31] (up to +-124 px/frame).
// Zero vector uses bits 0 and gets a light -4u SAD bias; non-zero vectors pay a
// small distance penalty in coarse stages so smooth cinema gradients with film
// grain do not drift along iso-luminance contours (aperture problem).
uint MvKeyQ(uint norm_sad, int qdx, int qdy) {
    uint mag_pen = min((uint)(abs(qdx) + abs(qdy)) * 2u, 32u);
    uint biased_sad = (qdx == 0 && qdy == 0) ? (norm_sad > 4u ? norm_sad - 4u : 0u)
                                             : min(norm_sad + mag_pen, 16383u);
    uint mv_bits = (qdx == 0 && qdy == 0) ? 0u : (((uint)(qdy + 32) << 6) | (uint)(qdx + 32));
    return (biased_sad << 12) | mv_bits;
}

int2 MvOfKeyQ(uint key) {
    uint mv_bits = key & 0xFFFu;
    if (mv_bits == 0u) return int2(0, 0);
    return int2((int)(mv_bits & 63u) - 32, (int)(mv_bits >> 6) - 32);
}

int Med3(int a, int b, int c) {
    return max(min(a, b), min(max(a, b), c));
}

int Med5(int a, int b, int c, int d, int e) {
    int mn1 = min(a, b), mx1 = max(a, b);
    int mn2 = min(c, d), mx2 = max(c, d);
    int lo = max(mn1, mn2);
    int hi = min(mx1, mx2);
    return Med3(lo, hi, e);
}

// QP-aware Lagrangian smoothness penalty (distance from temporal-spatial median
// predictor in quarter-pels, plus a light sub-pel tie-breaker when diverging from
// pred_mv so film grain never triggers noisy 4-tap interpolation).
uint MvCostPred(int dx, int dy, int2 pred_mv) {
    uint dist = (uint)(abs(dx - pred_mv.x) + abs(dy - pred_mv.y));
    uint cost = 0u;
    if (dist != 0u) {
        uint scale  = (tq_qp >= 32u) ? 6u : 4u;
        uint nz_pen = (tq_qp >= 32u) ? 16u : 8u;
        uint cap_c  = 112u;
        cost = min(nz_pen + dist * scale, cap_c);
        if (((dx | dy) & 3) != 0) cost += 24u;
    }
    if (dx == 0 && dy == 0) {
        cost = min(cost, 16u);
    }
    return cost;
}

// Slot-indexed exact-SAD InterlockedMin key for Stages 2/3/4 (unlimited MV range).
uint MvSlotKey(uint sad, uint slot) {
    return (min(sad, 65535u) << 4) | (slot & 15u);
}

// Dense 128-sample quincunx exact 4-tap HEVC luma SAD (2 rows x 8 staggered cols = 16
// samples/lane; 8 lanes = 128 samples covering all 16 X & 16 Y stride-2 positions 0..30).
uint EvalRow2Sad4Tap(uint tx, uint ty, uint lane, int ox, int oy) {
    int fx = (-ox) & 3;
    int fy = (-oy) & 3;
    int shx = ((-ox) >> 2) + 2;
    int shy = ((-oy) >> 2) + 2;
    int4 tx_t = TapsQ(fx);
    int4 ty_t = TapsQ(fy);
    int max_x = min(30, (int)width - 1 - (int)(tx * 32u));
    int max_y = min(30, (int)height - 1 - (int)(ty * 32u));
    uint sad = 0u;
    [loop] for (uint r = 0u; r < 2u; r++) {
        uint j = lane * 2u + r;
        int py_off = min((int)(j * 2u), max_y);
        int ry0 = py_off + shy;
        uint cur_row = j * 8u;
        int x_stag = (int)((j & 1u) << 1);
        if ((fx | fy) == 0) {
            int rrow = ry0 * 35;
            [loop] for (uint i = 0u; i < 8u; i++) {
                int rx0 = min((int)(i * 4u) + x_stag, max_x) + shx;
                sad += (uint)abs(gs_cur_q[cur_row + i] - gs_ref[rrow + rx0]);
            }
        } else if (fy == 0) {
            int rrow = ry0 * 35;
            [loop] for (uint i = 0u; i < 8u; i++) {
                int rb = rrow + min((int)(i * 4u) + x_stag, max_x) + shx - 1;
                int h = tx_t.x * gs_ref[rb] + tx_t.y * gs_ref[rb + 1] +
                        tx_t.z * gs_ref[rb + 2] + tx_t.w * gs_ref[rb + 3];
                int ry = clamp((h + 32) >> 6, 0, 255);
                sad += (uint)abs(gs_cur_q[cur_row + i] - ry);
            }
        } else if (fx == 0) {
            int rrow = (ry0 - 1) * 35;
            [loop] for (uint i = 0u; i < 8u; i++) {
                int cb = rrow + min((int)(i * 4u) + x_stag, max_x) + shx;
                int v = ty_t.x * gs_ref[cb] + ty_t.y * gs_ref[cb + 35] +
                        ty_t.z * gs_ref[cb + 70] + ty_t.w * gs_ref[cb + 105];
                int ry = clamp((v + 32) >> 6, 0, 255);
                sad += (uint)abs(gs_cur_q[cur_row + i] - ry);
            }
        } else {
            int r0 = (ry0 - 1) * 35;
            [loop] for (uint i = 0u; i < 8u; i++) {
                int rx = min((int)(i * 4u) + x_stag, max_x) + shx - 1;
                int b0 = r0 + rx;
                int h0 = tx_t.x * gs_ref[b0] + tx_t.y * gs_ref[b0 + 1] + tx_t.z * gs_ref[b0 + 2] + tx_t.w * gs_ref[b0 + 3];
                int h1 = tx_t.x * gs_ref[b0 + 35] + tx_t.y * gs_ref[b0 + 36] + tx_t.z * gs_ref[b0 + 37] + tx_t.w * gs_ref[b0 + 38];
                int h2 = tx_t.x * gs_ref[b0 + 70] + tx_t.y * gs_ref[b0 + 71] + tx_t.z * gs_ref[b0 + 72] + tx_t.w * gs_ref[b0 + 73];
                int h3 = tx_t.x * gs_ref[b0 + 105] + tx_t.y * gs_ref[b0 + 106] + tx_t.z * gs_ref[b0 + 107] + tx_t.w * gs_ref[b0 + 108];
                int acc = ty_t.x * h0 + ty_t.y * h1 + ty_t.z * h2 + ty_t.w * h3;
                int ry = clamp((acc + 2048) >> 12, 0, 255);
                sad += (uint)abs(gs_cur_q[cur_row + i] - ry);
            }
        }
    }
    return sad;
}

// Single-sample 4-tap HEVC luma fetch from g_prev_src (for temporal qpel predictor outside +-3 qpel).
int SamplePrevLuma4Tap(int px, int py, int dxq, int dyq) {
    int sxq = px * 4 - dxq;
    int syq = py * 4 - dyq;
    int ix = sxq >> 2;
    int iy = syq >> 2;
    int fx = sxq & 3;
    int fy = syq & 3;
    if ((fx | fy) == 0) {
        return LumaOf(g_prev_src.Load(int3(clamp(ix, 0, (int)width - 1),
                                           clamp(iy, 0, (int)height - 1), 0)).rgb);
    }
    int4 tx_t = TapsQ(fx);
    int4 ty_t = TapsQ(fy);
    int acc = 0;
    [loop] for (int j = 0; j < 4; j++) {
        int tv = (j == 0) ? ty_t.x : (j == 1) ? ty_t.y : (j == 2) ? ty_t.z : ty_t.w;
        int gy = clamp(iy - 1 + j, 0, (int)height - 1);
        int h = 0;
        [unroll] for (int i = 0; i < 4; i++) {
            int gx = clamp(ix - 1 + i, 0, (int)width - 1);
            int t = (i == 0) ? tx_t.x : (i == 1) ? tx_t.y : (i == 2) ? tx_t.z : tx_t.w;
            h += t * LumaOf(g_prev_src.Load(int3(gx, gy, 0)).rgb);
        }
        acc += tv * h;
    }
    return clamp((acc + 2048) >> 12, 0, 255);
}

[numthreads(64, 1, 1)]
void CSTileMV(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tx = gid.x;
    uint ty = gid.y;
    if (tx >= tiles_x || ty >= tiles_y) return;
    int4 hint = g_mv_hint[ty * tiles_x + tx];
    if (hint.x == 2) {
        if (gtid.x == 0u) g_tile_mvs[ty * tiles_x + tx] = int2(0, 0);
        return;
    }

    // Stage 0: load 64 full-res current luma samples in parallel (1 per thread)
    // and evaluate the zero vector (0, 0). Static tiles exit immediately.
    uint si = gtid.x & 7u;
    uint sj = gtid.x >> 3;
    int fpx = min((int)(tx * 32u + 2u + si * 4u), (int)width - 1);
    int fpy = min((int)(ty * 32u + 2u + sj * 4u), (int)height - 1);
    int fcy = LumaOf(g_src_rgba.Load(int3(fpx, fpy, 0)).rgb);
    gs_cur_full[gtid.x] = fcy;
    int fry = LumaOf(g_prev_src.Load(int3(fpx, fpy, 0)).rgb);
    gs_sad_red[gtid.x] = (uint)abs(fcy - fry);
    GroupMemoryBarrierWithGroupSync();

    [unroll] for (uint off = 32u; off > 0u; off >>= 1) {
        if (gtid.x < off) gs_sad_red[gtid.x] += gs_sad_red[gtid.x + off];
        GroupMemoryBarrierWithGroupSync();
    }
    uint zero_sad = gs_sad_red[0];
    GroupMemoryBarrierWithGroupSync();
    if (zero_sad <= 32u && hint.x != 1) {
        if (gtid.x == 0u) g_tile_mvs[ty * tiles_x + tx] = int2(0, 0);
        return;
    }

    bool qpel = (mode_flags & 16u) != 0u;
    int r = min((int)search_r, 124);

    // Compute temporal-spatial median MV predictor (in quarter-pels) from
    // previous frame's 5-tile neighbourhood (center + 4 cardinal neighbours).
    int2 pm0 = g_prev_mvs[ty * tiles_x + tx];
    int2 pm1 = g_prev_mvs[ty * tiles_x + ((tx > 0u) ? tx - 1u : tx)];
    int2 pm2 = g_prev_mvs[ty * tiles_x + min(tx + 1u, tiles_x - 1u)];
    int2 pm3 = g_prev_mvs[((ty > 0u) ? ty - 1u : ty) * tiles_x + tx];
    int2 pm4 = g_prev_mvs[min(ty + 1u, tiles_y - 1u) * tiles_x + tx];
    int2 pred_mv = int2(Med5(pm0.x, pm1.x, pm2.x, pm3.x, pm4.x),
                        Med5(pm0.y, pm1.y, pm2.y, pm3.y, pm4.y));
    if (!qpel) pred_mv *= 4;

    // Stage 0b: test temporal-spatial predictor at full res in parallel (64 threads),
    // and also fast-skip the coarse pyramid when zero_sad is already at film-grain
    // noise floor (<= 160 = 2.5 LSB/px) so Stage 3 refines +-1 px around (0,0) directly.
    bool skip_coarse = false;
    int2 coarse_mv = int2(0, 0); // whole pixels
    if (zero_sad <= 160u && pred_mv.x == 0 && pred_mv.y == 0 && hint.x != 1) {
        skip_coarse = true;
        coarse_mv = int2(0, 0);
    } else if ((pred_mv.x != 0 || pred_mv.y != 0) && r > 0) {
        int2 pi = clamp((pred_mv + 2) >> 2, int2(-r, -r), int2(r, r));
        int psx = clamp(fpx - pi.x, 0, (int)width - 1);
        int psy = clamp(fpy - pi.y, 0, (int)height - 1);
        int pry = LumaOf(g_prev_src.Load(int3(psx, psy, 0)).rgb);
        gs_sad_red[gtid.x] = (uint)abs(fcy - pry);
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint offp = 32u; offp > 0u; offp >>= 1) {
            if (gtid.x < offp) gs_sad_red[gtid.x] += gs_sad_red[gtid.x + offp];
            GroupMemoryBarrierWithGroupSync();
        }
        uint pred_sad = gs_sad_red[0];
        GroupMemoryBarrierWithGroupSync();
        if ((pred_sad <= 256u || (pred_sad <= 768u && pred_sad * 2u < zero_sad)) && hint.x != 1) {
            skip_coarse = true;
            coarse_mv = pi;
        } else if (zero_sad <= 160u && hint.x != 1) {
            skip_coarse = true;
            coarse_mv = int2(0, 0);
        }
    }

    if (!skip_coarse) {
        if (gtid.x == 0u) {
            gs_best_sad = 0xFFFFFFFFu;
            gs_best_mv  = int2(0, 0);
        }

        // Load 64 quarter-res current luma samples into gs_cur_low (1 per thread)
        // and cooperatively cache the inner +-60 px (8 + 2*qr_in)^2 window in gs_ref.
        int qw = (int)((width + 3u) >> 2);
        int qh = (int)((height + 3u) >> 2);
        int qpx = min((int)(tx * 8u + si), qw - 1);
        int qpy = min((int)(ty * 8u + sj), qh - 1);
        gs_cur_low[gtid.x] = (int)g_quarter_cur.Load(int3(qpx, qpy, 0));

        int qr = r >> 2;
        int qr_in = min(qr, 15);
        int qwin = 8 + 2 * qr_in;
        uint qtot = (uint)(qwin * qwin);
        int qbase_x = (int)(tx * 8u) - qr_in;
        int qbase_y = (int)(ty * 8u) - qr_in;
        [loop] for (uint qi = gtid.x; qi < qtot; qi += 64u) {
            int ly = (int)(qi / (uint)qwin);
            int lx = (int)(qi - (uint)ly * (uint)qwin);
            int gx = clamp(qbase_x + lx, 0, qw - 1);
            int gy = clamp(qbase_y + ly, 0, qh - 1);
            gs_ref[qi] = (int)g_quarter_prev.Load(int3(gx, gy, 0));
        }
        GroupMemoryBarrierWithGroupSync();

        // Stage 1a: quarter-res coarse search within +-60 px from groupshared gs_ref.
        // Exhaustive step-1 inside +-20 px (|qdx|,|qdy| <= 5) avoids step-2 aliasing
        // on normal/medium pans & zooms; 24..60 px uses checkerboard + 8-neighbour refine.
        int qmx_x = qw - 1 - (int)(tx * 8u);
        int qmx_y = qh - 1 - (int)(ty * 8u);
        int inner_r = min(qr_in, 5);
        int inner_span = 2 * inner_r + 1;
        int inner_cands = inner_span * inner_span;
        for (int ci = (int)gtid.x; ci < inner_cands; ci += 64) {
            int qdy = (ci / inner_span) - inner_r;
            int qdx = (ci % inner_span) - inner_r;
            uint norm_sad = 0u;
            [loop] for (int j = 0; j < 8; j++) {
                int rrow = (min(j, qmx_y) + qr_in - qdy) * qwin;
                [unroll] for (int i = 0; i < 8; i++) {
                    int ry = gs_ref[rrow + (min(i, qmx_x) + qr_in - qdx)];
                    norm_sad += (uint)abs(gs_cur_low[j * 8 + i] - ry);
                }
            }
            InterlockedMin(gs_best_sad, MvKeyQ(norm_sad, qdx, qdy));
        }
        if (qr_in > 5) {
            int full_span = 2 * qr_in + 1;
            int full_cands = full_span * full_span;
            int half_cands = (full_cands + 1) >> 1;
            for (int ch = (int)gtid.x; ch < half_cands; ch += 64) {
                int c = ch * 2;
                int qdy = (c / full_span) - qr_in;
                int qdx = (c % full_span) - qr_in;
                if (abs(qdx) <= 5 && abs(qdy) <= 5) continue;
                uint norm_sad = 0u;
                [loop] for (int j = 0; j < 8; j++) {
                    int rrow = (min(j, qmx_y) + qr_in - qdy) * qwin;
                    [unroll] for (int i = 0; i < 8; i++) {
                        int ry = gs_ref[rrow + (min(i, qmx_x) + qr_in - qdx)];
                        norm_sad += (uint)abs(gs_cur_low[j * 8 + i] - ry);
                    }
                }
                InterlockedMin(gs_best_sad, MvKeyQ(norm_sad, qdx, qdy));
            }
        }
        GroupMemoryBarrierWithGroupSync();

        if (qr_in > 5) {
            uint best_q2 = gs_best_sad;
            GroupMemoryBarrierWithGroupSync();
            if (gtid.x < 8u && best_q2 != 0xFFFFFFFFu) {
                uint o = (gtid.x < 4u) ? gtid.x : gtid.x + 1u;
                int2 q0 = MvOfKeyQ(best_q2);
                int qdx = q0.x + ((int)(o % 3u) - 1);
                int qdy = q0.y + ((int)(o / 3u) - 1);
                if (abs(qdx) <= qr_in && abs(qdy) <= qr_in) {
                    uint norm_sad = 0u;
                    [loop] for (int j = 0; j < 8; j++) {
                        int rrow = (min(j, qmx_y) + qr_in - qdy) * qwin;
                        [unroll] for (int i = 0; i < 8; i++) {
                            int ry = gs_ref[rrow + (min(i, qmx_x) + qr_in - qdx)];
                            norm_sad += (uint)abs(gs_cur_low[j * 8 + i] - ry);
                        }
                    }
                    InterlockedMin(gs_best_sad, MvKeyQ(norm_sad, qdx, qdy));
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }

        // Stage 1b: conditional wide-velocity outer ring (64..124 px/frame) on
        // L2-cached g_quarter_prev. Completely skipped (0 cost) whenever Stage 1a
        // already found a clean match within +-60 px (SAD <= 384 = 6 LSB/px).
        uint best_in_sad = (gs_best_sad == 0xFFFFFFFFu) ? 16383u : (gs_best_sad >> 12);
        GroupMemoryBarrierWithGroupSync();
        if (qr > 15 && best_in_sad > 384u) {
            int full_span = 2 * qr + 1;
            int full_cands = full_span * full_span;
            int half_cands = (full_cands + 1) >> 1;
            for (int ch = (int)gtid.x; ch < half_cands; ch += 64) {
                int c = ch * 2;
                int qdy = (c / full_span) - qr;
                int qdx = (c % full_span) - qr;
                if (abs(qdx) <= 15 && abs(qdy) <= 15) continue;
                uint norm_sad = EvalTileSadLowCached(g_quarter_prev, qw, qh,
                                                     tx * 8u, ty * 8u, 0u, 1u, qdx, qdy);
                InterlockedMin(gs_best_sad, MvKeyQ(norm_sad, qdx, qdy));
            }
            GroupMemoryBarrierWithGroupSync();
            uint best_out = gs_best_sad;
            GroupMemoryBarrierWithGroupSync();
            if (gtid.x < 8u && best_out != 0xFFFFFFFFu) {
                int2 q0 = MvOfKeyQ(best_out);
                if (abs(q0.x) > 15 || abs(q0.y) > 15) {
                    uint o = (gtid.x < 4u) ? gtid.x : gtid.x + 1u;
                    int qdx = q0.x + ((int)(o % 3u) - 1);
                    int qdy = q0.y + ((int)(o / 3u) - 1);
                    if (abs(qdx) <= qr && abs(qdy) <= qr) {
                        uint norm_sad = EvalTileSadLowCached(g_quarter_prev, qw, qh,
                                                             tx * 8u, ty * 8u, 0u, 1u, qdx, qdy);
                        InterlockedMin(gs_best_sad, MvKeyQ(norm_sad, qdx, qdy));
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }

        uint best_q = gs_best_sad;
        GroupMemoryBarrierWithGroupSync();
        if (gtid.x == 0u) {
            gs_best_sad = 0xFFFFFFFFu;
        }

        // Load 64 half-res current luma samples into gs_cur_low (1 per thread).
        int hw = (int)((width + 1u) >> 1);
        int hh = (int)((height + 1u) >> 1);
        int hpx = min((int)(tx * 16u + 1u + si * 2u), hw - 1);
        int hpy = min((int)(ty * 16u + 1u + sj * 2u), hh - 1);
        gs_cur_low[gtid.x] = (int)g_half_cur.Load(int3(hpx, hpy, 0));
        GroupMemoryBarrierWithGroupSync();

        // Stage 2: the quarter-res winner and its 8 neighbours at +-2 px, half-res luma.
        if (gtid.x < 9u && r > 0) {
            int2 qmv = (best_q == 0xFFFFFFFFu) ? int2(0, 0) : MvOfKeyQ(best_q);
            int hdx = qmv.x * 2 + ((int)(gtid.x % 3u) - 1);
            int hdy = qmv.y * 2 + ((int)(gtid.x / 3u) - 1);
            if (abs(hdx * 2) <= r && abs(hdy * 2) <= r) {
                uint norm_sad = EvalTileSadLowCached(g_half_prev, hw, hh,
                                                     tx * 16u, ty * 16u, 1u, 2u, hdx, hdy);
                uint mag_pen = min((uint)(abs(hdx) + abs(hdy)), 32u);
                uint biased_sad = (hdx == 0 && hdy == 0) ? (norm_sad > 4u ? norm_sad - 4u : 0u)
                                                         : min(norm_sad + mag_pen, 16383u);
                gs_cand_mv[gtid.x] = int2(hdx * 2, hdy * 2);
                InterlockedMin(gs_best_sad, MvSlotKey(biased_sad, gtid.x));
            }
        }
        GroupMemoryBarrierWithGroupSync();
        coarse_mv = (gs_best_sad == 0xFFFFFFFFu) ? int2(0, 0) : gs_cand_mv[gs_best_sad & 15u];
        GroupMemoryBarrierWithGroupSync();
    }

    if (gtid.x == 0u) {
        gs_cand_mv[0] = int2(0, 0);
        gs_cand_raw[0] = zero_sad;
        gs_best_sad = MvSlotKey(zero_sad + MvCostPred(0, 0, pred_mv), 0u);
    }
    GroupMemoryBarrierWithGroupSync();

    // Stage 3, full resolution: slot 0 = (0,0), slots 1..9 = half-res winner
    // and its 8 neighbours, slots 10..14 = previous-frame vectors of this tile
    // and its four neighbours, slot 15 = capture MoveRect hint.
    if (gtid.x >= 1u && gtid.x < 16u && r > 0) {
        uint cid = gtid.x - 1u;
        int2 cand = int2(0, 0);
        bool valid_cand = true;
        if (cid < 9u) {
            cand = coarse_mv + int2((int)(cid % 3u) - 1, (int)(cid / 3u) - 1);
        } else if (cid < 14u) {
            if (cid == 9u) cand = pm0;
            else if (cid == 10u) cand = pm1;
            else if (cid == 11u) cand = pm2;
            else if (cid == 12u) cand = pm3;
            else cand = pm4;
            if (qpel) cand = (cand + 2) >> 2;
        } else {
            valid_cand = (hint.x == 1);
            cand = hint.yz;
        }
        if (valid_cand && (cand.x != 0 || cand.y != 0) &&
            cand.x >= -r && cand.x <= r && cand.y >= -r && cand.y <= r) {
            uint norm_sad = EvalTileSadCached(tx * 32u, ty * 32u, cand.x, cand.y);
            int2 cmvq = cand * 4;
            gs_cand_mv[gtid.x] = cmvq;
            gs_cand_raw[gtid.x] = norm_sad;
            InterlockedMin(gs_best_sad, MvSlotKey(norm_sad + MvCostPred(cmvq.x, cmvq.y, pred_mv), gtid.x));
        }
    }
    GroupMemoryBarrierWithGroupSync();

    uint s3_slot = gs_best_sad & 15u;
    int2 best_mv = gs_cand_mv[s3_slot];
    uint src_raw_sad = gs_cand_raw[s3_slot];
    uint best_cost = gs_best_sad >> 4;
    GroupMemoryBarrierWithGroupSync();

    // Stage 4 (quarter-pel mode): cooperative 35x35 groupshared luma cache +
    // dense 128-sample quincunx current grid (16 rows x 8 staggered cols),
    // followed by 2-step hierarchical exact 4-tap HEVC refinement (Wave 4a: 8 half-pel
    // +-2 qpel x 8 lanes; Wave 4b: 8 quarter-pel +-1 qpel x 8 lanes).
    if (qpel && r > 0) {
        bool pred_is_frac = (((pred_mv.x | pred_mv.y) & 3) != 0);
        if (best_cost > 160u || (pred_is_frac && best_cost > 32u)) {
            int2 c0_q = best_mv;
            int2 c0_i = c0_q / 4;
            int base_x = (int)(tx * 32u) - c0_i.x;
            int base_y = (int)(ty * 32u) - c0_i.y;
            [loop] for (uint idx = gtid.x; idx < 1225u; idx += 64u) {
                int ly = (int)(idx / 35u);
                int lx = (int)(idx - (uint)ly * 35u);
                int gx = clamp(base_x - 2 + lx, 0, (int)width - 1);
                int gy = clamp(base_y - 2 + ly, 0, (int)height - 1);
                gs_ref[idx] = LumaOf(g_prev_src.Load(int3(gx, gy, 0)).rgb);
            }
            [unroll] for (uint s4 = 0u; s4 < 2u; s4++) {
                uint idx4 = gtid.x + s4 * 64u;
                uint sy4 = idx4 >> 3;
                uint sx4 = idx4 & 7u;
                int cx4 = min((int)(tx * 32u + sx4 * 4u + ((sy4 & 1u) << 1)), (int)width - 1);
                int cy4 = min((int)(ty * 32u + sy4 * 2u), (int)height - 1);
                gs_cur_q[idx4] = LumaOf(g_src_rgba.Load(int3(cx4, cy4, 0)).rgb);
            }
            GroupMemoryBarrierWithGroupSync();

            uint cid = gtid.x >> 3;
            uint lane = gtid.x & 7u;
            uint o = (cid < 4u) ? cid : cid + 1u;

            // Re-score integer center c0_q on the exact 128-sample quincunx metric.
            if (cid == 0u) {
                gs_sad_red[lane] = EvalRow2Sad4Tap(tx, ty, lane, 0, 0);
            }
            GroupMemoryBarrierWithGroupSync();
            if (gtid.x == 0u) {
                uint sad128_0 = gs_sad_red[0] + gs_sad_red[1] + gs_sad_red[2] + gs_sad_red[3] +
                                gs_sad_red[4] + gs_sad_red[5] + gs_sad_red[6] + gs_sad_red[7];
                uint sad64_0 = (sad128_0 + 1u) >> 1;
                gs_cand_mv[0] = c0_q;
                gs_cand_raw[0] = sad64_0;
                gs_best_sad = MvSlotKey(sad64_0 + MvCostPred(c0_q.x, c0_q.y, pred_mv), 0u);
            }
            GroupMemoryBarrierWithGroupSync();

            // Wave 4a: 8 half-pel neighbours (+-2 qpel) in parallel (8 threads/cand, 128 samples).
            int hx = ((int)(o % 3u) - 1) * 2;
            int hy = ((int)(o / 3u) - 1) * 2;
            int2 hq = c0_q + int2(hx, hy);
            bool hvalid = (abs(hq.x) <= 508 && abs(hq.y) <= 508);
            gs_sad_red[gtid.x] = hvalid ? EvalRow2Sad4Tap(tx, ty, lane, hx, hy) : 0u;
            GroupMemoryBarrierWithGroupSync();
            if (lane < 4u) gs_sad_red[gtid.x] += gs_sad_red[gtid.x + 4u];
            GroupMemoryBarrierWithGroupSync();
            if (lane < 2u) gs_sad_red[gtid.x] += gs_sad_red[gtid.x + 2u];
            GroupMemoryBarrierWithGroupSync();
            if (lane == 0u && hvalid) {
                uint sad64 = (gs_sad_red[gtid.x] + gs_sad_red[gtid.x + 1u] + 1u) >> 1;
                gs_cand_mv[cid + 1u] = hq;
                gs_cand_raw[cid + 1u] = sad64;
                InterlockedMin(gs_best_sad, MvSlotKey(sad64 + MvCostPred(hq.x, hq.y, pred_mv), cid + 1u));
            }
            GroupMemoryBarrierWithGroupSync();

            uint s4a_slot = gs_best_sad & 15u;
            best_mv = gs_cand_mv[s4a_slot];
            src_raw_sad = gs_cand_raw[s4a_slot];
            best_cost = gs_best_sad >> 4;
            GroupMemoryBarrierWithGroupSync();

            if (gtid.x == 0u) {
                gs_cand_mv[0] = best_mv;
                gs_cand_raw[0] = src_raw_sad;
                gs_best_sad = MvSlotKey(best_cost, 0u);
            }
            GroupMemoryBarrierWithGroupSync();

            // Wave 4b: 8 quarter-pel neighbours (+-1 qpel) around half-pel winner (128 samples).
            int2 h_off = best_mv - c0_q;
            int qx = h_off.x + ((int)(o % 3u) - 1);
            int qy = h_off.y + ((int)(o / 3u) - 1);
            int2 qq = c0_q + int2(qx, qy);
            bool qvalid = (abs(qx) <= 3 && abs(qy) <= 3 && abs(qq.x) <= 508 && abs(qq.y) <= 508);
            gs_sad_red[gtid.x] = qvalid ? EvalRow2Sad4Tap(tx, ty, lane, qx, qy) : 0u;
            GroupMemoryBarrierWithGroupSync();
            if (lane < 4u) gs_sad_red[gtid.x] += gs_sad_red[gtid.x + 4u];
            GroupMemoryBarrierWithGroupSync();
            if (lane < 2u) gs_sad_red[gtid.x] += gs_sad_red[gtid.x + 2u];
            GroupMemoryBarrierWithGroupSync();
            if (lane == 0u && qvalid) {
                uint sad64 = (gs_sad_red[gtid.x] + gs_sad_red[gtid.x + 1u] + 1u) >> 1;
                gs_cand_mv[cid + 1u] = qq;
                gs_cand_raw[cid + 1u] = sad64;
                InterlockedMin(gs_best_sad, MvSlotKey(sad64 + MvCostPred(qq.x, qq.y, pred_mv), cid + 1u));
            }
            GroupMemoryBarrierWithGroupSync();

            uint s4b_slot = gs_best_sad & 15u;
            best_mv = gs_cand_mv[s4b_slot];
            src_raw_sad = gs_cand_raw[s4b_slot];
            best_cost = gs_best_sad >> 4;
            GroupMemoryBarrierWithGroupSync();

            // Wave 4c: verify temporal-spatial median qpel predictor when outside c0_q +-3 qpel.
            if ((pred_mv.x != 0 || pred_mv.y != 0) && abs(pred_mv.x) <= 508 && abs(pred_mv.y) <= 508 &&
                (abs(pred_mv.x - c0_q.x) > 3 || abs(pred_mv.y - c0_q.y) > 3)) {
                int pry = SamplePrevLuma4Tap(fpx, fpy, pred_mv.x, pred_mv.y);
                gs_sad_red[gtid.x] = (uint)abs(gs_cur_full[gtid.x] - pry);
                GroupMemoryBarrierWithGroupSync();
                [unroll] for (uint off2 = 32u; off2 > 0u; off2 >>= 1) {
                    if (gtid.x < off2) gs_sad_red[gtid.x] += gs_sad_red[gtid.x + off2];
                    GroupMemoryBarrierWithGroupSync();
                }
                uint p_raw = gs_sad_red[0];
                uint p_cost = p_raw + MvCostPred(pred_mv.x, pred_mv.y, pred_mv);
                if (p_cost < best_cost) {
                    best_mv = pred_mv;
                    src_raw_sad = p_raw;
                    best_cost = p_cost;
                }
                GroupMemoryBarrierWithGroupSync();
            }

            // Wave 4d: bounded closed-loop +-1 qpel phase lock against g_prev_yuvd (128 samples).
            int2 s_q = best_mv;
            bool sq_is_frac = (((s_q.x | s_q.y) & 3) != 0);
            if (src_raw_sad <= 2560u &&
                (best_cost > 200u || (sq_is_frac && best_cost > 64u))) {
                int2 s_i = (s_q + 2) >> 2;
                int2 s_rem = s_q - s_i * 4;
                int rbase_x = (int)(tx * 32u) - s_i.x;
                int rbase_y = (int)(ty * 32u) - s_i.y;
                [loop] for (uint idx2 = gtid.x; idx2 < 1225u; idx2 += 64u) {
                    int ly = (int)(idx2 / 35u);
                    int lx = (int)(idx2 - (uint)ly * 35u);
                    int gx = clamp(rbase_x - 2 + lx, 0, (int)width - 1);
                    int gy = clamp(rbase_y - 2 + ly, 0, (int)height - 1);
                    gs_ref[idx2] = (int)(g_prev_yuvd[(uint)gy * width + (uint)gx] & 0xFFu);
                }
                GroupMemoryBarrierWithGroupSync();

                // Score center s_q on g_prev_yuvd (128 samples in parallel on cid == 0).
                if (cid == 0u) {
                    gs_sad_red[lane] = EvalRow2Sad4Tap(tx, ty, lane, s_rem.x, s_rem.y);
                }
                GroupMemoryBarrierWithGroupSync();
                if (gtid.x == 0u) {
                    uint sad0 = (gs_sad_red[0] + gs_sad_red[1] + gs_sad_red[2] + gs_sad_red[3] +
                                 gs_sad_red[4] + gs_sad_red[5] + gs_sad_red[6] + gs_sad_red[7] + 1u) >> 1;
                    gs_cand_mv[0] = s_q;
                    gs_best_sad = MvSlotKey(sad0 + MvCostPred(s_q.x, s_q.y, pred_mv), 0u);
                }
                GroupMemoryBarrierWithGroupSync();

                // Evaluate the 8 +-1 qpel neighbours around s_q on g_prev_yuvd (128 samples).
                int dx1 = (int)(o % 3u) - 1;
                int dy1 = (int)(o / 3u) - 1;
                int2 cand_q = s_q + int2(dx1, dy1);
                bool c1valid = (abs(cand_q.x) <= 508 && abs(cand_q.y) <= 508);
                gs_sad_red[gtid.x] = c1valid ? EvalRow2Sad4Tap(tx, ty, lane, s_rem.x + dx1, s_rem.y + dy1) : 0u;
                GroupMemoryBarrierWithGroupSync();
                if (lane < 4u) gs_sad_red[gtid.x] += gs_sad_red[gtid.x + 4u];
                GroupMemoryBarrierWithGroupSync();
                if (lane < 2u) gs_sad_red[gtid.x] += gs_sad_red[gtid.x + 2u];
                GroupMemoryBarrierWithGroupSync();
                if (lane == 0u && c1valid) {
                    uint c1_pen = (uint)(abs(dx1) + abs(dy1)) * ((tq_qp >= 28u) ? 16u : 8u);
                    uint sad64 = ((gs_sad_red[gtid.x] + gs_sad_red[gtid.x + 1u] + 1u) >> 1) + c1_pen;
                    gs_cand_mv[cid + 1u] = cand_q;
                    InterlockedMin(gs_best_sad, MvSlotKey(sad64 + MvCostPred(cand_q.x, cand_q.y, pred_mv), cid + 1u));
                }
                GroupMemoryBarrierWithGroupSync();
                best_mv = gs_cand_mv[gs_best_sad & 15u];
                GroupMemoryBarrierWithGroupSync();
            }
        }
    }

    if (gtid.x == 0u) {
        // Count unmatched tiles (mean source error > 32 LSB/px across 64 samples)
        // so Pass 1b detects abrupt scene cuts on Frame 0 before quantizing.
        if (src_raw_sad > 2048u) {
            uint dummy_cut;
            g_out_counter.InterlockedAdd(8, 1u, dummy_cut);
        }
        // Reject genuinely bogus false-match MVs (scene cuts / random distant outliers
        // that barely beat (0,0)), while preserving partial-overlap boundary tiles and
        // temporally/spatially coherent pans.
        uint pdist = (uint)(abs(best_mv.x - pred_mv.x) + abs(best_mv.y - pred_mv.y));
        bool bad_mv = (src_raw_sad > 2240u) ||
                      (src_raw_sad > 1408u && pdist > 16u && src_raw_sad * 10u >= zero_sad * 9u);
        int2 v = bad_mv ? int2(0, 0) : best_mv;
        g_tile_mvs[ty * tiles_x + tx] = qpel ? v : (v / 4);
    }
}

// =========================================================================
// Pass 1s: Closed-Loop Spatial-Median MV Coherence Filter (1 threadgroup/tile)
// Eliminates both isolated +-1..2 px / sub-pel tile jitter (smooth_thr) and
// wild false-match outlier tiles (diff_l1 > 8 healed only when closed-loop
// prediction error against g_prev_yuvd is genuinely improved), while leaving
// static HUDs and crisp moving-object boundaries untouched.
// =========================================================================
[numthreads(64, 1, 1)]
void CSMvSmooth(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tx = gid.x;
    uint ty = gid.y;
    if (tx >= tiles_x || ty >= tiles_y) return;

    uint tidx = ty * tiles_x + tx;
    int2 m0 = g_prev_mvs[tidx];
    int2 mL = g_prev_mvs[ty * tiles_x + ((tx > 0u) ? tx - 1u : tx)];
    int2 mR = g_prev_mvs[ty * tiles_x + min(tx + 1u, tiles_x - 1u)];
    int2 mU = g_prev_mvs[((ty > 0u) ? ty - 1u : ty) * tiles_x + tx];
    int2 mD = g_prev_mvs[min(ty + 1u, tiles_y - 1u) * tiles_x + tx];
    int2 m_med = int2(Med5(m0.x, mL.x, mR.x, mU.x, mD.x),
                      Med5(m0.y, mL.y, mR.y, mU.y, mD.y));
    int diff_l1 = abs(m0.x - m_med.x) + abs(m0.y - m_med.y);
    if (diff_l1 == 0) {
        if (gtid.x == 0u) g_tile_mvs[tidx] = m0;
        return;
    }

    uint si = gtid.x & 7u;
    uint sj = gtid.x >> 3;
    uint fpx = min(tx * 32u + 2u + si * 4u, width - 1u);
    uint fpy = min(ty * 32u + 2u + sj * 4u, height - 1u);
    int fcy = LumaOf(g_src_rgba.Load(int3((int)fpx, (int)fpy, 0)).rgb);
    int ry0 = (int)(FetchRefQ(fpx, fpy, MvQ(m0)) & 0xFFu);
    int rym = (int)(FetchRefQ(fpx, fpy, MvQ(m_med)) & 0xFFu);
    // Max 64-sample SAD is 64 * 255 = 16320 < 65535 -> pack both in 1 uint32.
    gs_sad_red[gtid.x] = ((uint)abs(fcy - ry0) << 16) | (uint)abs(fcy - rym);
    GroupMemoryBarrierWithGroupSync();

    [unroll] for (uint off = 32u; off > 0u; off >>= 1) {
        if (gtid.x < off) gs_sad_red[gtid.x] += gs_sad_red[gtid.x + off];
        GroupMemoryBarrierWithGroupSync();
    }
    if (gtid.x == 0u) {
        uint packed = gs_sad_red[0];
        uint sad0 = packed >> 16;
        uint sad_m = packed & 0xFFFFu;
        if (diff_l1 <= 8) {
            uint smooth_thr = (tq_qp >= 28u) ? 96u : 48u;
            g_tile_mvs[tidx] = (sad_m <= sad0 + smooth_thr) ? m_med : m0;
        } else {
            uint out_thr = (m0.x != 0 || m0.y != 0) ? 16u : 0u;
            g_tile_mvs[tidx] = (sad_m < sad0 + out_thr) ? m_med : m0;
        }
    }
}

// =========================================================================
// Pass 1a: Single-Pass Motion-Compensated Prediction + Decay (1 thread/pixel)
// Computes FetchRefQ + LeakyStep ONCE per pixel instead of repeating it in
// CSIntraTCount, CSEncode, and the drop-revert loop.
// Every 16x16 workgroup lies strictly inside one 32x32 tile (uniform mvq),
// so sub-pel phases cooperatively cache a 19x19 patch in groupshared (11x
// fewer global loads) and run a bit-exact separable 4-tap HEVC filter.
//   g_tflag_out (host binds zcvid_enc_pred_uav to u5): [decayed Y|Cb|Cr|orig_d]
//   g_cur_yuvd  (u0): pre-initialized to [decayed Y|Cb|Cr|half_d] (all-zero trits)
// =========================================================================
groupshared uint gs_mc_ref[361]; // 19x19 packed reference patch
groupshared int  gs_mc_h0[304];  // 19x16 unrounded horizontal 4-tap Y
groupshared int  gs_mc_h1[304];  // 19x16 unrounded horizontal 4-tap Cb
groupshared int  gs_mc_h2[304];  // 19x16 unrounded horizontal 4-tap Cr

[numthreads(16, 16, 1)]
void CSMotionPred(uint3 dtid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    // Piggyback CSTqClear (24 words) on workgroup (0,0) to eliminate a separate
    // dispatch + UAV pipeline-drain barrier before CSIntraTCount.
    if (gid.x == 0u && gid.y == 0u && (mode_flags & 4u) != 0u) {
        uint lid0 = gtid.y * 16u + gtid.x;
        if (lid0 < 24u) g_tq_scratch[TqHist() + lid0] = 0u;
    }
    uint gx0 = gid.x * 16u;
    uint gy0 = gid.y * 16u;
    int2 mvq = MvQ(g_prev_mvs[(gy0 >> 5) * tiles_x + (gx0 >> 5)]);
    int sxq0 = (int)gx0 * 4 - mvq.x;
    int syq0 = (int)gy0 * 4 - mvq.y;
    int ix0 = sxq0 >> 2;
    int iy0 = syq0 >> 2;
    int fx = sxq0 & 3;
    int fy = syq0 & 3;

    uint ref_w;
    if ((fx | fy) == 0) {
        if (dtid.x >= width || dtid.y >= height) return;
        ref_w = RefWordAt(ix0 + (int)gtid.x, iy0 + (int)gtid.y);
    } else {
        uint lid = gtid.y * 16u + gtid.x;
        [unroll] for (uint k = 0u; k < 2u; k++) {
            uint idx_k = lid + k * 256u;
            if (idx_k < 361u) {
                uint ry = idx_k / 19u;
                uint rx = idx_k - ry * 19u;
                gs_mc_ref[idx_k] = RefWordAt(ix0 - 1 + (int)rx, iy0 - 1 + (int)ry);
            }
        }
        GroupMemoryBarrierWithGroupSync();

        if (fy == 0) {
            if (dtid.x >= width || dtid.y >= height) return;
            int4 tx = TapsQ(fx);
            uint rb = (gtid.y + 1u) * 19u + gtid.x;
            uint w0 = gs_mc_ref[rb];
            uint w1 = gs_mc_ref[rb + 1u];
            uint w2 = gs_mc_ref[rb + 2u];
            uint w3 = gs_mc_ref[rb + 3u];
            int h0 = tx.x * (int)(w0 & 0xFFu) + tx.y * (int)(w1 & 0xFFu) +
                     tx.z * (int)(w2 & 0xFFu) + tx.w * (int)(w3 & 0xFFu);
            int h1 = tx.x * (int)((w0 >> 8) & 0xFFu) + tx.y * (int)((w1 >> 8) & 0xFFu) +
                     tx.z * (int)((w2 >> 8) & 0xFFu) + tx.w * (int)((w3 >> 8) & 0xFFu);
            int h2 = tx.x * (int)((w0 >> 16) & 0xFFu) + tx.y * (int)((w1 >> 16) & 0xFFu) +
                     tx.z * (int)((w2 >> 16) & 0xFFu) + tx.w * (int)((w3 >> 16) & 0xFFu);
            ref_w = (w1 & 0xFF000000u) |
                    (uint)clamp((h0 + 32) >> 6, 0, 255) |
                    ((uint)clamp((h1 + 32) >> 6, 0, 255) << 8) |
                    ((uint)clamp((h2 + 32) >> 6, 0, 255) << 16);
        } else if (fx == 0) {
            if (dtid.x >= width || dtid.y >= height) return;
            int4 ty = TapsQ(fy);
            uint cb = gtid.y * 19u + (gtid.x + 1u);
            uint w0 = gs_mc_ref[cb];
            uint w1 = gs_mc_ref[cb + 19u];
            uint w2 = gs_mc_ref[cb + 38u];
            uint w3 = gs_mc_ref[cb + 57u];
            int v0 = ty.x * (int)(w0 & 0xFFu) + ty.y * (int)(w1 & 0xFFu) +
                     ty.z * (int)(w2 & 0xFFu) + ty.w * (int)(w3 & 0xFFu);
            int v1 = ty.x * (int)((w0 >> 8) & 0xFFu) + ty.y * (int)((w1 >> 8) & 0xFFu) +
                     ty.z * (int)((w2 >> 8) & 0xFFu) + ty.w * (int)((w3 >> 8) & 0xFFu);
            int v2 = ty.x * (int)((w0 >> 16) & 0xFFu) + ty.y * (int)((w1 >> 16) & 0xFFu) +
                     ty.z * (int)((w2 >> 16) & 0xFFu) + ty.w * (int)((w3 >> 16) & 0xFFu);
            ref_w = (w1 & 0xFF000000u) |
                    (uint)clamp((v0 + 32) >> 6, 0, 255) |
                    ((uint)clamp((v1 + 32) >> 6, 0, 255) << 8) |
                    ((uint)clamp((v2 + 32) >> 6, 0, 255) << 16);
        } else {
            int4 tx = TapsQ(fx);
            [unroll] for (uint m2 = 0u; m2 < 2u; m2++) {
                uint m = lid + m2 * 256u;
                if (m < 304u) {
                    uint ry = m >> 4;
                    uint rx = m & 15u;
                    uint rb = ry * 19u + rx;
                    uint w0 = gs_mc_ref[rb];
                    uint w1 = gs_mc_ref[rb + 1u];
                    uint w2 = gs_mc_ref[rb + 2u];
                    uint w3 = gs_mc_ref[rb + 3u];
                    gs_mc_h0[m] = tx.x * (int)(w0 & 0xFFu) + tx.y * (int)(w1 & 0xFFu) +
                                  tx.z * (int)(w2 & 0xFFu) + tx.w * (int)(w3 & 0xFFu);
                    gs_mc_h1[m] = tx.x * (int)((w0 >> 8) & 0xFFu) + tx.y * (int)((w1 >> 8) & 0xFFu) +
                                  tx.z * (int)((w2 >> 8) & 0xFFu) + tx.w * (int)((w3 >> 8) & 0xFFu);
                    gs_mc_h2[m] = tx.x * (int)((w0 >> 16) & 0xFFu) + tx.y * (int)((w1 >> 16) & 0xFFu) +
                                  tx.z * (int)((w2 >> 16) & 0xFFu) + tx.w * (int)((w3 >> 16) & 0xFFu);
                }
            }
            GroupMemoryBarrierWithGroupSync();

            if (dtid.x >= width || dtid.y >= height) return;
            int4 ty = TapsQ(fy);
            uint c0 = gtid.y * 16u + gtid.x;
            int acc0 = ty.x * gs_mc_h0[c0] + ty.y * gs_mc_h0[c0 + 16u] +
                       ty.z * gs_mc_h0[c0 + 32u] + ty.w * gs_mc_h0[c0 + 48u];
            int acc1 = ty.x * gs_mc_h1[c0] + ty.y * gs_mc_h1[c0 + 16u] +
                       ty.z * gs_mc_h1[c0 + 32u] + ty.w * gs_mc_h1[c0 + 48u];
            int acc2 = ty.x * gs_mc_h2[c0] + ty.y * gs_mc_h2[c0 + 16u] +
                       ty.z * gs_mc_h2[c0 + 32u] + ty.w * gs_mc_h2[c0 + 48u];
            uint w00 = gs_mc_ref[(gtid.y + 1u) * 19u + (gtid.x + 1u)];
            ref_w = (w00 & 0xFF000000u) |
                    (uint)clamp((acc0 + 2048) >> 12, 0, 255) |
                    ((uint)clamp((acc1 + 2048) >> 12, 0, 255) << 8) |
                    ((uint)clamp((acc2 + 2048) >> 12, 0, 255) << 16);
        }
    }

    uint idx = dtid.y * width + dtid.x;
    int py  = LeakyStep((int)(ref_w & 0xFFu), 0, 0, (int)neutral_y);
    int pcb = LeakyStep((int)((ref_w >> 8) & 0xFFu), 0, 0, 128);
    int pcr = LeakyStep((int)((ref_w >> 16) & 0xFFu), 0, 0, 128);
    uint orig_d = (uint)max((int)delta_min, (int)((ref_w >> 24) & 0xFFu));
    uint half_d = (uint)max((int)delta_min, (int)(orig_d >> 1));
    uint ycbcr = (uint)py | ((uint)pcb << 8) | ((uint)pcr << 16);
    g_tflag_out[idx] = ycbcr | (orig_d << 24);
    g_cur_yuvd[idx]  = ycbcr | (half_d << 24);
}

// =========================================================================
// Pass 2: Full-Color Zcvid Encode + Jayant ADM + Disocclusion/Intra
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
};

uint PredWordAt(int x, int y) {
    x = clamp(x, 0, (int)width - 1);
    y = clamp(y, 0, (int)height - 1);
    return g_pred_yuvd[(uint)y * width + (uint)x];
}

// Closed-loop edge-preserving JND deadzone boost using only reference luma
// (encoder-only: computed lazily only when a channel exceeds base threshold).
int ComputeJnd(uint px, uint py, int ry) {
    if (threshold == 0u) return 0;
    int ry_l = (int)(PredWordAt((int)px - 1, (int)py) & 0xFFu);
    int ry_r = (int)(PredWordAt((int)px + 1, (int)py) & 0xFFu);
    int ry_u = (int)(PredWordAt((int)px, (int)py - 1) & 0xFFu);
    int ry_d = (int)(PredWordAt((int)px, (int)py + 1) & 0xFFu);
    int gx = abs(ry_r - ry_l);
    int gy = abs(ry_d - ry_u);
    int act_x = abs(ry_r - ry) + abs(ry - ry_l);
    int act_y = abs(ry_d - ry) + abs(ry - ry_u);
    int osc = (act_x - gx) + (act_y - gy);
    int coh = abs(gx - gy) + abs(act_x - act_y);
    int max_jump = max(max(abs(ry_r - ry), abs(ry - ry_l)),
                       max(abs(ry_d - ry), abs(ry - ry_u)));
    int tex_energy = max(osc - coh, 0);
    return (max_jump <= 64) ? clamp((tex_energy - 12) >> 3, 0, (int)threshold + 6) : 0;
}

PixState LoadPixFastXY(uint idx, uint px, uint py, uint num_px, uint intra_y0, uint intra_y1, bool allow_disocc) {
    PixState s;
    s.valid = (idx < num_px);
    if (!s.valid) {
        idx = num_px - 1u;
        py = height - 1u;
        px = width - 1u;
    }
    s.px = px;
    s.py = py;
    uint ref_w = g_pred_yuvd[idx];
    s.ry  = (int)(ref_w & 0xFFu);
    s.rcb = (int)((ref_w >> 8) & 0xFFu);
    s.rcr = (int)((ref_w >> 16) & 0xFFu);
    s.d   = max((int)delta_min, (int)((ref_w >> 24) & 0xFFu));
    float4 src_c = g_src_rgba.Load(int3(px, py, 0));
    RgbToYCbCr(src_c.rgb, s.cy, s.ccb, s.ccr);
    // When transform mode (bit 2) is active, CSIntraT handles rolling refresh
    // for all interior 4x4-aligned pixels; CSEncode only covers sub-4px borders.
    bool tq_roll_covered = ((mode_flags & 4u) != 0u) && (px < (width & ~3u)) && (py < (height & ~3u));
    s.rolling = !tq_roll_covered && (py >= intra_y0 && py < intra_y1);
    bool disocc = allow_disocc && (abs(s.cy - s.ry) > (int)intra_err_thr);
    s.want_intra = s.valid && (s.rolling || disocc);
    return s;
}

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

// Writes the reconstructed reference when any trit is non-zero.
// s.ry/s.rcb/s.rcr are already LeakyStep-decayed by CSMotionPred, and
// g_cur_yuvd already holds the all-zero-trit reconstruction.
void StoreDeltaNZ(uint idx, PixState s, int ty, int tcb, int tcr, int d_used, int next_d) {
    if ((ty | tcb | tcr) == 0) return;
    int ny  = clamp(s.ry  + ty  * d_used, 0, 255);
    int ncb = clamp(s.rcb + tcb * d_used, 0, 255);
    int ncr = clamp(s.rcr + tcr * d_used, 0, 255);
    g_cur_yuvd[idx] = (uint)ny | ((uint)ncb << 8) | ((uint)ncr << 16) | ((uint)next_d << 24);
}

// One pixel pair (idx_a even, pair k of its block). Writes the pair's
// reference (only when non-zero trits change it; g_cur_yuvd already holds the
// exact all-zero decay from CSMotionPred) and returns its trits + flags.
uint EncodePair(uint idx_a, uint k, uint num_pairs, uint num_px, uint intra_y0, uint intra_y1,
                bool allow_disocc, bool allow_chroma, bool budget_full, bool intra8, uint is_2x,
                out uint rec0, out uint rec1) {
    rec0 = 0u;
    rec1 = 0u;
    if (idx_a >= num_px) return 0x555u;

    uint py0 = idx_a / width;
    uint px0 = idx_a - py0 * width;
    // Early-skip blocks already coded by CSIntraT (zero wasted texture/buffer loads).
    if (TFlagAtXY(px0, py0) != 0u) {
        return 0x1555u;
    }
    // Since CSMotionPred already initialized g_cur_yuvd to the exact decay,
    // a full budget needs zero loads and zero writes.
    if (budget_full) {
        return 0x555u;
    }

    uint idx_b = idx_a + 1u;
    int y_thr = (int)threshold;
    int c_thr = (int)threshold * 2 + 4;

    // An odd pair is taken by its even neighbour when that one codes a 4-px
    // rolling intra.
    if ((k & 1u) != 0u && !intra8) {
        uint il = idx_a - 2u;
        uint ly = (px0 >= 2u) ? py0 : (il / width);
        uint lx = (px0 >= 2u) ? (px0 - 2u) : (il - ly * width);
        bool tq_rc = ((mode_flags & 4u) != 0u) && (lx < (width & ~3u)) && (ly < (height & ~3u));
        if (!tq_rc && ly >= intra_y0 && ly < intra_y1 && lx + 3u < width && il + 3u < num_px &&
            TFlagAtXY(lx, ly) == 0u && TFlagAtXY(lx + 2u, ly) == 0u) {
            return 0x555u;
        }
    }

    uint px_b = (px0 + 1u < width) ? (px0 + 1u) : 0u;
    uint py_b = (px0 + 1u < width) ? py0 : (py0 + 1u);
    PixState a = LoadPixFastXY(idx_a, px0, py0, num_px, intra_y0, intra_y1, allow_disocc);
    PixState b = LoadPixFastXY(idx_b, px_b, py_b, num_px, intra_y0, intra_y1, allow_disocc);

    int ty_a = 0, tcb_a = 0, tcr_a = 0;
    int ty_b = 0, tcb_b = 0, tcr_b = 0;
    uint flags = 0u;
    bool same_row_2 = (a.px + 1u < width);
    bool same_row_4 = (a.px + 3u < width);

    if (a.valid && same_row_2 && (a.want_intra || b.want_intra)) {
        if ((k & 1u) == 0u && (k + 1u) < num_pairs && a.rolling && !intra8 && same_row_4 && (idx_a + 3u < num_px) &&
            TFlagAtXY(a.px + 2u, a.py) == 0u) {
            int c_cy, c_ccb, c_ccr, d_cy, d_ccb, d_ccr;
            RgbToYCbCr(g_src_rgba.Load(int3(a.px + 2u, a.py, 0)).rgb, c_cy, c_ccb, c_ccr);
            RgbToYCbCr(g_src_rgba.Load(int3(a.px + 3u, a.py, 0)).rgb, d_cy, d_ccb, d_ccr);
            bool dith = (mode_flags & 64u) != 0u;
            uint brow = (a.py & 3u) * 4u;
            int y0_q6 = clamp((4 * a.cy + (dith ? (int)BAYER4[brow + ((a.px + 0u) & 3u)] : 8)) >> 4, 0, 63);
            int y1_q6 = clamp((4 * b.cy + (dith ? (int)BAYER4[brow + ((a.px + 1u) & 3u)] : 8)) >> 4, 0, 63);
            int y2_q6 = clamp((4 * c_cy + (dith ? (int)BAYER4[brow + ((a.px + 2u) & 3u)] : 8)) >> 4, 0, 63);
            int y3_q6 = clamp((4 * d_cy + (dith ? (int)BAYER4[brow + ((a.px + 3u) & 3u)] : 8)) >> 4, 0, 63);
            int cb_q6 = clamp(((a.ccb + b.ccb + c_ccb + d_ccb + 8) >> 4), 0, 63);
            int cr_q6 = clamp(((a.ccr + b.ccr + c_ccr + d_ccr + 8) >> 4), 0, 63);
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
        if (is_2x != 0u) {
            int ey  = b.valid ? (((a.cy  - a.ry)  + (b.cy  - b.ry))  >> 1) : (a.cy  - a.ry);
            int ecb = b.valid ? (((a.ccb - a.rcb) + (b.ccb - b.rcb)) >> 1) : (a.ccb - a.rcb);
            int ecr = b.valid ? (((a.ccr - a.rcr) + (b.ccr - b.rcr)) >> 1) : (a.ccr - a.rcr);
            int c_thr_p = (min(a.cy, b.valid ? b.cy : a.cy) < 32) ? min(c_thr, 6) : c_thr;
            bool need_q = (Quant(ey, y_thr, a.d) != 0) ||
                          (allow_chroma && (Quant(ecb, c_thr_p, a.d) != 0 || Quant(ecr, c_thr_p, a.d) != 0));
            if (need_q) {
                int ja = ComputeJnd(a.px, a.py, a.ry);
                int jnd_p = b.valid ? min(ja, ComputeJnd(b.px, b.py, b.ry)) : ja;
                ty_a  = Quant(ey, y_thr + jnd_p, a.d);
                tcb_a = allow_chroma ? Quant(ecb, c_thr_p + jnd_p * 2, a.d) : 0;
                tcr_a = allow_chroma ? Quant(ecr, c_thr_p + jnd_p * 2, a.d) : 0;

                int up = min((int)delta_max, a.d << 1);
                StoreDeltaNZ(idx_a, a, ty_a, tcb_a, tcr_a, a.d,
                             (ty_a == 0) ? max((int)delta_min, a.d >> 1) : up);
                if (b.valid) {
                    StoreDeltaNZ(idx_b, b, ty_a, tcb_a, tcr_a, a.d,
                                 (ty_a == 0) ? max((int)delta_min, b.d >> 1) : up);
                }
            }
        } else {
            int ey_a = a.cy - a.ry, ecb_a = a.ccb - a.rcb, ecr_a = a.ccr - a.rcr;
            int c_thr_a = (a.cy < 32) ? min(c_thr, 6) : c_thr;
            if (Quant(ey_a, y_thr, a.d) != 0 ||
                (allow_chroma && (Quant(ecb_a, c_thr_a, a.d) != 0 || Quant(ecr_a, c_thr_a, a.d) != 0))) {
                int ja = ComputeJnd(a.px, a.py, a.ry);
                ty_a  = Quant(ey_a, y_thr + ja, a.d);
                tcb_a = allow_chroma ? Quant(ecb_a, c_thr_a + ja * 2, a.d) : 0;
                tcr_a = allow_chroma ? Quant(ecr_a, c_thr_a + ja * 2, a.d) : 0;
                StoreDeltaNZ(idx_a, a, ty_a, tcb_a, tcr_a, a.d,
                             (ty_a == 0) ? max((int)delta_min, a.d >> 1)
                                         : min((int)delta_max, a.d << 1));
            }
            if (b.valid) {
                int ey_b = b.cy - b.ry, ecb_b = b.ccb - b.rcb, ecr_b = b.ccr - b.rcr;
                int c_thr_b = (b.cy < 32) ? min(c_thr, 6) : c_thr;
                if (Quant(ey_b, y_thr, b.d) != 0 ||
                    (allow_chroma && (Quant(ecb_b, c_thr_b, b.d) != 0 || Quant(ecr_b, c_thr_b, b.d) != 0))) {
                    int jb = ComputeJnd(b.px, b.py, b.ry);
                    ty_b  = Quant(ey_b, y_thr + jb, b.d);
                    tcb_b = allow_chroma ? Quant(ecb_b, c_thr_b + jb * 2, b.d) : 0;
                    tcr_b = allow_chroma ? Quant(ecr_b, c_thr_b + jb * 2, b.d) : 0;
                    StoreDeltaNZ(idx_b, b, ty_b, tcb_b, tcr_b, b.d,
                                 (ty_b == 0) ? max((int)delta_min, b.d >> 1)
                                             : min((int)delta_max, b.d << 1));
                }
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
        // 4-px rolling intra: 6-bit luma x4, shared 6-bit chroma (q6 << 2 so 32 -> 128 exact neutral).
        uint idx_a = (r0 & 0x3FFFFFu) << 1;
        uint cb_q6 = (r1 >> 24) & 63u;
        uint cr_q6 = (r0 >> 22) & 63u;
        uint chroma_d = ((cb_q6 << 2) << 8) | ((cr_q6 << 2) << 16) | dm;
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
groupshared uint gs_enc_scan[64];
groupshared uint gs_enc_base;

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

    // Hard rate control + folded CSTqFinal: compute exact accepted transform words
    // once per workgroup (eliminates the standalone CSTqFinal dispatch + UAV barrier).
    // g_out_counter[0] counts only CSEncode words; host adds counter[0] + counter[12].
    if (gtid.x == 0u) {
        if ((mode_flags & 4u) != 0u) {
            uint m = TqMisc();
            uint tq_tot = max(g_tq_scratch[m + 2u], g_tq_scratch[m + 3u]);
            gs_words = tq_tot;
            if (gid.x == 0u) {
                g_out_counter.Store(4, g_tq_scratch[m + 0u]);
                g_out_counter.Store(12, tq_tot);
            }
        } else {
            gs_words = 0u;
        }
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
    bool valid_blk = (gb < eff_blocks);
    uint blk = valid_blk ? ((gb + rot) % eff_blocks) : 0u;
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
    uint n_rec = 0u;
    if (valid_blk) {
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
        n_rec = n_intra + (nz_y ? 1u : 0u) + (nz_cb ? 1u : 0u) + (nz_cr ? 1u : 0u);
    }

    // Workgroup-aggregated record slot allocation: 1 global InterlockedAdd per 64 blocks.
    // Quiet workgroups skip the 12-barrier scan completely.
    if (gtid.x == 0u) gs_enc_base = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (n_rec != 0u) gs_enc_base = 1u;
    GroupMemoryBarrierWithGroupSync();
    if (gs_enc_base == 0u) return;

    uint my_words = 2u * n_rec;
    gs_enc_scan[gtid.x] = my_words;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint o = 1u; o < 64u; o <<= 1) {
        uint add = (gtid.x >= o) ? gs_enc_scan[gtid.x - o] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gs_enc_scan[gtid.x] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    uint local_off = gs_enc_scan[gtid.x] - my_words;
    if (gtid.x == 63u) {
        uint gtot = gs_enc_scan[63u];
        uint base = 0u;
        if (gtot > 0u) g_out_counter.InterlockedAdd(0, gtot, base);
        gs_enc_base = base;
    }
    GroupMemoryBarrierWithGroupSync();
    if (!valid_blk || n_rec == 0u) return;
    uint slot = cur_words + gs_enc_base + local_off;

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
    // plane(s) in g_cur_yuvd from precomputed g_pred_yuvd so Enc == Dec.
    if (drop_y || drop_cb || drop_cr) {
        for (uint p = 0; p < px_per_blk; p++) {
            uint pidx = base_px + p;
            if (pidx < num_px && ((intra_mask & (1u << p)) == 0u)) {
                uint pw = g_pred_yuvd[pidx];
                uint cur_w = g_cur_yuvd[pidx];
                if (drop_y) {
                    uint ny = pw & 0xFFu;
                    uint d  = (uint)max((int)delta_min, (int)((pw >> 24) & 0xFFu) >> 1);
                    cur_w = (cur_w & 0x00FFFF00u) | ny | (d << 24);
                }
                if (drop_cb) {
                    cur_w = (cur_w & 0xFFFF00FFu) | (pw & 0x0000FF00u);
                }
                if (drop_cr) {
                    cur_w = (cur_w & 0xFF00FFFFu) | (pw & 0x00FF0000u);
                }
                g_cur_yuvd[pidx] = cur_w;
            }
        }
    }
}

// =========================================================================
// Pass 1b: 16-Bin Global RDO Hybrid 8x8 + 4x4 Integer Transform
//
// Threads are grouped in 4s (lane 0..3 = TL, TR, BL, BR of an 8x8 superblock,
// 16 superblocks per 64-thread workgroup). Each lane evaluates its 4x4 block;
// when eff_tq_qp >= 16 or on a scene cut, lane 0 also evaluates an 8x8
// bilinear-upsampled transform (pred = 2 intra / 3 inter, anchored at TL) and
// all 4 lanes compute exact 64-pixel SSD. Whichever mode (8x8 vs 4x4) delivers
// more SSD gain per bit wins RDO with zero overlap and zero extra dispatches.
// =========================================================================
groupshared uint gs_tq[64];
groupshared uint gs_hist[16];
groupshared uint gs_stage_base;
groupshared uint gs_overflow_w;
groupshared uint gs_any_try8;
groupshared int  gs_it_c[1024]; // 64 threads x 16 ints for spill-free CSIntraT inverse
groupshared uint gs_it_ok8[16]; // 1 if superblock's lane 0 admitted an 8x8 block

// Superblock cooperation scratch for CSIntraTCount (~5.6 KB total).
groupshared uint gs_sb_s[256];    // 16 superblocks x 16 2x2 source cell averages [Y|Cb|Cr]
groupshared uint gs_sb_p[256];    // 16 superblocks x 16 2x2 pred cell averages   [Y|Cb|Cr]
groupshared int  gs_sb_c8[768];   // 16 superblocks x 48 inverse-transform 8x8 residual ints
groupshared uint gs_sb_lv8[256];  // 16 superblocks x 16 packed 8x8 levels (avoids per-thread spill)
groupshared uint gs_sb_4gain[64]; // per-lane 4x4 SSD gain
groupshared uint gs_sb_4n[64];    // per-lane 4x4 coef count
groupshared uint gs_sb_4bef[64];  // per-lane 4x4 SSD before
groupshared uint gs_sb_ssd8[64];  // per-lane 8x8 SSD after (exact 16-px quadrant)
groupshared uint gs_sb_win8[16];  // 1 if 8x8 won RDO for this superblock
groupshared uint gs_sb_roll[16];  // non-zero if any quadrant is on rolling refresh or OOB
groupshared uint gs_sb_pred8[16]; // pred8 (2/3 legacy or 4/5 DCT8) chosen by lane 0

// 4-lane parallel 12-mul Factored HEVC 8x8 DCT8 scratch (~11.5 KB).
groupshared int  gs_dct8[1024];     // 16 superblocks x 64 ints
groupshared uint gs_dct8_lv[1536];  // 64 lanes x 24 uints (3 planes x 8 packed int16 pairs)
groupshared uint gs_sb_n8[64];      // per-lane non-zero DCT8 coef count
groupshared uint gs_sb_sad8[64];    // per-lane SAD / AC-SAD
groupshared uint gs_sb_sad8_128[64];
groupshared uint gs_sb_sumy[64];
groupshared uint gs_sb_qpy8[16];
groupshared uint gs_sb_qpc8[16];
groupshared uint gs_sb_slot8[16];   // staging slot0 for winning DCT8 superblock
groupshared uint gs_sb_hf8[16];     // active HF sub-block count for RDO cost
groupshared uint gs_it_from8[16];   // CSIntraT: staging source offset
groupshared uint gs_it_dst8[16];    // CSIntraT: output record dst offset
groupshared uint gs_it_w8[16];      // CSIntraT: output record words
groupshared uint gs_it_b8[16];      // CSIntraT: TL block index b
groupshared uint gs_it_nz8[16];     // CSIntraT: per-plane non-zero flag

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

// Bijection mapping slot q in [0, nb) to a 4x4 block index b, grouping aligned
// 8x8 superblocks into 4 consecutive slots (lane 0..3 = TL, TR, BL, BR).
uint TqBlockOfSlot(uint q, out bool in_sb) {
    uint bw = (width + 3u) >> 2;
    uint bh = (height + 3u) >> 2;
    uint sbw = bw >> 1;
    uint sbh = bh >> 1;
    uint nsb = sbw * sbh;
    uint n_aligned = nsb * 4u;
    if (q < n_aligned) {
        in_sb = true;
        uint sq = q >> 2;
        uint lane = q & 3u;
        uint sb = (sq + (frame_id * 10133u) % nsb) % nsb;
        uint sbx = sb % sbw;
        uint sby = sb / sbw;
        uint bx = sbx * 2u + (lane & 1u);
        uint by = sby * 2u + (lane >> 1);
        return by * bw + bx;
    }
    in_sb = false;
    uint rem = q - n_aligned;
    uint odd_col = (bw & 1u) * (sbh * 2u);
    if (rem < odd_col) {
        return rem * bw + (bw - 1u);
    }
    uint bx = rem - odd_col;
    return (bh - 1u) * bw + bx;
}

// Intra-block exact integer bilinear upsample of a 4x4 grid to 8x8 pixels.
// Shared word-for-word with zcvid_decode.hlsl and zcvid_gpu_cpu_dec.h.
int TqSample8x8(int c[16], uint lx, uint ly) {
    uint u = lx >> 1;
    uint v = ly >> 1;
    uint un = ((lx & 1u) == 0u) ? ((u > 0u) ? (u - 1u) : 0u) : ((u < 3u) ? (u + 1u) : 3u);
    uint vn = ((ly & 1u) == 0u) ? ((v > 0u) ? (v - 1u) : 0u) : ((v < 3u) ? (v + 1u) : 3u);
    return (9 * c[v * 4u + u] + 3 * c[v * 4u + un] + 3 * c[vn * 4u + u] + c[vn * 4u + un] + 8) >> 4;
}

int TqSample8x8Gs(uint base, uint lx, uint ly) {
    uint u = lx >> 1;
    uint v = ly >> 1;
    uint un = ((lx & 1u) == 0u) ? ((u > 0u) ? (u - 1u) : 0u) : ((u < 3u) ? (u + 1u) : 3u);
    uint vn = ((ly & 1u) == 0u) ? ((v > 0u) ? (v - 1u) : 0u) : ((v < 3u) ? (v + 1u) : 3u);
    return (9 * gs_sb_c8[base + v * 4u + u] + 3 * gs_sb_c8[base + v * 4u + un] +
            3 * gs_sb_c8[base + vn * 4u + u] + gs_sb_c8[base + vn * 4u + un] + 8) >> 4;
}

uint TqRdoBin(uint ssd_gain, uint n_coefs) {
    uint rd_score = (ssd_gain << 2) / (n_coefs + 3u);
    uint hb = firstbithigh(max(rd_score, 1u));
    uint sub = (hb >= 1u) ? ((rd_score >> (hb - 1u)) & 1u) : 0u;
    uint raw_bin = (hb >= 5u) ? ((hb - 5u) * 2u + sub) : 0u;
    return min(raw_bin, 15u);
}

uint Eval4x4Core(uint sp[16], uint pp[16], uint pred, uint qp_y, uint qp_c,
                 uint ssd_before, out uint out_ssd_after, out uint lv_pk[16]) {
    [unroll] for (uint zi = 0u; zi < 16u; zi++) lv_pk[zi] = 0u;
    uint n = 0u;
    uint ssd_after = 0u;
    uint ssd_chroma = 0u;
    bool is_inter = ((pred & 1u) != 0u);
    [unroll] for (uint pl = 0u; pl < 3u; pl++) {
        uint sh = pl * 8u;
        int x[16];
        [unroll] for (uint k0 = 0u; k0 < 16u; k0++) {
            int sv = (int)((sp[k0] >> sh) & 0xFFu);
            int pv = is_inter ? (int)((pp[k0] >> sh) & 0xFFu) : 128;
            x[k0] = sv - pv;
        }
        [unroll] for (uint i = 0u; i < 4u; i++) {
            int s0 = x[i * 4u] + x[i * 4u + 3u], s1 = x[i * 4u + 1u] + x[i * 4u + 2u];
            int d0 = x[i * 4u] - x[i * 4u + 3u], d1 = x[i * 4u + 1u] - x[i * 4u + 2u];
            x[i * 4u] = s0 + s1; x[i * 4u + 2u] = s0 - s1;
            x[i * 4u + 1u] = 2 * d0 + d1; x[i * 4u + 3u] = d0 - 2 * d1;
        }
        [unroll] for (uint jc = 0u; jc < 4u; jc++) {
            int s0 = x[jc] + x[12u + jc], s1 = x[4u + jc] + x[8u + jc];
            int d0 = x[jc] - x[12u + jc], d1 = x[4u + jc] - x[8u + jc];
            x[jc] = s0 + s1; x[8u + jc] = s0 - s1;
            x[4u + jc] = 2 * d0 + d1; x[12u + jc] = d0 - 2 * d1;
        }
        uint qp = (pl == 0u) ? qp_y : qp_c;
        int qbits = 15 + (int)(qp / 6u);
        int fq_dc = (1 << qbits) / 3;
        int fq_ac = (!is_inter || qp < 16u) ? fq_dc : ((1 << qbits) / 6);
        int c[16];
        bool pl_nz = false;
        [unroll] for (uint k = 0u; k < 16u; k++) {
            uint r = TQ_ZZ[k];
            int w = x[r];
            int fq = (k == 0u) ? fq_dc : fq_ac;
            int z = (abs(w) * TQ_MF[(qp % 6u) * 3u + TqClass(r)] + fq) >> qbits;
            z = min(z, 255);
            int l = (w < 0) ? -z : z;
            lv_pk[k] |= ((uint)(l + 256) << (pl * 10u));
            c[r] = TqDequant(l, r, qp);
            if (l != 0) { n++; pl_nz = true; }
        }
        if (pl_nz) {
            TqInverse(c);
        }
        [unroll] for (uint k1 = 0u; k1 < 16u; k1++) {
            int sv = (int)((sp[k1] >> sh) & 0xFFu);
            int pv = is_inter ? (int)((pp[k1] >> sh) & 0xFFu) : 128;
            int rv = pl_nz ? clamp(pv + c[k1], 0, 255) : pv;
            int err = sv - rv;
            uint e2 = (uint)(err * err);
            if (pl == 0u) ssd_after += e2;
            else ssd_chroma += e2;
        }
    }
    out_ssd_after = ssd_after + (ssd_chroma >> 1);
    return n;
}

// --- 16-Bin Global RDO Deterministic Budget Allocation (6 tiny dispatches) ---
[numthreads(32, 1, 1)]
void CSTqClear(uint3 gtid : SV_GroupThreadID) {
    if (gtid.x < 24u) {
        g_tq_scratch[TqHist() + gtid.x] = 0u;
    }
}

[numthreads(64, 1, 1)]
void CSIntraTCount(uint3 dtid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    if (gtid.x < 16u) {
        gs_hist[gtid.x] = 0u;
        gs_sb_win8[gtid.x] = 0u;
        gs_sb_roll[gtid.x] = 0u;
        gs_sb_pred8[gtid.x] = 3u;
        gs_sb_qpy8[gtid.x] = 12u;
        gs_sb_qpc8[gtid.x] = 15u;
        gs_sb_slot8[gtid.x] = 0xFFFFFFFFu;
        gs_sb_hf8[gtid.x] = 0u;
    }
    if (gtid.x == 0u) { gs_overflow_w = 0u; gs_any_try8 = 0u; }
    gs_sb_4gain[gtid.x] = 0u;
    gs_sb_4n[gtid.x] = 0u;
    gs_sb_4bef[gtid.x] = 0u;
    gs_sb_ssd8[gtid.x] = 0u;
    gs_sb_n8[gtid.x] = 0u;
    gs_sb_sad8[gtid.x] = 0u;
    gs_sb_sad8_128[gtid.x] = 0u;
    gs_sb_sumy[gtid.x] = 0u;
    GroupMemoryBarrierWithGroupSync();

    uint bw = (width + 3u) >> 2;
    uint bh = (height + 3u) >> 2;
    uint nb = bw * bh;
    uint sbid = gtid.x >> 2;
    uint lane = gtid.x & 3u;

    bool in_sb = false;
    uint b = 0u;
    uint x0 = 0u, y0 = 0u;
    bool valid4 = false;
    if (dtid.x < nb && (mode_flags & 4u) != 0u) {
        b = TqBlockOfSlot(dtid.x, in_sb);
        x0 = (b % bw) * 4u;
        y0 = (b / bw) * 4u;
        valid4 = (x0 + 4u <= width) && (y0 + 4u <= height);
    }

    bool is_cut = TqIsSceneCut();
    uint cycles = max(intra_cycles, 1u);
    uint roll_bands = (bh + 1u) >> 1;
    uint sb_per_slot = max((roll_bands + cycles - 1u) / cycles, 1u);
    uint sb_y0 = (frame_id % cycles) * sb_per_slot;
    uint sb_y1 = sb_y0 + sb_per_slot;
    uint sb_y = y0 >> 3;
    bool hit_roll = valid4 && !is_cut && (sb_y >= sb_y0) && (sb_y < sb_y1);

    if (!in_sb || !valid4) {
        uint dummy_r;
        InterlockedOr(gs_sb_roll[sbid], 1u, dummy_r);
    }

    bool allow_inter = (mode_flags & 8u) != 0u;
    bool use_dct8 = (mode_flags & 256u) != 0u;
    uint eff_tq_qp = is_cut ? TqSceneCutQp() : tq_qp;
    bool may_8x8 = !use_dct8 && (is_cut || hit_roll || (eff_tq_qp >= 20u));
    uint sp[16], pp[16];
    uint ndis = 0u;
    int sad = 0, sad_128 = 0, max_abs_res = 0, sum_sy = 0;
    uint ssd_before = 0u;

    if (valid4) {
        [unroll] for (uint j = 0u; j < 16u; j++) {
            uint px = x0 + (j & 3u);
            uint py = y0 + (j >> 2);
            int sy_v, sb_v, sr_v;
            RgbToYCbCr(g_src_rgba.Load(int3(px, py, 0)).rgb, sy_v, sb_v, sr_v);
            sp[j] = (uint)sy_v | ((uint)sb_v << 8) | ((uint)sr_v << 16);
            sum_sy += sy_v;
            uint pw = g_pred_yuvd[py * width + px];
            pp[j] = pw;
            int py_v = (int)(pw & 0xFFu);
            int pb_v = (int)((pw >> 8) & 0xFFu);
            int pr_v = (int)((pw >> 16) & 0xFFu);
            int dy = sy_v - py_v;
            int db = sb_v - pb_v;
            int dr = sr_v - pr_v;
            int ady = abs(dy);
            int m_res = max(ady, max(abs(db), abs(dr)));
            if (m_res > max_abs_res) max_abs_res = m_res;
            if (ady > (int)intra_err_thr) ndis++;
            sad += ady;
            sad_128 += abs(sy_v - 128);
            ssd_before += (uint)(dy * dy) + ((uint)(db * db + dr * dr) >> 1);
        }
        gs_sb_4bef[gtid.x] = ssd_before;
        if (in_sb && may_8x8) {
            uint qx = (lane & 1u) * 2u;
            uint qy = (lane >> 1) * 2u;
            [unroll] for (uint cy2 = 0u; cy2 < 2u; cy2++) {
                [unroll] for (uint cx2 = 0u; cx2 < 2u; cx2++) {
                    uint i00 = (cy2 * 2u) * 4u + (cx2 * 2u);
                    uint s00 = sp[i00], s01 = sp[i00 + 1u], s10 = sp[i00 + 4u], s11 = sp[i00 + 5u];
                    uint p00 = pp[i00], p01 = pp[i00 + 1u], p10 = pp[i00 + 4u], p11 = pp[i00 + 5u];
                    uint sy_a = ((s00 & 0xFFu) + (s01 & 0xFFu) + (s10 & 0xFFu) + (s11 & 0xFFu) + 2u) >> 2;
                    uint sb_a = (((s00 >> 8) & 0xFFu) + ((s01 >> 8) & 0xFFu) + ((s10 >> 8) & 0xFFu) + ((s11 >> 8) & 0xFFu) + 2u) >> 2;
                    uint sr_a = (((s00 >> 16) & 0xFFu) + ((s01 >> 16) & 0xFFu) + ((s10 >> 16) & 0xFFu) + ((s11 >> 16) & 0xFFu) + 2u) >> 2;
                    uint py_a = ((p00 & 0xFFu) + (p01 & 0xFFu) + (p10 & 0xFFu) + (p11 & 0xFFu) + 2u) >> 2;
                    uint pb_a = (((p00 >> 8) & 0xFFu) + ((p01 >> 8) & 0xFFu) + ((p10 >> 8) & 0xFFu) + ((p11 >> 8) & 0xFFu) + 2u) >> 2;
                    uint pr_a = (((p00 >> 16) & 0xFFu) + ((p01 >> 16) & 0xFFu) + ((p10 >> 16) & 0xFFu) + ((p11 >> 16) & 0xFFu) + 2u) >> 2;
                    uint cell = (qy + cy2) * 4u + (qx + cx2);
                    gs_sb_s[sbid * 16u + cell] = sy_a | (sb_a << 8) | (sr_a << 16);
                    gs_sb_p[sbid * 16u + cell] = py_a | (pb_a << 8) | (pr_a << 16);
                }
            }
        }
    }

    // 1. Evaluate 4x4 block for every valid lane.
    uint w = 0u, qp_y = 12u, qp_c = 15u, pred = 0u, rdo_bin = 0u;
    uint lv_pk[16];
    [unroll] for (uint z0 = 0u; z0 < 16u; z0++) lv_pk[z0] = 0u;
    if (valid4) {
        int sad_gate = (eff_tq_qp >= 16u) ? max(16 * (int)tq_inter_thr, 24) : (16 * (int)tq_inter_thr);
        uint ssd_gate = (eff_tq_qp <= 8u) ? 32u
                      : (eff_tq_qp <= 12u) ? 64u
                      : (eff_tq_qp < 24u) ? 128u : 256u;
        bool force_kf = (mode_flags & 128u) != 0u;
        bool want_disocc = hit_roll || force_kf || (ndis >= 4u) || (is_cut && sad >= 64);
        bool want_inter  = allow_inter && (sad > sad_gate || ssd_before >= ssd_gate);
        if (want_disocc || want_inter) {
            pred = (is_cut || hit_roll || !allow_inter || (sad_128 + 32 < sad)) ? 0u : 1u;
            uint aq_qp4 = hit_roll ? clamp(eff_tq_qp, 12u, 42u) : eff_tq_qp;
            if (is_cut) {
                int mean_y4 = (sum_sy + 8) >> 4;
                uint ac_sad4 = 0u;
                [unroll] for (uint ja = 0u; ja < 16u; ja++) {
                    ac_sad4 += (uint)abs((int)(sp[ja] & 0xFFu) - mean_y4);
                }
                aq_qp4 = AqAdjustCutQp(eff_tq_qp, ac_sad4);
            }
            uint qp_floor = (pred != 0u && max_abs_res <= 64) ? 6u : 12u;
            qp_y = clamp(aq_qp4, qp_floor, (pred != 0u) ? TQ_INTER_QP_MAX : 51u);
            qp_c = TqChromaQp(qp_y);
            if (is_cut) qp_c = max(qp_c, (qp_y >= 26u) ? (qp_y - 6u) : qp_y);
            uint ssd_after4 = 0u;
            uint n4 = Eval4x4Core(sp, pp, pred, qp_y, qp_c, ssd_before, ssd_after4, lv_pk);
            bool ok4 = true;
            if (pred != 0u) {
                uint min_gain = (qp_y < 12u) ? 8u : 32u;
                if (n4 == 0u || ssd_after4 * 5u >= ssd_before * 4u || ssd_before <= ssd_after4 + min_gain) {
                    ok4 = false;
                }
            } else if (n4 == 0u) {
                n4 = 1u;
            }
            if (ok4) {
                uint gain4 = (ssd_before > ssd_after4) ? (ssd_before - ssd_after4) : 1u;
                gs_sb_4gain[gtid.x] = hit_roll ? ssd_after4 : gain4;
                gs_sb_4n[gtid.x] = n4;
                rdo_bin = TqRdoBin(gain4, n4);
                if (hit_roll) {
                    rdo_bin = (eff_tq_qp <= 34u) ? max(rdo_bin, 13u) : max(rdo_bin, 8u);
                }
                w = 2u * n4;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // 2. Evaluate 8x8 superblock candidate:
    //    - use_dct8 (mode_flags bit 8): 4-lane parallel exact 12-mul Factored HEVC DCT8
    //      (pred = 4 intra / 5 inter) with Parseval CSBF HF pruning & Parseval spectral SSD
    //      (proved < 0.086% of spatial SSD in Lean 4; saves 9 barriers + 2D IDCT8 in RDO).
    //    - legacy (!use_dct8): bilinear-upsampled 4x4 transform (pred = 2 / 3).
    uint base4 = sbid * 4u;
    uint bef_all_sb = gs_sb_4bef[base4] + gs_sb_4bef[base4 + 1u] +
                      gs_sb_4bef[base4 + 2u] + gs_sb_4bef[base4 + 3u];
    uint act4_any = gs_sb_4n[base4] | gs_sb_4n[base4 + 1u] |
                    gs_sb_4n[base4 + 2u] | gs_sb_4n[base4 + 3u];
    bool try_8x8 = (gs_sb_roll[sbid] == 0u) &&
                   (use_dct8 ? (is_cut || hit_roll || act4_any != 0u || bef_all_sb >= 256u)
                             : (is_cut || hit_roll || (eff_tq_qp >= 20u && bef_all_sb >= 512u)));
    uint n8 = 0u, qp_y8 = 12u, qp_c8 = 15u, pred8 = use_dct8 ? 5u : 3u;
    uint hf_sub8 = 0u;

    if (use_dct8) {
        if (try_8x8) {
            gs_any_try8 = 1u;
            gs_sb_sad8[gtid.x] = (uint)sad;
            gs_sb_sad8_128[gtid.x] = (uint)sad_128;
            gs_sb_sumy[gtid.x] = (uint)sum_sy;
        }
        GroupMemoryBarrierWithGroupSync();

        if (gs_any_try8 != 0u) {
            if (try_8x8 && lane == 0u) {
                uint sum8_y = gs_sb_sumy[base4] + gs_sb_sumy[base4 + 1u] +
                              gs_sb_sumy[base4 + 2u] + gs_sb_sumy[base4 + 3u];
                gs_sb_sumy[base4] = (sum8_y + 32u) >> 6; // mean8_y
            }
            GroupMemoryBarrierWithGroupSync();

            if (try_8x8 && is_cut) {
                int mean8_y = (int)gs_sb_sumy[base4];
                uint ac_lane = 0u;
                [unroll] for (uint ja8 = 0u; ja8 < 16u; ja8++) {
                    ac_lane += (uint)abs((int)(sp[ja8] & 0xFFu) - mean8_y);
                }
                gs_sb_sumy[gtid.x] = ac_lane;
            }
            GroupMemoryBarrierWithGroupSync();

            if (try_8x8 && lane == 0u) {
                uint sad8 = gs_sb_sad8[base4] + gs_sb_sad8[base4 + 1u] +
                            gs_sb_sad8[base4 + 2u] + gs_sb_sad8[base4 + 3u];
                uint sad8_128 = gs_sb_sad8_128[base4] + gs_sb_sad8_128[base4 + 1u] +
                                gs_sb_sad8_128[base4 + 2u] + gs_sb_sad8_128[base4 + 3u];
                uint ac_sad8 = 0u;
                if (is_cut) {
                    ac_sad8 = (gs_sb_sumy[base4] + gs_sb_sumy[base4 + 1u] +
                               gs_sb_sumy[base4 + 2u] + gs_sb_sumy[base4 + 3u]) >> 2;
                }
                bool is_inter8 = !is_cut && !hit_roll && allow_inter && (sad8_128 + 128u >= sad8);
                pred8 = is_inter8 ? 5u : 4u;
                uint q8_src = is_cut ? AqAdjustCutQp(eff_tq_qp, ac_sad8)
                                     : (hit_roll ? clamp(eff_tq_qp, 12u, 42u) : eff_tq_qp);
                qp_y8 = clamp(q8_src, 12u, 51u);
                qp_c8 = TqChromaQp(qp_y8);
                if (is_cut) qp_c8 = max(qp_c8, (qp_y8 >= 26u) ? (qp_y8 - 6u) : qp_y8);
                gs_sb_pred8[sbid] = pred8;
                gs_sb_qpy8[sbid] = qp_y8;
                gs_sb_qpc8[sbid] = qp_c8;
            }
            GroupMemoryBarrierWithGroupSync();

            uint p8_mode = gs_sb_pred8[sbid];
            bool is_inter8 = ((p8_mode & 1u) != 0u);
            uint qy_s = gs_sb_qpy8[sbid];
            uint qc_s = gs_sb_qpc8[sbid];
            uint sb_base = sbid * 64u;
            uint qx0 = (lane & 1u) * 4u;
            uint qy0 = (lane >> 1) * 4u;
            uint n8_lane = 0u;
            uint ssd8_q = 0u;
            uint lv_base = gtid.x * 24u;
            uint lam_qp_hf = clamp(is_cut ? eff_tq_qp : qy_s, 12u, 51u);
            uint qstep_hf = ((uint)TQ_V[(lam_qp_hf % 6u) * 3u] << (lam_qp_hf / 6u)) >> 2;
            uint lam_hf = max((qstep_hf * qstep_hf) >> 7, 2u);
            if (is_cut && max_record_words != 0u && max_record_words < nb) lam_hf <<= 1;

            [unroll] for (uint pl = 0u; pl < 3u; pl++) {
                uint sh = pl * 8u;
                if (try_8x8) {
                    [unroll] for (uint j = 0u; j < 16u; j++) {
                        int sv = (int)((sp[j] >> sh) & 0xFFu);
                        int pv = is_inter8 ? (int)((pp[j] >> sh) & 0xFFu) : 128;
                        gs_dct8[sb_base + (qy0 + (j >> 2)) * 8u + (qx0 + (j & 3u))] = sv - pv;
                    }
                }
                GroupMemoryBarrierWithGroupSync();

                // 4-lane parallel Forward Pass 1 (2 rows per lane, 12-mul Karatsuba + Gauss).
                if (try_8x8) {
                    [unroll] for (uint rr = 0u; rr < 2u; rr++) {
                        uint rb = sb_base + (lane * 2u + rr) * 8u;
                        int x0 = gs_dct8[rb + 0u], x1 = gs_dct8[rb + 1u], x2 = gs_dct8[rb + 2u], x3 = gs_dct8[rb + 3u];
                        int x4 = gs_dct8[rb + 4u], x5 = gs_dct8[rb + 5u], x6 = gs_dct8[rb + 6u], x7 = gs_dct8[rb + 7u];
                        Dct8Fwd1D(x0, x1, x2, x3, x4, x5, x6, x7, 2, 2);
                        gs_dct8[rb + 0u] = x0; gs_dct8[rb + 1u] = x1; gs_dct8[rb + 2u] = x2; gs_dct8[rb + 3u] = x3;
                        gs_dct8[rb + 4u] = x4; gs_dct8[rb + 5u] = x5; gs_dct8[rb + 6u] = x6; gs_dct8[rb + 7u] = x7;
                    }
                }
                GroupMemoryBarrierWithGroupSync();

                // 4-lane parallel Forward Pass 2 (2 cols per lane, 12-mul Karatsuba + Gauss).
                if (try_8x8) {
                    [unroll] for (uint cc = 0u; cc < 2u; cc++) {
                        uint cb = sb_base + lane * 2u + cc;
                        int x0 = gs_dct8[cb + 0u],  x1 = gs_dct8[cb + 8u],  x2 = gs_dct8[cb + 16u], x3 = gs_dct8[cb + 24u];
                        int x4 = gs_dct8[cb + 32u], x5 = gs_dct8[cb + 40u], x6 = gs_dct8[cb + 48u], x7 = gs_dct8[cb + 56u];
                        Dct8Fwd1D(x0, x1, x2, x3, x4, x5, x6, x7, 256, 9);
                        gs_dct8[cb + 0u]  = x0; gs_dct8[cb + 8u]  = x1; gs_dct8[cb + 16u] = x2; gs_dct8[cb + 24u] = x3;
                        gs_dct8[cb + 32u] = x4; gs_dct8[cb + 40u] = x5; gs_dct8[cb + 48u] = x6; gs_dct8[cb + 56u] = x7;
                    }
                }
                GroupMemoryBarrierWithGroupSync();

                // 4-lane parallel Quantize + Parseval HF Sub-block Pruning + Parseval Spectral SSD.
                // Lane 0=LL, 1=LH, 2=HL, 3=HH. Exact orthonormal scale: 16x -> 256x energy.
                if (try_8x8) {
                    uint qp = (pl == 0u) ? qy_s : qc_s;
                    uint qp_mod = qp % 6u;
                    uint qp_div = qp / 6u;
                    int mf = TQ8_MF[qp_mod];
                    int qbits = 18 + (int)qp_div;
                    int fq_dc = (1 << qbits) / 3;
                    int fq_ac = (!is_inter8 || qp < 16u) ? fq_dc : ((1 << qbits) / 6);
                    uint n_sub = 0u;
                    uint kept_hi = 0u, kept_lo = 0u;
                    uint zero_hi = 0u, zero_lo = 0u;
                    int qlv[16];
                    [unroll] for (uint k = 0u; k < 16u; k++) {
                        uint r = TQ_ZZ[k];
                        int wv = gs_dct8[sb_base + (qy0 + (r >> 2)) * 8u + (qx0 + (r & 3u))];
                        int fq = (lane == 0u && k == 0u) ? fq_dc : fq_ac;
                        int z = (abs(wv) * mf + fq) >> qbits;
                        z = min(z, 2047);
                        int l = (wv < 0) ? -z : z;
                        int dq_v = (l != 0) ? Tq8Dequant(l, qp) : 0;
                        qlv[k] = l;
                        if (l != 0) n_sub++;
                        uint ad = (uint)min(abs(wv - dq_v), 65535);
                        uint sq_d = ad * ad;
                        kept_hi += sq_d >> 8;
                        kept_lo += sq_d & 255u;
                        if (lane >= 1u) {
                            uint aw = (uint)min(abs(wv), 65535);
                            uint sq_w = aw * aw;
                            zero_hi += sq_w >> 8;
                            zero_lo += sq_w & 255u;
                        }
                    }
                    uint ssd_sub = kept_hi + ((kept_lo + 128u) >> 8);
                    if (lane >= 1u && n_sub > 0u) {
                        uint ssd_zero = zero_hi + ((zero_lo + 128u) >> 8);
                        uint gain_sub = (ssd_zero > ssd_sub) ? (ssd_zero - ssd_sub) : 0u;
                        uint w_gain = (pl == 0u) ? gain_sub : (gain_sub >> 1);
                        if (w_gain <= lam_hf * (n_sub + 1u)) {
                            n_sub = 0u;
                            ssd_sub = ssd_zero;
                            [unroll] for (uint kz = 0u; kz < 16u; kz++) qlv[kz] = 0;
                        } else {
                            uint dummy_hf;
                            InterlockedAdd(gs_sb_hf8[sbid], 1u, dummy_hf);
                        }
                    }
                    ssd8_q += (pl == 0u) ? ssd_sub : (ssd_sub >> 1);
                    n8_lane += n_sub;
                    uint pl_lv = lv_base + pl * 8u;
                    [unroll] for (uint kp = 0u; kp < 8u; kp++) {
                        gs_dct8_lv[pl_lv + kp] = ((uint)(qlv[kp * 2u] + 2048)) |
                                                 (((uint)(qlv[kp * 2u + 1u] + 2048)) << 16);
                    }
                }
                GroupMemoryBarrierWithGroupSync();
            }

            if (try_8x8) {
                gs_sb_ssd8[gtid.x] = ssd8_q;
                gs_sb_n8[gtid.x] = n8_lane;
            }
            GroupMemoryBarrierWithGroupSync();

            if (try_8x8 && lane == 0u) {
                n8 = gs_sb_n8[base4] + gs_sb_n8[base4 + 1u] +
                     gs_sb_n8[base4 + 2u] + gs_sb_n8[base4 + 3u];
                if (!is_inter8 && n8 == 0u) n8 = 1u;
                hf_sub8 = gs_sb_hf8[sbid];
            }
        }
    } else {
        if (try_8x8 && lane == 0u) {
            [unroll] for (uint z8 = 0u; z8 < 16u; z8++) gs_sb_lv8[sbid * 16u + z8] = 0u;
            int sad8 = 0, sad8_128 = 0, sum8_y = 0;
            [unroll] for (uint i8 = 0u; i8 < 16u; i8++) {
                uint svw = gs_sb_s[sbid * 16u + i8];
                uint pvw = gs_sb_p[sbid * 16u + i8];
                int sy = (int)(svw & 0xFFu), py = (int)(pvw & 0xFFu);
                sum8_y += sy;
                sad8 += abs(sy - py);
                sad8_128 += abs(sy - 128);
            }
            int mean8_y = (sum8_y + 8) >> 4;
            uint ac_sad8 = 0u;
            if (is_cut) {
                [unroll] for (uint ia8 = 0u; ia8 < 16u; ia8++) {
                    ac_sad8 += (uint)abs((int)(gs_sb_s[sbid * 16u + ia8] & 0xFFu) - mean8_y);
                }
            }
            bool is_inter8 = !is_cut && !hit_roll && allow_inter && (sad8_128 + 32 >= sad8);
            pred8 = is_inter8 ? 3u : 2u;
            gs_sb_pred8[sbid] = pred8;
            uint q8_src = is_cut ? AqAdjustCutQp(eff_tq_qp, ac_sad8)
                                 : (hit_roll ? clamp(eff_tq_qp, 12u, 42u) : eff_tq_qp);
            qp_y8 = clamp(q8_src, 12u, 51u);
            qp_c8 = TqChromaQp(qp_y8);
            if (is_cut) qp_c8 = max(qp_c8, (qp_y8 >= 26u) ? (qp_y8 - 6u) : qp_y8);

            [unroll] for (uint pl = 0u; pl < 3u; pl++) {
                uint sh = pl * 8u;
                int x[16];
                [unroll] for (uint k0 = 0u; k0 < 16u; k0++) {
                    int sv = (int)((gs_sb_s[sbid * 16u + k0] >> sh) & 0xFFu);
                    int pv = is_inter8 ? (int)((gs_sb_p[sbid * 16u + k0] >> sh) & 0xFFu) : 128;
                    x[k0] = sv - pv;
                }
                [unroll] for (uint i = 0u; i < 4u; i++) {
                    int s0 = x[i * 4u] + x[i * 4u + 3u], s1 = x[i * 4u + 1u] + x[i * 4u + 2u];
                    int d0 = x[i * 4u] - x[i * 4u + 3u], d1 = x[i * 4u + 1u] - x[i * 4u + 2u];
                    x[i * 4u] = s0 + s1; x[i * 4u + 2u] = s0 - s1;
                    x[i * 4u + 1u] = 2 * d0 + d1; x[i * 4u + 3u] = d0 - 2 * d1;
                }
                [unroll] for (uint jc = 0u; jc < 4u; jc++) {
                    int s0 = x[jc] + x[12u + jc], s1 = x[4u + jc] + x[8u + jc];
                    int d0 = x[jc] - x[12u + jc], d1 = x[4u + jc] - x[8u + jc];
                    x[jc] = s0 + s1; x[8u + jc] = s0 - s1;
                    x[4u + jc] = 2 * d0 + d1; x[12u + jc] = d0 - 2 * d1;
                }
                uint qp = (pl == 0u) ? qp_y8 : qp_c8;
                int qbits = 15 + (int)(qp / 6u);
                int fq_dc = (1 << qbits) / 3;
                int fq_ac = (!is_inter8 || qp < 16u) ? fq_dc : ((1 << qbits) / 6);
                int c[16];
                bool pl_nz = false;
                [unroll] for (uint k = 0u; k < 16u; k++) {
                    uint r = TQ_ZZ[k];
                    int wv = x[r];
                    int fq = (k == 0u) ? fq_dc : fq_ac;
                    int z = (abs(wv) * TQ_MF[(qp % 6u) * 3u + TqClass(r)] + fq) >> qbits;
                    z = min(z, 255);
                    int l = (wv < 0) ? -z : z;
                    gs_sb_lv8[sbid * 16u + k] |= ((uint)(l + 256) << (pl * 10u));
                    c[r] = TqDequant(l, r, qp);
                    if (l != 0) { n8++; pl_nz = true; }
                }
                if (pl_nz) {
                    TqInverse(c);
                } else {
                    [unroll] for (uint kz = 0u; kz < 16u; kz++) c[kz] = 0;
                }
                uint c8_off = sbid * 48u + pl * 16u;
                [unroll] for (uint kw = 0u; kw < 16u; kw++) gs_sb_c8[c8_off + kw] = c[kw];
            }
            if (!is_inter8 && n8 == 0u) n8 = 1u;
        }
        GroupMemoryBarrierWithGroupSync();

        // 3. All 4 lanes compute exact 16-pixel bilinear 8x8 SSD for their quadrant
        //    directly from groupshared gs_sb_c8 (zero register spills).
        if (try_8x8) {
            uint p8_mode = gs_sb_pred8[sbid];
            bool is_inter8 = ((p8_mode & 1u) != 0u);
            uint qx0 = (lane & 1u) * 4u;
            uint qy0 = (lane >> 1) * 4u;
            uint c8_base = sbid * 48u;
            uint ssd8_q = 0u;
            [loop] for (uint j = 0u; j < 16u; j++) {
                uint lx = qx0 + (j & 3u);
                uint ly = qy0 + (j >> 2);
                int ry = (is_inter8 ? (int)(pp[j] & 0xFFu) : 128) + TqSample8x8Gs(c8_base, lx, ly);
                int rb = (is_inter8 ? (int)((pp[j] >> 8) & 0xFFu) : 128) + TqSample8x8Gs(c8_base + 16u, lx, ly);
                int rr = (is_inter8 ? (int)((pp[j] >> 16) & 0xFFu) : 128) + TqSample8x8Gs(c8_base + 32u, lx, ly);
                int ey = (int)(sp[j] & 0xFFu) - clamp(ry, 0, 255);
                int eb = (int)((sp[j] >> 8) & 0xFFu) - clamp(rb, 0, 255);
                int er = (int)((sp[j] >> 16) & 0xFFu) - clamp(rr, 0, 255);
                ssd8_q += (uint)(ey * ey) + ((uint)(eb * eb + er * er) >> 1);
            }
            gs_sb_ssd8[gtid.x] = ssd8_q;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // 4. Lane 0 decides 8x8 vs 4x4 by exact H.264/HEVC Lagrangian RDO (J = D + lambda * R).
    if (try_8x8 && lane == 0u && n8 > 0u) {
        uint bef_all = bef_all_sb;
        uint aft8_all = gs_sb_ssd8[base4] + gs_sb_ssd8[base4 + 1u] +
                        gs_sb_ssd8[base4 + 2u] + gs_sb_ssd8[base4 + 3u];
        uint n4_sum = gs_sb_4n[base4] + gs_sb_4n[base4 + 1u] +
                      gs_sb_4n[base4 + 2u] + gs_sb_4n[base4 + 3u];
        uint act4 = (gs_sb_4n[base4] > 0u ? 1u : 0u) + (gs_sb_4n[base4 + 1u] > 0u ? 1u : 0u) +
                    (gs_sb_4n[base4 + 2u] > 0u ? 1u : 0u) + (gs_sb_4n[base4 + 3u] > 0u ? 1u : 0u);
        uint cost8 = n8 + (use_dct8 ? (2u + hf_sub8) : 3u);
        uint cost4 = n4_sum + act4 * 3u;
        bool win8 = false;
        uint rdo_gain8 = 1u;
        bool is_intra8 = ((pred8 & 1u) == 0u);
        if (hit_roll) {
            uint aft4_all = gs_sb_4gain[base4] + gs_sb_4gain[base4 + 1u] +
                            gs_sb_4gain[base4 + 2u] + gs_sb_4gain[base4 + 3u];
            uint lam_qp = clamp(qp_y8, 12u, 51u);
            uint qstep_q4 = ((uint)TQ_V[(lam_qp % 6u) * 3u] << (lam_qp / 6u)) >> 2;
            uint lam_roll = max((qstep_q4 * qstep_q4) >> 7, 4u);
            win8 = (act4 == 0u) || (aft8_all + lam_roll * cost8 <= aft4_all + lam_roll * cost4);
            rdo_gain8 = (bef_all > aft8_all) ? (bef_all - aft8_all) : 1u;
        } else {
            uint gain8 = (bef_all > aft8_all) ? (bef_all - aft8_all) : 0u;
            uint gain4_all = gs_sb_4gain[base4] + gs_sb_4gain[base4 + 1u] +
                             gs_sb_4gain[base4 + 2u] + gs_sb_4gain[base4 + 3u];
            bool tight_cut = is_cut && (max_record_words != 0u) && (max_record_words * 10u < nb);
            bool force_kf = (mode_flags & 128u) != 0u;
            bool good8 = is_intra8
                             ? (force_kf || bef_all > 128u || tight_cut)
                             : (aft8_all * 5u < bef_all * 4u && gain8 > (use_dct8 ? 32u : 96u));
            uint lam_qp = clamp(eff_tq_qp, 12u, 51u);
            uint qstep_q4 = ((uint)TQ_V[(lam_qp % 6u) * 3u] << (lam_qp / 6u)) >> 2;
            uint lam_cost = max((qstep_q4 * qstep_q4) >> 7, 1u);
            uint cut_lam = tight_cut ? (lam_cost * 2u) : lam_cost;
            win8 = good8 && (act4 == 0u ||
                             (gain8 + cut_lam * cost4 >= gain4_all + cut_lam * cost8));
            rdo_gain8 = max(gain8, 1u);
        }
        if (win8) {
            gs_sb_win8[sbid] = 1u;
            w = 2u * n8;
            qp_y = qp_y8;
            qp_c = qp_c8;
            pred = pred8;
            rdo_bin = TqRdoBin(rdo_gain8, n8);
            if (hit_roll) {
                rdo_bin = (eff_tq_qp <= 34u) ? max(rdo_bin, 13u) : max(rdo_bin, 8u);
            }
            if (!use_dct8) {
                [unroll] for (uint zc = 0u; zc < 16u; zc++) lv_pk[zc] = gs_sb_lv8[sbid * 16u + zc];
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (try_8x8 && lane != 0u && gs_sb_win8[sbid] != 0u) {
        w = 0u;
    }

    // Workgroup-aggregated staging slot allocation: 1 global InterlockedAdd per 64 blocks.
    uint local_off = GroupExclusive64(gtid.x, w);
    if (gtid.x == 63u) {
        uint gtot = local_off + w;
        uint base = 0u;
        if (gtot > 0u) InterlockedAdd(g_tq_scratch[TqMisc() + 1u], gtot, base);
        gs_stage_base = base;
    }
    GroupMemoryBarrierWithGroupSync();

    bool my_dct8_win = use_dct8 && (gs_sb_win8[sbid] != 0u);
    if (w != 0u) {
        uint stage_n = 0u, stride = 0u;
        g_tq_stage.GetDimensions(stage_n, stride);
        uint slot = gs_stage_base + local_off;
        if (slot + w <= stage_n) {
            g_tq_scratch[dtid.x] = w | (rdo_bin << 24);
            g_tq_scratch[TqPositions() + dtid.x] = slot;
            if (my_dct8_win) {
                gs_sb_slot8[sbid] = slot;
            } else {
                uint b_base = 0xF0000000u | (b * 48u);
                uint written_c = 0u;
                [unroll] for (uint pl = 0u; pl < 3u; pl++) {
                    uint qp = (pl == 0u) ? qp_y : qp_c;
                    uint qp_pred = (qp << 16) | (pred << 24);
                    uint p_off = pl * 16u;
                    uint sh = pl * 10u;
                    [unroll] for (uint k = 0u; k < 16u; k++) {
                        int l = (int)((lv_pk[k] >> sh) & 0x3FFu) - 256;
                        if (l != 0) {
                            g_tq_stage[slot]      = b_base + p_off + k;
                            g_tq_stage[slot + 1u] = ((uint)l & 0xFFFFu) | qp_pred;
                            slot += 2u;
                            written_c += 2u;
                        }
                    }
                }
                if (written_c == 0u) {
                    g_tq_stage[slot]      = b_base;
                    g_tq_stage[slot + 1u] = (qp_y << 16) | (pred << 24);
                }
            }
            uint dummy;
            InterlockedAdd(gs_hist[rdo_bin], w, dummy);
        } else {
            g_tq_scratch[dtid.x] = 0u;
            uint dummy;
            InterlockedAdd(gs_overflow_w, w, dummy);
        }
    } else {
        g_tq_scratch[dtid.x] = 0u;
    }

    GroupMemoryBarrierWithGroupSync();
    // 4-lane parallel staging for winning 8x8 DCT8 superblocks (zero serial bottleneck).
    if (my_dct8_win) {
        uint slot0 = gs_sb_slot8[sbid];
        if (slot0 != 0xFFFFFFFFu) {
            uint n0 = gs_sb_n8[base4 + 0u];
            uint n1 = gs_sb_n8[base4 + 1u];
            uint n2 = gs_sb_n8[base4 + 2u];
            uint n3 = gs_sb_n8[base4 + 3u];
            uint ntot = n0 + n1 + n2 + n3;
            if (ntot == 0u) {
                if (lane == 0u) {
                    g_tq_stage[slot0]      = 0xF0000000u | (b * 48u);
                    g_tq_stage[slot0 + 1u] = (gs_sb_qpy8[sbid] << 16) | (gs_sb_pred8[sbid] << 24);
                }
            } else {
                uint off_q = 0u;
                if (lane >= 1u) off_q += n0 * 2u;
                if (lane >= 2u) off_q += n1 * 2u;
                if (lane >= 3u) off_q += n2 * 2u;
                uint slot = slot0 + off_q;
                uint b_base = 0xF0000000u | (b * 48u);
                uint qy_s = gs_sb_qpy8[sbid];
                uint qc_s = gs_sb_qpc8[sbid];
                uint p8_s = gs_sb_pred8[sbid];
                uint lv_base = gtid.x * 24u;
                [unroll] for (uint pl = 0u; pl < 3u; pl++) {
                    uint qp = (pl == 0u) ? qy_s : qc_s;
                    uint qp_pred = (qp << 16) | (p8_s << 24);
                    uint p_off = pl * 16u;
                    uint pl_lv = lv_base + pl * 8u;
                    [unroll] for (uint kp = 0u; kp < 8u; kp++) {
                        uint pk = gs_dct8_lv[pl_lv + kp];
                        int l0 = (int)(pk & 0xFFFFu) - 2048;
                        int l1 = (int)(pk >> 16) - 2048;
                        if (l0 != 0) {
                            g_tq_stage[slot]      = b_base + p_off + (kp * 2u);
                            g_tq_stage[slot + 1u] = ((uint)l0 & 0xFFFFu) | qp_pred;
                            slot += 2u;
                        }
                        if (l1 != 0) {
                            g_tq_stage[slot]      = b_base + p_off + (kp * 2u + 1u);
                            g_tq_stage[slot + 1u] = ((uint)l1 & 0xFFFFu) | qp_pred;
                            slot += 2u;
                        }
                    }
                }
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();
    if (gtid.x < 16u && gs_hist[gtid.x] != 0u) {
        uint dummy;
        InterlockedAdd(g_tq_scratch[TqHist() + gtid.x], gs_hist[gtid.x], dummy);
    }
    if (gtid.x == 0u) {
        uint wg_tot = gs_overflow_w;
        [unroll] for (uint hb = 0u; hb < 16u; hb++) wg_tot += gs_hist[hb];
        if (wg_tot != 0u) {
            uint dummy;
            InterlockedAdd(g_tq_scratch[TqMisc() + 0u], wg_tot, dummy);
        }
    }
}

[numthreads(64, 1, 1)]
void CSTqMid(uint3 dtid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    if (gtid.x == 0u) {
        uint cap = TqInterCap();
        uint acc = 0u;
        uint cut = 0u;
        [unroll] for (int b = 15; b >= 0; b--) {
            uint h = g_tq_scratch[TqHist() + (uint)b];
            if (acc + h > cap) {
                cut = (uint)b;
                break;
            }
            acc += h;
        }
        gs_hist[0] = cut;
        gs_hist[1] = 0u;
        gs_hist[2] = 0u;
        if (gid.x == 0u) g_tq_scratch[TqMisc() + 4u] = cut;
    }
    GroupMemoryBarrierWithGroupSync();
    uint cut = gs_hist[0];
    uint c = g_tq_scratch[dtid.x];
    uint w = c & 0x00FFFFFFu;
    uint bin = (c >> 24) & 15u;
    uint w_hi = (w != 0u && bin > cut) ? w : 0u;
    uint w_eq = (w != 0u && bin == cut) ? w : 0u;
    if (w_hi != 0u) gs_hist[1] = 1u;
    if (w_eq != 0u) gs_hist[2] = 1u;
    GroupMemoryBarrierWithGroupSync();
    if (gs_hist[1] != 0u) {
        uint ex_hi = GroupExclusive64(gtid.x, w_hi);
        if (gtid.x == 63u) g_tq_scratch[TqSumsHi() + gid.x] = ex_hi + w_hi;
    } else if (gtid.x == 63u) {
        g_tq_scratch[TqSumsHi() + gid.x] = 0u;
    }
    if (gs_hist[2] != 0u) {
        uint ex_eq = GroupExclusive64(gtid.x, w_eq);
        if (gtid.x == 63u) g_tq_scratch[TqSumsEq() + gid.x] = ex_eq + w_eq;
    } else if (gtid.x == 63u) {
        g_tq_scratch[TqSumsEq() + gid.x] = 0u;
    }
}

groupshared uint gs_scan[1024];
void ScanArray(uint t, uint src_base, uint dst_base, bool store_hi_total) {
    uint p = TqPositions();
    uint ng = p >> 6;
    uint per = (ng + 1023u) >> 10;
    uint g0 = min(t * per, ng);
    uint g1 = min(g0 + per, ng);
    uint sum = 0u;
    for (uint g = g0; g < g1; g++) sum += g_tq_scratch[src_base + g];
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
        uint v = g_tq_scratch[src_base + g2];
        g_tq_scratch[dst_base + g2] = run;
        run += v;
    }
    if (store_hi_total && t == 1023u) {
        g_tq_scratch[TqMisc() + 2u] = run;
    }
}

[numthreads(1024, 1, 1)]
void CSTqScanA(uint3 gtid : SV_GroupThreadID) {
    ScanArray(gtid.x, TqSumsHi(), TqBasesHi(), true);
    GroupMemoryBarrierWithGroupSync();
    ScanArray(gtid.x, TqSumsEq(), TqBasesEq(), false);
}

[numthreads(64, 1, 1)]
void CSIntraT(uint3 dtid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint p = TqPositions();
    uint cap = TqInterCap();
    uint cut = g_tq_scratch[TqMisc() + 4u];
    uint hi_total = g_tq_scratch[TqMisc() + 2u];
    uint c = g_tq_scratch[dtid.x];
    uint w = c & 0x00FFFFFFu;
    uint bin = (c >> 24) & 15u;
    uint w_hi = (w != 0u && bin > cut) ? w : 0u;
    uint w_eq = (w != 0u && bin == cut) ? w : 0u;
    if (gtid.x < 16u) gs_it_ok8[gtid.x] = 0u;
    if (gtid.x == 0u) { gs_hist[1] = 0u; gs_hist[2] = 0u; gs_hist[3] = 0u; gs_hist[4] = 0u; }
    GroupMemoryBarrierWithGroupSync();
    if (w_hi != 0u) gs_hist[1] = 1u;
    if (w_eq != 0u) gs_hist[2] = 1u;
    GroupMemoryBarrierWithGroupSync();

    uint bw = (width + 3u) >> 2;
    uint bh = (height + 3u) >> 2;
    uint nb = bw * bh;
    if ((gs_hist[1] | gs_hist[2]) == 0u) {
        if (dtid.x < nb) {
            bool dummy_sb = false;
            g_tflag_out[TqBlockOfSlot(dtid.x, dummy_sb)] = 0u;
        }
        return;
    }

    uint a_hi = (gs_hist[1] != 0u) ? (g_tq_scratch[TqBasesHi() + gid.x] + GroupExclusive64(gtid.x, w_hi)) : 0u;
    uint a_eq = (gs_hist[2] != 0u) ? (g_tq_scratch[TqBasesEq() + gid.x] + GroupExclusive64(gtid.x, w_eq)) : 0u;

    bool ok = false;
    uint dst = 0u;
    if (w_hi != 0u) {
        ok = true;
        dst = a_hi;
    } else if (w_eq != 0u && hi_total + a_eq + w <= cap) {
        ok = true;
        dst = hi_total + a_eq;
        uint dummy;
        InterlockedMax(gs_hist[3], dst + w, dummy);
    }

    uint sbid = gtid.x >> 2;
    uint lane = gtid.x & 3u;

    bool in_sb = false;
    uint b = 0u;
    if (dtid.x < nb) {
        b = TqBlockOfSlot(dtid.x, in_sb);
        if (ok) {
            uint from = g_tq_scratch[p + dtid.x];
            uint x0 = (b % bw) * 4u;
            uint y0 = (b / bw) * 4u;
            uint pred_mode = (g_tq_stage[from + 1u] >> 24) & 0xFFu;
            bool is_inter = ((pred_mode & 1u) != 0u);
            bool is_8x8 = (pred_mode >= 2u);
            bool is_dct8 = (pred_mode >= 4u);
            if (is_8x8 && in_sb) {
                gs_it_ok8[sbid] = is_dct8 ? (is_inter ? 3u : 2u) : 1u;
                if (is_dct8) {
                    gs_it_from8[sbid] = from;
                    gs_it_dst8[sbid] = dst;
                    gs_it_w8[sbid] = w;
                    gs_it_b8[sbid] = b;
                    gs_hist[4] = 1u;
                }
            }
            if (!is_8x8) {
                uint out_yuv[16];
                [unroll] for (uint j0 = 0u; j0 < 16u; j0++) {
                    uint idx = (y0 + (j0 >> 2)) * width + x0 + (j0 & 3u);
                    out_yuv[j0] = is_inter ? ((g_pred_yuvd[idx] & 0x00FFFFFFu) | (delta_min << 24))
                                           : (0x00808080u | (delta_min << 24));
                }
                uint cur = 0u;
                uint cbase = gtid.x * 16u;
                [unroll] for (uint pl = 0u; pl < 3u; pl++) {
                    uint pl_lo = b * 48u + pl * 16u;
                    uint pl_hi = pl_lo + 16u;
                    bool any_c = false;
                    [unroll] for (uint z = 0u; z < 16u; z++) gs_it_c[cbase + z] = 0;
                    [loop] while (cur < w) {
                        uint rw0 = g_tq_stage[from + cur];
                        uint pid = rw0 & 0x0FFFFFFFu;
                        if (pid >= pl_hi) break;
                        uint rw1 = g_tq_stage[from + cur + 1u];
                        g_out_records[dst + cur]      = rw0;
                        g_out_records[dst + cur + 1u] = rw1;
                        cur += 2u;
                        int level = (int)(rw1 << 16) >> 16;
                        if (level != 0) {
                            uint qp = (rw1 >> 16) & 0xFFu;
                            uint r = TQ_ZZ[pid - pl_lo];
                            gs_it_c[cbase + r] = TqDequant(level, r, qp);
                            any_c = true;
                        }
                    }
                    if (any_c) {
                        int ic[16];
                        [unroll] for (uint z2 = 0u; z2 < 16u; z2++) ic[z2] = gs_it_c[cbase + z2];
                        TqInverse(ic);
                        uint sh = pl * 8u;
                        uint mask = ~(0xFFu << sh);
                        [unroll] for (uint q2 = 0u; q2 < 16u; q2++) {
                            int pv = (int)((out_yuv[q2] >> sh) & 0xFFu);
                            uint nv = (uint)clamp(pv + ic[q2], 0, 255);
                            out_yuv[q2] = (out_yuv[q2] & mask) | (nv << sh);
                        }
                    }
                }
                [unroll] for (uint j1 = 0u; j1 < 16u; j1++) {
                    uint idx = (y0 + (j1 >> 2)) * width + x0 + (j1 & 3u);
                    g_cur_yuvd[idx] = out_yuv[j1];
                }
            } else if (!is_dct8) {
                // Legacy 8x8 bilinear-upsampled transform reconstruction (64 pixels).
                uint cur = 0u;
                uint cbase = gtid.x * 16u;
                uint c8_base = sbid * 48u;
                [unroll] for (uint pl = 0u; pl < 3u; pl++) {
                    uint pl_lo = b * 48u + pl * 16u;
                    uint pl_hi = pl_lo + 16u;
                    bool any_c = false;
                    [unroll] for (uint z = 0u; z < 16u; z++) gs_it_c[cbase + z] = 0;
                    [loop] while (cur < w) {
                        uint rw0 = g_tq_stage[from + cur];
                        uint pid = rw0 & 0x0FFFFFFFu;
                        if (pid >= pl_hi) break;
                        uint rw1 = g_tq_stage[from + cur + 1u];
                        g_out_records[dst + cur]      = rw0;
                        g_out_records[dst + cur + 1u] = rw1;
                        cur += 2u;
                        int level = (int)(rw1 << 16) >> 16;
                        if (level != 0) {
                            uint qp = (rw1 >> 16) & 0xFFu;
                            uint r = TQ_ZZ[pid - pl_lo];
                            gs_it_c[cbase + r] = TqDequant(level, r, qp);
                            any_c = true;
                        }
                    }
                    uint c8_off = c8_base + pl * 16u;
                    if (any_c) {
                        int ic[16];
                        [unroll] for (uint z2 = 0u; z2 < 16u; z2++) ic[z2] = gs_it_c[cbase + z2];
                        TqInverse(ic);
                        [unroll] for (uint k0 = 0u; k0 < 16u; k0++) gs_sb_c8[c8_off + k0] = ic[k0];
                    } else {
                        [unroll] for (uint k0 = 0u; k0 < 16u; k0++) gs_sb_c8[c8_off + k0] = 0;
                    }
                }
                [loop] for (uint ly = 0u; ly < 8u; ly++) {
                    [unroll] for (uint lx = 0u; lx < 8u; lx++) {
                        uint idx = (y0 + ly) * width + x0 + lx;
                        uint pw = is_inter ? g_pred_yuvd[idx] : 0x00808080u;
                        int ry = clamp((int)(pw & 0xFFu) + TqSample8x8Gs(c8_base, lx, ly), 0, 255);
                        int rb = clamp((int)((pw >> 8) & 0xFFu) + TqSample8x8Gs(c8_base + 16u, lx, ly), 0, 255);
                        int rr = clamp((int)((pw >> 16) & 0xFFu) + TqSample8x8Gs(c8_base + 32u, lx, ly), 0, 255);
                        g_cur_yuvd[idx] = (uint)ry | ((uint)rb << 8) | ((uint)rr << 16) | (delta_min << 24);
                    }
                }
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // 4-lane parallel Exact 8x8 Factored HEVC IDCT8 reconstruction (pred = 4 intra / 5 inter).
    if (gs_hist[4] != 0u) {
        uint ok8_m = gs_it_ok8[sbid];
        bool do_dct8 = (ok8_m >= 2u);
        bool is_inter8 = (ok8_m == 3u);
        uint from8 = gs_it_from8[sbid];
        uint dst8 = gs_it_dst8[sbid];
        uint w8 = gs_it_w8[sbid];
        uint b_tl = gs_it_b8[sbid];
        uint x0_sb = (b_tl % bw) * 4u;
        uint y0_sb = (b_tl / bw) * 4u;
        uint qx0 = (lane & 1u) * 4u;
        uint qy0 = (lane >> 1) * 4u;
        uint sb_base = sbid * 64u;

        // Cooperatively copy all records of this 8x8 DCT8 superblock to g_out_records.
        if (do_dct8) {
            [loop] for (uint i = lane * 2u; i < w8; i += 8u) {
                g_out_records[dst8 + i]      = g_tq_stage[from8 + i];
                g_out_records[dst8 + i + 1u] = g_tq_stage[from8 + i + 1u];
            }
        }

        [unroll] for (uint pl = 0u; pl < 3u; pl++) {
            if (lane == 0u) gs_it_nz8[sbid] = 0u;
            if (do_dct8) {
                uint zbase = sb_base + lane * 16u;
                [unroll] for (uint z = 0u; z < 16u; z++) gs_dct8[zbase + z] = 0;
            }
            GroupMemoryBarrierWithGroupSync();

            if (do_dct8) {
                bool any_pl = false;
                [loop] for (uint i = lane * 2u; i < w8; i += 8u) {
                    uint rw0 = g_tq_stage[from8 + i];
                    uint pid = rw0 & 0x0FFFFFFFu;
                    uint rb = pid / 48u;
                    uint rem = pid - rb * 48u;
                    uint rpl = rem >> 4;
                    if (rpl == pl) {
                        uint rw1 = g_tq_stage[from8 + i + 1u];
                        int level = (int)(rw1 << 16) >> 16;
                        if (level != 0) {
                            uint qp = (rw1 >> 16) & 0xFFu;
                            uint rk = rem & 15u;
                            uint r_off = rb - b_tl;
                            uint rq = (r_off >= bw) ? (2u + (r_off - bw)) : r_off;
                            uint u0 = (rq & 1u) * 4u;
                            uint v0 = (rq >> 1) * 4u;
                            uint r = TQ_ZZ[rk];
                            gs_dct8[sb_base + (v0 + (r >> 2)) * 8u + (u0 + (r & 3u))] = Tq8Dequant(level, qp);
                            any_pl = true;
                        }
                    }
                }
                if (any_pl) gs_it_nz8[sbid] = 1u;
            }
            GroupMemoryBarrierWithGroupSync();

            bool nz_pl = do_dct8 && (gs_it_nz8[sbid] != 0u);
            if (nz_pl) {
                [unroll] for (uint rr = 0u; rr < 2u; rr++) {
                    uint rb = sb_base + (lane * 2u + rr) * 8u;
                    int c0 = gs_dct8[rb + 0u], c1 = gs_dct8[rb + 1u], c2 = gs_dct8[rb + 2u], c3 = gs_dct8[rb + 3u];
                    int c4 = gs_dct8[rb + 4u], c5 = gs_dct8[rb + 5u], c6 = gs_dct8[rb + 6u], c7 = gs_dct8[rb + 7u];
                    if ((c0 | c1 | c2 | c3 | c4 | c5 | c6 | c7) != 0) {
                        Dct8Inv1D(c0, c1, c2, c3, c4, c5, c6, c7, 64, 7);
                        gs_dct8[rb + 0u] = c0; gs_dct8[rb + 1u] = c1; gs_dct8[rb + 2u] = c2; gs_dct8[rb + 3u] = c3;
                        gs_dct8[rb + 4u] = c4; gs_dct8[rb + 5u] = c5; gs_dct8[rb + 6u] = c6; gs_dct8[rb + 7u] = c7;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();

            if (nz_pl) {
                [unroll] for (uint cc = 0u; cc < 2u; cc++) {
                    uint cb = sb_base + lane * 2u + cc;
                    int c0 = gs_dct8[cb + 0u],  c1 = gs_dct8[cb + 8u],  c2 = gs_dct8[cb + 16u], c3 = gs_dct8[cb + 24u];
                    int c4 = gs_dct8[cb + 32u], c5 = gs_dct8[cb + 40u], c6 = gs_dct8[cb + 48u], c7 = gs_dct8[cb + 56u];
                    Dct8Inv1D(c0, c1, c2, c3, c4, c5, c6, c7, 2048, 12);
                    gs_dct8[cb + 0u]  = c0; gs_dct8[cb + 8u]  = c1; gs_dct8[cb + 16u] = c2; gs_dct8[cb + 24u] = c3;
                    gs_dct8[cb + 32u] = c4; gs_dct8[cb + 40u] = c5; gs_dct8[cb + 48u] = c6; gs_dct8[cb + 56u] = c7;
                }
            }
            GroupMemoryBarrierWithGroupSync();

            if (do_dct8) {
                uint sh = pl * 8u;
                uint clear_m = ~(0xFFu << sh);
                if (nz_pl) {
                    [unroll] for (uint j = 0u; j < 16u; j++) {
                        uint lx = qx0 + (j & 3u);
                        uint ly = qy0 + (j >> 2);
                        uint idx = (y0_sb + ly) * width + x0_sb + lx;
                        uint pw = (pl == 0u) ? (is_inter8 ? g_pred_yuvd[idx] : 0x00808080u) : g_cur_yuvd[idx];
                        int pv = (int)((pw >> sh) & 0xFFu);
                        uint nv = (uint)clamp(pv + gs_dct8[sb_base + ly * 8u + lx], 0, 255);
                        uint out_w = (pw & clear_m) | (nv << sh);
                        if (pl == 0u) out_w = (out_w & 0x00FFFFFFu) | (delta_min << 24);
                        g_cur_yuvd[idx] = out_w;
                    }
                } else if (pl == 0u) {
                    [unroll] for (uint j = 0u; j < 16u; j++) {
                        uint lx = qx0 + (j & 3u);
                        uint ly = qy0 + (j >> 2);
                        uint idx = (y0_sb + ly) * width + x0_sb + lx;
                        uint pw = is_inter8 ? g_pred_yuvd[idx] : 0x00808080u;
                        g_cur_yuvd[idx] = (pw & 0x00FFFFFFu) | (delta_min << 24);
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    if (gtid.x == 0u && gs_hist[3] != 0u) {
        uint dummy;
        InterlockedMax(g_tq_scratch[TqMisc() + 3u], gs_hist[3], dummy);
    }
    if (dtid.x < nb) {
        bool covered = ok || (in_sb && gs_it_ok8[sbid] != 0u);
        g_tflag_out[b] = covered ? 1u : 0u;
    }
}

// Sets counter[0] = total accepted transform words, updates TqMisc()+2 for
// CSEncode, stores total transform demand in counter[4] for separated RC, and
// records exact tq_accepted in counter[12] so the host never reads stale VRAM
// if CSEncode overflows max_record_words after CSIntraT.
[numthreads(1, 1, 1)]
void CSTqFinal() {
    uint m = TqMisc();
    uint total = max(g_tq_scratch[m + 2u], g_tq_scratch[m + 3u]);
    g_tq_scratch[m + 2u] = total;
    g_out_counter.Store(0, total);
    g_out_counter.Store(4, g_tq_scratch[m + 0u]);
    g_out_counter.Store(12, total);
}
