// ternary_encode.hlsl - D3D11 Compute Shader (cs_5_0)
// Full-Color (YCbCr 4:4:4 / 4:2:0-ready) + 32x32 Tile MV + Jayant ADM +
// Rolling Intra-Refresh & Disocclusion Repair, Zero-Atomic-Contention Ref.
//
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
};

Texture2D<unorm float4>   g_src_rgba    : register(t0);
StructuredBuffer<uint>    g_prev_yuvd   : register(t1); // Ping-pong previous frame ref [Y|Cb|Cr|delta]
RWStructuredBuffer<uint>  g_cur_yuvd    : register(u0); // Ping-pong current frame ref
RWStructuredBuffer<int2>  g_tile_mvs    : register(u1); // (dx, dy) per 32x32 tile
RWStructuredBuffer<uint>  g_out_records : register(u2); // Compact output records
RWByteAddressBuffer       g_out_counter : register(u3); // [0]=record_words_cnt

static const uint POW3[10] = { 1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683 };

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

// =========================================================================
// Pass 1: Per-Tile 32x32 Motion Estimation (1 threadgroup per tile)
// =========================================================================
groupshared uint gs_best_sad;
groupshared int2 gs_best_mv;

[numthreads(16, 1, 1)]
void CSTileMV(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tx = gid.x;
    uint ty = gid.y;
    if (tx >= tiles_x || ty >= tiles_y) return;

    if (gtid.x == 0) {
        gs_best_sad = 0xFFFFFFFFu;
        gs_best_mv  = int2(0, 0);
    }
    GroupMemoryBarrierWithGroupSync();

    uint x0 = tx * 32u;
    uint y0 = ty * 32u;

    // Each of 16 threads evaluates candidate offsets in [-search_r, +search_r] (step 2)
    int r = (int)search_r;
    int span = 2 * r + 1;
    int total_cands = span * span;

    for (int c = (int)gtid.x; c < total_cands; c += 16) {
        int dy = (c / span) - r;
        int dx = (c % span) - r;
        // Coarse step-2 filter except around (0,0)
        if ((dx != 0 || dy != 0) && ((dx & 1) != 0 || (dy & 1) != 0)) {
            continue;
        }

        uint sad = 0;
        uint cnt = 0;
        for (uint ly = 2; ly < 32; ly += 4) {
            uint py = y0 + ly;
            int sy = (int)py - dy;
            if (py >= height || sy < 0 || sy >= (int)height) continue;
            for (uint lx = 2; lx < 32; lx += 4) {
                uint px = x0 + lx;
                int sx = (int)px - dx;
                if (px >= width || sx < 0 || sx >= (int)width) continue;

                float4 src_c = g_src_rgba.Load(int3(px, py, 0));
                int cy, ccb, ccr;
                RgbToYCbCr(src_c.rgb, cy, ccb, ccr);
                uint ref_w = g_prev_yuvd[(uint)sy * width + (uint)sx];
                int ry = (int)(ref_w & 0xFFu);
                sad += (uint)abs(cy - ry);
                cnt++;
            }
        }
        if (cnt > 0) {
            uint norm_sad = (sad * 64u) / cnt;
            if (dx == 0 && dy == 0) {
                norm_sad = (norm_sad > 32u) ? (norm_sad - 32u) : 0u;
            }
            // SAD in the high bits, MV in the low bits: one atomic min picks both.
            uint key = (norm_sad << 10) | ((uint)(dy + 16) << 5) | (uint)(dx + 16);
            InterlockedMin(gs_best_sad, key);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (gtid.x == 0) {
        // No candidate had a valid sample (tile mostly outside the frame): keep (0,0).
        uint best = gs_best_sad;
        int bdx = (best == 0xFFFFFFFFu) ? 0 : (int)(best & 31u) - 16;
        int bdy = (best == 0xFFFFFFFFu) ? 0 : (int)((best >> 5) & 31u) - 16;
        g_tile_mvs[ty * tiles_x + tx] = int2(bdx, bdy);
    }
}

// =========================================================================
// Pass 2: Full-Color Ternary Encode + Jayant ADM + Disocclusion/Intra
// =========================================================================
struct PixState {
    bool valid;       // idx is inside the frame
    bool want_intra;  // rolling refresh row or disocclusion
    int cy, ccb, ccr; // source
    int ry, rcb, rcr; // motion-compensated reference
    int d;            // adaptive step stored at the reference
};

PixState LoadPix(uint idx, uint num_px, uint intra_y0, uint intra_y1, bool allow_disocc) {
    PixState s;
    s.valid = (idx < num_px);
    // No early return: the caller's loop must stay unrollable by fxc.
    idx = min(idx, num_px - 1u);
    uint py = idx / width;
    uint px = idx - py * width;
    int2 mv = g_tile_mvs[(py >> 5) * tiles_x + (px >> 5)];
    int sx = clamp((int)px - mv.x, 0, (int)width - 1);
    int sy = clamp((int)py - mv.y, 0, (int)height - 1);
    uint ref_w = g_prev_yuvd[(uint)sy * width + (uint)sx];

    s.ry  = (int)(ref_w & 0xFFu);
    s.rcb = (int)((ref_w >> 8) & 0xFFu);
    s.rcr = (int)((ref_w >> 16) & 0xFFu);
    s.d   = max((int)delta_min, (int)((ref_w >> 24) & 0xFFu));

    float4 src_c = g_src_rgba.Load(int3(px, py, 0));
    RgbToYCbCr(src_c.rgb, s.cy, s.ccb, s.ccr);

    bool rolling = (py >= intra_y0 && py < intra_y1);
    bool disocc  = allow_disocc && (abs(s.cy - s.ry) > (int)intra_err_thr);
    s.want_intra = s.valid && (rolling || disocc);
    return s;
}

int Quant(int err, int thr) {
    return (err > thr) ? 1 : ((err < -thr) ? -1 : 0);
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
    uint blk = dtid.x;
    uint is_2x = mode_flags & 1u;
    uint eff_blocks = (is_2x != 0u) ? ((total_blocks + 1u) >> 1) : total_blocks;
    if (blk >= eff_blocks) return;

    uint num_px = width * height;
    uint cycles = max(intra_cycles, 1u);
    uint rows_per_slot = (height + cycles - 1u) / cycles;
    uint intra_y0 = (frame_id % cycles) * rows_per_slot;
    uint intra_y1 = intra_y0 + rows_per_slot;

    // Best-effort rate control: the counter is read while other groups are
    // still appending, so the cap is soft. Both sides stay in sync because
    // the reference is updated from what is actually emitted.
    uint cur_words = g_out_counter.Load(0);
    bool allow_disocc = (max_record_words == 0u) || (cur_words < (max_record_words >> 1));
    bool allow_chroma = (max_record_words == 0u) || (cur_words < max_record_words);

    int y_thr = (int)threshold;
    int c_thr = (int)threshold + 2;

    uint packed_y  = 0;
    uint packed_cb = 0;
    uint packed_cr = 0;
    bool nz_y  = false;
    bool nz_cb = false;
    bool nz_cr = false;

    // 1x: 5 pairs, 2 trits each (10 px). 2x: 10 pairs, 1 trit each (20 px).
    uint base_px = blk * ((is_2x != 0u) ? 20u : 10u);
    uint num_pairs = (is_2x != 0u) ? 10u : 5u;

    [unroll]
    for (uint k = 0; k < 10u; k++) {
        // Pairs past num_pairs (1x mode) resolve to idx >= num_px below and
        // contribute nothing: packed words only use POW3[0..9].
        uint idx_a = (k < num_pairs) ? (base_px + k * 2u) : num_px;
        uint idx_b = idx_a + 1u;
        PixState a = LoadPix(idx_a, num_px, intra_y0, intra_y1, allow_disocc);
        PixState b = LoadPix(idx_b, num_px, intra_y0, intra_y1, allow_disocc);

        int ty_a = 0, tcb_a = 0, tcr_a = 0;
        int ty_b = 0, tcb_b = 0, tcr_b = 0;

        if (a.valid && (a.want_intra || b.want_intra)) {
            int cy_b = b.valid ? b.cy : a.cy;
            int cb = b.valid ? ((a.ccb + b.ccb + 1) >> 1) : a.ccb;
            int cr = b.valid ? ((a.ccr + b.ccr + 1) >> 1) : a.ccr;

            uint slot = 0;
            g_out_counter.InterlockedAdd(0, 2u, slot);
            g_out_records[slot + 0u] = 0xC0000000u | (idx_a & 0x1FFFFFFFu);
            g_out_records[slot + 1u] = (uint)a.cy | ((uint)cb << 8) | ((uint)cy_b << 16) | ((uint)cr << 24);

            uint chroma_d = ((uint)cb << 8) | ((uint)cr << 16) | (delta_min << 24);
            g_cur_yuvd[idx_a] = (uint)a.cy | chroma_d;
            if (b.valid) g_cur_yuvd[idx_b] = (uint)cy_b | chroma_d;
        } else if (a.valid) {
            if (is_2x != 0u) {
                // Mean error of the pair (encoder-only decision, any rounding is fine).
                int ey  = b.valid ? (((a.cy  - a.ry)  + (b.cy  - b.ry))  >> 1) : (a.cy  - a.ry);
                int ecb = b.valid ? (((a.ccb - a.rcb) + (b.ccb - b.rcb)) >> 1) : (a.ccb - a.rcb);
                int ecr = b.valid ? (((a.ccr - a.rcr) + (b.ccr - b.rcr)) >> 1) : (a.ccr - a.rcr);
                ty_a  = Quant(ey, y_thr);
                tcb_a = allow_chroma ? Quant(ecb, c_thr) : 0;
                tcr_a = allow_chroma ? Quant(ecr, c_thr) : 0;

                int up = min((int)delta_max, a.d << 1);
                StoreDelta(idx_a, a, ty_a, tcb_a, tcr_a, a.d,
                           (ty_a == 0) ? max((int)delta_min, a.d >> 1) : up);
                if (b.valid) {
                    StoreDelta(idx_b, b, ty_a, tcb_a, tcr_a, a.d,
                               (ty_a == 0) ? max((int)delta_min, b.d >> 1) : up);
                }
            } else {
                ty_a  = Quant(a.cy - a.ry, y_thr);
                tcb_a = allow_chroma ? Quant(a.ccb - a.rcb, c_thr) : 0;
                tcr_a = allow_chroma ? Quant(a.ccr - a.rcr, c_thr) : 0;
                StoreDelta(idx_a, a, ty_a, tcb_a, tcr_a, a.d,
                           (ty_a == 0) ? max((int)delta_min, a.d >> 1)
                                       : min((int)delta_max, a.d << 1));
                if (b.valid) {
                    ty_b  = Quant(b.cy - b.ry, y_thr);
                    tcb_b = allow_chroma ? Quant(b.ccb - b.rcb, c_thr) : 0;
                    tcr_b = allow_chroma ? Quant(b.ccr - b.rcr, c_thr) : 0;
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

    uint hdr_base = (is_2x << 29) | (blk & 0x1FFFFFFFu);
    if (nz_y) {
        uint slot = 0;
        g_out_counter.InterlockedAdd(0, 2u, slot);
        g_out_records[slot + 0u] = (0u << 30) | hdr_base;
        g_out_records[slot + 1u] = packed_y;
    }
    if (nz_cb) {
        uint slot = 0;
        g_out_counter.InterlockedAdd(0, 2u, slot);
        g_out_records[slot + 0u] = (1u << 30) | hdr_base;
        g_out_records[slot + 1u] = packed_cb;
    }
    if (nz_cr) {
        uint slot = 0;
        g_out_counter.InterlockedAdd(0, 2u, slot);
        g_out_records[slot + 0u] = (2u << 30) | hdr_base;
        g_out_records[slot + 1u] = packed_cr;
    }
}
