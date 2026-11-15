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

// Record tags in upper 2 bits of 32-bit record header word:
// Delta record (2 words): word0 = (plane<<30) | (blk & 0x3FFFFFFF), word1 = packed10_u16
// Intra record (3 words): word0 = (3<<30) | (idx & 0x3FFFFFFF), word1 = YCbCr0, ... (or per-pixel)
// For maximum compactness in GPU->CPU->GPU, we emit:
//   Delta: (plane << 30) | (blk << 16) | packed10   [1 uint32 = 4 bytes for 10 pixels!]
//   Intra: (3 << 30) | (idx_div_2 << 16) ... or 2 words:
//          word0 = 0xC0000000 | (idx & 0x3FFFFFFF), word1 = Y | (Cb<<8) | (Cr<<16)

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
            // Pack (norm_sad << 12) for atomic min? Better: InterlockedMin on sad, then store mv
            uint key = (norm_sad << 10) | ((uint)(dy + 16) << 5) | (uint)(dx + 16);
            InterlockedMin(gs_best_sad, key);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (gtid.x == 0) {
        uint best = gs_best_sad;
        int bdx = (int)(best & 31u) - 16;
        int bdy = (int)((best >> 5) & 31u) - 16;
        g_tile_mvs[ty * tiles_x + tx] = int2(bdx, bdy);
    }
}

// =========================================================================
// Pass 2: Full-Color Ternary Encode + Jayant ADM + Disocclusion/Intra
// =========================================================================
[numthreads(64, 1, 1)]
void CSEncode(uint3 dtid : SV_DispatchThreadID) {
    uint blk = dtid.x;
    uint is_2x = mode_flags & 1u;
    uint eff_blocks = (is_2x != 0u) ? ((total_blocks + 1u) >> 1) : total_blocks;
    if (blk >= eff_blocks) return;

    uint step_px = (is_2x != 0u) ? 2u : 1u;
    uint base_px = blk * 10u * step_px;
    uint packed_y  = 0;
    uint packed_cb = 0;
    uint packed_cr = 0;
    bool nz_y  = false;
    bool nz_cb = false;
    bool nz_cr = false;

    uint intra_slot = frame_id % max(intra_cycles, 1u);
    uint rows_per_slot = (height + max(intra_cycles, 1u) - 1u) / max(intra_cycles, 1u);
    uint intra_y0 = intra_slot * rows_per_slot;
    uint intra_y1 = intra_y0 + rows_per_slot;

    uint cur_words = g_out_counter.Load(0);
    bool allow_disocc = (max_record_words == 0u) || (cur_words < (max_record_words >> 1));
    bool allow_chroma = (max_record_words == 0u) || (cur_words < max_record_words);

    [unroll]
    for (uint j = 0; j < 10u; j++) {
        uint idx = base_px + j * step_px;
        if (idx >= width * height) {
            packed_y  += 1u * POW3[j];
            packed_cb += 1u * POW3[j];
            packed_cr += 1u * POW3[j];
            continue;
        }
        uint py = idx / width;
        uint px = idx - py * width;

        uint tx = px >> 5;
        uint ty = py >> 5;
        int2 mv = g_tile_mvs[ty * tiles_x + tx];
        int sx = clamp((int)px - mv.x, 0, (int)width - 1);
        int sy = clamp((int)py - mv.y, 0, (int)height - 1);
        uint ref_w = g_prev_yuvd[(uint)sy * width + (uint)sx];

        int ry  = (int)(ref_w & 0xFFu);
        int rcb = (int)((ref_w >> 8) & 0xFFu);
        int rcr = (int)((ref_w >> 16) & 0xFFu);
        int d   = (int)((ref_w >> 24) & 0xFFu);
        if (d < (int)delta_min) d = (int)delta_min;

        float4 src_c = g_src_rgba.Load(int3(px, py, 0));
        int cy, ccb, ccr;
        RgbToYCbCr(src_c.rgb, cy, ccb, ccr);

        bool in_rolling_intra = (py >= intra_y0 && py < intra_y1);
        bool in_disocc = allow_disocc && (abs(cy - ry) > (int)intra_err_thr);

        if (in_rolling_intra || in_disocc) {
            // Emit 2-pixel 4:2:2 Intra pair (4 bytes/px instead of 8 bytes/px)
            if ((idx & 1u) == 0u || is_2x != 0u) {
                uint idx_next = min(idx + 1u, width * height - 1u);
                uint py1 = idx_next / width;
                uint px1 = idx_next - py1 * width;
                float4 src_c1 = g_src_rgba.Load(int3(px1, py1, 0));
                int cy1, ccb1, ccr1;
                RgbToYCbCr(src_c1.rgb, cy1, ccb1, ccr1);

                uint slot = 0;
                g_out_counter.InterlockedAdd(0, 2u, slot);
                g_out_records[slot + 0u] = 0xC0000000u | (idx & 0x1FFFFFFFu);
                g_out_records[slot + 1u] = (uint)cy | ((uint)ccb << 8) | ((uint)cy1 << 16) | ((uint)ccr << 24);
                g_cur_yuvd[idx_next] = (uint)cy1 | ((uint)ccb << 8) | ((uint)ccr << 16) | (delta_min << 24);
            }
            g_cur_yuvd[idx] = (uint)cy | ((uint)ccb << 8) | ((uint)ccr << 16) | (delta_min << 24);
            packed_y  += 1u * POW3[j];
            packed_cb += 1u * POW3[j];
            packed_cr += 1u * POW3[j];
        } else {
            int ey  = cy - ry;
            int ecb = ccb - rcb;
            int ecr = ccr - rcr;

            int ty_val  = (ey  > (int)threshold) ? 1 : ((ey  < -(int)threshold) ? -1 : 0);
            int c_thr   = (int)threshold + 2;
            int tcb_val = allow_chroma ? ((ecb > c_thr) ? 1 : ((ecb < -c_thr) ? -1 : 0)) : 0;
            int tcr_val = allow_chroma ? ((ecr > c_thr) ? 1 : ((ecr < -c_thr) ? -1 : 0)) : 0;

            if (ty_val  != 0) nz_y  = true;
            if (tcb_val != 0) nz_cb = true;
            if (tcr_val != 0) nz_cr = true;

            int ny  = LeakyStep(ry,  ty_val,  d, (int)neutral_y);
            int ncb = LeakyStep(rcb, tcb_val, d, 128);
            int ncr = LeakyStep(rcr, tcr_val, d, 128);

            int next_d = (ty_val == 0) ? max((int)delta_min, d >> 1)
                                       : min((int)delta_max, d << 1);

            uint yuvd_val = (uint)ny | ((uint)ncb << 8) | ((uint)ncr << 16) | ((uint)next_d << 24);
            g_cur_yuvd[idx] = yuvd_val;
            if (is_2x != 0u && idx + 1u < width * height) {
                g_cur_yuvd[idx + 1u] = yuvd_val;
            }

            packed_y  += (uint)(ty_val  + 1) * POW3[j];
            packed_cb += (uint)(tcb_val + 1) * POW3[j];
            packed_cr += (uint)(tcr_val + 1) * POW3[j];
        }
    }

    uint hdr_base = (is_2x << 29) | (blk & 0x1FFFFFFFu);
    if (nz_y) {
        uint slot = 0;
        g_out_counter.InterlockedAdd(0, 2u, slot);
        g_out_records[slot + 0u] = (0u << 30) | hdr_base;
        g_out_records[slot + 1u] = packed_y;
    }
    if (nz_cb && allow_chroma) {
        uint slot = 0;
        g_out_counter.InterlockedAdd(0, 2u, slot);
        g_out_records[slot + 0u] = (1u << 30) | hdr_base;
        g_out_records[slot + 1u] = packed_cb;
    }
    if (nz_cr && allow_chroma) {
        uint slot = 0;
        g_out_counter.InterlockedAdd(0, 2u, slot);
        g_out_records[slot + 0u] = (2u << 30) | hdr_base;
        g_out_records[slot + 1u] = packed_cr;
    }
}
