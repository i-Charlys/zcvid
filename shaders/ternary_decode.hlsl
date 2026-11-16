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
};

StructuredBuffer<int2>    g_tile_mvs   : register(t0);
StructuredBuffer<uint>    g_in_records : register(t1);
StructuredBuffer<uint>    g_prev_yuvd  : register(t2);
RWStructuredBuffer<uint>  g_cur_yuvd   : register(u0);
RWTexture2D<unorm float4> g_out_rgba   : register(u1);

int LeakyDecay(int prev_val, int neutral) {
    int diff = prev_val - neutral;
    return clamp(neutral + (diff * (int)gamma_q8) / 256, 0, 255);
}

// Motion-compensated reference word of a pixel (previous frame, tile MV applied).
uint PrevShifted(uint idx) {
    uint py = idx / width;
    uint px = idx - py * width;
    int2 mv = g_tile_mvs[(py >> 5) * tiles_x + (px >> 5)];
    int sx = clamp((int)px - mv.x, 0, (int)width - 1);
    int sy = clamp((int)py - mv.y, 0, (int)height - 1);
    return g_prev_yuvd[(uint)sy * width + (uint)sx];
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
void ApplyTrit(uint idx, int t, int d_used, uint tag) {
    // No early return: the caller's loop must stay unrollable by fxc.
    if (t != 0 && idx < width * height) {
        uint shift = tag * 8u;
        int neutral = (tag == 0u) ? (int)neutral_y : 128;
        int prev = (int)((PrevShifted(idx) >> shift) & 0xFFu);
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
        if (is_2x != 0u) {
            // 4-pixel 6-bit Intra: w0 = [tag=3|is_4px=1|cr_q6(6)|idx(22)], w1 = [y0..y3(24)|cb_q6(6)]
            uint idx4 = w0 & 0x3FFFFFu;
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

    [unroll]
    for (uint h = 0; h < 2u; h++) {
        uint v = halves[h];
        uint prev_q = v;
        [unroll]
        for (uint k = 0; k < 5u; k++) {
            uint q = (v * RECIP[k]) >> 16;
            int t = (int)(prev_q - q * 3u) - 1;
            prev_q = q;

            uint j = h * 5u + k;
            uint idx = base_px + j * step_px;
            if (t != 0 && idx < num_px) {
                int d = StepOf(PrevShifted(idx));
                ApplyTrit(idx, t, d, tag);
                if (is_2x != 0u) ApplyTrit(idx + 1u, t, d, tag);
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

    uint w = g_cur_yuvd[dtid.y * width + dtid.x];
    float y  = (float)(w & 0xFFu);
    float cb = (float)((w >> 8) & 0xFFu) - 128.0f;
    float cr = (float)((w >> 16) & 0xFFu) - 128.0f;

    float r = clamp((y + 1.402f * cr) / 255.0f, 0.0f, 1.0f);
    float g = clamp((y - 0.344136f * cb - 0.714136f * cr) / 255.0f, 0.0f, 1.0f);
    float b = clamp((y + 1.772f * cb) / 255.0f, 0.0f, 1.0f);

    g_out_rgba[dtid.xy] = float4(r, g, b, 1.0f);
}
