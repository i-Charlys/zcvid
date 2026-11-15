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

// =========================================================================
// Pass 1: Motion-Compensated Copy + Leaky Decay (1 thread per pixel)
// =========================================================================
[numthreads(16, 16, 1)]
void CSMotionDecay(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= width || dtid.y >= height) return;

    uint tx = dtid.x >> 5;
    uint ty = dtid.y >> 5;
    int2 mv = g_tile_mvs[ty * tiles_x + tx];
    int sx = clamp((int)dtid.x - mv.x, 0, (int)width - 1);
    int sy = clamp((int)dtid.y - mv.y, 0, (int)height - 1);

    uint w = g_prev_yuvd[(uint)sy * width + (uint)sx];
    int y  = LeakyDecay((int)(w & 0xFFu), (int)neutral_y);
    int cb = LeakyDecay((int)((w >> 8) & 0xFFu), 128);
    int cr = LeakyDecay((int)((w >> 16) & 0xFFu), 128);
    uint d = (w >> 24) & 0xFFu;
    if (d < delta_min) d = delta_min;

    g_cur_yuvd[dtid.y * width + dtid.x] = (uint)y | ((uint)cb << 8) | ((uint)cr << 16) | (d << 24);
}

// =========================================================================
// Pass 2: Apply Delta (Y/Cb/Cr) & Intra Records
// =========================================================================
[numthreads(64, 1, 1)]
void CSApply(uint3 dtid : SV_DispatchThreadID) {
    uint rec_idx = dtid.x * 2u;
    if (rec_idx + 1u >= record_words) return;

    uint w0 = g_in_records[rec_idx + 0u];
    uint w1 = g_in_records[rec_idx + 1u];
    uint tag = (w0 >> 30) & 3u;
    uint id  = w0 & 0x3FFFFFFFu;

    if (tag == 3u) {
        // Intra pixel anchor
        if (id < width * height) {
            g_cur_yuvd[id] = (w1 & 0x00FFFFFFu) | (delta_min << 24);
        }
        return;
    }

    // Delta 10-trit block on plane = tag (0=Y, 1=Cb, 2=Cr)
    uint base_px = id * 10u;
    uint lo = w1 % 243u;
    uint hi = w1 / 243u;
    uint shift = tag * 8u;
    uint mask = ~(0xFFu << shift);

    [unroll]
    for (uint j = 0; j < 5u; j++) {
        int t0 = (int)(lo % 3u) - 1; lo /= 3u;
        int t1 = (int)(hi % 3u) - 1; hi /= 3u;

        uint idx0 = base_px + j;
        if (idx0 < width * height) {
            uint cur_w = g_cur_yuvd[idx0];
            int d0 = (int)((cur_w >> 24) & 0xFFu);
            if (t0 != 0) {
                int v0 = clamp((int)((cur_w >> shift) & 0xFFu) + t0 * d0, 0, 255);
                InterlockedAnd(g_cur_yuvd[idx0], mask);
                InterlockedOr(g_cur_yuvd[idx0], (uint)v0 << shift);
            }
            if (tag == 0u) {
                int nd0 = (t0 == 0) ? max((int)delta_min, d0 >> 1) : min((int)delta_max, d0 << 1);
                InterlockedAnd(g_cur_yuvd[idx0], 0x00FFFFFFu);
                InterlockedOr(g_cur_yuvd[idx0], (uint)nd0 << 24);
            }
        }

        uint idx1 = base_px + 5u + j;
        if (idx1 < width * height) {
            uint cur_w = g_cur_yuvd[idx1];
            int d1 = (int)((cur_w >> 24) & 0xFFu);
            if (t1 != 0) {
                int v1 = clamp((int)((cur_w >> shift) & 0xFFu) + t1 * d1, 0, 255);
                InterlockedAnd(g_cur_yuvd[idx1], mask);
                InterlockedOr(g_cur_yuvd[idx1], (uint)v1 << shift);
            }
            if (tag == 0u) {
                int nd1 = (t1 == 0) ? max((int)delta_min, d1 >> 1) : min((int)delta_max, d1 << 1);
                InterlockedAnd(g_cur_yuvd[idx1], 0x00FFFFFFu);
                InterlockedOr(g_cur_yuvd[idx1], (uint)nd1 << 24);
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
