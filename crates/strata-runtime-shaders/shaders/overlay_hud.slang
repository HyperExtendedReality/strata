// GPU-rendered performance HUD overlay (backdrop + 120-frame rolling frametime sparkline
// + RAM/VRAM usage bars) composited directly onto an RGBA8_UNORM BDA framebuffer.

struct OverlayHudPush {
    uint64_t fb_addr;
    uint64_t frametimes_addr; // float[120] in milliseconds
    uint32_t fb_width;
    uint32_t fb_height;
    uint32_t box_x;
    uint32_t box_y;
    uint32_t box_w;
    uint32_t box_h;
    float vram_ratio;         // 0.0..1.0
    float ram_ratio;          // 0.0..1.0
    float backdrop_alpha;     // 0.0..1.0
    uint32_t sample_count;    // up to 120
};

[[vk::push_constant]]
ConstantBuffer<OverlayHudPush> pc;

uint32_t packRgba8(float4 c) {
    uint4 u = uint4(clamp(c, 0.0, 1.0) * 255.0 + 0.5);
    return (u.x & 0xFFu) | ((u.y & 0xFFu) << 8u) | ((u.z & 0xFFu) << 16u) | ((u.w & 0xFFu) << 24u);
}

float4 unpackRgba8(uint32_t p) {
    return float4(
        float((p >> 0u) & 0xFFu),
        float((p >> 8u) & 0xFFu),
        float((p >> 16u) & 0xFFu),
        float((p >> 24u) & 0xFFu)
    ) / 255.0;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint32_t lx = dtid.x;
    uint32_t ly = dtid.y;
    if (lx >= pc.box_w || ly >= pc.box_h) {
        return;
    }
    uint32_t px = pc.box_x + lx;
    uint32_t py = pc.box_y + ly;
    if (px >= pc.fb_width || py >= pc.fb_height) {
        return;
    }

    uint32_t *fb = (uint32_t *)pc.fb_addr;
    uint32_t idx = py * pc.fb_width + px;
    float4 dst = unpackRgba8(fb[idx]);

    // Dark translucent backdrop with 1px subtle border
    bool border = (lx == 0u || ly == 0u || lx + 1u == pc.box_w || ly + 1u == pc.box_h);
    float4 hud = border ? float4(0.35, 0.45, 0.65, 0.9) : float4(0.05, 0.06, 0.09, pc.backdrop_alpha);

    // Top 8px bar: VRAM utilization; next 8px bar: RAM utilization
    if (ly >= 8u && ly < 14u && lx >= 8u && lx + 8u < pc.box_w) {
        float frac_x = float(lx - 8u) / float(max(pc.box_w - 16u, 1u));
        hud = (frac_x <= pc.vram_ratio) ? float4(0.20, 0.85, 0.60, 0.95) : float4(0.15, 0.17, 0.22, 0.85);
    } else if (ly >= 18u && ly < 24u && lx >= 8u && lx + 8u < pc.box_w) {
        float frac_x = float(lx - 8u) / float(max(pc.box_w - 16u, 1u));
        hud = (frac_x <= pc.ram_ratio) ? float4(0.25, 0.60, 0.95, 0.95) : float4(0.15, 0.17, 0.22, 0.85);
    } else if (pc.sample_count > 0u && ly >= 30u && ly + 6u < pc.box_h && lx >= 8u && lx + 8u < pc.box_w) {
        // Rolling frametime sparkline (0..33.3 ms scale)
        uint32_t graph_w = max(pc.box_w - 16u, 1u);
        uint32_t graph_h = max(pc.box_h - 36u, 1u);
        uint32_t s_idx = min(((lx - 8u) * pc.sample_count) / graph_w, pc.sample_count - 1u);
        float *ft = (float *)pc.frametimes_addr;
        float ms = clamp(ft[s_idx], 0.0, 33.33);
        float norm_y = 1.0 - (float(ly - 30u) / float(graph_h));
        if (norm_y <= (ms / 33.33)) {
            hud = (ms > 16.7) ? float4(0.95, 0.35, 0.25, 0.9) : float4(0.25, 0.90, 0.45, 0.9);
        }
    }

    float3 blended = lerp(dst.rgb, hud.rgb, hud.a);
    fb[idx] = packRgba8(float4(blended, 1.0));
}
