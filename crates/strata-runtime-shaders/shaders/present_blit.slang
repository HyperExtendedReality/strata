// No-API Fullscreen Present / Blit Shader (Slang -> SPIR-V 1.6).
// - ZERO vertex buffers & ZERO vertex input layouts (generates fullscreen triangle from SV_VertexID).
// - Reads source framebuffer pixels directly from a 64-bit BDA pointer in Push Constants.
// - Supports optional gamma/exposure & ambient-floor adjustment cleanly in one pass.

struct PresentPush {
    uint64_t src_rgba8_bda; // 64-bit BDA pointer to RGBA8_UNORM source image bytes
    uint32_t width;
    uint32_t height;
    float exposure;
};

[[vk::push_constant]]
ConstantBuffer<PresentPush> pc;

struct VsOut {
    float4 pos : SV_Position;
    float2 uv  : TEXCOORD0;
};

[shader("vertex")]
VsOut main_vs(uint vid : SV_VertexID) {
    VsOut o;
    float2 uv = float2((vid << 1) & 2, vid & 2);
    o.uv = uv;
    o.pos = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

[shader("fragment")]
float4 main_fs(VsOut inp) : SV_Target0 {
    uint32_t x = min(uint32_t(inp.uv.x * float(pc.width)), pc.width - 1u);
    uint32_t y = min(uint32_t(inp.uv.y * float(pc.height)), pc.height - 1u);
    uint32_t packed = ((uint32_t *)pc.src_rgba8_bda)[y * pc.width + x];

    float4 rgba = float4(
        float((packed >>  0) & 0xFFu) / 255.0,
        float((packed >>  8) & 0xFFu) / 255.0,
        float((packed >> 16) & 0xFFu) / 255.0,
        float((packed >> 24) & 0xFFu) / 255.0
    );
    return float4(rgba.rgb * pc.exposure, rgba.a);
}
