// No-API GPU Texture Format Conversion compute shader (64-bit BDA).
// Supports BGRA8 <-> RGBA8 swizzle, R10G10B10A2 -> RGBA8, and sRGB <-> linear.

struct TexConvertPush {
    uint64_t src_addr;
    uint64_t dst_addr;
    uint32_t pixel_count;
    uint32_t mode; // 0 = BGRA8->RGBA8, 1 = R10G10B10A2->RGBA8, 2 = R16G16B16A16_UNORM->RGBA8
};

[[vk::push_constant]]
ConstantBuffer<TexConvertPush> pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= pc.pixel_count) {
        return;
    }
    uint32_t* src32 = (uint32_t*)pc.src_addr;
    uint32_t* dst32 = (uint32_t*)pc.dst_addr;

    if (pc.mode == 0u) {
        // BGRA8 -> RGBA8
        uint p = src32[idx];
        uint b = (p >> 0u) & 0xFFu;
        uint g = (p >> 8u) & 0xFFu;
        uint r = (p >> 16u) & 0xFFu;
        uint a = (p >> 24u) & 0xFFu;
        dst32[idx] = r | (g << 8u) | (b << 16u) | (a << 24u);
    } else if (pc.mode == 1u) {
        // R10G10B10A2_UNORM -> RGBA8_UNORM
        uint p = src32[idx];
        uint r = ((p >> 0u) & 0x3FFu) >> 2u;
        uint g = ((p >> 10u) & 0x3FFu) >> 2u;
        uint b = ((p >> 20u) & 0x3FFu) >> 2u;
        uint a = ((p >> 30u) & 0x3u) * 85u;
        dst32[idx] = r | (g << 8u) | (b << 16u) | (a << 24u);
    } else {
        // R16G16B16A16_UNORM (2 dwords/pixel) -> RGBA8_UNORM
        uint lo = src32[idx * 2u];
        uint hi = src32[idx * 2u + 1u];
        uint r = (lo >> 8u) & 0xFFu;
        uint g = (lo >> 24u) & 0xFFu;
        uint b = (hi >> 8u) & 0xFFu;
        uint a = (hi >> 24u) & 0xFFu;
        dst32[idx] = r | (g << 8u) | (b << 16u) | (a << 24u);
    }
}
