// GPU MSAA 2x/4x/8x resolve compute shader over 64-bit BDA surfaces.
// Replaces CPU MSAA resolve in strata-gpu/src/readback.rs.
// format_mode: 0 = RGBA8_UNORM average, 1 = RGBA32F average, 2 = R32F depth min, 3 = R32F depth max

struct MsaaResolvePush {
    uint64_t src_addr;
    uint64_t dst_addr;
    uint32_t width;
    uint32_t height;
    uint32_t sample_count;
    uint32_t format_mode;
};

[[vk::push_constant]]
ConstantBuffer<MsaaResolvePush> pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= pc.width || dtid.y >= pc.height) {
        return;
    }
    uint32_t samples = max(pc.sample_count, 1u);
    uint32_t pixel_idx = dtid.y * pc.width + dtid.x;
    uint32_t base_sample = pixel_idx * samples;

    if (pc.format_mode == 0u) {
        uint32_t *src = (uint32_t *)pc.src_addr;
        uint32_t *dst = (uint32_t *)pc.dst_addr;
        uint4 acc = uint4(0u, 0u, 0u, 0u);
        for (uint32_t s = 0u; s < samples; ++s) {
            uint32_t v = src[base_sample + s];
            acc.x += (v >> 0u) & 0xFFu;
            acc.y += (v >> 8u) & 0xFFu;
            acc.z += (v >> 16u) & 0xFFu;
            acc.w += (v >> 24u) & 0xFFu;
        }
        acc /= samples;
        dst[pixel_idx] = (acc.x & 0xFFu)
                       | ((acc.y & 0xFFu) << 8u)
                       | ((acc.z & 0xFFu) << 16u)
                       | ((acc.w & 0xFFu) << 24u);
    } else if (pc.format_mode == 1u) {
        float4 *src = (float4 *)pc.src_addr;
        float4 *dst = (float4 *)pc.dst_addr;
        float4 acc = float4(0.0, 0.0, 0.0, 0.0);
        for (uint32_t s = 0u; s < samples; ++s) {
            acc += src[base_sample + s];
        }
        dst[pixel_idx] = acc / float(samples);
    } else {
        float *src = (float *)pc.src_addr;
        float *dst = (float *)pc.dst_addr;
        float best = src[base_sample];
        for (uint32_t s = 1u; s < samples; ++s) {
            float v = src[base_sample + s];
            best = (pc.format_mode == 2u) ? min(best, v) : max(best, v);
        }
        dst[pixel_idx] = best;
    }
}
