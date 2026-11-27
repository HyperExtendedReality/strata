// No-API 2x2 Box-Filter Mipmap Generator (RGBA8_UNORM over 64-bit BDA).

struct MipGenPush {
    uint64_t src_addr;
    uint64_t dst_addr;
    uint32_t src_width;
    uint32_t src_height;
    uint32_t dst_width;
    uint32_t dst_height;
};

[[vk::push_constant]]
ConstantBuffer<MipGenPush> pc;

uint4 unpack_rgba8(uint p) {
    return uint4(p & 0xFFu, (p >> 8u) & 0xFFu, (p >> 16u) & 0xFFu, (p >> 24u) & 0xFFu);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.dst_width || tid.y >= pc.dst_height) {
        return;
    }
    uint32_t* src = (uint32_t*)pc.src_addr;
    uint32_t* dst = (uint32_t*)pc.dst_addr;

    uint sx0 = min(tid.x * 2u, pc.src_width - 1u);
    uint sx1 = min(tid.x * 2u + 1u, pc.src_width - 1u);
    uint sy0 = min(tid.y * 2u, pc.src_height - 1u);
    uint sy1 = min(tid.y * 2u + 1u, pc.src_height - 1u);

    uint4 c00 = unpack_rgba8(src[sy0 * pc.src_width + sx0]);
    uint4 c10 = unpack_rgba8(src[sy0 * pc.src_width + sx1]);
    uint4 c01 = unpack_rgba8(src[sy1 * pc.src_width + sx0]);
    uint4 c11 = unpack_rgba8(src[sy1 * pc.src_width + sx1]);

    uint4 avg = (c00 + c10 + c01 + c11 + uint4(2u)) >> 2u;
    dst[tid.y * pc.dst_width + tid.x] = avg.x | (avg.y << 8u) | (avg.z << 16u) | (avg.w << 24u);
}
