// 64-byte cache-line chunk dirty hashing & UAV write-detection sweep over 64-bit BDA.
// Each thread hashes one 64-byte chunk (16 dwords) and compares against prev_hash_addr,
// writing 1 to dirty_flags_addr[chunk] when changed and updating prev_hash_addr[chunk].

#include "common.slang"

struct ChunkDirtyHashPush {
    uint64_t src_addr;
    uint64_t prev_hash_addr;
    uint64_t dirty_flags_addr;
    uint32_t chunk_count;
};

[[vk::push_constant]]
ConstantBuffer<ChunkDirtyHashPush> pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint32_t chunk = dtid.x;
    if (chunk >= pc.chunk_count) {
        return;
    }
    uint32_t *src = (uint32_t *)(pc.src_addr + uint64_t(chunk) * 64ull);
    uint32_t h = 0x811C9DC5u ^ chunk;
    [unroll]
    for (uint32_t i = 0u; i < 16u; ++i) {
        h = mix32(h, src[i]);
    }
    uint32_t *prev = (uint32_t *)pc.prev_hash_addr;
    uint32_t *flags = (uint32_t *)pc.dirty_flags_addr;
    uint32_t old_h = prev[chunk];
    flags[chunk] = (old_h != h) ? 1u : 0u;
    prev[chunk] = h;
}
