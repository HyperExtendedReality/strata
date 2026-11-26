// GPU-side 64 KiB dirty-page hashing via 64-bit BDA pointers (No-API).
// One workgroup (64 lanes) hashes one 64 KiB page using WaveActiveBitXor
// to avoid CPU-side write-watch page-table walks on VRAM-resident buffers.

#include "common.slang"

[[vk::push_constant]]
ConstantBuffer<DirtyHashPush> pc;

groupshared uint32_t s_partial[2];

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    uint32_t block_idx = gid.x;
    if (block_idx >= pc.block_count) {
        return;
    }

    uint32_t *src = (uint32_t *)(pc.src_addr + uint64_t(block_idx) * uint64_t(pc.dwords_per_block) * 4ull);
    uint32_t h = 0x811C9DC5u ^ tid;

    for (uint32_t i = tid; i < pc.dwords_per_block; i += 64u) {
        h = mix32(h, src[i]);
    }

    uint32_t wave_h = WaveActiveBitXor(h);
    uint32_t lane_count = WaveGetLaneCount();
    uint32_t wave_id = tid / lane_count;

    if (WaveIsFirstLane()) {
        s_partial[wave_id] = wave_h;
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0u) {
        uint32_t final_h = s_partial[0];
        if (lane_count < 64u) {
            final_h ^= s_partial[1];
        }
        uint32_t *dst = (uint32_t *)pc.dst_hash_addr;
        dst[block_idx] = final_h;
    }
}
