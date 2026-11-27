// Clean-room GPU compute BCPack block unpacker over 64-bit BDA.
// Stage: reconstructs 8-byte (BC1/BC4) or 16-byte (BC2/BC3/BC5/BC6H/BC7) blocks
// from decoded symbol streams (endpoints + indices) in parallel across 64 lanes.

struct BcPackDecodePush {
    uint64_t endpoints_addr; // Decoded endpoint stream (BDA)
    uint64_t indices_addr;   // Decoded index stream (BDA)
    uint64_t dst_blocks_addr;// Output BC block buffer (BDA)
    uint32_t block_count;
    uint32_t block_bytes;    // 8 (BC1/BC4) or 16 (BC2/BC3/BC5/BC6H/BC7)
    uint32_t format_id;      // 1=BC1, 2=BC2, 3=BC3, 4=BC4, 5=BC5, 6=BC6H, 7=BC7
    uint32_t _pad;
};

[[vk::push_constant]]
ConstantBuffer<BcPackDecodePush> pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint32_t b = dtid.x;
    if (b >= pc.block_count) {
        return;
    }
    if (pc.block_bytes == 8u) {
        uint32_t *ep = (uint32_t *)pc.endpoints_addr;
        uint32_t *ix = (uint32_t *)pc.indices_addr;
        uint2 *dst = (uint2 *)pc.dst_blocks_addr;
        dst[b] = uint2(ep[b], ix[b]);
    } else {
        uint2 *ep = (uint2 *)pc.endpoints_addr;
        uint2 *ix = (uint2 *)pc.indices_addr;
        uint4 *dst = (uint4 *)pc.dst_blocks_addr;
        uint2 e = ep[b];
        uint2 i = ix[b];
        dst[b] = uint4(e.x, e.y, i.x, i.y);
    }
}
