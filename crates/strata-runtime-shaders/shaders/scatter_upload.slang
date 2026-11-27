// GPU scatter upload & in-place constant/buffer fill over 64-bit BDA.
// mode == 0: scatter (dword_offset, value) pairs from entries_addr into dst_addr
// mode == 1: fill count dwords at dst_addr with fill_value

struct ScatterEntry {
    uint32_t dword_offset;
    uint32_t value;
};

struct ScatterUploadPush {
    uint64_t dst_addr;
    uint64_t entries_addr;
    uint32_t count;
    uint32_t mode;
    uint32_t fill_value;
    uint32_t _pad;
};

[[vk::push_constant]]
ConstantBuffer<ScatterUploadPush> pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint32_t idx = dtid.x;
    if (idx >= pc.count) {
        return;
    }
    uint32_t *dst = (uint32_t *)pc.dst_addr;
    if (pc.mode == 0u) {
        ScatterEntry *entries = (ScatterEntry *)pc.entries_addr;
        ScatterEntry e = entries[idx];
        dst[e.dword_offset] = e.value;
    } else {
        dst[idx] = pc.fill_value;
    }
}
