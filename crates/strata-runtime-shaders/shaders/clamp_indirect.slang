// Batched indirect draw argument clamping via 64-bit BDA pointers.
// Inspired by VIctory-Belobug-AMD's hoisted indirect-clamp pass (225 ms -> 80 ms):
// clamps thousands of console indirect draw argument buffers in a single GPU dispatch
// instead of inserting a separate clamp dispatch before every draw.

struct ClampEntry {
    uint64_t args_addr;   // BDA pointer to VkDrawIndirectCommand / mesh dispatch args
    uint32_t max_count;   // Safe upper bound for vertex/group count
    uint32_t max_inst;    // Safe upper bound for instance count
};

struct ClampPush {
    uint64_t entries_addr;
    uint32_t entry_count;
};

[[vk::push_constant]]
ConstantBuffer<ClampPush> pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint32_t idx = dtid.x;
    if (idx >= pc.entry_count) {
        return;
    }

    ClampEntry e = ((ClampEntry *)pc.entries_addr)[idx];
    uint32_t *args = (uint32_t *)e.args_addr;
    args[0] = min(args[0], e.max_count);
    args[1] = min(args[1], e.max_inst);
}
