// Propagate changed bytes and preserve pending CPU writes during readback.
struct MirrorPush {
    uint64_t vram;
    uint64_t host;
    uint64_t baseline;
    uint64_t counter;
    uint wordCount;
    uint firstByte;
    uint byteCount;
    uint upload;
};

[[vk::push_constant]] ConstantBuffer<MirrorPush> pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 thread : SV_DispatchThreadID) {
    uint i = thread.x;
    uint bytes = 0;
    if (i < pc.wordCount) {
        uint* src = (uint*)(pc.upload != 0 ? pc.host : pc.vram);
        uint* dst = (uint*)(pc.upload != 0 ? pc.vram : pc.host);
        uint* baseline = (uint*)pc.baseline;
        uint old = baseline[i];
        uint incoming = src[i];
        if (incoming != old) {
            uint current = dst[i];
            uint mask = 0;
            for (uint b = 0; b < 4; ++b) {
                uint address = i * 4 + b;
                uint bits = 255u << (b * 8);
                if (address >= pc.firstByte && address - pc.firstByte < pc.byteCount &&
                    ((incoming ^ old) & bits) != 0 &&
                    (pc.upload != 0 || ((current ^ old) & bits) == 0)) {
                    mask |= bits;
                    ++bytes;
                }
            }
            if (mask != 0) {
                dst[i] = (current & ~mask) | (incoming & mask);
                baseline[i] = (old & ~mask) | (incoming & mask);
            }
        }
    }
    uint total = WaveActiveSum(bytes);
    if (total != 0 && WaveIsFirstLane()) {
        InterlockedAdd(((uint*)pc.counter)[0], total);
    }
}
