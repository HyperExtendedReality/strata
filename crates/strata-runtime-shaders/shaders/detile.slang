// Generic Xbox _X tiled texture detiling compute shader using BDA pointers.
// Uses Slang interfaces so new RDNA2 swizzle modes specialize at compile time.

#include "common.slang"

interface ITileAddress {
    static uint32_t tiledOffset(uint32_t x, uint32_t y, uint32_t width, uint32_t bpp);
};

// Standard 2D Morton / 64KB_S-style micro-tile addressing (8x8 micro-tile).
struct Swizzle64KS : ITileAddress {
    static uint32_t tiledOffset(uint32_t x, uint32_t y, uint32_t width, uint32_t bpp) {
        uint32_t tile_x = x >> 3;
        uint32_t tile_y = y >> 3;
        uint32_t tiles_per_row = (width + 7u) >> 3;
        uint32_t lx = x & 7u;
        uint32_t ly = y & 7u;
        uint32_t morton = ((lx & 1u) << 0) | ((ly & 1u) << 1)
                        | ((lx & 2u) << 1) | ((ly & 2u) << 2)
                        | ((lx & 4u) << 2) | ((ly & 4u) << 3);
        uint32_t elem = ((tile_y * tiles_per_row + tile_x) << 6) | morton;
        return elem * bpp;
    }
};

void detilePixel<T : ITileAddress>(uint2 p) {
    if (p.x >= pc.width || p.y >= pc.height) {
        return;
    }
    uint32_t src_off = T::tiledOffset(p.x, p.y, pc.width, pc.bpp);
    uint32_t dst_off = (p.y * pc.width + p.x) * pc.bpp;

    uint32_t *src = (uint32_t *)(pc.src_addr + uint64_t(src_off));
    uint32_t *dst = (uint32_t *)(pc.dst_addr + uint64_t(dst_off));
    *dst = *src;
}

[[vk::push_constant]]
ConstantBuffer<DetilePush> pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    detilePixel<Swizzle64KS>(dtid.xy);
}
