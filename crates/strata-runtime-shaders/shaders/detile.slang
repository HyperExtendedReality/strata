// Generic Xbox _X tiled texture detiling & retiling compute shader using 64-bit BDA pointers.
// Supports 1, 2, 4, 8, and 16 BPP (including BC1-BC7 blocks) and multiple GFX10 swizzle modes:
//   mode & 0xFF == 0: Linear
//   mode & 0xFF == 1: 64KB_S (8x8 Morton)
//   mode & 0xFF == 2: 64KB_D (display 2D interleaved)
//   mode & 0xFF == 3: 64KB_S_X (8x8 Morton + pipe-bank XOR)
//   mode & 0x100 != 0: Retile (linear -> tiled) instead of Detile (tiled -> linear)

#include "common.slang"

uint32_t morton8x8(uint32_t lx, uint32_t ly) {
    return ((lx & 1u) << 0) | ((ly & 1u) << 1)
         | ((lx & 2u) << 1) | ((ly & 2u) << 2)
         | ((lx & 4u) << 2) | ((ly & 4u) << 3);
}

uint32_t computeTiledOffset(uint32_t x, uint32_t y, uint32_t width, uint32_t bpp, uint32_t mode) {
    uint32_t sw = mode & 0xFFu;
    if (sw == 0u) {
        return (y * width + x) * bpp;
    }
    uint32_t tile_x = x >> 3;
    uint32_t tile_y = y >> 3;
    uint32_t tiles_per_row = (width + 7u) >> 3;
    uint32_t lx = x & 7u;
    uint32_t ly = y & 7u;
    uint32_t intra = morton8x8(lx, ly);
    if (sw == 2u) {
        // 64KB_D display micro-tile rotates the upper 2 bits
        intra = ((intra & 0x0Fu) | (((intra >> 4) ^ (intra & 3u)) << 4));
    } else if (sw == 3u) {
        // Pipe-bank XOR across macro-tile coordinates
        uint32_t xor_bits = ((tile_x ^ tile_y) & 3u) << 4;
        intra ^= xor_bits;
    }
    uint32_t elem = ((tile_y * tiles_per_row + tile_x) << 6) | intra;
    return elem * bpp;
}

void copyElement(uint64_t src_addr, uint64_t dst_addr, uint32_t bpp) {
    if (bpp == 4u) {
        uint32_t *s = (uint32_t *)src_addr;
        uint32_t *d = (uint32_t *)dst_addr;
        *d = *s;
    } else if (bpp == 8u) {
        uint2 *s = (uint2 *)src_addr;
        uint2 *d = (uint2 *)dst_addr;
        *d = *s;
    } else if (bpp == 16u) {
        uint4 *s = (uint4 *)src_addr;
        uint4 *d = (uint4 *)dst_addr;
        *d = *s;
    } else if (bpp == 2u) {
        uint16_t *s = (uint16_t *)src_addr;
        uint16_t *d = (uint16_t *)dst_addr;
        *d = *s;
    } else {
        uint8_t *s = (uint8_t *)src_addr;
        uint8_t *d = (uint8_t *)dst_addr;
        for (uint32_t i = 0u; i < bpp; ++i) {
            d[i] = s[i];
        }
    }
}

[[vk::push_constant]]
ConstantBuffer<DetilePush> pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint2 p = dtid.xy;
    if (p.x >= pc.width || p.y >= pc.height) {
        return;
    }
    uint32_t tiled_off = computeTiledOffset(p.x, p.y, pc.width, pc.bpp, pc.swizzle_mode);
    uint32_t linear_off = (p.y * pc.width + p.x) * pc.bpp;
    bool retile = (pc.swizzle_mode & 0x100u) != 0u;
    uint64_t src = pc.src_addr + uint64_t(retile ? linear_off : tiled_off);
    uint64_t dst = pc.dst_addr + uint64_t(retile ? tiled_off : linear_off);
    copyElement(src, dst, pc.bpp);
}
