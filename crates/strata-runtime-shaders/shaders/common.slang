// Shared No-API types & helpers for Strata's Slang runtime shaders.
// Uses native 64-bit PhysicalStorageBuffer pointers (`Ptr<T>`) and Push Constants.

struct DirtyHashPush {
    uint64_t src_addr;      // 64-bit BDA pointer to buffer being hashed
    uint64_t dst_hash_addr; // 64-bit BDA pointer to u32 output hash array
    uint32_t block_count;   // Number of 64 KiB pages
    uint32_t dwords_per_block;
};

struct DetilePush {
    uint64_t src_addr;      // Console tiled surface (BDA)
    uint64_t dst_addr;      // Linear / host staging surface (BDA)
    uint32_t width;
    uint32_t height;
    uint32_t bpp;           // Bytes per pixel / block
    uint32_t swizzle_mode;
};

struct BvhOraclePush {
    uint64_t rays_addr;     // Input Ray array (BDA)
    uint64_t nodes_addr;    // Input Box32Node array (BDA)
    uint64_t results_addr;  // Output uint4 hit child IDs (BDA)
    uint32_t count;
};

// Fast FNV-1a / Murmur-style avalanche mixer for 64 KiB GPU dirty-page detection.
uint32_t mix32(uint32_t h, uint32_t v) {
    h ^= v * 0x85EBCA6Bu;
    h = (h << 13) | (h >> 19);
    return h * 5u + 0xE6546B64u;
}
