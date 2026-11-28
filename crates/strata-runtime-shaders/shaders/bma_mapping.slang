// BMA (BVH Mapping Algorithm, Chen/Xie/Yuan, Computers 2026 15(6) 363):
// the Mapping stage on the GPU, run at a frame boundary.
//
// Table layout: see strata_shader::bma (64-byte header, then one u32
// shortcut entry per (tile, ray slot), then the sampled paths: one u32 per
// (tile, ray slot, depth), node byte offset from the root + 1, 0 = none).
//
// mode 0 (map): one thread per (tile, ray slot). Over the 3x3 neighbouring
//   tiles' sampled nodes at each depth, accumulate kernel weight per node;
//   while the best node's weight >= threshold go one level deeper. The
//   deepest passing node (below the root) becomes the shortcut. Border tiles
//   keep the root. Thread 0 publishes the sampled build generation.
// mode 1 (reset): one thread per path dword: clear it for the next frame's
//   sampling. Thread 0 applies a new root / flags when given.

static const uint32_t H_MAP_BUILD = 4;
static const uint32_t H_TILES_X = 8;
static const uint32_t H_TILES_Y = 12;
static const uint32_t H_RAYS = 20;
static const uint32_t H_DEPTH = 24;
static const uint32_t H_FLAGS = 28;
static const uint32_t H_ROOT_LO = 32;
static const uint32_t H_ROOT_HI = 36;
static const uint32_t H_SAMPLE_BUILD = 40;
static const uint32_t H_PATHS_OFFSET = 48;
static const uint32_t HEADER_BYTES = 64;

struct BmaMappingPush {
    uint64_t table_addr;
    uint64_t new_root;      // mode 1: non-zero replaces the root
    float threshold;
    uint32_t mode;
    uint32_t flags;         // mode 1: written to the header when new_root != 0
    uint32_t count;         // threads with work
    float kernel[9];        // row-major [dy][dx], centre at 4
};

[[vk::push_constant]]
ConstantBuffer<BmaMappingPush> pc;

uint32_t hdr(uint32_t off) {
    uint32_t *t = (uint32_t *)pc.table_addr;
    return t[off / 4];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint32_t idx = dtid.x;
    uint32_t *t = (uint32_t *)pc.table_addr;
    if (pc.mode == 1u) {
        if (idx == 0u && pc.new_root != 0ull) {
            t[H_ROOT_LO / 4] = uint32_t(pc.new_root & 0xFFFFFFFFull);
            t[H_ROOT_HI / 4] = uint32_t(pc.new_root >> 32);
            t[H_FLAGS / 4] = pc.flags;
        }
        if (idx < pc.count) {
            t[hdr(H_PATHS_OFFSET) / 4 + idx] = 0u;
        }
        return;
    }

    if (idx == 0u) {
        t[H_MAP_BUILD / 4] = hdr(H_SAMPLE_BUILD);
    }
    if (idx >= pc.count) {
        return;
    }
    uint32_t tiles_x = hdr(H_TILES_X);
    uint32_t tiles_y = hdr(H_TILES_Y);
    uint32_t rays = hdr(H_RAYS);
    uint32_t depth = hdr(H_DEPTH);
    uint32_t paths = hdr(H_PATHS_OFFSET) / 4;
    uint32_t r = idx % rays;
    uint32_t tile = idx / rays;
    uint32_t tx = tile % tiles_x;
    uint32_t ty = tile / tiles_x;
    uint32_t entry = HEADER_BYTES / 4 + idx;
    if (tx == 0u || ty == 0u || tx + 1u >= tiles_x || ty + 1u >= tiles_y) {
        t[entry] = 0u;
        return;
    }

    uint32_t best = 0u;
    for (uint32_t d = 0u; d < depth; ++d) {
        uint32_t keys[9];
        float weights[9];
        uint32_t n = 0u;
        [unroll]
        for (uint32_t j = 0u; j < 3u; ++j) {
            [unroll]
            for (uint32_t k = 0u; k < 3u; ++k) {
                uint32_t nt = (ty + j - 1u) * tiles_x + (tx + k - 1u);
                uint32_t node = t[paths + (nt * rays + r) * depth + d];
                if (node == 0u) {
                    continue;
                }
                float w = pc.kernel[j * 3u + k];
                bool found = false;
                for (uint32_t q = 0u; q < n; ++q) {
                    if (keys[q] == node) {
                        weights[q] += w;
                        found = true;
                    }
                }
                if (!found) {
                    keys[n] = node;
                    weights[n] = w;
                    n += 1u;
                }
            }
        }
        if (n == 0u) {
            break;
        }
        uint32_t arg = 0u;
        for (uint32_t q = 1u; q < n; ++q) {
            if (weights[q] > weights[arg] || (weights[q] == weights[arg] && keys[q] < keys[arg])) {
                arg = q;
            }
        }
        if (weights[arg] + 1e-6 < pc.threshold) {
            break;
        }
        best = keys[arg];
    }
    // Entry 1 is the root itself (offset 0): no shortcut.
    t[entry] = best > 1u ? best : 0u;
}
