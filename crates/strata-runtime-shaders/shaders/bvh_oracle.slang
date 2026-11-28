// GPU reference oracle for RDNA2 `image_bvh64_intersect_ray` (box32 node).
// Compiled via Slang to SPIR-V 1.6 with BDA pointers to differentially test
// `strata-shader`'s direct SPIR-V BVH lowering on real GPU hardware.

#include "common.slang"

struct RayData {
    float t_max;
    float3 origin;
    float3 dir;
    float3 inv_dir;
};

struct Box32NodeData {
    uint32_t children[4];
    float3 bmin[4];
    float3 bmax[4];
    // Strata trailer [112..128]: parent ptr, depth|leaf_count<<16, then
    // four binary16 child surface-area priors (strata_shader::bvh::Box32Meta).
    uint32_t meta[4];
};

[[vk::push_constant]]
ConstantBuffer<BvhOraclePush> pc;

void cmpSwap(inout float da, inout float db, inout uint32_t ha, inout uint32_t hb, inout float pa, inout float pb) {
    if (db < da) {
        float td = da; da = db; db = td;
        uint32_t th = ha; ha = hb; hb = th;
        float tp = pa; pa = pb; pb = tp;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint32_t idx = dtid.x;
    if (idx >= pc.count) {
        return;
    }

    RayData ray = ((RayData *)pc.rays_addr)[idx];
    Box32NodeData node = ((Box32NodeData *)pc.nodes_addr)[idx];

    uint32_t hits[4] = { 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu };
    float dists[4] = { 1.0 / 0.0, 1.0 / 0.0, 1.0 / 0.0, 1.0 / 0.0 };
    float priors[4] = {
        f16tof32(node.meta[2] & 0xFFFFu), f16tof32(node.meta[2] >> 16),
        f16tof32(node.meta[3] & 0xFFFFu), f16tof32(node.meta[3] >> 16)
    };

    [unroll]
    for (uint32_t i = 0; i < 4; ++i) {
        if (node.children[i] != 0xFFFFFFFFu) {
            float3 t0 = (node.bmin[i] - ray.origin) * ray.inv_dir;
            float3 t1 = (node.bmax[i] - ray.origin) * ray.inv_dir;
            float3 n = min(t0, t1);
            float3 f = max(t0, t1);
            float tmin = max(max(n.x, n.y), max(n.z, 0.0));
            float tmax = min(min(f.x, f.y), min(f.z, ray.t_max));
            if (tmin <= tmax) {
                hits[i] = node.children[i];
                dists[i] = tmin;
            }
        }
    }

    // 4-element sorting network matching RDNA2 box sort
    cmpSwap(dists[0], dists[2], hits[0], hits[2], priors[0], priors[2]);
    cmpSwap(dists[1], dists[3], hits[1], hits[3], priors[1], priors[3]);
    cmpSwap(dists[0], dists[1], hits[0], hits[1], priors[0], priors[1]);
    cmpSwap(dists[2], dists[3], hits[2], hits[3], priors[2], priors[3]);
    cmpSwap(dists[1], dists[2], hits[1], hits[2], priors[1], priors[2]);

    // Optional surface-area-prior pruning of the non-nearest children
    // (strata_shader::bvh::intersect_box32_prior_pruned; 0 = exact)
    if (pc.bma_threshold > 0.0) {
        [unroll]
        for (uint32_t k = 1; k < 4; ++k) {
            if (hits[k] != 0xFFFFFFFFu) {
                float score = priors[k] / (1.0 + max(dists[k], 0.0));
                if (score < pc.bma_threshold) {
                    hits[k] = 0xFFFFFFFFu;
                }
            }
        }
    }

    uint4 *out_res = (uint4 *)pc.results_addr;
    out_res[idx] = uint4(hits[0], hits[1], hits[2], hits[3]);
}
