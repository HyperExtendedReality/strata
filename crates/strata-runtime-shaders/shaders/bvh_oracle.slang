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
};

[[vk::push_constant]]
ConstantBuffer<BvhOraclePush> pc;

void cmpSwap(inout float da, inout float db, inout uint32_t ha, inout uint32_t hb) {
    if (db < da) {
        float td = da; da = db; db = td;
        uint32_t th = ha; ha = hb; hb = th;
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
    cmpSwap(dists[0], dists[2], hits[0], hits[2]);
    cmpSwap(dists[1], dists[3], hits[1], hits[3]);
    cmpSwap(dists[0], dists[1], hits[0], hits[1]);
    cmpSwap(dists[2], dists[3], hits[2], hits[3]);
    cmpSwap(dists[1], dists[2], hits[1], hits[2]);

    uint4 *out_res = (uint4 *)pc.results_addr;
    out_res[idx] = uint4(hits[0], hits[1], hits[2], hits[3]);
}
