// GPU RectList topology expansion: converts N 3-vertex rectangles (v0, v1, v2)
// into 6-vertex triangle-list pairs (v0, v1, v2, v2, v1, v3) where v3 = v1 + v2 - v0.

struct RectExpandPush {
    uint64_t src_verts_addr;
    uint64_t dst_verts_addr;
    uint32_t rect_count;
    uint32_t _pad;
};

[[vk::push_constant]]
ConstantBuffer<RectExpandPush> pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint32_t r = dtid.x;
    if (r >= pc.rect_count) {
        return;
    }
    float4 *src = (float4 *)pc.src_verts_addr;
    float4 *dst = (float4 *)pc.dst_verts_addr;
    float4 v0 = src[r * 3u + 0u];
    float4 v1 = src[r * 3u + 1u];
    float4 v2 = src[r * 3u + 2u];
    float4 v3 = float4(v1.xyz + v2.xyz - v0.xyz, v0.w);

    uint32_t out_base = r * 6u;
    dst[out_base + 0u] = v0;
    dst[out_base + 1u] = v1;
    dst[out_base + 2u] = v2;
    dst[out_base + 3u] = v2;
    dst[out_base + 4u] = v1;
    dst[out_base + 5u] = v3;
}
