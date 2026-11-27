// Optional RT-off ambient/reflection clamp & NaN-scrubbing compute pass over RGBA32F BDA.

struct AmbientClampPush {
    uint64_t surface_addr;
    uint32_t width;
    uint32_t height;
    float ambient_floor;
    float max_luminance;
    uint32_t scrub_nans;
    uint32_t _pad;
};

[[vk::push_constant]]
ConstantBuffer<AmbientClampPush> pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= pc.width || dtid.y >= pc.height) {
        return;
    }
    uint32_t idx = dtid.y * pc.width + dtid.x;
    float4 *buf = (float4 *)pc.surface_addr;
    float4 c = buf[idx];
    if (pc.scrub_nans != 0u) {
        c.x = (isnan(c.x) || isinf(c.x)) ? 0.0 : c.x;
        c.y = (isnan(c.y) || isinf(c.y)) ? 0.0 : c.y;
        c.z = (isnan(c.z) || isinf(c.z)) ? 0.0 : c.z;
        c.w = (isnan(c.w) || isinf(c.w)) ? 1.0 : c.w;
    }
    c.xyz = clamp(max(c.xyz, float3(pc.ambient_floor, pc.ambient_floor, pc.ambient_floor)),
                  float3(0.0, 0.0, 0.0),
                  float3(pc.max_luminance, pc.max_luminance, pc.max_luminance));
    buf[idx] = c;
}
