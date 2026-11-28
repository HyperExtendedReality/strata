struct RayProbePush {
    uint64_t scene;
    uint4* result;
    float4 origin_tmin;
    float4 direction_tmax;
};

[[vk::push_constant]]
ConstantBuffer<RayProbePush> pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
    RayDesc ray;
    ray.Origin = pc.origin_tmin.xyz;
    ray.TMin = pc.origin_tmin.w;
    ray.Direction = pc.direction_tmax.xyz;
    ray.TMax = pc.direction_tmax.w;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(RaytracingAccelerationStructure(pc.scene), 0, 0xff, ray);
    while (query.Proceed()) {}
    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        *pc.result = uint4(
            query.CommittedPrimitiveIndex(),
            asuint(query.CommittedRayT()),
            asuint(query.CommittedTriangleBarycentrics())
        );
    } else {
        *pc.result = uint4(0xffffffff, 0, 0, 0);
    }
}
