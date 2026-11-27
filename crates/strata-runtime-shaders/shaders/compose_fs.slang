[[vk::binding(0, 0)]] Texture2D<float4> planes[1];

[shader("fragment")]
float4 main(float2 uv : TEXCOORD0) : SV_Target0 {
    uint width, height;
    planes[0].GetDimensions(width, height);
    int2 edge = int2(width, height) - 1;
    float2 p = uv * float2(width, height) - 0.5;
    int2 lo = int2(floor(p));
    float2 weight = frac(p);
    float4 a = planes[0].Load(int3(clamp(lo, 0, edge), 0));
    float4 b = planes[0].Load(int3(clamp(lo + int2(1, 0), 0, edge), 0));
    float4 c = planes[0].Load(int3(clamp(lo + int2(0, 1), 0, edge), 0));
    float4 d = planes[0].Load(int3(clamp(lo + int2(1, 1), 0, edge), 0));
    return lerp(lerp(a, b, weight.x), lerp(c, d, weight.x), weight.y);
}
