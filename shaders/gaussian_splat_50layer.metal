#include <metal_stdlib>
using namespace metal;

// High-capacity OIT for Gaussian Splatting
// Uses per-pixel linked list approach with compute shader sorting
// Can handle 50+ overlapping splats per pixel

#define MAX_FRAGMENTS_PER_PIXEL 50

struct SplatInstance {
    packed_float4 rotation;
    packed_float3 scale;
    packed_float3 position;
    packed_float4 color;
};

struct Uniforms {
    float4x4 viewProjectionMatrix;
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float2 viewportSize;
};

struct VertexOut {
    float4 position [[position]];
    float4 color;
    float2 uv;
    float depth;
};

// Fragment data for per-pixel lists
struct FragmentNode {
    half3 color;        // Premultiplied RGB
    half visibility;    // 1 - alpha (for under operator)
    half depth;         // View space depth
};

// Per-pixel fragment list (stored in 2D texture array or buffer)
struct PixelFragmentList {
    atomic_uint count;          // Number of fragments at this pixel
    FragmentNode fragments[MAX_FRAGMENTS_PER_PIXEL];
};

// Quad vertices
constant float2 quadVertices[6] = {
    float2(-1.0, -1.0), float2( 1.0, -1.0), float2(-1.0,  1.0),
    float2(-1.0,  1.0), float2( 1.0, -1.0), float2( 1.0,  1.0)
};

float3x3 quaternionToMatrix(float4 q) {
    float4 nq = normalize(q);
    float w = nq[0], x = nq[1], y = nq[2], z = nq[3];
    return float3x3(
        float3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
        float3(2.0 * (x * y - w * z),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),
        float3(2.0 * (x * z + w * y),       2.0 * (y * z - w * x),       1.0 - 2.0 * (x * x + y * y))
    );
}

float3x3 computeCovariance3D(float4 rotation, float3 scale) {
    float3x3 R = quaternionToMatrix(rotation);
    float3x3 S = float3x3(
        float3(scale.x, 0.0, 0.0),
        float3(0.0, scale.y, 0.0),
        float3(0.0, 0.0, scale.z)
    );
    float3x3 M = R * S;
    return M * transpose(M);
}

static inline void eigenSym2x2(
    float a, float b, float c,
    thread float2 &e1, thread float2 &e2,
    thread float  &l1, thread float  &l2
) {
    float tr = a + c;
    float det = a * c - b * b;
    float disc = max(0.0, 0.25 * tr * tr - det);
    float s = sqrt(disc);
    l1 = 0.5 * tr + s;
    l2 = 0.5 * tr - s;
    if (abs(b) > 1e-8) {
        e1 = normalize(float2(l1 - c, b));
    } else {
        e1 = (a >= c) ? float2(1.0, 0.0) : float2(0.0, 1.0);
    }
    e2 = float2(-e1.y, e1.x);
}

vertex VertexOut vertex_main(
    uint vertexID [[vertex_id]],
    uint instanceID [[instance_id]],
    constant SplatInstance* instances [[buffer(0)]],
    constant Uniforms& uniforms [[buffer(1)]]
) {
    VertexOut out;
    const SplatInstance instance = instances[instanceID];
    const float4 viewPos4 = uniforms.viewMatrix * float4(instance.position, 1.0);
    const float3 viewPos = viewPos4.xyz;
    const float zFront = -viewPos.z;

    if (zFront < 1e-4) {
        out.position = float4(0);
        out.color = instance.color;
        out.uv = float2(0);
        out.depth = -1.0;
        return out;
    }

    const float3x3 Sigma = computeCovariance3D(instance.rotation, instance.scale);
    const float3x3 viewRot = float3x3(
        uniforms.viewMatrix[0].xyz,
        uniforms.viewMatrix[1].xyz,
        uniforms.viewMatrix[2].xyz
    );
    const float3x3 viewCov = (viewRot * Sigma) * transpose(viewRot);
    const float focalX = uniforms.projectionMatrix[0][0] * (uniforms.viewportSize.x * 0.5);
    const float focalY = uniforms.projectionMatrix[1][1] * (uniforms.viewportSize.y * 0.5);
    const float invZ = 1.0 / zFront;
    const float invZ2 = invZ * invZ;
    const float3 J0 = float3(focalX * invZ, 0.0, -focalX * viewPos.x * invZ2);
    const float3 J1 = float3(0.0, focalY * invZ, -focalY * viewPos.y * invZ2);

    float a = dot(J0, viewCov * J0) + 1e-4;
    float b = dot(J0, viewCov * J1);
    float c = dot(J1, viewCov * J1) + 1e-4;

    float2 e1, e2;
    float l1, l2;
    eigenSym2x2(a, b, c, e1, e2, l1, l2);
    const float r1 = 3.0 * sqrt(max(l1, 0.0));
    const float r2 = 3.0 * sqrt(max(l2, 0.0));

    const float2 q = quadVertices[vertexID];
    const float2 offsetPx = (q.x * r1) * e1 + (q.y * r2) * e2;
    const float4 clip = uniforms.viewProjectionMatrix * float4(instance.position, 1.0);
    const float invW = 1.0 / clip.w;
    const float2 centerNDC = clip.xy * invW;
    const float2 offsetNDC = offsetPx * (2.0 / uniforms.viewportSize);

    out.position = float4(centerNDC + offsetNDC, clip.z * invW, 1.0);
    out.color = instance.color;
    out.uv = q * 3.0;
    out.depth = zFront;
    return out;
}

// Fragment shader: accumulate fragments into per-pixel lists
fragment void fragment_accumulate(
    VertexOut in [[stage_in]],
    device PixelFragmentList* fragmentLists [[buffer(0)]],
    constant Uniforms& uniforms [[buffer(1)]]
) {
    if (in.depth < 0.001) discard_fragment();

    float gaussianValue = exp(-0.5 * dot(in.uv, in.uv));
    if (gaussianValue < 0.01) discard_fragment();

    float alpha = gaussianValue * in.color.a;

    // Get pixel coordinates from fragment position
    uint2 pixelCoord = uint2(in.position.xy);

    // Get pixel index
    uint viewportWidth = uint(uniforms.viewportSize.x);
    uint pixelIndex = pixelCoord.y * viewportWidth + pixelCoord.x;
    device PixelFragmentList& list = fragmentLists[pixelIndex];

    // Atomically reserve a slot
    uint index = atomic_fetch_add_explicit(&list.count, 1, memory_order_relaxed);

    if (index < MAX_FRAGMENTS_PER_PIXEL) {
        list.fragments[index].color = half3(in.color.rgb);
        list.fragments[index].visibility = half(1.0 - alpha);
        list.fragments[index].depth = half(in.depth);
    }
}

// Compute shader: sort and composite per-pixel fragments
kernel void compute_sort_composite(
    device PixelFragmentList* fragmentLists [[buffer(0)]],
    texture2d<half, access::write> outputTexture [[texture(0)]],
    constant Uniforms& uniforms [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint viewportWidth = uint(uniforms.viewportSize.x);
    uint pixelIndex = gid.y * viewportWidth + gid.x;
    device PixelFragmentList& list = fragmentLists[pixelIndex];

    uint count = min(atomic_load_explicit(&list.count, memory_order_relaxed), uint(MAX_FRAGMENTS_PER_PIXEL));

    if (count == 0) {
        outputTexture.write(half4(0.0h), gid);
        return;
    }

    // Insertion sort (small lists, good cache locality)
    for (uint i = 1; i < count; i++) {
        FragmentNode key = list.fragments[i];
        int j = i - 1;
        while (j >= 0 && list.fragments[j].depth < key.depth) {
            list.fragments[j + 1] = list.fragments[j];
            j--;
        }
        list.fragments[j + 1] = key;
    }

    // Composite front-to-back using under operator
    half3 accumColor = half3(0.0h);
    half accumVis = 1.0h;

    for (uint i = 0; i < count; i++) {
        FragmentNode frag = list.fragments[i];
        // Premultiplied: color already has alpha baked in
        // Under operator: Cfinal += Csrc * visibility_behind
        accumColor += frag.color * accumVis;
        accumVis *= frag.visibility;

        if (accumVis < 0.01h) break;  // Early termination
    }

    outputTexture.write(half4(accumColor, 1.0h - accumVis), gid);
}
