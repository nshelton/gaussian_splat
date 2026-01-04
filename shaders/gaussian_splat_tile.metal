#include <metal_stdlib>
using namespace metal;

// Tile-based Gaussian Splat Renderer
// Uses Metal's tile shading pipeline for order-independent transparency

#define MAX_FRAGMENTS_PER_PIXEL 32

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

// Quad vertices (2 triangles forming a billboard)
constant float2 quadVertices[6] = {
    float2(-1.0, -1.0),
    float2( 1.0, -1.0),
    float2(-1.0,  1.0),
    float2(-1.0,  1.0),
    float2( 1.0, -1.0),
    float2( 1.0,  1.0)
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
        float3(scale.x, 0.0,    0.0),
        float3(0.0,    scale.y, 0.0),
        float3(0.0,    0.0,    scale.z)
    );
    float3x3 M = R * S;
    return M * transpose(M);
}

static inline void eigenSym2x2(
    float a, float b, float c,
    thread float2 &e1,
    thread float2 &e2,
    thread float  &l1,
    thread float  &l2
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
    const float zView = viewPos.z;
    const float zFront = -zView;

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

    float a = dot(J0, viewCov * J0);
    float b = dot(J0, viewCov * J1);
    float c = dot(J1, viewCov * J1);

    const float eps = 1e-4;
    a += eps;
    c += eps;

    float2 e1, e2;
    float l1, l2;
    eigenSym2x2(a, b, c, e1, e2, l1, l2);
    l1 = max(l1, 0.0);
    l2 = max(l2, 0.0);

    const float r1 = 3.0 * sqrt(l1);
    const float r2 = 3.0 * sqrt(l2);

    const float2 q = quadVertices[vertexID];
    const float2 offsetPx = (q.x * r1) * e1 + (q.y * r2) * e2;

    const float4 clip = uniforms.viewProjectionMatrix * float4(instance.position, 1.0);
    const float invW = 1.0 / clip.w;
    const float2 centerNDC = clip.xy * invW;

    const float2 offsetNDC = offsetPx * (2.0 / uniforms.viewportSize);
    float2 posNDC = centerNDC + offsetNDC;

    out.position = float4(posNDC, clip.z * invW, 1.0);
    out.color = instance.color;
    out.uv = q * 3.0;
    out.depth = zFront;
    return out;
}

// Fragment data for tile shader
struct FragmentData {
    half depth;
    half3 color;
    half alpha;
};

// Threadgroup memory for fragment accumulation
struct TileData {
    FragmentData fragments[MAX_FRAGMENTS_PER_PIXEL];
    atomic_uint count;
};

// Fragment shader - writes to threadgroup memory (tile memory)
// Using programmable blending to accumulate fragments per pixel
struct FragmentOut {
    half4 color [[color(0)]];
};

fragment FragmentOut fragment_main(
    VertexOut in [[stage_in]],
    device atomic_uint* fragmentCounter [[buffer(0)]],
    device FragmentData* fragmentBuffer [[buffer(1)]],
    uint2 pixelCoord [[thread_position_in_grid]]
) {
    // Evaluate 2D Gaussian
    float2 d = in.uv;

    if (in.depth < 0.001) {
        discard_fragment();
    }

    float gaussianValue = exp(-0.5 * dot(d, d));

    if (gaussianValue < 0.01) {
        discard_fragment();
    }

    float alpha = gaussianValue * in.color.a;

    // Store fragment in global buffer
    uint index = atomic_fetch_add_explicit(fragmentCounter, 1, memory_order_relaxed);

    if (index < MAX_FRAGMENTS_PER_PIXEL) {
        fragmentBuffer[index].depth = half(in.depth);
        fragmentBuffer[index].color = half3(in.color.rgb);
        fragmentBuffer[index].alpha = half(alpha);
    }

    // Don't write to framebuffer yet - tile shader will do final composite
    FragmentOut out;
    out.color = half4(0.0h);
    return out;
}

// Tile shader - sorts and composites fragments
kernel void tile_sort_composite(
    texture2d<half, access::write> outputTexture [[texture(0)]],
    device FragmentData* fragmentBuffer [[buffer(0)]],
    device atomic_uint* fragmentCounter [[buffer(1)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tileSize [[threads_per_threadgroup]]
) {
    // Get fragment count for this pixel
    uint fragCount = atomic_load_explicit(fragmentCounter, memory_order_relaxed);
    fragCount = min(fragCount, uint(MAX_FRAGMENTS_PER_PIXEL));

    if (fragCount == 0) {
        outputTexture.write(half4(0.0h, 0.0h, 0.0h, 0.0h), tid);
        return;
    }

    // Sort fragments by depth (insertion sort)
    FragmentData sorted[MAX_FRAGMENTS_PER_PIXEL];

    for (uint i = 0; i < fragCount; i++) {
        sorted[i] = fragmentBuffer[i];
    }

    // Insertion sort (front-to-back)
    for (uint i = 1; i < fragCount; i++) {
        FragmentData key = sorted[i];
        int j = i - 1;

        while (j >= 0 && sorted[j].depth < key.depth) {
            sorted[j + 1] = sorted[j];
            j--;
        }
        sorted[j + 1] = key;
    }

    // Composite front-to-back
    half3 accumColor = half3(0.0h);
    half accumAlpha = 0.0h;

    for (uint i = 0; i < fragCount; i++) {
        FragmentData frag = sorted[i];
        half srcAlpha = frag.alpha * (1.0h - accumAlpha);
        accumColor += frag.color * srcAlpha;
        accumAlpha += srcAlpha;

        if (accumAlpha >= 0.99h) {
            break;
        }
    }

    outputTexture.write(half4(accumColor, accumAlpha), tid);
}
