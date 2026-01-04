#include <metal_stdlib>
#include "gaussian_splat_types.h"
using namespace metal;

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

    // Columns of column-major rotation matrix
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
    // Sigma = M * M^T
    return M * transpose(M);
}

// Robust eigen basis for symmetric 2x2
static inline void eigenSym2x2(
    float a, float b, float c,   // [a b; b c]
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

    // eigenvector for l1
    // handle near-diagonal / near-isotropic cases robustly
    if (abs(b) > 1e-8) {
        // (l1 - c, b) is a valid eigenvector
        e1 = normalize(float2(l1 - c, b));
    } else {
        // matrix is ~diagonal
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

    // World -> view for the splat center
    const float4 viewPos4 = uniforms.viewMatrix * float4(instance.position, 1.0);
    const float3 viewPos = viewPos4.xyz;
    const float zView = viewPos.z;
    const float zFront = -zView; // camera looks down -Z

    // Cull anything too close to the plane or behind
    if (zFront < 1e-4) {
        out.position = float4(0);
        out.color = instance.color;
        out.uv = float2(0);
        out.depth = -1.0;  // Negative depth will be culled
        return out;
    }

    // 3D covariance in world space
    const float3x3 Sigma = computeCovariance3D(instance.rotation, instance.scale);

    // View rotation (world -> view) directly from the upper-left 3x3.
    const float3x3 viewRot = float3x3(
        uniforms.viewMatrix[0].xyz,
        uniforms.viewMatrix[1].xyz,
        uniforms.viewMatrix[2].xyz
    );

    // Covariance in view space
    const float3x3 viewCov = (viewRot * Sigma) * transpose(viewRot);

    // Pixel-space focal lengths from projection
    const float focalX = uniforms.projectionMatrix[0][0] * (uniforms.viewportSize.x * 0.5);
    const float focalY = uniforms.projectionMatrix[1][1] * (uniforms.viewportSize.y * 0.5);

    // Jacobian rows of perspective projection (view -> pixel)
    const float invZ = 1.0 / zFront;
    const float invZ2 = invZ * invZ;
    const float3 J0 = float3(focalX * invZ, 0.0, -focalX * viewPos.x * invZ2);
    const float3 J1 = float3(0.0, focalY * invZ, -focalY * viewPos.y * invZ2);

    // 2x2 covariance in pixel space
    float a = dot(J0, viewCov * J0);
    float b = dot(J0, viewCov * J1);
    float c = dot(J1, viewCov * J1);

    // Small diagonal lift to keep things stable
    const float eps = 1e-4;
    a += eps;
    c += eps;

    float2 e1, e2;
    float l1, l2;
    eigenSym2x2(a, b, c, e1, e2, l1, l2);
    l1 = max(l1, 0.0);
    l2 = max(l2, 0.0);

    // 3-sigma radii in pixels
    const float r1 = 3.0 * sqrt(l1);
    const float r2 = 3.0 * sqrt(l2);

    // Billboard corner in [-1,1]
    const float2 q = quadVertices[vertexID];

    // Pixel offset aligned to ellipse axes
    const float2 offsetPx = (q.x * r1) * e1 + (q.y * r2) * e2;

    // Center in clip/NDC
    const float4 clip = uniforms.viewProjectionMatrix * float4(instance.position, 1.0);
    const float invW = 1.0 / clip.w;
    const float2 centerNDC = clip.xy * invW;

    // Apply pixel offset in NDC units
    const float2 offsetNDC = offsetPx * (2.0 / uniforms.viewportSize);
    float2 posNDC = centerNDC + offsetNDC;

    out.position = float4(posNDC, clip.z * invW, 1.0);
    out.color = instance.color;
    out.uv = q * 3.0;     // [-1,1] -> [-3,3] sigma space
    out.depth = zFront;   // positive forward depth
    return out;
}


fragment void fragment_main(
    VertexOut in [[stage_in]],
    imageblock_data<ImageBlockData> imageblock [[imageblock_data]]
) {
    // Evaluate 2D Gaussian
    // UV coordinates are in standard deviation units: [-3σ, 3σ] range
    float2 d = in.uv;

    // Cull splats behind camera
    if (in.depth < 0.001) {
        discard_fragment();
    }

    // Canonical Gaussian in sigma space: exp(-0.5 * ||d||^2)
    float gaussianValue = exp(-0.5 * dot(d, d));

    // Discard fragments that contribute very little
    if (gaussianValue < 0.01) {
        discard_fragment();
    }

    // Apply per-splat opacity
    float alpha = gaussianValue * in.color.a;

    // Write to imageblock instead of framebuffer
    uint index = imageblock.count;

    if (index >= MAX_FRAGMENTS_PER_PIXEL) {
        // OVERFLOW HANDLING: Replace furthest fragment if current is closer
        uint furthest_idx = 0;
        half furthest_depth = imageblock.fragments[0].depth;

        for (uint i = 1; i < MAX_FRAGMENTS_PER_PIXEL; i++) {
            if (imageblock.fragments[i].depth < furthest_depth) {
                furthest_depth = imageblock.fragments[i].depth;
                furthest_idx = i;
            }
        }

        // Only replace if current fragment is closer
        if (half(in.depth) > furthest_depth) {
            index = furthest_idx;
        } else {
            discard_fragment();  // Current fragment is further, discard
        }
    }

    // Write fragment data
    imageblock.fragments[index].depth = half(in.depth);
    imageblock.fragments[index].color = half3(in.color.rgb);
    imageblock.fragments[index].alpha = half(alpha);

    if (index == imageblock.count) {
        imageblock.count++;  // Increment only if new fragment
    }
}

