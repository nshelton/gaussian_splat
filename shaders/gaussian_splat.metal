#include <metal_stdlib>
using namespace metal;

struct SplatInstance {
    packed_float4 rotation;
    packed_float3 scale;
    packed_float3 position;
    packed_float4 color;        // DC coefficients (RGB) + opacity
    float sh_rest[45];           // Remaining SH coefficients (degree 1-3)
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
    // Quaternion q = (w, x, y, z) where w is the scalar part
    float w = q[0], x = q[1], y = q[2], z = q[3];

    // Convert quaternion to rotation matrix (column-major for Metal)
    // Standard formula for unit quaternion to rotation matrix
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

    // eigenvector for l1 (largest eigenvalue)
    // handle near-diagonal / near-isotropic cases robustly
    if (abs(b) > 1e-8) {
        // Use (b, l1 - a) as eigenvector - more numerically stable
        e1 = normalize(float2(b, l1 - a));
    } else {
        // matrix is ~diagonal
        e1 = (a >= c) ? float2(1.0, 0.0) : float2(0.0, 1.0);
    }
    // Ensure right-handed coordinate system (counter-clockwise rotation)
    e2 = float2(-e1.y, e1.x);
}

// Evaluate spherical harmonics for view-dependent color
// dir: normalized view direction in world space
// sh_rest: 45 SH coefficients for degrees 1-3 (15 per channel)
float3 evaluateSH(float3 dir, const float sh_rest[45], float3 colorDC) {
    // SH basis functions (up to degree 3)
    float x = dir.x, y = dir.y, z = dir.z;

    // Degree 0 (DC) - apply SH C0 constant
    const float C0 = 0.28209479177387814;
    float3 result = C0 * colorDC;

    // Degree 1 (3 coefficients per channel)
    float sh1_0 = -0.4886025119029199 * y;
    float sh1_1 =  0.4886025119029199 * z;
    float sh1_2 = -0.4886025119029199 * x;

    result.r += sh1_0 * sh_rest[0] + sh1_1 * sh_rest[1] + sh1_2 * sh_rest[2];
    result.g += sh1_0 * sh_rest[3] + sh1_1 * sh_rest[4] + sh1_2 * sh_rest[5];
    result.b += sh1_0 * sh_rest[6] + sh1_1 * sh_rest[7] + sh1_2 * sh_rest[8];

    // Degree 2 (5 coefficients per channel)
    float sh2_0 =  1.0925484305920792 * x * y;
    float sh2_1 = -1.0925484305920792 * y * z;
    float sh2_2 =  0.31539156525252005 * (2.0 * z * z - x * x - y * y);
    float sh2_3 = -1.0925484305920792 * x * z;
    float sh2_4 =  0.5462742152960396 * (x * x - y * y);

    result.r += sh2_0 * sh_rest[9]  + sh2_1 * sh_rest[10] + sh2_2 * sh_rest[11] + sh2_3 * sh_rest[12] + sh2_4 * sh_rest[13];
    result.g += sh2_0 * sh_rest[14] + sh2_1 * sh_rest[15] + sh2_2 * sh_rest[16] + sh2_3 * sh_rest[17] + sh2_4 * sh_rest[18];
    result.b += sh2_0 * sh_rest[19] + sh2_1 * sh_rest[20] + sh2_2 * sh_rest[21] + sh2_3 * sh_rest[22] + sh2_4 * sh_rest[23];

    // Degree 3 (7 coefficients per channel)
    float sh3_0 = -0.5900435899266435 * y * (3.0 * x * x - y * y);
    float sh3_1 =  2.890611442640554 * x * y * z;
    float sh3_2 = -0.4570457994644658 * y * (4.0 * z * z - x * x - y * y);
    float sh3_3 =  0.3731763325901154 * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y);
    float sh3_4 = -0.4570457994644658 * x * (4.0 * z * z - x * x - y * y);
    float sh3_5 =  1.445305721320277 * z * (x * x - y * y);
    float sh3_6 = -0.5900435899266435 * x * (x * x - 3.0 * y * y);

    result.r += sh3_0 * sh_rest[24] + sh3_1 * sh_rest[25] + sh3_2 * sh_rest[26] + sh3_3 * sh_rest[27] + sh3_4 * sh_rest[28] + sh3_5 * sh_rest[29] + sh3_6 * sh_rest[30];
    result.g += sh3_0 * sh_rest[31] + sh3_1 * sh_rest[32] + sh3_2 * sh_rest[33] + sh3_3 * sh_rest[34] + sh3_4 * sh_rest[35] + sh3_5 * sh_rest[36] + sh3_6 * sh_rest[37];
    result.b += sh3_0 * sh_rest[38] + sh3_1 * sh_rest[39] + sh3_2 * sh_rest[40] + sh3_3 * sh_rest[41] + sh3_4 * sh_rest[42] + sh3_5 * sh_rest[43] + sh3_6 * sh_rest[44];

    // Clamp result to valid color range
    return saturate(result);
}

vertex VertexOut vertex_main(
    uint vertexID [[vertex_id]],
    uint instanceID [[instance_id]],
    constant SplatInstance* instances [[buffer(0)]],
    constant Uniforms& uniforms [[buffer(1)]],
    constant uint* sortedIndices [[buffer(2)]]
) {
    VertexOut out;

    // Use sorted index for depth ordering (back-to-front)
    uint actualIndex = sortedIndices[instanceID];
    const SplatInstance instance = instances[actualIndex];

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

    // Compute view-dependent color using spherical harmonics
    // Camera position in world space = -(R^T * t) where viewMatrix = [R | t]
    float3x3 R = float3x3(uniforms.viewMatrix[0].xyz, uniforms.viewMatrix[1].xyz, uniforms.viewMatrix[2].xyz);
    float3 t = float3(uniforms.viewMatrix[3][0], uniforms.viewMatrix[3][1], uniforms.viewMatrix[3][2]);
    float3 cameraPos = transpose(R) * (-t);

    // View direction in world space (from splat to camera)
    float3 viewDir = normalize(cameraPos - float3(instance.position));

    // Evaluate SH to get view-dependent color
    // TEMPORARY: Disable SH to test base rendering
    // float3 shColor = evaluateSH(viewDir, instance.sh_rest, instance.color.rgb);
    float3 shColor = instance.color.rgb;  // Just use DC component

    out.position = float4(posNDC, clip.z * invW, 1.0);
    out.color = float4(shColor, instance.color.a);  // SH color + opacity
    out.uv = q * 3.0;     // [-1,1] -> [-3,3] sigma space
    out.depth = zFront;   // positive forward depth
    return out;
}


fragment float4 fragment_main(VertexOut in [[stage_in]]) 
{
    // Evaluate 2D Gaussian
    // UV coordinates are in standard deviation units: [-3σ, 3σ] range
    float2 d = in.uv;

    // // Cull splats behind camera
    if (in.depth < 0.001) {
        discard_fragment();
    }

    // Canonical Gaussian in sigma space: exp(-0.5 * ||d||^2)
    float gaussianValue = exp(-0.5 * dot(d, d));

    // Apply per-splat opacity
    float alpha = gaussianValue * in.color.a;

    // Discard fragments that contribute very little
    if (gaussianValue < 0.1) {
        discard_fragment();
    }
    // Return color with gaussian-modulated alpha
    return float4(in.color.rgb, alpha);
}

