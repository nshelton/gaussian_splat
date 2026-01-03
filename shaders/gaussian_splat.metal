#include <metal_stdlib>
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
    float2 conic;  // For 2D Gaussian evaluation
    float radius;  // For depth testing
    float depth;   // For depth-based weighting
};

// Output for Weighted Blended OIT
struct FragmentOut {
    float4 accumulation [[color(0)]];  // RGB * alpha * weight, A = alpha * weight
    float revealage [[color(1)]];      // Product of (1 - alpha)
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

// Build rotation matrix from quaternion
float3x3 quaternionToMatrix(float4 q) {
    float x = q.x, y = q.y, z = q.z, w = q.w;

    float3x3 R;
    R[0][0] = 1.0 - 2.0 * (y * y + z * z);
    R[0][1] = 2.0 * (x * y - w * z);
    R[0][2] = 2.0 * (x * z + w * y);

    R[1][0] = 2.0 * (x * y + w * z);
    R[1][1] = 1.0 - 2.0 * (x * x + z * z);
    R[1][2] = 2.0 * (y * z - w * x);

    R[2][0] = 2.0 * (x * z - w * y);
    R[2][1] = 2.0 * (y * z + w * x);
    R[2][2] = 1.0 - 2.0 * (x * x + y * y);

    return R;
}

// Compute 3D covariance matrix: Sigma = R * S * S^T * R^T
float3x3 computeCovariance3D(float4 rotation, float3 scale) {
    float3x3 R = quaternionToMatrix(rotation);

    // S is diagonal scale matrix
    float3x3 S = float3x3(
        float3(scale.x, 0.0, 0.0),
        float3(0.0, scale.y, 0.0),
        float3(0.0, 0.0, scale.z)
    );

    float3x3 M = R * S;

    // Sigma = M * M^T
    float3x3 Sigma;
    Sigma[0] = float3(dot(M[0], M[0]), dot(M[0], M[1]), dot(M[0], M[2]));
    Sigma[1] = float3(dot(M[1], M[0]), dot(M[1], M[1]), dot(M[1], M[2]));
    Sigma[2] = float3(dot(M[2], M[0]), dot(M[2], M[1]), dot(M[2], M[2]));

    return Sigma;
}

vertex VertexOut vertex_main(
    uint vertexID [[vertex_id]],
    uint instanceID [[instance_id]],
    constant SplatInstance* instances [[buffer(0)]],
    constant Uniforms& uniforms [[buffer(1)]]
) {
    VertexOut out;

    // Get instance data
    SplatInstance instance = instances[instanceID];

    // Transform splat center to view space
    float3 viewPos = (uniforms.viewMatrix * float4(instance.position, 1.0)).xyz;

    // Compute 3D covariance matrix
    float3x3 Sigma = computeCovariance3D(instance.rotation, instance.scale);

    // Transform covariance to view space: V * Sigma * V^T
    float3x3 viewMatrix3x3 = float3x3(
        uniforms.viewMatrix[0].xyz,
        uniforms.viewMatrix[1].xyz,
        uniforms.viewMatrix[2].xyz
    );
    float3x3 T = viewMatrix3x3 * Sigma;
    float3x3 viewCov = T * transpose(viewMatrix3x3);

    // Project to 2D using EWA splatting (compute Jacobian of perspective projection)
    float z = viewPos.z;
    float focal_x = uniforms.projectionMatrix[0][0] * uniforms.viewportSize.x * 0.5;
    float focal_y = uniforms.projectionMatrix[1][1] * uniforms.viewportSize.y * 0.5;

    // Jacobian of perspective projection
    float2x3 J = float2x3(
        float3(focal_x / z, 0.0, -focal_x * viewPos.x / (z * z)),
        float3(0.0, focal_y / z, -focal_y * viewPos.y / (z * z))
    );

    // Project covariance: Sigma2D = J * Sigma3D * J^T
    float2x2 cov2D;
    cov2D[0][0] = dot(J[0], viewCov * J[0]);
    cov2D[0][1] = dot(J[0], viewCov * J[1]);
    cov2D[1][0] = dot(J[1], viewCov * J[0]);
    cov2D[1][1] = dot(J[1], viewCov * J[1]);

    // Add a small epsilon to diagonal for numerical stability
    cov2D[0][0] += 0.01;
    cov2D[1][1] += 0.01;

    // Compute eigenvalues and eigenvectors for oriented quad
    float det = cov2D[0][0] * cov2D[1][1] - cov2D[0][1] * cov2D[1][0];
    float trace = cov2D[0][0] + cov2D[1][1];
    float sqrtDisc = sqrt(max(0.0, trace * trace * 0.25 - det));
    float eigenvalue1 = trace * 0.5 + sqrtDisc;
    float eigenvalue2 = trace * 0.5 - sqrtDisc;

    // Compute eigenvector for major axis
    float2 eigenvec1 = normalize(float2(cov2D[0][1], eigenvalue1 - cov2D[0][0]));
    float2 eigenvec2 = float2(-eigenvec1.y, eigenvec1.x);

    // Quad extends to 3 standard deviations
    float radius1 = 3.0 * sqrt(eigenvalue1);
    float radius2 = 3.0 * sqrt(eigenvalue2);

    // Get quad vertex position
    float2 quadPos = quadVertices[vertexID];

    // Construct oriented quad vertex in screen space
    float2 offset = quadPos.x * radius1 * eigenvec1 + quadPos.y * radius2 * eigenvec2;

    // Project center to screen
    float4 clipPos = uniforms.viewProjectionMatrix * float4(instance.position, 1.0);
    clipPos /= clipPos.w;

    // Add offset in NDC space
    clipPos.xy += offset / (uniforms.viewportSize * 0.5);

    out.position = clipPos;
    out.color = instance.color;
    out.uv = quadPos;
    out.radius = max(radius1, radius2);
    out.depth = clipPos.z / clipPos.w;  // Normalized device coordinate depth

    return out;
}

fragment FragmentOut fragment_main(VertexOut in [[stage_in]]) {
    // Evaluate 2D Gaussian
    // The UV coordinates are already in [-1, 1] range from quad vertices
    float2 d = in.uv;

    // Gaussian evaluation: exp(-0.5 * d^T * d)
    // Since we scaled the quad by the eigenvalues,
    // we can use simple distance-based Gaussian
    float power = -0.5 * dot(d, d);
    float alpha = exp(power);

    // Discard fragments that contribute very little
    if (alpha < 0.001) {
        discard_fragment();
    }

    // Final alpha
    alpha *= in.color.a;

    // Weighted Blended OIT weight function
    // Using depth-based weighting: w = alpha * max(10^-2, min(3*10^3, 10 / (eps + |z|)))
    float z = in.depth;
    float weight = max(1e-2, min(3e3, 10.0 / (1e-5 + abs(z))));
    weight *= alpha;

    // Output for accumulation buffer: premultiplied RGB with weight
    float3 weightedColor = in.color.rgb * alpha * weight;

    FragmentOut out;
    out.accumulation = float4(weightedColor, alpha * weight);
    out.revealage = alpha;  // Will be multiplied as (1 - alpha) in blend mode

    return out;
}

// ============================================================================
// Composite Pass - Blend OIT buffers with background
// ============================================================================

struct CompositeVertexOut {
    float4 position [[position]];
    float2 texCoord;
};

// Fullscreen quad vertices
vertex CompositeVertexOut composite_vertex(uint vertexID [[vertex_id]]) {
    // Generate fullscreen triangle
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2( 3.0, -1.0),
        float2(-1.0,  3.0)
    };

    float2 texCoords[3] = {
        float2(0.0, 1.0),
        float2(2.0, 1.0),
        float2(0.0, -1.0)
    };

    CompositeVertexOut out;
    out.position = float4(positions[vertexID], 0.0, 1.0);
    out.texCoord = texCoords[vertexID];

    return out;
}

fragment float4 composite_fragment(
    CompositeVertexOut in [[stage_in]],
    texture2d<float> accumulationTexture [[texture(0)]],
    texture2d<float> revealageTexture [[texture(1)]]
) {
    constexpr sampler texSampler(mag_filter::nearest, min_filter::nearest);

    // Sample OIT buffers
    float4 accum = accumulationTexture.sample(texSampler, in.texCoord);
    float reveal = revealageTexture.sample(texSampler, in.texCoord).r;

    // Compute final composite
    // revealage = product of (1 - alpha), so transmittance = revealage
    float transmittance = reveal;

    // Prevent division by zero
    if (accum.a >= 1e-5) {
        accum.rgb /= accum.a;
    }

    // Composite: C = accum.rgb + background * transmittance
    // For now, assume black background
    float3 finalColor = accum.rgb;

    return float4(finalColor, 1.0 - transmittance);
}
