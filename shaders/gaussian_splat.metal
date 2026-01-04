#include <metal_stdlib>
using namespace metal;

// Multi-Layer Alpha Blending (MLAB) configuration
// Based on Apple's OIT sample - uses color attachments with raster order groups
// Metal supports max 8 color attachments
//
// Current implementation: 6 layers
// - 6 attachments for layer RGBA (premultiplied color + visibility)
// - 2 attachments for packed depths (3 depths per RGBA channel)
#define NUM_OIT_LAYERS 6

// K-Buffer: stores sorted layers of transparent fragments
typedef struct {
    half4 layer0 [[color(0)]];  // RGBA: premultiplied color + visibility
    half4 layer1 [[color(1)]];
    half4 layer2 [[color(2)]];
    half4 layer3 [[color(3)]];
    half4 layer4 [[color(4)]];
    half4 layer5 [[color(5)]];
    half4 depths01  [[color(6)]];  // RG: depths for layers 0-1, BA: depths for layers 2-3
    half4 depths23  [[color(7)]];  // RG: depths for layers 4-5, BA: unused
} KBuffer;

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


// Fragment shader - Multi-Layer Alpha Blending with K-Buffer
// This reads AND writes the K-Buffer using raster order groups
fragment KBuffer fragment_main(
    VertexOut in [[stage_in]],
    KBuffer kbuffer  // Input: read current K-Buffer state
) {
    // Evaluate 2D Gaussian
    float2 d = in.uv;

    // Cull splats behind camera
    if (in.depth < 0.001) {
        return kbuffer;  // Return unchanged
    }

    // Canonical Gaussian in sigma space: exp(-0.5 * ||d||^2)
    float gaussianValue = exp(-0.5 * dot(d, d));

    // Discard fragments that contribute very little
    if (gaussianValue < 0.01) {
        return kbuffer;
    }

    // Apply per-splat opacity
    float alpha = gaussianValue * in.color.a;
    half newDepth = half(in.depth);

    // Premultiplied color + visibility (under operator)
    half4 newLayer = half4(half3(in.color.rgb) * half(alpha), 1.0h - half(alpha));

    // Read K-Buffer into array for sorting
    half4 layers[NUM_OIT_LAYERS];
    half depths[NUM_OIT_LAYERS];

    layers[0] = kbuffer.layer0;
    layers[1] = kbuffer.layer1;
    layers[2] = kbuffer.layer2;
    layers[3] = kbuffer.layer3;
    layers[4] = kbuffer.layer4;
    layers[5] = kbuffer.layer5;

    depths[0] = kbuffer.depths01.r;
    depths[1] = kbuffer.depths01.g;
    depths[2] = kbuffer.depths01.b;
    depths[3] = kbuffer.depths01.a;
    depths[4] = kbuffer.depths23.r;
    depths[5] = kbuffer.depths23.g;

    // Insertion sort: insert new fragment in depth order (front to back)
    const short lastLayer = NUM_OIT_LAYERS - 1;
    for (short i = 0; i < NUM_OIT_LAYERS; ++i) {
        half layerDepth = depths[i];
        bool insert = (newDepth >= layerDepth);  // Front-to-back (larger depth = closer)

        if (insert) {
            // Insert here, shift current layer down
            half4 tempLayer = layers[i];
            half tempDepth = depths[i];

            layers[i] = newLayer;
            depths[i] = newDepth;

            newLayer = tempLayer;
            newDepth = tempDepth;
        }
    }

    // Merge last two layers if buffer overflows
    half4 lastLayerColor = layers[lastLayer];
    half lastLayerDepth = depths[lastLayer];

    bool newDepthCloser = (newDepth >= lastLayerDepth);
    half4 frontLayer = newDepthCloser ? newLayer : lastLayerColor;
    half4 backLayer = newDepthCloser ? lastLayerColor : newLayer;

    // Under operator: front UNDER back = back.rgb + front.rgb * back.a
    layers[lastLayer] = half4(backLayer.rgb + frontLayer.rgb * backLayer.a,
                              frontLayer.a * backLayer.a);
    depths[lastLayer] = newDepthCloser ? newDepth : lastLayerDepth;

    // Write K-Buffer back
    KBuffer output;
    output.layer0 = layers[0];
    output.layer1 = layers[1];
    output.layer2 = layers[2];
    output.layer3 = layers[3];
    output.layer4 = layers[4];
    output.layer5 = layers[5];

    output.depths01 = half4(depths[0], depths[1], depths[2], depths[3]);
    output.depths23 = half4(depths[4], depths[5], 0.0h, 0.0h);

    return output;
}

// Fullscreen triangle for resolve pass
struct FullscreenVertexOut {
    float4 position [[position]];
};

vertex FullscreenVertexOut fullscreentriangle_vertex(uint vid [[vertex_id]]) {
    FullscreenVertexOut out;
    switch(vid) {
        case 0:
            out.position = float4(-1, -3, 0, 1);
            break;
        case 1:
            out.position = float4(-1, 1, 0, 1);
            break;
        case 2:
            out.position = float4(3, 1, 0, 1);
            break;
    }
    return out;
}

// Resolve pass: composite K-Buffer layers into final image
// Reads from K-Buffer attachments 1-6, writes to output attachment 0
typedef struct {
    half4 layer0 [[color(1)]];  // K-Buffer starts at attachment 1
    half4 layer1 [[color(2)]];
    half4 layer2 [[color(3)]];
    half4 layer3 [[color(4)]];
    half4 layer4 [[color(5)]];
    half4 layer5 [[color(6)]];
} KBufferNoDepth;

struct ResolveOut {
    half4 color [[color(0)]];  // Output to attachment 0 (drawable)
};

fragment ResolveOut resolve_main(KBufferNoDepth kbuffer) {
    // Composite layers front-to-back using under operator
    half3 finalColor = half3(0.0h);
    half alphaTotal = 1.0h;

    // Layer 0 (closest)
    finalColor += kbuffer.layer0.rgb * alphaTotal;
    alphaTotal *= kbuffer.layer0.a;

    // Layer 1
    finalColor += kbuffer.layer1.rgb * alphaTotal;
    alphaTotal *= kbuffer.layer1.a;

    // Layer 2
    finalColor += kbuffer.layer2.rgb * alphaTotal;
    alphaTotal *= kbuffer.layer2.a;

    // Layer 3
    finalColor += kbuffer.layer3.rgb * alphaTotal;
    alphaTotal *= kbuffer.layer3.a;

    // Layer 4
    finalColor += kbuffer.layer4.rgb * alphaTotal;
    alphaTotal *= kbuffer.layer4.a;

    // Layer 5 (furthest)
    finalColor += kbuffer.layer5.rgb * alphaTotal;
    alphaTotal *= kbuffer.layer5.a;

    ResolveOut out;
    out.color = half4(finalColor, 1.0h - alphaTotal);
    return out;
}

