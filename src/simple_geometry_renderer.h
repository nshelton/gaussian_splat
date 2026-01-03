#pragma once

#include <simd/simd.h>
#include <vector>

class Renderable;

// Renders simple geometry (lines, points, triangles) using basic vertex rendering
class SimpleGeometryRenderer {
public:
    SimpleGeometryRenderer();
    ~SimpleGeometryRenderer();

    bool initialize(void* device);
    void uploadBuffers(const std::vector<Renderable*>& objects);

    // New simplified API - creates its own render pass
    void render(void* commandBuffer,
                void* drawableTexture,
                const std::vector<Renderable*>& objects,
                const simd_float4x4& viewMatrix,
                const simd_float4x4& projectionMatrix);

private:
    class Impl;
    Impl* impl;
};
