#pragma once

#include <simd/simd.h>
#include <vector>

struct SplatInstance {
    float rotation[4];  // 12 bytes, no padding
    float scale[3];  // 12 bytes, no padding
    float position[3];  // 12 bytes, no padding
    float color[4];     // 16 bytes
};

class InstancedSplatRenderer {
public:
    InstancedSplatRenderer(std::string filepath);
    ~InstancedSplatRenderer();

    bool initialize(void* device);

    // New simplified API - creates its own OIT render passes
    void render(void* commandBuffer,
                void* drawableTexture,
                const simd_float4x4& viewMatrix,
                const simd_float4x4& projectionMatrix,
                float viewportWidth,
                float viewportHeight);

    int getPointCount() const { return _instances.size(); }

private:
    std::vector<SplatInstance> _instances;
    class Impl;
    Impl* impl;
};
