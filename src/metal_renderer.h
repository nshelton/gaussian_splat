#pragma once

#include <simd/simd.h>
#include <functional>

#ifdef __OBJC__
@class CAMetalLayer;
#else
typedef void CAMetalLayer;
#endif

// Pure Metal infrastructure - handles frame pacing, command buffers, drawables, ImGui
// No rendering logic - renderers are passed in as callbacks
class MetalRenderer {
public:
    MetalRenderer();
    ~MetalRenderer();

    bool initialize(CAMetalLayer* layer, void* window);
    void resize(int width, int height);

    // Frame management - simplified!
    void beginFrame();
    void endFrame();

    // Get resources for external renderers
    void* getDevice() const;
    void* getCommandBuffer() const;
    void* getDrawableTexture() const;

    // Get GPU timing info (in milliseconds)
    float getLastGpuTime() const;

private:
    class Impl;
    Impl* impl;
};
