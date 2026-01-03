#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include "metal_renderer.h"
#include <iostream>
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_metal.h"
#include <SDL2/SDL.h>

class MetalRenderer::Impl {
public:
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    CAMetalLayer* metalLayer;

    // GPU timing
    float lastGpuTimeMs = 0.0f;

    // Cached drawable for frame
    id<CAMetalDrawable> currentDrawable = nil;

    // Current render encoder (for the frame)
    id<MTLRenderCommandEncoder> currentEncoder = nil;

    // Current command buffer (for the frame)
    id<MTLCommandBuffer> currentCommandBuffer = nil;

    // Semaphore for frame pacing (limit 3 frames in flight)
    dispatch_semaphore_t frameSemaphore;
};

MetalRenderer::MetalRenderer() : impl(new Impl()) {
    impl->device = nil;
    impl->commandQueue = nil;
    impl->metalLayer = nil;
}

MetalRenderer::~MetalRenderer() {
    delete impl;
}

bool MetalRenderer::initialize(CAMetalLayer* layer, void* window) {
    impl->metalLayer = layer;
    impl->device = MTLCreateSystemDefaultDevice();

    if (!impl->device) {
        std::cerr << "Metal is not supported on this device" << std::endl;
        return false;
    }

    std::cout << "Metal device: " << [impl->device.name UTF8String] << std::endl;

    impl->commandQueue = [impl->device newCommandQueue];
    impl->metalLayer.device = impl->device;

    // Initialize semaphore to allow 3 frames in flight
    impl->frameSemaphore = dispatch_semaphore_create(3);
    impl->metalLayer.pixelFormat = MTLPixelFormatBGRA8Unorm;

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    ImGui_ImplSDL2_InitForMetal((SDL_Window*)window);
    ImGui_ImplMetal_Init(impl->device);

    std::cout << "MetalRenderer: Infrastructure initialized successfully" << std::endl;
    return true;
}

void MetalRenderer::beginFrame() {
    // Wait for a frame slot to be available (limits to 3 frames in flight)
    dispatch_semaphore_wait(impl->frameSemaphore, DISPATCH_TIME_FOREVER);

    // Get drawable for this frame
    id<CAMetalDrawable> drawable = [impl->metalLayer nextDrawable];
    if (!drawable) {
        dispatch_semaphore_signal(impl->frameSemaphore);
        return;
    }

    impl->currentDrawable = drawable;

    // Create command buffer for this frame
    id<MTLCommandBuffer> commandBuffer = [impl->commandQueue commandBuffer];
    impl->currentCommandBuffer = commandBuffer;

    // Setup ImGui frame
    MTLRenderPassDescriptor* imguiPassDescTemp = [MTLRenderPassDescriptor renderPassDescriptor];
    imguiPassDescTemp.colorAttachments[0].texture = drawable.texture;
    imguiPassDescTemp.colorAttachments[0].loadAction = MTLLoadActionLoad;
    imguiPassDescTemp.colorAttachments[0].storeAction = MTLStoreActionStore;
    ImGui_ImplMetal_NewFrame(imguiPassDescTemp);
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();
}

void MetalRenderer::endFrame() {
    @autoreleasepool {
        id<CAMetalDrawable> drawable = impl->currentDrawable;
        id<MTLCommandBuffer> commandBuffer = impl->currentCommandBuffer;

        if (!drawable || !commandBuffer) return;

        // Render ImGui in a separate render pass
        ImGui::Render();
        MTLRenderPassDescriptor* imguiPassDesc = [MTLRenderPassDescriptor renderPassDescriptor];
        imguiPassDesc.colorAttachments[0].texture = drawable.texture;
        imguiPassDesc.colorAttachments[0].loadAction = MTLLoadActionLoad;  // Keep existing content
        imguiPassDesc.colorAttachments[0].storeAction = MTLStoreActionStore;

        id<MTLRenderCommandEncoder> imguiEncoder = [commandBuffer renderCommandEncoderWithDescriptor:imguiPassDesc];
        ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), commandBuffer, imguiEncoder);
        [imguiEncoder endEncoding];

        [commandBuffer presentDrawable:drawable];

        // Add completion handler to measure GPU time and signal semaphore
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
            CFTimeInterval gpuTime = buffer.GPUEndTime - buffer.GPUStartTime;
            impl->lastGpuTimeMs = gpuTime * 1000.0f; // Convert to milliseconds

            // Signal semaphore to allow next frame
            dispatch_semaphore_signal(impl->frameSemaphore);
        }];

        [commandBuffer commit];

        // Clear cached references for next frame
        impl->currentCommandBuffer = nil;
        impl->currentDrawable = nil;
    }
}

float MetalRenderer::getLastGpuTime() const {
    return impl->lastGpuTimeMs;
}

void* MetalRenderer::getDevice() const {
    return (__bridge void*)impl->device;
}

void* MetalRenderer::getCommandBuffer() const {
    return (__bridge void*)impl->currentCommandBuffer;
}

void* MetalRenderer::getDrawableTexture() const {
    if (impl->currentDrawable) {
        return (__bridge void*)impl->currentDrawable.texture;
    }
    return nullptr;
}

void MetalRenderer::resize(int width, int height) {
    impl->metalLayer.drawableSize = CGSizeMake(width, height);
}
