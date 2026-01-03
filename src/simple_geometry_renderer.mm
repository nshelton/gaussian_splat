#import <Metal/Metal.h>
#include "simple_geometry_renderer.h"
#include "renderable.h"
#include <iostream>
#include <vector>

class SimpleGeometryRenderer::Impl {
public:
    id<MTLDevice> device;
    id<MTLRenderPipelineState> pipelineState;
    id<MTLBuffer> uniformBuffer;

    // Cached vertex buffers per object
    std::vector<id<MTLBuffer>> vertexBuffers;
    std::vector<size_t> vertexCounts;

    bool initializePipeline() {
        NSError* error = nil;

        // Shader source for basic geometry rendering
        NSString* shaderSource = @R"(
            #include <metal_stdlib>
            using namespace metal;

            struct Vertex {
                packed_float3 position;
                uchar4 color;
            };

            struct Uniforms {
                float4x4 viewProjectionMatrix;
            };

            struct VertexOut {
                float4 position [[position]];
                uchar4 color;
                float pointSize [[point_size]];
            };

            vertex VertexOut vertex_main(uint vertexID [[vertex_id]],
                                         constant Vertex* vertices [[buffer(0)]],
                                         constant Uniforms& uniforms [[buffer(1)]]) {
                VertexOut out;
                Vertex v = vertices[vertexID];
                out.position = uniforms.viewProjectionMatrix * float4(float3(v.position), 1.0);

                out.color = uchar4(v.color);
                out.pointSize = 2.0;
                return out;
            }

            fragment float4 fragment_main(VertexOut in [[stage_in]]) {
                return float4(in.color) / 255.0;
            }
        )";

        id<MTLLibrary> library = [device newLibraryWithSource:shaderSource
                                                      options:nil
                                                        error:&error];
        if (!library) {
            std::cerr << "SimpleGeometryRenderer: Failed to create library: "
                      << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }

        id<MTLFunction> vertexFunc = [library newFunctionWithName:@"vertex_main"];
        id<MTLFunction> fragmentFunc = [library newFunctionWithName:@"fragment_main"];

        MTLRenderPipelineDescriptor* pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
        pipelineDesc.vertexFunction = vertexFunc;
        pipelineDesc.fragmentFunction = fragmentFunc;
        pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;

        pipelineState = [device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
        if (!pipelineState) {
            std::cerr << "SimpleGeometryRenderer: Failed to create pipeline state: "
                      << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }

        // Create uniform buffer for MVP matrix
        uniformBuffer = [device newBufferWithLength:sizeof(simd_float4x4)
                                            options:MTLResourceStorageModeShared];

        std::cout << "SimpleGeometryRenderer: Pipeline initialized successfully" << std::endl;
        return true;
    }
};

SimpleGeometryRenderer::SimpleGeometryRenderer() : impl(new Impl()) {
    impl->device = nil;
    impl->pipelineState = nil;
    impl->uniformBuffer = nil;
}

SimpleGeometryRenderer::~SimpleGeometryRenderer() {
    delete impl;
}

bool SimpleGeometryRenderer::initialize(void* device) {
    impl->device = (__bridge id<MTLDevice>)device;
    return impl->initializePipeline();
}

void SimpleGeometryRenderer::uploadBuffers(const std::vector<Renderable*>& objects) {
    // Clear old buffers
    impl->vertexBuffers.clear();
    impl->vertexCounts.clear();

    // Create GPU buffers for each object
    for (const auto* obj : objects) {
        const auto& vertices = obj->getVertices();

        if (vertices.empty()) {
            impl->vertexBuffers.push_back(nil);
            impl->vertexCounts.push_back(0);
            continue;
        }

        // Upload to GPU once
        id<MTLBuffer> buffer = [impl->device newBufferWithBytes:vertices.data()
                                                         length:vertices.size() * sizeof(Vertex)
                                                        options:MTLResourceStorageModeShared];

        impl->vertexBuffers.push_back(buffer);
        impl->vertexCounts.push_back(vertices.size());
    }

    std::cout << "SimpleGeometryRenderer: Uploaded " << impl->vertexBuffers.size()
              << " buffers to GPU" << std::endl;
}

void SimpleGeometryRenderer::render(void* commandBuffer,
                                    void* drawableTexture,
                                    const std::vector<Renderable*>& objects,
                                    const simd_float4x4& viewMatrix,
                                    const simd_float4x4& projectionMatrix) {
    if (!commandBuffer || !drawableTexture) {
        return;  // No valid frame resources
    }

    id<MTLCommandBuffer> cmdBuffer = (__bridge id<MTLCommandBuffer>)commandBuffer;
    id<MTLTexture> drawable = (__bridge id<MTLTexture>)drawableTexture;

    // Create our own render pass
    MTLRenderPassDescriptor* passDesc = [MTLRenderPassDescriptor renderPassDescriptor];
    passDesc.colorAttachments[0].texture = drawable;
    passDesc.colorAttachments[0].loadAction = MTLLoadActionClear;
    passDesc.colorAttachments[0].clearColor = MTLClearColorMake(0.1, 0.1, 0.15, 1.0);
    passDesc.colorAttachments[0].storeAction = MTLStoreActionStore;

    id<MTLRenderCommandEncoder> encoder = [cmdBuffer renderCommandEncoderWithDescriptor:passDesc];
    [encoder setLabel:@"Simple Geometry Pass"];
    [encoder setRenderPipelineState:impl->pipelineState];

    // Render each object using cached buffers
    for (size_t i = 0; i < objects.size(); i++) {
        if (i >= impl->vertexBuffers.size() || impl->vertexBuffers[i] == nil) continue;

        const auto* obj = objects[i];

        // Calculate MVP matrix (model-view-projection)
        simd_float4x4 modelMatrix = obj->getModelMatrix();
        simd_float4x4 mvpMatrix = simd_mul(projectionMatrix, simd_mul(viewMatrix, modelMatrix));
        memcpy([impl->uniformBuffer contents], &mvpMatrix, sizeof(simd_float4x4));

        // Use cached vertex buffer
        [encoder setVertexBuffer:impl->vertexBuffers[i] offset:0 atIndex:0];
        [encoder setVertexBuffer:impl->uniformBuffer offset:0 atIndex:1];

        // Determine primitive type
        MTLPrimitiveType primitiveType;
        switch (obj->getPrimitiveType()) {
            case PrimitiveType::Triangle:
                primitiveType = MTLPrimitiveTypeTriangle;
                break;
            case PrimitiveType::Line:
                primitiveType = MTLPrimitiveTypeLine;
                break;
            case PrimitiveType::Point:
                primitiveType = MTLPrimitiveTypePoint;
                break;
        }

        [encoder drawPrimitives:primitiveType vertexStart:0 vertexCount:(int)impl->vertexCounts[i]];
    }

    // End our render pass
    [encoder endEncoding];
}
