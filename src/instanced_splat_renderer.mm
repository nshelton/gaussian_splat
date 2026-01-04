#include "instanced_splat_renderer.h"
#include "ply_loader.h"
#import <Metal/Metal.h>
#include <objc/objc.h>
#include <iostream>
#import <simd/simd.h>
#include <vector>
#import <Foundation/Foundation.h>

class InstancedSplatRenderer::Impl {
public:
  id<MTLDevice> device;
  id<MTLRenderPipelineState> pipelineState;
  id<MTLBuffer> instanceBuffer;
  id<MTLBuffer> uniformBuffer;
  id<MTLTexture> depthTexture;
  id<MTLDepthStencilState> depthStencilState;
  size_t instanceCount = 0;

  // Hot reload support
  NSString *shaderPath = nil;
  NSTimeInterval lastModifiedTime = 0;
  bool shaderNeedsReload = false;

  // Current viewport dimensions for depth texture
  int currentWidth = 0;
  int currentHeight = 0;

  bool initializePipeline() {
    NSError *error = nil;

    // Load shader from file (store path for hot reload)
    if (!shaderPath) {
      shaderPath = [[NSBundle mainBundle] pathForResource:@"gaussian_splat" ofType:@"metal" inDirectory:@"shaders"];

      // If not in bundle, try relative to executable (for development builds)
      if (!shaderPath) {
        NSString *executablePath = [[NSBundle mainBundle] executablePath];
        NSString *buildDir = [executablePath stringByDeletingLastPathComponent];
        shaderPath = [buildDir stringByAppendingPathComponent:@"../shaders/gaussian_splat.metal"];
      }
    }

    NSString *shaderSource = [NSString stringWithContentsOfFile:shaderPath
                                                       encoding:NSUTF8StringEncoding
                                                          error:&error];
    if (!shaderSource) {
      std::cerr << "❌ Failed to load shader file: " << [shaderPath UTF8String] << std::endl;
      if (error) {
        std::cerr << "   Error: " << [[error localizedDescription] UTF8String] << std::endl;
      }
      return false;
    }

    std::cout << "📄 " << (pipelineState ? "Reloading" : "Loading")
              << " shader from: " << [shaderPath UTF8String] << std::endl;

    // Enable shader debugging for Instruments profiling
    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    if (@available(macOS 15.0, *)) {
        options.mathMode = MTLMathModeRelaxed;  // Better for debugging
    } else {
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = NO;
        #pragma clang diagnostic pop
    }

    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource
                                                  options:options
                                                  error:&error];
    if (!library) {
      std::cerr << "❌ Shader compilation failed!" << std::endl;
      std::cerr << "   " << [[error localizedDescription] UTF8String] << std::endl;

      // Print detailed compilation errors if available
      if (error.userInfo) {
        NSString *compilerOutput = error.userInfo[@"MTLLibraryErrorCompilerOutput"];
        if (compilerOutput) {
          std::cerr << "\n📝 Compiler output:\n" << [compilerOutput UTF8String] << std::endl;
        }
      }
      return false;
    }

    id<MTLFunction> vertexFunc = [library newFunctionWithName:@"vertex_main"];
    id<MTLFunction> fragmentFunc = [library newFunctionWithName:@"fragment_main"];

    if (!vertexFunc || !fragmentFunc) {
      std::cerr << "❌ Failed to find shader functions (vertex_main, fragment_main)" << std::endl;
      return false;
    }

    // Create simple pipeline with alpha blending
    MTLRenderPipelineDescriptor *pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDesc.vertexFunction = vertexFunc;
    pipelineDesc.fragmentFunction = fragmentFunc;

    // Single color attachment
    pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    pipelineDesc.colorAttachments[0].blendingEnabled = NO;  // Opaque rendering

    // Single color attachment with standard alpha blending
    // pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    // pipelineDesc.colorAttachments[0].blendingEnabled = YES;
    // pipelineDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    // pipelineDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    // pipelineDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    // pipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    // pipelineDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    // pipelineDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;

    // Enable depth testing for opaque quads
    pipelineDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;

    pipelineState = [device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];

    if (!pipelineState) {
      std::cerr << "❌ Failed to create splat pipeline state!" << std::endl;
      std::cerr << "   " << [[error localizedDescription] UTF8String] << std::endl;
      return false;
    }

    std::cout << "✅ Shader pipeline compiled successfully\n" << std::endl;

    // Create uniform buffer - use the actual struct definition to ensure correct size
    // Metal has strict alignment rules, so we define the struct to match the shader
    struct UniformData {
      simd_float4x4 viewProjectionMatrix;
      simd_float4x4 viewMatrix;
      simd_float4x4 projectionMatrix;
      simd_float2 viewportSize;
    };

    uniformBuffer = [device newBufferWithLength:sizeof(UniformData) options:MTLResourceStorageModeShared];

    // Create depth stencil state for opaque rendering
    MTLDepthStencilDescriptor *depthDesc = [[MTLDepthStencilDescriptor alloc] init];
    depthDesc.depthCompareFunction = MTLCompareFunctionGreater;  // Try Greater if Less appears backwards
    depthDesc.depthWriteEnabled = YES;
    depthStencilState = [device newDepthStencilStateWithDescriptor:depthDesc];

    return true;
  }

  void ensureDepthTexture(int width, int height) {
    if (depthTexture && currentWidth == width && currentHeight == height) {
      return;  // Texture already exists with correct size
    }

    currentWidth = width;
    currentHeight = height;

    // Create depth texture
    MTLTextureDescriptor *depthDesc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                     width:width
                                    height:height
                                 mipmapped:NO];
    depthDesc.usage = MTLTextureUsageRenderTarget;
    depthDesc.storageMode = MTLStorageModePrivate;

    depthTexture = [device newTextureWithDescriptor:depthDesc];
    [depthTexture setLabel:@"Splat Depth Buffer"];
  }


  void initializeFileWatcher() {
    if (!shaderPath) return;

    // Get initial modification time
    NSError *error = nil;
    NSDictionary *attributes = [[NSFileManager defaultManager] attributesOfItemAtPath:shaderPath error:&error];
    if (attributes) {
      lastModifiedTime = [[attributes fileModificationDate] timeIntervalSinceReferenceDate];
      std::cout << "Shader hot-reload enabled - watching: " << [shaderPath UTF8String] << std::endl;
    }
  }

  void checkForShaderChanges() {
    if (!shaderPath) return;

    NSError *error = nil;
    NSDictionary *attributes = [[NSFileManager defaultManager] attributesOfItemAtPath:shaderPath error:&error];
    if (attributes) {
      NSTimeInterval modTime = [[attributes fileModificationDate] timeIntervalSinceReferenceDate];
      if (modTime > lastModifiedTime) {
        lastModifiedTime = modTime;
        shaderNeedsReload = true;
        std::cout << "\n🔄 Shader file modified, will reload on next frame..." << std::endl;
      }
    }
  }

  bool checkAndReloadShader() {
    if (!shaderNeedsReload) return true;

    shaderNeedsReload = false;

    // Save old pipeline state in case reload fails
    id<MTLRenderPipelineState> oldPipeline = pipelineState;

    if (initializePipeline()) {
      std::cout << "✅ Shader hot-reload successful!\n" << std::endl;
      return true;
    } else {
      std::cerr << "⚠️  Shader reload failed - keeping previous working version\n" << std::endl;
      // Restore old pipeline state
      pipelineState = oldPipeline;
      return false;
    }
  }
};

InstancedSplatRenderer::InstancedSplatRenderer(std::string plyPath)
    : impl(new Impl()) {

  std::vector<PointData> points;
  // Try to load as PLY first
  bool success = PLYLoader::load(plyPath, points);

  if (!success) {
    std::cerr << "Failed to load Gaussian splat from: " << plyPath << std::endl;
    return;
  }

  // int total = 50000;
  // if (points.size() > total) {
  //   points.resize(total);
  // }

  std::cout << "Loaded " << points.size() << " points from: " << plyPath
            << std::endl;

  _instances.reserve(points.size());
  for (const auto &point : points) {
    SplatInstance inst;
    // flip y 
    inst.position[0] = point.x;
    inst.position[1] = point.y;
    inst.position[2] = point.z;
  
    inst.color[0] = point.r;
    inst.color[1] = point.g;
    inst.color[2] = point.b;
    inst.color[3] = point.opacity;

    inst.scale[0] = point.scale_x;
    inst.scale[1] = point.scale_y;
    inst.scale[2] = point.scale_z;

    // flip quaternion to match the y flip
    inst.rotation[0] = point.rot_0;
    inst.rotation[1] = point.rot_1;
    inst.rotation[2] = point.rot_2;
    inst.rotation[3] = point.rot_3;

    // normalize quaternion herespo

    _instances.push_back(inst);
  }

  impl->instanceCount = _instances.size();
  std::cout << "InstancedSplatRenderer: loaded " << _instances.size()
            << " instances" << std::endl;
}

InstancedSplatRenderer::~InstancedSplatRenderer() {
  delete impl;
}

bool InstancedSplatRenderer::initialize(void *device) {
  impl->device = (__bridge id<MTLDevice>)device;

  if (!impl->initializePipeline()) {
    return false;
  }

  // Enable shader hot-reload for development
  impl->initializeFileWatcher();

  // Now that we have a device, create the instance buffer
  if (!_instances.empty()) {
    size_t bufferSize = _instances.size() * sizeof(SplatInstance);
    impl->instanceBuffer =
        [impl->device newBufferWithBytes:_instances.data()
                                  length:bufferSize
                                 options:MTLResourceStorageModeShared];

    std::cout << "InstancedSplatRenderer: uploaded " << _instances.size()
              << " instances to GPU" << std::endl;
  }

  return true;
}

void InstancedSplatRenderer::render(void *commandBuffer,
                                    void *drawableTexture,
                                    const simd_float4x4 &viewMatrix,
                                    const simd_float4x4 &projectionMatrix,
                                    float viewportWidth,
                                    float viewportHeight) {
  // Check for shader file changes and hot-reload if needed
  impl->checkForShaderChanges();
  impl->checkAndReloadShader();

  if (impl->instanceCount == 0) {
    return;
  }

  if (!commandBuffer || !drawableTexture) {
    return;  // No valid frame resources
  }

  id<MTLCommandBuffer> cmdBuffer = (__bridge id<MTLCommandBuffer>)commandBuffer;
  id<MTLTexture> drawable = (__bridge id<MTLTexture>)drawableTexture;

  // Ensure depth texture matches viewport size
  impl->ensureDepthTexture(viewportWidth, viewportHeight);

  // Calculate view-projection matrix
  simd_float4x4 viewProjectionMatrix = simd_mul(projectionMatrix, viewMatrix);

  // Update uniforms
  struct UniformData {
    simd_float4x4 viewProjectionMatrix;
    simd_float4x4 viewMatrix;
    simd_float4x4 projectionMatrix;
    simd_float2 viewportSize;
  };

  UniformData uniforms;
  uniforms.viewProjectionMatrix = viewProjectionMatrix;
  uniforms.viewMatrix = viewMatrix;
  uniforms.projectionMatrix = projectionMatrix;
  uniforms.viewportSize = simd_make_float2(viewportWidth, viewportHeight);

  memcpy([impl->uniformBuffer contents], &uniforms, sizeof(UniformData));

  // Render splats directly to drawable with depth testing
  MTLRenderPassDescriptor *passDesc = [MTLRenderPassDescriptor renderPassDescriptor];
  passDesc.colorAttachments[0].texture = drawable;
  passDesc.colorAttachments[0].loadAction = MTLLoadActionLoad;  // Preserve existing content
  passDesc.colorAttachments[0].storeAction = MTLStoreActionStore;

  // Configure depth attachment for opaque rendering
  passDesc.depthAttachment.texture = impl->depthTexture;
  passDesc.depthAttachment.loadAction = MTLLoadActionClear;
  passDesc.depthAttachment.storeAction = MTLStoreActionDontCare;
  passDesc.depthAttachment.clearDepth = 0.0;  // Clear to 0.0 for Greater comparison

  id<MTLRenderCommandEncoder> encoder = [cmdBuffer renderCommandEncoderWithDescriptor:passDesc];
  [encoder setLabel:@"Gaussian Splat Pass"];
  [encoder setRenderPipelineState:impl->pipelineState];
  [encoder setDepthStencilState:impl->depthStencilState];
  [encoder setCullMode:MTLCullModeNone];  // Render both sides of billboards

  // Bind buffers
  [encoder setVertexBuffer:impl->instanceBuffer offset:0 atIndex:0];
  [encoder setVertexBuffer:impl->uniformBuffer offset:0 atIndex:1];

  // Draw instanced quads
  [encoder drawPrimitives:MTLPrimitiveTypeTriangle
                vertexStart:0
                vertexCount:6
              instanceCount:impl->instanceCount];

  [encoder endEncoding];
}
