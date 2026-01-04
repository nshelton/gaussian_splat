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
  id<MTLRenderPipelineState> pipelineState;       // Splat rendering pipeline (MLAB or 50-layer)
  id<MTLRenderPipelineState> resolvePipelineState; // K-Buffer resolve pipeline (MLAB)
  id<MTLComputePipelineState> computeSortPipeline; // Compute sort pipeline (50-layer)
  id<MTLBuffer> instanceBuffer;
  id<MTLBuffer> uniformBuffer;

  // K-Buffer textures (8 total: 6 layers + 2 depth buffers) - for MLAB mode
  id<MTLTexture> kbufferTextures[8];

  // Per-pixel fragment list buffer - for 50-layer mode
  id<MTLBuffer> fragmentListBuffer;
  id<MTLTexture> outputTexture;

  size_t instanceCount = 0;

  // Hot reload support
  NSString *shaderPath = nil;
  NSTimeInterval lastModifiedTime = 0;
  bool shaderNeedsReload = false;

  // Current viewport dimensions
  int currentWidth = 0;
  int currentHeight = 0;

  // Rendering mode
  bool use50LayerMode = true;  // Toggle between MLAB (6 layers) and compute (50 layers)

  bool initializePipeline() {
    NSError *error = nil;

    // Load shader from file (store path for hot reload)
    if (!shaderPath) {
      // For development: watch the SOURCE file, not the build directory copy
      NSString *executablePath = [[NSBundle mainBundle] executablePath];
      NSString *buildDir = [executablePath stringByDeletingLastPathComponent];

      // Choose shader based on mode
      NSString *shaderFilename = use50LayerMode ? @"gaussian_splat_50layer.metal" : @"gaussian_splat.metal";

      // Go up from build/gaussian_splat.app/Contents/MacOS to the source directory
      shaderPath = [buildDir stringByAppendingPathComponent:[NSString stringWithFormat:@"../../../../shaders/%@", shaderFilename]];
      shaderPath = [shaderPath stringByStandardizingPath];

      // Verify the source file exists
      if (![[NSFileManager defaultManager] fileExistsAtPath:shaderPath]) {
        std::cerr << "⚠️  Source shader not found at: " << [shaderPath UTF8String] << std::endl;

        // Fallback to build directory copy
        shaderPath = [[NSBundle mainBundle] pathForResource:[shaderFilename stringByDeletingPathExtension]
                                                     ofType:@"metal"
                                                inDirectory:@"shaders"];
        if (!shaderPath) {
          shaderPath = [buildDir stringByAppendingPathComponent:[NSString stringWithFormat:@"../shaders/%@", shaderFilename]];
        }
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
    id<MTLFunction> fragmentFunc = nil;
    id<MTLFunction> resolveFunc = nil;
    id<MTLFunction> computeSortFunc = nil;

    if (use50LayerMode) {
      fragmentFunc = [library newFunctionWithName:@"fragment_accumulate"];
      computeSortFunc = [library newFunctionWithName:@"compute_sort_composite"];

      if (!vertexFunc || !fragmentFunc || !computeSortFunc) {
        std::cerr << "❌ Failed to find 50-layer shader functions" << std::endl;
        return false;
      }
    } else {
      fragmentFunc = [library newFunctionWithName:@"fragment_main"];
      resolveFunc = [library newFunctionWithName:@"resolve_main"];

      if (!vertexFunc || !fragmentFunc || !resolveFunc) {
        std::cerr << "❌ Failed to find MLAB shader functions" << std::endl;
        return false;
      }
    }

    // Create rendering pipeline
    MTLRenderPipelineDescriptor *splatDesc = [[MTLRenderPipelineDescriptor alloc] init];
    splatDesc.vertexFunction = vertexFunc;
    splatDesc.fragmentFunction = fragmentFunc;

    if (use50LayerMode) {
      // 50-layer mode: fragment shader doesn't write to color attachments
      // It only writes to the fragment list buffer
      splatDesc.label = @"Gaussian Splat 50-Layer Accumulate Pipeline";
      // No color attachments needed - fragment shader writes to buffer
      splatDesc.colorAttachments[0].pixelFormat = MTLPixelFormatInvalid;
      splatDesc.rasterizationEnabled = YES;  // Still need rasterization
    } else {
      // MLAB mode: Configure 8 color attachments for K-Buffer (6 layers + 2 depth)
      splatDesc.label = @"Gaussian Splat K-Buffer Pipeline";
      for (int i = 0; i < 8; i++) {
        splatDesc.colorAttachments[i].pixelFormat = MTLPixelFormatRGBA16Float;
        splatDesc.colorAttachments[i].blendingEnabled = NO;
      }
    }

    pipelineState = [device newRenderPipelineStateWithDescriptor:splatDesc error:&error];

    if (!pipelineState) {
      std::cerr << "❌ Failed to create pipeline state!" << std::endl;
      std::cerr << "   " << [[error localizedDescription] UTF8String] << std::endl;
      return false;
    }

    // Create secondary pipeline based on mode
    if (use50LayerMode) {
      // Create compute pipeline for sorting and compositing
      computeSortPipeline = [device newComputePipelineStateWithFunction:computeSortFunc error:&error];

      if (!computeSortPipeline) {
        std::cerr << "❌ Failed to create compute sort pipeline!" << std::endl;
        std::cerr << "   " << [[error localizedDescription] UTF8String] << std::endl;
        return false;
      }
    } else {
      // Create resolve pipeline (composites K-Buffer to final image)
      MTLRenderPipelineDescriptor *resolveDesc = [[MTLRenderPipelineDescriptor alloc] init];
      resolveDesc.label = @"K-Buffer Resolve Pipeline";
      resolveDesc.vertexFunction = [library newFunctionWithName:@"fullscreentriangle_vertex"];
      resolveDesc.fragmentFunction = resolveFunc;

      // Output attachment (drawable)
      resolveDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
      resolveDesc.colorAttachments[0].blendingEnabled = YES;
      resolveDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
      resolveDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;

      // Input attachments (K-Buffer layers) - attachments 1-6
      for (int i = 1; i <= 6; i++) {
        resolveDesc.colorAttachments[i].pixelFormat = MTLPixelFormatRGBA16Float;
        resolveDesc.colorAttachments[i].blendingEnabled = NO;
      }

      resolvePipelineState = [device newRenderPipelineStateWithDescriptor:resolveDesc error:&error];

      if (!resolvePipelineState) {
        std::cerr << "❌ Failed to create resolve pipeline state!" << std::endl;
        std::cerr << "   " << [[error localizedDescription] UTF8String] << std::endl;
        return false;
      }
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
    // MTLDepthStencilDescriptor *depthDesc = [[MTLDepthStencilDescriptor alloc] init];
    // depthDesc.depthCompareFunction = MTLCompareFunctionGreater;  // Try Greater if Less appears backwards
    // depthDesc.depthWriteEnabled = YES;
    // depthStencilState = [device newDepthStencilStateWithDescriptor:depthDesc];

    return true;
  }

  void ensureKBufferTextures(int width, int height) {
    if (kbufferTextures[0] && currentWidth == width && currentHeight == height) {
      return;  // Textures already exist with correct size
    }

    currentWidth = width;
    currentHeight = height;

    // Create 8 K-Buffer textures (6 layers + 2 depth buffers)
    MTLTextureDescriptor *kbufferDesc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                     width:width
                                    height:height
                                 mipmapped:NO];
    kbufferDesc.usage = MTLTextureUsageRenderTarget;
    kbufferDesc.storageMode = MTLStorageModePrivate;

    for (int i = 0; i < 8; i++) {
      kbufferTextures[i] = [device newTextureWithDescriptor:kbufferDesc];
      [kbufferTextures[i] setLabel:[NSString stringWithFormat:@"K-Buffer Attachment %d", i]];
    }
  }

  void ensureFragmentListBuffer(int width, int height, MTLPixelFormat drawableFormat) {
    if (fragmentListBuffer && currentWidth == width && currentHeight == height &&
        outputTexture && outputTexture.pixelFormat == drawableFormat) {
      return;  // Buffer already exists with correct size and format
    }

    currentWidth = width;
    currentHeight = height;

    // Create per-pixel fragment list buffer
    // Structure: struct PixelFragmentList { atomic_uint count; FragmentNode fragments[50]; }
    // FragmentNode: half3 color (6 bytes) + half visibility (2 bytes) + half depth (2 bytes) = 10 bytes
    // PixelFragmentList: 4 bytes (count) + 50 * 10 bytes (fragments) = 504 bytes per pixel
    size_t bytesPerPixel = 4 + (50 * 10);  // Slightly larger for alignment
    size_t totalPixels = width * height;
    size_t bufferSize = totalPixels * bytesPerPixel;

    fragmentListBuffer = [device newBufferWithLength:bufferSize
                                             options:MTLResourceStorageModePrivate];
    [fragmentListBuffer setLabel:@"Per-Pixel Fragment Lists"];

    // Create output texture for compute shader - match drawable format for blit compatibility
    MTLTextureDescriptor *outputDesc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:drawableFormat
                                     width:width
                                    height:height
                                 mipmapped:NO];
    outputDesc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    outputDesc.storageMode = MTLStorageModePrivate;

    outputTexture = [device newTextureWithDescriptor:outputDesc];
    [outputTexture setLabel:@"50-Layer Output Texture"];

    std::cout << "📊 Created fragment list buffer: " << (bufferSize / 1024 / 1024) << " MB for "
              << width << "x" << height << " pixels" << std::endl;
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
        std::cout << "\n🔄 Shader file modified: " << [shaderPath UTF8String] << std::endl;
        std::cout << "   Old time: " << lastModifiedTime << ", New time: " << modTime << std::endl;
        lastModifiedTime = modTime;
        shaderNeedsReload = true;
      }
    } else if (error) {
      static bool errorPrinted = false;
      if (!errorPrinted) {
        std::cerr << "⚠️  Error checking shader file: " << [[error localizedDescription] UTF8String] << std::endl;
        errorPrinted = true;
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

    // clip to [-r, r]
    float r = 5;
    if ( abs(point.x) < r && abs(point.y)<  r && abs(point.z) < r ) {
        _instances.push_back(inst);
    }  

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

  // Ensure appropriate resources based on mode
  if (impl->use50LayerMode) {
    impl->ensureFragmentListBuffer(viewportWidth, viewportHeight, drawable.pixelFormat);
  } else {
    impl->ensureKBufferTextures(viewportWidth, viewportHeight);
  }

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

  if (impl->use50LayerMode) {
    // ===== 50-LAYER MODE: 3-pass rendering =====

    // PASS 1: Clear fragment list buffer
    id<MTLBlitCommandEncoder> blitEncoder = [cmdBuffer blitCommandEncoder];
    [blitEncoder fillBuffer:impl->fragmentListBuffer range:NSMakeRange(0, impl->fragmentListBuffer.length) value:0];
    [blitEncoder endEncoding];

    // PASS 2: Accumulate fragments (rasterization pass)
    MTLRenderPassDescriptor *accumPass = [MTLRenderPassDescriptor renderPassDescriptor];
    // No color attachments - fragment shader writes to buffer
    accumPass.renderTargetWidth = viewportWidth;
    accumPass.renderTargetHeight = viewportHeight;
    accumPass.defaultRasterSampleCount = 1;  // Required when no color attachments

    id<MTLRenderCommandEncoder> accumEncoder = [cmdBuffer renderCommandEncoderWithDescriptor:accumPass];
    [accumEncoder setLabel:@"50-Layer Fragment Accumulation"];
    [accumEncoder setRenderPipelineState:impl->pipelineState];
    [accumEncoder setCullMode:MTLCullModeNone];

    [accumEncoder setVertexBuffer:impl->instanceBuffer offset:0 atIndex:0];
    [accumEncoder setVertexBuffer:impl->uniformBuffer offset:0 atIndex:1];
    [accumEncoder setFragmentBuffer:impl->fragmentListBuffer offset:0 atIndex:0];
    [accumEncoder setFragmentBuffer:impl->uniformBuffer offset:0 atIndex:1];

    [accumEncoder drawPrimitives:MTLPrimitiveTypeTriangle
                        vertexStart:0
                        vertexCount:6
                      instanceCount:impl->instanceCount];
    [accumEncoder endEncoding];

    // PASS 3: Sort and composite (compute shader)
    id<MTLComputeCommandEncoder> computeEncoder = [cmdBuffer computeCommandEncoder];
    [computeEncoder setLabel:@"50-Layer Sort & Composite"];
    [computeEncoder setComputePipelineState:impl->computeSortPipeline];

    [computeEncoder setBuffer:impl->fragmentListBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:impl->uniformBuffer offset:0 atIndex:1];
    [computeEncoder setTexture:impl->outputTexture atIndex:0];

    MTLSize gridSize = MTLSizeMake(viewportWidth, viewportHeight, 1);
    MTLSize threadgroupSize = MTLSizeMake(8, 8, 1);

    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [computeEncoder endEncoding];

    // PASS 4: Blit output texture to drawable
    id<MTLBlitCommandEncoder> finalBlit = [cmdBuffer blitCommandEncoder];
    [finalBlit copyFromTexture:impl->outputTexture
                   sourceSlice:0
                   sourceLevel:0
                  sourceOrigin:MTLOriginMake(0, 0, 0)
                    sourceSize:MTLSizeMake(viewportWidth, viewportHeight, 1)
                     toTexture:drawable
              destinationSlice:0
              destinationLevel:0
             destinationOrigin:MTLOriginMake(0, 0, 0)];
    [finalBlit endEncoding];

  } else {
    // ===== MLAB MODE: 2-pass rendering =====

    // PASS 1: Render splats to K-Buffer
    MTLRenderPassDescriptor *kbufferPass = [MTLRenderPassDescriptor renderPassDescriptor];

    for (int i = 0; i < 8; i++) {
      kbufferPass.colorAttachments[i].texture = impl->kbufferTextures[i];
      kbufferPass.colorAttachments[i].loadAction = MTLLoadActionClear;
      kbufferPass.colorAttachments[i].storeAction = MTLStoreActionStore;
      kbufferPass.colorAttachments[i].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
    }

    id<MTLRenderCommandEncoder> kbufferEncoder = [cmdBuffer renderCommandEncoderWithDescriptor:kbufferPass];
    [kbufferEncoder setLabel:@"K-Buffer Splat Pass"];
    [kbufferEncoder setRenderPipelineState:impl->pipelineState];
    [kbufferEncoder setCullMode:MTLCullModeNone];

    [kbufferEncoder setVertexBuffer:impl->instanceBuffer offset:0 atIndex:0];
    [kbufferEncoder setVertexBuffer:impl->uniformBuffer offset:0 atIndex:1];

    [kbufferEncoder drawPrimitives:MTLPrimitiveTypeTriangle
                        vertexStart:0
                        vertexCount:6
                      instanceCount:impl->instanceCount];
    [kbufferEncoder endEncoding];

    // PASS 2: Resolve K-Buffer to final framebuffer
    MTLRenderPassDescriptor *resolvePass = [MTLRenderPassDescriptor renderPassDescriptor];
    resolvePass.colorAttachments[0].texture = drawable;
    resolvePass.colorAttachments[0].loadAction = MTLLoadActionLoad;
    resolvePass.colorAttachments[0].storeAction = MTLStoreActionStore;

    for (int i = 0; i < 6; i++) {
      resolvePass.colorAttachments[i + 1].texture = impl->kbufferTextures[i];
      resolvePass.colorAttachments[i + 1].loadAction = MTLLoadActionLoad;
      resolvePass.colorAttachments[i + 1].storeAction = MTLStoreActionDontCare;
    }

    id<MTLRenderCommandEncoder> resolveEncoder = [cmdBuffer renderCommandEncoderWithDescriptor:resolvePass];
    [resolveEncoder setLabel:@"K-Buffer Resolve Pass"];
    [resolveEncoder setRenderPipelineState:impl->resolvePipelineState];

    [resolveEncoder drawPrimitives:MTLPrimitiveTypeTriangle
                        vertexStart:0
                        vertexCount:3];
    [resolveEncoder endEncoding];
  }
}
