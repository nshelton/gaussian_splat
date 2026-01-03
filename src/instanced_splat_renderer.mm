#include "instanced_splat_renderer.h"
#include "ply_loader.h"
#import <Metal/Metal.h>
#include <objc/objc.h>
#include <iostream>
#import <simd/simd.h>
#include <vector>

class InstancedSplatRenderer::Impl {
public:
  id<MTLDevice> device;
  id<MTLRenderPipelineState> pipelineState;
  id<MTLRenderPipelineState> compositePipelineState;
  id<MTLBuffer> instanceBuffer;
  id<MTLBuffer> uniformBuffer;
  id<MTLTexture> accumulationTexture;
  id<MTLTexture> revealageTexture;
  size_t instanceCount = 0;
  int currentWidth = 0;
  int currentHeight = 0;

  bool initializePipeline() {
    NSError *error = nil;

    // Load shader from file
    NSString *shaderPath = [[NSBundle mainBundle] pathForResource:@"gaussian_splat" ofType:@"metal" inDirectory:@"shaders"];

    // If not in bundle, try relative to executable (for development builds)
    if (!shaderPath) {
      NSString *executablePath = [[NSBundle mainBundle] executablePath];
      NSString *buildDir = [executablePath stringByDeletingLastPathComponent];
      shaderPath = [buildDir stringByAppendingPathComponent:@"../shaders/gaussian_splat.metal"];
    }

    NSString *shaderSource = [NSString stringWithContentsOfFile:shaderPath
                                                       encoding:NSUTF8StringEncoding
                                                          error:&error];
    if (!shaderSource) {
      std::cerr << "Error: failed to load shader file: " << [shaderPath UTF8String] << std::endl;
      if (error) {
        std::cerr << "  " << [[error localizedDescription] UTF8String] << std::endl;
      }
      return false;
    }

    std::cout << "InstancedSplatRenderer: Loaded shader from: " << [shaderPath UTF8String] << std::endl;

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
      std::cerr << "Error: failed to create shader library: " <<
          [[error localizedDescription] UTF8String] << std::endl;
      return false;
    }

    std::cout << "InstancedSplatRenderer: Shader library created successfully"
              << std::endl;

    id<MTLFunction> vertexFunc = [library newFunctionWithName:@"vertex_main"];
    id<MTLFunction> fragmentFunc =
        [library newFunctionWithName:@"fragment_main"];

    if (!vertexFunc || !fragmentFunc) {
      std::cerr << "Error: failed to find shader functions" << std::endl;
      return false;
    }

    std::cout << "InstancedSplatRenderer: Shader functions found" << std::endl;

    // Create OIT pipeline with dual render targets
    MTLRenderPipelineDescriptor *pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDesc.vertexFunction = vertexFunc;
    pipelineDesc.fragmentFunction = fragmentFunc;

    // Accumulation buffer (color attachment 0)
    pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA16Float;
    pipelineDesc.colorAttachments[0].blendingEnabled = YES;
    pipelineDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
    pipelineDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOne;
    pipelineDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    pipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    pipelineDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOne;
    pipelineDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;

    // Revealage buffer (color attachment 1)
    pipelineDesc.colorAttachments[1].pixelFormat = MTLPixelFormatR16Float;
    pipelineDesc.colorAttachments[1].blendingEnabled = YES;
    pipelineDesc.colorAttachments[1].sourceRGBBlendFactor = MTLBlendFactorZero;
    pipelineDesc.colorAttachments[1].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceColor;
    pipelineDesc.colorAttachments[1].rgbBlendOperation = MTLBlendOperationAdd;

    pipelineState = [device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
  
    if (!pipelineState) {
      std::cerr << "Error: failed to create pipeline state: " <<
          [[error localizedDescription] UTF8String] << std::endl;
      return false;
    }

    // Create composite pipeline
    id<MTLFunction> compositeVertexFunc = [library newFunctionWithName:@"composite_vertex"];
    id<MTLFunction> compositeFragmentFunc = [library newFunctionWithName:@"composite_fragment"];

    if (!compositeVertexFunc || !compositeFragmentFunc) {
      std::cerr << "Error: failed to find composite shader functions" << std::endl;
      return false;
    }

    MTLRenderPipelineDescriptor *compositeDesc = [[MTLRenderPipelineDescriptor alloc] init];
    compositeDesc.vertexFunction = compositeVertexFunc;
    compositeDesc.fragmentFunction = compositeFragmentFunc;
    compositeDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    compositeDesc.colorAttachments[0].blendingEnabled = YES;
    compositeDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    compositeDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    compositeDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;

    compositePipelineState = [device newRenderPipelineStateWithDescriptor:compositeDesc error:&error];

    if (!compositePipelineState) {
      std::cerr << "Error: failed to create composite pipeline state: " <<
          [[error localizedDescription] UTF8String] << std::endl;
      return false;
    }

    std::cout << "InstancedSplatRenderer: Composite pipeline created successfully" << std::endl;

    // Create uniform buffer (viewProjection + view + projection + viewport)
    size_t uniformSize = sizeof(simd_float4x4) * 3 + sizeof(simd_float2);
    uniformBuffer = [device newBufferWithLength:uniformSize options:MTLResourceStorageModeShared];

    return true;
  }

  void ensureTextures(int width, int height) {
    if (currentWidth == width && currentHeight == height && accumulationTexture && revealageTexture) {
      return;  // Textures already created at this size
    }

    currentWidth = width;
    currentHeight = height;

    // Create accumulation texture (RGBA16Float)
    MTLTextureDescriptor *accumDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                          width:width
                                                                                         height:height
                                                                                      mipmapped:NO];
    accumDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    accumDesc.storageMode = MTLStorageModePrivate;
    accumulationTexture = [device newTextureWithDescriptor:accumDesc];

    // Create revealage texture (R16Float)
    MTLTextureDescriptor *revealDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                                                                           width:width
                                                                                          height:height
                                                                                       mipmapped:NO];
    revealDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    revealDesc.storageMode = MTLStorageModePrivate;
    revealageTexture = [device newTextureWithDescriptor:revealDesc];

    std::cout << "InstancedSplatRenderer: Created OIT textures " << width << "x" << height << std::endl;
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

  int total = 10000;
  if (points.size() > total) {
    points.resize(total);
  }

  std::cout << "Loaded " << points.size() << " points from: " << plyPath
            << std::endl;

  _instances.reserve(points.size());
  for (const auto &point : points) {
    SplatInstance inst;
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

    inst.rotation[0] = point.rot_0;
    inst.rotation[1] = point.rot_1;
    inst.rotation[2] = point.rot_2;
    inst.rotation[3] = point.rot_3;

    _instances.push_back(inst);
  }

  impl->instanceCount = _instances.size();
  std::cout << "InstancedSplatRenderer: loaded " << _instances.size()
            << " instances" << std::endl;
}

InstancedSplatRenderer::~InstancedSplatRenderer() { delete impl; }

bool InstancedSplatRenderer::initialize(void *device) {
  impl->device = (__bridge id<MTLDevice>)device;

  if (!impl->initializePipeline()) {
    return false;
  }

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
  if (impl->instanceCount == 0) {
    return;
  }

  if (!commandBuffer || !drawableTexture) {
    return;  // No valid frame resources
  }

  id<MTLCommandBuffer> cmdBuffer = (__bridge id<MTLCommandBuffer>)commandBuffer;
  id<MTLTexture> drawable = (__bridge id<MTLTexture>)drawableTexture;

  // Ensure OIT textures are created at the right size
  impl->ensureTextures((int)viewportWidth, (int)viewportHeight);

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

  // ===========================================================================
  // PASS 1: Render splats to OIT buffers
  // ===========================================================================

  MTLRenderPassDescriptor *oitPassDesc = [MTLRenderPassDescriptor renderPassDescriptor];

  // Accumulation buffer - clear to (0, 0, 0, 0)
  oitPassDesc.colorAttachments[0].texture = impl->accumulationTexture;
  oitPassDesc.colorAttachments[0].loadAction = MTLLoadActionClear;
  oitPassDesc.colorAttachments[0].storeAction = MTLStoreActionStore;
  oitPassDesc.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);

  // Revealage buffer - clear to 1.0 (fully transparent)
  oitPassDesc.colorAttachments[1].texture = impl->revealageTexture;
  oitPassDesc.colorAttachments[1].loadAction = MTLLoadActionClear;
  oitPassDesc.colorAttachments[1].storeAction = MTLStoreActionStore;
  oitPassDesc.colorAttachments[1].clearColor = MTLClearColorMake(1.0, 1.0, 1.0, 1.0);

  id<MTLRenderCommandEncoder> oitEncoder = [cmdBuffer renderCommandEncoderWithDescriptor:oitPassDesc];
  [oitEncoder setLabel:@"OIT Accumulation Pass"];
  [oitEncoder setRenderPipelineState:impl->pipelineState];

  // Bind buffers
  [oitEncoder setVertexBuffer:impl->instanceBuffer offset:0 atIndex:0];
  [oitEncoder setVertexBuffer:impl->uniformBuffer offset:0 atIndex:1];

  // Draw instanced quads
  [oitEncoder drawPrimitives:MTLPrimitiveTypeTriangle
              vertexStart:0
              vertexCount:6
            instanceCount:impl->instanceCount];

  [oitEncoder endEncoding];

  // ===========================================================================
  // PASS 2: Composite OIT buffers to final framebuffer
  // ===========================================================================

  MTLRenderPassDescriptor *finalPassDesc = [MTLRenderPassDescriptor renderPassDescriptor];
  finalPassDesc.colorAttachments[0].texture = drawable;
  finalPassDesc.colorAttachments[0].loadAction = MTLLoadActionLoad;  // Preserve existing content
  finalPassDesc.colorAttachments[0].storeAction = MTLStoreActionStore;

  id<MTLRenderCommandEncoder> compositeEncoder = [cmdBuffer renderCommandEncoderWithDescriptor:finalPassDesc];
  [compositeEncoder setLabel:@"OIT Composite Pass"];
  [compositeEncoder setRenderPipelineState:impl->compositePipelineState];

  // Bind OIT textures
  [compositeEncoder setFragmentTexture:impl->accumulationTexture atIndex:0];
  [compositeEncoder setFragmentTexture:impl->revealageTexture atIndex:1];

  // Draw fullscreen triangle
  [compositeEncoder drawPrimitives:MTLPrimitiveTypeTriangle
                        vertexStart:0
                        vertexCount:3];

  [compositeEncoder endEncoding];
}
