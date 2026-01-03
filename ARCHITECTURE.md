# Gaussian Splat Renderer - Architecture

## Overview

Clean, composable architecture where each renderer is **independent** and manages its own Metal render passes.

## Core Design Principles

1. **No callbacks** - Direct, sequential rendering
2. **Independent renderers** - Each creates its own render pass(es)
3. **Shared resources** - Command buffer and drawable texture passed explicitly
4. **Clear separation** - Infrastructure vs. rendering logic

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         main.mm                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  renderer.beginFrame()                                  │ │
│  │    ↓                                                    │ │
│  │  geometryRenderer.render(cmdBuffer, texture, ...)      │ │
│  │    ↓                                                    │ │
│  │  splatRenderer.render(cmdBuffer, texture, ...)         │ │
│  │    ↓                                                    │ │
│  │  renderer.endFrame()                                   │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────┴──────────────────────┐
        │                                            │
┌───────▼────────┐                         ┌────────▼─────────┐
│ MetalRenderer  │                         │   Renderers      │
│  (Infrastructure)                        │  (Rendering Logic)│
├────────────────┤                         ├──────────────────┤
│ • Command queue│                         │ SimpleGeometry   │
│ • Frame pacing │                         │ InstancedSplat   │
│ • ImGui        │                         └──────────────────┘
│ • Present      │
└────────────────┘
```

## Metal Concepts Explained

### Command Buffer
A container for GPU commands. Created once per frame.
```cpp
id<MTLCommandBuffer> = [commandQueue commandBuffer];
```

### Render Pass
A complete rendering operation targeting specific texture(s).
```cpp
MTLRenderPassDescriptor* = describes what textures to render to
id<MTLRenderCommandEncoder> = records drawing commands
```

### Pipeline State
Describes HOW to render (shaders, blend modes, formats).
```cpp
id<MTLRenderPipelineState> = compiled pipeline configuration
```

## Component Responsibilities

### MetalRenderer (Infrastructure Only)
**What it does:**
- Creates command buffers
- Manages frame pacing (semaphore)
- Renders ImGui overlay
- Presents to screen
- Measures GPU timing

**What it provides:**
```cpp
beginFrame()              // Sets up frame resources
getCommandBuffer()        // Returns shared command buffer
getDrawableTexture()      // Returns output target
endFrame()                // ImGui + present
```

### SimpleGeometryRenderer
**What it does:**
- Creates ONE render pass (clears background)
- Renders grid, axes, simple geometry

**API:**
```cpp
render(commandBuffer,     // Shared command buffer
       drawableTexture,   // Output target
       objects,           // What to render
       viewMatrix,        // Camera transform
       projectionMatrix)  // Camera projection
```

**Internally:**
1. Creates render pass descriptor
2. Clears background to dark blue
3. Creates encoder from command buffer
4. Draws geometry
5. Ends encoder

### InstancedSplatRenderer (OIT-based)
**What it does:**
- Creates TWO render passes:
  1. **OIT Accumulation Pass** → Renders to internal buffers
  2. **Composite Pass** → Blends to drawable texture

**API:**
```cpp
render(commandBuffer,     // Shared command buffer
       drawableTexture,   // Output target
       viewMatrix,        // Camera transform
       projectionMatrix,  // Camera projection
       width, height)     // Viewport size
```

**Internally:**
1. **Pass 1:** Render splats to accumulation + revealage textures
2. **Pass 2:** Composite buffers to final drawable (fullscreen quad)

## Weighted Blended OIT

### What is OIT?
**Order-Independent Transparency** - Renders transparent objects in any order and still gets correct blending.

### How does it work?
1. **Accumulation Buffer** (RGBA16Float):
   - Stores weighted sum of colors: `Σ(color * alpha * weight)`

2. **Revealage Buffer** (R16Float):
   - Stores product of (1 - alpha): `Π(1 - alpha)`

3. **Composite Pass**:
   - Divides accumulated color by accumulated weight
   - Uses revealage as transmittance
   - Final = `accum.rgb/accum.a` with `alpha = 1 - revealage`

### Weight Function
```glsl
weight = alpha * max(0.01, min(3000, 10.0 / (eps + |depth|)))
```
Closer objects → higher weight → more influence

## Benefits of New Architecture

### Before (Callback Pattern)
```cpp
renderer.render([&](void* encoder) {
  // What encoder is this?
  // What render pass are we in?
  // Can we create our own pass?
  geometryRenderer.render(encoder, ...);
  splatRenderer.render(???, encoder, ...); // Confusion!
});
```

**Problems:**
- Unclear control flow
- Hard to create multiple render passes
- Tight coupling
- Confusing ownership

### After (Direct Composition)
```cpp
renderer.beginFrame();

geometryRenderer.render(
    renderer.getCommandBuffer(),
    renderer.getDrawableTexture(),
    ...);

splatRenderer.render(
    renderer.getCommandBuffer(),
    renderer.getDrawableTexture(),
    ...);

renderer.endFrame();
```

**Benefits:**
- ✅ Crystal clear flow
- ✅ Each renderer independent
- ✅ Easy to add/remove renderers
- ✅ Simple to understand
- ✅ Easy to profile/debug

## Adding a New Renderer

1. Create your renderer class
2. Implement this signature:
```cpp
void render(void* commandBuffer,
            void* drawableTexture,
            /* your parameters */);
```

3. In the implementation:
```cpp
// Cast parameters
id<MTLCommandBuffer> cmdBuffer = (__bridge id<MTLCommandBuffer>)commandBuffer;
id<MTLTexture> drawable = (__bridge id<MTLTexture>)drawableTexture;

// Create your render pass
MTLRenderPassDescriptor* desc = [MTLRenderPassDescriptor renderPassDescriptor];
desc.colorAttachments[0].texture = drawable;
desc.colorAttachments[0].loadAction = MTLLoadActionLoad; // or Clear
desc.colorAttachments[0].storeAction = MTLStoreActionStore;

// Encode drawing commands
id<MTLRenderCommandEncoder> encoder = [cmdBuffer renderCommandEncoderWithDescriptor:desc];
// ... draw stuff ...
[encoder endEncoding];
```

4. Call it from main.mm:
```cpp
myRenderer.render(
    renderer.getCommandBuffer(),
    renderer.getDrawableTexture(),
    ...);
```

Done! No callbacks, no complexity.

## Performance Notes

- All render passes use the same command buffer → Good for GPU efficiency
- OIT uses 2 passes but NO sorting → Fast for 1000s of overlapping splats
- ImGui renders in separate pass → Doesn't interfere with main rendering
- Frame pacing semaphore → Limits to 3 frames in flight

## File Structure

```
src/
├── main.mm                          # Composition layer
├── metal_renderer.{h,mm}            # Infrastructure only
├── simple_geometry_renderer.{h,mm}  # Grid/axes rendering
├── instanced_splat_renderer.{h,mm}  # Gaussian splat OIT rendering
└── shaders/
    └── gaussian_splat.metal         # OIT shaders
```

## Future Improvements

- [ ] Depth buffer support
- [ ] Sorting pass for splats (depth-based)
- [ ] Multiple viewports
- [ ] Render to texture
- [ ] Post-processing passes

## Summary

**Old way:** Complex callback-based coupling
**New way:** Simple, independent, composable renderers
**Result:** Readable, maintainable, extensible architecture ✨
