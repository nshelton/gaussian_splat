# Gaussian Splat Renderer

Minimal SDL2 + Metal renderer for Gaussian splatting on macOS.

## Setup

### Install dependencies

```bash
brew install sdl2 cmake
```

### Build

```bash
cd gaussian_splat
mkdir build && cd build
cmake ..
make
./gaussian_splat
```

## Current Status

- ✅ SDL2 window with Metal backend
- ✅ Basic Metal rendering pipeline
- ✅ Test triangle rendering
- 🔲 Gaussian splat data structures
- 🔲 Splat rendering shader
- 🔲 Camera controls
- 🔲 ImGui integration

## Next Steps

1. Add basic math library (float3, float4, matrices)
2. Define Gaussian splat data structure
3. Implement splat rendering shader
4. Add camera/view controls
5. Load splat data from file
6. Optimize rendering (sorting, culling, etc.)
