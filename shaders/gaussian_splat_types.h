#ifndef GAUSSIAN_SPLAT_TYPES_H
#define GAUSSIAN_SPLAT_TYPES_H

#include <metal_stdlib>
using namespace metal;

// Configuration: 24 fragments per pixel to stay within 64KB tile memory limit
// Per pixel: 24 × 10 bytes + 4 bytes = 244 bytes
// Per tile (16×16): 244 × 256 = 62,464 bytes ≈ 61 KB (fits in M1's 64 KB limit)
#define MAX_FRAGMENTS_PER_PIXEL 24

// Fragment data stored in imageblock
struct Fragment {
    half depth;      // 2 bytes - view-space depth (positive = closer)
    half3 color;     // 6 bytes - premultiplied RGB
    half alpha;      // 2 bytes - opacity
};
// Total: 10 bytes per fragment

// Per-pixel imageblock data
struct ImageBlockData {
    Fragment fragments[MAX_FRAGMENTS_PER_PIXEL];
    uint count;      // Number of fragments written (0 to MAX_FRAGMENTS_PER_PIXEL)
};

#endif
