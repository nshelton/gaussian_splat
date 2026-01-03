#include "renderable.h"
#include "ply_loader.h"
#include <iostream>

Grid::Grid(int size, float spacing) {
    const float halfGrid = size * spacing * 0.5f;
    const uint8_t gridColor[4] = {255, 255, 255, 230};  // White with slight transparency

    // Grid lines parallel to X axis
    for (int i = 0; i <= size; i++) {
        float z = -halfGrid + i * spacing;
        Vertex v1 = {{-halfGrid, 0.0f, z}, {gridColor[0], gridColor[1], gridColor[2], gridColor[3]}};
        Vertex v2 = {{ halfGrid, 0.0f, z}, {gridColor[0], gridColor[1], gridColor[2], gridColor[3]}};
        vertices.push_back(v1);
        vertices.push_back(v2);
    }

    // Grid lines parallel to Z axis
    for (int i = 0; i <= size; i++) {
        float x = -halfGrid + i * spacing;
        Vertex v1 = {{x, 0.0f, -halfGrid}, {gridColor[0], gridColor[1], gridColor[2], gridColor[3]}};
        Vertex v2 = {{x, 0.0f,  halfGrid}, {gridColor[0], gridColor[1], gridColor[2], gridColor[3]}};
        vertices.push_back(v1);
        vertices.push_back(v2);
    }
}

Axes::Axes(float length) {
    // X axis - Red
    vertices.push_back({{0.0f, 0.0f, 0.0f}, {255, 0, 0, 255}});
    vertices.push_back({{length, 0.0f, 0.0f}, {255, 0, 0, 255}});

    // Y axis - Green
    vertices.push_back({{0.0f, 0.0f, 0.0f}, {0, 255, 0, 255}});
    vertices.push_back({{0.0f, length, 0.0f}, {0, 255, 0, 255}});

    // Z axis - Blue
    vertices.push_back({{0.0f, 0.0f, 0.0f}, {0, 0, 255, 255}});
    vertices.push_back({{0.0f, 0.0f, length}, {0, 0, 255, 255}});
}

TriangleMesh::TriangleMesh(const std::vector<Vertex>& verts)
    : vertices(verts)
    , modelMatrix(matrix_identity_float4x4)
{
}

GaussianSplat::GaussianSplat(const std::string& filepath)
    : loaded(false)
{
    std::vector<PointData> points;

    // Try to load as PLY first
    bool success = PLYLoader::load(filepath, points);

    if (!success) {
        std::cerr << "Failed to load Gaussian splat from: " << filepath << std::endl;
        return;
    }

    // Convert PointData to Vertex format
    vertices.reserve(points.size());
    for (const auto& pt : points) {
        Vertex v;
        v.position[0] = pt.x;
        v.position[1] = pt.y;
        v.position[2] = pt.z;
        // Convert float [0-1] to uint8 [0-255]
        v.color[0] = static_cast<uint8_t>(pt.r * 255.0f);
        v.color[1] = static_cast<uint8_t>(pt.g * 255.0f);
        v.color[2] = static_cast<uint8_t>(pt.b * 255.0f);
        v.color[3] = static_cast<uint8_t>(pt.opacity * 255.0f);
        vertices.push_back(v);
    }

    loaded = true;
    std::cout << "GaussianSplat loaded: " << vertices.size() << " points" << std::endl;
}
