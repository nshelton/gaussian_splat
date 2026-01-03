#pragma once

#include <simd/simd.h>
#include <vector>
#include <cstdint>

// Basic vertex structure
struct Vertex {
    float position[3];
    uint8_t color[4];  // RGBA as uint8 [0-255]
};

// Primitive types for rendering
enum class PrimitiveType {
    Triangle,
    Line,
    Point
};

// Base renderable object
class Renderable {
public:
    virtual ~Renderable() = default;

    virtual const std::vector<Vertex>& getVertices() const = 0;
    virtual PrimitiveType getPrimitiveType() const = 0;
    virtual simd_float4x4 getModelMatrix() const { return matrix_identity_float4x4; }
};

// Grid renderable
class Grid : public Renderable {
public:
    Grid(int size = 10, float spacing = 1.0f);

    const std::vector<Vertex>& getVertices() const override { return vertices; }
    PrimitiveType getPrimitiveType() const override { return PrimitiveType::Line; }

private:
    std::vector<Vertex> vertices;
};

// Coordinate axes renderable
class Axes : public Renderable {
public:
    Axes(float length = 2.0f);

    const std::vector<Vertex>& getVertices() const override { return vertices; }
    PrimitiveType getPrimitiveType() const override { return PrimitiveType::Line; }

private:
    std::vector<Vertex> vertices;
};

// Triangle mesh renderable
class TriangleMesh : public Renderable {
public:
    TriangleMesh(const std::vector<Vertex>& verts);

    const std::vector<Vertex>& getVertices() const override { return vertices; }
    PrimitiveType getPrimitiveType() const override { return PrimitiveType::Triangle; }

    void setModelMatrix(const simd_float4x4& matrix) { modelMatrix = matrix; }
    simd_float4x4 getModelMatrix() const override { return modelMatrix; }

private:
    std::vector<Vertex> vertices;
    simd_float4x4 modelMatrix;
};

// Point cloud renderable
class PointCloud : public Renderable {
public:
    PointCloud(int numPoints, float radius = 5.0f);

    const std::vector<Vertex>& getVertices() const override { return vertices; }
    PrimitiveType getPrimitiveType() const override { return PrimitiveType::Point; }

private:
    std::vector<Vertex> vertices;
};

// Gaussian splat renderable (loaded from PLY file)
class GaussianSplat : public Renderable {
public:
    GaussianSplat(const std::string& filepath);

    const std::vector<Vertex>& getVertices() const override { return vertices; }
    PrimitiveType getPrimitiveType() const override { return PrimitiveType::Point; }

    bool isLoaded() const { return loaded; }
    int getPointCount() const { return static_cast<int>(vertices.size()); }

private:
    std::vector<Vertex> vertices;
    bool loaded;
};
