#pragma once

#include <string>
#include <vector>

// Structure to hold full Gaussian splat data from PLY files
struct PointData {
    float x, y, z;           // Position
    float nx, ny, nz;        // Normal
    float r, g, b;           // Color (from f_dc_0, f_dc_1, f_dc_2 or direct RGB)
    float opacity;           // Opacity/alpha
    float scale_x, scale_y, scale_z;     // Scale (scale_0, scale_1, scale_2)
    float rot_0, rot_1, rot_2, rot_3;    // Rotation quaternion (w, x, y, z)

    // Spherical harmonics coefficients (45 f_rest values)
    float sh_rest[45];

    PointData()
        : x(0), y(0), z(0)
        , nx(0), ny(0), nz(0)
        , r(0), g(0), b(0)
        , opacity(1.0f)
        , scale_x(0.01f), scale_y(0.01f), scale_z(0.01f)
        , rot_0(1), rot_1(0), rot_2(0), rot_3(0)
    {
        for (int i = 0; i < 45; i++) sh_rest[i] = 0.0f;
    }
};

class PLYLoader {
public:
    // Load a PLY file and return point cloud data
    static bool load(const std::string& filepath, std::vector<PointData>& points);

private:
    struct PropertyInfo {
        std::string name;
        std::string type;
    };

    static bool parseHeader(std::istream& file, int& vertexCount,
                           std::vector<PropertyInfo>& properties,
                           bool& isBinary);
};
