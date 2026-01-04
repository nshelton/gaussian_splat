#include "ply_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>

// SH coefficient to RGB conversion (simplified - using only DC component)
static float SH_C0 = 0.28209479177387814f;

static void shToRGB(float sh0, float sh1, float sh2, float& r, float& g, float& b) {
    r = 0.5f + SH_C0 * sh0;
    g = 0.5f + SH_C0 * sh1;
    b = 0.5f + SH_C0 * sh2;

    // Clamp to [0, 1]
    r = std::max(0.0f, std::min(1.0f, r));
    g = std::max(0.0f, std::min(1.0f, g));
    b = std::max(0.0f, std::min(1.0f, b));
}

bool PLYLoader::load(const std::string& filepath, std::vector<PointData>& points) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open PLY file: " << filepath << std::endl;
        return false;
    }

    int vertexCount = 0;
    std::vector<PropertyInfo> properties;
    bool isBinary = false;

    auto fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    if (!parseHeader(file, vertexCount, properties, isBinary)) {
        std::cerr << "Failed to parse PLY header" << std::endl;
        return false;
    }

    std::cout << "Loading PLY: " << vertexCount << " vertices, "
              << properties.size() << " properties" << std::endl;
    std::cout << "  Format: " << (isBinary ? "binary" : "ascii") << std::endl;


    const size_t splatSize = sizeof(PointData);

    std::cout << "  File size: " << fileSize << " bytes" << std::endl;
    std::cout << "  Splat size: " << splatSize << " bytes" << std::endl;

    points.clear();
    points.resize(vertexCount); // Pre-allocate all memory

    if (isBinary) {
        // Build property index map once (avoid string comparisons in loop)
        std::vector<int> propIndices(properties.size(), -1);
        for (size_t i = 0; i < properties.size(); i++) {
            const auto& name = properties[i].name;
            if (name == "x") propIndices[i] = 0;
            else if (name == "y") propIndices[i] = 1;
            else if (name == "z") propIndices[i] = 2;
            else if (name == "nx") propIndices[i] = 3;
            else if (name == "ny") propIndices[i] = 4;
            else if (name == "nz") propIndices[i] = 5;
            else if (name == "f_dc_0") propIndices[i] = 6;
            else if (name == "f_dc_1") propIndices[i] = 7;
            else if (name == "f_dc_2") propIndices[i] = 8;
            else if (name == "opacity") propIndices[i] = 9;
            else if (name == "scale_0") propIndices[i] = 10;
            else if (name == "scale_1") propIndices[i] = 11;
            else if (name == "scale_2") propIndices[i] = 12;
            else if (name == "rot_0") propIndices[i] = 13;
            else if (name == "rot_1") propIndices[i] = 14;
            else if (name == "rot_2") propIndices[i] = 15;
            else if (name == "rot_3") propIndices[i] = 16;
            else if (name.find("f_rest_") == 0) {
                int idx = std::stoi(name.substr(7));
                if (idx >= 0 && idx < 45) {
                    propIndices[i] = 100 + idx; // f_rest indices start at 100
                }
            }
        }

        // Calculate stride (bytes per vertex)
        size_t stride = properties.size() * sizeof(float);

        // Read in chunks for better I/O performance
        const int CHUNK_SIZE = 10000;
        std::vector<float> buffer(CHUNK_SIZE * properties.size());

        for (int chunkStart = 0; chunkStart < vertexCount; chunkStart += CHUNK_SIZE) {
            int chunkSize = std::min(CHUNK_SIZE, vertexCount - chunkStart);
            size_t bytesToRead = chunkSize * stride;

            file.read(reinterpret_cast<char*>(buffer.data()), bytesToRead);

            // Process chunk
            for (int i = 0; i < chunkSize; i++) {
                PointData& pt = points[chunkStart + i];
                float* vertex = buffer.data() + (i * properties.size());

                for (size_t j = 0; j < properties.size(); j++) {
                    float value = vertex[j];
                    int idx = propIndices[j];

                    switch(idx) {
                        case 0: pt.x = value; break;
                        case 1: pt.y = value; break;
                        case 2: pt.z = value; break;
                        case 3: pt.nx = value; break;
                        case 4: pt.ny = value; break;
                        case 5: pt.nz = value; break;
                        case 6: pt.r = value; break;
                        case 7: pt.g = value; break;
                        case 8: pt.b = value; break;
                        case 9: pt.opacity = 1.0f / (1.0f + std::exp(-value)); break;
                        case 10: pt.scale_x = std::exp(value); break;
                        case 11: pt.scale_y = std::exp(value); break;
                        case 12: pt.scale_z = std::exp(value); break;
                        case 13: pt.rot_0 = value; break;
                        case 14: pt.rot_1 = value; break;
                        case 15: pt.rot_2 = value; break;
                        case 16: pt.rot_3 = value; break;
                        default:
                            if (idx >= 100 && idx < 145) {
                                pt.sh_rest[idx - 100] = value;
                            }
                            break;
                    }
                }

                // Convert SH to RGB
                if (pt.r != 0 || pt.g != 0 || pt.b != 0) {
                    float r, g, b;
                    shToRGB(pt.r, pt.g, pt.b, r, g, b);
                    pt.r = r;
                    pt.g = g;
                    pt.b = b;
                }
            }

            // Progress indicator for large files
            if (chunkStart % 100000 == 0 && chunkStart > 0) {
                std::cout << "\r  Loading: " << (chunkStart * 100 / vertexCount) << "%" << std::flush;
            }
        }

        if (vertexCount > 100000) {
            std::cout << "\r  Loading: 100%    " << std::endl;
        }
    } else {
        // ASCII format
        std::string line;
        for (int i = 0; i < vertexCount; i++) {
            if (!std::getline(file, line)) break;

            std::istringstream iss(line);
            PointData pt;

            for (const auto& prop : properties) {
                float value = 0.0f;
                iss >> value;

                // Map property to struct field (same as binary)
                if (prop.name == "x") pt.x = value;
                else if (prop.name == "y") pt.y = value;
                else if (prop.name == "z") pt.z = value;
                else if (prop.name == "nx") pt.nx = value;
                else if (prop.name == "ny") pt.ny = value;
                else if (prop.name == "nz") pt.nz = value;
                else if (prop.name == "f_dc_0") pt.r = value;
                else if (prop.name == "f_dc_1") pt.g = value;
                else if (prop.name == "f_dc_2") pt.b = value;
                else if (prop.name.find("f_rest_") == 0) {
                    int idx = std::stoi(prop.name.substr(7));
                    if (idx >= 0 && idx < 45) {
                        pt.sh_rest[idx] = value;
                    }
                }
                else if (prop.name == "opacity") pt.opacity = 1.0f / (1.0f + std::exp(-value));
                else if (prop.name == "scale_0") pt.scale_x = std::exp(value);
                else if (prop.name == "scale_1") pt.scale_y = std::exp(value);
                else if (prop.name == "scale_2") pt.scale_z = std::exp(value);
                else if (prop.name == "rot_0") pt.rot_0 = value;
                else if (prop.name == "rot_1") pt.rot_1 = value;
                else if (prop.name == "rot_2") pt.rot_2 = value;
                else if (prop.name == "rot_3") pt.rot_3 = value;
            }

            // Convert SH to RGB if we have f_dc values
            if (pt.r != 0 || pt.g != 0 || pt.b != 0) {
                float r, g, b;
                shToRGB(pt.r, pt.g, pt.b, r, g, b);
                pt.r = r;
                pt.g = g;
                pt.b = b;
            }

            points.push_back(pt);
        }
    }

    std::cout << "Successfully loaded " << points.size() << " points" << std::endl;
    return !points.empty();
}

bool PLYLoader::parseHeader(std::istream& file, int& vertexCount,
                            std::vector<PropertyInfo>& properties,
                            bool& isBinary) {
    std::string line;

    // Read magic number
    if (!std::getline(file, line) || line != "ply") {
        return false;
    }

    properties.clear();

    // Parse header
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "format") {
            std::string format;
            iss >> format;
            isBinary = (format == "binary_little_endian" || format == "binary_big_endian");
        }
        else if (token == "element") {
            std::string elementType;
            iss >> elementType;
            if (elementType == "vertex") {
                iss >> vertexCount;
            }
        }
        else if (token == "property") {
            PropertyInfo prop;
            iss >> prop.type >> prop.name;
            properties.push_back(prop);
        }
        else if (token == "end_header") {
            break;
        }
    }

    return vertexCount > 0 && !properties.empty();
}
