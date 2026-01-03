#pragma once

#include <simd/simd.h>

class TrackballCamera {
public:
    TrackballCamera();

    // Camera parameters
    void setViewportSize(int width, int height);
    void setTarget(simd_float3 target);
    void setPosition(simd_float3 position);
    void setDistance(float distance);

    // Input handling (similar to Three.js TrackballControls)
    void handleMouseDown(float x, float y, int button);
    void handleMouseMove(float x, float y);
    void handleMouseUp();
    void handleScroll(float delta);

    // Get matrices
    simd_float4x4 getViewMatrix() const;
    simd_float4x4 getProjectionMatrix() const;
    simd_float3 getPosition() const { return position; }
    simd_float3 getTarget() const { return target; }

    // Camera properties (matching Three.js TrackballControls)
    float rotateSpeed = 1.0f;
    float zoomSpeed = 1.2f;
    float panSpeed = 0.3f;
    float minDistance = 0.1f;
    float maxDistance = 100.0f;

    // Field of view
    float fov = 45.0f;
    float nearPlane = 0.1f;
    float farPlane = 1000.0f;

private:
    // Camera state
    simd_float3 position;
    simd_float3 target;
    simd_float3 up;
    float distance;

    // Viewport
    int viewportWidth;
    int viewportHeight;

    // Mouse state
    bool isRotating;
    bool isPanning;
    simd_float2 lastMousePos;
    simd_float2 mouseDownPos;

    // Trackball helpers
    simd_float3 projectToSphere(float x, float y);
    simd_float3 screenToWorld(float x, float y);

    // Math helpers
    simd_float4x4 makeLookAt(simd_float3 eye, simd_float3 center, simd_float3 up) const;
    simd_float4x4 makePerspective(float fovRadians, float aspect, float near, float far) const;
    simd_quatf rotationBetweenVectors(simd_float3 start, simd_float3 dest);
};
