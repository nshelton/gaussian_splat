#include "trackball_camera.h"
#include <cmath>
#include <cstdio>

TrackballCamera::TrackballCamera()
    : position(simd_make_float3(0, 0, 5))
    , target(simd_make_float3(0, 0, 0))
    , up(simd_make_float3(0, 1, 0))
    , distance(5.0f)
    , viewportWidth(800)
    , viewportHeight(600)
    , isRotating(false)
    , isPanning(false)
    , lastMousePos(simd_make_float2(0, 0))
    , mouseDownPos(simd_make_float2(0, 0))
{
}

void TrackballCamera::setViewportSize(int width, int height) {
    viewportWidth = width;
    viewportHeight = height;
}

void TrackballCamera::setTarget(simd_float3 newTarget) {
    target = newTarget;
}

void TrackballCamera::setPosition(simd_float3 newPosition) {
    position = newPosition;
    distance = simd_length(position - target);
}

void TrackballCamera::setDistance(float newDistance) {
    distance = simd_clamp(newDistance, minDistance, maxDistance);
    simd_float3 direction = simd_normalize(position - target);
    position = target + direction * distance;
}

void TrackballCamera::handleMouseDown(float x, float y, int button) {
    lastMousePos = simd_make_float2(x, y);
    mouseDownPos = simd_make_float2(x, y);

    if (button == 0) { // Left mouse button - rotate
        isRotating = true;
    } else if (button == 2 || button == 1) { // Right or middle mouse button - pan
        isPanning = true;
    }
}

void TrackballCamera::handleMouseUp() {
    isRotating = false;
    isPanning = false;
}

void TrackballCamera::handleMouseMove(float x, float y) {
    simd_float2 currentPos = simd_make_float2(x, y);
    simd_float2 delta = currentPos - lastMousePos;

    if (isRotating) {
        // Trackball rotation: horizontal drag rotates around world Y, vertical drag rotates around camera right
        float deltaX = delta.x * rotateSpeed * 0.01f;
        float deltaY = delta.y * rotateSpeed * 0.01f;

        // Get camera right vector (perpendicular to view direction and world up)
        simd_float3 viewDir = simd_normalize(target - position);
        simd_float3 right = simd_normalize(simd_cross(viewDir, simd_make_float3(0, 1, 0)));

        // Rotate around camera's right axis (vertical mouse movement)
        simd_quatf rotX = simd_quaternion(-deltaY, right);

        // Rotate around WORLD up axis (horizontal mouse movement)
        simd_quatf rotY = simd_quaternion(-deltaX, simd_make_float3(0, 1, 0));

        // Combine rotations: apply horizontal rotation first (world space), then vertical (camera space)
        simd_quatf rotation = simd_mul(rotY, rotX);

        // Rotate the camera position around the target
        simd_float3 offset = position - target;
        offset = simd_act(rotation, offset);
        position = target + offset;

        // Update the up vector to maintain camera orientation
        up = simd_act(rotation, up);
        up = simd_normalize(up);
    } else if (isPanning) {
        // Pan the camera
        simd_float3 right = simd_normalize(simd_cross(target - position, up));
        simd_float3 upVec = simd_normalize(simd_cross(right, target - position));

        float panX = -delta.x * panSpeed * distance / viewportHeight;
        float panY = delta.y * panSpeed * distance / viewportHeight;

        simd_float3 offset = right * panX + upVec * panY;
        position += offset;
        target += offset;
    }

    lastMousePos = currentPos;
}

void TrackballCamera::handleScroll(float delta) {
    // Zoom in/out by changing distance
    float zoomFactor = pow(0.95f, delta * zoomSpeed);
    setDistance(distance * zoomFactor);
}

simd_float3 TrackballCamera::projectToSphere(float x, float y) {
    // Convert screen coordinates to normalized device coordinates [-1, 1]
    float ndcX = (2.0f * x / viewportWidth) - 1.0f;
    float ndcY = 1.0f - (2.0f * y / viewportHeight); // Flip Y

    // Project onto virtual trackball sphere
    float length = ndcX * ndcX + ndcY * ndcY;
    float z;

    if (length <= 0.5f) {
        // Inside sphere
        z = sqrt(1.0f - length);
    } else {
        // Outside sphere - use hyperbolic sheet
        z = 0.5f / sqrt(length);
    }

    return simd_normalize(simd_make_float3(ndcX, ndcY, z));
}

simd_float4x4 TrackballCamera::getViewMatrix() const {
    return makeLookAt(position, target, up);
}

simd_float4x4 TrackballCamera::getProjectionMatrix() const {
    float aspect = (float)viewportWidth / (float)viewportHeight;
    return makePerspective(fov * M_PI / 180.0f, aspect, nearPlane, farPlane);
}

simd_float4x4 TrackballCamera::makeLookAt(simd_float3 eye, simd_float3 center, simd_float3 up) const {
    simd_float3 f = simd_normalize(center - eye);
    simd_float3 s = simd_normalize(simd_cross(f, up));
    simd_float3 u = simd_cross(s, f);

    simd_float4x4 result = matrix_identity_float4x4;
    result.columns[0] = simd_make_float4(s.x, u.x, -f.x, 0);
    result.columns[1] = simd_make_float4(s.y, u.y, -f.y, 0);
    result.columns[2] = simd_make_float4(s.z, u.z, -f.z, 0);
    result.columns[3] = simd_make_float4(-simd_dot(s, eye), -simd_dot(u, eye), simd_dot(f, eye), 1);

    return result;
}

simd_float4x4 TrackballCamera::makePerspective(float fovRadians, float aspect, float near, float far) const {
    float yScale = 1.0f / tan(fovRadians * 0.5f);
    float xScale = yScale / aspect;
    float zRange = far - near;
    float zScale = -(far + near) / zRange;
    float wzScale = -2.0f * far * near / zRange;

    simd_float4 P = simd_make_float4(xScale, 0, 0, 0);
    simd_float4 Q = simd_make_float4(0, yScale, 0, 0);
    simd_float4 R = simd_make_float4(0, 0, zScale, -1);
    simd_float4 S = simd_make_float4(0, 0, wzScale, 0);

    return simd_matrix(P, Q, R, S);
}

simd_quatf TrackballCamera::rotationBetweenVectors(simd_float3 start, simd_float3 dest) {
    start = simd_normalize(start);
    dest = simd_normalize(dest);

    float cosTheta = simd_dot(start, dest);
    simd_float3 rotationAxis;

    if (cosTheta < -0.999999f) {
        // Vectors are opposite - choose any perpendicular axis
        rotationAxis = simd_cross(simd_make_float3(0, 0, 1), start);
        if (simd_length(rotationAxis) < 0.01f) {
            rotationAxis = simd_cross(simd_make_float3(1, 0, 0), start);
        }
        rotationAxis = simd_normalize(rotationAxis);
        return simd_quaternion(M_PI, rotationAxis);
    }

    rotationAxis = simd_cross(start, dest);

    float s = sqrt((1.0f + cosTheta) * 2.0f);
    float invS = 1.0f / s;

    return simd_quaternion(s * 0.5f, rotationAxis.x * invS, rotationAxis.y * invS, rotationAxis.z * invS);
}
