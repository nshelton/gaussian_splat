cmake_minimum_required(VERSION 3.20)
project(GaussianSplat)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fetch ImGui
include(FetchContent)
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.90.1
)
FetchContent_MakeAvailable(imgui)

# Find SDL2
find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

# Metal framework
find_library(METAL_LIBRARY Metal)
find_library(QUARTZCORE_LIBRARY QuartzCore)
find_library(FOUNDATION_LIBRARY Foundation)
find_library(CORESERVICES_LIBRARY CoreServices)

# ImGui sources
set(IMGUI_DIR ${imgui_SOURCE_DIR})
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdl2.cpp
    ${IMGUI_DIR}/backends/imgui_impl_metal.mm
)

# Source files
set(SOURCES
    src/main.mm
    src/metal_renderer.mm
    src/simple_geometry_renderer.mm
    src/instanced_splat_renderer.mm
    src/trackball_camera.mm
    src/renderable.cpp
    src/ply_loader.cpp
    ${IMGUI_SOURCES}
)

add_executable(gaussian_splat MACOSX_BUNDLE ${SOURCES})

# Set Info.plist for Metal GPU capture
set_target_properties(gaussian_splat PROPERTIES
    MACOSX_BUNDLE_INFO_PLIST ${CMAKE_SOURCE_DIR}/Info.plist
)

# Include ImGui headers
target_include_directories(gaussian_splat PRIVATE
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)

target_link_libraries(gaussian_splat
    ${SDL2_LIBRARIES}
    ${METAL_LIBRARY}
    ${QUARTZCORE_LIBRARY}
    ${FOUNDATION_LIBRARY}
    ${CORESERVICES_LIBRARY}  

)

# Copy shaders to build directory for development
file(COPY ${CMAKE_SOURCE_DIR}/shaders DESTINATION ${CMAKE_BINARY_DIR})

# Also include shaders in the macOS bundle resources
set_source_files_properties(
    ${CMAKE_SOURCE_DIR}/shaders/gaussian_splat.metal
    PROPERTIES MACOSX_PACKAGE_LOCATION Resources/shaders
)
target_sources(gaussian_splat PRIVATE ${CMAKE_SOURCE_DIR}/shaders/gaussian_splat.metal)
