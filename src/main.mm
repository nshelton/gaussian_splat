#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "instanced_splat_renderer.h"
#include "metal_renderer.h"
#include "renderable.h"
#include "simple_geometry_renderer.h"
#include "trackball_camera.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_metal.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <simd/matrix.h>
#include <simd/types.h>

int main(int argc, char *argv[]) {
  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    std::cerr << "SDL_Init Error: " << SDL_GetError() << std::endl;
    return 1;
  }

  const int WINDOW_WIDTH = 1280;
  const int WINDOW_HEIGHT = 720;

  SDL_Window *window = SDL_CreateWindow(
      "Gaussian Splat Renderer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
      WINDOW_WIDTH, WINDOW_HEIGHT,
      SDL_WINDOW_METAL | SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_RESIZABLE);

  if (!window) {
    std::cerr << "SDL_CreateWindow Error: " << SDL_GetError() << std::endl;
    SDL_Quit();
    return 1;
  }

  // Get the Metal layer from SDL
  CAMetalLayer *metalLayer =
      (__bridge CAMetalLayer *)SDL_Metal_GetLayer(SDL_Metal_CreateView(window));

  // Initialize Metal infrastructure
  MetalRenderer renderer;
  if (!renderer.initialize(metalLayer, window)) {
    std::cerr << "Failed to initialize Metal renderer" << std::endl;
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 1;
  }

  // Set initial drawable size
  int drawableWidth, drawableHeight;
  SDL_Metal_GetDrawableSize(window, &drawableWidth, &drawableHeight);
  renderer.resize(drawableWidth, drawableHeight);

  // Initialize camera
  TrackballCamera camera;
  camera.setViewportSize(drawableWidth, drawableHeight);
  camera.setPosition(simd_make_float3(0, 2, 5));
  camera.setTarget(simd_make_float3(0, 0, 0));

  // Create scene objects
  Grid grid(10, 1.0f);
  Axes axes(2.0f);

  // Initialize specialized renderers
  SimpleGeometryRenderer geometryRenderer;
  geometryRenderer.initialize(renderer.getDevice());

  // Load Gaussian splat from PLY file
  std::string plyPath = "/Users/nshelton/cpp_practice/gaussian_splat/models/"
                        "truck/point_cloud/iteration_30000/point_cloud.ply";
  InstancedSplatRenderer splatRenderer(plyPath);
  splatRenderer.initialize(renderer.getDevice());

  // Collect all renderable objects and upload to geometry renderer
  std::vector<Renderable *> objects = {&grid, &axes};
  geometryRenderer.uploadBuffers(objects);

  std::cout << "Renderer initialized successfully" << std::endl;

  // Main loop

  bool running = true;
  SDL_Event event;

  // Stats tracking
  auto lastTime = std::chrono::high_resolution_clock::now();
  int frameCount = 0;
  float fps = 0.0f;
  float avgFrameTime = 0.0f;

  // Timing history for plots
  const int historySize = 120; // 120 frames of history
  float cpuTimeHistory[120] = {0};
  float gpuTimeHistory[120] = {0};
  int historyOffset = 0;

  while (running) {
    @autoreleasepool {
      auto frameStart = std::chrono::high_resolution_clock::now();

      while (SDL_PollEvent(&event)) {
      ImGui_ImplSDL2_ProcessEvent(&event);

      // Get ImGui IO to check if it wants to capture input
      ImGuiIO &io = ImGui::GetIO();

      switch (event.type) {
      case SDL_QUIT:
        running = false;
        break;

      case SDL_KEYDOWN:
        if (event.key.keysym.sym == SDLK_ESCAPE) {
          running = false;
        }
        break;

      case SDL_WINDOWEVENT:
        if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
          SDL_Metal_GetDrawableSize(window, &drawableWidth, &drawableHeight);
          renderer.resize(drawableWidth, drawableHeight);
          camera.setViewportSize(drawableWidth, drawableHeight);
        }
        break;

      case SDL_MOUSEBUTTONDOWN:
        if (!io.WantCaptureMouse) {
          camera.handleMouseDown(event.button.x, event.button.y,
                                 event.button.button - 1);
        }
        break;

      case SDL_MOUSEBUTTONUP:
        if (!io.WantCaptureMouse) {
          camera.handleMouseUp();
        }
        break;

      case SDL_MOUSEMOTION:
        if (!io.WantCaptureMouse) {
          camera.handleMouseMove(event.motion.x, event.motion.y);
        }
        break;

      case SDL_MOUSEWHEEL:
        if (!io.WantCaptureMouse) {
          camera.handleScroll(event.wheel.y);
        }
        break;
      }
    }

    // Start frame
    renderer.beginFrame();

    // Update timing history
    cpuTimeHistory[historyOffset] = splatRenderer.getCpuSortTimeMs();
    gpuTimeHistory[historyOffset] = renderer.getLastGpuTime();
    historyOffset = (historyOffset + 1) % historySize;

    // Draw ImGui debug window
    ImGui::Begin("Debug Info", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    ImGui::Text("FPS: %.1f (%d x %d)", fps, drawableWidth, drawableHeight);

    ImGui::Separator();

    // CPU timing with plot
    ImGui::Text("CPU: %.2f ms", splatRenderer.getCpuSortTimeMs());
    ImGui::PlotLines("##cpu", cpuTimeHistory, historySize, historyOffset,
                     nullptr, 0.0f, 20.0f, ImVec2(300, 60));

    // GPU timing with plot
    ImGui::Text("GPU: %.2f ms", renderer.getLastGpuTime());
    ImGui::PlotLines("##gpu", gpuTimeHistory, historySize, historyOffset,
                     nullptr, 0.0f, 20.0f, ImVec2(300, 60));

    ImGui::Separator();
    ImGui::Text("Points: %d", splatRenderer.getPointCount());
    auto camPos = camera.getPosition();
    ImGui::Text("Camera: (%.1f, %.1f, %.1f)", camPos.x, camPos.y, camPos.z);
    ImGui::End();

    // Clean, independent rendering - each renderer manages its own passes!
    geometryRenderer.render(
        renderer.getCommandBuffer(),
        renderer.getDrawableTexture(),
        objects,
        camera.getViewMatrix(),
        camera.getProjectionMatrix());

    splatRenderer.render(
        renderer.getCommandBuffer(),
        renderer.getDrawableTexture(),
        camera.getViewMatrix(),
        camera.getProjectionMatrix(),
        drawableWidth,
        drawableHeight);

    // End frame (renders ImGui and presents)
    renderer.endFrame();

    // Calculate FPS and stats
    frameCount++;
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTime)
            .count();

    // Cap at ~60 FPS
    SDL_Delay(16);
    }
  }

  std::cout << std::endl; // New line after stats loop ends

  SDL_DestroyWindow(window);
  SDL_Quit();

  return 0;
}
