// SwiftSim Raylib + ImGui Visualizer
// Real-time 3D visualization of drone swarm training

#pragma once

#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"

// ImGui integration
#include "imgui.h"
#include "rlImGui.h"

#include <vector>
#include <string>
#include <cmath>

namespace swiftsim {
namespace ui {

// ============================================================================
// DRONE MESH
// ============================================================================
class DroneMesh {
public:
    Model body;
    Model propeller;
    bool loaded = false;

    void init() {
        // Create simple drone mesh (body + 4 propellers)
        body = LoadModelFromMesh(GenMeshCube(0.3f, 0.1f, 0.3f));
        propeller = LoadModelFromMesh(GenMeshCylinder(0.1f, 0.02f, 8));
        loaded = true;
    }

    void draw(Vector3 pos, Quaternion quat, Color color) {
        if (!loaded) init();

        // Convert quaternion to rotation matrix
        Matrix rot = QuaternionToMatrix(quat);

        // Draw body
        DrawModelEx(body, pos, {0, 1, 0}, 0, {1, 1, 1}, color);

        // Draw propellers at corners
        float offset = 0.16f;
        Vector3 props[4] = {
            {offset, 0.05f, offset},
            {offset, 0.05f, -offset},
            {-offset, 0.05f, -offset},
            {-offset, 0.05f, offset}
        };

        for (int i = 0; i < 4; i++) {
            Vector3 prop_pos = Vector3Add(pos, Vector3Transform(props[i], rot));
            DrawModel(propeller, prop_pos, 1.0f, DARKGRAY);
        }
    }

    void unload() {
        if (loaded) {
            UnloadModel(body);
            UnloadModel(propeller);
            loaded = false;
        }
    }
};

// ============================================================================
// TRAINING STATS
// ============================================================================
struct TrainingStats {
    std::vector<float> rewards_history;
    std::vector<float> pg_loss_history;
    std::vector<float> vf_loss_history;
    std::vector<float> entropy_history;

    float current_reward = 0.0f;
    float mean_reward = 0.0f;
    float pg_loss = 0.0f;
    float vf_loss = 0.0f;
    float entropy = 0.0f;

    int total_steps = 0;
    int episodes = 0;
    float fps = 0.0f;
    float physics_rate = 0.0f;

    void add_reward(float r) {
        current_reward = r;
        rewards_history.push_back(r);
        if (rewards_history.size() > 1000) {
            rewards_history.erase(rewards_history.begin());
        }

        // Compute mean
        float sum = 0;
        for (float v : rewards_history) sum += v;
        mean_reward = sum / rewards_history.size();
    }

    void add_losses(float pg, float vf, float ent) {
        pg_loss = pg;
        vf_loss = vf;
        entropy = ent;

        pg_loss_history.push_back(pg);
        vf_loss_history.push_back(vf);
        entropy_history.push_back(ent);

        if (pg_loss_history.size() > 500) {
            pg_loss_history.erase(pg_loss_history.begin());
            vf_loss_history.erase(vf_loss_history.begin());
            entropy_history.erase(entropy_history.begin());
        }
    }
};

// ============================================================================
// MAIN VISUALIZER
// ============================================================================
class SwiftSimVisualizer {
public:
    // Window settings
    int screen_width = 1600;
    int screen_height = 900;
    bool running = false;

    // Camera
    Camera3D camera = {0};

    // Drone rendering
    DroneMesh drone_mesh;
    std::vector<Color> drone_colors;

    // Training stats
    TrainingStats stats;

    // UI state
    bool show_stats = true;
    bool show_settings = true;
    bool paused = false;
    float sim_speed = 1.0f;
    int selected_drone = 0;

    // Settings
    float target_altitude = 10.0f;
    float learning_rate = 3e-4f;
    float clip_range = 0.2f;
    float entropy_coef = 0.01f;

    void init() {
        SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE);
        InitWindow(screen_width, screen_height, "SwiftSim - CUDA RL Training");
        SetTargetFPS(60);

        // Setup camera
        camera.position = {30.0f, 20.0f, 30.0f};
        camera.target = {0.0f, 10.0f, 0.0f};
        camera.up = {0.0f, 1.0f, 0.0f};
        camera.fovy = 45.0f;
        camera.projection = CAMERA_PERSPECTIVE;

        // Init ImGui
        rlImGuiSetup(true);

        // Init drone mesh
        drone_mesh.init();

        // Generate colors for drones
        for (int i = 0; i < 1000; i++) {
            drone_colors.push_back({
                (unsigned char)(50 + (i * 37) % 200),
                (unsigned char)(50 + (i * 73) % 200),
                (unsigned char)(50 + (i * 113) % 200),
                255
            });
        }

        running = true;
    }

    void begin_frame() {
        BeginDrawing();
        ClearBackground(RAYWHITE);

        // Update camera
        UpdateCamera(&camera, CAMERA_ORBITAL);
    }

    void draw_ground() {
        // Draw ground plane
        DrawPlane({0, 0, 0}, {100, 100}, DARKGREEN);

        // Draw grid
        DrawGrid(50, 2.0f);
    }

    void draw_drones(const float* positions, const float* quaternions, int n_drones) {
        BeginMode3D(camera);

        draw_ground();

        // Draw target altitude plane (transparent)
        DrawPlane({0, target_altitude, 0}, {20, 20}, Fade(BLUE, 0.2f));

        // Draw drones
        for (int i = 0; i < n_drones; i++) {
            // Position (convert NED to Y-up)
            Vector3 pos = {
                positions[i * 3 + 0],
                -positions[i * 3 + 2],  // NED Z (down) -> Y (up)
                positions[i * 3 + 1]
            };

            // Quaternion
            Quaternion quat = {
                quaternions[i * 4 + 1],  // x
                quaternions[i * 4 + 3],  // z -> y
                quaternions[i * 4 + 2],  // y -> z
                quaternions[i * 4 + 0]   // w
            };

            // Color based on index
            Color color = drone_colors[i % drone_colors.size()];
            if (i == selected_drone) {
                color = YELLOW;
            }

            drone_mesh.draw(pos, quat, color);
        }

        // Draw axes
        DrawLine3D({0, 0, 0}, {5, 0, 0}, RED);
        DrawLine3D({0, 0, 0}, {0, 5, 0}, GREEN);
        DrawLine3D({0, 0, 0}, {0, 0, 5}, BLUE);

        EndMode3D();
    }

    void draw_imgui() {
        rlImGuiBegin();

        // Stats window
        if (show_stats) {
            ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(300, 400), ImGuiCond_FirstUseEver);

            if (ImGui::Begin("Training Stats", &show_stats)) {
                ImGui::Text("FPS: %.1f", stats.fps);
                ImGui::Text("Physics Rate: %.1f M/s", stats.physics_rate);
                ImGui::Separator();

                ImGui::Text("Steps: %d", stats.total_steps);
                ImGui::Text("Episodes: %d", stats.episodes);
                ImGui::Separator();

                ImGui::Text("Current Reward: %.3f", stats.current_reward);
                ImGui::Text("Mean Reward: %.3f", stats.mean_reward);
                ImGui::Separator();

                ImGui::Text("Policy Loss: %.6f", stats.pg_loss);
                ImGui::Text("Value Loss: %.6f", stats.vf_loss);
                ImGui::Text("Entropy: %.4f", stats.entropy);

                // Reward plot
                if (!stats.rewards_history.empty()) {
                    ImGui::PlotLines("Rewards",
                        stats.rewards_history.data(),
                        (int)stats.rewards_history.size(),
                        0, nullptr, -100, 10, ImVec2(280, 100));
                }

                // Loss plot
                if (!stats.pg_loss_history.empty()) {
                    ImGui::PlotLines("Policy Loss",
                        stats.pg_loss_history.data(),
                        (int)stats.pg_loss_history.size(),
                        0, nullptr, 0, 1, ImVec2(280, 60));
                }
            }
            ImGui::End();
        }

        // Settings window
        if (show_settings) {
            ImGui::SetNextWindowPos(ImVec2(screen_width - 310, 10), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(300, 350), ImGuiCond_FirstUseEver);

            if (ImGui::Begin("Settings", &show_settings)) {
                ImGui::Text("Simulation");
                ImGui::Checkbox("Paused", &paused);
                ImGui::SliderFloat("Speed", &sim_speed, 0.1f, 10.0f);
                ImGui::Separator();

                ImGui::Text("Environment");
                ImGui::SliderFloat("Target Alt", &target_altitude, 1.0f, 50.0f);
                ImGui::Separator();

                ImGui::Text("PPO Hyperparameters");
                ImGui::SliderFloat("Learning Rate", &learning_rate, 1e-5f, 1e-2f, "%.6f", ImGuiSliderFlags_Logarithmic);
                ImGui::SliderFloat("Clip Range", &clip_range, 0.1f, 0.5f);
                ImGui::SliderFloat("Entropy Coef", &entropy_coef, 0.0f, 0.1f);
                ImGui::Separator();

                ImGui::Text("View");
                ImGui::SliderInt("Selected Drone", &selected_drone, 0, 99);

                if (ImGui::Button("Reset Camera")) {
                    camera.position = {30.0f, 20.0f, 30.0f};
                    camera.target = {0.0f, 10.0f, 0.0f};
                }

                if (ImGui::Button("Save Model")) {
                    // Trigger save
                }
                ImGui::SameLine();
                if (ImGui::Button("Load Model")) {
                    // Trigger load
                }
            }
            ImGui::End();
        }

        // Help overlay
        ImGui::SetNextWindowPos(ImVec2(10, screen_height - 80), ImGuiCond_Always);
        ImGui::SetNextWindowBgAlpha(0.5f);
        if (ImGui::Begin("Help", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove)) {
            ImGui::Text("Controls: Mouse - Orbit | Scroll - Zoom | SPACE - Pause");
            ImGui::Text("Press F1 - Stats | F2 - Settings | ESC - Quit");
        }
        ImGui::End();

        rlImGuiEnd();
    }

    void end_frame() {
        // Draw FPS
        DrawFPS(screen_width - 100, 10);

        EndDrawing();

        // Update screen size if resized
        screen_width = GetScreenWidth();
        screen_height = GetScreenHeight();

        // Handle input
        if (IsKeyPressed(KEY_F1)) show_stats = !show_stats;
        if (IsKeyPressed(KEY_F2)) show_settings = !show_settings;
        if (IsKeyPressed(KEY_SPACE)) paused = !paused;
        if (IsKeyPressed(KEY_ESCAPE) || WindowShouldClose()) running = false;

        stats.fps = GetFPS();
    }

    void close() {
        drone_mesh.unload();
        rlImGuiShutdown();
        CloseWindow();
    }

    bool should_continue() const {
        return running;
    }
};

} // namespace ui
} // namespace swiftsim
