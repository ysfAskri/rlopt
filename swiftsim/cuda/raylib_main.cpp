// SwiftSim Raylib Demo - Main (C++ only, no CUDA includes)
#include "physics_wrapper.cuh"
#include "raylib.h"
#include "raymath.h"
#include <vector>
#include <cmath>
#include <iostream>

// Configuration
constexpr int SCREEN_WIDTH = 1280;
constexpr int SCREEN_HEIGHT = 720;
constexpr size_t N_DRONES = 100;
constexpr float TARGET_ALTITUDE = 10.0f;

// Draw a simple drone representation
void DrawDrone(Vector3 pos, float altitude_error, float time) {
    // Body color based on altitude
    Color body_color;
    if (std::abs(altitude_error) < 0.5f) {
        body_color = GREEN;
    } else if (std::abs(altitude_error) < 2.0f) {
        body_color = YELLOW;
    } else {
        body_color = RED;
    }

    // Draw body
    DrawCube(pos, 0.4f, 0.15f, 0.4f, body_color);
    DrawCubeWires(pos, 0.4f, 0.15f, 0.4f, BLACK);

    // Draw rotors (spinning effect)
    float spin = time * 20.0f;
    Vector3 rotor_offsets[4] = {
        {0.25f, 0.08f, 0.25f},
        {0.25f, 0.08f, -0.25f},
        {-0.25f, 0.08f, -0.25f},
        {-0.25f, 0.08f, 0.25f}
    };

    for (int i = 0; i < 4; i++) {
        Vector3 rotor_pos = Vector3Add(pos, rotor_offsets[i]);
        Color rotor_color = ((int)(spin + i * 90) % 180 < 90) ? ORANGE : DARKGRAY;
        DrawSphere(rotor_pos, 0.08f, rotor_color);
    }
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  SwiftSim Raylib + CUDA Demo\n";
    std::cout << "========================================\n\n";

    // Initialize physics
    PhysicsWrapper physics;
    physics.init(N_DRONES);
    physics.reset(TARGET_ALTITUDE);

    std::cout << "Physics initialized with " << N_DRONES << " drones\n";

    // Host buffers
    std::vector<float> h_obs(N_DRONES * 13);
    std::vector<float> h_actions(N_DRONES * 4, 0.58f);
    std::vector<float> h_rewards(N_DRONES);

    // Initialize Raylib
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "SwiftSim CUDA + Raylib Demo");
    SetTargetFPS(60);

    // Camera
    Camera3D camera = { 0 };
    camera.position = { 35.0f, 25.0f, 35.0f };
    camera.target = { 0.0f, TARGET_ALTITUDE, 0.0f };
    camera.up = { 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float sim_time = 0.0f;
    bool paused = false;
    int frame = 0;
    float mean_altitude = TARGET_ALTITUDE;

    std::cout << "\nControls:\n";
    std::cout << "  SPACE - Pause/Resume\n";
    std::cout << "  R     - Reset simulation\n";
    std::cout << "  Mouse - Rotate camera\n\n";

    while (!WindowShouldClose()) {
        // Input
        if (IsKeyPressed(KEY_SPACE)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            physics.reset(TARGET_ALTITUDE);
            sim_time = 0.0f;
        }

        UpdateCamera(&camera, CAMERA_ORBITAL);

        if (!paused) {
            // Vary throttle for interesting motion
            float base = 0.58f;
            float variation = 0.03f * std::sin(sim_time * 2.0f) +
                             0.02f * std::sin(sim_time * 5.0f);

            for (size_t i = 0; i < N_DRONES; i++) {
                float phase = i * 0.1f;
                h_actions[i * 4 + 0] = base + variation + 0.01f * std::sin(sim_time + phase);
                h_actions[i * 4 + 1] = base + variation + 0.01f * std::sin(sim_time + phase + 1.57f);
                h_actions[i * 4 + 2] = base + variation + 0.01f * std::sin(sim_time + phase + 3.14f);
                h_actions[i * 4 + 3] = base + variation + 0.01f * std::sin(sim_time + phase + 4.71f);
            }

            physics.set_actions(h_actions.data());

            // Step physics (10 sub-steps)
            for (int s = 0; s < 10; s++) {
                physics.step(0.001f);
            }

            physics.compute_rewards(-TARGET_ALTITUDE);
            sim_time += 0.016f;
        }

        // Get observations
        physics.get_observations(h_obs.data());
        physics.get_rewards(h_rewards.data());

        // Calculate mean altitude
        mean_altitude = 0.0f;
        for (size_t i = 0; i < N_DRONES; i++) {
            mean_altitude += -h_obs[i * 13 + 2];
        }
        mean_altitude /= N_DRONES;

        // Render
        BeginDrawing();
        ClearBackground(RAYWHITE);

        BeginMode3D(camera);

        // Ground
        DrawPlane({0, 0, 0}, {60, 60}, DARKGREEN);

        // Grid
        for (int i = -30; i <= 30; i += 5) {
            DrawLine3D({(float)i, 0.01f, -30}, {(float)i, 0.01f, 30}, GRAY);
            DrawLine3D({-30, 0.01f, (float)i}, {30, 0.01f, (float)i}, GRAY);
        }

        // Target altitude plane
        DrawPlane({0, TARGET_ALTITUDE, 0}, {40, 40}, Fade(SKYBLUE, 0.2f));

        // Draw drones
        for (size_t i = 0; i < N_DRONES; i++) {
            float* obs = &h_obs[i * 13];
            Vector3 pos = {obs[0], -obs[2], obs[1]};
            float altitude_error = (-obs[2]) - TARGET_ALTITUDE;
            DrawDrone(pos, altitude_error, sim_time);
        }

        EndMode3D();

        // UI Panel
        DrawRectangle(10, 10, 280, 140, Fade(BLACK, 0.7f));
        DrawText("SwiftSim CUDA + Raylib", 20, 20, 20, WHITE);
        DrawText(TextFormat("Drones: %d", (int)N_DRONES), 20, 50, 16, WHITE);
        DrawText(TextFormat("Mean Altitude: %.2f m", mean_altitude), 20, 70, 16, WHITE);
        DrawText(TextFormat("Target: %.1f m", TARGET_ALTITUDE), 20, 90, 16, SKYBLUE);
        DrawText(TextFormat("FPS: %d", GetFPS()), 20, 110, 16, GREEN);
        DrawText(paused ? "[SPACE] Resume" : "[SPACE] Pause", 20, 130, 16, GRAY);

        const char* status = paused ? "PAUSED" : "RUNNING";
        Color status_color = paused ? YELLOW : GREEN;
        DrawText(status, SCREEN_WIDTH - 100, 20, 20, status_color);

        DrawText("[R] Reset  |  Mouse to rotate", 10, SCREEN_HEIGHT - 25, 16, DARKGRAY);

        EndDrawing();
        frame++;
    }

    CloseWindow();
    std::cout << "Simulation ended after " << frame << " frames.\n";

    return 0;
}
