// SwiftSim Physics Test
// Tests the extracted quadrotor physics model

#include "core/quadrotor_physics.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cmath>

using namespace swiftsim;

// ============================================================================
// TEST 1: Hover Test
// ============================================================================
void test_hover() {
    std::cout << "\n=== TEST 1: HOVER ===\n";

    QuadrotorParams params;
    QuadrotorPhysics physics(params);
    DroneState state;

    // Start at 10m altitude (NED: -10 is 10m up)
    state.pos_z = -10.0f;

    // Set hover throttle
    float hover = physics.getHoverThrottle();
    std::cout << "Hover throttle: " << hover * 100 << "%\n";

    physics.setMotorInputs(state, hover, hover, hover, hover);

    // Simulate 5 seconds
    float dt = 0.001f; // 1000 Hz
    int steps = 5000;

    std::cout << "\nSimulating hover for 5 seconds...\n";
    std::cout << std::setw(8) << "Time" << std::setw(12) << "Z (m)"
              << std::setw(12) << "Vz (m/s)" << "\n";

    for (int i = 0; i <= steps; ++i) {
        if (i % 1000 == 0) {
            std::cout << std::fixed << std::setprecision(3)
                      << std::setw(8) << i * dt
                      << std::setw(12) << -state.pos_z  // Convert to altitude
                      << std::setw(12) << -state.vel_z << "\n";
        }
        physics.step(state, dt);
    }

    float final_altitude = -state.pos_z;
    bool success = std::abs(final_altitude - 10.0f) < 0.5f;
    std::cout << "\nResult: " << (success ? "PASS" : "FAIL")
              << " (altitude drift: " << std::abs(final_altitude - 10.0f) << "m)\n";
}

// ============================================================================
// TEST 2: Free Fall
// ============================================================================
void test_freefall() {
    std::cout << "\n=== TEST 2: FREE FALL ===\n";

    QuadrotorPhysics physics;
    DroneState state;

    // Start at 50m altitude
    state.pos_z = -50.0f;

    // No throttle
    physics.setMotorInputs(state, 0, 0, 0, 0);

    float dt = 0.001f;
    int steps = 0;
    float max_steps = 10000;

    std::cout << "Dropping from 50m with zero throttle...\n";

    while (!state.is_grounded && steps < max_steps) {
        physics.step(state, dt);
        steps++;
    }

    float fall_time = steps * dt;
    float theoretical_time = std::sqrt(2 * 50.0f / GRAVITY); // t = sqrt(2h/g)

    std::cout << "Hit ground after: " << fall_time << "s\n";
    std::cout << "Theoretical (vacuum): " << theoretical_time << "s\n";

    bool success = std::abs(fall_time - theoretical_time) < 0.5f;
    std::cout << "Result: " << (success ? "PASS" : "FAIL") << "\n";
}

// ============================================================================
// TEST 3: Takeoff
// ============================================================================
void test_takeoff() {
    std::cout << "\n=== TEST 3: TAKEOFF ===\n";

    QuadrotorPhysics physics;
    DroneState state;

    // Start on ground
    state.pos_z = 0.0f;
    state.is_grounded = true;

    // Full throttle
    physics.setMotorInputs(state, 0.8f, 0.8f, 0.8f, 0.8f);

    float dt = 0.001f;
    int steps = 3000; // 3 seconds

    std::cout << "Taking off with 80% throttle...\n";
    std::cout << std::setw(8) << "Time" << std::setw(12) << "Alt (m)"
              << std::setw(12) << "Vz (m/s)" << "\n";

    for (int i = 0; i <= steps; ++i) {
        if (i % 500 == 0) {
            std::cout << std::fixed << std::setprecision(3)
                      << std::setw(8) << i * dt
                      << std::setw(12) << -state.pos_z
                      << std::setw(12) << -state.vel_z << "\n";
        }
        physics.step(state, dt);
    }

    bool success = -state.pos_z > 5.0f; // Should reach > 5m altitude
    std::cout << "\nResult: " << (success ? "PASS" : "FAIL")
              << " (reached " << -state.pos_z << "m altitude)\n";
}

// ============================================================================
// TEST 4: Yaw Rotation
// ============================================================================
void test_yaw() {
    std::cout << "\n=== TEST 4: YAW ROTATION ===\n";

    QuadrotorPhysics physics;
    DroneState state;

    state.pos_z = -10.0f; // Start at 10m

    // Hover with yaw torque (motors 0,2 faster than 1,3)
    float hover = physics.getHoverThrottle();
    float delta = 0.1f;
    physics.setMotorInputs(state,
                          hover + delta,  // motor 0 (CW)
                          hover - delta,  // motor 1 (CCW)
                          hover + delta,  // motor 2 (CW)
                          hover - delta); // motor 3 (CCW)

    float dt = 0.001f;
    int steps = 2000;

    std::cout << "Applying yaw torque for 2 seconds...\n";

    for (int i = 0; i < steps; ++i) {
        physics.step(state, dt);
    }

    // Check yaw rate
    std::cout << "Final angular velocity (yaw): " << state.omega_z << " rad/s\n";
    std::cout << "Final angular velocity (yaw): " << state.omega_z * 180.0f / PI << " deg/s\n";

    bool success = std::abs(state.omega_z) > 0.5f;
    std::cout << "Result: " << (success ? "PASS" : "FAIL") << "\n";
}

// ============================================================================
// BENCHMARK: Performance Test
// ============================================================================
void benchmark() {
    std::cout << "\n=== BENCHMARK: SINGLE DRONE PERFORMANCE ===\n";

    QuadrotorPhysics physics;
    DroneState state;
    state.pos_z = -10.0f;

    float hover = physics.getHoverThrottle();
    physics.setMotorInputs(state, hover, hover, hover, hover);

    float dt = 0.001f;
    int steps = 1000000; // 1 million steps

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < steps; ++i) {
        physics.step(state, dt);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double seconds = duration.count() / 1e6;
    double steps_per_sec = steps / seconds;
    double sim_time = steps * dt;

    std::cout << "Steps: " << steps << "\n";
    std::cout << "Wall time: " << seconds << " s\n";
    std::cout << "Simulated time: " << sim_time << " s\n";
    std::cout << "Steps/second: " << std::fixed << std::setprecision(0) << steps_per_sec << "\n";
    std::cout << "Real-time factor: " << sim_time / seconds << "x\n";
}

// ============================================================================
// BENCHMARK: Multi-Drone (AoS baseline)
// ============================================================================
void benchmark_multi_drone() {
    std::cout << "\n=== BENCHMARK: MULTI-DRONE (AoS BASELINE) ===\n";

    std::vector<int> drone_counts = {10, 100, 1000, 10000};

    for (int n_drones : drone_counts) {
        QuadrotorPhysics physics;
        std::vector<DroneState> drones(n_drones);

        float hover = physics.getHoverThrottle();
        for (auto& drone : drones) {
            drone.pos_z = -10.0f;
            physics.setMotorInputs(drone, hover, hover, hover, hover);
        }

        float dt = 0.001f;
        int steps = 1000; // 1 second of simulation

        auto start = std::chrono::high_resolution_clock::now();

        for (int s = 0; s < steps; ++s) {
            for (auto& drone : drones) {
                physics.step(drone, dt);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        double seconds = duration.count() / 1e6;
        double drone_steps_per_sec = (n_drones * steps) / seconds;

        std::cout << std::setw(6) << n_drones << " drones: "
                  << std::fixed << std::setprecision(2) << seconds * 1000 << " ms, "
                  << std::setprecision(0) << drone_steps_per_sec << " drone-steps/s\n";
    }

    std::cout << "\n(Note: SoA + SIMD version will be 5-10x faster)\n";
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           SWIFTSIM - Quadrotor Physics Test                  ║\n";
    std::cout << "║                                                              ║\n";
    std::cout << "║  Extracted from AirSim (MIT License)                         ║\n";
    std::cout << "║  Modernized for high-performance RL training                 ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";

    test_hover();
    test_freefall();
    test_takeoff();
    test_yaw();
    benchmark();
    benchmark_multi_drone();

    std::cout << "\n✓ All tests completed!\n";
    return 0;
}
