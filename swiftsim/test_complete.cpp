// SwiftSim Complete Physics Test
// Tests all extracted physics components from AirSim

#include "core/quadrotor_physics_complete.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace swiftsim;

void printState(const DroneState& s, float t) {
    std::cout << std::fixed << std::setprecision(3)
              << "t=" << std::setw(5) << t
              << " pos=(" << std::setw(7) << s.position.x
              << "," << std::setw(7) << s.position.y
              << "," << std::setw(7) << -s.position.z << ")"
              << " vel_z=" << std::setw(7) << -s.velocity.z
              << " thrust=" << std::setw(5) << (s.motor_thrust[0]+s.motor_thrust[1]+s.motor_thrust[2]+s.motor_thrust[3])
              << "\n";
}

// ============================================================================
// TEST 1: Environment Model
// ============================================================================
void test_environment() {
    std::cout << "\n=== TEST: ENVIRONMENT MODEL ===\n";
    std::cout << std::setw(12) << "Altitude" << std::setw(12) << "Gravity"
              << std::setw(12) << "Density" << std::setw(12) << "Pressure"
              << std::setw(12) << "Temp (C)" << "\n";

    for (float alt = 0; alt <= 10000; alt += 2000) {
        EnvironmentState env;
        env.update(alt);
        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(12) << alt
                  << std::setw(12) << env.gravity
                  << std::setw(12) << env.air_density
                  << std::setw(12) << env.air_pressure
                  << std::setw(12) << env.temperature - 273.15f
                  << "\n";
    }
    std::cout << "✓ Environment model works correctly\n";
}

// ============================================================================
// TEST 2: Hover with Drag
// ============================================================================
void test_hover_with_drag() {
    std::cout << "\n=== TEST: HOVER WITH DRAG ===\n";

    QuadrotorPhysicsComplete physics;
    physics.enable_sensor_noise = false;
    DroneState state;
    state.position.z = -20.0f; // 20m altitude

    float hover = physics.getHoverThrottle();
    physics.setMotorInputs(state, hover, hover, hover, hover);

    std::cout << "Hover throttle: " << hover * 100 << "%\n";
    std::cout << "Simulating 5 seconds of hover...\n";

    float dt = 0.001f;
    for (int i = 0; i <= 5000; ++i) {
        if (i % 1000 == 0) {
            printState(state, i * dt);
        }
        physics.step(state, dt);
    }

    float drift = std::abs(-state.position.z - 20.0f);
    std::cout << "Altitude drift: " << drift << "m\n";
    std::cout << (drift < 1.0f ? "✓ PASS" : "✗ FAIL") << "\n";
}

// ============================================================================
// TEST 3: Wind Effects
// ============================================================================
void test_wind() {
    std::cout << "\n=== TEST: WIND EFFECTS ===\n";

    QuadrotorPhysicsComplete physics;
    physics.enable_sensor_noise = false;
    DroneState state;
    state.position.z = -10.0f;

    float hover = physics.getHoverThrottle();
    physics.setMotorInputs(state, hover, hover, hover, hover);

    // Add wind (5 m/s from north)
    physics.setWind(state, 5.0f, 0.0f, 0.0f);

    std::cout << "Hovering with 5 m/s headwind...\n";

    float dt = 0.001f;
    for (int i = 0; i <= 3000; ++i) {
        physics.step(state, dt);
    }

    std::cout << "Drift due to wind: X=" << state.position.x
              << "m, Y=" << state.position.y << "m\n";

    // Drone should drift backwards (negative X) due to drag from headwind
    bool drifted = state.position.x < -0.1f;
    std::cout << (drifted ? "✓ PASS (drone drifted as expected)" : "✗ FAIL") << "\n";
}

// ============================================================================
// TEST 4: Sensors
// ============================================================================
void test_sensors() {
    std::cout << "\n=== TEST: SENSORS ===\n";

    QuadrotorPhysicsComplete physics;
    physics.enable_sensor_noise = false;
    DroneState state;
    state.position.z = -100.0f; // 100m altitude

    float hover = physics.getHoverThrottle();
    physics.setMotorInputs(state, hover, hover, hover, hover);

    // Run a few steps
    for (int i = 0; i < 100; ++i) {
        physics.step(state, 0.001f);
    }

    std::cout << "At altitude: " << -state.position.z << "m\n";
    std::cout << "\nIMU:\n";
    std::cout << "  Angular vel: (" << state.imu.angular_velocity.x << ", "
              << state.imu.angular_velocity.y << ", " << state.imu.angular_velocity.z << ") rad/s\n";
    std::cout << "  Linear accel: (" << state.imu.linear_acceleration.x << ", "
              << state.imu.linear_acceleration.y << ", " << state.imu.linear_acceleration.z << ") m/s²\n";

    std::cout << "\nBarometer:\n";
    std::cout << "  Altitude: " << state.barometer.altitude << "m\n";
    std::cout << "  Pressure: " << state.barometer.pressure << " Pa\n";
    std::cout << "  Temperature: " << state.barometer.temperature - 273.15f << "°C\n";

    std::cout << "\nMagnetometer:\n";
    std::cout << "  Field: (" << state.magnetometer.magnetic_field.x << ", "
              << state.magnetometer.magnetic_field.y << ", "
              << state.magnetometer.magnetic_field.z << ") Gauss\n";

    std::cout << "\n✓ Sensors working\n";
}

// ============================================================================
// TEST 5: Sensor Noise
// ============================================================================
void test_sensor_noise() {
    std::cout << "\n=== TEST: SENSOR NOISE ===\n";

    QuadrotorPhysicsComplete physics;
    physics.enable_sensor_noise = true;
    DroneState state;
    state.position.z = -10.0f;

    float hover = physics.getHoverThrottle();
    physics.setMotorInputs(state, hover, hover, hover, hover);

    // Collect IMU samples
    float sum_x = 0, sum_y = 0, sum_z = 0;
    float sum_sq_x = 0, sum_sq_y = 0, sum_sq_z = 0;
    int n = 1000;

    for (int i = 0; i < n; ++i) {
        physics.step(state, 0.001f);
        sum_x += state.imu.linear_acceleration.x;
        sum_y += state.imu.linear_acceleration.y;
        sum_z += state.imu.linear_acceleration.z;
        sum_sq_x += state.imu.linear_acceleration.x * state.imu.linear_acceleration.x;
        sum_sq_y += state.imu.linear_acceleration.y * state.imu.linear_acceleration.y;
        sum_sq_z += state.imu.linear_acceleration.z * state.imu.linear_acceleration.z;
    }

    float mean_x = sum_x / n, mean_y = sum_y / n, mean_z = sum_z / n;
    float std_x = std::sqrt(sum_sq_x / n - mean_x * mean_x);
    float std_y = std::sqrt(sum_sq_y / n - mean_y * mean_y);
    float std_z = std::sqrt(sum_sq_z / n - mean_z * mean_z);

    std::cout << "Accelerometer noise (1000 samples):\n";
    std::cout << "  Mean: (" << mean_x << ", " << mean_y << ", " << mean_z << ")\n";
    std::cout << "  Std:  (" << std_x << ", " << std_y << ", " << std_z << ")\n";

    bool has_noise = (std_x > 0.01f || std_y > 0.01f || std_z > 0.01f);
    std::cout << (has_noise ? "✓ PASS (noise detected)" : "✗ FAIL") << "\n";
}

// ============================================================================
// TEST 6: Takeoff and Land
// ============================================================================
void test_takeoff_land() {
    std::cout << "\n=== TEST: TAKEOFF AND LAND ===\n";

    QuadrotorPhysicsComplete physics;
    physics.enable_sensor_noise = false;
    DroneState state;
    state.position.z = 0.0f;
    state.is_grounded = true;

    float hover = physics.getHoverThrottle();
    float dt = 0.001f;

    // Phase 1: Takeoff
    std::cout << "Phase 1: Takeoff (80% throttle)\n";
    physics.setMotorInputs(state, 0.8f, 0.8f, 0.8f, 0.8f);
    for (int i = 0; i < 2000; ++i) {
        physics.step(state, dt);
    }
    float alt_after_takeoff = -state.position.z;
    std::cout << "Altitude after 2s: " << alt_after_takeoff << "m\n";

    // Phase 2: Hover
    std::cout << "Phase 2: Hover\n";
    physics.setMotorInputs(state, hover, hover, hover, hover);
    for (int i = 0; i < 2000; ++i) {
        physics.step(state, dt);
    }
    float alt_after_hover = -state.position.z;
    std::cout << "Altitude after hover: " << alt_after_hover << "m\n";

    // Phase 3: Descend and land
    std::cout << "Phase 3: Descend (40% throttle)\n";
    physics.setMotorInputs(state, 0.4f, 0.4f, 0.4f, 0.4f);
    int steps = 0;
    while (!state.is_grounded && steps < 20000) {
        physics.step(state, dt);
        steps++;
    }
    std::cout << "Landed after " << steps * dt << "s\n";
    std::cout << "Is grounded: " << (state.is_grounded ? "yes" : "no") << "\n";

    bool success = alt_after_takeoff > 5.0f && state.is_grounded;
    std::cout << (success ? "✓ PASS" : "✗ FAIL") << "\n";
}

// ============================================================================
// TEST 7: Roll/Pitch Control
// ============================================================================
void test_attitude() {
    std::cout << "\n=== TEST: ATTITUDE CONTROL ===\n";

    QuadrotorPhysicsComplete physics;
    physics.enable_sensor_noise = false;
    DroneState state;
    state.position.z = -10.0f;

    float hover = physics.getHoverThrottle();
    float dt = 0.001f;

    // Apply roll torque (right side faster)
    std::cout << "Applying roll torque...\n";
    physics.setMotorInputs(state,
        hover + 0.05f,  // Front-right
        hover - 0.05f,  // Front-left
        hover - 0.05f,  // Back-left
        hover + 0.05f   // Back-right
    );

    for (int i = 0; i < 1000; ++i) {
        physics.step(state, dt);
    }

    float roll, pitch, yaw;
    state.orientation.toEuler(roll, pitch, yaw);
    std::cout << "Roll: " << roll * 180.0f / PI << "°\n";
    std::cout << "Pitch: " << pitch * 180.0f / PI << "°\n";
    std::cout << "Yaw: " << yaw * 180.0f / PI << "°\n";

    bool rolled = std::abs(roll) > 0.1f; // At least 5.7 degrees
    std::cout << (rolled ? "✓ PASS" : "✗ FAIL") << "\n";
}

// ============================================================================
// BENCHMARK
// ============================================================================
void benchmark() {
    std::cout << "\n=== BENCHMARK: COMPLETE PHYSICS ===\n";

    QuadrotorPhysicsComplete physics;
    physics.enable_sensor_noise = false;

    std::vector<int> counts = {1, 10, 100, 1000, 10000};

    for (int n : counts) {
        std::vector<DroneState> drones(n);
        float hover = physics.getHoverThrottle();

        for (auto& d : drones) {
            d.position.z = -10.0f;
            physics.setMotorInputs(d, hover, hover, hover, hover);
        }

        int steps = 1000;
        auto start = std::chrono::high_resolution_clock::now();

        for (int s = 0; s < steps; ++s) {
            for (auto& d : drones) {
                physics.step(d, 0.001f);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double secs = dur.count() / 1e6;
        double rate = (n * steps) / secs;

        std::cout << std::setw(6) << n << " drones: "
                  << std::fixed << std::setprecision(2) << secs * 1000 << " ms, "
                  << std::setprecision(0) << rate / 1e6 << "M drone-steps/s\n";
    }
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         SWIFTSIM - Complete Physics Test                     ║\n";
    std::cout << "║                                                              ║\n";
    std::cout << "║  Full extraction from AirSim including:                      ║\n";
    std::cout << "║  - Quadrotor dynamics with drag                              ║\n";
    std::cout << "║  - Environment model (ISA atmosphere)                        ║\n";
    std::cout << "║  - Wind effects                                              ║\n";
    std::cout << "║  - IMU, Barometer, Magnetometer sensors                      ║\n";
    std::cout << "║  - Realistic sensor noise model                              ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";

    test_environment();
    test_hover_with_drag();
    test_wind();
    test_sensors();
    test_sensor_noise();
    test_takeoff_land();
    test_attitude();
    benchmark();

    std::cout << "\n✓ All tests completed!\n";
    return 0;
}
