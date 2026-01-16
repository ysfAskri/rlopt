// SwiftSim - Quadrotor Physics Core
// Based on AirSim FastPhysicsEngine (MIT License)
// Modernized with SoA layout for SIMD optimization

#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <algorithm>

namespace swiftsim {

// ============================================================================
// CONSTANTS
// ============================================================================
constexpr float GRAVITY = 9.81f;
constexpr float AIR_DENSITY_SEA_LEVEL = 1.225f; // kg/m^3
constexpr float PI = 3.14159265358979f;

// ============================================================================
// QUATERNION OPERATIONS (for orientation)
// ============================================================================
struct Quaternion {
    float w, x, y, z;

    static Quaternion identity() { return {1.0f, 0.0f, 0.0f, 0.0f}; }

    Quaternion normalized() const {
        float mag = std::sqrt(w*w + x*x + y*y + z*z);
        return {w/mag, x/mag, y/mag, z/mag};
    }

    // Multiply quaternions: this * other
    Quaternion operator*(const Quaternion& other) const {
        return {
            w*other.w - x*other.x - y*other.y - z*other.z,
            w*other.x + x*other.w + y*other.z - z*other.y,
            w*other.y - x*other.z + y*other.w + z*other.x,
            w*other.z + x*other.y - y*other.x + z*other.w
        };
    }
};

// Create quaternion from axis-angle
inline Quaternion quaternionFromAxisAngle(float ax, float ay, float az, float angle) {
    float half_angle = angle * 0.5f;
    float s = std::sin(half_angle);
    return {std::cos(half_angle), ax * s, ay * s, az * s};
}

// ============================================================================
// ROTOR PARAMETERS (from AirSim RotorParams)
// ============================================================================
struct RotorParams {
    float max_thrust = 4.179446268f;      // N (newtons) at full throttle
    float max_torque = 0.05f;             // N·m
    float max_speed = 6396.667f;          // RPM
    float propeller_diameter = 0.2286f;   // m (9 inches)
    float propeller_height = 0.0254f;     // m
    float control_signal_filter_tc = 0.005f; // time constant for motor response
};

// ============================================================================
// QUADROTOR PARAMETERS
// ============================================================================
struct QuadrotorParams {
    float mass = 1.0f;                    // kg
    float arm_length = 0.2275f;           // m (distance from center to rotor)

    // Inertia tensor (diagonal approximation)
    float Ixx = 0.0029125f;               // kg·m²
    float Iyy = 0.0029125f;
    float Izz = 0.0055225f;

    // Body dimensions for drag
    float body_width = 0.18f;             // m
    float body_length = 0.11f;            // m
    float body_height = 0.04f;            // m

    float linear_drag_coefficient = 0.4f;
    float restitution = 0.55f;            // bounce coefficient
    float friction = 0.5f;

    RotorParams rotor_params;

    // Rotor layout (X configuration)
    //     0 (CW)
    //       \
    //   3----+----1
    //  (CCW)   (CCW)
    //       /
    //     2 (CW)
    std::array<float, 4> rotor_x = { 0.16f, 0.16f, -0.16f, -0.16f};
    std::array<float, 4> rotor_y = { 0.16f, -0.16f, -0.16f, 0.16f};
    std::array<int, 4> rotor_direction = {1, -1, 1, -1}; // CW=1, CCW=-1
};

// ============================================================================
// SINGLE DRONE STATE (AoS - for reference/testing)
// ============================================================================
struct DroneState {
    // Position (world frame, NED: North-East-Down)
    float pos_x = 0, pos_y = 0, pos_z = 0;

    // Velocity (world frame)
    float vel_x = 0, vel_y = 0, vel_z = 0;

    // Acceleration (world frame)
    float acc_x = 0, acc_y = 0, acc_z = 0;

    // Orientation (quaternion)
    Quaternion orientation = Quaternion::identity();

    // Angular velocity (body frame)
    float omega_x = 0, omega_y = 0, omega_z = 0;

    // Angular acceleration (body frame)
    float alpha_x = 0, alpha_y = 0, alpha_z = 0;

    // Motor states (0-1 normalized throttle)
    std::array<float, 4> motor_inputs = {0, 0, 0, 0};
    std::array<float, 4> motor_filtered = {0, 0, 0, 0};

    // Flags
    bool is_grounded = false;
    bool has_collided = false;
};

// ============================================================================
// PHYSICS ENGINE (Single Drone - AoS version for testing)
// ============================================================================
class QuadrotorPhysics {
public:
    QuadrotorParams params;

    QuadrotorPhysics(const QuadrotorParams& p = QuadrotorParams()) : params(p) {}

    // Main physics step (Semi-implicit Euler / Verlet)
    void step(DroneState& state, float dt) {
        // 1. Always update motors first
        updateMotors(state, dt);

        if (state.is_grounded) {
            handleGrounded(state);
            if (state.is_grounded) return; // Still grounded, skip physics
        }

        // 2. Compute forces and torques from rotors
        float thrust_total = 0;
        float torque_x = 0, torque_y = 0, torque_z = 0;

        for (int i = 0; i < 4; ++i) {
            float throttle = state.motor_filtered[i];
            float thrust = throttle * params.rotor_params.max_thrust;
            float torque = throttle * params.rotor_params.max_torque * params.rotor_direction[i];

            thrust_total += thrust;

            // Torque from thrust offset (roll and pitch)
            torque_x += thrust * params.rotor_y[i]; // roll
            torque_y -= thrust * params.rotor_x[i]; // pitch (negative for NED)
            torque_z += torque;                      // yaw
        }

        // 3. Transform thrust to world frame
        float fx_body = 0, fy_body = 0, fz_body = -thrust_total; // thrust along -Z in body
        float fx_world, fy_world, fz_world;
        rotateBodyToWorld(state.orientation, fx_body, fy_body, fz_body,
                         fx_world, fy_world, fz_world);

        // 4. Add gravity (NED: down is positive Z)
        fz_world += params.mass * GRAVITY;

        // 5. Compute linear acceleration
        float new_acc_x = fx_world / params.mass;
        float new_acc_y = fy_world / params.mass;
        float new_acc_z = fz_world / params.mass;

        // 6. Semi-implicit Euler: update velocity first
        state.vel_x += (state.acc_x + new_acc_x) * 0.5f * dt;
        state.vel_y += (state.acc_y + new_acc_y) * 0.5f * dt;
        state.vel_z += (state.acc_z + new_acc_z) * 0.5f * dt;

        state.acc_x = new_acc_x;
        state.acc_y = new_acc_y;
        state.acc_z = new_acc_z;

        // 7. Update position using new velocity
        state.pos_x += state.vel_x * dt;
        state.pos_y += state.vel_y * dt;
        state.pos_z += state.vel_z * dt;

        // 8. Compute angular acceleration (Euler's rotation equations)
        // τ = I·α + ω × (I·ω)
        float Lx = params.Ixx * state.omega_x;
        float Ly = params.Iyy * state.omega_y;
        float Lz = params.Izz * state.omega_z;

        // Cross product ω × L
        float cross_x = state.omega_y * Lz - state.omega_z * Ly;
        float cross_y = state.omega_z * Lx - state.omega_x * Lz;
        float cross_z = state.omega_x * Ly - state.omega_y * Lx;

        float new_alpha_x = (torque_x - cross_x) / params.Ixx;
        float new_alpha_y = (torque_y - cross_y) / params.Iyy;
        float new_alpha_z = (torque_z - cross_z) / params.Izz;

        // 9. Update angular velocity
        state.omega_x += (state.alpha_x + new_alpha_x) * 0.5f * dt;
        state.omega_y += (state.alpha_y + new_alpha_y) * 0.5f * dt;
        state.omega_z += (state.alpha_z + new_alpha_z) * 0.5f * dt;

        state.alpha_x = new_alpha_x;
        state.alpha_y = new_alpha_y;
        state.alpha_z = new_alpha_z;

        // 10. Update orientation
        float omega_mag = std::sqrt(state.omega_x*state.omega_x +
                                    state.omega_y*state.omega_y +
                                    state.omega_z*state.omega_z);
        if (omega_mag > 1e-6f) {
            float ax = state.omega_x / omega_mag;
            float ay = state.omega_y / omega_mag;
            float az = state.omega_z / omega_mag;
            Quaternion dq = quaternionFromAxisAngle(ax, ay, az, omega_mag * dt);
            state.orientation = (state.orientation * dq).normalized();
        }

        // 11. Simple ground collision
        if (state.pos_z > 0) { // NED: positive Z is down/ground
            state.pos_z = 0;
            state.vel_z = -state.vel_z * params.restitution;
            if (std::abs(state.vel_z) < 0.1f) {
                state.is_grounded = true;
                state.vel_z = 0;
            }
        }
    }

    // Set motor inputs (0-1 for each motor)
    void setMotorInputs(DroneState& state, float m0, float m1, float m2, float m3) {
        state.motor_inputs[0] = std::clamp(m0, 0.0f, 1.0f);
        state.motor_inputs[1] = std::clamp(m1, 0.0f, 1.0f);
        state.motor_inputs[2] = std::clamp(m2, 0.0f, 1.0f);
        state.motor_inputs[3] = std::clamp(m3, 0.0f, 1.0f);
    }

    // Compute hover throttle (analytical)
    float getHoverThrottle() const {
        float weight = params.mass * GRAVITY;
        float thrust_per_motor = weight / 4.0f;
        return thrust_per_motor / params.rotor_params.max_thrust;
    }

private:
    void updateMotors(DroneState& state, float dt) {
        // First-order filter for motor response
        float alpha = dt / (params.rotor_params.control_signal_filter_tc + dt);
        for (int i = 0; i < 4; ++i) {
            state.motor_filtered[i] += alpha * (state.motor_inputs[i] - state.motor_filtered[i]);
        }
    }

    void handleGrounded(DroneState& state) {
        // Check if thrust exceeds weight (with some margin for stability)
        float total_thrust = 0;
        for (int i = 0; i < 4; ++i) {
            total_thrust += state.motor_filtered[i] * params.rotor_params.max_thrust;
        }
        float weight = params.mass * GRAVITY;
        if (total_thrust > weight * 1.01f) { // 1% margin to break ground
            state.is_grounded = false;
            // Give initial upward velocity
            state.vel_z = -(total_thrust - weight) / params.mass * 0.01f;
        }
    }

    // Rotate vector from body frame to world frame using quaternion
    static void rotateBodyToWorld(const Quaternion& q,
                                  float bx, float by, float bz,
                                  float& wx, float& wy, float& wz) {
        // v' = q * v * q^-1
        // Optimized version for unit quaternion
        float qw = q.w, qx = q.x, qy = q.y, qz = q.z;

        float t2 = qw * qx;
        float t3 = qw * qy;
        float t4 = qw * qz;
        float t5 = -qx * qx;
        float t6 = qx * qy;
        float t7 = qx * qz;
        float t8 = -qy * qy;
        float t9 = qy * qz;
        float t10 = -qz * qz;

        wx = 2 * ((t8 + t10) * bx + (t6 - t4) * by + (t3 + t7) * bz) + bx;
        wy = 2 * ((t4 + t6) * bx + (t5 + t10) * by + (t9 - t2) * bz) + by;
        wz = 2 * ((t7 - t3) * bx + (t2 + t9) * by + (t5 + t8) * bz) + bz;
    }
};

} // namespace swiftsim
