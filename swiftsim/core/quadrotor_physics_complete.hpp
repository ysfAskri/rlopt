// SwiftSim - Complete Quadrotor Physics
// Fully extracted from AirSim (MIT License)
// Includes: Dynamics, Drag, Wind, Environment, Sensors, Collision

#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <random>

namespace swiftsim {

// ============================================================================
// CONSTANTS & EARTH UTILS (from AirSim EarthUtils)
// ============================================================================
constexpr float GRAVITY_SEA_LEVEL = 9.80665f;
constexpr float AIR_DENSITY_SEA_LEVEL = 1.225f;     // kg/m³
constexpr float AIR_PRESSURE_SEA_LEVEL = 101325.0f; // Pa
constexpr float TEMPERATURE_SEA_LEVEL = 288.15f;    // K (15°C)
constexpr float TEMPERATURE_LAPSE_RATE = 0.0065f;   // K/m
constexpr float MOLAR_MASS_AIR = 0.0289644f;        // kg/mol
constexpr float GAS_CONSTANT = 8.31447f;            // J/(mol·K)
constexpr float PI = 3.14159265358979f;
constexpr float SPEED_OF_LIGHT = 299792458.0f;      // Velocity clamp

// Earth utils functions
inline float getGravity(float altitude_m) {
    // Gravity decreases with altitude: g = g0 * (R/(R+h))^2
    constexpr float EARTH_RADIUS = 6371000.0f;
    float ratio = EARTH_RADIUS / (EARTH_RADIUS + altitude_m);
    return GRAVITY_SEA_LEVEL * ratio * ratio;
}

inline float getAirDensity(float altitude_m) {
    // International Standard Atmosphere model
    if (altitude_m < 0) altitude_m = 0;
    float temperature = TEMPERATURE_SEA_LEVEL - TEMPERATURE_LAPSE_RATE * altitude_m;
    float pressure = AIR_PRESSURE_SEA_LEVEL *
        std::pow(1.0f - (TEMPERATURE_LAPSE_RATE * altitude_m) / TEMPERATURE_SEA_LEVEL,
                 (GRAVITY_SEA_LEVEL * MOLAR_MASS_AIR) / (GAS_CONSTANT * TEMPERATURE_LAPSE_RATE));
    return pressure * MOLAR_MASS_AIR / (GAS_CONSTANT * temperature);
}

inline float getAirPressure(float altitude_m) {
    if (altitude_m < 0) altitude_m = 0;
    return AIR_PRESSURE_SEA_LEVEL *
        std::pow(1.0f - (TEMPERATURE_LAPSE_RATE * altitude_m) / TEMPERATURE_SEA_LEVEL,
                 (GRAVITY_SEA_LEVEL * MOLAR_MASS_AIR) / (GAS_CONSTANT * TEMPERATURE_LAPSE_RATE));
}

inline float getTemperature(float altitude_m) {
    return TEMPERATURE_SEA_LEVEL - TEMPERATURE_LAPSE_RATE * altitude_m;
}

// ============================================================================
// VECTOR3 & QUATERNION
// ============================================================================
struct Vector3 {
    float x = 0, y = 0, z = 0;

    Vector3() = default;
    Vector3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    Vector3 operator+(const Vector3& v) const { return {x + v.x, y + v.y, z + v.z}; }
    Vector3 operator-(const Vector3& v) const { return {x - v.x, y - v.y, z - v.z}; }
    Vector3 operator*(float s) const { return {x * s, y * s, z * s}; }
    Vector3 operator/(float s) const { return {x / s, y / s, z / s}; }
    Vector3& operator+=(const Vector3& v) { x += v.x; y += v.y; z += v.z; return *this; }
    Vector3& operator-=(const Vector3& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
    Vector3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }

    float dot(const Vector3& v) const { return x*v.x + y*v.y + z*v.z; }
    Vector3 cross(const Vector3& v) const {
        return {y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x};
    }
    float norm() const { return std::sqrt(x*x + y*y + z*z); }
    float squaredNorm() const { return x*x + y*y + z*z; }
    Vector3 normalized() const {
        float n = norm();
        return n > 1e-10f ? *this / n : Vector3{0,0,0};
    }

    static Vector3 zero() { return {0, 0, 0}; }
};

struct Quaternion {
    float w = 1, x = 0, y = 0, z = 0;

    static Quaternion identity() { return {1, 0, 0, 0}; }

    Quaternion normalized() const {
        float mag = std::sqrt(w*w + x*x + y*y + z*z);
        return {w/mag, x/mag, y/mag, z/mag};
    }

    Quaternion conjugate() const { return {w, -x, -y, -z}; }

    Quaternion operator*(const Quaternion& q) const {
        return {
            w*q.w - x*q.x - y*q.y - z*q.z,
            w*q.x + x*q.w + y*q.z - z*q.y,
            w*q.y - x*q.z + y*q.w + z*q.x,
            w*q.z + x*q.y - y*q.x + z*q.w
        };
    }

    // Rotate vector from body to world frame
    Vector3 rotateVector(const Vector3& v) const {
        // Optimized quaternion-vector rotation
        float t2 = w * x, t3 = w * y, t4 = w * z;
        float t5 = -x * x, t6 = x * y, t7 = x * z;
        float t8 = -y * y, t9 = y * z, t10 = -z * z;
        return {
            2*((t8+t10)*v.x + (t6-t4)*v.y + (t3+t7)*v.z) + v.x,
            2*((t4+t6)*v.x + (t5+t10)*v.y + (t9-t2)*v.z) + v.y,
            2*((t7-t3)*v.x + (t2+t9)*v.y + (t5+t8)*v.z) + v.z
        };
    }

    // Rotate vector from world to body frame
    Vector3 rotateVectorInverse(const Vector3& v) const {
        return conjugate().rotateVector(v);
    }

    // Convert to Euler angles (roll, pitch, yaw)
    void toEuler(float& roll, float& pitch, float& yaw) const {
        // Roll (x-axis rotation)
        float sinr_cosp = 2 * (w * x + y * z);
        float cosr_cosp = 1 - 2 * (x * x + y * y);
        roll = std::atan2(sinr_cosp, cosr_cosp);
        // Pitch (y-axis rotation)
        float sinp = 2 * (w * y - z * x);
        pitch = std::abs(sinp) >= 1 ? std::copysign(PI / 2, sinp) : std::asin(sinp);
        // Yaw (z-axis rotation)
        float siny_cosp = 2 * (w * z + x * y);
        float cosy_cosp = 1 - 2 * (y * y + z * z);
        yaw = std::atan2(siny_cosp, cosy_cosp);
    }
};

inline Quaternion quaternionFromAxisAngle(const Vector3& axis, float angle) {
    float half = angle * 0.5f;
    float s = std::sin(half);
    return {std::cos(half), axis.x * s, axis.y * s, axis.z * s};
}

// ============================================================================
// WRENCH (Force + Torque)
// ============================================================================
struct Wrench {
    Vector3 force;
    Vector3 torque;

    static Wrench zero() { return {{0,0,0}, {0,0,0}}; }

    Wrench operator+(const Wrench& w) const {
        return {force + w.force, torque + w.torque};
    }
    Wrench& operator+=(const Wrench& w) {
        force += w.force;
        torque += w.torque;
        return *this;
    }
};

// ============================================================================
// ENVIRONMENT STATE
// ============================================================================
struct EnvironmentState {
    float altitude = 0;          // m (positive up, unlike NED)
    float gravity = GRAVITY_SEA_LEVEL;
    float air_density = AIR_DENSITY_SEA_LEVEL;
    float air_pressure = AIR_PRESSURE_SEA_LEVEL;
    float temperature = TEMPERATURE_SEA_LEVEL;
    Vector3 wind = {0, 0, 0};    // Wind velocity in world frame

    void update(float alt) {
        altitude = alt;
        gravity = getGravity(alt);
        air_density = getAirDensity(alt);
        air_pressure = getAirPressure(alt);
        temperature = getTemperature(alt);
    }
};

// ============================================================================
// ROTOR PARAMETERS (from AirSim)
// ============================================================================
struct RotorParams {
    float max_thrust = 4.179446268f;      // N at full throttle
    float max_torque = 0.05f;             // N·m reaction torque
    float max_speed_square = 40923225.0f; // (RPM)² = 6396.667²
    float propeller_diameter = 0.2286f;   // m
    float propeller_height = 0.0254f;     // m
    float control_signal_filter_tc = 0.005f; // Motor response time constant
};

// ============================================================================
// QUADROTOR PARAMETERS
// ============================================================================
struct QuadrotorParams {
    float mass = 1.0f;
    float arm_length = 0.2275f;

    // Inertia tensor (diagonal)
    float Ixx = 0.0029125f;
    float Iyy = 0.0029125f;
    float Izz = 0.0055225f;

    // Body box for drag calculation
    float body_x = 0.18f;
    float body_y = 0.11f;
    float body_z = 0.04f;

    float linear_drag_coefficient = 0.4f;
    float angular_drag_coefficient = 0.1f;
    float restitution = 0.55f;
    float friction = 0.5f;

    RotorParams rotor;

    // Rotor positions (X config) and directions
    std::array<Vector3, 4> rotor_positions = {{
        {0.16f, 0.16f, 0.0f},   // Front-right
        {0.16f, -0.16f, 0.0f},  // Front-left
        {-0.16f, -0.16f, 0.0f}, // Back-left
        {-0.16f, 0.16f, 0.0f}   // Back-right
    }};
    std::array<int, 4> rotor_directions = {1, -1, 1, -1}; // CW=1, CCW=-1
};

// ============================================================================
// DRAG VERTEX (for aerodynamic drag calculation)
// ============================================================================
struct DragVertex {
    Vector3 position;
    Vector3 normal;
    float drag_factor;
};

// ============================================================================
// SENSOR OUTPUTS
// ============================================================================
struct ImuOutput {
    Vector3 angular_velocity;     // rad/s (body frame)
    Vector3 linear_acceleration;  // m/s² (body frame, without gravity)
    Quaternion orientation;
    uint64_t timestamp_ns = 0;
};

struct GpsOutput {
    double latitude = 0;   // degrees
    double longitude = 0;  // degrees
    float altitude = 0;    // m
    Vector3 velocity;      // m/s (NED)
    float eph = 1.0f;      // horizontal accuracy (m)
    float epv = 1.5f;      // vertical accuracy (m)
    uint64_t timestamp_ns = 0;
};

struct BarometerOutput {
    float pressure = AIR_PRESSURE_SEA_LEVEL;  // Pa
    float altitude = 0;                        // m (from pressure)
    float temperature = TEMPERATURE_SEA_LEVEL; // K
    uint64_t timestamp_ns = 0;
};

struct MagnetometerOutput {
    Vector3 magnetic_field;  // Gauss (body frame)
    uint64_t timestamp_ns = 0;
};

// ============================================================================
// COMPLETE DRONE STATE
// ============================================================================
struct DroneState {
    // Kinematics (NED frame: X=North, Y=East, Z=Down)
    Vector3 position = {0, 0, 0};
    Vector3 velocity = {0, 0, 0};
    Vector3 acceleration = {0, 0, 0};
    Quaternion orientation = Quaternion::identity();
    Vector3 angular_velocity = {0, 0, 0};      // body frame
    Vector3 angular_acceleration = {0, 0, 0};  // body frame

    // Motors
    std::array<float, 4> motor_inputs = {0, 0, 0, 0};
    std::array<float, 4> motor_filtered = {0, 0, 0, 0};
    std::array<float, 4> motor_thrust = {0, 0, 0, 0};
    std::array<float, 4> motor_speed = {0, 0, 0, 0}; // RPM

    // Sensors
    ImuOutput imu;
    GpsOutput gps;
    BarometerOutput barometer;
    MagnetometerOutput magnetometer;

    // Collision
    bool is_grounded = false;
    bool has_collided = false;
    Vector3 collision_normal = {0, 0, 0};
    Vector3 collision_position = {0, 0, 0};

    // Environment
    EnvironmentState environment;
};

// ============================================================================
// SENSOR NOISE MODEL (from AirSim ImuSimple)
// ============================================================================
class SensorNoise {
public:
    struct ImuNoiseParams {
        // Gyroscope
        float gyro_arw = 0.004f;           // Angular random walk (rad/s/√Hz)
        float gyro_bias_stability = 0.01f; // Bias stability (rad/s)
        float gyro_tau = 500.0f;           // Correlation time (s)
        Vector3 gyro_turn_on_bias = {0.01f, 0.01f, 0.01f};

        // Accelerometer
        float accel_vrw = 0.008f;          // Velocity random walk (m/s²/√Hz)
        float accel_bias_stability = 0.03f;
        float accel_tau = 500.0f;
        Vector3 accel_turn_on_bias = {0.05f, 0.05f, 0.05f};
    };

    ImuNoiseParams params;
    std::mt19937 rng;
    std::normal_distribution<float> gauss{0.0f, 1.0f};
    Vector3 gyro_bias = {0, 0, 0};
    Vector3 accel_bias = {0, 0, 0};

    SensorNoise(uint32_t seed = 42) : rng(seed) {
        reset();
    }

    void reset() {
        gyro_bias = params.gyro_turn_on_bias;
        accel_bias = params.accel_turn_on_bias;
    }

    Vector3 randomVector3() {
        return {gauss(rng), gauss(rng), gauss(rng)};
    }

    void addImuNoise(Vector3& accel, Vector3& gyro, float dt) {
        float sqrt_dt = std::sqrt(std::max(dt, 0.0001f));

        // Gyroscope noise
        float gyro_sigma = params.gyro_arw / sqrt_dt;
        gyro += randomVector3() * gyro_sigma + gyro_bias;
        float gyro_bias_sigma = (params.gyro_bias_stability / std::sqrt(params.gyro_tau)) * sqrt_dt;
        gyro_bias += randomVector3() * gyro_bias_sigma;

        // Accelerometer noise
        float accel_sigma = params.accel_vrw / sqrt_dt;
        accel += randomVector3() * accel_sigma + accel_bias;
        float accel_bias_sigma = (params.accel_bias_stability / std::sqrt(params.accel_tau)) * sqrt_dt;
        accel_bias += randomVector3() * accel_bias_sigma;
    }
};

// ============================================================================
// COMPLETE PHYSICS ENGINE
// ============================================================================
class QuadrotorPhysicsComplete {
public:
    QuadrotorParams params;
    SensorNoise sensor_noise;
    bool enable_drag = true;
    bool enable_ground_lock = true;
    bool enable_sensor_noise = true;

    static constexpr float DRAG_MIN_VELOCITY = 0.1f;
    static constexpr float RESTING_VELOCITY_MAX = 0.1f;

    QuadrotorPhysicsComplete(const QuadrotorParams& p = QuadrotorParams(), uint32_t seed = 42)
        : params(p), sensor_noise(seed) {
        initDragVertices();
    }

    void reset(DroneState& state) {
        state = DroneState{};
        sensor_noise.reset();
    }

    void setMotorInputs(DroneState& state, float m0, float m1, float m2, float m3) {
        state.motor_inputs[0] = std::clamp(m0, 0.0f, 1.0f);
        state.motor_inputs[1] = std::clamp(m1, 0.0f, 1.0f);
        state.motor_inputs[2] = std::clamp(m2, 0.0f, 1.0f);
        state.motor_inputs[3] = std::clamp(m3, 0.0f, 1.0f);
    }

    void setWind(DroneState& state, float wx, float wy, float wz) {
        state.environment.wind = {wx, wy, wz};
    }

    float getHoverThrottle() const {
        float weight = params.mass * GRAVITY_SEA_LEVEL;
        return (weight / 4.0f) / params.rotor.max_thrust;
    }

    // Main physics step
    void step(DroneState& state, float dt) {
        // Update environment
        float altitude = -state.position.z; // NED to altitude
        state.environment.update(altitude);

        // Update motors
        updateMotors(state, dt);

        // Handle grounded state
        if (state.is_grounded) {
            if (!handleGrounded(state)) {
                return; // Still grounded
            }
        }

        // Compute body wrench (from rotors)
        Wrench body_wrench = computeRotorWrench(state);

        // Compute drag wrench
        Wrench drag_wrench = Wrench::zero();
        if (enable_drag) {
            drag_wrench = computeDragWrench(state);
        }

        // Total wrench
        Wrench total_wrench = body_wrench + drag_wrench;

        // Add gravity (NED: positive Z is down)
        Vector3 gravity_force = {0, 0, params.mass * state.environment.gravity};
        total_wrench.force += gravity_force;

        // Linear dynamics (Verlet integration)
        Vector3 new_accel = total_wrench.force / params.mass;

        // Velocity clamp
        if (new_accel.squaredNorm() > SPEED_OF_LIGHT * SPEED_OF_LIGHT) {
            new_accel = new_accel.normalized() * SPEED_OF_LIGHT;
        }

        // Semi-implicit Euler
        state.velocity += (state.acceleration + new_accel) * (0.5f * dt);
        state.acceleration = new_accel;
        state.position += state.velocity * dt;

        // Angular dynamics (Euler's equations)
        Vector3 L = {
            params.Ixx * state.angular_velocity.x,
            params.Iyy * state.angular_velocity.y,
            params.Izz * state.angular_velocity.z
        };
        Vector3 omega_cross_L = state.angular_velocity.cross(L);
        Vector3 new_alpha = {
            (total_wrench.torque.x - omega_cross_L.x) / params.Ixx,
            (total_wrench.torque.y - omega_cross_L.y) / params.Iyy,
            (total_wrench.torque.z - omega_cross_L.z) / params.Izz
        };

        state.angular_velocity += (state.angular_acceleration + new_alpha) * (0.5f * dt);
        state.angular_acceleration = new_alpha;

        // Update orientation
        float omega_mag = state.angular_velocity.norm();
        if (omega_mag > 1e-8f) {
            Vector3 axis = state.angular_velocity / omega_mag;
            Quaternion dq = quaternionFromAxisAngle(axis, omega_mag * dt);
            state.orientation = (state.orientation * dq).normalized();
        }

        // Ground collision
        handleGroundCollision(state, dt);

        // Update sensors
        updateSensors(state, dt);
    }

private:
    std::vector<DragVertex> drag_vertices;

    void initDragVertices() {
        // 6 faces of the body box
        float prop_area = PI * params.rotor.propeller_diameter * params.rotor.propeller_diameter / 4.0f;
        float prop_xsection = PI * params.rotor.propeller_diameter * params.rotor.propeller_height;

        float top_bottom = params.body_x * params.body_y;
        float front_back = params.body_y * params.body_z;
        float left_right = params.body_x * params.body_z;

        float cd = params.linear_drag_coefficient / 2.0f;

        drag_vertices = {
            {{0, 0, -params.body_z/2}, {0, 0, -1}, (top_bottom + 4*prop_area) * cd},
            {{0, 0, params.body_z/2}, {0, 0, 1}, (top_bottom + 4*prop_area) * cd},
            {{0, -params.body_y/2, 0}, {0, -1, 0}, (left_right + 4*prop_xsection) * cd},
            {{0, params.body_y/2, 0}, {0, 1, 0}, (left_right + 4*prop_xsection) * cd},
            {{-params.body_x/2, 0, 0}, {-1, 0, 0}, (front_back + 4*prop_xsection) * cd},
            {{params.body_x/2, 0, 0}, {1, 0, 0}, (front_back + 4*prop_xsection) * cd}
        };
    }

    void updateMotors(DroneState& state, float dt) {
        float alpha = dt / (params.rotor.control_signal_filter_tc + dt);
        for (int i = 0; i < 4; ++i) {
            state.motor_filtered[i] += alpha * (state.motor_inputs[i] - state.motor_filtered[i]);
            state.motor_thrust[i] = state.motor_filtered[i] * params.rotor.max_thrust;
            state.motor_speed[i] = std::sqrt(state.motor_filtered[i] * params.rotor.max_speed_square);
        }
    }

    bool handleGrounded(DroneState& state) {
        float total_thrust = 0;
        for (int i = 0; i < 4; ++i) {
            total_thrust += state.motor_thrust[i];
        }
        float weight = params.mass * state.environment.gravity;
        if (total_thrust > weight * 1.01f) {
            state.is_grounded = false;
            return true; // Lift off
        }
        // Keep grounded
        state.velocity = Vector3::zero();
        state.angular_velocity = Vector3::zero();
        state.acceleration = Vector3::zero();
        state.angular_acceleration = Vector3::zero();
        return false;
    }

    Wrench computeRotorWrench(const DroneState& state) {
        Wrench wrench = Wrench::zero();
        float air_ratio = state.environment.air_density / AIR_DENSITY_SEA_LEVEL;

        for (int i = 0; i < 4; ++i) {
            float thrust = state.motor_thrust[i] * air_ratio;
            float torque = state.motor_filtered[i] * params.rotor.max_torque * params.rotor_directions[i] * air_ratio;

            // Thrust is along -Z in body frame
            Vector3 thrust_force = {0, 0, -thrust};
            wrench.force += thrust_force;

            // Torque from thrust offset
            wrench.torque += params.rotor_positions[i].cross(thrust_force);

            // Reaction torque (yaw)
            wrench.torque.z += torque;
        }

        // Transform force to world frame
        wrench.force = state.orientation.rotateVector(wrench.force);

        return wrench;
    }

    Wrench computeDragWrench(const DroneState& state) {
        Wrench wrench = Wrench::zero();

        // Relative velocity (account for wind)
        Vector3 rel_vel = state.velocity - state.environment.wind;
        Vector3 vel_body = state.orientation.rotateVectorInverse(rel_vel);

        for (const auto& vertex : drag_vertices) {
            Vector3 vel_at_vertex = vel_body + state.angular_velocity.cross(vertex.position);
            float vel_comp = vertex.normal.dot(vel_at_vertex);

            if (vel_comp > DRAG_MIN_VELOCITY) {
                float drag_mag = -vertex.drag_factor * state.environment.air_density * vel_comp * vel_comp;
                Vector3 drag_force = vertex.normal * drag_mag;
                Vector3 drag_torque = vertex.position.cross(drag_force);

                wrench.force += drag_force;
                wrench.torque += drag_torque;
            }
        }

        // Transform force to world frame
        wrench.force = state.orientation.rotateVector(wrench.force);

        return wrench;
    }

    void handleGroundCollision(DroneState& state, float dt) {
        if (state.position.z > 0) { // Below ground in NED
            state.position.z = 0;

            if (state.velocity.z > RESTING_VELOCITY_MAX) {
                // Bounce
                state.velocity.z = -state.velocity.z * params.restitution;
                // Friction on horizontal velocity
                state.velocity.x *= (1.0f - params.friction);
                state.velocity.y *= (1.0f - params.friction);
                state.has_collided = true;
            } else {
                // Come to rest
                if (enable_ground_lock) {
                    state.is_grounded = true;
                    state.velocity = Vector3::zero();
                    state.angular_velocity = state.angular_velocity * 0.9f;
                    // Level the drone
                    float roll, pitch, yaw;
                    state.orientation.toEuler(roll, pitch, yaw);
                    state.orientation = quaternionFromAxisAngle({0, 0, 1}, yaw);
                }
            }
            state.collision_normal = {0, 0, -1};
            state.collision_position = state.position;
        }
    }

    void updateSensors(DroneState& state, float dt) {
        // IMU
        Vector3 gravity_world = {0, 0, state.environment.gravity};
        Vector3 accel_world = state.acceleration - gravity_world;
        state.imu.linear_acceleration = state.orientation.rotateVectorInverse(accel_world);
        state.imu.angular_velocity = state.angular_velocity;
        state.imu.orientation = state.orientation;

        if (enable_sensor_noise) {
            sensor_noise.addImuNoise(state.imu.linear_acceleration, state.imu.angular_velocity, dt);
        }

        // Barometer
        state.barometer.altitude = -state.position.z;
        state.barometer.pressure = state.environment.air_pressure;
        state.barometer.temperature = state.environment.temperature;

        // GPS (simplified - no delay/noise for now)
        state.gps.altitude = -state.position.z;
        state.gps.velocity = state.velocity;

        // Magnetometer (simplified - constant field)
        Vector3 mag_world = {0.22f, 0.0f, 0.42f}; // Approximate Earth field (Gauss)
        state.magnetometer.magnetic_field = state.orientation.rotateVectorInverse(mag_world);
    }
};

} // namespace swiftsim
