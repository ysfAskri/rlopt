// SwiftSim - SoA Swarm Physics Engine
// Optimized for SIMD vectorization and multi-threading
// Target: 50-100M drone-steps/second

#pragma once

#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace swiftsim {

// ============================================================================
// CONSTANTS
// ============================================================================
constexpr float GRAVITY = 9.80665f;
constexpr float AIR_DENSITY = 1.225f;
constexpr size_t SIMD_WIDTH = 8; // AVX processes 8 floats at once

// ============================================================================
// ALIGNED MEMORY ALLOCATION
// ============================================================================
inline void* aligned_alloc_32(size_t size) {
#ifdef _MSC_VER
    return _aligned_malloc(size, 32);
#else
    void* ptr = nullptr;
    posix_memalign(&ptr, 32, size);
    return ptr;
#endif
}

inline void aligned_free(void* ptr) {
#ifdef _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// ============================================================================
// SOA DRONE SWARM - Core Data Structure
// ============================================================================
class DroneSwarmSoA {
public:
    size_t count = 0;
    size_t capacity = 0;

    // Position (NED frame)
    float* pos_x = nullptr;
    float* pos_y = nullptr;
    float* pos_z = nullptr;

    // Velocity
    float* vel_x = nullptr;
    float* vel_y = nullptr;
    float* vel_z = nullptr;

    // Acceleration
    float* acc_x = nullptr;
    float* acc_y = nullptr;
    float* acc_z = nullptr;

    // Orientation (quaternion)
    float* quat_w = nullptr;
    float* quat_x = nullptr;
    float* quat_y = nullptr;
    float* quat_z = nullptr;

    // Angular velocity (body frame)
    float* omega_x = nullptr;
    float* omega_y = nullptr;
    float* omega_z = nullptr;

    // Angular acceleration
    float* alpha_x = nullptr;
    float* alpha_y = nullptr;
    float* alpha_z = nullptr;

    // Motor inputs (4 per drone, interleaved: m0,m1,m2,m3,m0,m1,m2,m3,...)
    float* motor_0 = nullptr;
    float* motor_1 = nullptr;
    float* motor_2 = nullptr;
    float* motor_3 = nullptr;

    // Motor filtered outputs
    float* motor_f0 = nullptr;
    float* motor_f1 = nullptr;
    float* motor_f2 = nullptr;
    float* motor_f3 = nullptr;

    // Flags (packed as floats for SIMD, 0.0 or 1.0)
    float* is_grounded = nullptr;
    float* is_active = nullptr;

    DroneSwarmSoA() = default;

    explicit DroneSwarmSoA(size_t n) {
        resize(n);
    }

    ~DroneSwarmSoA() {
        deallocate();
    }

    // No copy (owns memory)
    DroneSwarmSoA(const DroneSwarmSoA&) = delete;
    DroneSwarmSoA& operator=(const DroneSwarmSoA&) = delete;

    // Move OK
    DroneSwarmSoA(DroneSwarmSoA&& other) noexcept {
        *this = std::move(other);
    }

    DroneSwarmSoA& operator=(DroneSwarmSoA&& other) noexcept {
        if (this != &other) {
            deallocate();
            count = other.count;
            capacity = other.capacity;
            // Transfer all pointers
            pos_x = other.pos_x; pos_y = other.pos_y; pos_z = other.pos_z;
            vel_x = other.vel_x; vel_y = other.vel_y; vel_z = other.vel_z;
            acc_x = other.acc_x; acc_y = other.acc_y; acc_z = other.acc_z;
            quat_w = other.quat_w; quat_x = other.quat_x;
            quat_y = other.quat_y; quat_z = other.quat_z;
            omega_x = other.omega_x; omega_y = other.omega_y; omega_z = other.omega_z;
            alpha_x = other.alpha_x; alpha_y = other.alpha_y; alpha_z = other.alpha_z;
            motor_0 = other.motor_0; motor_1 = other.motor_1;
            motor_2 = other.motor_2; motor_3 = other.motor_3;
            motor_f0 = other.motor_f0; motor_f1 = other.motor_f1;
            motor_f2 = other.motor_f2; motor_f3 = other.motor_f3;
            is_grounded = other.is_grounded; is_active = other.is_active;
            // Null out other
            other.count = other.capacity = 0;
            other.pos_x = other.pos_y = other.pos_z = nullptr;
            other.vel_x = other.vel_y = other.vel_z = nullptr;
            other.acc_x = other.acc_y = other.acc_z = nullptr;
            other.quat_w = other.quat_x = other.quat_y = other.quat_z = nullptr;
            other.omega_x = other.omega_y = other.omega_z = nullptr;
            other.alpha_x = other.alpha_y = other.alpha_z = nullptr;
            other.motor_0 = other.motor_1 = other.motor_2 = other.motor_3 = nullptr;
            other.motor_f0 = other.motor_f1 = other.motor_f2 = other.motor_f3 = nullptr;
            other.is_grounded = other.is_active = nullptr;
        }
        return *this;
    }

    void resize(size_t n) {
        if (n > capacity) {
            deallocate();
            // Round up to SIMD width for alignment
            capacity = ((n + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;
            allocate(capacity);
        }
        count = n;
    }

    void reset() {
        // Initialize all drones to default state
        for (size_t i = 0; i < capacity; ++i) {
            pos_x[i] = pos_y[i] = 0.0f;
            pos_z[i] = -10.0f; // 10m altitude
            vel_x[i] = vel_y[i] = vel_z[i] = 0.0f;
            acc_x[i] = acc_y[i] = acc_z[i] = 0.0f;
            quat_w[i] = 1.0f;
            quat_x[i] = quat_y[i] = quat_z[i] = 0.0f;
            omega_x[i] = omega_y[i] = omega_z[i] = 0.0f;
            alpha_x[i] = alpha_y[i] = alpha_z[i] = 0.0f;
            motor_0[i] = motor_1[i] = motor_2[i] = motor_3[i] = 0.0f;
            motor_f0[i] = motor_f1[i] = motor_f2[i] = motor_f3[i] = 0.0f;
            is_grounded[i] = 0.0f;
            is_active[i] = 1.0f;
        }
    }

private:
    void allocate(size_t n) {
        size_t bytes = n * sizeof(float);
        pos_x = (float*)aligned_alloc_32(bytes);
        pos_y = (float*)aligned_alloc_32(bytes);
        pos_z = (float*)aligned_alloc_32(bytes);
        vel_x = (float*)aligned_alloc_32(bytes);
        vel_y = (float*)aligned_alloc_32(bytes);
        vel_z = (float*)aligned_alloc_32(bytes);
        acc_x = (float*)aligned_alloc_32(bytes);
        acc_y = (float*)aligned_alloc_32(bytes);
        acc_z = (float*)aligned_alloc_32(bytes);
        quat_w = (float*)aligned_alloc_32(bytes);
        quat_x = (float*)aligned_alloc_32(bytes);
        quat_y = (float*)aligned_alloc_32(bytes);
        quat_z = (float*)aligned_alloc_32(bytes);
        omega_x = (float*)aligned_alloc_32(bytes);
        omega_y = (float*)aligned_alloc_32(bytes);
        omega_z = (float*)aligned_alloc_32(bytes);
        alpha_x = (float*)aligned_alloc_32(bytes);
        alpha_y = (float*)aligned_alloc_32(bytes);
        alpha_z = (float*)aligned_alloc_32(bytes);
        motor_0 = (float*)aligned_alloc_32(bytes);
        motor_1 = (float*)aligned_alloc_32(bytes);
        motor_2 = (float*)aligned_alloc_32(bytes);
        motor_3 = (float*)aligned_alloc_32(bytes);
        motor_f0 = (float*)aligned_alloc_32(bytes);
        motor_f1 = (float*)aligned_alloc_32(bytes);
        motor_f2 = (float*)aligned_alloc_32(bytes);
        motor_f3 = (float*)aligned_alloc_32(bytes);
        is_grounded = (float*)aligned_alloc_32(bytes);
        is_active = (float*)aligned_alloc_32(bytes);
    }

    void deallocate() {
        aligned_free(pos_x); aligned_free(pos_y); aligned_free(pos_z);
        aligned_free(vel_x); aligned_free(vel_y); aligned_free(vel_z);
        aligned_free(acc_x); aligned_free(acc_y); aligned_free(acc_z);
        aligned_free(quat_w); aligned_free(quat_x);
        aligned_free(quat_y); aligned_free(quat_z);
        aligned_free(omega_x); aligned_free(omega_y); aligned_free(omega_z);
        aligned_free(alpha_x); aligned_free(alpha_y); aligned_free(alpha_z);
        aligned_free(motor_0); aligned_free(motor_1);
        aligned_free(motor_2); aligned_free(motor_3);
        aligned_free(motor_f0); aligned_free(motor_f1);
        aligned_free(motor_f2); aligned_free(motor_f3);
        aligned_free(is_grounded); aligned_free(is_active);
        capacity = 0;
    }
};

// ============================================================================
// QUADROTOR PARAMS (shared across all drones)
// ============================================================================
struct SwarmParams {
    float mass = 1.0f;
    float max_thrust = 4.179446268f;  // N per motor
    float max_torque = 0.05f;
    float motor_tc = 0.005f;          // Motor time constant

    // Rotor positions (X config)
    float rotor_x[4] = {0.16f, 0.16f, -0.16f, -0.16f};
    float rotor_y[4] = {0.16f, -0.16f, -0.16f, 0.16f};
    int rotor_dir[4] = {1, -1, 1, -1};

    // Inertia
    float Ixx = 0.0029125f;
    float Iyy = 0.0029125f;
    float Izz = 0.0055225f;

    float hover_throttle() const {
        return (mass * GRAVITY / 4.0f) / max_thrust;
    }
};

// ============================================================================
// SIMD PHYSICS ENGINE
// ============================================================================
class SwarmPhysicsSoA {
public:
    SwarmParams params;

    // ========================================================================
    // SCALAR FALLBACK (for reference and small counts)
    // ========================================================================
    void step_scalar(DroneSwarmSoA& s, float dt) {
        const float alpha = dt / (params.motor_tc + dt);
        const float inv_mass = 1.0f / params.mass;
        const float half_dt = 0.5f * dt;

        for (size_t i = 0; i < s.count; ++i) {
            if (s.is_active[i] < 0.5f) continue;

            // Skip grounded drones (simplified)
            if (s.is_grounded[i] > 0.5f) {
                float total_thrust = (s.motor_f0[i] + s.motor_f1[i] +
                                     s.motor_f2[i] + s.motor_f3[i]) * params.max_thrust;
                if (total_thrust > params.mass * GRAVITY * 1.01f) {
                    s.is_grounded[i] = 0.0f;
                } else {
                    continue;
                }
            }

            // Motor filter
            s.motor_f0[i] += alpha * (s.motor_0[i] - s.motor_f0[i]);
            s.motor_f1[i] += alpha * (s.motor_1[i] - s.motor_f1[i]);
            s.motor_f2[i] += alpha * (s.motor_2[i] - s.motor_f2[i]);
            s.motor_f3[i] += alpha * (s.motor_3[i] - s.motor_f3[i]);

            // Compute thrust (body -Z)
            float t0 = s.motor_f0[i] * params.max_thrust;
            float t1 = s.motor_f1[i] * params.max_thrust;
            float t2 = s.motor_f2[i] * params.max_thrust;
            float t3 = s.motor_f3[i] * params.max_thrust;
            float thrust_total = t0 + t1 + t2 + t3;

            // Torques
            float torque_x = (t0 + t3) * params.rotor_y[0] + (t1 + t2) * params.rotor_y[1];
            float torque_y = -(t0 + t1) * params.rotor_x[0] - (t2 + t3) * params.rotor_x[2];
            float torque_z = (s.motor_f0[i] - s.motor_f1[i] + s.motor_f2[i] - s.motor_f3[i]) *
                            params.max_torque;

            // Rotate thrust to world frame (simplified - assumes near-level)
            // Full quaternion rotation: f_world = q * f_body * q^-1
            float qw = s.quat_w[i], qx = s.quat_x[i], qy = s.quat_y[i], qz = s.quat_z[i];

            // Body thrust is (0, 0, -thrust_total)
            // Rotate to world frame
            float t2_ = qw * qx;
            float t3_ = qw * qy;
            float t5_ = -qx * qx;
            float t6_ = qx * qy;
            float t7_ = qx * qz;
            float t8_ = -qy * qy;
            float t9_ = qy * qz;
            float t10_ = -qz * qz;

            float fz_body = -thrust_total;
            float fx_world = 2 * ((t3_ + t7_) * fz_body);
            float fy_world = 2 * ((t9_ - t2_) * fz_body);
            float fz_world = 2 * ((t5_ + t8_) * fz_body) + fz_body;

            // Add gravity
            fz_world += params.mass * GRAVITY;

            // Linear acceleration
            float new_acc_x = fx_world * inv_mass;
            float new_acc_y = fy_world * inv_mass;
            float new_acc_z = fz_world * inv_mass;

            // Semi-implicit Euler
            s.vel_x[i] += (s.acc_x[i] + new_acc_x) * half_dt;
            s.vel_y[i] += (s.acc_y[i] + new_acc_y) * half_dt;
            s.vel_z[i] += (s.acc_z[i] + new_acc_z) * half_dt;

            s.acc_x[i] = new_acc_x;
            s.acc_y[i] = new_acc_y;
            s.acc_z[i] = new_acc_z;

            s.pos_x[i] += s.vel_x[i] * dt;
            s.pos_y[i] += s.vel_y[i] * dt;
            s.pos_z[i] += s.vel_z[i] * dt;

            // Angular dynamics (simplified - Euler equations)
            float Lx = params.Ixx * s.omega_x[i];
            float Ly = params.Iyy * s.omega_y[i];
            float Lz = params.Izz * s.omega_z[i];

            float cross_x = s.omega_y[i] * Lz - s.omega_z[i] * Ly;
            float cross_y = s.omega_z[i] * Lx - s.omega_x[i] * Lz;
            float cross_z = s.omega_x[i] * Ly - s.omega_y[i] * Lx;

            float new_alpha_x = (torque_x - cross_x) / params.Ixx;
            float new_alpha_y = (torque_y - cross_y) / params.Iyy;
            float new_alpha_z = (torque_z - cross_z) / params.Izz;

            s.omega_x[i] += (s.alpha_x[i] + new_alpha_x) * half_dt;
            s.omega_y[i] += (s.alpha_y[i] + new_alpha_y) * half_dt;
            s.omega_z[i] += (s.alpha_z[i] + new_alpha_z) * half_dt;

            s.alpha_x[i] = new_alpha_x;
            s.alpha_y[i] = new_alpha_y;
            s.alpha_z[i] = new_alpha_z;

            // Update orientation (quaternion integration)
            float omega_mag = std::sqrt(s.omega_x[i]*s.omega_x[i] +
                                        s.omega_y[i]*s.omega_y[i] +
                                        s.omega_z[i]*s.omega_z[i]);
            if (omega_mag > 1e-8f) {
                float half_angle = omega_mag * dt * 0.5f;
                float s_ha = std::sin(half_angle) / omega_mag;
                float c_ha = std::cos(half_angle);

                float dqw = c_ha;
                float dqx = s.omega_x[i] * s_ha;
                float dqy = s.omega_y[i] * s_ha;
                float dqz = s.omega_z[i] * s_ha;

                // q = q * dq
                float nqw = qw*dqw - qx*dqx - qy*dqy - qz*dqz;
                float nqx = qw*dqx + qx*dqw + qy*dqz - qz*dqy;
                float nqy = qw*dqy - qx*dqz + qy*dqw + qz*dqx;
                float nqz = qw*dqz + qx*dqy - qy*dqx + qz*dqw;

                // Normalize
                float norm = std::sqrt(nqw*nqw + nqx*nqx + nqy*nqy + nqz*nqz);
                s.quat_w[i] = nqw / norm;
                s.quat_x[i] = nqx / norm;
                s.quat_y[i] = nqy / norm;
                s.quat_z[i] = nqz / norm;
            }

            // Ground collision
            if (s.pos_z[i] > 0.0f) {
                s.pos_z[i] = 0.0f;
                if (s.vel_z[i] > 0.1f) {
                    s.vel_z[i] = -s.vel_z[i] * 0.5f;
                } else {
                    s.is_grounded[i] = 1.0f;
                    s.vel_x[i] = s.vel_y[i] = s.vel_z[i] = 0.0f;
                }
            }
        }
    }

    // ========================================================================
    // AVX2 SIMD VERSION (8 drones at once)
    // ========================================================================
#ifdef __AVX2__
    void step_avx(DroneSwarmSoA& s, float dt) {
        const __m256 v_alpha = _mm256_set1_ps(dt / (params.motor_tc + dt));
        const __m256 v_inv_mass = _mm256_set1_ps(1.0f / params.mass);
        const __m256 v_half_dt = _mm256_set1_ps(0.5f * dt);
        const __m256 v_dt = _mm256_set1_ps(dt);
        const __m256 v_gravity = _mm256_set1_ps(params.mass * GRAVITY);
        const __m256 v_max_thrust = _mm256_set1_ps(params.max_thrust);
        const __m256 v_max_torque = _mm256_set1_ps(params.max_torque);
        const __m256 v_rotor_y0 = _mm256_set1_ps(params.rotor_y[0]);
        const __m256 v_rotor_y1 = _mm256_set1_ps(params.rotor_y[1]);
        const __m256 v_rotor_x0 = _mm256_set1_ps(params.rotor_x[0]);
        const __m256 v_rotor_x2 = _mm256_set1_ps(params.rotor_x[2]);
        const __m256 v_Ixx_inv = _mm256_set1_ps(1.0f / params.Ixx);
        const __m256 v_Iyy_inv = _mm256_set1_ps(1.0f / params.Iyy);
        const __m256 v_Izz_inv = _mm256_set1_ps(1.0f / params.Izz);
        const __m256 v_Ixx = _mm256_set1_ps(params.Ixx);
        const __m256 v_Iyy = _mm256_set1_ps(params.Iyy);
        const __m256 v_Izz = _mm256_set1_ps(params.Izz);
        const __m256 v_two = _mm256_set1_ps(2.0f);
        const __m256 v_half = _mm256_set1_ps(0.5f);
        const __m256 v_zero = _mm256_setzero_ps();
        const __m256 v_one = _mm256_set1_ps(1.0f);
        const __m256 v_eps = _mm256_set1_ps(1e-8f);

        size_t n = (s.count / SIMD_WIDTH) * SIMD_WIDTH;

        for (size_t i = 0; i < n; i += SIMD_WIDTH) {
            // Load motor inputs and filtered values
            __m256 m0 = _mm256_load_ps(&s.motor_0[i]);
            __m256 m1 = _mm256_load_ps(&s.motor_1[i]);
            __m256 m2 = _mm256_load_ps(&s.motor_2[i]);
            __m256 m3 = _mm256_load_ps(&s.motor_3[i]);
            __m256 mf0 = _mm256_load_ps(&s.motor_f0[i]);
            __m256 mf1 = _mm256_load_ps(&s.motor_f1[i]);
            __m256 mf2 = _mm256_load_ps(&s.motor_f2[i]);
            __m256 mf3 = _mm256_load_ps(&s.motor_f3[i]);

            // Motor filter: mf += alpha * (m - mf)
            mf0 = _mm256_fmadd_ps(v_alpha, _mm256_sub_ps(m0, mf0), mf0);
            mf1 = _mm256_fmadd_ps(v_alpha, _mm256_sub_ps(m1, mf1), mf1);
            mf2 = _mm256_fmadd_ps(v_alpha, _mm256_sub_ps(m2, mf2), mf2);
            mf3 = _mm256_fmadd_ps(v_alpha, _mm256_sub_ps(m3, mf3), mf3);

            _mm256_store_ps(&s.motor_f0[i], mf0);
            _mm256_store_ps(&s.motor_f1[i], mf1);
            _mm256_store_ps(&s.motor_f2[i], mf2);
            _mm256_store_ps(&s.motor_f3[i], mf3);

            // Compute thrusts
            __m256 t0 = _mm256_mul_ps(mf0, v_max_thrust);
            __m256 t1 = _mm256_mul_ps(mf1, v_max_thrust);
            __m256 t2 = _mm256_mul_ps(mf2, v_max_thrust);
            __m256 t3 = _mm256_mul_ps(mf3, v_max_thrust);
            __m256 thrust_total = _mm256_add_ps(_mm256_add_ps(t0, t1), _mm256_add_ps(t2, t3));

            // Load quaternion
            __m256 qw = _mm256_load_ps(&s.quat_w[i]);
            __m256 qx = _mm256_load_ps(&s.quat_x[i]);
            __m256 qy = _mm256_load_ps(&s.quat_y[i]);
            __m256 qz = _mm256_load_ps(&s.quat_z[i]);

            // Rotate thrust (0,0,-thrust) to world frame
            __m256 fz_body = _mm256_sub_ps(v_zero, thrust_total);

            // Quaternion rotation for (0,0,fz)
            __m256 t2_ = _mm256_mul_ps(qw, qx);
            __m256 t3_ = _mm256_mul_ps(qw, qy);
            __m256 t5_ = _mm256_sub_ps(v_zero, _mm256_mul_ps(qx, qx));
            __m256 t7_ = _mm256_mul_ps(qx, qz);
            __m256 t8_ = _mm256_sub_ps(v_zero, _mm256_mul_ps(qy, qy));
            __m256 t9_ = _mm256_mul_ps(qy, qz);

            __m256 fx_world = _mm256_mul_ps(v_two, _mm256_mul_ps(_mm256_add_ps(t3_, t7_), fz_body));
            __m256 fy_world = _mm256_mul_ps(v_two, _mm256_mul_ps(_mm256_sub_ps(t9_, t2_), fz_body));
            __m256 fz_world = _mm256_fmadd_ps(v_two, _mm256_mul_ps(_mm256_add_ps(t5_, t8_), fz_body), fz_body);

            // Add gravity
            fz_world = _mm256_add_ps(fz_world, v_gravity);

            // Linear acceleration
            __m256 new_acc_x = _mm256_mul_ps(fx_world, v_inv_mass);
            __m256 new_acc_y = _mm256_mul_ps(fy_world, v_inv_mass);
            __m256 new_acc_z = _mm256_mul_ps(fz_world, v_inv_mass);

            // Load current values
            __m256 vel_x = _mm256_load_ps(&s.vel_x[i]);
            __m256 vel_y = _mm256_load_ps(&s.vel_y[i]);
            __m256 vel_z = _mm256_load_ps(&s.vel_z[i]);
            __m256 acc_x = _mm256_load_ps(&s.acc_x[i]);
            __m256 acc_y = _mm256_load_ps(&s.acc_y[i]);
            __m256 acc_z = _mm256_load_ps(&s.acc_z[i]);
            __m256 pos_x = _mm256_load_ps(&s.pos_x[i]);
            __m256 pos_y = _mm256_load_ps(&s.pos_y[i]);
            __m256 pos_z = _mm256_load_ps(&s.pos_z[i]);

            // Semi-implicit Euler: vel += (acc + new_acc) * half_dt
            vel_x = _mm256_fmadd_ps(_mm256_add_ps(acc_x, new_acc_x), v_half_dt, vel_x);
            vel_y = _mm256_fmadd_ps(_mm256_add_ps(acc_y, new_acc_y), v_half_dt, vel_y);
            vel_z = _mm256_fmadd_ps(_mm256_add_ps(acc_z, new_acc_z), v_half_dt, vel_z);

            // pos += vel * dt
            pos_x = _mm256_fmadd_ps(vel_x, v_dt, pos_x);
            pos_y = _mm256_fmadd_ps(vel_y, v_dt, pos_y);
            pos_z = _mm256_fmadd_ps(vel_z, v_dt, pos_z);

            // Store
            _mm256_store_ps(&s.vel_x[i], vel_x);
            _mm256_store_ps(&s.vel_y[i], vel_y);
            _mm256_store_ps(&s.vel_z[i], vel_z);
            _mm256_store_ps(&s.acc_x[i], new_acc_x);
            _mm256_store_ps(&s.acc_y[i], new_acc_y);
            _mm256_store_ps(&s.acc_z[i], new_acc_z);
            _mm256_store_ps(&s.pos_x[i], pos_x);
            _mm256_store_ps(&s.pos_y[i], pos_y);
            _mm256_store_ps(&s.pos_z[i], pos_z);

            // Angular dynamics (torques)
            __m256 torque_x = _mm256_fmadd_ps(_mm256_add_ps(t0, t3), v_rotor_y0,
                              _mm256_mul_ps(_mm256_add_ps(t1, t2), v_rotor_y1));
            __m256 torque_y = _mm256_sub_ps(v_zero,
                              _mm256_fmadd_ps(_mm256_add_ps(t0, t1), v_rotor_x0,
                              _mm256_mul_ps(_mm256_add_ps(t2, t3), v_rotor_x2)));
            __m256 torque_z = _mm256_mul_ps(
                _mm256_sub_ps(_mm256_add_ps(mf0, mf2), _mm256_add_ps(mf1, mf3)),
                v_max_torque);

            __m256 omega_x = _mm256_load_ps(&s.omega_x[i]);
            __m256 omega_y = _mm256_load_ps(&s.omega_y[i]);
            __m256 omega_z = _mm256_load_ps(&s.omega_z[i]);
            __m256 alpha_x = _mm256_load_ps(&s.alpha_x[i]);
            __m256 alpha_y = _mm256_load_ps(&s.alpha_y[i]);
            __m256 alpha_z = _mm256_load_ps(&s.alpha_z[i]);

            // L = I * omega
            __m256 Lx = _mm256_mul_ps(v_Ixx, omega_x);
            __m256 Ly = _mm256_mul_ps(v_Iyy, omega_y);
            __m256 Lz = _mm256_mul_ps(v_Izz, omega_z);

            // cross = omega x L
            __m256 cross_x = _mm256_sub_ps(_mm256_mul_ps(omega_y, Lz), _mm256_mul_ps(omega_z, Ly));
            __m256 cross_y = _mm256_sub_ps(_mm256_mul_ps(omega_z, Lx), _mm256_mul_ps(omega_x, Lz));
            __m256 cross_z = _mm256_sub_ps(_mm256_mul_ps(omega_x, Ly), _mm256_mul_ps(omega_y, Lx));

            __m256 new_alpha_x = _mm256_mul_ps(_mm256_sub_ps(torque_x, cross_x), v_Ixx_inv);
            __m256 new_alpha_y = _mm256_mul_ps(_mm256_sub_ps(torque_y, cross_y), v_Iyy_inv);
            __m256 new_alpha_z = _mm256_mul_ps(_mm256_sub_ps(torque_z, cross_z), v_Izz_inv);

            omega_x = _mm256_fmadd_ps(_mm256_add_ps(alpha_x, new_alpha_x), v_half_dt, omega_x);
            omega_y = _mm256_fmadd_ps(_mm256_add_ps(alpha_y, new_alpha_y), v_half_dt, omega_y);
            omega_z = _mm256_fmadd_ps(_mm256_add_ps(alpha_z, new_alpha_z), v_half_dt, omega_z);

            _mm256_store_ps(&s.omega_x[i], omega_x);
            _mm256_store_ps(&s.omega_y[i], omega_y);
            _mm256_store_ps(&s.omega_z[i], omega_z);
            _mm256_store_ps(&s.alpha_x[i], new_alpha_x);
            _mm256_store_ps(&s.alpha_y[i], new_alpha_y);
            _mm256_store_ps(&s.alpha_z[i], new_alpha_z);

            // Quaternion update (scalar fallback for now - complex SIMD)
            // TODO: Vectorize quaternion integration
        }

        // Handle remaining drones with scalar
        for (size_t i = n; i < s.count; ++i) {
            // Simple scalar update for stragglers
            s.motor_f0[i] += (dt/(params.motor_tc+dt)) * (s.motor_0[i] - s.motor_f0[i]);
            s.motor_f1[i] += (dt/(params.motor_tc+dt)) * (s.motor_1[i] - s.motor_f1[i]);
            s.motor_f2[i] += (dt/(params.motor_tc+dt)) * (s.motor_2[i] - s.motor_f2[i]);
            s.motor_f3[i] += (dt/(params.motor_tc+dt)) * (s.motor_3[i] - s.motor_f3[i]);
        }
    }
#endif

    // ========================================================================
    // MAIN STEP FUNCTION (auto-selects best implementation)
    // ========================================================================
    void step(DroneSwarmSoA& s, float dt) {
#ifdef __AVX2__
        if (s.count >= SIMD_WIDTH) {
            step_avx(s, dt);
        } else {
            step_scalar(s, dt);
        }
#else
        step_scalar(s, dt);
#endif
    }

    // ========================================================================
    // PARALLEL STEP (OpenMP)
    // ========================================================================
#ifdef _OPENMP
    void step_parallel(DroneSwarmSoA& s, float dt, int num_threads = 0) {
        if (num_threads > 0) {
            omp_set_num_threads(num_threads);
        }

        const float alpha = dt / (params.motor_tc + dt);
        const float inv_mass = 1.0f / params.mass;
        const float half_dt = 0.5f * dt;

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(s.count); ++i) {
            // Motor filter
            s.motor_f0[i] += alpha * (s.motor_0[i] - s.motor_f0[i]);
            s.motor_f1[i] += alpha * (s.motor_1[i] - s.motor_f1[i]);
            s.motor_f2[i] += alpha * (s.motor_2[i] - s.motor_f2[i]);
            s.motor_f3[i] += alpha * (s.motor_3[i] - s.motor_f3[i]);

            // Compute forces (simplified - full version in scalar)
            float thrust = (s.motor_f0[i] + s.motor_f1[i] + s.motor_f2[i] + s.motor_f3[i])
                          * params.max_thrust;

            float new_acc_z = (GRAVITY - thrust * inv_mass);

            s.vel_z[i] += (s.acc_z[i] + new_acc_z) * half_dt;
            s.acc_z[i] = new_acc_z;
            s.pos_z[i] += s.vel_z[i] * dt;
        }
    }
#endif

    // ========================================================================
    // UTILITY: Set hover throttle for all drones
    // ========================================================================
    void setHoverThrottle(DroneSwarmSoA& s) {
        float hover = params.hover_throttle();
        for (size_t i = 0; i < s.count; ++i) {
            s.motor_0[i] = s.motor_1[i] = s.motor_2[i] = s.motor_3[i] = hover;
        }
    }
};

} // namespace swiftsim
