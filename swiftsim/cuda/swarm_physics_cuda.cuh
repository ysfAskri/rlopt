// SwiftSim CUDA Physics Engine
// Massively parallel drone swarm simulation on GPU
// Target: 10B+ drone-steps/second

#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <stdexcept>

namespace swiftsim {
namespace cuda {

// ============================================================================
// CONSTANTS
// ============================================================================
constexpr float GRAVITY = 9.80665f;
constexpr int BLOCK_SIZE = 256;  // Threads per block

// ============================================================================
// CUDA ERROR CHECKING
// ============================================================================
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

// ============================================================================
// DRONE SWARM DATA (GPU Memory)
// ============================================================================
struct DroneSwarmGPU {
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

    // Angular velocity
    float* omega_x = nullptr;
    float* omega_y = nullptr;
    float* omega_z = nullptr;

    // Motor inputs & filtered
    float* motor_0 = nullptr;
    float* motor_1 = nullptr;
    float* motor_2 = nullptr;
    float* motor_3 = nullptr;
    float* motor_f0 = nullptr;
    float* motor_f1 = nullptr;
    float* motor_f2 = nullptr;
    float* motor_f3 = nullptr;

    // Rewards (for RL)
    float* rewards = nullptr;
    float* dones = nullptr;

    void allocate(size_t n) {
        if (n > capacity) {
            deallocate();
            capacity = n;
            count = n;

            size_t bytes = n * sizeof(float);

            CUDA_CHECK(cudaMalloc(&pos_x, bytes));
            CUDA_CHECK(cudaMalloc(&pos_y, bytes));
            CUDA_CHECK(cudaMalloc(&pos_z, bytes));
            CUDA_CHECK(cudaMalloc(&vel_x, bytes));
            CUDA_CHECK(cudaMalloc(&vel_y, bytes));
            CUDA_CHECK(cudaMalloc(&vel_z, bytes));
            CUDA_CHECK(cudaMalloc(&acc_x, bytes));
            CUDA_CHECK(cudaMalloc(&acc_y, bytes));
            CUDA_CHECK(cudaMalloc(&acc_z, bytes));
            CUDA_CHECK(cudaMalloc(&quat_w, bytes));
            CUDA_CHECK(cudaMalloc(&quat_x, bytes));
            CUDA_CHECK(cudaMalloc(&quat_y, bytes));
            CUDA_CHECK(cudaMalloc(&quat_z, bytes));
            CUDA_CHECK(cudaMalloc(&omega_x, bytes));
            CUDA_CHECK(cudaMalloc(&omega_y, bytes));
            CUDA_CHECK(cudaMalloc(&omega_z, bytes));
            CUDA_CHECK(cudaMalloc(&motor_0, bytes));
            CUDA_CHECK(cudaMalloc(&motor_1, bytes));
            CUDA_CHECK(cudaMalloc(&motor_2, bytes));
            CUDA_CHECK(cudaMalloc(&motor_3, bytes));
            CUDA_CHECK(cudaMalloc(&motor_f0, bytes));
            CUDA_CHECK(cudaMalloc(&motor_f1, bytes));
            CUDA_CHECK(cudaMalloc(&motor_f2, bytes));
            CUDA_CHECK(cudaMalloc(&motor_f3, bytes));
            CUDA_CHECK(cudaMalloc(&rewards, bytes));
            CUDA_CHECK(cudaMalloc(&dones, bytes));
        }
        count = n;
    }

    void deallocate() {
        if (capacity > 0) {
            cudaFree(pos_x); cudaFree(pos_y); cudaFree(pos_z);
            cudaFree(vel_x); cudaFree(vel_y); cudaFree(vel_z);
            cudaFree(acc_x); cudaFree(acc_y); cudaFree(acc_z);
            cudaFree(quat_w); cudaFree(quat_x); cudaFree(quat_y); cudaFree(quat_z);
            cudaFree(omega_x); cudaFree(omega_y); cudaFree(omega_z);
            cudaFree(motor_0); cudaFree(motor_1); cudaFree(motor_2); cudaFree(motor_3);
            cudaFree(motor_f0); cudaFree(motor_f1); cudaFree(motor_f2); cudaFree(motor_f3);
            cudaFree(rewards); cudaFree(dones);
            capacity = 0;
        }
    }

    // NOTE: No destructor! Destructor makes struct non-trivially-copyable
    // which breaks CUDA kernel parameter passing. Call deallocate() explicitly.
};

// ============================================================================
// PHYSICS PARAMETERS (Constant memory for fast access)
// ============================================================================
struct PhysicsParams {
    float mass = 1.0f;
    float max_thrust = 4.179446268f;
    float max_torque = 0.05f;
    float motor_tc = 0.005f;
    float Ixx = 0.0029125f;
    float Iyy = 0.0029125f;
    float Izz = 0.0055225f;
    float rotor_x[4] = {0.16f, 0.16f, -0.16f, -0.16f};
    float rotor_y[4] = {0.16f, -0.16f, -0.16f, 0.16f};

    // RL parameters
    float target_altitude = 10.0f;
    float pos_reward_scale = 1.0f;
    float vel_penalty_scale = 0.1f;
    float omega_penalty_scale = 0.05f;

    __host__ __device__ float hover_throttle() const {
        return (mass * GRAVITY / 4.0f) / max_thrust;
    }
};

// Constant memory for parameters (fast broadcast to all threads)
__constant__ PhysicsParams d_params;

// ============================================================================
// CUDA KERNELS
// ============================================================================

// Reset all drones to initial state
__global__ void kernel_reset(DroneSwarmGPU swarm, float init_altitude) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= swarm.count) return;

    swarm.pos_x[i] = 0.0f;
    swarm.pos_y[i] = 0.0f;
    swarm.pos_z[i] = -init_altitude;  // NED: negative is up

    swarm.vel_x[i] = 0.0f;
    swarm.vel_y[i] = 0.0f;
    swarm.vel_z[i] = 0.0f;

    swarm.acc_x[i] = 0.0f;
    swarm.acc_y[i] = 0.0f;
    swarm.acc_z[i] = 0.0f;

    swarm.quat_w[i] = 1.0f;
    swarm.quat_x[i] = 0.0f;
    swarm.quat_y[i] = 0.0f;
    swarm.quat_z[i] = 0.0f;

    swarm.omega_x[i] = 0.0f;
    swarm.omega_y[i] = 0.0f;
    swarm.omega_z[i] = 0.0f;

    float hover = d_params.hover_throttle();
    swarm.motor_0[i] = hover;
    swarm.motor_1[i] = hover;
    swarm.motor_2[i] = hover;
    swarm.motor_3[i] = hover;
    swarm.motor_f0[i] = hover;
    swarm.motor_f1[i] = hover;
    swarm.motor_f2[i] = hover;
    swarm.motor_f3[i] = hover;

    swarm.rewards[i] = 0.0f;
    swarm.dones[i] = 0.0f;
}

// Main physics step kernel
__global__ void kernel_step(DroneSwarmGPU swarm, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= swarm.count) return;

    // Motor filter
    float alpha = dt / (d_params.motor_tc + dt);
    swarm.motor_f0[i] += alpha * (swarm.motor_0[i] - swarm.motor_f0[i]);
    swarm.motor_f1[i] += alpha * (swarm.motor_1[i] - swarm.motor_f1[i]);
    swarm.motor_f2[i] += alpha * (swarm.motor_2[i] - swarm.motor_f2[i]);
    swarm.motor_f3[i] += alpha * (swarm.motor_3[i] - swarm.motor_f3[i]);

    // Compute thrust
    float t0 = swarm.motor_f0[i] * d_params.max_thrust;
    float t1 = swarm.motor_f1[i] * d_params.max_thrust;
    float t2 = swarm.motor_f2[i] * d_params.max_thrust;
    float t3 = swarm.motor_f3[i] * d_params.max_thrust;
    float thrust_total = t0 + t1 + t2 + t3;

    // Load quaternion
    float qw = swarm.quat_w[i];
    float qx = swarm.quat_x[i];
    float qy = swarm.quat_y[i];
    float qz = swarm.quat_z[i];

    // Rotate thrust from body to world frame
    // Body thrust is (0, 0, -thrust_total)
    float fz_body = -thrust_total;

    float t2_ = qw * qx;
    float t3_ = qw * qy;
    float t5_ = -qx * qx;
    float t7_ = qx * qz;
    float t8_ = -qy * qy;
    float t9_ = qy * qz;

    float fx_world = 2.0f * ((t3_ + t7_) * fz_body);
    float fy_world = 2.0f * ((t9_ - t2_) * fz_body);
    float fz_world = 2.0f * ((t5_ + t8_) * fz_body) + fz_body;

    // Add gravity
    fz_world += d_params.mass * GRAVITY;

    // Linear acceleration
    float inv_mass = 1.0f / d_params.mass;
    float new_acc_x = fx_world * inv_mass;
    float new_acc_y = fy_world * inv_mass;
    float new_acc_z = fz_world * inv_mass;

    // Semi-implicit Euler integration
    float half_dt = 0.5f * dt;
    swarm.vel_x[i] += (swarm.acc_x[i] + new_acc_x) * half_dt;
    swarm.vel_y[i] += (swarm.acc_y[i] + new_acc_y) * half_dt;
    swarm.vel_z[i] += (swarm.acc_z[i] + new_acc_z) * half_dt;

    swarm.acc_x[i] = new_acc_x;
    swarm.acc_y[i] = new_acc_y;
    swarm.acc_z[i] = new_acc_z;

    swarm.pos_x[i] += swarm.vel_x[i] * dt;
    swarm.pos_y[i] += swarm.vel_y[i] * dt;
    swarm.pos_z[i] += swarm.vel_z[i] * dt;

    // Torques
    float torque_x = (t0 + t3) * d_params.rotor_y[0] + (t1 + t2) * d_params.rotor_y[1];
    float torque_y = -(t0 + t1) * d_params.rotor_x[0] - (t2 + t3) * d_params.rotor_x[2];
    float torque_z = (swarm.motor_f0[i] - swarm.motor_f1[i] +
                      swarm.motor_f2[i] - swarm.motor_f3[i]) * d_params.max_torque;

    // Angular dynamics (Euler equations)
    float Lx = d_params.Ixx * swarm.omega_x[i];
    float Ly = d_params.Iyy * swarm.omega_y[i];
    float Lz = d_params.Izz * swarm.omega_z[i];

    float cross_x = swarm.omega_y[i] * Lz - swarm.omega_z[i] * Ly;
    float cross_y = swarm.omega_z[i] * Lx - swarm.omega_x[i] * Lz;
    float cross_z = swarm.omega_x[i] * Ly - swarm.omega_y[i] * Lx;

    float alpha_x = (torque_x - cross_x) / d_params.Ixx;
    float alpha_y = (torque_y - cross_y) / d_params.Iyy;
    float alpha_z = (torque_z - cross_z) / d_params.Izz;

    swarm.omega_x[i] += alpha_x * dt;
    swarm.omega_y[i] += alpha_y * dt;
    swarm.omega_z[i] += alpha_z * dt;

    // Quaternion integration
    float omega_mag = sqrtf(swarm.omega_x[i]*swarm.omega_x[i] +
                           swarm.omega_y[i]*swarm.omega_y[i] +
                           swarm.omega_z[i]*swarm.omega_z[i]);

    if (omega_mag > 1e-8f) {
        float half_angle = omega_mag * dt * 0.5f;
        float s_ha = sinf(half_angle) / omega_mag;
        float c_ha = cosf(half_angle);

        float dqw = c_ha;
        float dqx = swarm.omega_x[i] * s_ha;
        float dqy = swarm.omega_y[i] * s_ha;
        float dqz = swarm.omega_z[i] * s_ha;

        // q = q * dq
        float nqw = qw*dqw - qx*dqx - qy*dqy - qz*dqz;
        float nqx = qw*dqx + qx*dqw + qy*dqz - qz*dqy;
        float nqy = qw*dqy - qx*dqz + qy*dqw + qz*dqx;
        float nqz = qw*dqz + qx*dqy - qy*dqx + qz*dqw;

        // Normalize
        float norm = rsqrtf(nqw*nqw + nqx*nqx + nqy*nqy + nqz*nqz);
        swarm.quat_w[i] = nqw * norm;
        swarm.quat_x[i] = nqx * norm;
        swarm.quat_y[i] = nqy * norm;
        swarm.quat_z[i] = nqz * norm;
    }

    // Ground collision
    if (swarm.pos_z[i] > 0.0f) {
        swarm.pos_z[i] = 0.0f;
        swarm.vel_z[i] = 0.0f;
        swarm.dones[i] = 1.0f;  // Episode done
    }

    // Out of bounds check
    float altitude = -swarm.pos_z[i];
    if (altitude > 100.0f || altitude < -1.0f) {
        swarm.dones[i] = 1.0f;
    }
}

// Compute rewards kernel
__global__ void kernel_compute_rewards(DroneSwarmGPU swarm, float target_z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= swarm.count) return;

    // Position error
    float px = swarm.pos_x[i];
    float py = swarm.pos_y[i];
    float pz = swarm.pos_z[i] - target_z;  // Error from target altitude
    float pos_error = sqrtf(px*px + py*py + pz*pz);

    // Velocity penalty
    float vx = swarm.vel_x[i];
    float vy = swarm.vel_y[i];
    float vz = swarm.vel_z[i];
    float vel_mag = sqrtf(vx*vx + vy*vy + vz*vz);

    // Angular velocity penalty
    float ox = swarm.omega_x[i];
    float oy = swarm.omega_y[i];
    float oz = swarm.omega_z[i];
    float omega_mag = sqrtf(ox*ox + oy*oy + oz*oz);

    // Reward calculation
    float reward = -d_params.pos_reward_scale * pos_error
                   - d_params.vel_penalty_scale * vel_mag
                   - d_params.omega_penalty_scale * omega_mag;

    // Bonus for being close
    if (pos_error < 0.5f) {
        reward += 1.0f;
    }

    swarm.rewards[i] = reward;
}

// Set actions (motor inputs) from neural network output
__global__ void kernel_set_actions(DroneSwarmGPU swarm, const float* actions) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= swarm.count) return;

    // Actions are (N, 4) - 4 motor commands per drone
    int base = i * 4;
    swarm.motor_0[i] = fminf(fmaxf(actions[base + 0], 0.0f), 1.0f);
    swarm.motor_1[i] = fminf(fmaxf(actions[base + 1], 0.0f), 1.0f);
    swarm.motor_2[i] = fminf(fmaxf(actions[base + 2], 0.0f), 1.0f);
    swarm.motor_3[i] = fminf(fmaxf(actions[base + 3], 0.0f), 1.0f);
}

// Get observations for neural network
__global__ void kernel_get_observations(DroneSwarmGPU swarm, float* obs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= swarm.count) return;

    // Observation: [pos(3), vel(3), quat(4), omega(3)] = 13 dims
    int base = i * 13;
    obs[base + 0] = swarm.pos_x[i];
    obs[base + 1] = swarm.pos_y[i];
    obs[base + 2] = swarm.pos_z[i];
    obs[base + 3] = swarm.vel_x[i];
    obs[base + 4] = swarm.vel_y[i];
    obs[base + 5] = swarm.vel_z[i];
    obs[base + 6] = swarm.quat_w[i];
    obs[base + 7] = swarm.quat_x[i];
    obs[base + 8] = swarm.quat_y[i];
    obs[base + 9] = swarm.quat_z[i];
    obs[base + 10] = swarm.omega_x[i];
    obs[base + 11] = swarm.omega_y[i];
    obs[base + 12] = swarm.omega_z[i];
}

// ============================================================================
// HOST API
// ============================================================================
class SwarmPhysicsCUDA {
public:
    PhysicsParams params;
    DroneSwarmGPU swarm;

    // GPU buffers for RL
    float* d_observations = nullptr;
    float* d_actions = nullptr;
    size_t obs_size = 0;
    size_t act_size = 0;

    void init(size_t n_drones) {
        swarm.allocate(n_drones);

        // Allocate observation/action buffers
        obs_size = n_drones * 13;
        act_size = n_drones * 4;
        CUDA_CHECK(cudaMalloc(&d_observations, obs_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_actions, act_size * sizeof(float)));

        // Copy params to constant memory
        CUDA_CHECK(cudaMemcpyToSymbol(d_params, &params, sizeof(PhysicsParams)));
    }

    void reset(float altitude = 10.0f) {
        int blocks = (swarm.count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernel_reset<<<blocks, BLOCK_SIZE>>>(swarm, altitude);
        CUDA_CHECK(cudaGetLastError());
    }

    void step(float dt) {
        int blocks = (swarm.count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernel_step<<<blocks, BLOCK_SIZE>>>(swarm, dt);
        CUDA_CHECK(cudaGetLastError());
    }

    void compute_rewards(float target_altitude = -10.0f) {
        int blocks = (swarm.count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernel_compute_rewards<<<blocks, BLOCK_SIZE>>>(swarm, target_altitude);
        CUDA_CHECK(cudaGetLastError());
    }

    void set_actions(const float* h_actions) {
        CUDA_CHECK(cudaMemcpy(d_actions, h_actions, act_size * sizeof(float),
                              cudaMemcpyHostToDevice));
        int blocks = (swarm.count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernel_set_actions<<<blocks, BLOCK_SIZE>>>(swarm, d_actions);
        CUDA_CHECK(cudaGetLastError());
    }

    void get_observations(float* h_obs) {
        int blocks = (swarm.count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernel_get_observations<<<blocks, BLOCK_SIZE>>>(swarm, d_observations);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_obs, d_observations, obs_size * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    void get_rewards(float* h_rewards) {
        CUDA_CHECK(cudaMemcpy(h_rewards, swarm.rewards, swarm.count * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    void get_dones(float* h_dones) {
        CUDA_CHECK(cudaMemcpy(h_dones, swarm.dones, swarm.count * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    void sync() {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    ~SwarmPhysicsCUDA() {
        swarm.deallocate();  // Manually deallocate since DroneSwarmGPU has no destructor
        if (d_observations) cudaFree(d_observations);
        if (d_actions) cudaFree(d_actions);
    }
};

} // namespace cuda
} // namespace swiftsim
