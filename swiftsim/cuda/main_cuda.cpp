// SwiftSim CUDA Training with Raylib UI
// Full C++ stack: CUDA Physics + LibTorch PPO + Raylib/ImGui

#include "swarm_physics_cuda.cuh"
#include "ppo_agent.hpp"
#include "../ui/visualizer_raylib.hpp"

#include <iostream>
#include <chrono>
#include <thread>

using namespace swiftsim;
using Clock = std::chrono::high_resolution_clock;

// ============================================================================
// CONFIGURATION
// ============================================================================
struct Config {
    // Environment
    size_t n_envs = 1000;           // Number of parallel drones
    float dt = 0.01f;               // Physics timestep
    int sub_steps = 10;             // Sub-steps per env step
    float target_altitude = 10.0f;

    // PPO
    size_t buffer_size = 2048;      // Steps before update
    float learning_rate = 3e-4f;
    float gamma = 0.99f;
    float clip_range = 0.2f;
    float entropy_coef = 0.01f;

    // Training
    int total_timesteps = 10000000;
    int log_interval = 10;
    int save_interval = 100000;
    bool render = true;

    // Observation/Action dims
    int obs_dim = 13;               // pos(3) + vel(3) + quat(4) + omega(3)
    int act_dim = 4;                // 4 motors
};

// ============================================================================
// MAIN TRAINING LOOP
// ============================================================================
int main(int argc, char** argv) {
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║        SwiftSim CUDA - PPO Training with Raylib UI           ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";

    // Check CUDA
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "║  GPU: " << prop.name << "\n";
    std::cout << "║  CUDA Cores: " << prop.multiProcessorCount * 128 << "\n";
    std::cout << "║  Memory: " << prop.totalGlobalMem / (1024*1024) << " MB\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    Config cfg;

    // Parse args
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--no-render") cfg.render = false;
        else if (arg == "--envs" && i + 1 < argc) cfg.n_envs = std::stoi(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc) cfg.learning_rate = std::stof(argv[++i]);
    }

    std::cout << "Configuration:\n";
    std::cout << "  Environments: " << cfg.n_envs << "\n";
    std::cout << "  Buffer size: " << cfg.buffer_size << "\n";
    std::cout << "  Learning rate: " << cfg.learning_rate << "\n";
    std::cout << "  Render: " << (cfg.render ? "yes" : "no") << "\n\n";

    // ========================================================================
    // INITIALIZE
    // ========================================================================

    // Physics (CUDA)
    cuda::SwarmPhysicsCUDA physics;
    physics.params.target_altitude = cfg.target_altitude;
    physics.init(cfg.n_envs);
    physics.reset(cfg.target_altitude);

    std::cout << "Physics initialized on GPU\n";

    // PPO Agent (LibTorch)
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "PyTorch device: " << device << "\n";

    rl::PPOAgent agent(cfg.obs_dim, cfg.act_dim, cfg.n_envs, cfg.buffer_size, device);
    agent.learning_rate = cfg.learning_rate;
    agent.clip_range = cfg.clip_range;
    agent.ent_coef = cfg.entropy_coef;

    std::cout << "PPO Agent initialized\n";

    // Visualizer (Raylib + ImGui)
    ui::SwiftSimVisualizer viz;
    if (cfg.render) {
        viz.init();
        viz.target_altitude = cfg.target_altitude;
        std::cout << "Visualizer initialized\n";
    }

    // Host buffers
    std::vector<float> h_observations(cfg.n_envs * cfg.obs_dim);
    std::vector<float> h_actions(cfg.n_envs * cfg.act_dim);
    std::vector<float> h_rewards(cfg.n_envs);
    std::vector<float> h_dones(cfg.n_envs);
    std::vector<float> h_positions(cfg.n_envs * 3);
    std::vector<float> h_quaternions(cfg.n_envs * 4);

    // ========================================================================
    // TRAINING LOOP
    // ========================================================================

    int total_steps = 0;
    int episodes = 0;
    int updates = 0;
    auto start_time = Clock::now();
    auto last_log_time = start_time;

    std::cout << "\nStarting training...\n\n";

    while (total_steps < cfg.total_timesteps) {
        // Check if we should stop (window closed)
        if (cfg.render && !viz.should_continue()) {
            break;
        }

        // Get observations
        physics.get_observations(h_observations.data());
        auto obs_tensor = torch::from_blob(h_observations.data(),
            {(int64_t)cfg.n_envs, cfg.obs_dim}, torch::kFloat32).to(device);

        // Select actions
        auto actions_tensor = agent.select_action(obs_tensor);

        // Copy actions to host
        auto actions_cpu = actions_tensor.to(torch::kCPU);
        std::memcpy(h_actions.data(), actions_cpu.data_ptr<float>(),
                   cfg.n_envs * cfg.act_dim * sizeof(float));

        // Set actions in physics
        physics.set_actions(h_actions.data());

        // Step physics (multiple sub-steps)
        for (int s = 0; s < cfg.sub_steps; s++) {
            physics.step(cfg.dt);
        }

        // Compute rewards
        physics.compute_rewards(-cfg.target_altitude);  // NED: negative altitude

        // Get rewards and dones
        physics.get_rewards(h_rewards.data());
        physics.get_dones(h_dones.data());

        // Convert to tensors
        auto rewards_tensor = torch::from_blob(h_rewards.data(),
            {(int64_t)cfg.n_envs, 1}, torch::kFloat32);
        auto dones_tensor = torch::from_blob(h_dones.data(),
            {(int64_t)cfg.n_envs, 1}, torch::kFloat32);

        // Collect rollout
        agent.collect_rollout(obs_tensor, actions_tensor, rewards_tensor, dones_tensor);

        total_steps += cfg.n_envs;

        // Count episodes
        for (size_t i = 0; i < cfg.n_envs; i++) {
            if (h_dones[i] > 0.5f) {
                episodes++;
                // Reset this drone
                // (In CUDA kernel, we could do selective reset)
            }
        }

        // Update policy when buffer is full
        if (agent.buffer->full) {
            physics.get_observations(h_observations.data());
            auto last_obs = torch::from_blob(h_observations.data(),
                {(int64_t)cfg.n_envs, cfg.obs_dim}, torch::kFloat32).to(device);

            auto [pg_loss, vf_loss, entropy] = agent.update(last_obs);
            updates++;

            viz.stats.add_losses(pg_loss, vf_loss, entropy);
        }

        // Update stats
        float mean_reward = 0;
        for (size_t i = 0; i < cfg.n_envs; i++) {
            mean_reward += h_rewards[i];
        }
        mean_reward /= cfg.n_envs;
        viz.stats.add_reward(mean_reward);
        viz.stats.total_steps = total_steps;
        viz.stats.episodes = episodes;

        // Logging
        auto now = Clock::now();
        float elapsed = std::chrono::duration<float>(now - last_log_time).count();
        if (elapsed > 1.0f) {  // Log every second
            float total_elapsed = std::chrono::duration<float>(now - start_time).count();
            float sps = total_steps / total_elapsed;
            viz.stats.physics_rate = sps * cfg.sub_steps / 1e6f;

            std::cout << "Steps: " << total_steps
                      << " | Episodes: " << episodes
                      << " | Mean Reward: " << mean_reward
                      << " | SPS: " << (int)sps
                      << " | Physics: " << viz.stats.physics_rate << " M/s"
                      << "\n";

            last_log_time = now;
        }

        // Render
        if (cfg.render && !viz.paused) {
            // Get positions and quaternions for rendering
            // (Would need to add kernel for this, using observations for now)
            for (size_t i = 0; i < std::min(cfg.n_envs, (size_t)100); i++) {
                h_positions[i * 3 + 0] = h_observations[i * cfg.obs_dim + 0];
                h_positions[i * 3 + 1] = h_observations[i * cfg.obs_dim + 1];
                h_positions[i * 3 + 2] = h_observations[i * cfg.obs_dim + 2];
                h_quaternions[i * 4 + 0] = h_observations[i * cfg.obs_dim + 6];
                h_quaternions[i * 4 + 1] = h_observations[i * cfg.obs_dim + 7];
                h_quaternions[i * 4 + 2] = h_observations[i * cfg.obs_dim + 8];
                h_quaternions[i * 4 + 3] = h_observations[i * cfg.obs_dim + 9];
            }

            viz.begin_frame();
            viz.draw_drones(h_positions.data(), h_quaternions.data(),
                           std::min((int)cfg.n_envs, 100));
            viz.draw_imgui();
            viz.end_frame();

            // Sync visualization settings back
            cfg.target_altitude = viz.target_altitude;
            agent.learning_rate = viz.learning_rate;
            agent.clip_range = viz.clip_range;
            agent.ent_coef = viz.entropy_coef;
        }

        // Save periodically
        if (total_steps % cfg.save_interval == 0) {
            agent.save("swiftsim_ppo_" + std::to_string(total_steps) + ".pt");
            std::cout << "Model saved at step " << total_steps << "\n";
        }
    }

    // ========================================================================
    // CLEANUP
    // ========================================================================

    auto total_time = std::chrono::duration<float>(Clock::now() - start_time).count();

    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    Training Complete                          ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Total Steps: " << total_steps << "\n";
    std::cout << "║  Total Episodes: " << episodes << "\n";
    std::cout << "║  Total Updates: " << updates << "\n";
    std::cout << "║  Total Time: " << total_time << " seconds\n";
    std::cout << "║  Average SPS: " << total_steps / total_time << "\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";

    // Final save
    agent.save("swiftsim_ppo_final.pt");

    if (cfg.render) {
        viz.close();
    }

    return 0;
}
