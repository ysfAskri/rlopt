// SwiftSim CUDA Benchmark
// Pure CUDA physics performance test

#include "swarm_physics_cuda.cuh"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

using namespace swiftsim::cuda;
using Clock = std::chrono::high_resolution_clock;

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           SwiftSim CUDA Physics Benchmark                    ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    // Check CUDA
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "CUDA Cores: " << prop.multiProcessorCount * 128 << "\n";
    std::cout << "Memory: " << prop.totalGlobalMem / (1024*1024) << " MB\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n\n";

    // Test configurations
    std::vector<size_t> drone_counts = {1000, 10000, 100000, 1000000};
    int steps = 1000;
    float dt = 0.001f;

    std::cout << "Benchmarking physics step (" << steps << " steps)...\n\n";
    std::cout << "  Drones       Time (ms)    Rate (M/s)    Speedup vs CPU\n";
    std::cout << "  ──────────────────────────────────────────────────────\n";

    // CPU baseline (from previous benchmarks): ~7 M/s with SoA
    float cpu_baseline = 7.0f;  // Million drone-steps/s

    for (size_t n : drone_counts) {
        SwarmPhysicsCUDA physics;
        physics.init(n);
        physics.reset(10.0f);

        // Warmup
        for (int i = 0; i < 10; i++) {
            physics.step(dt);
        }
        physics.sync();

        // Benchmark
        auto start = Clock::now();

        for (int i = 0; i < steps; i++) {
            physics.step(dt);
        }
        physics.sync();

        auto end = Clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        double rate = (n * steps) / elapsed / 1e6;  // Million drone-steps/s
        double speedup = rate / cpu_baseline;

        std::cout << "  " << std::setw(8) << n
                  << "     " << std::setw(8) << std::fixed << std::setprecision(2) << elapsed * 1000
                  << "      " << std::setw(8) << std::setprecision(1) << rate
                  << "         " << std::setw(5) << std::setprecision(0) << speedup << "x\n";
    }

    std::cout << "\n";

    // Full step test (physics + rewards)
    std::cout << "Full RL step (physics + rewards + observations)...\n\n";

    size_t n = 100000;
    SwarmPhysicsCUDA physics;
    physics.init(n);
    physics.reset(10.0f);

    std::vector<float> h_obs(n * 13);
    std::vector<float> h_actions(n * 4, 0.58f);  // Hover throttle
    std::vector<float> h_rewards(n);
    std::vector<float> h_dones(n);

    // Warmup
    for (int i = 0; i < 10; i++) {
        physics.set_actions(h_actions.data());
        physics.step(dt);
        physics.compute_rewards(-10.0f);
        physics.get_observations(h_obs.data());
        physics.get_rewards(h_rewards.data());
        physics.get_dones(h_dones.data());
    }
    physics.sync();

    // Benchmark
    auto start = Clock::now();

    for (int i = 0; i < steps; i++) {
        physics.set_actions(h_actions.data());
        physics.step(dt);
        physics.compute_rewards(-10.0f);
        physics.get_observations(h_obs.data());
        physics.get_rewards(h_rewards.data());
    }
    physics.sync();

    auto end = Clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    double rate = (n * steps) / elapsed / 1e6;

    std::cout << "  100,000 drones x " << steps << " steps\n";
    std::cout << "  Time: " << elapsed * 1000 << " ms\n";
    std::cout << "  Rate: " << rate << " M drone-steps/s\n";
    std::cout << "  Per step: " << elapsed / steps * 1000 << " ms\n";

    // Show sample data
    std::cout << "\nSample drone state (drone 0):\n";
    std::cout << "  Position: (" << h_obs[0] << ", " << h_obs[1] << ", " << h_obs[2] << ")\n";
    std::cout << "  Velocity: (" << h_obs[3] << ", " << h_obs[4] << ", " << h_obs[5] << ")\n";
    std::cout << "  Reward: " << h_rewards[0] << "\n";

    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    Benchmark Complete                         ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";

    return 0;
}
