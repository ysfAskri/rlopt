// SwiftSim - SoA vs AoS Benchmark
// Compares performance of different physics implementations

#include "core/quadrotor_physics_complete.hpp"
#include "core/swarm_physics_soa.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

using namespace swiftsim;
using Clock = std::chrono::high_resolution_clock;

// ============================================================================
// AoS Benchmark (baseline)
// ============================================================================
double benchmark_aos(size_t n_drones, int steps) {
    QuadrotorPhysicsComplete physics;
    physics.enable_sensor_noise = false;
    physics.enable_drag = false; // Fair comparison

    std::vector<DroneState> drones(n_drones);
    float hover = physics.getHoverThrottle();

    for (auto& d : drones) {
        d.position.z = -10.0f;
        physics.setMotorInputs(d, hover, hover, hover, hover);
    }

    auto start = Clock::now();

    for (int s = 0; s < steps; ++s) {
        for (auto& d : drones) {
            physics.step(d, 0.001f);
        }
    }

    auto end = Clock::now();
    return std::chrono::duration<double>(end - start).count();
}

// ============================================================================
// SoA Scalar Benchmark
// ============================================================================
double benchmark_soa_scalar(size_t n_drones, int steps) {
    SwarmPhysicsSoA physics;
    DroneSwarmSoA swarm(n_drones);
    swarm.reset();
    physics.setHoverThrottle(swarm);

    auto start = Clock::now();

    for (int s = 0; s < steps; ++s) {
        physics.step_scalar(swarm, 0.001f);
    }

    auto end = Clock::now();
    return std::chrono::duration<double>(end - start).count();
}

// ============================================================================
// SoA SIMD Benchmark
// ============================================================================
double benchmark_soa_simd(size_t n_drones, int steps) {
    SwarmPhysicsSoA physics;
    DroneSwarmSoA swarm(n_drones);
    swarm.reset();
    physics.setHoverThrottle(swarm);

    auto start = Clock::now();

    for (int s = 0; s < steps; ++s) {
        physics.step(swarm, 0.001f); // Auto-selects AVX if available
    }

    auto end = Clock::now();
    return std::chrono::duration<double>(end - start).count();
}

// ============================================================================
// SoA Parallel Benchmark (OpenMP)
// ============================================================================
#ifdef _OPENMP
double benchmark_soa_parallel(size_t n_drones, int steps, int threads) {
    SwarmPhysicsSoA physics;
    DroneSwarmSoA swarm(n_drones);
    swarm.reset();
    physics.setHoverThrottle(swarm);

    auto start = Clock::now();

    for (int s = 0; s < steps; ++s) {
        physics.step_parallel(swarm, 0.001f, threads);
    }

    auto end = Clock::now();
    return std::chrono::duration<double>(end - start).count();
}
#endif

// ============================================================================
// MAIN
// ============================================================================
int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              SWIFTSIM - AoS vs SoA BENCHMARK                     ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════╣\n";
#ifdef __AVX2__
    std::cout << "║  AVX2: ENABLED                                                   ║\n";
#else
    std::cout << "║  AVX2: DISABLED                                                  ║\n";
#endif
#ifdef _OPENMP
    std::cout << "║  OpenMP: ENABLED                                                 ║\n";
#else
    std::cout << "║  OpenMP: DISABLED                                                ║\n";
#endif
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";

    std::vector<size_t> drone_counts = {100, 1000, 10000, 100000};
    int steps = 1000;

    // Header
    std::cout << std::setw(10) << "Drones"
              << std::setw(15) << "AoS (M/s)"
              << std::setw(15) << "SoA-Scalar"
              << std::setw(15) << "SoA-SIMD"
#ifdef _OPENMP
              << std::setw(15) << "SoA-Parallel"
#endif
              << std::setw(12) << "Speedup"
              << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (size_t n : drone_counts) {
        double t_aos = benchmark_aos(n, steps);
        double t_soa_scalar = benchmark_soa_scalar(n, steps);
        double t_soa_simd = benchmark_soa_simd(n, steps);

        double rate_aos = (n * steps) / t_aos / 1e6;
        double rate_soa_scalar = (n * steps) / t_soa_scalar / 1e6;
        double rate_soa_simd = (n * steps) / t_soa_simd / 1e6;

        double speedup = rate_soa_simd / rate_aos;

        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(10) << n
                  << std::setw(15) << rate_aos
                  << std::setw(15) << rate_soa_scalar
                  << std::setw(15) << rate_soa_simd;

#ifdef _OPENMP
        double t_soa_par = benchmark_soa_parallel(n, steps, 0);
        double rate_soa_par = (n * steps) / t_soa_par / 1e6;
        std::cout << std::setw(15) << rate_soa_par;
        speedup = rate_soa_par / rate_aos;
#endif

        std::cout << std::setw(10) << speedup << "x"
                  << "\n";
    }

    std::cout << std::string(80, '=') << "\n";

    // Memory comparison
    std::cout << "\n=== MEMORY LAYOUT ===\n";
    std::cout << "AoS DroneState size: " << sizeof(DroneState) << " bytes\n";
    std::cout << "SoA: 28 arrays x 4 bytes = 112 bytes/drone (no padding waste)\n\n";

    // Cache analysis
    std::cout << "=== CACHE EFFICIENCY ===\n";
    std::cout << "AoS: Processing vel_z requires loading full " << sizeof(DroneState) << " bytes/drone\n";
    std::cout << "     Cache line (64B) holds " << 64 / sizeof(DroneState) << " drones\n";
    std::cout << "SoA: Processing vel_z loads only 4 bytes/drone\n";
    std::cout << "     Cache line (64B) holds 16 consecutive vel_z values\n\n";

    // SIMD analysis
    std::cout << "=== SIMD UTILIZATION ===\n";
    std::cout << "AoS: Cannot vectorize (data interleaved)\n";
    std::cout << "SoA: AVX2 processes 8 drones per instruction\n";
    std::cout << "     Theoretical max speedup: 8x\n\n";

    return 0;
}
