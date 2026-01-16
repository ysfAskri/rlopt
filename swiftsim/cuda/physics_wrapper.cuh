// Physics Wrapper - Isolates CUDA from Raylib
#pragma once

#include <cstddef>

// Forward declaration - hides CUDA implementation
class PhysicsWrapper {
public:
    PhysicsWrapper();
    ~PhysicsWrapper();

    void init(size_t n_drones);
    void reset(float altitude);
    void set_actions(const float* actions);
    void step(float dt);
    void compute_rewards(float target_z);
    void get_observations(float* obs);
    void get_rewards(float* rewards);
    void sync();

    size_t count() const;

private:
    void* impl;  // Opaque pointer to SwarmPhysicsCUDA
};
