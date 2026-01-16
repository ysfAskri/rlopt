// Physics Wrapper - CUDA Implementation
#include "physics_wrapper.cuh"
#include "swarm_physics_cuda.cuh"

using namespace swiftsim::cuda;

PhysicsWrapper::PhysicsWrapper() {
    impl = new SwarmPhysicsCUDA();
}

PhysicsWrapper::~PhysicsWrapper() {
    delete static_cast<SwarmPhysicsCUDA*>(impl);
}

void PhysicsWrapper::init(size_t n_drones) {
    static_cast<SwarmPhysicsCUDA*>(impl)->init(n_drones);
}

void PhysicsWrapper::reset(float altitude) {
    static_cast<SwarmPhysicsCUDA*>(impl)->reset(altitude);
}

void PhysicsWrapper::set_actions(const float* actions) {
    static_cast<SwarmPhysicsCUDA*>(impl)->set_actions(actions);
}

void PhysicsWrapper::step(float dt) {
    static_cast<SwarmPhysicsCUDA*>(impl)->step(dt);
}

void PhysicsWrapper::compute_rewards(float target_z) {
    static_cast<SwarmPhysicsCUDA*>(impl)->compute_rewards(target_z);
}

void PhysicsWrapper::get_observations(float* obs) {
    static_cast<SwarmPhysicsCUDA*>(impl)->get_observations(obs);
}

void PhysicsWrapper::get_rewards(float* rewards) {
    static_cast<SwarmPhysicsCUDA*>(impl)->get_rewards(rewards);
}

void PhysicsWrapper::sync() {
    static_cast<SwarmPhysicsCUDA*>(impl)->sync();
}

size_t PhysicsWrapper::count() const {
    return static_cast<SwarmPhysicsCUDA*>(impl)->swarm.count;
}
