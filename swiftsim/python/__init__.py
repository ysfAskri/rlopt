"""
SwiftSim - High-performance drone swarm physics simulator for RL training

Usage:
    import swiftsim

    # Create a swarm of 1000 drones
    swarm = swiftsim.DroneSwarm(1000)
    swarm.reset()

    # Create physics engine
    physics = swiftsim.SwarmPhysics()
    physics.set_hover_throttle(swarm)

    # Step physics
    for _ in range(1000):
        physics.step(swarm, 0.001)

    # Get positions as numpy array
    positions = swarm.get_positions()  # (N, 3) array

Gymnasium Environments:
    from swiftsim.swiftsim_env import DroneHoverEnv, DroneSwarmEnv

    # Single drone hover task
    env = DroneHoverEnv()
    obs, info = env.reset()
    obs, reward, term, trunc, info = env.step([0.58, 0.58, 0.58, 0.58])

    # Vectorized swarm (1000 parallel drones)
    env = DroneSwarmEnv(n_envs=1000)
    obs, info = env.reset()
    actions = np.random.uniform(0.5, 0.7, (1000, 4))
    obs, reward, term, trunc, info = env.step(actions)
"""

from swiftsim import (
    DroneSwarm,
    SwarmPhysics,
    SwarmParams,
    HAS_AVX2,
    HAS_OPENMP,
    __version__,
)

__all__ = [
    "DroneSwarm",
    "SwarmPhysics",
    "SwarmParams",
    "HAS_AVX2",
    "HAS_OPENMP",
    "__version__",
]
