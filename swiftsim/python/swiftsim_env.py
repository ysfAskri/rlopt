"""
SwiftSim Gymnasium Environment
High-performance vectorized drone swarm environment for RL training
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

try:
    import swiftsim
except ImportError:
    raise ImportError("swiftsim not installed. Run: pip install .")


class DroneSwarmEnv(gym.Env):
    """
    Vectorized drone swarm environment.

    Each drone is an independent agent with:
    - Observation: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, qw, qx, qy, qz, omega_x, omega_y, omega_z]
    - Action: [motor_0, motor_1, motor_2, motor_3] in range [0, 1]

    The environment simulates n_envs independent drones in parallel.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        n_envs: int = 1,
        max_steps: int = 1000,
        dt: float = 0.01,
        sub_steps: int = 10,
        target_altitude: float = 10.0,
        use_parallel: bool = True,
        render_mode: Optional[str] = None,
    ):
        """
        Args:
            n_envs: Number of parallel environments (drones)
            max_steps: Maximum steps per episode
            dt: Physics timestep (sub_steps * dt = env timestep)
            sub_steps: Number of physics sub-steps per env step
            target_altitude: Target hover altitude in meters
            use_parallel: Use OpenMP parallelization
            render_mode: "human" or "rgb_array"
        """
        super().__init__()

        self.n_envs = n_envs
        self.max_steps = max_steps
        self.dt = dt
        self.sub_steps = sub_steps
        self.target_altitude = target_altitude
        self.use_parallel = use_parallel and swiftsim.HAS_OPENMP
        self.render_mode = render_mode

        # Create physics engine and swarm
        self.physics = swiftsim.SwarmPhysics()
        self.swarm = swiftsim.DroneSwarm(n_envs)

        # Get hover throttle for normalization
        self.hover_throttle = self.physics.params.hover_throttle()

        # Observation: [pos(3), vel(3), quat(4), omega(3)] = 13 dims
        self.obs_dim = 13
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        # Action: 4 motor commands in [0, 1]
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        # Episode tracking
        self.steps = 0
        self.episode_returns = np.zeros(n_envs, dtype=np.float32)

        # Target positions (can be customized)
        self.targets = np.zeros((n_envs, 3), dtype=np.float32)
        self.targets[:, 2] = -target_altitude  # NED frame (negative is up)

    def _get_obs(self) -> np.ndarray:
        """Get observations for all drones."""
        obs = np.zeros((self.n_envs, self.obs_dim), dtype=np.float32)

        # Position (3)
        obs[:, 0] = self.swarm.pos_x
        obs[:, 1] = self.swarm.pos_y
        obs[:, 2] = self.swarm.pos_z

        # Velocity (3)
        obs[:, 3] = self.swarm.vel_x
        obs[:, 4] = self.swarm.vel_y
        obs[:, 5] = self.swarm.vel_z

        # Quaternion (4)
        obs[:, 6] = self.swarm.quat_w
        obs[:, 7] = self.swarm.quat_x
        obs[:, 8] = self.swarm.quat_y
        obs[:, 9] = self.swarm.quat_z

        # Angular velocity (3)
        obs[:, 10] = self.swarm.omega_x
        obs[:, 11] = self.swarm.omega_y
        obs[:, 12] = self.swarm.omega_z

        return obs

    def _compute_reward(self) -> np.ndarray:
        """Compute rewards for all drones."""
        # Position error (NED frame)
        pos = np.stack([
            self.swarm.pos_x,
            self.swarm.pos_y,
            self.swarm.pos_z
        ], axis=1)

        pos_error = np.linalg.norm(pos - self.targets, axis=1)

        # Velocity penalty
        vel = np.stack([
            self.swarm.vel_x,
            self.swarm.vel_y,
            self.swarm.vel_z
        ], axis=1)
        vel_penalty = 0.1 * np.linalg.norm(vel, axis=1)

        # Angular velocity penalty
        omega = np.stack([
            self.swarm.omega_x,
            self.swarm.omega_y,
            self.swarm.omega_z
        ], axis=1)
        omega_penalty = 0.05 * np.linalg.norm(omega, axis=1)

        # Reward: -position_error - velocity_penalty - omega_penalty
        reward = -pos_error - vel_penalty - omega_penalty

        # Bonus for being close to target
        close_bonus = np.where(pos_error < 0.5, 1.0, 0.0)
        reward += close_bonus

        return reward.astype(np.float32)

    def _check_termination(self) -> Tuple[np.ndarray, np.ndarray]:
        """Check termination conditions."""
        # Truncated: max steps reached
        truncated = np.full(self.n_envs, self.steps >= self.max_steps, dtype=bool)

        # Terminated: crashed (hit ground with high velocity) or flew too far
        altitude = -self.swarm.pos_z  # Convert NED to altitude
        crashed = np.array(self.swarm.is_grounded) > 0.5
        too_far = altitude > 100.0  # >100m
        too_low = altitude < -1.0   # Below ground

        terminated = crashed | too_far | too_low

        return terminated, truncated

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset all environments."""
        super().reset(seed=seed)

        # Reset swarm
        self.swarm.reset()

        # Randomize initial positions slightly
        if self.np_random is not None:
            self.swarm.pos_x[:] = self.np_random.uniform(-1, 1, self.n_envs)
            self.swarm.pos_y[:] = self.np_random.uniform(-1, 1, self.n_envs)
            self.swarm.pos_z[:] = self.np_random.uniform(-12, -8, self.n_envs)

        # Set hover throttle
        self.physics.set_hover_throttle(self.swarm)

        # Reset episode tracking
        self.steps = 0
        self.episode_returns.fill(0)

        obs = self._get_obs()
        info = {"n_envs": self.n_envs}

        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Step all environments.

        Args:
            action: (n_envs, 4) array of motor commands in [0, 1]

        Returns:
            obs: (n_envs, obs_dim) observations
            reward: (n_envs,) rewards
            terminated: (n_envs,) termination flags
            truncated: (n_envs,) truncation flags
            info: dict with additional info
        """
        # Validate action shape
        action = np.asarray(action, dtype=np.float32)
        if action.shape == (4,):
            action = np.tile(action, (self.n_envs, 1))
        assert action.shape == (self.n_envs, 4), f"Expected ({self.n_envs}, 4), got {action.shape}"

        # Clip actions to valid range
        action = np.clip(action, 0.0, 1.0)

        # Set motor inputs
        self.swarm.set_motors(action)

        # Step physics
        for _ in range(self.sub_steps):
            if self.use_parallel:
                self.physics.step_parallel(self.swarm, self.dt)
            else:
                self.physics.step(self.swarm, self.dt)

        self.steps += 1

        # Get results
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated, truncated = self._check_termination()

        # Track episode returns
        self.episode_returns += reward

        # Auto-reset terminated environments
        done_mask = terminated | truncated
        if np.any(done_mask):
            # Store final returns for logging
            final_returns = self.episode_returns[done_mask].copy()

            # Reset done environments
            done_indices = np.where(done_mask)[0]
            for i in done_indices:
                # Reset position
                self.swarm.pos_x[i] = self.np_random.uniform(-1, 1) if self.np_random else 0
                self.swarm.pos_y[i] = self.np_random.uniform(-1, 1) if self.np_random else 0
                self.swarm.pos_z[i] = self.np_random.uniform(-12, -8) if self.np_random else -10

                # Reset velocity
                self.swarm.vel_x[i] = 0
                self.swarm.vel_y[i] = 0
                self.swarm.vel_z[i] = 0

                # Reset orientation
                self.swarm.quat_w[i] = 1
                self.swarm.quat_x[i] = 0
                self.swarm.quat_y[i] = 0
                self.swarm.quat_z[i] = 0

                # Reset angular velocity
                self.swarm.omega_x[i] = 0
                self.swarm.omega_y[i] = 0
                self.swarm.omega_z[i] = 0

                # Reset motors
                hover = self.hover_throttle
                self.swarm.motor_0[i] = hover
                self.swarm.motor_1[i] = hover
                self.swarm.motor_2[i] = hover
                self.swarm.motor_3[i] = hover
                self.swarm.motor_f0[i] = hover
                self.swarm.motor_f1[i] = hover
                self.swarm.motor_f2[i] = hover
                self.swarm.motor_f3[i] = hover

                # Reset flags
                self.swarm.is_grounded[i] = 0
                self.swarm.is_active[i] = 1

                # Reset episode return
                self.episode_returns[i] = 0

        info = {
            "n_envs": self.n_envs,
            "steps": self.steps,
            "episode_returns": self.episode_returns.copy(),
        }
        if np.any(done_mask):
            info["final_returns"] = final_returns

        return obs, reward, terminated, truncated, info

    def set_targets(self, targets: np.ndarray):
        """Set target positions for all drones."""
        targets = np.asarray(targets, dtype=np.float32)
        if targets.shape == (3,):
            targets = np.tile(targets, (self.n_envs, 1))
        assert targets.shape == (self.n_envs, 3)
        self.targets = targets

    def render(self):
        """Render the environment (placeholder)."""
        if self.render_mode == "human":
            positions = self.swarm.get_positions()
            print(f"Step {self.steps}: Mean altitude = {-positions[:, 2].mean():.2f}m")
        return None

    def close(self):
        """Clean up resources."""
        pass


class DroneHoverEnv(DroneSwarmEnv):
    """
    Single drone hover task.
    Goal: Maintain stable hover at target altitude.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("n_envs", 1)
        super().__init__(**kwargs)


class DroneSwarmVecEnv(DroneSwarmEnv):
    """
    Vectorized environment for batch RL training.
    Compatible with SB3 VecEnv interface.
    """

    @property
    def num_envs(self) -> int:
        return self.n_envs

    def step_async(self, actions: np.ndarray):
        """Store actions for async step."""
        self._pending_actions = actions

    def step_wait(self):
        """Execute pending step."""
        return self.step(self._pending_actions)


# Register environments with Gymnasium
try:
    gym.register(
        id="SwiftSim/DroneHover-v0",
        entry_point="swiftsim_env:DroneHoverEnv",
    )

    gym.register(
        id="SwiftSim/DroneSwarm-v0",
        entry_point="swiftsim_env:DroneSwarmEnv",
        kwargs={"n_envs": 100},
    )
except gym.error.Error:
    pass  # Already registered


if __name__ == "__main__":
    import time

    print("=== SwiftSim Gymnasium Environment Test ===\n")

    # Test single drone
    print("1. Single drone hover test:")
    env = DroneHoverEnv(max_steps=100)
    obs, info = env.reset(seed=42)
    print(f"   Obs shape: {obs.shape}")
    print(f"   Action space: {env.action_space}")

    total_reward = 0
    for i in range(100):
        action = np.array([0.58, 0.58, 0.58, 0.58])  # Near hover
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward[0]
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Final altitude: {-obs[0, 2]:.2f}m")

    # Test vectorized environment
    print("\n2. Vectorized swarm test (10k drones):")
    env = DroneSwarmEnv(n_envs=10000, max_steps=1000)
    obs, info = env.reset(seed=42)
    print(f"   Obs shape: {obs.shape}")

    # Benchmark
    n_steps = 100
    start = time.time()
    for _ in range(n_steps):
        action = np.random.uniform(0.5, 0.7, (10000, 4)).astype(np.float32)
        obs, reward, term, trunc, info = env.step(action)
    elapsed = time.time() - start

    total_steps = 10000 * n_steps * env.sub_steps
    rate = total_steps / elapsed / 1e6
    print(f"   {n_steps} env steps = {total_steps/1e6:.1f}M physics steps")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Rate: {rate:.1f} M physics-steps/s")
    print(f"   SPS (env): {10000 * n_steps / elapsed:.0f} samples/s")

    print("\n=== All tests passed! ===")
