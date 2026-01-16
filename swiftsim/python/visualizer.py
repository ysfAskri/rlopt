"""
SwiftSim 3D Visualizer with PyVista
Real-time drone swarm visualization for RL research
"""

import numpy as np
import pyvista as pv
from pyvista import examples
import time
import threading

try:
    import swiftsim
except ImportError:
    raise ImportError("swiftsim not installed. Run: pip install .")


class DroneVisualizer:
    """Real-time 3D visualization of drone swarm."""

    def __init__(self, n_drones: int = 10, drone_size: float = 0.3):
        self.n_drones = n_drones
        self.drone_size = drone_size

        # Create physics
        self.swarm = swiftsim.DroneSwarm(n_drones)
        self.physics = swiftsim.SwarmPhysics()
        self.swarm.reset()
        self.physics.set_hover_throttle(self.swarm)

        # Simulation state
        self.running = False
        self.dt = 0.01
        self.sim_speed = 1.0

        # Setup plotter
        self.plotter = pv.Plotter()
        self.plotter.set_background('black')

        # Create ground plane
        ground = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1),
                          i_size=50, j_size=50, i_resolution=10, j_resolution=10)
        self.plotter.add_mesh(ground, color='darkgreen', opacity=0.5)

        # Create grid
        for i in range(-25, 26, 5):
            line_x = pv.Line((i, -25, 0.01), (i, 25, 0.01))
            line_y = pv.Line((-25, i, 0.01), (25, i, 0.01))
            self.plotter.add_mesh(line_x, color='gray', line_width=1)
            self.plotter.add_mesh(line_y, color='gray', line_width=1)

        # Create drone meshes (simple quadrotor shape)
        self.drone_actors = []
        self.trail_points = [[] for _ in range(n_drones)]
        self.trail_actors = []

        for i in range(n_drones):
            # Drone body (sphere)
            drone = pv.Sphere(radius=drone_size, center=(0, 0, 10))

            # Color based on drone index
            color = self._get_drone_color(i)
            actor = self.plotter.add_mesh(drone, color=color, name=f'drone_{i}')
            self.drone_actors.append(actor)

        # Add axes
        self.plotter.add_axes()

        # Camera setup
        self.plotter.camera_position = [(30, 30, 30), (0, 0, 10), (0, 0, 1)]

        # Add text
        self.text_actor = self.plotter.add_text(
            "SwiftSim Visualizer\nPress SPACE to start/stop\nPress R to reset",
            position='upper_left', font_size=10, color='white'
        )

        # Key bindings
        self.plotter.add_key_event('space', self._toggle_simulation)
        self.plotter.add_key_event('r', self._reset_simulation)
        self.plotter.add_key_event('Up', lambda: self._adjust_throttle(0.02))
        self.plotter.add_key_event('Down', lambda: self._adjust_throttle(-0.02))

    def _get_drone_color(self, index: int) -> str:
        colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta',
                  'orange', 'purple', 'pink', 'lime']
        return colors[index % len(colors)]

    def _toggle_simulation(self):
        self.running = not self.running
        print(f"Simulation {'RUNNING' if self.running else 'PAUSED'}")

    def _reset_simulation(self):
        self.swarm.reset()
        self.physics.set_hover_throttle(self.swarm)
        self.trail_points = [[] for _ in range(self.n_drones)]
        print("Simulation RESET")

    def _adjust_throttle(self, delta: float):
        hover = self.physics.params.hover_throttle()
        new_throttle = np.clip(hover + delta, 0, 1)
        self.swarm.set_throttle(new_throttle)
        print(f"Throttle: {new_throttle*100:.1f}%")

    def _update(self):
        """Update drone positions."""
        if self.running:
            # Step physics
            for _ in range(10):  # 10 sub-steps
                self.physics.step(self.swarm, self.dt / 10)

        # Get positions
        positions = self.swarm.get_positions()

        # Update drone meshes
        for i in range(self.n_drones):
            pos = positions[i]
            # Convert NED to visualization (Z up)
            viz_pos = (pos[0], pos[1], -pos[2])

            # Create new sphere at position
            drone = pv.Sphere(radius=self.drone_size, center=viz_pos)
            color = self._get_drone_color(i)

            # Update mesh
            self.plotter.remove_actor(f'drone_{i}')
            self.plotter.add_mesh(drone, color=color, name=f'drone_{i}')

            # Add to trail
            if self.running:
                self.trail_points[i].append(viz_pos)
                if len(self.trail_points[i]) > 100:
                    self.trail_points[i].pop(0)

        # Update info text
        mean_alt = -positions[:, 2].mean()
        info = f"SwiftSim Visualizer\n"
        info += f"Drones: {self.n_drones}\n"
        info += f"Mean altitude: {mean_alt:.1f}m\n"
        info += f"Status: {'RUNNING' if self.running else 'PAUSED'}\n"
        info += f"\nControls:\n"
        info += f"SPACE - Start/Stop\n"
        info += f"R - Reset\n"
        info += f"UP/DOWN - Throttle"

        self.plotter.remove_actor('info_text')
        self.plotter.add_text(info, position='upper_left', font_size=9,
                             color='white', name='info_text')

    def run(self):
        """Start the visualization."""
        print("="*50)
        print("SwiftSim 3D Visualizer")
        print("="*50)
        print(f"Drones: {self.n_drones}")
        print(f"AVX2: {swiftsim.HAS_AVX2}")
        print(f"OpenMP: {swiftsim.HAS_OPENMP}")
        print("="*50)
        print("\nControls:")
        print("  SPACE - Start/Stop simulation")
        print("  R     - Reset simulation")
        print("  UP    - Increase throttle")
        print("  DOWN  - Decrease throttle")
        print("  Mouse - Rotate view")
        print("="*50)

        # Add timer callback for updates
        self.plotter.add_callback(self._update, interval=50)  # 20 FPS

        # Show
        self.plotter.show()


class SwarmAnimation:
    """Non-interactive animation for recording/demos."""

    def __init__(self, n_drones: int = 100, duration: float = 10.0):
        self.n_drones = n_drones
        self.duration = duration

        # Create physics
        self.swarm = swiftsim.DroneSwarm(n_drones)
        self.physics = swiftsim.SwarmPhysics()

    def create_takeoff_animation(self, filename: str = "swarm_takeoff.gif"):
        """Create a takeoff animation."""
        print(f"Creating animation with {self.n_drones} drones...")

        # Reset
        self.swarm.reset()

        # Randomize starting positions
        self.swarm.pos_x[:] = np.random.uniform(-10, 10, self.n_drones)
        self.swarm.pos_y[:] = np.random.uniform(-10, 10, self.n_drones)
        self.swarm.pos_z[:] = 0  # On ground

        # Set takeoff throttle
        self.swarm.set_throttle(0.7)

        # Setup plotter
        plotter = pv.Plotter(off_screen=True)
        plotter.set_background('black')

        # Ground
        ground = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1),
                          i_size=30, j_size=30)
        plotter.add_mesh(ground, color='darkgreen', opacity=0.5)

        # Open GIF
        plotter.open_gif(filename)

        frames = 100
        for frame in range(frames):
            # Physics step
            for _ in range(10):
                self.physics.step(self.swarm, 0.01)

            # Switch to hover after 3 seconds
            if frame == 30:
                self.physics.set_hover_throttle(self.swarm)

            # Get positions
            positions = self.swarm.get_positions()

            # Clear and redraw
            plotter.clear()
            plotter.add_mesh(ground, color='darkgreen', opacity=0.5)

            # Draw drones as points
            points = np.column_stack([
                positions[:, 0],
                positions[:, 1],
                -positions[:, 2]  # NED to Z-up
            ])
            cloud = pv.PolyData(points)
            plotter.add_mesh(cloud, color='cyan', point_size=10,
                           render_points_as_spheres=True)

            # Camera
            plotter.camera_position = [(40, 40, 30), (0, 0, 10), (0, 0, 1)]

            # Write frame
            plotter.write_frame()

            if frame % 10 == 0:
                print(f"  Frame {frame}/{frames}")

        plotter.close()
        print(f"Animation saved to {filename}")


def quick_demo():
    """Quick matplotlib demo (no PyVista window needed)."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    print("SwiftSim Quick Demo (Matplotlib)")
    print("="*40)

    # Create swarm
    n_drones = 20
    swarm = swiftsim.DroneSwarm(n_drones)
    physics = swiftsim.SwarmPhysics()

    # Random starting positions
    swarm.reset()
    swarm.pos_x[:] = np.random.uniform(-5, 5, n_drones)
    swarm.pos_y[:] = np.random.uniform(-5, 5, n_drones)
    swarm.pos_z[:] = np.random.uniform(-15, -5, n_drones)

    # Set hover
    physics.set_hover_throttle(swarm)

    # Simulate
    history = []
    for step in range(200):
        physics.step(swarm, 0.01)
        if step % 10 == 0:
            history.append(swarm.get_positions().copy())

    # Plot
    fig = plt.figure(figsize=(12, 5))

    # 3D view
    ax1 = fig.add_subplot(121, projection='3d')
    positions = swarm.get_positions()
    ax1.scatter(positions[:, 0], positions[:, 1], -positions[:, 2],
                c=range(n_drones), cmap='rainbow', s=50)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Altitude (m)')
    ax1.set_title(f'SwiftSim - {n_drones} Drones')

    # Altitude over time
    ax2 = fig.add_subplot(122)
    for i in range(min(5, n_drones)):
        altitudes = [-h[i, 2] for h in history]
        ax2.plot(altitudes, label=f'Drone {i}')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Altitude (m)')
    ax2.set_title('Altitude History')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('swiftsim_demo.png', dpi=150)
    plt.show()
    print("Saved to swiftsim_demo.png")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--matplotlib':
        quick_demo()
    elif len(sys.argv) > 1 and sys.argv[1] == '--animate':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        anim = SwarmAnimation(n_drones=n)
        anim.create_takeoff_animation()
    else:
        # Interactive 3D visualization
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
        viz = DroneVisualizer(n_drones=n)
        viz.run()
