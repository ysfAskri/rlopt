"""
SwiftSim Real-Time Animated Demo
Continuous physics simulation with live 3D visualization
"""
import numpy as np
import pyvista as pv
from pyvista import themes
import swiftsim
import time

class RealtimeSwarmVisualizer:
    def __init__(self, n_drones=50):
        self.n_drones = n_drones

        # Physics
        self.swarm = swiftsim.DroneSwarm(n_drones)
        self.physics = swiftsim.SwarmPhysics()
        self.dt = 0.002
        self.hover = 0.58
        self.time = 0.0

        # Initialize positions in a grid
        self.reset()

        # PyVista setup
        self.plotter = pv.Plotter(off_screen=False)
        self.plotter.set_background('black')

        # Ground
        ground = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1),
                          i_size=40, j_size=40)
        self.plotter.add_mesh(ground, color='darkgreen', opacity=0.5)

        # Grid lines
        for i in range(-20, 21, 5):
            self.plotter.add_mesh(pv.Line((i, -20, 0.1), (i, 20, 0.1)),
                                  color='gray', line_width=1)
            self.plotter.add_mesh(pv.Line((-20, i, 0.1), (20, i, 0.1)),
                                  color='gray', line_width=1)

        # Target altitude indicator
        target = pv.Plane(center=(0, 0, 10), direction=(0, 0, 1),
                          i_size=40, j_size=40)
        self.plotter.add_mesh(target, color='cyan', opacity=0.1)

        # Create drone actors (will update positions)
        self.drone_actors = []
        self.rotor_actors = []

        positions = self._get_positions()
        for i in range(n_drones):
            pos = positions[i]
            # Drone body
            drone = pv.Sphere(radius=0.25, center=pos)
            actor = self.plotter.add_mesh(drone, color='blue', name=f'drone_{i}')
            self.drone_actors.append(actor)

            # Rotors (4 per drone)
            rotor_offsets = [(0.2, 0.2, 0), (0.2, -0.2, 0),
                            (-0.2, -0.2, 0), (-0.2, 0.2, 0)]
            drone_rotors = []
            for j, (dx, dy, dz) in enumerate(rotor_offsets):
                rotor_pos = (pos[0]+dx, pos[1]+dy, pos[2]+dz)
                rotor = pv.Sphere(radius=0.1, center=rotor_pos)
                color = 'red' if j < 2 else 'orange'
                actor = self.plotter.add_mesh(rotor, color=color,
                                              name=f'rotor_{i}_{j}')
                drone_rotors.append(actor)
            self.rotor_actors.append(drone_rotors)

        # Camera
        self.plotter.camera_position = [(50, 50, 40), (0, 0, 10), (0, 0, 1)]

        # Stats text
        self.plotter.add_text("SwiftSim Real-Time Demo", position='upper_left',
                              font_size=14, color='white')

    def reset(self):
        """Reset drones to grid formation at 10m"""
        self.swarm.reset()

        grid_size = int(np.ceil(np.sqrt(self.n_drones)))
        spacing = 2.0

        pos_x = self.swarm.pos_x
        pos_y = self.swarm.pos_y
        pos_z = self.swarm.pos_z

        for i in range(self.n_drones):
            row = i // grid_size
            col = i % grid_size
            pos_x[i] = (col - grid_size/2) * spacing
            pos_y[i] = (row - grid_size/2) * spacing
            pos_z[i] = -10.0  # NED: negative is up

        self.swarm.set_throttle(self.hover)
        self.time = 0.0

    def _get_positions(self):
        """Get drone positions in visualization coords (Z up)"""
        pos_x = np.array(self.swarm.pos_x)
        pos_y = np.array(self.swarm.pos_y)
        pos_z = np.array(self.swarm.pos_z)
        return np.column_stack([pos_x, pos_y, -pos_z])  # Flip Z for viz

    def step(self):
        """Advance physics simulation"""
        # Vary throttle for interesting motion
        base = self.hover

        # Create some oscillating patterns
        t = self.time
        throttle = base + 0.03 * np.sin(t * 2.0) + 0.02 * np.sin(t * 5.0)
        self.swarm.set_throttle(throttle)

        # Step physics multiple times for stability
        for _ in range(5):
            self.physics.step(self.swarm, self.dt)

        self.time += self.dt * 5

    def update_visuals(self):
        """Update drone positions in the plotter"""
        positions = self._get_positions()

        for i in range(self.n_drones):
            pos = positions[i]

            # Update drone body
            self.plotter.remove_actor(f'drone_{i}')

            # Color based on altitude (10m target)
            alt = pos[2]
            if alt > 9.5 and alt < 10.5:
                color = 'lime'  # Good altitude
            elif alt > 8 and alt < 12:
                color = 'yellow'  # Close
            else:
                color = 'red'  # Too far

            drone = pv.Sphere(radius=0.25, center=pos)
            self.plotter.add_mesh(drone, color=color, name=f'drone_{i}')

            # Update rotors
            rotor_offsets = [(0.2, 0.2, 0), (0.2, -0.2, 0),
                            (-0.2, -0.2, 0), (-0.2, 0.2, 0)]
            for j, (dx, dy, dz) in enumerate(rotor_offsets):
                self.plotter.remove_actor(f'rotor_{i}_{j}')
                rotor_pos = (pos[0]+dx, pos[1]+dy, pos[2]+dz)
                rotor = pv.Sphere(radius=0.1, center=rotor_pos)
                # Spinning effect - alternate colors
                spin = int(self.time * 50) % 2
                color = 'red' if (j + spin) % 2 == 0 else 'orange'
                self.plotter.add_mesh(rotor, color=color, name=f'rotor_{i}_{j}')

    def run(self, duration=30.0):
        """Run the real-time simulation"""
        print("=" * 50)
        print("SwiftSim Real-Time Demo")
        print("=" * 50)
        print(f"Drones: {self.n_drones}")
        print(f"Duration: {duration}s")
        print("=" * 50)
        print("\nControls:")
        print("  Mouse drag - Rotate view")
        print("  Scroll     - Zoom")
        print("  Close window to exit")
        print("=" * 50)

        self.plotter.show(interactive_update=True, auto_close=False)

        start = time.time()
        frame = 0

        while time.time() - start < duration:
            # Physics step
            self.step()

            # Update visuals every few frames
            if frame % 2 == 0:
                self.update_visuals()
                self.plotter.update()

            frame += 1

            # Small delay to control frame rate
            time.sleep(0.016)  # ~60 FPS target

        print(f"\nSimulation complete! {frame} frames rendered.")
        self.plotter.close()


def main():
    viz = RealtimeSwarmVisualizer(n_drones=30)
    viz.run(duration=60.0)  # Run for 60 seconds


if __name__ == "__main__":
    main()
