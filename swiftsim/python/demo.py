"""
SwiftSim Quick Demo - Simple 3D visualization
"""
import numpy as np
import pyvista as pv
import swiftsim
import time

def run_demo():
    print("=" * 50)
    print("SwiftSim Quick Demo")
    print("=" * 50)

    # Create swarm
    n_drones = 20
    swarm = swiftsim.DroneSwarm(n_drones)
    physics = swiftsim.SwarmPhysics()

    print(f"Drones: {n_drones}")
    print(f"AVX2: {swiftsim.HAS_AVX2}")
    print(f"OpenMP: {swiftsim.HAS_OPENMP}")
    print("=" * 50)

    # Reset to hover at 10m altitude
    swarm.reset()

    # Spread drones in a grid
    grid_size = int(np.ceil(np.sqrt(n_drones)))
    pos_x = swarm.pos_x
    pos_y = swarm.pos_y
    pos_z = swarm.pos_z
    for i in range(n_drones):
        row = i // grid_size
        col = i % grid_size
        pos_x[i] = (col - grid_size/2) * 2  # X
        pos_y[i] = (row - grid_size/2) * 2  # Y
        pos_z[i] = -10.0  # Z (NED, negative is up)

    # Set hover throttle (applies to all drones)
    hover = 0.58  # Approximate hover throttle
    swarm.set_throttle(hover)

    # Create plotter
    plotter = pv.Plotter()
    plotter.set_background('lightgray')

    # Ground plane
    ground = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1),
                      i_size=30, j_size=30)
    plotter.add_mesh(ground, color='darkgreen', opacity=0.7)

    # Target altitude plane
    target_plane = pv.Plane(center=(0, 0, 10), direction=(0, 0, 1),
                            i_size=30, j_size=30)
    plotter.add_mesh(target_plane, color='yellow', opacity=0.2)

    # Simulate for a few seconds
    print("Simulating drone swarm physics...")
    dt = 0.001
    steps = 1000

    for step in range(steps):
        # Slightly perturb throttle for some movement
        noise = 0.02 * np.sin(step * 0.01)
        swarm.set_throttle(hover + noise)
        physics.step(swarm, dt)

    # Get final positions
    pos_x = np.array(swarm.pos_x)
    pos_y = np.array(swarm.pos_y)
    pos_z = np.array(swarm.pos_z)

    # Convert NED to visualization (Z up)
    vis_positions = np.column_stack([pos_x, pos_y, -pos_z])

    print(f"Final drone altitudes: min={-pos_z.min():.2f}m, max={-pos_z.max():.2f}m")

    # Add drones as spheres
    for i, pos in enumerate(vis_positions):
        drone = pv.Sphere(radius=0.3, center=pos)
        color = 'blue' if pos[2] > 9.5 else 'red'  # Blue if near target, red otherwise
        plotter.add_mesh(drone, color=color)

        # Add rotor indicators
        for dx, dy in [(0.2, 0.2), (0.2, -0.2), (-0.2, -0.2), (-0.2, 0.2)]:
            rotor = pv.Sphere(radius=0.1, center=(pos[0]+dx, pos[1]+dy, pos[2]))
            plotter.add_mesh(rotor, color='red' if dx > 0 else 'gray')

    # Camera
    plotter.camera_position = [(40, 40, 30), (0, 0, 10), (0, 0, 1)]

    # Add text
    plotter.add_text(f"SwiftSim: {n_drones} Drones @ 10m altitude", font_size=12)

    print("\nVisualization window opened!")
    print("Close the window to exit.")

    plotter.show()

if __name__ == "__main__":
    run_demo()
