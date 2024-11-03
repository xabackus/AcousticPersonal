import numpy as np
import pygame
import sys
import math
import matplotlib.pyplot as plt
import cv2  # For video writing
import json  # For saving/loading key press data
import os   # For file path operations

# Set the mode: 'write', 'read', or 'simulation'
# Uncomment the desired mode
# mode = 'write'       # Record key presses
# mode = 'read'        # Replay simulation based on recorded key presses
mode = 'simulation'    # ML-based control to optimize action timing

# Constants and Initialization
PI = 3.1415926
res = 512  # Simulation area size (512x512 pixels)
dashboard_width = 400  # Width for the dashboard area

# Define dashboard heights
text_dashboard_height = 300  # Height for displaying text information
average_graph_height = 200   # Height for the average neighbors graph
cumulative_graph_height = 200  # Height for the cumulative neighbor proportions graph

# Calculate window dimensions
window_width = res + dashboard_width
window_height = max(res, text_dashboard_height + average_graph_height + cumulative_graph_height)

# Pygame Initialization
pygame.init()
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Bulk Acoustic 2D Simulation with Dashboards")
clock = pygame.time.Clock()

# Video recording setup
record_video = True  # Set to True to record video
video_filename = 'simulation_video.mp4'
video_writer = None
video_fps = 30  # Frames per second

if record_video:
    frame_width = window_width
    frame_height = window_height
    # Use 'mp4v' codec for better compatibility
    video_writer = cv2.VideoWriter(
        video_filename,
        cv2.VideoWriter_fourcc(*'mp4v'),
        video_fps,
        (frame_width, frame_height)
    )

# Simulation Parameters
paused = False
running = True  # Flag to control the main loop
N = 12  # Number of particles (changed to 12)
particle_m_max = 5.0  # Maximum mass of particles
nest_size = 0.6  # Nest size
particle_radius_max = 10.0 / float(res)  # Maximum particle radius (~0.0195)
init_vel = 100  # Initial velocity

h = 1e-5  # Time step
substepping = 3  # Number of sub-iterations within a time step

# Set the distance threshold for clustering
d = 1.5 * particle_radius_max  # Global variable 'd'

# Fields for particle properties
pos = np.zeros((N, 2), dtype=np.float32)  # Positions
vel = np.zeros((N, 2), dtype=np.float32)  # Velocities
force = np.zeros((N, 2), dtype=np.float32)  # Forces

particle_radius = np.zeros(N, dtype=np.float32)  # Radii
particle_m = np.zeros(N, dtype=np.float32)  # Masses

energy = np.zeros(2, dtype=np.float32)  # [1] current energy, [0] initial energy
particle_color = np.zeros((N, 3), dtype=np.float32)  # Colors (not used in rendering)

# Acoustic properties
po = 10e7  # Acoustic pressure

ax = np.array([1.0])  # Amplitude in x-direction
ay = np.array([1.0])  # Amplitude in y-direction
kx = np.array([1], dtype=int)  # Wave number in x-direction (initially 1)
ky = np.array([1], dtype=int)  # Wave number in y-direction

ax_field = ax
ay_field = ay
kx_field = kx
ky_field = ky

num_waves_x = len(ax)
num_waves_y = len(ay)

# Drag properties
drag = 3e6

# Clustering variables
neighbors = np.zeros(N, dtype=np.int32)  # Number of neighbors for each particle
particle_node_assignments = np.full(N, -1, dtype=np.int32)  # Cluster assignments
cluster_id = 0  # Current cluster ID

# For plotting
average_neighbors_over_time = []  # Track average neighbors over time
neighbor_count_over_time = {i: [] for i in range(7)}  # Track counts of neighbor numbers over time
neighbor_counts_field = np.zeros(7, dtype=np.int32)  # Field to store neighbor counts
weighted_avg_density_over_time = []  # List to store weighted average cluster density over time
weighted_avg_aspect_ratio_over_time = []  # List to store weighted average aspect ratio over time

# Initialize key press tracking variables
key_press_data_indices = []  # Stores data indices when number keys are pressed
time_step = 0  # Tracks the current time step

key_press_data = []  # List of tuples: (time_step, key_name)

if mode == 'read':
    # Load key_press_data from file
    with open('key_press_data.txt', 'r') as f:
        key_press_data = json.load(f)
    key_press_index = 0
else:
    key_press_data = []

# Set random seed for reproducibility
np.random.seed(0)

# Initialize particles randomly
def initialize():
    global energy
    energy[0] = 0.0
    energy[1] = 0.0
    for i in range(N):
        pos[i] = np.random.rand(2)
        vel[i] = np.zeros(2)
        particle_radius[i] = 0.5 * particle_radius_max
        particle_m[i] = (particle_radius[i] / particle_radius_max) ** 2 * particle_m_max
        particle_color[i] = 0.3 + 0.7 * np.random.rand(3)

# Compute forces acting on particles
def compute_force():
    # Clear forces
    force[:, :] = 0.0

    # Compute acoustic force
    for i in range(N):
        f_x = 0.0
        f_y = 0.0

        for wave in range(num_waves_x):
            f_x += particle_radius[i] ** 3 * 1e6 * ax_field[wave] * np.sin(2 * PI * pos[i][0] * kx_field[wave])

        for wave in range(num_waves_y):
            f_y += particle_radius[i] ** 3 * 1e6 * ay_field[wave] * np.sin(2 * PI * pos[i][1] * ky_field[wave])

        f_vector = np.array([f_x, f_y]) * po
        force[i] += f_vector

    # Force due to drag
    for i in range(N):
        f = -drag * particle_radius[i] * vel[i]
        force[i] += f

    # Compute collision forces and add to force
    compute_collision_force()

# Compute collision forces between particles
def compute_collision_force():
    global force
    # Initialize collision force
    collision_force = np.zeros_like(force)
    EPSILON = 1e-4
    DAMPING = 0.1
    K = 1e11  # Stiffness constant

    # Calculate pairwise differences and distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # Shape (N, N, 2)
    r = np.linalg.norm(diff, axis=2) + EPSILON  # Shape (N, N)

    # Calculate overlaps
    overlap = particle_radius[:, np.newaxis] + particle_radius[np.newaxis, :] - r  # Shape (N, N)
    overlap = np.maximum(overlap, 0)  # Only positive overlaps (compressions)

    # Normal force magnitude (Hertzian contact model)
    normal_force_magnitude = K * overlap ** (3 / 2)  # Shape (N, N)

    # Compute unit vectors along diff
    normal_vector = diff / r[..., np.newaxis]  # Shape (N, N, 2)

    # Compute relative velocities
    relative_velocity = vel[:, np.newaxis, :] - vel[np.newaxis, :, :]  # Shape (N, N, 2)

    # Compute relative velocity along the normal direction
    vel_along_normal = np.sum(relative_velocity * normal_vector, axis=2)  # Shape (N, N)

    # Damping force magnitude
    damping_force_magnitude = -DAMPING * vel_along_normal  # Shape (N, N)

    # Total force magnitude
    total_force_magnitude = normal_force_magnitude + damping_force_magnitude  # Shape (N, N)

    # Total force vector
    total_force = total_force_magnitude[..., np.newaxis] * normal_vector  # Shape (N, N, 2)

    # Compute collision force on each particle
    collision_force = np.sum(total_force, axis=1) - np.sum(total_force, axis=0)  # Shape (N, 2)

    # Add collision force to total force
    force += collision_force

# Update positions and velocities
def update():
    dt = h / substepping
    for i in range(N):
        vel[i] += dt * force[i] / particle_m[i]
        pos[i] += dt * vel[i]
        # Collision detection at edges
        if pos[i][0] < 0.0 + particle_radius[i]:
            pos[i][0] = 0.0 + particle_radius[i]
            vel[i][0] *= -1
        if pos[i][0] > 1.0 - particle_radius[i]:
            pos[i][0] = 1.0 - particle_radius[i]
            vel[i][0] *= -1
        if pos[i][1] < 0.0 + particle_radius[i]:
            pos[i][1] = 0.0 + particle_radius[i]
            vel[i][1] *= -1
        if pos[i][1] > 1.0 - particle_radius[i]:
            pos[i][1] = 1.0 - particle_radius[i]
            vel[i][1] *= -1

# Compute total kinetic energy
def compute_energy():
    energy[1] = 0.0
    for i in range(N):
        energy[1] += 0.5 * particle_m[i] * np.sum(vel[i] ** 2)

# Clustering logic
def calculate_neighbors_and_init_clusters(d):
    global neighbors, particle_node_assignments
    neighbors[:] = 0
    particle_node_assignments[:] = -1
    for i in range(N):
        for j in range(i + 1, N):
            r = np.linalg.norm(pos[j] - pos[i])
            if r < particle_radius[i] + particle_radius[j] + 2e-3:
                neighbors[i] += 1
                neighbors[j] += 1

def expand_cluster(cluster_id_value, particle_id, d):
    stack = []
    stack.append(particle_id)
    particle_node_assignments[particle_id] = cluster_id_value

    while stack:
        current_particle = stack.pop()
        for j in range(N):
            if particle_node_assignments[j] == -1:
                r = np.linalg.norm(pos[j] - pos[current_particle])
                if r < particle_radius[current_particle] + particle_radius[j] + 2e-3:
                    particle_node_assignments[j] = cluster_id_value
                    stack.append(j)
                elif r < d:
                    particle_node_assignments[j] = cluster_id_value
                    stack.append(j)

def assign_clusters(d):
    global cluster_id
    cluster_id = 0
    for i in range(N):
        if neighbors[i] >= 2 and particle_node_assignments[i] == -1:
            particle_node_assignments[i] = cluster_id
            expand_cluster(cluster_id, i, d)
            cluster_id += 1

def run_clustering(d):
    calculate_neighbors_and_init_clusters(d)
    assign_clusters(d)

def calc_neighbors():
    global neighbors, neighbor_counts_field
    neighbors[:] = 0
    for i in range(N):
        for j in range(i + 1, N):
            r = np.linalg.norm(pos[j] - pos[i])
            if r < particle_radius[i] + particle_radius[j] + 2e-3:
                neighbors[i] += 1
                neighbors[j] += 1

    neighbor_counts_field[:] = 0
    for i in range(N):
        if neighbors[i] <= 6:
            neighbor_counts_field[neighbors[i]] += 1

def update_average_neighbors():
    avg_neighbors = np.mean(neighbors)
    average_neighbors_over_time.append(avg_neighbors)

def update_neighbor_count_over_time():
    counts = neighbor_counts_field.copy()
    for i in range(7):
        neighbor_count_over_time[i].append(counts[i])

def calculate_weighted_avg_density():
    global weighted_avg_density_over_time
    total_weighted_density = 0.0
    total_particles = 0
    for cluster_idx in range(cluster_id):
        indices = np.where(particle_node_assignments == cluster_idx)[0]
        num_particles_in_cluster = len(indices)
        if num_particles_in_cluster > 1:
            # For each particle, compute distance to its nearest neighbor in the cluster
            distances = []
            for i in indices:
                # Get positions of other particles in the same cluster
                other_indices = indices[indices != i]
                if len(other_indices) == 0:
                    continue  # Only one particle left, skip
                dists = np.linalg.norm(pos[other_indices] - pos[i], axis=1)
                min_dist = np.min(dists)
                distances.append(min_dist)
            if len(distances) > 0:
                avg_distance = np.mean(distances)
                # Weight by number of particles in the cluster
                total_weighted_density += avg_distance * num_particles_in_cluster
                total_particles += num_particles_in_cluster
    if total_particles > 0:
        weighted_avg_density = total_weighted_density / total_particles
    else:
        weighted_avg_density = 0.0  # No clusters, density is zero
    weighted_avg_density_over_time.append(weighted_avg_density)

def calculate_aspect_ratios():
    global weighted_avg_aspect_ratio_over_time
    total_weighted_aspect_ratio = 0.0
    total_particles_in_clusters = 0
    cluster_aspect_ratios = {}

    for cluster_idx in range(cluster_id):
        indices = np.where(particle_node_assignments == cluster_idx)[0]
        num_particles_in_cluster = len(indices)
        if num_particles_in_cluster > 1:
            # Get positions of particles in the cluster
            cluster_positions = pos[indices]
            # Compute covariance matrix
            covariance_matrix = np.cov(cluster_positions.T)
            # Compute eigenvalues
            eigenvalues, _ = np.linalg.eig(covariance_matrix)
            # Sort eigenvalues
            sorted_eigenvalues = np.sort(eigenvalues)
            # Avoid division by zero
            if sorted_eigenvalues[0] == 0:
                aspect_ratio = np.inf  # Set to infinity if division by zero
            else:
                aspect_ratio = np.sqrt(sorted_eigenvalues[-1] / sorted_eigenvalues[0])
            cluster_aspect_ratios[cluster_idx] = aspect_ratio
            # Weighted sum
            total_weighted_aspect_ratio += aspect_ratio * num_particles_in_cluster
            total_particles_in_clusters += num_particles_in_cluster
        else:
            # For clusters with only one particle, aspect ratio is 1
            cluster_aspect_ratios[cluster_idx] = 1.0
            total_weighted_aspect_ratio += 1.0 * num_particles_in_cluster
            total_particles_in_clusters += num_particles_in_cluster

    if total_particles_in_clusters > 0:
        weighted_avg_aspect_ratio = total_weighted_aspect_ratio / total_particles_in_clusters
    else:
        weighted_avg_aspect_ratio = 0.0

    weighted_avg_aspect_ratio_over_time.append(weighted_avg_aspect_ratio)
    return cluster_aspect_ratios

# Plotting functions using Matplotlib
def plot_average_neighbors_over_time():
    plt.figure(figsize=(10, 6))
    plt.plot(average_neighbors_over_time, label="Average Neighbors Over Time", color="blue", linewidth=2)

    # Add vertical lines at key_press_data_indices
    for idx, data_index in enumerate(key_press_data_indices):
        plt.axvline(x=data_index, color='red', linestyle='--', linewidth=1, label='Key Press' if idx == 0 else "")

    plt.xlabel("Data Index")
    plt.ylabel("Average Number of Neighbors")
    plt.title("Average Neighbors Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the figure
    filename = f"average_neighbors_over_time_{mode}.png"
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.close()

def plot_neighbor_distribution_over_time():
    plt.figure(figsize=(10, 6))

    # Number of neighbor categories
    max_neighbors = 6

    # Convert the neighbor_count_over_time to a numpy array for easier manipulation
    neighbor_counts_array = np.array([neighbor_count_over_time[i] for i in range(7)])

    # Compute cumulative counts along the time axis (axis=0)
    cumulative_counts = np.cumsum(neighbor_counts_array, axis=0)

    # Plot each cumulative count
    for i in range(max_neighbors + 1):
        plt.plot(cumulative_counts[i], label=f"Up to {i} neighbors", linewidth=2)

    # Add vertical lines at key_press_data_indices
    for idx, data_index in enumerate(key_press_data_indices):
        plt.axvline(x=data_index, color='red', linestyle='--', linewidth=1, label='Key Press' if idx == 0 else "")

    plt.xlabel("Data Index")
    plt.ylabel("Number of Particles")
    plt.title("Cumulative Neighbor Distribution Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the figure
    filename = f"neighbor_distribution_over_time_{mode}.png"
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.close()

def plot_weighted_avg_density_over_time():
    plt.figure(figsize=(10, 6))
    plt.plot(weighted_avg_density_over_time, label="Weighted Avg Cluster Density Over Time", color="purple", linewidth=2)

    # Add vertical lines at key_press_data_indices
    for idx, data_index in enumerate(key_press_data_indices):
        plt.axvline(x=data_index, color='red', linestyle='--', linewidth=1, label='Key Press' if idx == 0 else "")

    plt.xlabel("Data Index")
    plt.ylabel("Weighted Average Cluster Density")
    plt.title("Weighted Average Cluster Density Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the figure
    filename = f"weighted_avg_density_over_time_{mode}.png"
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.close()

def plot_weighted_avg_aspect_ratio_over_time():
    plt.figure(figsize=(10, 6))
    plt.plot(weighted_avg_aspect_ratio_over_time, label="Weighted Avg Aspect Ratio Over Time", color="orange", linewidth=2)

    # Add vertical lines at key_press_data_indices
    for idx, data_index in enumerate(key_press_data_indices):
        plt.axvline(x=data_index, color='red', linestyle='--', linewidth=1, label='Key Press' if idx == 0 else "")

    plt.xlabel("Data Index")
    plt.ylabel("Weighted Average Aspect Ratio")
    plt.title("Weighted Average Aspect Ratio Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the figure
    filename = f"weighted_avg_aspect_ratio_over_time_{mode}.png"
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.close()

# Initialize the simulation
initialize()
xchanging = True  # Flag to indicate whether changing kx or ky
message = "Changing number of nodes along X-axis"

# Colors for clusters
colors = [
    0xFF0000,  # Red
    0x00FF00,  # Green
    0x0000FF,  # Blue
    0xFFFF00,  # Yellow
    0xFF00FF,  # Magenta
    0x00FFFF,  # Cyan
    0xFFFFFF,  # White
    0xFFA500,  # Orange
    0x800080,  # Purple
    0x00FF7F,  # Spring Green
    0x808000,  # Olive
    0x008080,  # Teal
    0x000000,  # Black
]

def adjust_brightness(color, factor):
    """Adjust the brightness of a color."""
    r = (color >> 16) & 0xFF
    g = (color >> 8) & 0xFF
    b = color & 0xFF

    r = int(min(max(r * factor, 0), 255))
    g = int(min(max(g * factor, 0), 255))
    b = int(min(max(b * factor, 0), 255))

    return (r, g, b)

# Font for text
pygame.font.init()
font = pygame.font.SysFont('Arial', 16)

# Define dashboard areas
text_area_rect = pygame.Rect(res, 0, dashboard_width, text_dashboard_height)
average_graph_rect = pygame.Rect(res, text_dashboard_height, dashboard_width, average_graph_height)
cumulative_graph_rect = pygame.Rect(res, text_dashboard_height + average_graph_height, dashboard_width, cumulative_graph_height)

# Colors for neighbor counts in cumulative graph
neighbor_colors = [
    (255, 0, 0),     # Red for 0 neighbors
    (255, 165, 0),   # Orange for 1 neighbor
    (255, 255, 0),   # Yellow for 2 neighbors
    (0, 255, 0),     # Green for 3 neighbors
    (0, 127, 255),   # Light blue for 4 neighbors
    (0, 0, 255),     # Blue for 5 neighbors
    (139, 0, 255),   # Purple for 6 neighbors
]

neighbor_labels = ["0", "1", "2", "3", "4", "5", "6"]

def process_key_press(key):
    """Process key presses to control simulation parameters."""
    global xchanging, message, kx_field, ky_field, kx, ky, paused, running, key_press_data_indices, record_video, video_writer
    if key == pygame.K_ESCAPE:
        running = False
    elif key == pygame.K_r:
        initialize()
    elif key == pygame.K_SPACE:
        paused = not paused
    elif key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]:
        val = int(pygame.key.name(key))
        if xchanging:
            kx_field[0] = kx[0] = val
        else:
            ky_field[0] = ky[0] = val
        # Record the current data index when a number key is pressed
        key_press_data_indices.append(len(average_neighbors_over_time))
    elif key == pygame.K_x:
        xchanging = True
        message = "Changing number of nodes along X-axis"
    elif key == pygame.K_y:
        xchanging = False
        message = "Changing number of nodes along Y-axis"
    elif key == pygame.K_g:
        # Plot graphs using Matplotlib
        plot_average_neighbors_over_time()
        plot_neighbor_distribution_over_time()
        plot_weighted_avg_density_over_time()
        plot_weighted_avg_aspect_ratio_over_time()
        # Release the video writer
        if record_video and video_writer is not None:
            video_writer.release()
            video_writer = None
            print(f"Video saved as {video_filename}")
        if mode == 'write':
            # Save key_press_data to file
            with open('key_press_data.txt', 'w') as f:
                json.dump(key_press_data, f)
            print("Key press data saved to key_press_data.txt")
        elif mode == 'read':
            # In 'read' mode, when 'g' is pressed, we stop the simulation
            running = False
        elif mode == 'simulation':
            # In 'simulation' mode, when 'g' is pressed, we stop the simulation
            running = False

####################################################################################
# ML code for 'simulation' mode begins here

# Function to compute the reward based on the current particle positions
def compute_reward(total_steps):
    # Run clustering to identify clusters
    d_local = d  # Use same distance threshold
    run_clustering(d_local)
    # Define groups based on x-coordinate
    group1 = pos[:, 0] < 0.5
    group2 = pos[:, 0] >= 0.5
    fraction_group1 = np.sum(group1) / N
    fraction_group2 = np.sum(group2) / N
    reward = (-10000 * abs(fraction_group1 - fraction_group2)) - total_steps
    return reward, fraction_group1, fraction_group2

# Function to run the simulation with a given action time step and return the reward
def run_simulation_with_action(action_time_step, render=True):
    global pos, vel, kx_field, ky_field, time_step, paused, running, key_press_data, cluster_id, particle_node_assignments
    # Reset simulation state
    initialize()
    kx_field[0] = kx[0] = 1  # Start with kx = 1
    time_step = 0
    total_steps = 0
    action_taken = False
    cluster_id = 0
    particle_node_assignments[:] = -1  # Reset cluster assignments

    sim_running = True
    while sim_running:
        if not paused:
            for _ in range(substepping):
                compute_force()
                update()
                compute_energy()
                calc_neighbors()
                run_clustering(d)
                update_average_neighbors()
                update_neighbor_count_over_time()
                calculate_weighted_avg_density()
                cluster_aspect_ratios = calculate_aspect_ratios()

            time_step += 1
            total_steps += 1

            # Apply action if it's time and action hasn't been taken yet
            if not action_taken and time_step >= action_time_step:
                kx_field[0] = kx[0] = 2  # Change kx to 2
                action_taken = True
                action_taken_time = time_step

            # If action has been taken, check if 200 steps have passed since then
            if action_taken and time_step >= action_taken_time + 200:
                # Compute reward
                reward, fraction_group1, fraction_group2 = compute_reward(total_steps)
                print(f"Action taken at step {action_time_step}: Reward = {reward}")
                return reward, action_time_step, fraction_group1, fraction_group2

            # Render the simulation if required
            if render:
                render_simulation(cluster_aspect_ratios)
                # Process events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sim_running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            sim_running = False
                clock.tick(60)  # Limit to 60 FPS

# Gradient descent optimization function
def optimize_action():
    # Initial guesses for action time steps
    initial_guesses = [50, 100, 200, 300, 400, 500, 700, 1000]
    best_reward = None
    best_action_time_step = None

    print("Evaluating initial guesses...")
    for guess in initial_guesses:
        reward, action_time_step, fraction_group1, fraction_group2 = run_simulation_with_action(guess, render=False)
        print(f"Action at step {action_time_step}: Reward = {reward}, Group1 fraction = {fraction_group1:.2f}, Group2 fraction = {fraction_group2:.2f}")
        if best_reward is None or reward > best_reward:
            best_reward = reward
            best_action_time_step = action_time_step

    # Use the best initial guess for gradient descent
    action_time_step = best_action_time_step
    learning_rate = 10  # Adjust the time step by this amount
    previous_reward = best_reward

    print("\nStarting gradient descent optimization...")
    while True:
        # Try action earlier
        earlier_action_time_step = max(1, action_time_step - learning_rate)
        reward_earlier, _, _, _ = run_simulation_with_action(earlier_action_time_step, render=False)
        # Try action later
        later_action_time_step = action_time_step + learning_rate
        reward_later, _, _, _ = run_simulation_with_action(later_action_time_step, render=False)

        # Print the rewards
        print(f"Trying action at step {earlier_action_time_step}: Reward = {reward_earlier}")
        print(f"Trying action at step {later_action_time_step}: Reward = {reward_later}")

        # Check if moving earlier improves the reward
        if reward_earlier > previous_reward:
            action_time_step = earlier_action_time_step
            previous_reward = reward_earlier
            print(f"Action time updated to {action_time_step} (earlier)")
            continue
        # Check if moving later improves the reward
        elif reward_later > previous_reward:
            action_time_step = later_action_time_step
            previous_reward = reward_later
            print(f"Action time updated to {action_time_step} (later)")
            continue
        else:
            # Neither earlier nor later improves the reward; optimal time found
            print("\nOptimal action time found.")
            print(f"Final action at step {action_time_step}: Reward = {previous_reward}")
            # Run the simulation one more time with rendering
            reward, _, fraction_group1, fraction_group2 = run_simulation_with_action(action_time_step, render=True)
            print(f"Final fractions: Group1 = {fraction_group1:.2f}, Group2 = {fraction_group2:.2f}")
            break

    # Output the key presses similar to 'write' mode
    key_press_data.clear()
    # Record key presses for 'x' and number '2' at the action time step
    key_press_data.append((action_time_step, 'x'))
    key_press_data.append((action_time_step + 1, '2'))

    # Save key_press_data to file
    with open('key_press_data.txt', 'w') as f:
        json.dump(key_press_data, f)
    print("Optimized key press data saved to key_press_data.txt")
    print(f"Optimal action time step: {action_time_step}")

# Function to render the simulation
def render_simulation(cluster_aspect_ratios):
    global assignments, cluster_id, pos
    # Process events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

    # Clear the screen
    window.fill((17, 47, 65))  # Dark background

    # Draw pressure minima lines
    lines = []
    for x_line in np.linspace(.5 / kx[0], 1 - .5 / kx[0], kx[0]):
        start_pos = (int(x_line * res), 0)
        end_pos = (int(x_line * res), res)
        lines.append((start_pos, end_pos))

    for y_line in np.linspace(.5 / ky[0], 1 - .5 / ky[0], ky[0]):
        start_pos = (0, int(y_line * res))
        end_pos = (res, int(y_line * res))
        lines.append((start_pos, end_pos))

    for line in lines:
        pygame.draw.line(window, (128, 128, 128), line[0], line[1], 1)  # Gray lines

    # Draw particles
    num_colors = len(colors)
    assignments = particle_node_assignments.copy()

    unassigned_mask = assignments == -1

    assignments_unassigned = assignments.copy()
    assignments_unassigned[unassigned_mask] = cluster_id

    total_clusters_with_unassigned = cluster_id + 1

    counts = np.bincount(assignments_unassigned.astype(np.int32), minlength=total_clusters_with_unassigned)

    for i in range(N):
        cluster_id_i = assignments[i]
        num_neighbors = neighbors[i]
        brightness_factor = min(max(num_neighbors / 6, 0), 1)

        if cluster_id_i == -1:
            base_color = 0x808080  # Gray for unassigned particles
        else:
            base_color = colors[cluster_id_i % num_colors]

        color = adjust_brightness(base_color, brightness_factor)

        # Ensure positions are within [0,1]
        pos_clipped = np.clip(pos[i], 0.0, 1.0)
        screen_pos = (int(pos_clipped[0] * res), int(pos_clipped[1] * res))
        radius = int(max(particle_radius[i] * res, 2))  # Minimum radius of 2 pixels for visibility
        pygame.draw.circle(window, color, screen_pos, radius)

    # Draw Text Dashboard
    pygame.draw.rect(window, (30, 30, 30), text_area_rect)  # Background for text area

    # Display cluster information
    start_x = text_area_rect.left + 10
    start_y = text_area_rect.top + 10  # Starting y position for text
    spacing = 20  # Spacing between lines

    # Render cluster information
    y_pos = start_y
    for cluster_idx in range(cluster_id):
        aspect_ratio = cluster_aspect_ratios.get(cluster_idx, 1.0)
        cluster_text = f"Cluster {cluster_idx + 1}: {counts[cluster_idx]} particles; aspect ratio = {aspect_ratio:.2f}"
        if y_pos + spacing > text_area_rect.bottom:
            break
        text_surface = font.render(cluster_text, True, (255, 255, 255))
        window.blit(text_surface, (start_x, y_pos))
        y_pos += spacing

    # Display unassigned particles
    if y_pos + spacing <= text_area_rect.bottom:
        cluster_text = f"Unassigned: {counts[cluster_id]} particles"
        text_surface = font.render(cluster_text, True, (255, 255, 255))
        window.blit(text_surface, (start_x, y_pos))
        y_pos += spacing

    # Display messages
    if y_pos + spacing <= text_area_rect.bottom:
        message_surface = font.render(message, True, (255, 255, 255))
        window.blit(message_surface, (start_x, y_pos))
        y_pos += spacing

    # Display average neighbors
    if y_pos + spacing <= text_area_rect.bottom:
        avg_neighbors = average_neighbors_over_time[-1] if average_neighbors_over_time else 0
        avg_neighbors_surface = font.render(f"Average Neighbors: {avg_neighbors:.2f}", True, (255, 255, 255))
        window.blit(avg_neighbors_surface, (start_x, y_pos))
        y_pos += spacing

    # Display number of clusters
    if y_pos + spacing <= text_area_rect.bottom:
        num_clusters_surface = font.render(f"Number of Clusters: {cluster_id}", True, (255, 255, 255))
        window.blit(num_clusters_surface, (start_x, y_pos))
        y_pos += spacing

    # Display weighted average cluster density
    if y_pos + spacing <= text_area_rect.bottom:
        weighted_avg_density = weighted_avg_density_over_time[-1] if weighted_avg_density_over_time else 0
        density_surface = font.render(f"Weighted Avg Cluster Density: {weighted_avg_density:.4f}", True, (255, 255, 255))
        window.blit(density_surface, (start_x, y_pos))
        y_pos += spacing

    # Display weighted average aspect ratio
    if y_pos + spacing <= text_area_rect.bottom:
        weighted_avg_aspect_ratio = weighted_avg_aspect_ratio_over_time[-1] if weighted_avg_aspect_ratio_over_time else 0
        aspect_ratio_surface = font.render(f"Weighted Avg Aspect Ratio: {weighted_avg_aspect_ratio:.2f}", True, (255, 255, 255))
        window.blit(aspect_ratio_surface, (start_x, y_pos))
        y_pos += spacing

    # Draw Average Neighbors Over Time Graph
    pygame.draw.rect(window, (30, 30, 30), average_graph_rect)  # Background for graph area
    data = average_neighbors_over_time
    data_length = len(data)
    max_data_points = average_graph_rect.width  # Number of pixels in x-direction

    if data_length > max_data_points:
        data_to_plot = data[-max_data_points:]
    else:
        data_to_plot = data

    x_scale = average_graph_rect.width / max_data_points
    y_min = 0
    y_max = 6
    y_range = y_max - y_min
    y_scale = (average_graph_rect.height - 40) / y_range  # Leave space for labels

    points = []

    for i, value in enumerate(data_to_plot):
        x = average_graph_rect.left + i * x_scale
        y = average_graph_rect.bottom - 20 - (value - y_min) * y_scale
        points.append((x, y))

    if len(points) >= 2:
        pygame.draw.lines(window, (0, 255, 0), False, points, 2)  # Green line

    # Draw labels and axes for average neighbors graph
    text_surface = font.render("Average Neighbors Over Time", True, (255, 255, 255))
    window.blit(text_surface, (average_graph_rect.left + 10, average_graph_rect.top + 10))

    # Draw y-axis labels
    for i in range(int(y_min), int(y_max) + 1):
        y = average_graph_rect.bottom - 20 - (i - y_min) * y_scale
        label_surface = font.render(str(i), True, (255, 255, 255))
        window.blit(label_surface, (average_graph_rect.left + 5, y - 8))
        pygame.draw.line(window, (100, 100, 100), (average_graph_rect.left + 30, y), (average_graph_rect.right, y), 1)

    # Draw Cumulative Neighbor Proportions Graph
    pygame.draw.rect(window, (30, 30, 30), cumulative_graph_rect)  # Background for graph area
    neighbor_counts_array = np.array([neighbor_count_over_time[i] for i in range(7)])
    data_length = neighbor_counts_array.shape[1]

    max_data_points = cumulative_graph_rect.width  # Number of pixels in x-direction

    if data_length > max_data_points:
        neighbor_counts_array = neighbor_counts_array[:, -max_data_points:]
        data_length = max_data_points

    cumulative_counts_array = np.cumsum(neighbor_counts_array, axis=0)
    cumulative_proportions_array = cumulative_counts_array / N  # Convert to proportions

    x_scale = cumulative_graph_rect.width / max_data_points
    y_scale = (cumulative_graph_rect.height - 40)  # Leave space for labels

    # Draw cumulative proportions for each neighbor count
    for neighbor in range(6, -1, -1):  # Draw from 6 to 0 neighbors
        data = cumulative_proportions_array[neighbor]
        points = []
        for i, value in enumerate(data):
            x = cumulative_graph_rect.left + i * x_scale
            y = cumulative_graph_rect.bottom - 20 - value * y_scale
            points.append((x, y))
        if len(points) >= 2:
            pygame.draw.lines(window, neighbor_colors[neighbor], False, points, 2)

    # Draw labels and axes for cumulative graph
    text_surface = font.render("Cumulative Neighbor Proportions Over Time", True, (255, 255, 255))
    window.blit(text_surface, (cumulative_graph_rect.left + 10, cumulative_graph_rect.top + 10))

    # Draw legend for cumulative graph
    legend_y = cumulative_graph_rect.top + 30
    legend_x = cumulative_graph_rect.left + 10
    for neighbor in range(7):
        color = neighbor_colors[neighbor]
        label = neighbor_labels[neighbor]
        pygame.draw.line(window, color, (legend_x, legend_y), (legend_x + 20, legend_y), 2)
        label_surface = font.render(f"Neighbors = {label}", True, (255, 255, 255))
        window.blit(label_surface, (legend_x + 25, legend_y - 8))
        legend_y += 20

    # Draw numbering labels to clusters on the simulation area
    pos_np = pos.copy()
    for cluster_idx in range(cluster_id):
        indices = np.where(assignments == cluster_idx)[0]
        if len(indices) > 0:
            cluster_positions = pos_np[indices]
            centroid = np.mean(cluster_positions, axis=0)
            centroid = np.clip(centroid, 0.05, 0.95)
            text_surface = font.render(str(cluster_idx + 1), True, (255, 255, 255))
            screen_pos = (int(centroid[0] * res), int(centroid[1] * res))
            window.blit(text_surface, screen_pos)

    pygame.display.flip()

    # Capture the screen and write to video
    if record_video:
        # Capture the screen
        frame = pygame.surfarray.array3d(window)
        # Convert from (width, height, 3) to (height, width, 3)
        frame = np.transpose(frame, (1, 0, 2))
        # Convert RGB to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Write frame to video
        video_writer.write(frame)

# End of ML code
####################################################################################

# Main simulation loop
if mode == 'simulation':
    # In 'simulation' mode, run the optimization
    optimize_action()
    # After optimization, we can exit the program
    # Release the video writer
    if video_writer is not None:
        video_writer.release()
        video_writer = None
        print(f"Video saved as {video_filename}")
    pygame.quit()
    sys.exit()
else:
    running = True
    while running:
        if mode == 'write':
            # In 'write' mode, capture key presses
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    key_name = pygame.key.name(event.key)
                    key_press_data.append((time_step, key_name))
                    process_key_press(event.key)
        elif mode == 'read':
            # In 'read' mode, replay recorded key presses
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            # Process virtual key presses from key_press_data
            while key_press_index < len(key_press_data) and key_press_data[key_press_index][0] == time_step:
                key_name = key_press_data[key_press_index][1]
                key = pygame.key.key_code(key_name)
                process_key_press(key)
                if key_name == 'g':
                    running = False
                key_press_index += 1

        if not paused:
            for _ in range(substepping):
                compute_force()
                update()
                compute_energy()
                calc_neighbors()
                run_clustering(d)
                update_average_neighbors()
                update_neighbor_count_over_time()
                calculate_weighted_avg_density()
                cluster_aspect_ratios = calculate_aspect_ratios()

            time_step += 1  # Increment the time step after each simulation step

            total_clusters = cluster_id

            assignments = particle_node_assignments.copy()

            unassigned_mask = assignments == -1

            assignments_unassigned = assignments.copy()
            assignments_unassigned[unassigned_mask] = total_clusters

            total_clusters_with_unassigned = total_clusters + 1

            counts = np.bincount(assignments_unassigned.astype(np.int32), minlength=total_clusters_with_unassigned)

        # Clear the screen
        window.fill((17, 47, 65))  # Dark background

        # Draw pressure minima lines
        lines = []
        for x_line in np.linspace(.5 / kx[0], 1 - .5 / kx[0], kx[0]):
            start_pos = (int(x_line * res), 0)
            end_pos = (int(x_line * res), res)
            lines.append((start_pos, end_pos))

        for y_line in np.linspace(.5 / ky[0], 1 - .5 / ky[0], ky[0]):
            start_pos = (0, int(y_line * res))
            end_pos = (res, int(y_line * res))
            lines.append((start_pos, end_pos))

        for line in lines:
            pygame.draw.line(window, (128, 128, 128), line[0], line[1], 1)  # Gray lines

        # Draw particles
        num_colors = len(colors)
        for i in range(N):
            cluster_id_i = assignments[i]
            num_neighbors = neighbors[i]
            brightness_factor = min(max(num_neighbors / 6, 0), 1)

            if cluster_id_i == -1:
                base_color = 0x808080  # Gray for unassigned particles
            else:
                base_color = colors[cluster_id_i % num_colors]

            color = adjust_brightness(base_color, brightness_factor)

            # Ensure positions are within [0,1]
            pos_clipped = np.clip(pos[i], 0.0, 1.0)
            screen_pos = (int(pos_clipped[0] * res), int(pos_clipped[1] * res))
            radius = int(max(particle_radius[i] * res, 2))  # Minimum radius of 2 pixels for visibility
            pygame.draw.circle(window, color, screen_pos, radius)

        # Draw Text Dashboard
        pygame.draw.rect(window, (30, 30, 30), text_area_rect)  # Background for text area

        # Display cluster information
        start_x = text_area_rect.left + 10
        start_y = text_area_rect.top + 10  # Starting y position for text
        spacing = 20  # Spacing between lines

        # Render cluster information
        y_pos = start_y
        for cluster_idx in range(total_clusters):
            aspect_ratio = cluster_aspect_ratios.get(cluster_idx, 1.0)
            cluster_text = f"Cluster {cluster_idx + 1}: {counts[cluster_idx]} particles; aspect ratio = {aspect_ratio:.2f}"
            if y_pos + spacing > text_area_rect.bottom:
                break
            text_surface = font.render(cluster_text, True, (255, 255, 255))
            window.blit(text_surface, (start_x, y_pos))
            y_pos += spacing

        # Display unassigned particles
        if y_pos + spacing <= text_area_rect.bottom:
            cluster_text = f"Unassigned: {counts[total_clusters]} particles"
            text_surface = font.render(cluster_text, True, (255, 255, 255))
            window.blit(text_surface, (start_x, y_pos))
            y_pos += spacing

        # Display messages
        if y_pos + spacing <= text_area_rect.bottom:
            message_surface = font.render(message, True, (255, 255, 255))
            window.blit(message_surface, (start_x, y_pos))
            y_pos += spacing

        # Display average neighbors
        if y_pos + spacing <= text_area_rect.bottom:
            avg_neighbors = average_neighbors_over_time[-1] if average_neighbors_over_time else 0
            avg_neighbors_surface = font.render(f"Average Neighbors: {avg_neighbors:.2f}", True, (255, 255, 255))
            window.blit(avg_neighbors_surface, (start_x, y_pos))
            y_pos += spacing

        # Display number of clusters
        if y_pos + spacing <= text_area_rect.bottom:
            num_clusters_surface = font.render(f"Number of Clusters: {total_clusters}", True, (255, 255, 255))
            window.blit(num_clusters_surface, (start_x, y_pos))
            y_pos += spacing

        # Display weighted average cluster density
        if y_pos + spacing <= text_area_rect.bottom:
            weighted_avg_density = weighted_avg_density_over_time[-1] if weighted_avg_density_over_time else 0
            density_surface = font.render(f"Weighted Avg Cluster Density: {weighted_avg_density:.4f}", True, (255, 255, 255))
            window.blit(density_surface, (start_x, y_pos))
            y_pos += spacing

        # Display weighted average aspect ratio
        if y_pos + spacing <= text_area_rect.bottom:
            weighted_avg_aspect_ratio = weighted_avg_aspect_ratio_over_time[-1] if weighted_avg_aspect_ratio_over_time else 0
            aspect_ratio_surface = font.render(f"Weighted Avg Aspect Ratio: {weighted_avg_aspect_ratio:.2f}", True, (255, 255, 255))
            window.blit(aspect_ratio_surface, (start_x, y_pos))
            y_pos += spacing

        # Draw Average Neighbors Over Time Graph
        pygame.draw.rect(window, (30, 30, 30), average_graph_rect)  # Background for graph area
        data = average_neighbors_over_time
        data_length = len(data)
        max_data_points = average_graph_rect.width  # Number of pixels in x-direction

        if data_length > max_data_points:
            data_to_plot = data[-max_data_points:]
        else:
            data_to_plot = data

        x_scale = average_graph_rect.width / max_data_points
        y_min = 0
        y_max = 6
        y_range = y_max - y_min
        y_scale = (average_graph_rect.height - 40) / y_range  # Leave space for labels

        points = []

        for i, value in enumerate(data_to_plot):
            x = average_graph_rect.left + i * x_scale
            y = average_graph_rect.bottom - 20 - (value - y_min) * y_scale
            points.append((x, y))

        if len(points) >= 2:
            pygame.draw.lines(window, (0, 255, 0), False, points, 2)  # Green line

        # Draw labels and axes for average neighbors graph
        text_surface = font.render("Average Neighbors Over Time", True, (255, 255, 255))
        window.blit(text_surface, (average_graph_rect.left + 10, average_graph_rect.top + 10))

        # Draw y-axis labels
        for i in range(int(y_min), int(y_max) + 1):
            y = average_graph_rect.bottom - 20 - (i - y_min) * y_scale
            label_surface = font.render(str(i), True, (255, 255, 255))
            window.blit(label_surface, (average_graph_rect.left + 5, y - 8))
            pygame.draw.line(window, (100, 100, 100), (average_graph_rect.left + 30, y), (average_graph_rect.right, y), 1)

        # Draw Cumulative Neighbor Proportions Graph
        pygame.draw.rect(window, (30, 30, 30), cumulative_graph_rect)  # Background for graph area
        neighbor_counts_array = np.array([neighbor_count_over_time[i] for i in range(7)])
        data_length = neighbor_counts_array.shape[1]

        max_data_points = cumulative_graph_rect.width  # Number of pixels in x-direction

        if data_length > max_data_points:
            neighbor_counts_array = neighbor_counts_array[:, -max_data_points:]
            data_length = max_data_points

        cumulative_counts_array = np.cumsum(neighbor_counts_array, axis=0)
        cumulative_proportions_array = cumulative_counts_array / N  # Convert to proportions

        x_scale = cumulative_graph_rect.width / max_data_points
        y_scale = (cumulative_graph_rect.height - 40)  # Leave space for labels

        # Draw cumulative proportions for each neighbor count
        for neighbor in range(6, -1, -1):  # Draw from 6 to 0 neighbors
            data = cumulative_proportions_array[neighbor]
            points = []
            for i, value in enumerate(data):
                x = cumulative_graph_rect.left + i * x_scale
                y = cumulative_graph_rect.bottom - 20 - value * y_scale
                points.append((x, y))
            if len(points) >= 2:
                pygame.draw.lines(window, neighbor_colors[neighbor], False, points, 2)

        # Draw labels and axes for cumulative graph
        text_surface = font.render("Cumulative Neighbor Proportions Over Time", True, (255, 255, 255))
        window.blit(text_surface, (cumulative_graph_rect.left + 10, cumulative_graph_rect.top + 10))

        # Draw legend for cumulative graph
        legend_y = cumulative_graph_rect.top + 30
        legend_x = cumulative_graph_rect.left + 10
        for neighbor in range(7):
            color = neighbor_colors[neighbor]
            label = neighbor_labels[neighbor]
            pygame.draw.line(window, color, (legend_x, legend_y), (legend_x + 20, legend_y), 2)
            label_surface = font.render(f"Neighbors = {label}", True, (255, 255, 255))
            window.blit(label_surface, (legend_x + 25, legend_y - 8))
            legend_y += 20

        # Draw numbering labels to clusters on the simulation area
        pos_np = pos.copy()
        for cluster_idx in range(total_clusters):
            indices = np.where(assignments == cluster_idx)[0]
            if len(indices) > 0:
                cluster_positions = pos_np[indices]
                centroid = np.mean(cluster_positions, axis=0)
                centroid = np.clip(centroid, 0.05, 0.95)
                text_surface = font.render(str(cluster_idx + 1), True, (255, 255, 255))
                screen_pos = (int(centroid[0] * res), int(centroid[1] * res))
                window.blit(text_surface, screen_pos)

        pygame.display.flip()

        # Capture the screen and write to video
        if record_video:
            # Capture the screen
            frame = pygame.surfarray.array3d(window)
            # Convert from (width, height, 3) to (height, width, 3)
            frame = np.transpose(frame, (1, 0, 2))
            # Convert RGB to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Write frame to video
            video_writer.write(frame)

        clock.tick(30)  # Limit to 30 FPS

    # When the simulation ends, ensure the video writer is released
    if video_writer is not None:
        video_writer.release()
        video_writer = None
        print(f"Video saved as {video_filename}")

    pygame.quit()
    sys.exit()
