import numpy as np
import pygame
import sys
import math
import matplotlib.pyplot as plt

# Constants and Initialization
PI = 3.1415926
res = 512
dashboard_height = 200  # Height for the dashboard area

# Pygame Initialization
pygame.init()
window = pygame.display.set_mode((res, res + dashboard_height))
pygame.display.set_caption("Bulk Acoustic 2D Simulation with Dashboard")
clock = pygame.time.Clock()

# Simulation Parameters
paused = False
N = 61  # number of particles
particle_m_max = 5.0  # mass
nest_size = 0.6  # nest size
particle_radius_max = 10.0 / float(res)  # particle radius for rendering (~0.0195)
init_vel = 100  # initial velocity

h = 1e-5  # time step
substepping = 3  # the number of sub-iterations within a time step

# Fields
pos = np.zeros((N, 2), dtype=np.float32)
vel = np.zeros((N, 2), dtype=np.float32)
force = np.zeros((N, 2), dtype=np.float32)

particle_radius = np.zeros(N, dtype=np.float32)
particle_m = np.zeros(N, dtype=np.float32)

energy = np.zeros(2, dtype=np.float32)  # [1] current energy [0] initial energy
particle_color = np.zeros((N, 3), dtype=np.float32)

# Acoustic properties
po = 10e6  # acoustic pressure

ax = np.array([1.0])
ay = np.array([1.0])
kx = np.array([3], dtype=int)
ky = np.array([1], dtype=int)

ax_field = ax
ay_field = ay
kx_field = kx
ky_field = ky

num_waves_x = len(ax)
num_waves_y = len(ay)

# Drag properties
drag = 3e6

# Clustering variables
neighbors = np.zeros(N, dtype=np.int32)
particle_node_assignments = np.full(N, -1, dtype=np.int32)
cluster_id = 0

# For plotting
average_neighbors_over_time = []
neighbor_count_over_time = {i: [] for i in range(7)}
neighbor_counts_field = np.zeros(7, dtype=np.int32)

# Initialize key press tracking variables
key_press_time_steps = []  # Stores time steps when number keys are pressed
time_step = 0  # Tracks the current time step

# Initialize particles randomly
def initialize():
    global energy
    energy[0] = 0.0
    energy[1] = 0.0
    for i in range(N):
        theta = np.random.rand() * 2 * PI
        r = (np.sqrt(np.random.rand()) * 0.7 + 0.3) * nest_size
        pos[i] = np.array([np.random.rand(), np.random.rand()])
        offset = init_vel * r * np.array([np.cos(theta), np.sin(theta)])
        vel[i] = np.array([-offset[1], offset[0]])

        particle_radius[i] = 0.5 * particle_radius_max
        particle_m[i] = (particle_radius[i] / particle_radius_max)**2 * particle_m_max

        energy[0] += 0.5 * particle_m[i] * np.sum(vel[i]**2)
        energy[1] += 0.5 * particle_m[i] * np.sum(vel[i]**2)

        # particle_color[i] = 1 - particle_m[i] / particle_m_max  # Not used in rendering

# Initialize particles in a hexagonal cluster
cluster_x = 1/6
cluster_y = 1/2
cluster_radius = particle_radius_max

def initialize_cluster():
    global energy
    energy[0] = 0.0
    energy[1] = 0.0
    theta = 0.0
    curr = np.array([cluster_x, cluster_y])
    pos[0] = curr.copy()
    index = 1
    layer = 1

    while index < N:
        curr += cluster_radius * np.array([np.cos(theta), np.sin(theta)])
        for j in range(2, 8):
            angle = j * PI / 3 + theta
            for _ in range(layer):
                curr += cluster_radius * np.array([np.cos(angle), np.sin(angle)])
                if index < N:
                    pos[index] = curr.copy()
                    index += 1
                else:
                    break
            if index >= N:
                break
        layer += 1

    for i in range(N):
        theta = np.random.rand() * 2 * PI
        r = (np.sqrt(np.random.rand()) * 0.7 + 0.3) * nest_size
        offset = init_vel * r * np.array([np.cos(theta), np.sin(theta)])
        vel[i] = np.array([-offset[1], offset[0]])

        particle_radius[i] = 0.5 * particle_radius_max
        particle_m[i] = (particle_radius[i] / particle_radius_max)**2 * particle_m_max

        energy[0] += 0.5 * particle_m[i] * np.sum(vel[i]**2)
        energy[1] += 0.5 * particle_m[i] * np.sum(vel[i]**2)

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
            f_x += particle_radius[i]**3 * 1e6 * ax_field[wave] * np.sin(2 * PI * pos[i][0] * kx_field[wave])

        for wave in range(num_waves_y):
            f_y += particle_radius[i]**3 * 1e6 * ay_field[wave] * np.sin(2 * PI * pos[i][1] * ky_field[wave])

        f_vector = np.array([f_x, f_y]) * po
        force[i] += f_vector

    # Force due to drag
    for i in range(N):
        f = -drag * particle_radius[i] * vel[i]
        force[i] += f

# Update particle collisions
def collision_update():
    for i in range(N - 1):
        for j in range(i + 1, N):
            diff = pos[j] - pos[i]
            r = np.linalg.norm(diff)
            overlap = particle_radius[i] + particle_radius[j] - r
            if overlap > 0:
                # Particles are overlapping
                # Compute the normal vector
                if r > 0:
                    normal = diff / r
                else:
                    # If r is zero (particles are exactly on top of each other), choose a default normal
                    normal = np.array([1.0, 0.0])

                # Compute relative velocity in normal direction
                rel_vel = vel[j] - vel[i]
                vel_along_normal = np.dot(rel_vel, normal)

                # Only resolve if particles are moving towards each other
                if vel_along_normal < 0:
                    # Compute impulse scalar
                    restitution = 1.0  # Perfectly elastic collision
                    impulse_scalar = -(1 + restitution) * vel_along_normal
                    impulse_scalar /= (1 / particle_m[i] + 1 / particle_m[j])

                    # Apply impulse to velocities
                    impulse = impulse_scalar * normal
                    vel[i] -= impulse / particle_m[i]
                    vel[j] += impulse / particle_m[j]

                # Adjust positions to resolve overlap
                correction = normal * (overlap / (particle_m[i] + particle_m[j]))
                pos[i] -= correction * particle_m[j]
                pos[j] += correction * particle_m[i]

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
        energy[1] += 0.5 * particle_m[i] * np.sum(vel[i]**2)

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
        for j in range(N):
            if particle_node_assignments[j] == -1:
                r = np.linalg.norm(pos[j] - pos[current_particle])
                if r < d:
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

# Plotting functions
c = 0  # Initialize counter for plot filenames

def plot_average_neighbors_over_time():
    global c
    plt.figure(figsize=(10, 6))
    plt.plot(average_neighbors_over_time, label="Average Neighbors Over Time", color="blue", linewidth=2)
    
    # Add vertical lines at key_press_time_steps
    for idx, step in enumerate(key_press_time_steps):
        plt.axvline(x=step, color='red', linestyle='--', linewidth=1, label='Key Press' if idx == 0 else "")
    
    plt.xlabel("Time Step")
    plt.ylabel("Average Number of Neighbors")
    plt.title("Average Neighbors Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"average_neighbors_{c}.png")
    plt.show()
    c += 1

def plot_neighbor_distribution_over_time():
    global c
    plt.figure(figsize=(10, 6))
    
    # Number of neighbor categories
    max_neighbors = 6
    
    # Convert the neighbor_count_over_time to a numpy array for easier manipulation
    neighbor_counts_array = np.array([neighbor_count_over_time[i] for i in range(max_neighbors + 1)])
    
    # Compute cumulative counts along the time axis (axis=0)
    cumulative_counts = np.cumsum(neighbor_counts_array, axis=0)
    
    # Plot each cumulative count
    for i in range(max_neighbors + 1):
        plt.plot(cumulative_counts[i], label=f"Up to {i} neighbors", linewidth=2)
    
    # Add vertical lines at key_press_time_steps
    for idx, step in enumerate(key_press_time_steps):
        plt.axvline(x=step, color='red', linestyle='--', linewidth=1, label='Key Press' if idx == 0 else "")
    
    plt.xlabel("Time Step")
    plt.ylabel("Number of Particles")
    plt.title("Cumulative Neighbor Distribution Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"cumulative_neighbor_distribution_{c}.png")
    plt.show()
    c += 1

def plot_radial_distribution():
    global c
    positions = pos.copy()
    rs = np.linspace(0, np.sqrt(2), 500)
    gs = np.zeros_like(rs)
    mid = np.mean(positions, axis=0)

    plt.figure(figsize=(10, 6))
    for j in range(N):
        distance = np.linalg.norm(positions[j] - mid)
        for i, r in enumerate(rs):
            if particle_radius[j] == 0:
                continue  # Avoid division by zero
            contribution = np.sqrt(max(0., 1. - ((distance - r) ** 2) / (particle_radius[j] ** 2)))
            gs[i] += contribution

    plt.plot(rs, gs, label="Radial Distribution Function", color="black", linewidth=2, linestyle="--")
    plt.xlabel("Distance from Center")
    plt.ylabel("Radial Distribution")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"radial_distribution_{c}.png")
    plt.show()
    c += 1

# Initialize the simulation
initialize_cluster()
xchanging = True
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

# Main simulation loop
d = 1.5 * particle_radius_max  # Set the distance threshold for clustering
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_r:
                initialize_cluster()
            elif event.key == pygame.K_SPACE:
                paused = not paused
            elif event.unicode in "123456789":
                val = int(event.unicode)
                if xchanging:
                    kx_field[0] = kx[0] = val
                else:
                    ky_field[0] = ky[0] = val
                # Record the current time step when a number key is pressed
                key_press_time_steps.append(time_step)
            elif event.key == pygame.K_x:
                xchanging = True
                message = "Changing number of nodes along X-axis"
            elif event.key == pygame.K_y:
                xchanging = False
                message = "Changing number of nodes along Y-axis"
            elif event.key == pygame.K_g:
                plot_radial_distribution()
                plot_average_neighbors_over_time()
                plot_neighbor_distribution_over_time()

    if not paused:
        for _ in range(substepping):
            compute_force()
            collision_update()
            update()
            compute_energy()
            calc_neighbors()
            run_clustering(d)
            update_average_neighbors()
            update_neighbor_count_over_time()

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
    for x_line in np.linspace(.5/kx[0], 1-.5/kx[0], kx[0]):
        start_pos = (int(x_line * res), 0)
        end_pos = (int(x_line * res), res)
        lines.append((start_pos, end_pos))

    for y_line in np.linspace(.5/ky[0], 1-.5/ky[0], ky[0]):
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

    # Draw dashboard background
    dashboard_rect = pygame.Rect(0, res, res, dashboard_height)
    pygame.draw.rect(window, (30, 30, 30), dashboard_rect)  # Dark gray background for dashboard

    # Display cluster information on dashboard
    start_x = 10
    start_y = res + 10  # Starting y position for dashboard text
    spacing = 20  # Spacing between lines

    # Render cluster information
    for cluster_idx in range(total_clusters):
        cluster_text = f"Cluster {cluster_idx + 1}: {counts[cluster_idx]} particles"
        y_pos = start_y + cluster_idx * spacing
        if y_pos + spacing > res + dashboard_height:
            break
        text_surface = font.render(cluster_text, True, (255, 255, 255))
        window.blit(text_surface, (start_x, y_pos))

    # Display unassigned particles
    cluster_text = f"Unassigned: {counts[total_clusters]} particles"
    y_pos = start_y + total_clusters * spacing
    if y_pos + spacing <= res + dashboard_height:
        text_surface = font.render(cluster_text, True, (255, 255, 255))
        window.blit(text_surface, (start_x, y_pos))

    # Add numbering labels to clusters
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

    # Display messages on dashboard
    message_surface = font.render(message, True, (255, 255, 255))
    window.blit(message_surface, (start_x, res + dashboard_height - 30))  # Positioned at the bottom of dashboard

    number_of_clusters_surface = font.render(f"Number of clusters: {total_clusters}", True, (255, 255, 255))
    window.blit(number_of_clusters_surface, (start_x, res + dashboard_height - 60))  # Positioned above the message

    pygame.display.flip()
    clock.tick(30)  # Limit to 30 FPS

pygame.quit()
sys.exit()
