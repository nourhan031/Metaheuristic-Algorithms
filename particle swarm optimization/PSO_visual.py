import random
import numpy as np
import math
import matplotlib.pyplot as plt


def f(x):
    x1, x2 = x
    return np.sin((2 * x1) - (0.5 * math.pi)) + (3 * np.cos(x2)) + (0.5 * x1)

x1_bound = [-2, 3]
x2_bound = [-2, 1]
v_bound = [-0.1, 1]
max_iter=500

def pso(cost_func, dim=2, num_particles=50, c1=2, c2=2):

    # Initialize particles and velocities
    particles = np.random.uniform([x1_bound[0], x2_bound[0]], [x1_bound[1], x2_bound[1]], (num_particles, dim))
    velocities = np.random.uniform(v_bound[0], v_bound[1], (num_particles, dim))

    # Initialize the best positions and fitness values
    best_positions = np.copy(particles)
    best_fitness = np.array([cost_func(p) for p in particles])
    swarm_best_position = best_positions[np.argmin(best_fitness)]
    swarm_best_fitness = np.min(best_fitness)

    # Track best fitness and particle positions over iterations
    fitness_over_time = [swarm_best_fitness]
    particles_over_time = [np.copy(particles)]

    # Updating the velocity and position of each particle at each iteration
    for i in range(max_iter):
        r1 = np.random.uniform(0, 1, (num_particles, dim))
        r2 = np.random.uniform(0, 1, (num_particles, dim))
        velocities = velocities + c1 * r1 * (best_positions - particles) + c2 * r2 * (
                    swarm_best_position - particles)

        # Apply velocity bounds
        velocities = np.clip(velocities, v_bound[0], v_bound[1])

        # Update positions
        particles += velocities

        # Apply position bounds
        particles[:, 0] = np.clip(particles[:, 0], x1_bound[0], x1_bound[1])
        particles[:, 1] = np.clip(particles[:, 1], x2_bound[0], x2_bound[1])

        # Evaluate fitness of each particle
        fitness_values = np.array([cost_func(p) for p in particles])

        # Update best positions and fitness values
        improved_indices = np.where(fitness_values < best_fitness)
        best_positions[improved_indices] = particles[improved_indices]
        best_fitness[improved_indices] = fitness_values[improved_indices]
        if np.min(fitness_values) < swarm_best_fitness:
            swarm_best_position = particles[np.argmin(fitness_values)]
            swarm_best_fitness = np.min(fitness_values)

        # Track the best fitness value and particle positions
        fitness_over_time.append(swarm_best_fitness)
        particles_over_time.append(np.copy(particles))

    # The best solution found, fitness tracking, and particle positions over time
    return swarm_best_position, swarm_best_fitness, fitness_over_time, particles_over_time


# Problem dimensions
dim = 2
n_runs = 1  # For visualization
results = []

# Perform a single run with a specific random seed
np.random.seed(0)
solution, fitness, fitness_over_time, particles_over_time = pso(f, dim=dim)
results.append((solution, fitness, fitness_over_time, particles_over_time))

# Convergence curve
plt.figure()
plt.plot(fitness_over_time, label='Fitness over time')
plt.xlabel('Iteration')
plt.ylabel('Best Fitness')
plt.title('PSO Convergence')
plt.legend()
plt.show()

# Particles trajectory
plt.figure()
for i in range(len(particles_over_time[0])):
    particle_positions = np.array([particles_over_time[t][i] for t in range(len(particles_over_time))])
    plt.plot(particle_positions[:, 0], particle_positions[:, 1], marker='o', markersize=2, label=f'Particle {i + 1}')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Particle Trajectories')
plt.show()
