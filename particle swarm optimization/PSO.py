import random
import numpy as np
import math
import matplotlib.pyplot as plt

def f(x):
    x1, x2 = x
    return np.sin((2 * x1) - (0.5 * math.pi)) + (3 * np.cos(x2)) + (0.5 * x1)


def pso(cost_func, dim=2, num_particles=50, n_iterations=500, c1=2, c2=2):
    x1_bound = [-2, 3]
    x2_bound = [-2, 1]
    v_bound = [-0.1, 1]

    # Initialize particles and velocities
    particles = np.random.uniform([x1_bound[0], x2_bound[0]], [x1_bound[1], x2_bound[1]], (num_particles, dim))
    velocities = np.random.uniform(v_bound[0], v_bound[1], (num_particles, dim))

    # Initialize the best positions and fitness values
    best_positions = np.copy(particles)
    best_fitness = np.array([cost_func(p) for p in particles])
    swarm_best_position = best_positions[np.argmin(best_fitness)]
    swarm_best_fitness = np.min(best_fitness)

    # Kepp track of the best fitness over iterations
    fitness_over_time = [swarm_best_fitness]

    for i in range(n_iterations):
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

        # Evaluate each particle's fitness
        fitness_values = np.array([cost_func(p) for p in particles])

        # Update best positions and fitness values
        improved_indices = np.where(fitness_values < best_fitness)
        best_positions[improved_indices] = particles[improved_indices]
        best_fitness[improved_indices] = fitness_values[improved_indices]
        if np.min(fitness_values) < swarm_best_fitness:
            swarm_best_position = particles[np.argmin(fitness_values)]
            swarm_best_fitness = np.min(fitness_values)

        fitness_over_time.append(swarm_best_fitness)

    # The best solution found
    return swarm_best_position, swarm_best_fitness, fitness_over_time


# Problem dimensions
dim = 2
n_runs = 5
results = []

# Several runs with different random seeds
for run in range(n_runs):
    np.random.seed(run)
    solution, fitness, fitness_over_time = pso(f, dim=dim)
    results.append((solution, fitness, fitness_over_time))
    # solution, fitness = pso(f, dim=dim)
    # results.append((solution, fitness))

# Solution and fitness value for each run
for i, (solution, fitness, fitness_over_time) in enumerate(results):
    print(f'Run {i + 1}:')
    print('Solution:', solution)
    print('Fitness:', fitness)
    print()

    plt.plot(fitness_over_time, label=f'Run {i + 1}')

plt.xlabel('Iteration')
plt.ylabel('Best Fitness')
plt.title('Convergence')
plt.legend()
plt.show()

