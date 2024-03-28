import random
import numpy as np
import pandas as pd
from scipy.optimize import minimize

pop = 20

def init_pop(pop_size=pop, R_max=20, R_min=0):
    return np.random.uniform(low=R_min, high=R_max, size=(pop_size,))

def tournament(pop, k):
    tournament_pool = random.sample(list(pop), k)  # select k random individuals
    best = max(tournament_pool)
    return best, tournament_pool

def arithmetic_cross(parent1, parent2, pCross=0.6):
    alpha = 0.01
    if np.random.rand() < pCross:
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = alpha * parent2 + (1 - alpha) * parent1
        return child1, child2
    else:
        return parent1, parent2

def gaussian_mutate(individual, sigma=0.5, pMut=0.05):
    if np.random.rand() < pMut:
        individual + np.random.normal(loc=0, scale=sigma)
    return individual

def objective(x):
    return - (8 - (x[0] + 0.0317) ** 2 + x[1] ** 2)

# Define the bounds for x1 and x2
bounds = [(-2, 2), (-2, 2)]

runs = 10
generations = 10

for run in range(runs):
    print(f"Run {run + 1}")

    population = init_pop()

    for generation in range(generations):
        print(f"  Generation {generation + 1}")

        parent1, tournament_pool1 = tournament(population, k=3)
        parent2, tournament_pool2 = tournament(population, k=3)

        child1, child2 = arithmetic_cross(parent1, parent2)

        mutated_child1 = gaussian_mutate(child1)
        mutated_child2 = gaussian_mutate(child2)

        res = minimize(objective, [0, 0], bounds=bounds)

        print(f"  Chosen individuals for first tournament: {tournament_pool1}")
        print(f"  Chosen individuals for second tournament: {tournament_pool2}")
        print()
        print(f"  Best individual from first tournament: {parent1}")
        print(f"  Best individual from second tournament: {parent2}")
        print()
        print(f"  Child 1: {child1}")
        print(f"  Child 2: {child2}")
        print()
        print(f"  Child 1 (after mutation): {mutated_child1}")
        print(f"  Child 2 (after mutation): {mutated_child2}")
        print()
        print(f"  Optimal solution: {res.x}")
        print(f"  Maximum value: {-res.fun}")
        print()
