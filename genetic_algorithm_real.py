import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# setting pandas display options so that the entire dataframe is printed wout truncation
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)

# function to decode binary to real
def decode(binary, min, max):
    decimal = int(''.join([str(bit) for bit in binary]), 2)

    return min + (decimal / (2**len(binary) - 1)) * (max - min)

# objective funtion
def f(x1, x2):
    return 8 - (x1 + 0.0317)**2 + x2**2
    #   return abs(x1+x2-1)

# fitness function
def fitness(individual, x_range, y_range):
    x1 = decode(individual[:len(individual)//2], * x_range)
    x2 = decode(individual[:len(individual)//2], * y_range)

    return f(x1, x2)

def rank_fitness(fitnesses, sp):
    N = len(fitnesses)
    sorted_indices = np.argsort(fitnesses)
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(N)
    rank_fitnesses = (2 - sp) + 2 * (sp - 1) * ((ranks) / (N - 1))
    return rank_fitnesses


pop = 10
generations =10
x_range = (-2, 2)
y_range = (-2, 2)

# increased number of bits in each chromosome to 10 in order ot inc the precision (2^10=1024 different values that can be represented)
generation = np.random.randint(2, size=(pop, 10))
best_fitnesses = []
avg_fitnesses = []

for gen in range(generations):
    fitnesses = []
    decoded_vals = []

    # iterate over each row in the matrix
    for row in generation:
        fitnesses.append(fitness(row, x_range, y_range))
        # decode binary and append to the decoded list
        x1 = decode(row[:len(row)//2], *x_range)
        x2 = decode(row[:len(row) // 2], *y_range)
        decoded_vals.append((x1, x2))

    rank_fitnesses = rank_fitness(fitnesses, sp=2)
    best_fitness = max(fitnesses)
    avg_fitness = sum(fitnesses)/len(fitnesses)
    best_fitnesses.append(best_fitness)
    avg_fitnesses.append(avg_fitness)

    df = pd.DataFrame({
        'chromosome': [row.tolist() for row in generation],
        'decoded values': decoded_vals,
        'fitness': fitnesses,
        'rank fitness': rank_fitnesses,
        'relative fitness': rank_fitnesses/rank_fitnesses.sum(),
        'cumulative fitness': np.cumsum(rank_fitnesses/rank_fitnesses.sum())
    })

    print(f"Generation {gen + 1}:")
    print(df)
    print(f"Best Fitness: {best_fitness}")
    print(f"Average Fitness: {avg_fitness}\n")

# Plotting the function
x1 = np.linspace(-2, 2, 100)  # replace with your range for x1
x2 = np.linspace(-2, 2, 100)  # replace with your range for x2

X1, X2 = np.meshgrid(x1, x2)
Y = f(X1, X2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Y)

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('F(X1, X2)')

plt.show()


# Plotting the fitness values
plt.figure()
plt.plot(best_fitnesses, label='Best Fitness')
plt.plot(avg_fitnesses, label='Average Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.show()