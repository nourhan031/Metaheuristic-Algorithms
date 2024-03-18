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

# objective function
def f(x1, x2):
    return 8 - (x1 + 0.0317)**2 + x2**2

# fitness function - (UNCONSTRAINED)
# def fitness(individual, x_range, y_range):
#     x1 = decode(individual[:len(individual)//2], * x_range)
#     x2 = decode(individual[:len(individual)//2], * y_range)
#
#     return f(x1, x2)

# fitness function (CONSTRAINED)
def fitness(individual, x_range, y_range):
    x1 = decode(individual[:len(individual)//2], * x_range)
    x2 = decode(individual[:len(individual)//2], * y_range)

    # objective function value
    f_val = f(x1, x2)

    # constraint penalty
    penalty = abs(x1 + x2 - 1)

    # fitness after penalty
    return f_val - penalty

def rank_fitness(fitnesses, sp):
    N = len(fitnesses)
    sorted_indices = np.argsort(fitnesses)
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(N)
    rank_fitnesses = (2 - sp) + 2 * (sp - 1) * ((ranks) / (N - 1))
    return rank_fitnesses

# crossover
def two_point_crossover(p1, p2, pCross=0.6):
    rand = random.random()

    if rand < pCross:
        # select the random crossover pts (exclude the first and the last indices)
        crossover_pt1 = random.randint(1, len(p1)-2)
        crossover_pt2 = random.randint(crossover_pt1, len(p1) - 1)

        p1 = p1.tolist()
        p2 = p2.tolist()

        # perform crossover
        offspring1 = np.concatenate(
            (p1[:crossover_pt1], p2[crossover_pt1:crossover_pt2], p1[crossover_pt2:]))
        offspring2 = np.concatenate(
            (p2[:crossover_pt1], p1[crossover_pt1:crossover_pt2], p2[crossover_pt2:]))

        # convert offspring to binary strings
        offspring1 = [int(bit) for bit in offspring1]
        offspring2 = [int(bit) for bit in offspring2]

        return np.array(offspring1), np.array(offspring2)
    else:
        return np.copy(p1), np.copy(p2)

# mutation
def mutation(offspring, offspring_number, pMut=0.05):
    # Generate a random number
    rand = random.random()

    # print mutation probability
    # print("Mutation probability for offspring " + str(offspring_number) + ":", rand)

    # If rand < pMut, perform mutation
    if rand < pMut:
        # Select a random index for the bit to mutate
        bit_index = random.randint(0, len(offspring) - 1)

        # Perform mutation (flip the bit)
        offspring[bit_index] = 1 - offspring[bit_index]

        # Print the mutation
        # print("Mutation occurred at bit", bit_index)

    return offspring

pop = 10
generations = 100
x_range = (-2, 2)
y_range = (-2, 2)
runs = 10
all_runs_avg_fitnesses_no_elitism = []
all_runs_avg_fitnesses_elitism = []

for run in range(runs):
    # Reset the initial generation for each run
    generation = np.random.randint(2, size=(pop, 10))
    avg_fitnesses_no_elitism = []
    avg_fitnesses_elitism = []

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

        rank_fitnesses = rank_fitness(fitnesses, sp=2) # selection pressure
        best_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses)/len(fitnesses)
        avg_fitnesses_no_elitism.append(avg_fitness)
        cumulative_fitness = np.cumsum(rank_fitnesses/rank_fitnesses.sum())

        df = pd.DataFrame({
            'chromosome': [row.tolist() for row in generation],
            'decoded values': decoded_vals,
            'fitness': fitnesses,
            'rank fitness': rank_fitnesses,
            'relative fitness': rank_fitnesses/rank_fitnesses.sum(),
            'cumulative fitness': cumulative_fitness
        })

        print(f"Generation {gen + 1}:")
        print(df)
        print(f"Best Fitness: {best_fitness}")
        print(f"Average Fitness: {avg_fitness}\n")

        new_generation = []
        num_offspring = pop // 2

        for i in range(num_offspring):
            # parent 1
            rand1 = random.random()
            i1 = next((i for i, cf in enumerate(cumulative_fitness) if rand1 < cf), 0)-1

            if i1 == -1:
                i1 = len(cumulative_fitness)-1
            p1 = generation[i1]

            # parent 2
            rand2 = random.random()
            i2 = next((i for i, cf in enumerate(cumulative_fitness) if rand2 < cf), 0) - 1
            if i2 == -1:
                i2 = len(cumulative_fitness) - 1
            p2 = generation[i2]

            # perform crossover
            offspring1, offspring2 = two_point_crossover(p1, p2, pCross=0.6)

            # perform mutation
            offspring1 = mutation(offspring1, i)
            offspring2 = mutation(offspring2, i)

            # add offspring to the new generation
            new_generation.append(offspring1)
            new_generation.append(offspring2)

        # replace the current generation with the new one
        generation = np.array(new_generation)

        # Implement elitism
        elite_indices = np.argsort(fitnesses)[-2:]  # Get the indices of the 2 fittest individuals
        elites = np.copy(generation[elite_indices])  # Copy the elites

        new_generation[-2:] = elites  # Replace the last 2 individuals with the elites
        generation = np.array(new_generation)

        # Calculate the average fitness of the current generation after elitism
        fitnesses = [fitness(individual, x_range, y_range) for individual in generation]
        avg_fitness = sum(fitnesses)/len(fitnesses)
        avg_fitnesses_elitism.append(avg_fitness)

    # Calculate the average fitness of all generations for the current run
    run_avg_fitness_no_elitism = sum(avg_fitnesses_no_elitism)/len(avg_fitnesses_no_elitism)
    all_runs_avg_fitnesses_no_elitism.append(run_avg_fitness_no_elitism)

    run_avg_fitness_elitism = sum(avg_fitnesses_elitism)/len(avg_fitnesses_elitism)
    all_runs_avg_fitnesses_elitism.append(run_avg_fitness_elitism)

# Calculate the average of the average fitnesses of all runs
overall_avg_fitness_no_elitism = sum(all_runs_avg_fitnesses_no_elitism)/len(all_runs_avg_fitnesses_no_elitism)
overall_avg_fitness_elitism = sum(all_runs_avg_fitnesses_elitism)/len(all_runs_avg_fitnesses_elitism)

print(f'Overall Average Fitness without Elitism: {overall_avg_fitness_no_elitism}')
print(f'Overall Average Fitness with Elitism: {overall_avg_fitness_elitism}')

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

# After the runs loop
plt.figure()
plt.plot(all_runs_avg_fitnesses_no_elitism, label='Average Fitness per Run without Elitism')
plt.plot(all_runs_avg_fitnesses_elitism, label='Average Fitness per Run with Elitism')
plt.xlabel('Run')
plt.ylabel('Average Fitness')
plt.legend()
plt.title('Average Fitness per Run')
plt.show()
