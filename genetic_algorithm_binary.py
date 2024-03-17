import numpy as np
import random
import pandas as pd

pop = 20
generations = 10

generation = np.random.randint(2, size=(pop, 5))
best_fitness_for_each_gen = []  # To store the best fitness values for each generation
avg_fitness_for_each_gen = []

# Iterate over each generation
for gen in range(generations):
    fitness = []

    # Iterate over each row in the matrix
    for row in generation:
        # Count the number of ones in the row and append to the fitness list
        fitness.append(np.sum(row))

    # sum of fitness values
    total_fitness = sum(fitness)
    # each fitness val/sum of fitness vals
    relative_fitness = [f / total_fitness for f in fitness]
    # cumulative fitness of each row
    cumulative_fitness = np.cumsum(relative_fitness)

    # Create a DataFrame for the current generation
    df = pd.DataFrame({
        'chromosome': [row.tolist() for row in generation],
        'fitness': fitness,
        'relative fitness': relative_fitness,
        'cumulative fitness': cumulative_fitness
    })

    print(f"Generation {gen + 1}:")
    print(df)
    print("")

    new_generation = []

    # number of offspring (half of the population size)
    num_offspring = pop // 2

    # Generate offspring using crossover and mutation
    for _ in range(num_offspring):
        # parent 1
        random_number1 = random.random()
        index1 = next((i for i, cf in enumerate(cumulative_fitness) if random_number1 < cf), 0) - 1
        if index1 == -1:
            index1 = len(cumulative_fitness) - 1
        parent1 = generation[index1]

        # parent 2
        random_number2 = random.random()
        index2 = next((i for i, cf in enumerate(cumulative_fitness) if random_number2 < cf), 0) - 1
        if index2 == -1:
            index2 = len(cumulative_fitness) - 1
        parent2 = generation[index2]


        def one_point_crossover(parent1, parent2, pCross=0.6):
            rand = random.random()
            # print probability
            print("Crossover probability:", rand)

            # If rand < pCross, perform crossover
            if rand < pCross:
                # Select a random crossover point (not the first or the last index)
                crossover_point = random.randint(1, len(parent1) - 1)

                # Convert numpy arrays to lists
                parent1 = parent1.tolist()
                parent2 = parent2.tolist()

                # Create offspring by combining the genes from the parents
                offspring1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                offspring2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

                return offspring1, offspring2
            else:
                # don't perform crossover (the offspring are clones of the parents)
                return parent1.tolist(), parent2.tolist()


        def mutation(offspring, offspring_number, pMut=0.05):
            # Generate a random number
            rand = random.random()

            # print mutation probability
            print("Mutation probability for offspring " + str(offspring_number) + ":", rand)

            # If rand < pMut, perform mutation
            if rand < pMut:
                # Select a random index for the bit to mutate
                bit_index = random.randint(0, len(offspring) - 1)

                # Perform mutation (flip the bit)
                offspring[bit_index] = 1 - offspring[bit_index]

                # Print the mutation
                print("Mutation occurred at bit", bit_index)

            return offspring


        # crossover
        offspring1, offspring2 = one_point_crossover(parent1, parent2)

        # mutation
        offspring1 = mutation(offspring1, 1)
        offspring2 = mutation(offspring2, 2)

        # Append offspring to new_generation list
        new_generation.append(offspring1)
        new_generation.append(offspring2)

    # Update the current generation with the new generation
    generation = np.array(new_generation)

    # Calculate and store best and average fitness values for the current generation
    best_fitness = max([np.sum(row) for row in generation])
    average_fitness = np.mean([np.sum(row) for row in generation])

    best_fitness_for_each_gen.append(best_fitness)
    avg_fitness_for_each_gen.append(average_fitness)

    # Print information for each generation
    print(f"Generation {gen + 1}: Best Fitness: {best_fitness}, Average Fitness: {average_fitness}")
    print("")

# Final generation
print("Final Generation:")
print(pd.DataFrame({
    'chromosome': [row.tolist() for row in generation],
    'fitness': [np.sum(row) for row in generation]
}))

# Print fitness values for the final generation
final_best_fitness = max([np.sum(row) for row in generation])
final_average_fitness = np.mean([np.sum(row) for row in generation])
print("Final Generation - Best Fitness: {}, Average Fitness: {}".format(final_best_fitness, final_average_fitness))
