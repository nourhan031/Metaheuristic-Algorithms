import numpy as np
import random
import pandas as pd

# number of runs
num_runs = 10

# number of generations
num_generations = 100

# matrix of 20 x 5
pop = 20

for run in range(num_runs):
    # Set the seed for reproducibility
    np.random.seed(run)

    # from 0(inclusive) to 2(exclusive)
    generation = np.random.randint(2, size=(pop, 5))

    # Initialize the history vectors
    best_hist = []
    avg_hist = []

    for gen in range(num_generations):
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

        # Append the max fitness and average fitness to the history vectors
        best_hist.append(max(fitness))
        avg_hist.append(total_fitness / pop)

        df = pd.DataFrame({
            'chromosome': [row.tolist() for row in generation],
            'fitness': fitness,
            'relative fitness': relative_fitness,
            'cumulative fitness': cumulative_fitness
        })

        print("Run " + str(run+1) + ", Generation " + str(gen+1) + ":")
        print(df)
        print("Best fitness: ", best_hist[-1])
        print("Average fitness: ", avg_hist[-1])
        print("")

        new_generation = []

        # Elitism: Keep the best two individuals
        elite_indices = np.argsort(fitness)[-2:]  # Get the indices of the best two individuals
        elites = generation[elite_indices]  # Get the best two individuals

        # number of generations
        for _ in range((pop - 2) // 2):  # Adjust the range to account for the two elites
            # parent 1
            # generate a rand num between 0-1
            random_number1 = random.random()
            # find the index that the random number belongs to
            index1 = next((i for i, cf in enumerate(cumulative_fitness) if random_number1 < cf), 0)-1

            # if index is -1, set it to the last index
            if index1 == -1:
                index1 = len(cumulative_fitness) - 1

            # get the row in the generation matrix that corresponds to the index
            parent1 = generation[index1]

            # parent 2
            # generate a rand num between 0-1
            random_number2 = random.random()
            # find the index that the random number belongs to
            index2 = next((i for i, cf in enumerate(cumulative_fitness) if random_number2 < cf), 0)-1

            # if index is -1, set it to the last index
            if index2 == -1:
                index2 = len(cumulative_fitness) - 1

            # get the row in the generation matrix that corresponds to the index
            parent2 = generation[index2]

            # crossover
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


            # perform crossover
            offspring1, offspring2 = one_point_crossover(parent1, parent2)

            # Perform mutation
            offspring1 = mutation(offspring1,1)
            offspring2 = mutation(offspring2,2)

            # Append offsprings to new_generation list
            new_generation.append(offspring1)
            new_generation.append(offspring2)

            # Print information for each step
            print("rand generated number:", random_number1)
            print("index:", index1)
            print("Parent 1:", parent1.tolist())
            print("")
            print("rand generated number:", random_number2)

        # Add the elites to the new generation
        new_generation.extend(elites.tolist())

        # Replace the old generation with the new one
        generation = np.array(new_generation)
