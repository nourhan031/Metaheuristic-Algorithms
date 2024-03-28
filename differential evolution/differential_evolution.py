import numpy as np
import random

# init a pop of 30 vectors, each with 3 elements rand generated from uniform dist with the specified ranges
pop = np.random.uniform(low = [-1, 2, 0], high = [1, 10, 5], size = (30, 3))
print(pop.shape)  # shape of arr
print(pop)  # contents of arr

# objective function is to maximize f(x) = x + y + z
fitness = []
# calculate the fitness of each vector in the pop
for arr in pop:
    fitness.append(np.sum(arr))
print(sorted(fitness, reverse=True))

# for each iteration you need to select 3 parents and a target parent,
# the 3 parents should not be the same and also the target parent should not be on of the three.
# How you can exclude a vector from a numpy array?


# removes the target parent (first vector) from the pop arr
# using np.delete() to del the row at index 0 along the specified axis (axis=0 -> rows)
pop_without_taregt_parent = np.delete(pop, 0, axis=0)
print(pop_without_taregt_parent)

# randomly select 3 indices from the remaining pop(after removing the target parent)
# using .choice() to select indices wout replacement
random_indices = np.random.choice(pop_without_taregt_parent.shape[0], size=3, replace=False)
print(random_indices)

# selects the 3 parents from the remaining pop based on the randomly chosen indices
selected_parents = pop_without_taregt_parent[random_indices]
print(selected_parents)


def get_three_parents(pop, parent_index_to_exclude):
    # Exclude the target parent from the population
    pop_without_target = np.delete(pop, parent_index_to_exclude, axis=0)

    # Randomly select 3 parents without replacement
    random_indices = np.random.choice(pop_without_target.shape[0], size=3, replace=False)

    selected_parents = pop_without_target[random_indices]

    return selected_parents

# selects 3 parents for the target parent at index 0
get_three_parents(pop, 0)



# init parameters for the DE
F = 2 # scaling factor fot mutation
CR = 0.9 # crossover prob
number_of_generations = 10
fitness_over_generations = []

for generation_number in range(0, number_of_generations):
    new_generation = []
    fitness = []
    # iterate over each vector in the current pop
    # enumerate gets both the vector and the index itself
    for target_vector_index, target_vector in enumerate(pop):
        # get 3 random parents using the function you made
        parents = get_three_parents(pop, target_vector_index)

        # subtract p1 from p2 --> use indexing to get the parent from the 3 parents you have
        difference_vector = parents[1] - parents[0]

        # the third parent + f times the difference vector
        mutant_vector = parents[2] + F * difference_vector

        # create a trial vector by performing crossover bet the target vector and the mutant based on CR's prob
        trial_vector = []
        for i in range(len(target_vector)):
            # generate random number using np.random.rand() and if it's smaller than or equal CR then
            if np.random.rand() <= CR:
                # append the mutant_vector in the trial_vector
                trial_vector.append(mutant_vector[i])
            else:
                # append the target_vector in the trial_vector
                trial_vector.append(target_vector[i])

        # append the better vector to the new generation
        new_generation.append(trial_vector if np.sum(trial_vector) > np.sum(target_vector) else target_vector)

    # calculate the fitness of the new generation
    for arr in new_generation:
        fitness.append(np.sum(arr))

    # print the best fitness of the current generation
    print("Generation {}".format(generation_number), sorted(fitness, reverse=True)[0])
    print("---------------------------")

    # append the best fitness to the fitness_over_generations list
    fitness_over_generations.append(sorted(fitness, reverse=True)[0])

    # replace the old population with the new generation
    pop = np.array(new_generation)

print(max(fitness_over_generations))

# objective funtion is to maximize f(x) = x + y + z
# calc and print the fitnesses of the vectors in the final pop
fitness = []
for arr in pop:
    fitness.append(np.sum(arr))
print(sorted(fitness, reverse=True))
