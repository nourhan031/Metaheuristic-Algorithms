import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def dist_bet_cities(cities):
    n = len(cities)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # cities[i][0] gets the x-coordinate of the i-th city
            # cities[i][1] gets the y-coordinate of the i-th city
            x1, y1 = cities[i][0], cities[i][1]
            x2, y2 = cities[j][0], cities[j][1]
            # Euclidean distance between i and j
            dist[i][j] = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

def tour_length(dist):
    # choose a random city to start the tour
    start_city = random.randint(0, len(dist) - 1)
    # init the tour with the start city
    tour = [start_city]
    # init tour length
    tour_length = 0

    # while there are still cities to visit
    while len(tour) < len(dist):
        # get the last visited city
        last_city = tour[-1]
        # get the dist from the last city to all other cities
        dist_to_others = dist[last_city].copy()
        # set the dist to the already visited to infinity
        dist_to_others[tour] = float("inf")
        # find the city closest to the last city
        closest_city = np.argmin(dist_to_others)
        # add the closest city to the tour
        tour.append(closest_city)
        # add dist to the closest city to the tour length
        tour_length += dist[last_city][closest_city]
    # add the dist from the last city back to the start city
    tour_length += dist[tour[-1]][tour[0]]
    return tour, tour_length

def choose_next_node(current_node, visited, pheromone, distances, alpha, beta):
    probabilities = []
    for i in range(len(cities)):
        if i not in visited:
            # Probability calculation using pheromone and distance
            prob = (pheromone[current_node][i] ** alpha) * ((1.0 / distances[current_node][i]) ** beta)
            probabilities.append(prob)
        else:
            probabilities.append(0)
    # Normalize probabilities to sum to 1
    probabilities = probabilities / np.sum(probabilities)
    next_node = np.random.choice(range(len(cities)), p=probabilities)
    return next_node
    # probabilities = np.zeros(len(distances))
    # total_probability = 0
    # for i in range(len(distances)):
    #     if i not in visited:
    #         probabilities[i] = (pheromone[current_node][i] ** alpha) * ((1.0 / distances[current_node][i]) ** beta)
    #         total_probability += probabilities[i]
    # if total_probability == 0:
    #     return -1
    # probabilities /= total_probability
    # next_node = np.random.choice(range(len(distances)), p=probabilities)
    #
    # return next_node

def optimization(dist, num_runs, m, alpha, beta, evaporation_rate):
    num_nodes = len(dist)
    results = []
    tour_lengths = []

    for run in range(num_runs):
        # Reset pheromone matrix with equal values for all edges
        tau = np.ones((num_nodes, num_nodes)) / num_nodes

        best_tour = None
        best_tour_length = float("inf")

        for ant in range(m):
            visited = [random.randint(0, num_nodes - 1)]
            current_node = visited[0]
            while len(visited) < num_nodes:
                current_node = choose_next_node(current_node, visited, tau, dist, alpha, beta)
                if current_node == -1:
                    break
                visited.append(current_node)

            # Make it a valid tour
            tour = list(set(visited))

            # Ensure it visits all cities once
            if len(tour) == num_nodes:
                tour_length = sum(dist[tour[i - 1]][tour[i]] for i in range(1, len(tour))) + dist[tour[-1]][tour[0]]
                if tour_length < best_tour_length:
                    best_tour = tour
                    best_tour_length = tour_length

                # Update pheromones for all ants (done for each ant after it constructs its tour)
                for i in range(num_nodes - 1):
                    tau[tour[i]][tour[i + 1]] += 1.0 / (tour_length ** 2)
                tau[tour[-1]][tour[0]] += 1.0 / (tour_length ** 2)

        # Apply evaporation
        tau *= (1 - evaporation_rate)

        # the best tour and its length for this run
        results.append(best_tour)
        tour_lengths.append(best_tour_length)
        print(f"Run {run + 1}: Tour length = {best_tour_length}")

    return results, tour_lengths

# [city_id, x_coordinate, y_coordinate]
cities_data = np.array([
    [1, 0, 1], [2, 0.009174316, 0.995412849], [3, 0.064220188, 0.438073402],
    [4, 0.105504591, 0.594036699], [5, 0.105504591, 0.706422024], [6, 0.105504591, 0.917431193],
    [7, 0.185779816, 0.454128438], [8, 0.240825687, 0.614678901], [9, 0.254587155, 0.396788998],
    [10, 0.38302753, 0.830275235], [11, 0.394495416, 0.839449537], [12, 0.419724769, 0.646788988],
    [13, 0.458715603, 0.470183489], [14, 0.593922025, 0.348509173], [15, 0.729357795, 0.642201837],
    [16, 0.731651377, 0.098623857], [17, 0.749999997, 0.403669732], [18, 0.770642198, 0.495412842],
    [19, 0.786697249, 0.552752296], [20, 0.811926602, 0.254587155], [21, 0.852217125, 0.442928131],
    [22, 0.861850152, 0.493004585], [23, 0.869762996, 0.463761466], [24, 0.871559638, 0],
    [25, 0.880733941, 0.08486239], [26, 0.880733941, 0.268348623], [27, 0.885321106, 0.075688073],
    [28, 0.908256877, 0.353211012], [29, 0.912270643, 0.43470948]
])
# convert the list of tuples to a NumPy array
cities = np.array([city[1:] for city in cities_data])

# Distance matrix
distance = dist_bet_cities(cities)

# Parameters
num_runs = 20
m = 50  # Number of ants
alpha = 1.0  # Relative importance of pheromone
beta = 1.5  # Relative importance of distance
evaporation_rate = 0.5  # Evaporation rate to ensure pheromones persist

# testing
results, tour_lengths = optimization(distance, num_runs, m, alpha, beta, evaporation_rate)

print("\nAll Tours:")
for i, tour in enumerate(results):
    tour_len = tour_lengths[i]
    print(f"Run {i + 1}: {tour} with length {tour_len}")

plt.plot(range(1, num_runs + 1), tour_lengths, '-o')
plt.xlabel('Run')
plt.ylabel('Tour Length')
plt.title('Tour Length over Runs')
plt.show()