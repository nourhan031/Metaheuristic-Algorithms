import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# a- calculate distance between cities
def dist_bet_cities(cities):
    # cities: 2D array where each row represents a city and the 2 columns represent the x and y coordinates
    n = len(cities)
    dist = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            # cities[i][0] gets the x-coordinate of the i-th city
            # cities[i][1] gets the y-coordinate of the i-th city
            x1, y1 = cities[i][0], cities[i][1]
            x2, y2 = cities[j][0], cities[j][1]
            # Euclidean distance between i and j
            dist[i][j] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist


# b- eta = reciprocal of distances
def eta(dist):
    eta = np.zeros_like(dist)
    mask = dist != 0
    eta[mask] = 1.0 / dist[mask]
    max_eta = np.nanmax(eta)
    eta[~mask] = max_eta  # Assign maximum value to elements where division by zero occurred
    return eta


# c- tour length using nearest neighbor heuristic Lnn
def tour_length(dist):
    # choose a random city to start the tour
    start_city = random.randint(0, len(dist)-1)
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
         dist_to_others[tour] = float('inf')
        # find the city closest to the last city
         closest_city = np.argmin(dist_to_others)
        # add the closest city to the tour
         tour.append(closest_city)
        # add dist to the closest city to the tour length
         tour_length += dist[last_city][closest_city]
    # add the dist from the last city back to the start city
    tour_length += dist[tour[-1]][tour[0]]
    # return the tour and tour length
    return tour, tour_length

# [city_id, x_coordinate, y_coordinate]
cities_data = np.array([
    [1, 0, 1],
    [2, 0.009174316, 0.995412849],
    [3, 0.064220188, 0.438073402],
    [4, 0.105504591, 0.594036699],
    [5, 0.105504591, 0.706422024],
    [6, 0.105504591, 0.917431193],
    [7, 0.185779816, 0.454128438],
    [8, 0.240825687, 0.614678901],
    [9, 0.254587155, 0.396788998],
    [10, 0.38302753,  0.830275235],
    [11, 0.394495416, 0.839449537],
    [12, 0.419724769, 0.646788988],
    [13, 0.458715603, 0.470183489],
    [14, 0.593922025, 0.348509173],
    [15, 0.729357795, 0.642201837],
    [16, 0.731651377, 0.098623857],
    [17, 0.749999997, 0.403669732],
    [18, 0.770642198, 0.495412842],
    [19, 0.786697249, 0.552752296],
    [20, 0.811926602, 0.254587155],
    [21, 0.852217125, 0.442928131],
    [22, 0.861850152, 0.493004585],
    [23, 0.869762996, 0.463761466],
    [24, 0.871559638, 0],
    [25, 0.880733941, 0.08486239],
    [26, 0.880733941, 0.268348623],
    [27, 0.885321106, 0.075688073],
    [28, 0.908256877, 0.353211012],
    [29, 0.912270643, 0.43470948]
])
# Convert the list of tuples to a NumPy array
cities = np.array([city[1:] for city in cities_data])
# print(cities[1])


# TESTING
# distance between cities
distance = dist_bet_cities(cities)
df_distance = pd.DataFrame(distance)
# print("distance matrix values: ")
# print(df_distance)
#
# print()

# eta
eta = eta(distance)
df_eta = pd.DataFrame(eta)
# print("eta matrix values: ")
# print(df_eta)
#
# print()

# tour length
n = len(cities)
tour, Lnn = tour_length(distance)

# set the initial pheromone value
tau = np.zeros((n,n))
tau_0 = 1 / (n * Lnn)
# print("initial tau: ", tau_0)

# Create an n*n matrix of pheromones and set each element to tau_0
tau = np.full((n, n), tau_0)
tau_data = np.array([tau])
df_tau = pd.DataFrame(tau)
# print("tau matrix values: ")
# print(df_tau)


# Number of ants
m = 50
# Generate m ants and place them over the cities
ants = np.random.permutation(m)

# Evaporation rate of the pheromone
evaporation_rate = 0.5
# Alpha parameter for the relative importance of pheromone
alpha = 1.0
# Beta parameter for the relative importance distance
beta = 1.5
# Number of cities
num_nodes = len(cities)

# POINT 2
def choose_next_node(current_node, visited, pheromone, distances):
    probabilities = []
    for i in range(num_nodes):
        if i not in visited:
            # Probability calculation using pheromone and distance
            prob = (pheromone[current_node][i] ** alpha) * ((1.0 / distances[current_node][i]) ** beta)
        else:
            prob = 0  # Set probability to 0 for visited nodes
        probabilities.append(prob)

    # Normalize probabilities to sum to 1
    probabilities = probabilities / np.sum(probabilities)
    # Choose the next node based on the probabilities
    next_node = np.random.choice(range(num_nodes), p=probabilities)

    return next_node


# POINT 3
def remove_cycles(tour, distance):
    n = len(tour)
    improvement = True
    while improvement:
        improvement = False
        for i in range(1, n - 2):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue  # No need to reverse two adjacent edges
                new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
                new_length = sum(distance[new_tour[k - 1]][new_tour[k]] for k in range(1, n)) + distance[new_tour[-1]][new_tour[0]]
                if new_length < sum(distance[tour[k - 1]][tour[k]] for k in range(1, n)) + distance[tour[-1]][tour[0]]:
                    tour[:] = new_tour
                    improvement = True
                    break
            if improvement:
                break
    return tour



num_runs = 20
results = []

for run in range(num_runs):
    # Reset pheromone matrix for each run
    tau = np.full((num_nodes, num_nodes), tau_0)

    # Initialize the visited cities and the current city for each ant
    for ant in range(m):
        visited = [ant % num_nodes] # ensure that it doesn't access an index outside the valid range of num_nodes
        current_node = ant % num_nodes

        while len(visited) < num_nodes:
            # Apply the state transition rule to find the next city to visit for each ant
            current_node = choose_next_node(current_node, visited, tau, distance)
            visited.append(current_node)

        # After constructing the tour, remove cycles from tour
        tour = visited.copy()
        tour = remove_cycles(tour, distance)

        # Update pheromone for edges traversed by the ant
        for i in range(num_nodes - 1):
            tau[tour[i]][tour[i + 1]] += 1.0 / Lnn
        tau[tour[-1]][tour[0]] += 1.0 / Lnn

        # Apply evaporation
        tau *= (1 - evaporation_rate)

    # Store the tour constructed in this run
    results.append(tour)

    print(f"Run {run + 1}: Tour length = {sum(distance[tour[i - 1]][tour[i]] for i in range(1, len(tour))) + distance[tour[-1]][tour[0]]}")

# Print all the tours
print("\nAll Tours:")
for i, tour in enumerate(results):
    print(f"Run {i + 1}: {tour}")


