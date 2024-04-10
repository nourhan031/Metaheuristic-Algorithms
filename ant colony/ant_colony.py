import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# a- calculate distance between citites
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
    eta = 1.0 / dist
    eta[eta == np.inf] = np.nanmax(eta)
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

# TESTING

# distance between cities
distance = dist_bet_cities(cities)
df_distance = pd.DataFrame(distance)
print("distance matrix values: ")
print(df_distance)

print()

# eta
eta = eta(distance)
df_eta = pd.DataFrame(eta)
print("eta matrix values: ")
print(df_eta)

print()

# tour length
n = len(cities)
tour, Lnn = tour_length(distance)



# set the initial pheromone value
tau = np.zeros((n,n))
tau_0 = 1 / (n * Lnn)
print("initial tau: ", tau_0)
# Create an n*n matrix of pheromones and set each element to tau_0
tau = np.full((n, n), tau_0)
tau_data = np.array([tau])
df_tau = pd.DataFrame(tau)
print("tau matrix values: ")
print(df_tau)


# Number of ants
m = len(cities)
# Generate m ants and place them over the cities
ants = np.random.permutation(m)

# Evaporation rate of the pheromone
evaporation_rate = 0.5
# Alpha parameter for the relative importance of pheromone
alpha = 1.0
# Beta parameter for the relative importance distance
beta = 2.0
# Number of cities
num_nodes = len(cities)

# Initialize the visited cities and the current city
visited = [0]
current_node = 0

def choose_next_node(current_node, visited, pheromone, distances):
    probabilities = []
    for i in range(num_nodes):
        if i not in visited:
            # Probability calculation using pheromone and distance
            prob = (pheromone[current_node][i] ** alpha) * ((1.0 / distances[current_node][i]) ** beta)
            probabilities.append(prob)
        else:
            probabilities.append(0)
    # Normalize probabilities to sum to 1
    probabilities = probabilities / np.sum(probabilities)
    next_node = np.random.choice(range(num_nodes), p=probabilities)
    return next_node

# While there are still cities to visit
while len(visited) < num_nodes:
    # Apply the state transition rule to find the next city to visit
    current_node = choose_next_node(current_node, visited, tau, distance)
    # Add the next city to the visited cities
    visited.append(current_node)

    # After constructing the tour, update pheromone matrix
    for i in range(num_nodes):
        for j in range(num_nodes):
            tau[i][j] *= evaporation_rate
            if j in visited:
                tau[i][j] += (1.0 / tour_length)

# Print the visited cities
print(visited)


def plot_tour(cities, tour):
    # Create a new figure
    plt.figure(figsize=(10, 10))
    # Plot the cities as points
    plt.scatter(cities[:, 0], cities[:, 1])
    # Plot the tour
    for i in range(len(tour) - 1):
        # Get the coordinates of the current city and the next city
        city1 = cities[tour[i]]
        city2 = cities[tour[i + 1]]
        # Plot a line between the current city and the next city
        plt.plot([city1[0], city2[0]], [city1[1], city2[1]], 'r-')
        # Plot a line from the last city back to the first city
    city1 = cities[tour[-1]]
    city2 = cities[tour[0]]
    plt.plot([city1[0], city2[0]], [city1[1], city2[1]], 'r-')
    # Show the plot
    plt.show()

# Plot the tour
plot_tour(cities, tour)
#
# # set the initial pheromone value
# tau = np.zeros((n,n))
# tau_0 = 1 / (n * Lnn)
# print("initial tau: ", tau_0)
# # Create an n*n matrix of pheromones and set each element to tau_0
# tau = np.full((n, n), tau_0)
# tau_data = np.array([tau])
# df_tau = pd.DataFrame(tau)
# print("tau matrix values: ")
# print(df_tau)