import random

import numpy as np
import pandas as pd
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
         dist_to_others = dist[last_city]
        # set the dist to the already visited to infinity
         dist_to_others[tour] = float('inf')
        # find the city closest to the last city
         closest_city = np.argmin(dist_to_others)
        # add the closest city to the tour
         tour.append(closest_city)
        # add dist to the closest city to the tour length
         tour_length += dist[last_city][closest_city]
        # add the dist from the last city back to the start city
         tour_length += dist[[tour][-1]][tour[0]]
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
print(df_distance)

print()

# eta
eta = eta(distance)
df_eta = pd.DataFrame(eta)
print(df_eta)

print()

# tour length