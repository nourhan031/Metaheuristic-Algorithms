import math
import numpy as np
import matplotlib.pyplot as plt
import random


def distance(point1, point2):
    return math.dist(point1,point2)


def choose_next_node(current_point, visited, pheromone, points, alpha, beta):
    probabilities = []
    for i, visited_status in enumerate(visited):
        if not visited_status:
            prob = (pheromone[current_point, i] ** alpha) / (distance(points[current_point], points[i]) ** beta)
            probabilities.append(prob)
        else:
            probabilities.append(0)
    probabilities /= np.sum(probabilities)
    next_point = np.random.choice(np.arange(len(visited)), p=probabilities)
    return next_point


def aco(cities, n_ants, n_iterations, alpha, beta, evaporation_rate, Q):
    points = cities[:, 1:]
    n_points = len(points)
    pheromone = np.ones((n_points, n_points))
    best_path = None
    best_path_length = np.inf

    for iteration in range(n_iterations):
        paths = []
        path_lengths = []

        for ant in range(n_ants):
            visited = [False] * n_points
            current_point = random.randint(0, n_points - 1)
            visited[current_point] = True
            path = [current_point]
            path_length = 0

            while False in visited:
                next_point = choose_next_node(current_point, visited, pheromone, points, alpha, beta)
                path.append(next_point)
                path_length += distance(points[current_point], points[next_point])
                visited[next_point] = True
                current_point = next_point

            paths.append(path)
            path_lengths.append(path_length)

            if path_length < best_path_length:
                best_path = path
                best_path_length = path_length

        pheromone *= evaporation_rate

        for path, path_length in zip(paths, path_lengths):
            for i in range(n_points - 1):
                pheromone[path[i], path[i + 1]] += Q / path_length
            pheromone[path[-1], path[0]] += Q / path_length

    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], c='r', marker='o')
    for i in range(n_points - 1):
        plt.plot([points[best_path[i], 0], points[best_path[i + 1], 0]],
                 [points[best_path[i], 1], points[best_path[i + 1], 1]],
                 c='g', linestyle='-', linewidth=2, marker='o')
    plt.plot([points[best_path[0], 0], points[best_path[-1], 0]],
             [points[best_path[0], 1], points[best_path[-1], 1]],
             c='g', linestyle='-', linewidth=2, marker='o')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Optimized Path')
    plt.show()


# Example usage:
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

aco(cities_data, n_ants=10, n_iterations=100, alpha=1, beta=1, evaporation_rate=0.5, Q=1)
