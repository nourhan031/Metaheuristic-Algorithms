import numpy as np
import matplotlib.pyplot as plt

def hill_climbing(starting_point, step_size, max_iterations=100, convergence_threshold=1e-6):
    objective_function = input("Enter the objective function in terms of x: ")
    # eval to dynamically create a lambda function from the user input
    f = lambda x: eval(objective_function)

    x = starting_point
    iterations = 0

    # storing x and y's values for plotting
    x_values = [x]
    y_values = [f(x)]

    while iterations < max_iterations:
        # generate a neighbor by adding the step size to the current x
        n = x + step_size
        # hill climbing step
        # if the objective value at the new neighbor > the current objective value, update x to the neighbor
        # else, break out of the loop (since no improvement is found)
        if f(n) > f(x):
            x = n
        else:
            break

        x_values.append(x)
        y_values.append(f(x))

        # check for convergence by comparing the improvement between consecutive iterations to the convergence threshold
        # where, 'convergence threshold' is
        # a small value that determines when the optimization process is considered converged.
        # if so, it checks if the absolute difference between the objective function values at consecutive steps is below this threshold.
        # if the improvement is smaller than the threshold, the algorithm stops, assuming it has converged to a solution
        if len(x_values) > 1 and np.abs(y_values[-1] - y_values[-2]) < convergence_threshold:
            break

        iterations += 1

    print("Optimal x:", x)
    print("Optimal Objective Value:", f(x))

    # Plot the objective function and the steps taken by the algorithm
    plot_objective_function(f, x_values, y_values)

def plot_objective_function(f, x_values, y_values):
    x_range = np.linspace(min(x_values) - 1, max(x_values) + 1, 100)
    y_range = [f(x) for x in x_range]

    plt.plot(x_range, y_range, label='Objective Function')
    plt.scatter(x_values, y_values, color='red', label='Steps')
    plt.xlabel('x')
    plt.ylabel('Objective Value')
    plt.legend()
    plt.show()

# Example usage with advanced features
starting_point = float(input("Enter the starting point for hill climbing: "))
step_size = float(input("Enter the step size for hill climbing: "))

hill_climbing(starting_point, step_size)



# def hill_climbing():
#     # Accepts user input for the objective function
#     objective_function = input("Enter the objective function in terms of x: ")
#
#     try:
#         # Dynamically create a lambda function from the user input
#         f = lambda x: eval(objective_function)
#
#         # Initialize starting point and step size
#         x = 0
#         step_size = 1
#
#         # Main loop for hill climbing
#         while True:
#             n = x + step_size
#             if f(n) > f(x):
#                 x = n
#             else:
#                 break
#
#         # Display optimal x and its corresponding objective value
#         print("Optimal x:", x)
#         print("Optimal Objective Value:", f(x))
#
#     except Exception as e:
#         print("Error:", e)
#
# # Execute the hill climbing function
# hill_climbing()
#
