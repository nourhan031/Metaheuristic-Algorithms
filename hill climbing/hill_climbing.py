import numpy as np

def hill_climbing(starting_point, step_size, max_iterations=100, convergence_threshold=1e-6):
    objective_function = input("Enter the objective function in terms of x: ")
    f = lambda x: eval(objective_function)

    x = starting_point
    iterations = 0

    while iterations < max_iterations:
        n = x + step_size
        if f(n) > f(x):
            x = n
        else:
            break

        iterations += 1

    optimal_x = x
    optimal_objective_value = f(x)

    return optimal_x, optimal_objective_value

starting_point = float(input("Enter the starting point for hill climbing: "))
step_size = float(input("Enter the step size for hill climbing: "))

optimal_x, optimal_objective_value = hill_climbing(starting_point, step_size)
print("Optimal x:", optimal_x)
print("Optimal Objective Value:", optimal_objective_value)
