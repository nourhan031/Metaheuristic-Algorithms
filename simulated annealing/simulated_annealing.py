import random
import math
import matplotlib.pyplot as plt

# generates a neighboring solution close to C by adding a small random change.
# 'step_size' parameter controls the size of the change
def Next(C, step_size=1):
    # A solution close to C is C plus a small random change
    return C + random.uniform(-step_size, step_size)

def custom_objective_function(C):
    # E(C) = C * sin(C)
    return C * math.sin(C)

def simulated_annealing():
    T_max = float(input("Enter the maximum temperature: "))
    T_min = float(input("Enter the minimum temperature: "))
    C_init = float(input("Enter the initial solution: "))
    step_size = float(input("Enter the step size for generating neighboring solutions: "))

    C = C_init
    T = T_max

    # Lists to store the progression of solutions and energies
    solution_history = [C]
    energy_history = [custom_objective_function(C)]

    # simulated annealing loop
    # until the temperature T drops below T_min
    while T > T_min:
        EC = custom_objective_function(C)
        N = Next(C, step_size)
        EN = custom_objective_function(N)
        # Evaluate the current energy (EC)
        # and generate a neighboring solution (N) with its energy (EN).
        delta_E = EN - EC

        # acceptance criteria:
        # If the new solution (N) has higher energy, accept it.
        # If the new solution has lower energy, accept it with a probability determined by the criterion
        if delta_E > 0:
            C = N
        else:
            if math.exp(delta_E / T) > random.random():
                C = N

        # Decrease temperature
        T *= 0.99  # cooling schedule

        # Record the progression
        solution_history.append(C)
        energy_history.append(EC)

    print("Optimal Solution:", C)
    print("Optimal Energy:", custom_objective_function(C))

    # Visualize the convergence
    plot_convergence(solution_history, energy_history)

def plot_convergence(solution_history, energy_history):
    plt.plot(solution_history, energy_history, marker='o', linestyle='-', color='b')
    plt.xlabel('Solution')
    plt.ylabel('Energy')
    plt.title('Convergence of Simulated Annealing')
    plt.show()

simulated_annealing()



# import random
# import math
#
# def Next(C):
#     # A solution close to C is C plus a small random change
#     return C + random.uniform(0, 1)
#
# def E(C):
#     # The energy of a solution is just its square
#     return C * C
#
# def simulated_annealing():
#     T_max = float(input("Enter the maximum temperature: "))
#     T_min = float(input("Enter the minimum temperature: "))
#     C_init = float(input("Enter the initial solution: "))
#
#     C = C_init
#     T = T_max
#
#     while T > T_min:
#         EC = E(C)
#         N = Next(C)
#         EN = E(N)
#         delta_E = EN - EC
#
#         if delta_E > 0:
#             C = N
#         else:
#             if math.exp(delta_E / T) > random.random():  # Fix the random.random() call
#                 C = N
#
#         # Decrease temperature
#         T *= 0.99  # Adjust the cooling schedule
#
#     print("Optimal Solution:", C)
#     print("Optimal Energy:", E(C))
#
# simulated_annealing()
