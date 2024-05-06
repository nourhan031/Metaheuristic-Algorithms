import numpy as np
import math
import matplotlib.pyplot as plt

def f(x1, x2):
    return np.sin((2 * x1) - (0.5 * math.pi)) + (3 * np.cos(x2)) + (0.5 * x1)

n_iterations = 500
n_particles = 50
c1 = c2 = 2
v_max = 0.1 # max velocity

x1_bound = [-2, 3]
x2_bound = [-2, 1]

best_fitness = []

# perform several runs with different random seeds
for run in range(5):
    np.random.seed(run)

    # initialize the st position that lies within the lower and upper bounds of x1
    x_current = np.random.uniform(low=[x1_bound[0], x2_bound[0]],
                                  high=[x1_bound[1], x2_bound[1]],
                                  size=(n_particles, 2)) # specifies the shape of the output array
    # init the st velocity
    v_current = np.random.uniform(low=-v_max,
                                  high=v_max,
                                  size=(n_particles, 2)) # specifies that there should be 'n_particles' rows and 2 cols
                                                        # corresponding to the x and y components of each particle's velocity

    pbest = x_current.copy()
    # best min st pt among all particles
    """
    -> create a list of the func vals for each particle's init position.
    -> loop over each row in 'x_current', extracting 'x1' and 'x2' from each row to compute f(x1, x2)
    -> 'np.argmin()': returns the index of the smallest val in the list
    -> x_current[] : selects the row in 'x_current' corresponding to the gbest init position
    """
    gbest = x_current[np.argmin([f(x[0], x[1]) for x in x_current])]

    for iteration in range(n_iterations):
        """
        -> gen 2 sets of rand numbers from a uniform dist bet 0 and 1
        -> 'n_particles': #particles in each set
        -> 1" indicates that each set contains a single rand num for eah particle
        """
        r1, r2 = np.random.rand(2, n_particles, 1)
        for i in range(n_particles):
            v_current[i] = v_current[i] + c1 * r1[i] * (pbest[i] - x_current[i]) + c2 * r2[i] * (gbest - x_current[i])
            v_current[i] = np.clip(v_current[i], -v_max, v_max)  # clamp the velocity
            x_current[i] = x_current[i] + v_current[i]
            x_current[i] = np.clip(x_current[i], [x1_bound[0], x2_bound[0]],
                                   [x1_bound[1], x2_bound[1]])  # enforce boundary conditions

            # update personal best
            if f(x_current[i][0], x_current[i][1]) < f(pbest[i][0], pbest[i][1]):
                pbest[i] = x_current[i]

            # update global best
            if f(pbest[i][0], pbest[i][1]) < f(gbest[0], gbest[1]):
                gbest = pbest[i]


        best_fitness.append(f(gbest[0], gbest[1]))


    # plot the best fitness value in each iteration
    plt.scatter(x_current[:, 0], x_current[:, 1])
    plt.title(f"Particle Positions at the End of Run {run + 1}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()
