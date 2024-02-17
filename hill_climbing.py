# this is a simplified hill climbing example where the problem is one-dimensional

def hill_climbing():
    objective_function = input("Enter the objective function in terms of x: ")
    f = lambda x: eval(objective_function)

    x = 0
    step_size = 1

    while True:
        n = x + step_size
        if f(n) > f(x):
            x = n
        else:
            break

    print("Optimal x:", x)
    print("Optimal Objective Value:", f(x))

hill_climbing()
