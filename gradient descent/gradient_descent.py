import matplotlib.pyplot as plt
import numpy as np

# Your functions and gradient descent algorithm
def f(x, y):
    return x**2 + y**2

def df_dx(x, y):
    return 2 * x

def df_dy(x, y):
    return 2 * y


def gradient_descent(start_x, start_y, learning_rate, num_iterations):
    x = start_x
    y = start_y

    for i in range(num_iterations):
        # Calculate the gradients
        grad_x = df_dx(x, y)
        grad_y = df_dy(x, y)

        # Update the parameters
        x = x - learning_rate * grad_x
        y = y - learning_rate * grad_y

    return x, y, f(x, y)

start_x, start_y = 8, 8
learning_rate = 0.1
num_iterations = 20
x_opt, y_opt, f_opt = gradient_descent(start_x, start_y, learning_rate, num_iterations)

print("Optimal Solution:")
print("x =", x_opt)
print("y =", y_opt)

print("Optimal Value of the Function:")
print("f(x, y) =", f_opt)
