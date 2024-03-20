import matplotlib.pyplot as plt
import numpy as np

# Your functions and gradient descent algorithm
def f(x):
    return x**2

def df(x):
    return 2*x

def gradient_descent(f, df, init_pt, learning_rate=0.1, num_iter=1000):
    x = init_pt
    history = [x]  # to store all points
    for _ in range(num_iter):
        x_prev = x
        grad = df(x_prev)
        x = x_prev - learning_rate * grad
        history.append(x)
    return x, history

# Run gradient descent and get history
init_pt = 5
final_pt, history = gradient_descent(f, df, init_pt)

# Generate x values and corresponding y values
x_values = np.linspace(-init_pt, init_pt, 400)
y_values = f(x_values)

# Create the plot
plt.figure(figsize=(10, 6))
# plt.plot(x_values, y_values, label='f(x) = x^2')
plt.plot(x_values, y_values, label='f(x) = sin(x)')
plt.scatter(history, [f(x) for x in history], color='red', zorder=5)
plt.plot(history, [f(x) for x in history], linestyle='dashed', color='red', zorder=5)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
