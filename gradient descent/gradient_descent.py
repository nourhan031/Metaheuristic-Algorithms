def f(x):
    return x**2

def df(x):
    return 2*x

def gradient_descent(f, df, init_pt, learning_rate=0.1, num_iterations=1000):
    x = init_pt
    for i in range(num_iterations):
        gradient = df(x)
        x = x - learning_rate * gradient
    return x

print(gradient_descent(f, df, init_pt=10))