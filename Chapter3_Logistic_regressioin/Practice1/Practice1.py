<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.array([0.245, 0.247, 0.285, 0.299, 0.327, 0.347, 0.356, 0.36, 0.363, 0.364, 0.398, 0.4, 0.409, 0.421,
              0.432, 0.473, 0.509, 0.529, 0.561, 0.569, 0.594, 0.638, 0.656, 0.816, 0.853, 0.938, 1.036, 1.045])
y = np.array([0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

# Initialize params
np.random.seed(0)   # ensurred the same random numbers generated every time
theta0 = np.random.rand()
theta1 = np.random.rand()
alpha = 1e-4
iterations = 1000

# Logistic function
def logistic_function(z):
    return 1 / (1 + np.exp(-z))

# Predict function
def h(x, theta0, theta1):
    z = theta0 + theta1 * x
    g_z = logistic_function(z)
    return g_z

# Cost function
def cost_function(x, y, theta0, theta1):
    m = len(x)
    epsilon = 1e-15   #ensured log dont go to infinity
    return (-1/m) * np.sum(y * np.log(h(x, theta0, theta1) + epsilon) + (1 - y) * np.log(1 - h(x, theta0, theta1)))

# Gradient descent
def gradient_descent(x, y, theta0, theta1, alpha, iterations):
    m = len(x)
    cost_history = []
    for i in range(iterations):
        grad0 = (1/m) * np.sum(h(x, theta0, theta1) - y)
        grad1 = (1/m) * np.sum((h(x, theta0, theta1) - y) * x)
        theta0 = theta0 - alpha * grad0
        theta1 = theta1 - alpha * grad1
        cost = cost_function(x, y, theta0, theta1)
        cost_history.append(cost)
    return theta0, theta1, cost_history

def plot_final_result(x, y, theta0, theta1, cost, iterations):
    plt.close('all')
    fig, ax = plt.subplots()
    plt.xlabel('Grains size')
    plt.ylabel('Spider appearance')

    # Plot the original data points
    ax.scatter(x, y, color="blue", label="Data")

    # Plot the final prediction line based on theta0 and theta1
    line = h(x, theta0, theta1) # Logistic function (sigmoid)
    ax.plot(x, line, color="red", label="Prediction")

    ax.legend()

    # Add text information about the cost and theta values
    plt.figtext(0.3, 1.07, f"Iteration: {iterations}, Cost: {cost:.4f}", transform=ax.transAxes)
    plt.figtext(0.3, 1.02, f"Theta0: {theta0:.4f}, Theta1: {theta1:.4f}", transform=ax.transAxes)

    plt.show()

theta0, theta1, cost_history = gradient_descent(x, y, theta0, theta1, alpha, iterations)
plot_final_result(x, y, theta0, theta1, cost_history[-1], iterations)
=======
import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.array([0.245, 0.247, 0.285, 0.299, 0.327, 0.347, 0.356, 0.36, 0.363, 0.364, 0.398, 0.4, 0.409, 0.421,
              0.432, 0.473, 0.509, 0.529, 0.561, 0.569, 0.594, 0.638, 0.656, 0.816, 0.853, 0.938, 1.036, 1.045])
y = np.array([0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

# Initialize params
np.random.seed(0)   # ensurred the same random numbers generated every time
theta0 = np.random.rand()
theta1 = np.random.rand()
alpha = 1e-4
iterations = 1000

# Logistic function
def logistic_function(z):
    return 1 / (1 + np.exp(-z))

# Predict function
def h(x, theta0, theta1):
    z = theta0 + theta1 * x
    g_z = logistic_function(z)
    return g_z

# Cost function
def cost_function(x, y, theta0, theta1):
    m = len(x)
    epsilon = 1e-15   #ensured log dont go to infinity
    return (-1/m) * np.sum(y * np.log(h(x, theta0, theta1) + epsilon) + (1 - y) * np.log(1 - h(x, theta0, theta1)))

# Gradient descent
def gradient_descent(x, y, theta0, theta1, alpha, iterations):
    m = len(x)
    cost_history = []
    for i in range(iterations):
        grad0 = (1/m) * np.sum(h(x, theta0, theta1) - y)
        grad1 = (1/m) * np.sum((h(x, theta0, theta1) - y) * x)
        theta0 = theta0 - alpha * grad0
        theta1 = theta1 - alpha * grad1
        cost = cost_function(x, y, theta0, theta1)
        cost_history.append(cost)
    return theta0, theta1, cost_history

def plot_final_result(x, y, theta0, theta1, cost, iterations):
    plt.close('all')
    fig, ax = plt.subplots()
    plt.xlabel('Grains size')
    plt.ylabel('Spider appearance')

    # Plot the original data points
    ax.scatter(x, y, color="blue", label="Data")

    # Plot the final prediction line based on theta0 and theta1
    line = h(x, theta0, theta1) # Logistic function (sigmoid)
    ax.plot(x, line, color="red", label="Prediction")

    ax.legend()

    # Add text information about the cost and theta values
    plt.figtext(0.3, 1.07, f"Iteration: {iterations}, Cost: {cost:.4f}", transform=ax.transAxes)
    plt.figtext(0.3, 1.02, f"Theta0: {theta0:.4f}, Theta1: {theta1:.4f}", transform=ax.transAxes)

    plt.show()

theta0, theta1, cost_history = gradient_descent(x, y, theta0, theta1, alpha, iterations)
plot_final_result(x, y, theta0, theta1, cost_history[-1], iterations)
>>>>>>> 3be0b58 (Add Chapter1, Chapter2, and Chapter3)
