<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt

# Data
times = np.array([155, 180, 164, 162, 181, 182, 173, 190, 171, 170, 181, 182, 189, 184, 209, 210])
scores = np.array([51, 52, 54, 53, 55, 59, 61, 59, 63, 76, 64, 66, 69, 72, 70, 80])

# Normalized data
normalized_times = (times - np.mean(times)) / np.std(times)

# Initialize params
theta0 = 0
theta1 = 0
alpha = 1e-6
iterations = 1000

# Predict function
def h(x, theta0, theta1):
    return theta0 + theta1 * x

# Cost function
def cost_func(times, scores, theta0, theta1):
    m = len(times)
    return (1/(2*m)) * np.sum((h(times, theta0, theta1) - scores)**2)

# Gradient descent
def gradient_descent(times, scores, theta0, theta1, alpha, iterations):
    m = len(times)
    cost_history = [] 
    for i in range(iterations):
      grad0 = (1/m) * np.sum(h(times, theta0, theta1) - scores)
      grad1 = (1/m) * np.sum((h(times, theta0, theta1) - scores) * times)
      theta0 = theta0 - alpha * grad0
      theta1 = theta1 - alpha * grad1
      cost = cost_func(times, scores, theta0, theta1)
      cost_history.append(cost)
    return theta0, theta1, cost_history

def plot_final_result(times, scores, theta0, theta1, cost, iterations):
    plt.close("all")
    fig, ax = plt.subplots()

    # Plot the original data points
    ax.scatter(times, scores, color="blue", label="Data")

    # Plot the final prediction line based on theta0 and theta1
    line = theta0 + theta1 * times
    ax.plot(times, line, color="red", label="Prediction")
    ax.legend()

    ax.text(0.3, 0.95, f"Iteration: {iterations}, Cost: {cost:.4f}", transform=ax.transAxes)
    ax.text(0.3, 0.90, f"Theta0: {theta0:.4f}, Theta1: {theta1:.4f}", transform=ax.transAxes)

    plt.show()

theta0, theta1, cost_history = gradient_descent(times, scores, theta0, theta1, alpha, iterations)
plot_final_result(times, scores, theta0, theta1, cost_history[-1], iterations)
=======
import numpy as np
import matplotlib.pyplot as plt

# Data
times = np.array([155, 180, 164, 162, 181, 182, 173, 190, 171, 170, 181, 182, 189, 184, 209, 210])
scores = np.array([51, 52, 54, 53, 55, 59, 61, 59, 63, 76, 64, 66, 69, 72, 70, 80])

# Normalized data
normalized_times = (times - np.mean(times)) / np.std(times)

# Initialize params
theta0 = 0
theta1 = 0
alpha = 1e-6
iterations = 1000

# Predict function
def h(x, theta0, theta1):
    return theta0 + theta1 * x

# Cost function
def cost_func(times, scores, theta0, theta1):
    m = len(times)
    return (1/(2*m)) * np.sum((h(times, theta0, theta1) - scores)**2)

# Gradient descent
def gradient_descent(times, scores, theta0, theta1, alpha, iterations):
    m = len(times)
    cost_history = [] 
    for i in range(iterations):
      grad0 = (1/m) * np.sum(h(times, theta0, theta1) - scores)
      grad1 = (1/m) * np.sum((h(times, theta0, theta1) - scores) * times)
      theta0 = theta0 - alpha * grad0
      theta1 = theta1 - alpha * grad1
      cost = cost_func(times, scores, theta0, theta1)
      cost_history.append(cost)
    return theta0, theta1, cost_history

def plot_final_result(times, scores, theta0, theta1, cost, iterations):
    plt.close("all")
    fig, ax = plt.subplots()

    # Plot the original data points
    ax.scatter(times, scores, color="blue", label="Data")

    # Plot the final prediction line based on theta0 and theta1
    line = theta0 + theta1 * times
    ax.plot(times, line, color="red", label="Prediction")
    ax.legend()

    ax.text(0.3, 0.95, f"Iteration: {iterations}, Cost: {cost:.4f}", transform=ax.transAxes)
    ax.text(0.3, 0.90, f"Theta0: {theta0:.4f}, Theta1: {theta1:.4f}", transform=ax.transAxes)

    plt.show()

theta0, theta1, cost_history = gradient_descent(times, scores, theta0, theta1, alpha, iterations)
plot_final_result(times, scores, theta0, theta1, cost_history[-1], iterations)
>>>>>>> 3be0b58 (Add Chapter1, Chapter2, and Chapter3)
