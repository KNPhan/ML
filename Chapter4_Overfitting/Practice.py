import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Variables to store means and deviation for each feature
mu = []
std = []

def load(filename):
    df = pd.read_csv(filename)
    data = df.to_numpy(dtype=float)
    normalize(data)

    # Extract features and target
    x = data[:, :-1]  # All columns except the last one
    y = data[:, -1]   # Only the last column
    return x, y

def normalize(data):
    mu.clear()
    std.clear()
    # Loop through all columns except the last one
    for i in range(data.shape[1] - 1):
        mu.append(np.mean(data[:, i]))
        std.append(np.std(data[:, i]))
        data[:, i] = (data[:, i] - np.mean(data[:, i])) / np.std(data[:, i])

def h(x,theta):
	  return np.matmul(x, theta)

# Updated cost_function
def cost_function(x, y, theta, lambda_):
    m = y.shape[0]
    regularization_term = (lambda_ / (2 * m)) * np.sum(np.square(theta[1:]))   # Exclude theta[0] from regularization
    return (1 / (2 * m)) * np.sum((h(x, theta) - y) ** 2) + regularization_term

# Updated gradient descent
def gradient_descent(x, y, theta, alpha, lambda_, iterations):
    m = x.shape[0]
    cost_history = []

    for i in range(iterations):
        gradient = (1/m) * (x.T @ (h(x, theta) - y))
        theta[1:] = theta[1:] - alpha * (gradient[1:] + (lambda_ / m) * theta[1:])
        theta[0] = theta[0] - alpha * gradient[0]   # Exclude theta[0] from regularization
        cost = cost_function(x, y, theta, lambda_)
        cost_history.append(cost)
    return theta, cost_history

def plot_cost(cost_history, iterations, alpha):
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title(f'Cost vs. Iterations (alpha = {alpha})')
    plt.plot(iterations, cost_history, 'r', linewidth = "5")
    plt.show()

def predict(x, theta):
    x = np.array(x)
    # Normalize the input using the same mu and std from the training data
    x = (x - mu) / std
    # Add intercept term
    x = np.hstack((np.ones(1), x))

    y = np.dot(x, theta)
    print('Predicted sales:', y)

x, y = load('Chapter4_Overfitting/Practice2_Chapter2.csv')
# Dynamically determine the number of rows in y
y = np.reshape(y, (y.shape[0], 1))
# Add a column of ones to x for the theta0 term
x = np.hstack((np.ones((x.shape[0], 1)), x))
# Initialize theta and another params
theta = np.zeros((x.shape[1], 1))
alpha = 0.001
iterations = 10000
#lambda_values = [0, 0.01, 0.1, 1, 10, 100]
#best_lambda = test_lambda_values(x, y, alpha, iterations, lambda_values)
lambda_ = 0.01

theta, cost_history = gradient_descent(x, y, theta, alpha, lambda_, iterations)
C = cost_function(x, y, theta, lambda_)
print("Cost: ", C)
for i in range(len(theta)):
    print(f"Theta{i}: ", theta[i])

# Check for optimal alpha and iterations
iter = list(range(iterations))
plot_cost(cost_history, iter, alpha)

# Test prediction
test_set = [
    [230.1, 37.8, 69.2],
    [44.5, 39.3, 45.1],
    [17.2, 39.3, 45.1],
    [151.5, 41.3, 58.5],
    [180.8, 10.8, 58.4],]

for i, test_features in enumerate(test_set):
    print(f"\nTest case {i+1}:")
    predict(test_features, theta)

