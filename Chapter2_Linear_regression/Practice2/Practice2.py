<<<<<<< HEAD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# variables to store means and deviation for each feature
mu = []
std = []

def load(filename):
    df = pd.read_csv(filename)
    data = df.to_numpy(dtype=float)

    # Normalize the feature columns
    normalize(data)

    # Extract features and target
    x = data[:, :-1]  # All columns except the last one
    y = data[:, -1]   # Only the last column ('Sales')
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

def cost_function(x, y, theta):
	  return (1/(2*y.shape[0])) * np.sum((h(x, theta) - y)**2)

def gradient_descent(x, y, theta, alpha, iterations):
    m = x.shape[0]
    cost_history = []

    for i in range(iterations):
        gradient = (1/m) * (x.T @ (h(x, theta) - y))
        theta = theta - alpha * gradient
        cost = cost_function(x, y, theta)
        cost_history.append(cost)
    return theta, cost_history

def plot_cost(cost_history, iterations):
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.plot(iterations, cost_history, 'r', linewidth = "5")
    plt.show()

def predict(x, theta):
    x = np.array(x)
    # Print to verify the normalization step
    print(f"Original input: {x}")
    
    # Normalize the input using the same mu and std from the training data
    x = (x - mu) / std
    print(f"Normalized input: {x}")
    
    # Add intercept term
    x = np.hstack((np.ones(1), x))
    
    # Compute the predicted value
    y = np.dot(x, theta)
    print('Predicted sales:', y)

x, y = load('Practice2_Chapter2.csv')
# Dynamically determine the number of rows in y
y = np.reshape(y, (y.shape[0], 1))
# Add a column of ones to x for the theta0 term
x = np.hstack((np.ones((x.shape[0], 1)), x))
# Initialize theta and another params
theta = np.zeros((x.shape[1], 1))

alpha = 0.0001
iterations = 100000

theta, cost_history = gradient_descent(x, y, theta, alpha, iterations)
C = cost_function(x, y, theta)
print("Cost: ", C)
print("Theta: ", theta)

# Check for optimal alpha and iterations
iter = list(range(iterations))
plot_cost(cost_history, iter)

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
=======
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# variables to store means and deviation for each feature
mu = []
std = []

def load(filename):
    df = pd.read_csv(filename)
    data = df.to_numpy(dtype=float)

    # Normalize the feature columns
    normalize(data)

    # Extract features and target
    x = data[:, :-1]  # All columns except the last one
    y = data[:, -1]   # Only the last column ('Sales')
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

def cost_function(x, y, theta):
	  return (1/(2*y.shape[0])) * np.sum((h(x, theta) - y)**2)

def gradient_descent(x, y, theta, alpha, iterations):
    m = x.shape[0]
    cost_history = []

    for i in range(iterations):
        gradient = (1/m) * (x.T @ (h(x, theta) - y))
        theta = theta - alpha * gradient
        cost = cost_function(x, y, theta)
        cost_history.append(cost)
    return theta, cost_history

def plot_cost(cost_history, iterations):
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.plot(iterations, cost_history, 'r', linewidth = "5")
    plt.show()

def predict(x, theta):
    x = np.array(x)
    # Print to verify the normalization step
    print(f"Original input: {x}")
    
    # Normalize the input using the same mu and std from the training data
    x = (x - mu) / std
    print(f"Normalized input: {x}")
    
    # Add intercept term
    x = np.hstack((np.ones(1), x))
    
    # Compute the predicted value
    y = np.dot(x, theta)
    print('Predicted sales:', y)

x, y = load('Practice2_Chapter2.csv')
# Dynamically determine the number of rows in y
y = np.reshape(y, (y.shape[0], 1))
# Add a column of ones to x for the theta0 term
x = np.hstack((np.ones((x.shape[0], 1)), x))
# Initialize theta and another params
theta = np.zeros((x.shape[1], 1))

alpha = 0.0001
iterations = 100000

theta, cost_history = gradient_descent(x, y, theta, alpha, iterations)
C = cost_function(x, y, theta)
print("Cost: ", C)
print("Theta: ", theta)

# Check for optimal alpha and iterations
iter = list(range(iterations))
plot_cost(cost_history, iter)

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
>>>>>>> 3be0b58 (Add Chapter1, Chapter2, and Chapter3)
