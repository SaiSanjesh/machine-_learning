import numpy as np

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function (Binary Cross-Entropy)
def compute_cost(X, y, weights):
    m = len(y)
    predictions = sigmoid(np.dot(X, weights))
    cost = -1/m * (np.dot(y, np.log(predictions)) + np.dot((1 - y), np.log(1 - predictions)))
    return cost

# Gradient Descent function
def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        predictions = sigmoid(np.dot(X, weights))
        gradient = 1/m * np.dot(X.T, (predictions - y))
        weights -= learning_rate * gradient
        
        # Save the cost at each iteration for analysis
        cost_history.append(compute_cost(X, y, weights))

    return weights, cost_history

# Logistic Regression model
def logistic_regression(X, y, learning_rate=0.01, iterations=1000):
    # Add intercept column (1s) to the feature matrix
    X = np.c_[np.ones((X.shape[0], 1)), X]
    
    # Initialize weights (including intercept)
    weights = np.zeros(X.shape[1])
    
    # Train the model using Gradient Descent
    weights, cost_history = gradient_descent(X, y, weights, learning_rate, iterations)
    
    return weights, cost_history

# Prediction function
def predict(X, weights):
    X = np.c_[np.ones((X.shape[0], 1)), X]  # Add intercept term
    predictions = sigmoid(np.dot(X, weights))
    return [1 if i >= 0.5 else 0 for i in predictions]

# Example Usage:
# Assuming X (features) and y (target) are numpy arrays
# X = np.array([[feature1, feature2], ...])
# y = np.array([0, 1, 0, 1, ...])

# Initialize some example data for X and y
X = np.array([[2.5, 3.5], [1.5, 2.5], [3.0, 4.0], [4.5, 5.5], [3.5, 4.5]])
y = np.array([0, 0, 1, 1, 1])

# Train the logistic regression model
weights, cost_history = logistic_regression(X, y, learning_rate=0.1, iterations=1000)

# Display the learned weights and cost over iterations
print("Learned Weights:", weights)
print("Final Cost:", cost_history[-1])

# Make predictions on new data
X_new = np.array([[3.0, 4.0], [2.0, 3.0]])
predictions = predict(X_new, weights)
print("Predictions:", predictions)
