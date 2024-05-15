import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import axes3d

# Set the style for matplotlib
plt.style.use("_mpl-gallery")

# Define the data
data_X = [0.5, 2.5, 0.6]
data_Y = [0.2, 0.9, 0.8]


def sigmoid(x, weight, bias):
    """Calculate the sigmoid function."""
    return 1.0 / (1.0 + np.exp(-(x * weight + bias)))


def calculate_error(weight, bias):
    """Calculate the mean squared error."""
    error = 0
    for x, y in zip(data_X, data_Y):
        error += 0.5 * (sigmoid(x, weight, bias) - y) ** 2
    return error


def calculate_gradient_weight(x, y, weight, bias):
    """Calculate the gradient of the error function with respect to the weight."""
    return (
        (sigmoid(x, weight, bias) - y)
        * sigmoid(x, weight, bias)
        * (1 - sigmoid(x, weight, bias))
        * x
    )


def calculate_gradient_bias(x, y, weight, bias):
    """Calculate the gradient of the error function with respect to the bias."""
    return (
        (sigmoid(x, weight, bias) - y)
        * sigmoid(x, weight, bias)
        * (1 - sigmoid(x, weight, bias))
    )


def perform_gradient_descent():
    """Perform the gradient descent optimization algorithm."""
    weight, bias, learning_rate, max_iterations = -1, 1, 0.1, 10000
    weight_values = []
    bias_values = []
    weight_values.append(weight)
    bias_values.append(bias)
    for _ in range(max_iterations):
        gradient_weight, gradient_bias = 0, 0
        for x, y in zip(data_X, data_Y):
            gradient_weight += calculate_gradient_weight(x, y, weight, bias)
            gradient_bias += calculate_gradient_bias(x, y, weight, bias)
        weight -= learning_rate * gradient_weight
        bias -= learning_rate * gradient_bias
        weight_values.append(weight)
        bias_values.append(bias)
    return weight_values, bias_values


weight_range = bias_range = np.arange(-20.0, 20.0, 0.5)
weight_range, bias_range = np.meshgrid(weight_range, bias_range)

error_values = []
for i in range(len(weight_range[0])):
    error_row = []
    for j in range(len(weight_range)):
        error_row.append(calculate_error(weight_range[i][j], bias_range[i][j]))
    error_values.append(error_row)

error_values = np.array(error_values)

weight_values, bias_values = perform_gradient_descent()
error_values_gradient_descent = [
    calculate_error(weight_values[i], bias_values[i]) for i in range(len(weight_values))
]

# Create the 3D plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surface = ax.plot_surface(
    weight_range,
    bias_range,
    error_values,
    antialiased=False,
    linewidth=0,
    alpha=0.9,
    color="blue",
)

ax.set_xlabel("Weight")
ax.set_ylabel("Bias")
ax.set_zlabel("Error")
scatter = ax.scatter(weight_values, bias_values, error_values_gradient_descent, c="r")

plt.show()
