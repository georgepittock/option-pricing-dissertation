import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return ((x * np.cos(x / 5) + 2 * x) ** 2) / 75


def f_prime(x):
    return (
        2
        * (x * np.cos(x / 5) + 2 * x)
        * (np.cos(x / 5) - x * np.sin(x / 5) / 5 + 2)
        / 75
    )


def plot_update_process(x, learning_rate, ax):
    x_vals = [x]
    y_vals = [f(x)]

    while abs(f_prime(x)) > 1e-9:
        x = x - learning_rate * f_prime(x)
        x_vals.append(x)
        y_vals.append(f(x))

    ax.scatter(x_vals[0], y_vals[0], color="yellow")
    ax.scatter(x_vals[1:-1], y_vals[1:-1], color="red")
    ax.scatter(x_vals[-1], y_vals[-1], color="green")
    for i in range(len(x_vals)):
        if i:
            ax.plot(
                [x_vals[i - 1], x_vals[i]],
                [y_vals[i - 1], y_vals[i]],
                color="green",
                linestyle="--",
            )

    x_plot = np.linspace(-15, 21, 10000)
    y_plot = f(x_plot)
    ax.plot(x_plot, y_plot, color="blue")
    ax.set_title(
        f"Learning Rate: {lr}, iterations: {len(x_vals)}, x: {x_vals[-1]:.2f}, y: {y_vals[-1]:.2f}"
    )


x_0 = 20
learning_rates = [0.1, 3, 5, 10]
fig, axes = plt.subplots(len(learning_rates), 1, figsize=(8, 6 * len(learning_rates)))

for lr, ax in zip(learning_rates, axes):
    plot_update_process(x_0, lr, ax)

fig.subplots_adjust(hspace=1)
plt.show()
