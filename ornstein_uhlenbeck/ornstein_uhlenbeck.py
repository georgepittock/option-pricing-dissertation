import matplotlib.pyplot as plt
import numpy as np

# generate plot for Ornstein-Uhlenbeck process
kappa = 2.0
theta = 0.01
rho = 0.0
xi = 0.1
dt = 0.01
T = 200
num_steps = int(T / dt)

fig, ax = plt.subplots()

time = np.linspace(0.0, T, num_steps)
for _ in range(5):
    sigma_squared = np.zeros(num_steps)
    dB = np.sqrt(dt) * np.random.normal(size=num_steps)
    dW = rho * dB + np.sqrt(1 - rho**2) * np.sqrt(dt) * np.random.normal(
        size=num_steps
    )
    sigma_squared[0] = 0.01
    for i in range(1, num_steps):
        sigma_squared[i] = (
            sigma_squared[i - 1]
            + kappa * (theta - sigma_squared[i - 1]) * dt
            + xi * np.sqrt(sigma_squared[i - 1]) * dW[i]
        )
    ax.plot(time, sigma_squared)

ax.axhline(theta, color="r", linestyle="--", label="Mean Long-Run Variance")
ax.set_xlabel("Time")
ax.set_ylabel("Variance")
ax.set_title("Volatility Simulation via Heston Model")
ax.grid(True)
ax.legend()
plt.show()
