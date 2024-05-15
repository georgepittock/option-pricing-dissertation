from typing import List

import matplotlib.pyplot as plt
import numpy as np


def call_payoff(price, strike_price):
    return np.maximum(price - strike_price, 0)


def put_payoff(price, strike_price):
    return np.maximum(strike_price - price, 0)


price_at_maturity = np.arange(50, 151)

# Calculate payoffs for call and put all_options for each strike price
call_payoffs: np.ndarray = call_payoff(price_at_maturity, 100)
put_payoffs: np.ndarray = put_payoff(price_at_maturity, 100)

# Create subplots
fig, axs = plt.subplots()

axs.plot(price_at_maturity, call_payoffs, color="blue")

axs.set_xlabel("Price at S(T)")
axs.set_xticks([100], labels=["K"])
axs.set_xticklabels(["K"], color="black")

axs.set_yticks([0])
axs.set_ylabel("Payoff")
axs.set_title("European Call Option Payoff")

plt.show()

fig, axs = plt.subplots()

axs.plot(price_at_maturity, put_payoffs, color="red")
axs.set_xlabel("Price at initial_stock_price(maturity_time)")
axs.set_xticks([100], labels=["K"])
axs.set_xticklabels(["K"], color="black")

axs.set_yticks([0])
axs.set_ylabel("Payoff")

axs.set_title("European Put Option Payoff")

plt.show()
