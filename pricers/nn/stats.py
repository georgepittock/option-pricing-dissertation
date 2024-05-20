import matplotlib.pyplot as plt
import numpy as np
from pricers.create_options import create_options
from pricers.market_data import historical_data_df
from pricers.nn.nn_pricer import NeuralNetwork


def calculate_errors(prices, payoffs):
    mae = np.mean(np.abs(prices - payoffs))
    rmse = np.sqrt(np.mean((prices - payoffs) ** 2))
    return mae, rmse


def plot_data(options):
    fig, ax = plt.subplots()
    _max_val = max(
        np.max([option.payoff() for option in options]),
        np.max(NeuralNetwork(options).price()),
    )

    for option_type, color in [("put", "red"), ("call", "blue")]:
        options_type = [
            option for option in options if option.option_type == option_type
        ]
        prices = NeuralNetwork(options_type).price()
        payoffs = np.array([option.payoff() for option in options_type])
        ax.scatter(payoffs, prices, color=color, label=option_type)

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    ax.plot([0, _max_val], [0, _max_val], color="black")
    ax.legend()
    plt.show()


def run(n):
    all_options = list(create_options(n, historical_data_df))

    put_options = []
    call_options = []
    put_payoffs = []
    call_payoffs = []
    for option in all_options:
        print(option.option_type)
        if option.option_type == "put":
            put_options.append(option)
            put_payoffs.append(option.payoff())
        if option.option_type == "call":
            call_options.append(option)
            call_payoffs.append(option.payoff())
    mae, rmse = calculate_errors(
        np.concatenate(
            [NeuralNetwork(put_options).price(), NeuralNetwork(call_options).price()]
        ),
        np.concatenate([put_payoffs, call_payoffs]),
    )
    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")
    plot_data(all_options)


if __name__ == "__main__":
    run(500)
