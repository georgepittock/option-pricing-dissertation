import itertools

import matplotlib.pyplot as plt
import numpy as np
from pricers.black_scholes.black_scholes import BlackScholes
from pricers.create_options import create_options
from pricers.market_data import historical_data_df
from pricers.monte_carlo.monte_carlo import MonteCarlo
from pricers.nn.nn_pricer import NeuralNetwork

_option_types = ("call", "put")


def calculate_differences(pricer, options):
    prices = pricer.price()
    return prices - np.array([option.payoff() for option in options])


def plot_differences(all_differences, classes):
    fig, axs = plt.subplots(2, sharex=True, sharey=True)
    colors = ["blue", "green", "red"]

    for i, option_type in enumerate(_option_types):
        ax = axs[i]
        ax.set_xlabel("Difference")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{option_type.capitalize()} Option")
        for j, clazz in enumerate(classes):
            diffs = all_differences[clazz.__name__][option_type]
            ax.hist(
                diffs,
                bins=30,
                color=colors[j],
                alpha=0.5,
                label=clazz.__name__,
            )
        ax.legend()

    fig.subplots_adjust(hspace=0.5)
    plt.show()


def _mae(a):
    return np.mean(np.abs(a))


def _rmse(a):
    return np.sqrt(np.mean(np.square(a)))


if __name__ == "__main__":
    n = 1000
    all_options = list(create_options(n, historical_data_df))

    class_batch_size = {
        NeuralNetwork: n,
        BlackScholes: n,
        MonteCarlo: 1,
    }

    differences = {
        clazz.__name__: {opt_type: [] for opt_type in _option_types}
        for clazz in class_batch_size
    }

    for pricer_class, batch_size in class_batch_size.items():
        class_diffs = differences[pricer_class.__name__]
        for batch in itertools.batched(all_options, batch_size):
            diff = calculate_differences(pricer_class(batch), batch)
            for idx, d in enumerate(diff):
                class_diffs[batch[idx].option_type].append(d)

        print(f"Pricer: {pricer_class.__name__}")

        metrics = {
            option_type: [_mae, _rmse] for option_type in _option_types + ("<ALL>",)
        }

        for option_type, metric_fns in metrics.items():
            for metric_fn in metric_fns:
                if option_type == "<ALL>":
                    data = [
                        item
                        for sublist in [
                            value for k, value in class_diffs.items() if k != "<ALL>"
                        ]
                        for item in sublist
                    ]
                else:
                    data = class_diffs[option_type]
                name = metric_fn.__name__.strip("_").upper()
                print(
                    f"{name} {option_type if option_type != '<ALL>' else 'all'}: {metric_fn(data)}"
                )
        print("-" * 50)

    plot_differences(differences, list(class_batch_size.keys()))
