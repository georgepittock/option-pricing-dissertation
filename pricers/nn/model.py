import torch.nn as nn
from pricers.create_options import create_options
from pricers.market_data import historical_data_df
from pricers.nn.feature_extractor import extract_features

_hidden_size = (256, 128, 64)


class OptionPricingNN(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(OptionPricingNN, self).__init__()

        layers = []

        previous_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.ReLU())
            previous_size = hidden_size

        layers.append(nn.Linear(hidden_sizes[-1], 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def create_model():
    input_size = len(extract_features(create_options(1, historical_data_df)[0]))
    return OptionPricingNN(input_size, _hidden_size)


if __name__ == "__main__":
    print(create_model())
