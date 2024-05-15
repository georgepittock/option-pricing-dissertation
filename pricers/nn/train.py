import multiprocessing
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pricers.create_options import create_options
from pricers.market_data import historical_data_df
from pricers.nn.feature_extractor import extract_features
from pricers.nn.model import create_model
from torch.utils.data import DataLoader, TensorDataset


def process_option(option):
    """
    returns features and moneyness of the option
    """
    return extract_features(option), option.payoff() / option.strike_price


def create_options_with_features(n, df, option_type):
    """
    multiprocessing method to extract payoffs and features
    """
    options = [opt for opt in create_options(n, df) if opt.option_type == option_type]
    with multiprocessing.Pool() as pool:
        return pool.map(process_option, options)


def _train(option_type):
    model = create_model()
    features, payoffs = zip(
        *create_options_with_features(10000, historical_data_df, option_type)
    )

    dataloader = DataLoader(
        TensorDataset(
            torch.stack(features), torch.tensor(payoffs, dtype=torch.float32)
        ),
        batch_size=64,
        shuffle=True,
    )

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    for epoch in range(num_epochs):
        for batch_idx, (features, payoffs) in enumerate(dataloader):
            optimizer.zero_grad()
            loss = loss_fn(model(features), payoffs.unsqueeze(1))
            loss.backward()
            optimizer.step()
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}"
            )

    torch.save(
        model.state_dict(),
        os.path.join(os.path.dirname(__file__), f"{option_type}_model.pth"),
    )


if __name__ == "__main__":
    for opt_type in ("put", "call"):
        _train(opt_type)
