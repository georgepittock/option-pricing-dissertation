import multiprocessing
import os
import types

import torch
from pricers.base.base_pricer import BasePricer
from pricers.nn.feature_extractor import extract_features
from pricers.nn.model import create_model


class NeuralNetwork(BasePricer):
    def __init__(self, options):
        if isinstance(options, types.GeneratorType):
            options = list(options)
        super().__init__(options)
        with multiprocessing.Pool() as pool:
            self.features = pool.map(extract_features, options)

        self.put_model = create_model()
        self.call_model = create_model()

        put_model_path = os.path.join(os.path.dirname(__file__), "put_model.pth")
        call_model_path = os.path.join(os.path.dirname(__file__), "call_model.pth")

        self.put_model.load_state_dict(torch.load(put_model_path))
        self.put_model.eval()

        self.call_model.load_state_dict(torch.load(call_model_path))
        self.call_model.eval()

    def price(self):
        return (self.option_types == "call") * self.call_price() + (
            self.option_types == "put"
        ) * self.put_price()

    def put_price(self):
        with torch.no_grad():
            return [
                self.put_model(feature).item() * strike
                for feature, strike in zip(self.features, self.strike_prices)
            ]

    def call_price(self):
        with torch.no_grad():
            return [
                self.call_model(feature).item() * strike
                for feature, strike in zip(self.features, self.strike_prices)
            ]
