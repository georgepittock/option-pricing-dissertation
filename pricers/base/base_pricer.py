from abc import ABC, abstractmethod

import numpy as np


class BasePricer(ABC):
    def __init__(self, options):
        # TODO make this one for loop
        self.initial_stock_prices = np.array(
            [option.initial_stock_price for option in options]
        )
        self.strike_prices = np.array([option.strike_price for option in options])
        self.maturity_times = np.array([option.maturity_time for option in options])
        self.risk_free_rates = np.array([option.risk_free_rate for option in options])
        self.volatilities = np.array([option.volatility for option in options])
        self.option_types = np.array([option.option_type for option in options])
        self.option_start_dates = np.array([option.start_date for option in options])

        self.__options = options

    @property
    def volatilities(self):
        return self._volatilities

    @volatilities.setter
    def volatilities(self, value):
        self._volatilities = value

    def payoff(self, prices):
        payoff = (self.option_types == "call") * (prices - self.strike_prices) + (
            self.option_types == "put"
        ) * (self.strike_prices - prices)
        return np.maximum(0, payoff)

    def __call__(self, *args, **kwargs):
        return self.price()

    @abstractmethod
    def price(self):
        return NotImplementedError("Must implement price method")
