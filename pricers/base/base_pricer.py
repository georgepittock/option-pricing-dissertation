from abc import ABC, abstractmethod

import numpy as np


class BasePricer(ABC):
    def __init__(self, options):
        self.initial_stock_prices = np.array([])
        self.strike_prices = np.array([])
        self.maturity_times = np.array([])
        self.risk_free_rates = np.array([])
        self.volatilities = np.array([])
        self.option_types = np.array([])
        self.option_start_dates = np.array([], dtype="datetime64")

        for option in options:
            self.initial_stock_prices = np.append(
                self.initial_stock_prices, option.initial_stock_price
            )
            self.strike_prices = np.append(self.strike_prices, option.strike_price)
            self.maturity_times = np.append(self.maturity_times, option.maturity_time)
            self.risk_free_rates = np.append(
                self.risk_free_rates, option.risk_free_rate
            )
            self.volatilities = np.append(self.volatilities, option.volatility)
            self.option_types = np.append(self.option_types, option.option_type)
            self.option_start_dates = np.append(
                self.option_start_dates, option.start_date
            )

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
