import numpy as np
from pricers.base.base_pricer import BasePricer
from scipy.stats import norm


class BlackScholes(BasePricer):
    @property
    def d1(self):
        return (
            np.log(self.initial_stock_prices / self.strike_prices)
            + (self.risk_free_rates + 0.5 * self.volatilities**2)
            * self.maturity_times
        ) / (self.volatilities * np.sqrt(self.maturity_times))

    @property
    def d2(self):
        return self.d1 - self.volatilities * np.sqrt(self.maturity_times)

    def price(self):
        return (self.option_types == "call") * self.call_price() + (
            self.option_types == "put"
        ) * self.put_price()

    def call_price(self):
        return self.initial_stock_prices * norm.cdf(
            self.d1
        ) - self.strike_prices * np.exp(
            -self.risk_free_rates * self.maturity_times
        ) * norm.cdf(
            self.d2
        )

    def put_price(self):
        return self.strike_prices * np.exp(
            -self.risk_free_rates * self.maturity_times
        ) * norm.cdf(-self.d2) - self.initial_stock_prices * norm.cdf(-self.d1)
