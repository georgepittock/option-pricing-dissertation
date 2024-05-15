import numpy as np
from pricers.base.base_pricer import BasePricer
from pricers.monte_carlo.heston.heston_model import Heston


class MonteCarlo(BasePricer):
    NUM_INTERVALS = 2500
    NUM_SIMULATIONS = 2500

    def __init__(self, options):
        super().__init__(options)
        self._init_volatilities = self._volatilities
        hestons = [Heston(option.start_date) for option in options]
        self.rhos = [h.rho() for h in hestons]
        self.kappas = [h.kappa() for h in hestons]
        self.thetas = [h.theta() for h in hestons]
        self.xis = [h.xi() for h in hestons]

        self._wiener_processes = [
            np.random.multivariate_normal(
                [0, 0],
                np.array([[1, rho], [rho, 1]]),
                (self.NUM_SIMULATIONS, self.NUM_INTERVALS),
            )
            for rho in self.rhos
        ]

    def _variances(self):
        dt = self.maturity_times / self.NUM_INTERVALS
        variances = np.zeros((self.NUM_SIMULATIONS, self.NUM_INTERVALS))
        for wiener_process in self._wiener_processes:
            variance = self._init_volatilities[0] ** 2 * np.ones(
                (self.NUM_SIMULATIONS,)
            )
            for t in range(self.NUM_INTERVALS):
                variance = np.abs(
                    variance
                    + self.kappas * (self.thetas - variance) * dt
                    + self.xis * np.sqrt(variance) * wiener_process[:, t, 0]
                )
                variances[:, t] = variance
        return variances

    @property
    def volatilities(self):
        return np.sqrt(self._variances())

    @volatilities.setter
    def volatilities(self, value):
        self._volatilities = value

    def price(self):
        dt = self.maturity_times / self.NUM_INTERVALS
        prices = np.zeros((len(self._wiener_processes), self.NUM_SIMULATIONS))
        for idx, wiener_process in enumerate(self._wiener_processes):
            wiener_process = wiener_process[:, :, 1] * np.sqrt(dt)
            prices[idx] = (
                self.initial_stock_prices
                * np.exp(
                    np.cumsum(
                        (self.risk_free_rates - 0.5 * self.volatilities[idx] ** 2) * dt
                        + self.volatilities[idx] * wiener_process,
                        axis=1,
                    )
                )[:, -1]
            )
        payoffs = self.payoff(prices)
        return np.exp(-self.risk_free_rates * self.maturity_times) * np.mean(payoffs)
