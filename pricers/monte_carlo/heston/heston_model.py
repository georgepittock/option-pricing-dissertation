import numpy as np
import pandas as pd
from pricers.market_data import historical_data_df


class Heston:
    def __init__(self, start_date):
        self.start_date = start_date

    def filtered_df(self):
        ninety_days_ago = self.start_date - pd.DateOffset(days=90)
        return historical_data_df[
            (historical_data_df["Date"] >= ninety_days_ago)
            & (historical_data_df["Date"] <= self.start_date)
        ]

    def theta(self):
        return np.mean(self.filtered_df()["90 Day Rolling Volatility"].values) ** 2

    def xi(self):
        return self.filtered_df()["90 Day Rolling Volatility"].std()

    def rho(self):
        return self.filtered_df()["rho"].values[-1]

    def kappa(self):
        return self.filtered_df()["kappa"].values[-1]
