import functools
import os.path
import random

import numpy as np
import pandas as pd

_ROLLING_WINDOW_SIZE = 5
_root_dir = os.path.split(os.path.dirname(__file__))[0]
_external_data_path = os.path.join(_root_dir, "external")


def _load_bank_rate():
    return pd.read_excel(
        os.path.join(
            _external_data_path,
            "Bank Rate history and data  Bank of England Database.xlsx",
        )
    )


def _load_historical_data():
    df = pd.read_excel(
        os.path.join(
            _external_data_path,
            "FTSE 100 Historical Price Data With Heston Params.xlsx",
        )
    ).sort_values(by="Date", ascending=True)

    df["Log Returns"] = np.log(df["Price"] / df["Price"].shift(1))

    def _rolling_volatility(days):
        return df["Log Returns"].rolling(window=days).std() * np.sqrt(days)

    df["5 Day Rolling Volatility"] = _rolling_volatility(5)
    df["10 Day Rolling Volatility"] = _rolling_volatility(10)
    df["21 Day Rolling Volatility"] = _rolling_volatility(21)
    df["90 Day Rolling Volatility"] = _rolling_volatility(90)
    return df


def _bank_rate(start_date, df):
    best_date = df[df["Date Changed"] <= start_date]["Date Changed"].max()
    return df[df["Date Changed"] == best_date]["Rate"].values[0] / 100


bank_rate_df = _load_bank_rate()
bank_rate = functools.partial(_bank_rate, df=bank_rate_df)
historical_data_df = _load_historical_data()
historical_data_df = historical_data_df[
    historical_data_df["21 Day Rolling Volatility"]
    < historical_data_df["21 Day Rolling Volatility"].quantile(0.5)
]
