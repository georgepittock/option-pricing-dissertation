import datetime
from calendar import monthrange

import numpy as np
import torch
from pricers.market_data import historical_data_df


def extract_features(option):
    option_date_data = historical_data_df[
        historical_data_df["Date"] == option.start_date
    ]
    twenty_one_day_volatility = option_date_data["21 Day Rolling Volatility"].values[0]
    ninety_day_volatility = option_date_data["90 Day Rolling Volatility"].values[0]

    # Convert start date to python datetime as Numpy datetime64 does not support
    # extracting day, month, year, etc.
    # start_data_py_dt = datetime.datetime.strptime(
    #     np.datetime_as_string(option.start_date, unit="s"), "%Y-%m-%dT%H:%M:%S"
    # )
    # day_decimal = (
    #     start_data_py_dt.day
    #     / monthrange(start_data_py_dt.year, start_data_py_dt.month)[1]
    # )
    option_data = [
        option.initial_stock_price / option.strike_price,
        option.maturity_time,
        option.risk_free_rate,
        # int(option.option_type == "call"),
        # start_data_py_dt.month / 12,
        # day_decimal,
        # (start_data_py_dt.year - 2010) / 13,
        option.volatility,
        option.volatility * np.sqrt(option.maturity_time),
        twenty_one_day_volatility,
        twenty_one_day_volatility * np.sqrt(option.maturity_time),
        ninety_day_volatility,
        ninety_day_volatility * np.sqrt(option.maturity_time),
    ]
    return torch.tensor(option_data).float()
