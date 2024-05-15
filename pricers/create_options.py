import functools
import multiprocessing
import random

import numpy as np
from option_contract.contract import OptionContract
from pricers.market_data import bank_rate

_CONTRACT_LENGTHS = [7, 14, 28, 30, 31, 56, 60, 90, 120, 180, 365, 365 * 2, 365 * 3]


def _create_option(df):
    df = df.dropna()
    while True:
        contract_start_date = df["Date"].sample(1).values[0]
        contract_start_date_ftse_data = df[df["Date"] == contract_start_date]
        if (
            contract_start_date + np.timedelta64(np.min(_CONTRACT_LENGTHS), "D")
            > df["Date"].max()
        ):
            continue

        contract_length_days = random.choice(_CONTRACT_LENGTHS)
        expiration_date = contract_start_date + np.timedelta64(
            contract_length_days, "D"
        )
        if expiration_date not in df["Date"].values:
            continue

        contract_length_years = contract_length_days / 365
        price_on_start_date = contract_start_date_ftse_data["Price"].values[0]
        expiration_price = df[df["Date"] == expiration_date]["Price"].values[0]

        delta = random.randint(1, 40) / 100
        option_type = random.choice(["call", "put"])
        strike_price = price_on_start_date * (
            1 - delta if option_type == "call" else 1 + delta
        )
        volatility = (
            contract_start_date_ftse_data["10 Day Rolling Volatility"].values[0] ** 2
        )

        risk_free_rate = bank_rate(start_date=contract_start_date)

        return OptionContract(
            price_on_start_date,
            strike_price,
            contract_length_years,
            risk_free_rate,
            volatility,
            option_type,
            contract_start_date,
            expiration_price,
        )


def _multiprocess_create_option(_, df):
    return _create_option(df)


def create_options(n, df):
    create_option_partial = functools.partial(_multiprocess_create_option, df=df)
    with multiprocessing.Pool() as pool:
        return pool.map(create_option_partial, range(n))
