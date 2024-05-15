import functools
import pathlib

import numpy as np
import pandas as pd
import scipy.optimize
from pricers.market_data import historical_data_df

_external_data_path = pathlib.Path(__file__).parents[3] / "external"


def heston_model(params, time, theta, v0, xi):
    kappa, _ = params
    v = np.full(time, v0)
    wiener = np.random.standard_normal(time)
    for t in range(1, time):
        v_ = v[t - 1] + kappa * (theta - v[t - 1]) + xi * np.sqrt(v[t - 1]) * wiener[t]
        v[t] = max(v_, 1e-6)
    return v


def get_historical_data(date, window_size=90):
    end_date = date
    start_date = end_date - pd.DateOffset(days=window_size)
    return historical_data_df[
        (historical_data_df["Date"] >= start_date)
        & (historical_data_df["Date"] <= end_date)
    ]


def calibrate_heston_model(date):
    historical_data = get_historical_data(date)
    v0 = historical_data["5 Day Rolling Volatility"].values[-1] ** 2
    theta = np.mean(historical_data["90 Day Rolling Volatility"]) ** 2
    xi = historical_data["90 Day Rolling Volatility"].std()

    future_vol = get_historical_data(date + pd.DateOffset(days=90))[
        "5 Day Rolling Volatility"
    ]
    hest = functools.partial(
        heston_model, time=len(future_vol), theta=theta, v0=v0, xi=xi
    )

    params = scipy.optimize.minimize(
        lambda p: np.mean((hest(p) - future_vol) ** 2), [2, 0], method="Nelder-Mead"
    )
    return params.x


if __name__ == "__main__":
    calibrated_params_columns = ["kappa", "rho"]

    for idx, row in historical_data_df.iterrows():
        if get_historical_data(row["Date"]).isna().values.any():
            continue
        print(f"Row Index: {idx}")

        calibrated_params = calibrate_heston_model(row["Date"])
        historical_data_df.loc[idx, calibrated_params_columns] = calibrated_params

    output_filename = "FTSE 100 Historical Price Data With Heston Params.xlsx"
    historical_data_df.to_excel(_external_data_path / output_filename, index=False)
    print(f"Calibrated Parameters saved to {output_filename}")
