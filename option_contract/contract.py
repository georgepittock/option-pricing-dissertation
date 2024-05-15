class OptionContract:
    def __init__(
        self,
        initial_stock_price,
        strike_price,
        maturity_time,
        risk_free_rate,
        volatility,
        option_type,
        start_date,
        spot_on_maturity,
    ):
        self.initial_stock_price = initial_stock_price
        self.strike_price = strike_price
        self.maturity_time = maturity_time
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.option_type = option_type
        self.start_date = start_date
        self.spot_on_maturity = spot_on_maturity

    def payoff(self) -> float:
        if self.option_type == "call":
            return max(0, self.spot_on_maturity - self.strike_price)
        else:  # put
            return max(0, self.strike_price - self.spot_on_maturity)
