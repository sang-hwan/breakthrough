# trading/trade_executor.py
from logs.logger_config import setup_logger
import pandas as pd
from trading.calculators import (
    calculate_atr,
    calculate_dynamic_stop_and_take,
    adjust_trailing_stop,
    calculate_partial_exit_targets
)

logger = setup_logger(__name__)

class TradeExecutor:
    @staticmethod
    def compute_atr(data: pd.DataFrame, period: int = 14):
        return calculate_atr(data, period)

    @staticmethod
    def calculate_dynamic_stop_and_take(entry_price: float, atr: float, risk_params: dict):
        return calculate_dynamic_stop_and_take(entry_price, atr, risk_params)

    @staticmethod
    def adjust_trailing_stop(current_stop: float, current_price: float, highest_price: float, trailing_percentage: float):
        return adjust_trailing_stop(current_stop, current_price, highest_price, trailing_percentage)

    @staticmethod
    def calculate_partial_exit_targets(entry_price: float, partial_exit_ratio: float = 0.5,
                                       partial_profit_ratio: float = 0.03, final_profit_ratio: float = 0.06):
        return calculate_partial_exit_targets(entry_price, partial_exit_ratio, partial_profit_ratio, final_profit_ratio)
