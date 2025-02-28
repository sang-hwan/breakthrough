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
    def compute_atr(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        try:
            result = calculate_atr(data, period)
            logger.debug("ATR computed successfully.")
            return result
        except Exception as e:
            logger.error("Error in compute_atr: {}".format(e), exc_info=True)
            raise

    @staticmethod
    def calculate_dynamic_stop_and_take(entry_price: float, atr: float, risk_params: dict):
        try:
            stop_loss, take_profit = calculate_dynamic_stop_and_take(entry_price, atr, risk_params)
            logger.debug(f"Dynamic stop/take computed: stop_loss={stop_loss:.2f}, take_profit={take_profit:.2f} "
                        f"(entry_price={entry_price}, atr={atr})")
            return stop_loss, take_profit
        except Exception as e:
            logger.error("Error in calculate_dynamic_stop_and_take: {}".format(e), exc_info=True)
            raise

    @staticmethod
    def adjust_trailing_stop(current_stop: float, current_price: float, highest_price: float, trailing_percentage: float):
        try:
            new_stop = adjust_trailing_stop(current_stop, current_price, highest_price, trailing_percentage)
            # 핵심 거래 정보: 조정된 trailing stop info 레벨로 출력
            logger.debug(f"Trailing stop adjusted to {new_stop:.2f} (current_price={current_price}, highest_price={highest_price})")
            return new_stop
        except Exception as e:
            logger.error("Error in adjust_trailing_stop: {}".format(e), exc_info=True)
            raise

    @staticmethod
    def calculate_partial_exit_targets(entry_price: float, partial_exit_ratio: float = 0.5,
                                       partial_profit_ratio: float = 0.03, final_profit_ratio: float = 0.06):
        try:
            targets = calculate_partial_exit_targets(entry_price, partial_exit_ratio, partial_profit_ratio, final_profit_ratio)
            # 핵심 거래 정보: 부분 청산 목표 info 레벨로 출력
            logger.debug(f"Partial exit targets computed: {targets} (entry_price={entry_price})")
            return targets
        except Exception as e:
            logger.error("Error in calculate_partial_exit_targets: {}".format(e), exc_info=True)
            raise
