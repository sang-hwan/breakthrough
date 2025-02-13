# trading/trade_executor.py
from logs.logger_config import setup_logger
import numpy as np
import pandas as pd
from trading.calculators import calculate_atr, calculate_dynamic_stop_and_take, adjust_trailing_stop

logger = setup_logger(__name__)

class TradeExecutor:
    """
    거래 실행 및 주문 체결 관련 로직을 제공하며,
    ATR, 스탑로스, 테이크 프로핏 계산 등은 calculators.py의 함수들을 호출합니다.
    """
    
    @staticmethod
    def compute_atr(data: pd.DataFrame, period: int = 14):
        return calculate_atr(data, period)

    @staticmethod
    def calculate_dynamic_stop_and_take(entry_price: float, atr: float, risk_params: dict):
        return calculate_dynamic_stop_and_take(entry_price, atr, risk_params)

    @staticmethod
    def adjust_trailing_stop(current_stop: float, current_price: float, highest_price: float, trailing_percentage: float):
        # 간단히 calculators.py의 함수 호출
        return adjust_trailing_stop(current_stop, current_price, highest_price, trailing_percentage)
    
    @staticmethod
    def calculate_partial_exit_targets(entry_price: float, partial_exit_ratio: float = 0.5,
                                       partial_profit_ratio: float = 0.03, final_profit_ratio: float = 0.06):
        # 계산 로직을 그대로 포함 (또는 calculators.py로 이동 가능)
        from trading.calculators import calculate_partial_exit_targets
        return calculate_partial_exit_targets(entry_price, partial_exit_ratio, partial_profit_ratio, final_profit_ratio)
