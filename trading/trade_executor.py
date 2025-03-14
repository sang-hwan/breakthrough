# trading/trade_executor.py

from logs.logger_config import setup_logger
import pandas as pd
from trading.calculators import (
    calculate_atr,
    calculate_dynamic_stop_and_take,
    adjust_trailing_stop,
    calculate_partial_exit_targets
)

# 전역 변수 및 객체 정의
# logger: 이 모듈의 거래 실행 관련 정보를 기록하는 로깅 객체입니다.
logger = setup_logger(__name__)


class TradeExecutor:
    @staticmethod
    def compute_atr(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        시장 변동성 지표인 ATR(Average True Range)을 계산합니다.
        
        Parameters:
          - data (pd.DataFrame): 시장 데이터가 포함된 데이터프레임.
          - period (int): ATR 계산에 사용될 기간.
        
        Returns:
          - pd.DataFrame: ATR 결과가 포함된 데이터프레임.
        
        동작 방식:
          - 외부의 calculate_atr 함수를 호출하여 데이터프레임에 ATR 값을 계산하고 반환합니다.
        """
        try:
            result = calculate_atr(data, period)
            logger.debug("ATR computed successfully.")
            return result
        except Exception as e:
            logger.error("Error in compute_atr: {}".format(e), exc_info=True)
            raise

    @staticmethod
    def calculate_dynamic_stop_and_take(entry_price: float, atr: float, risk_params: dict):
        """
        ATR 및 위험 파라미터를 기반으로 동적인 손절(stop loss) 및 이익 실현(take profit) 가격을 계산합니다.
        
        Parameters:
          - entry_price (float): 진입 가격.
          - atr (float): Average True Range 값.
          - risk_params (dict): 위험 관리 관련 파라미터 (예: 손절/이익 실현 배수).
        
        Returns:
          - tuple: (stop_loss, take_profit) 값.
        
        동작 방식:
          - 외부의 calculate_dynamic_stop_and_take 함수를 호출하여 계산된 stop loss와 take profit 값을 반환합니다.
        """
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
        """
        주어진 시장 데이터를 바탕으로 trailing stop(후행 손절)을 조정합니다.
        
        Parameters:
          - current_stop (float): 현재 trailing stop 값.
          - current_price (float): 현재 시장 가격.
          - highest_price (float): 최근 최고 가격.
          - trailing_percentage (float): trailing stop 조정을 위한 백분율 (예: 0.02는 2%).
        
        Returns:
          - float: 조정된 trailing stop 값.
        
        동작 방식:
          - 외부의 adjust_trailing_stop 함수를 호출하여 새 trailing stop 값을 계산한 후 반환합니다.
        """
        try:
            new_stop = adjust_trailing_stop(current_stop, current_price, highest_price, trailing_percentage)
            logger.debug(f"Trailing stop adjusted to {new_stop:.2f} (current_price={current_price}, highest_price={highest_price})")
            return new_stop
        except Exception as e:
            logger.error("Error in adjust_trailing_stop: {}".format(e), exc_info=True)
            raise

    @staticmethod
    def calculate_partial_exit_targets(entry_price: float, partial_exit_ratio: float = 0.5,
                                       partial_profit_ratio: float = 0.03, final_profit_ratio: float = 0.06):
        """
        포지션의 일부 청산을 위한 목표 가격들을 계산합니다.
        
        Parameters:
          - entry_price (float): 진입 가격.
          - partial_exit_ratio (float): 부분 청산 시 청산할 비율.
          - partial_profit_ratio (float): 부분 이익 목표 비율.
          - final_profit_ratio (float): 최종 이익 목표 비율.
        
        Returns:
          - list of tuples: 각 청산 단계에 대해 (target_price, exit_ratio)를 담은 튜플 리스트.
        
        동작 방식:
          - 외부의 calculate_partial_exit_targets 함수를 호출하여 부분 및 최종 청산 목표 가격을 계산 후 반환합니다.
        """
        try:
            targets = calculate_partial_exit_targets(entry_price, partial_exit_ratio, partial_profit_ratio, final_profit_ratio)
            logger.debug(f"Partial exit targets computed: {targets} (entry_price={entry_price})")
            return targets
        except Exception as e:
            logger.error("Error in calculate_partial_exit_targets: {}".format(e), exc_info=True)
            raise
