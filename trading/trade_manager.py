# trading/trade_manager.py
import pandas as pd
import numpy as np
import ta
from logs.logger_config import setup_logger

logger = setup_logger(__name__)

class TradeManager:
    @staticmethod
    def calculate_atr_stop_loss(
        data: pd.DataFrame,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        dynamic_sl_adjustment: float = 1.0,
        stop_loss_col: str = 'stop_loss_price',
        entry_price_col: str = 'entry_price',
        atr_col: str = 'atr',
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
        entry_signal_col: str = 'long_entry'
    ) -> pd.DataFrame:
        if len(data) < atr_period:
            data[atr_col] = data[high_col] - data[low_col]
        else:
            try:
                atr_indicator = ta.volatility.AverageTrueRange(
                    high=data[high_col],
                    low=data[low_col],
                    close=data[close_col],
                    window=atr_period,
                    fillna=True
                )
                data[atr_col] = atr_indicator.average_true_range()
            except Exception as e:
                logger.error(f"ATR 계산 에러: {e}")
                data[atr_col] = data[high_col] - data[low_col]
        data['close_ma'] = data[close_col].rolling(window=atr_period, min_periods=1).mean()
        data['close_std'] = data[close_col].rolling(window=atr_period, min_periods=1).std()
        data['std_ratio'] = data['close_std'] / data['close_ma']
        data['dynamic_multiplier'] = atr_multiplier * (1 + data['std_ratio'])
        data[entry_price_col] = np.where(data.get(entry_signal_col, False), data[close_col], np.nan)
        data[entry_price_col] = data[entry_price_col].ffill()
        data[stop_loss_col] = data[entry_price_col] - (data[atr_col] * data['dynamic_multiplier'] * dynamic_sl_adjustment)
        data.drop(columns=['close_ma', 'close_std', 'std_ratio', 'dynamic_multiplier'], inplace=True)
        logger.debug("ATR 기반 스탑로스 계산 완료.")
        return data

    @staticmethod
    def adjust_trailing_stop(
        current_stop: float,
        current_price: float,
        highest_price: float,
        trailing_percentage: float,
        volatility: float = 0.0
    ) -> float:
        if current_stop is None:
            current_stop = highest_price * (1 - trailing_percentage * (1 + volatility))
        new_stop = highest_price * (1.0 - trailing_percentage * (1 + volatility))
        adjusted_stop = new_stop if new_stop > current_stop and new_stop < current_price else current_stop
        logger.debug(f"조정된 트레일링 스탑: {adjusted_stop}")
        return adjusted_stop

    @staticmethod
    def set_fixed_take_profit(
        data: pd.DataFrame,
        profit_ratio: float = 0.05,
        take_profit_col: str = 'take_profit_price',
        entry_price_col: str = 'entry_price'
    ) -> pd.DataFrame:
        data[take_profit_col] = data[entry_price_col] * (1 + profit_ratio)
        logger.debug("고정 테이크 프로핏 설정 완료.")
        return data

    @staticmethod
    def should_exit_trend(
        data: pd.DataFrame,
        current_time,
        window_length: int = 20,
        price_column: str = 'close'
    ) -> bool:
        if current_time not in data.index:
            data_sub = data.loc[:current_time]
            if len(data_sub) < window_length:
                return False
            window_data = data_sub.iloc[-window_length:]
        else:
            idx = data.index.get_loc(current_time)
            if idx < window_length:
                return False
            window_data = data.iloc[idx - window_length + 1: idx + 1]
        recent_min = window_data[price_column].min()
        current_price = data.loc[current_time, price_column] if current_time in data.index else data.iloc[-1][price_column]
        decision = current_price < recent_min
        logger.debug(f"should_exit_trend 결정: {decision} (current_price={current_price}, recent_min={recent_min})")
        return decision

    @staticmethod
    def calculate_partial_exit_targets(
        entry_price: float,
        partial_exit_ratio: float = 0.5,
        partial_profit_ratio: float = 0.03,
        final_profit_ratio: float = 0.06,
        final_exit_ratio: float = 1.0
    ):
        partial_target = entry_price * (1.0 + partial_profit_ratio)
        final_target = entry_price * (1.0 + final_profit_ratio)
        logger.debug(f"부분 청산 목표 계산: partial_target={partial_target}, final_target={final_target}")
        return [(partial_target, partial_exit_ratio), (final_target, final_exit_ratio)]

    @staticmethod
    def compute_atr(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        try:
            atr_indicator = ta.volatility.AverageTrueRange(
                high=data['high'],
                low=data['low'],
                close=data['close'],
                window=period,
                fillna=True
            )
            data['atr'] = atr_indicator.average_true_range()
        except Exception as e:
            logger.error(f"compute_atr 에러: {e}")
            data['atr'] = data['high'] - data['low']
        logger.debug("ATR 계산 완료.")
        return data

    @staticmethod
    def calculate_dynamic_stop_and_take(
        entry_price: float,
        atr: float,
        risk_params: dict
    ):
        """
        동적 스탑로스와 테이크 프로핏 가격을 계산합니다.
        - entry_price: 진입 가격
        - atr: 현재 ATR 값
        - risk_params: 리스크 파라미터 (예: 'atr_multiplier', 'profit_ratio', 'volatility_multiplier' 등 포함)
        """
        atr_multiplier = risk_params.get("atr_multiplier", 2.0)
        profit_ratio = risk_params.get("profit_ratio", 0.05)
        volatility_multiplier = risk_params.get("volatility_multiplier", 1.0)
        stop_loss_price = entry_price - (atr * atr_multiplier * volatility_multiplier)
        take_profit_price = entry_price * (1 + profit_ratio)
        logger.debug(f"동적 스탑로스/테이크 프로핏 계산: entry_price={entry_price}, atr={atr}, stop_loss={stop_loss_price}, take_profit={take_profit_price}")
        return stop_loss_price, take_profit_price
