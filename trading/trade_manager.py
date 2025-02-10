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
        # ATR 계산
        if len(data) < atr_period:
            data[atr_col] = data[high_col] - data[low_col]
            logger.info("ATR 계산: 데이터 길이가 짧아 high-low 차이 사용.")
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
                logger.info(f"ATR 계산 성공: 첫 5행 {data[atr_col].head().tolist()}")
            except Exception as e:
                logger.error(f"ATR 계산 에러: {e}", exc_info=True)
                data[atr_col] = data[high_col] - data[low_col]

        # 중간 계산: rolling 평균, 표준편차, 비율 및 동적 승수 계산
        data['close_ma'] = data[close_col].rolling(window=atr_period, min_periods=1).mean()
        data['close_std'] = data[close_col].rolling(window=atr_period, min_periods=1).std()
        data['std_ratio'] = data['close_std'] / data['close_ma']
        data['dynamic_multiplier'] = atr_multiplier * (1 + data['std_ratio'])
        logger.info(
            f"중간 계산 결과: close_ma 첫 5행={data['close_ma'].head().tolist()}, "
            f"close_std 첫 5행={data['close_std'].head().tolist()}, "
            f"std_ratio 첫 5행={data['std_ratio'].head().tolist()}, "
            f"dynamic_multiplier 첫 5행={data['dynamic_multiplier'].head().tolist()}"
        )
        
        # entry_price 컬럼 채우기 (입력 신호 발생 시 close 값을 기록하고, ffill)
        data[entry_price_col] = np.where(data.get(entry_signal_col, False), data[close_col], np.nan)
        data[entry_price_col] = data[entry_price_col].ffill()
        
        # 스탑로스 가격 계산
        data[stop_loss_col] = data[entry_price_col] - (data[atr_col] * data['dynamic_multiplier'] * dynamic_sl_adjustment)
        logger.info(f"스탑로스 계산: 첫 5행 {data[stop_loss_col].head().tolist()}")
        
        # 중간 계산에 사용한 임시 컬럼 제거
        data.drop(columns=['close_ma', 'close_std', 'std_ratio', 'dynamic_multiplier'], inplace=True)

        logger.info(f"ATR 기반 스탑로스 계산 완료: 총 {len(data)} 행 처리")
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
        logger.info(
            f"트레일링 스탑 조정: current_price={current_price:.2f}, highest_price={highest_price:.2f}, "
            f"조정 후 스탑={adjusted_stop:.2f}"
        )
        return adjusted_stop

    @staticmethod
    def set_fixed_take_profit(
        data: pd.DataFrame,
        profit_ratio: float = 0.05,
        take_profit_col: str = 'take_profit_price',
        entry_price_col: str = 'entry_price'
    ) -> pd.DataFrame:
        data[take_profit_col] = data[entry_price_col] * (1 + profit_ratio)
        logger.info(f"고정 테이크 프로핏 설정 완료: 총 {len(data)} 행 처리")
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
        current_price = (
            data.loc[current_time, price_column]
            if current_time in data.index
            else data.iloc[-1][price_column]
        )
        decision = current_price < recent_min
        logger.info(
            f"should_exit_trend 결정: current_price={current_price}, recent_min={recent_min}, exit_decision={decision}"
        )
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
        logger.info(
            f"부분 청산 목표 계산: entry_price={entry_price}, partial_profit_ratio={partial_profit_ratio}, "
            f"final_profit_ratio={final_profit_ratio}, 계산된 partial_target={partial_target:.2f}, final_target={final_target:.2f}"
        )
        logger.info(f"부분 청산 목표 계산 완료: partial_target = {partial_target:.2f}, final_target = {final_target:.2f}")
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
            logger.info(f"compute_atr: 첫 5행 ATR 값: {data['atr'].head().tolist()}")
        except Exception as e:
            logger.error(f"compute_atr 에러: {e}", exc_info=True)
            data['atr'] = data['high'] - data['low']
        logger.info(f"ATR 계산 완료: 총 {len(data)} 행 처리")
        return data

    @staticmethod
    def calculate_dynamic_stop_and_take(
        entry_price: float,
        atr: float,
        risk_params: dict
    ):
        atr_multiplier = risk_params.get("atr_multiplier", 2.0)
        profit_ratio = risk_params.get("profit_ratio", 0.05)
        volatility_multiplier = risk_params.get("volatility_multiplier", 1.0)
        stop_loss_price = entry_price - (atr * atr_multiplier * volatility_multiplier)
        take_profit_price = entry_price * (1 + profit_ratio)
        logger.info(
            f"동적 스탑로스/테이크 프로핏 계산: entry_price={entry_price:.2f}, ATR={atr:.2f}, "
            f"atr_multiplier={atr_multiplier}, volatility_multiplier={volatility_multiplier}, profit_ratio={profit_ratio}, "
            f"계산된 stop_loss={stop_loss_price:.2f}, take_profit={take_profit_price:.2f}"
        )
        return stop_loss_price, take_profit_price
