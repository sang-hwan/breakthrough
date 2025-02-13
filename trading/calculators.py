# trading/calculators.py
import pandas as pd
import numpy as np
import ta
from logs.logger_config import setup_logger

logger = setup_logger(__name__)

def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    try:
        if len(data) < period:
            data['atr'] = data['high'] - data['low']
            logger.debug("ATR 계산: 데이터 길이 부족, high-low 차이 사용")
        else:
            atr_indicator = ta.volatility.AverageTrueRange(
                high=data['high'],
                low=data['low'],
                close=data['close'],
                window=period,
                fillna=True
            )
            data['atr'] = atr_indicator.average_true_range()
            logger.debug(f"ATR 계산 성공: 첫 5행 {data['atr'].head().tolist()}")
    except Exception as e:
        logger.error(f"calculate_atr 에러: {e}", exc_info=True)
        data['atr'] = data['high'] - data['low']
    logger.debug(f"ATR 계산 완료: 총 {len(data)} 행 처리")
    return data

def calculate_dynamic_stop_and_take(entry_price: float, atr: float, risk_params: dict):
    atr_multiplier = risk_params.get("atr_multiplier", 2.0)
    profit_ratio = risk_params.get("profit_ratio", 0.05)
    volatility_multiplier = risk_params.get("volatility_multiplier", 1.0)
    stop_loss_price = entry_price - (atr * atr_multiplier * volatility_multiplier)
    take_profit_price = entry_price * (1 + profit_ratio)
    logger.debug(
        f"동적 스탑로스/테이크 프로핏 계산: entry_price={entry_price:.2f}, ATR={atr:.2f}, "
        f"atr_multiplier={atr_multiplier}, volatility_multiplier={volatility_multiplier}, profit_ratio={profit_ratio}, "
        f"계산된 stop_loss={stop_loss_price:.2f}, take_profit={take_profit_price:.2f}"
    )
    return stop_loss_price, take_profit_price

def adjust_trailing_stop(current_stop: float, current_price: float, highest_price: float, trailing_percentage: float,
                           volatility: float = 0.0, weekly_high: float = None, weekly_volatility: float = None) -> float:
    if current_stop is None:
        current_stop = highest_price * (1 - trailing_percentage * (1 + volatility))
    new_stop_intraday = highest_price * (1.0 - trailing_percentage * (1 + volatility))
    if weekly_high is not None:
        w_vol = weekly_volatility if weekly_volatility is not None else 0.0
        new_stop_weekly = weekly_high * (1 - trailing_percentage * (1 + w_vol))
        candidate_stop = max(new_stop_intraday, new_stop_weekly)
    else:
        candidate_stop = new_stop_intraday
    adjusted_stop = candidate_stop if candidate_stop > current_stop and candidate_stop < current_price else current_stop
    logger.debug(
        f"calculators.adjust_trailing_stop: current_price={current_price:.2f}, highest_price={highest_price:.2f}, "
        f"volatility={volatility:.4f}, trailing_percentage={trailing_percentage}, "
        f"weekly_high={weekly_high}, weekly_volatility={weekly_volatility}, 조정 후 stop={adjusted_stop:.2f}"
    )
    return adjusted_stop

def calculate_partial_exit_targets(entry_price: float, partial_exit_ratio: float = 0.5,
                                     partial_profit_ratio: float = 0.03, final_profit_ratio: float = 0.06,
                                     final_exit_ratio: float = 1.0, use_weekly_target: bool = False,
                                     weekly_momentum: float = None, weekly_adjustment_factor: float = 0.5):
    if use_weekly_target and weekly_momentum is not None:
        adjusted_partial = partial_profit_ratio + weekly_adjustment_factor * weekly_momentum
        adjusted_final = final_profit_ratio + weekly_adjustment_factor * weekly_momentum
    else:
        adjusted_partial = partial_profit_ratio
        adjusted_final = final_profit_ratio
    partial_target = entry_price * (1.0 + adjusted_partial)
    final_target = entry_price * (1.0 + adjusted_final)
    logger.debug(
        f"calculators.calculate_partial_exit_targets: entry_price={entry_price}, 기본 partial_profit_ratio={partial_profit_ratio}, "
        f"final_profit_ratio={final_profit_ratio}, "
        f"{'주간 목표 반영: weekly_momentum=' + str(weekly_momentum) if use_weekly_target else '기본 계산'}, "
        f"계산된 partial_target={partial_target:.2f}, final_target={final_target:.2f}"
    )
    return [(partial_target, partial_exit_ratio), (final_target, final_exit_ratio)]
