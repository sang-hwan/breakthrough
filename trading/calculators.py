# trading/calculators.py
import pandas as pd
import ta
from logs.logger_config import setup_logger

logger = setup_logger(__name__)

def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    try:
        if len(data) < period:
            data['atr'] = data['high'] - data['low']
        else:
            atr_indicator = ta.volatility.AverageTrueRange(
                high=data['high'],
                low=data['low'],
                close=data['close'],
                window=period,
                fillna=True
            )
            data['atr'] = atr_indicator.average_true_range()
    except Exception as e:
        logger.error(f"calculate_atr error: {e}", exc_info=True)
        data['atr'] = data['high'] - data['low']
    return data

def calculate_dynamic_stop_and_take(entry_price: float, atr: float, risk_params: dict):
    # 입력값 검증
    if entry_price <= 0:
        logger.error("Invalid entry_price <= 0: {}".format(entry_price))
        raise ValueError("entry_price must be positive.")
    if atr <= 0:
        logger.error("Invalid atr <= 0: {}".format(atr))
        raise ValueError("atr must be positive.")
    try:
        atr_multiplier = risk_params.get("atr_multiplier", 2.0)
        profit_ratio = risk_params.get("profit_ratio", 0.05)
        volatility_multiplier = risk_params.get("volatility_multiplier", 1.0)
        # 극단적인 값 방지를 위해 클리핑
        atr_multiplier = max(0.1, min(atr_multiplier, 10))
        profit_ratio = max(0.001, min(profit_ratio, 1))
        stop_loss_price = entry_price - (atr * atr_multiplier * volatility_multiplier)
        take_profit_price = entry_price * (1 + profit_ratio)
        # 손절 가격이 0 이하인 경우 최소 50% 수준으로 조정
        if stop_loss_price <= 0:
            logger.warning("Computed stop_loss_price is non-positive; adjusting to at least 50% of entry_price.")
            stop_loss_price = entry_price * 0.5
        logger.debug(f"Calculated stop_loss={stop_loss_price:.2f}, take_profit={take_profit_price:.2f} "
                     f"(entry_price={entry_price}, atr={atr}, atr_multiplier={atr_multiplier}, profit_ratio={profit_ratio})")
        return stop_loss_price, take_profit_price
    except Exception as e:
        logger.error(f"calculate_dynamic_stop_and_take error: {e}", exc_info=True)
        raise

def adjust_trailing_stop(current_stop: float, current_price: float, highest_price: float, trailing_percentage: float,
                           volatility: float = 0.0, weekly_high: float = None, weekly_volatility: float = None) -> float:
    if current_price <= 0 or highest_price <= 0:
        logger.error("Invalid current_price or highest_price: current_price={}, highest_price={}".format(current_price, highest_price))
        raise ValueError("current_price and highest_price must be positive.")
    if trailing_percentage < 0:
        logger.error("Invalid trailing_percentage < 0: {}".format(trailing_percentage))
        raise ValueError("trailing_percentage must be non-negative.")
    try:
        if current_stop is None or current_stop <= 0:
            current_stop = highest_price * (1 - trailing_percentage * (1 + volatility))
        new_stop_intraday = highest_price * (1 - trailing_percentage * (1 + volatility))
        if weekly_high is not None:
            w_vol = weekly_volatility if weekly_volatility is not None else 0.0
            new_stop_weekly = weekly_high * (1 - trailing_percentage * (1 + w_vol))
            candidate_stop = max(new_stop_intraday, new_stop_weekly)
        else:
            candidate_stop = new_stop_intraday
        adjusted_stop = candidate_stop if candidate_stop > current_stop and candidate_stop < current_price else current_stop
        logger.debug(f"Adjusted trailing stop: {adjusted_stop:.2f} "
                     f"(current_stop={current_stop}, candidate_stop={candidate_stop}, current_price={current_price})")
        return adjusted_stop
    except Exception as e:
        logger.error(f"adjust_trailing_stop error: {e}", exc_info=True)
        raise

def calculate_partial_exit_targets(entry_price: float, partial_exit_ratio: float = 0.5,
                                     partial_profit_ratio: float = 0.03, final_profit_ratio: float = 0.06,
                                     final_exit_ratio: float = 1.0, use_weekly_target: bool = False,
                                     weekly_momentum: float = None, weekly_adjustment_factor: float = 0.5):
    if entry_price <= 0:
        logger.error("Invalid entry_price <= 0: {}".format(entry_price))
        raise ValueError("entry_price must be positive.")
    try:
        if use_weekly_target and weekly_momentum is not None:
            adjusted_partial = partial_profit_ratio + weekly_adjustment_factor * weekly_momentum
            adjusted_final = final_profit_ratio + weekly_adjustment_factor * weekly_momentum
        else:
            adjusted_partial = partial_profit_ratio
            adjusted_final = final_profit_ratio
        partial_target = entry_price * (1 + adjusted_partial)
        final_target = entry_price * (1 + adjusted_final)
        logger.debug(f"Partial targets: partial={partial_target:.2f}, final={final_target:.2f} "
                     f"(entry_price={entry_price}, adjusted_partial={adjusted_partial}, adjusted_final={adjusted_final})")
        return [(partial_target, partial_exit_ratio), (final_target, final_exit_ratio)]
    except Exception as e:
        logger.error(f"calculate_partial_exit_targets error: {e}", exc_info=True)
        raise
