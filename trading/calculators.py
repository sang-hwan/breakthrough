# trading/calculators.py
import pandas as pd
import ta
from logs.logger_config import setup_logger

logger = setup_logger(__name__)

class InvalidEntryPriceError(ValueError):
    pass

def calculate_atr(data: pd.DataFrame, period: int = 14, min_atr: float = None) -> pd.DataFrame:
    required_cols = ['high', 'low', 'close']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
    
    data = data[data['high'] >= data['low']].copy()
    
    data.loc[data['close'] < data['low'], 'close'] = data['low']
    data.loc[data['close'] > data['high'], 'close'] = data['high']
    
    range_series = data['high'] - data['low']
    typical_range = range_series.median()
    if typical_range > 0:
        data = data[range_series <= (3 * typical_range)]
    else:
        logger.debug("Typical range is zero; skipping outlier filtering.")
    
    effective_period = period if len(data) >= period else len(data)
    
    try:
        if effective_period < 1:
            data['atr'] = 0
        elif len(data) < effective_period:
            data['atr'] = data['high'] - data['low']
        else:
            atr_indicator = ta.volatility.AverageTrueRange(
                high=data['high'],
                low=data['low'],
                close=data['close'],
                window=effective_period,
                fillna=True
            )
            data['atr'] = atr_indicator.average_true_range()
    except Exception as e:
        logger.error("calculate_atr error: " + str(e), exc_info=True)
        data['atr'] = data['high'] - data['low']
    
    avg_close = data['close'].mean()
    if min_atr is None:
        min_atr = avg_close * 0.01

    data['atr'] = data['atr'].apply(lambda x: max(x, min_atr))
    
    return data

def calculate_dynamic_stop_and_take(entry_price: float, atr: float, risk_params: dict):
    if entry_price <= 0:
        logger.error(f"Invalid entry_price: {entry_price}. Must be positive.", exc_info=True)
        raise InvalidEntryPriceError(f"Invalid entry_price: {entry_price}. Must be positive.")
    if atr <= 0:
        logger.error(f"ATR value is non-positive ({atr}). Using fallback ATR value from risk_params if available.", exc_info=True)
        fallback_atr = risk_params.get("fallback_atr", entry_price * 0.01)
        if fallback_atr <= 0:
            fallback_atr = entry_price * 0.01
        atr = fallback_atr
    try:
        atr_multiplier = risk_params.get("atr_multiplier", 2.0)
        profit_ratio = risk_params.get("profit_ratio", 0.05)
        volatility_multiplier = risk_params.get("volatility_multiplier", 1.0)
        atr_multiplier = max(0.1, min(atr_multiplier, 10))
        profit_ratio = max(0.001, min(profit_ratio, 1))
        stop_loss_price = entry_price - (atr * atr_multiplier * volatility_multiplier)
        take_profit_price = entry_price * (1 + profit_ratio)
        if stop_loss_price <= 0:
            logger.error("Computed stop_loss_price is non-positive; adjusting to at least 50% of entry_price.", exc_info=True)
            stop_loss_price = entry_price * 0.5
        logger.debug(f"Calculated stop_loss={stop_loss_price:.2f}, take_profit={take_profit_price:.2f} (entry_price={entry_price}, atr={atr}, atr_multiplier={atr_multiplier}, profit_ratio={profit_ratio})")
        return stop_loss_price, take_profit_price
    except Exception as e:
        logger.error("calculate_dynamic_stop_and_take error: " + str(e), exc_info=True)
        raise

def adjust_trailing_stop(current_stop: float, current_price: float, highest_price: float, trailing_percentage: float,
                           volatility: float = 0.0, weekly_high: float = None, weekly_volatility: float = None) -> float:
    if current_price <= 0 or highest_price <= 0:
        logger.error(f"Invalid current_price ({current_price}) or highest_price ({highest_price}).", exc_info=True)
        raise ValueError("current_price and highest_price must be positive.")
    if trailing_percentage < 0:
        logger.error(f"Invalid trailing_percentage ({trailing_percentage}). Must be non-negative.", exc_info=True)
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
        logger.debug(f"Adjusted trailing stop: {adjusted_stop:.2f} (current_stop={current_stop}, candidate_stop={candidate_stop}, current_price={current_price})")
        return adjusted_stop
    except Exception as e:
        logger.error("adjust_trailing_stop error: " + str(e), exc_info=True)
        raise

def calculate_partial_exit_targets(entry_price: float, partial_exit_ratio: float = 0.5,
                                     partial_profit_ratio: float = 0.03, final_profit_ratio: float = 0.06,
                                     final_exit_ratio: float = 1.0, use_weekly_target: bool = False,
                                     weekly_momentum: float = None, weekly_adjustment_factor: float = 0.5):
    if entry_price <= 0:
        logger.error(f"Invalid entry_price: {entry_price}. Must be positive.", exc_info=True)
        raise InvalidEntryPriceError(f"Invalid entry_price: {entry_price}. Must be positive.")
    try:
        if use_weekly_target and weekly_momentum is not None:
            adjusted_partial = partial_profit_ratio + weekly_adjustment_factor * weekly_momentum
            adjusted_final = final_profit_ratio + weekly_adjustment_factor * weekly_momentum
        else:
            adjusted_partial = partial_profit_ratio
            adjusted_final = final_profit_ratio
        partial_target = entry_price * (1 + adjusted_partial)
        final_target = entry_price * (1 + adjusted_final)
        logger.debug(f"Partial targets: partial={partial_target:.2f}, final={final_target:.2f} (entry_price={entry_price}, adjusted_partial={adjusted_partial}, adjusted_final={adjusted_final})")
        return [(partial_target, partial_exit_ratio), (final_target, final_exit_ratio)]
    except Exception as e:
        logger.error("calculate_partial_exit_targets error: " + str(e), exc_info=True)
        raise
