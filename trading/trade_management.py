# trading/trade_management.py
import pandas as pd
import numpy as np
import ta

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
            data[atr_col] = data[high_col] - data[low_col]
    data['close_ma'] = data[close_col].rolling(window=atr_period, min_periods=1).mean()
    data['close_std'] = data[close_col].rolling(window=atr_period, min_periods=1).std()
    data['std_ratio'] = data['close_std'] / data['close_ma']
    data['dynamic_multiplier'] = atr_multiplier * (1 + data['std_ratio'])
    data[entry_price_col] = np.where(data.get(entry_signal_col, False), data[close_col], np.nan)
    data[entry_price_col] = data[entry_price_col].ffill()
    data[stop_loss_col] = data[entry_price_col] - (data[atr_col] * data['dynamic_multiplier'] * dynamic_sl_adjustment)
    data.drop(columns=['close_ma', 'close_std', 'std_ratio', 'dynamic_multiplier'], inplace=True)
    return data

def adjust_trailing_stop(
    current_stop: float,
    current_price: float,
    highest_price: float,
    trailing_percentage: float
) -> float:
    new_stop = highest_price * (1.0 - trailing_percentage)
    return new_stop if new_stop > current_stop and new_stop < current_price else current_stop

def set_fixed_take_profit(
    data: pd.DataFrame,
    profit_ratio: float = 0.05,
    take_profit_col: str = 'take_profit_price',
    entry_price_col: str = 'entry_price'
) -> pd.DataFrame:
    data[take_profit_col] = data[entry_price_col] * (1 + profit_ratio)
    return data

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
    return current_price < recent_min

def calculate_partial_exit_targets(
    entry_price: float,
    partial_exit_ratio: float = 0.5,
    partial_profit_ratio: float = 0.03,
    final_profit_ratio: float = 0.06,
    final_exit_ratio: float = 1.0
):
    partial_target = entry_price * (1.0 + partial_profit_ratio)
    final_target = entry_price * (1.0 + final_profit_ratio)
    return [
        (partial_target, partial_exit_ratio),
        (final_target, final_exit_ratio)
    ]
