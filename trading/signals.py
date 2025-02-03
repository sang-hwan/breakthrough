# trading/signals.py
import pandas as pd
from trading.indicators import compute_sma, compute_macd, compute_rsi, compute_bollinger_bands

def generate_breakout_signals(
    data: pd.DataFrame,
    lookback_window: int = 20,
    volume_factor: float = 1.5,
    confirmation_bars: int = 2,
    use_high_price: bool = False,
    breakout_buffer: float = 0.0,
    high_max_prefix: str = "highest_",
    breakout_flag_col: str = "breakout_signal",
    vol_ma_prefix: str = "vol_ma_",
    confirmed_breakout_flag_col: str = "confirmed_breakout",
    high_col: str = "high",
    close_col: str = "close",
    volume_col: str = "volume",
    volume_condition_col: str = "volume_condition"
) -> pd.DataFrame:
    data[f'{high_max_prefix}{lookback_window}'] = data[high_col].shift(1).rolling(lookback_window).max()
    price_source = data[high_col] if use_high_price else data[close_col]
    data[breakout_flag_col] = price_source > (data[f'{high_max_prefix}{lookback_window}'] * (1 + breakout_buffer))
    data[f'{vol_ma_prefix}{lookback_window}'] = data[volume_col].shift(1).rolling(lookback_window).mean()
    data[volume_condition_col] = data[volume_col] > (volume_factor * data[f'{vol_ma_prefix}{lookback_window}'])
    data[breakout_flag_col] = data[breakout_flag_col] & data[volume_condition_col]
    data[confirmed_breakout_flag_col] = (data[breakout_flag_col].rolling(confirmation_bars).sum() == confirmation_bars).fillna(False)
    return data

def generate_retest_signals(
    data: pd.DataFrame,
    retest_threshold: float = 0.005,
    confirmation_bars: int = 2,
    breakout_reference_col: str = "highest_20",
    breakout_signal_col: str = "breakout_signal",
    retest_signal_col: str = "retest_signal",
    low_col: str = "low",
    close_col: str = "close"
) -> pd.DataFrame:
    data[retest_signal_col] = False
    breakout_indices = data.index[data[breakout_signal_col] == True]
    for br_idx in breakout_indices:
        breakout_level = data.loc[br_idx, breakout_reference_col]
        try:
            pos = data.index.get_loc(br_idx)
        except Exception:
            continue
        for offset in range(1, confirmation_bars + 1):
            if pos + offset >= len(data):
                break
            current_row = data.iloc[pos + offset]
            if current_row[low_col] <= breakout_level and current_row[low_col] >= breakout_level * (1 - retest_threshold):
                if pos + offset + 1 < len(data):
                    next_row = data.iloc[pos + offset + 1]
                    if next_row[close_col] > breakout_level:
                        retest_idx = next_row.name
                        data.at[retest_idx, retest_signal_col] = True
                        break
                else:
                    break
    return data

def filter_long_trend_relaxed(
    data: pd.DataFrame,
    price_column: str = 'close',
    sma_period: int = 200,
    macd_slow_period: int = 26,
    macd_fast_period: int = 12,
    macd_signal_period: int = 9,
    rsi_period: int = 14,
    rsi_threshold: float = 75.0,
    bb_period: int = 20,
    bb_std_multiplier: float = 2.0,
    fillna: bool = False,
    macd_diff_column: str = 'macd_diff',
    rsi_column: str = 'rsi',
    bb_upper_band_column: str = 'bb_hband',
    use_sma: bool = True,
    use_macd: bool = True,
    use_rsi: bool = True,
    use_bb: bool = True,
    macd_diff_threshold: float = -0.5
) -> pd.DataFrame:
    if use_sma:
        sma_col = f"sma{sma_period}"
        data = compute_sma(data, price_column=price_column, period=sma_period, fillna=fillna, output_col=sma_col)
        sma_condition = data[price_column] >= data[sma_col]
    else:
        sma_condition = pd.Series(True, index=data.index)
    if use_macd:
        data = compute_macd(data, price_column=price_column, slow_period=macd_slow_period, fast_period=macd_fast_period, signal_period=macd_signal_period, fillna=fillna, prefix='macd_')
        macd_condition = data[macd_diff_column] > macd_diff_threshold
    else:
        macd_condition = pd.Series(False, index=data.index)
    if use_rsi:
        data = compute_rsi(data, price_column=price_column, period=rsi_period, fillna=fillna, output_col=rsi_column)
        rsi_condition = data[rsi_column] < rsi_threshold
    else:
        rsi_condition = pd.Series(False, index=data.index)
    if use_bb:
        data = compute_bollinger_bands(data, price_column=price_column, period=bb_period, std_multiplier=bb_std_multiplier, fillna=fillna, prefix='bb_')
        bb_condition = data[bb_upper_band_column] > data[price_column]
    else:
        bb_condition = pd.Series(False, index=data.index)
    optional_pass = macd_condition | rsi_condition | bb_condition
    data['long_filter_pass'] = sma_condition & optional_pass
    return data
