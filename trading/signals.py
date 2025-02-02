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
    confirmed_breakout_flag_col: str = "confirmed_breakout"
) -> pd.DataFrame:
    """
    돌파 전략 신호 생성 함수.
    - 과거 봉 기준 최고가와 거래량 조건을 적용하여 신호를 생성하고,
    - confirmation_bars만큼 연속해서 조건이 충족될 때 최종 신호를 확정합니다.
    """
    # 1) 이전 봉들 중 최고가 (현재 봉 제외)
    data[f'{high_max_prefix}{lookback_window}'] = (
        data['high'].shift(1).rolling(lookback_window).max()
    )
    # 2) 돌파 신호 (고가 또는 종가 기준)
    price_source = data['high'] if use_high_price else data['close']
    data[breakout_flag_col] = price_source > (data[f'{high_max_prefix}{lookback_window}'] * (1 + breakout_buffer))
    # 3) 거래량 이동 평균
    data[f'{vol_ma_prefix}{lookback_window}'] = (
        data['volume'].shift(1).rolling(lookback_window).mean()
    )
    data['volume_condition'] = data['volume'] > (volume_factor * data[f'{vol_ma_prefix}{lookback_window}'])
    
    # 4) 최종 신호: 돌파 및 거래량 조건 모두 충족되어야 함.
    data[breakout_flag_col] = data[breakout_flag_col] & data['volume_condition']
    
    # 5) 확정 돌파 (연속 확인)
    data[confirmed_breakout_flag_col] = (
        data[breakout_flag_col].rolling(confirmation_bars).sum() == confirmation_bars
    ).fillna(False)
    
    return data

def generate_retest_signals(
    data: pd.DataFrame,
    retest_threshold: float = 0.005,
    confirmation_bars: int = 2,
    breakout_reference_col: str = "highest_20",
    breakout_signal_col: str = "breakout_signal",
    retest_signal_col: str = "retest_signal"
) -> pd.DataFrame:
    """
    돌파 신호 발생 후 일정 기간 내에 가격이 돌파 기준가 근처로 풀백하고, 이후 반등하는 경우를 리테스트 신호로 생성합니다.
    
    Parameters:
      - data: OHLCV 데이터가 포함된 DataFrame. 반드시 breakout_reference_col과 breakout_signal_col이 포함되어 있어야 합니다.
      - retest_threshold: 돌파 기준가 대비 허용 풀백 범위 (예: 0.005는 0.5% 풀백 허용)
      - confirmation_bars: 리테스트 신호를 확인하기 위해 확인할 연속 캔들 수
      - breakout_reference_col: 돌파 기준가(예: 이전 lookback_window 동안의 최고가) 컬럼명
      - breakout_signal_col: 기존 돌파 신호 컬럼명 (예: breakout_signal)
      - retest_signal_col: 생성할 리테스트 신호 컬럼명 (예: retest_signal)
    
    Returns:
      - DataFrame에 retest_signal_col 컬럼이 추가되어 리테스트 신호(True/False)가 기록됩니다.
    """
    data[retest_signal_col] = False
    
    # 돌파 신호가 발생한 인덱스 위치들을 확인합니다.
    breakout_indices = data.index[data[breakout_signal_col] == True]
    for br_idx in breakout_indices:
        # 해당 행의 돌파 기준 가격
        breakout_level = data.loc[br_idx, breakout_reference_col]
        try:
            pos = data.index.get_loc(br_idx)
        except Exception:
            continue
        # 이후 confirmation_bars 기간 동안 리테스트 조건을 확인합니다.
        for offset in range(1, confirmation_bars + 1):
            if pos + offset >= len(data):
                break
            current_row = data.iloc[pos + offset]
            # 조건: 현재 행의 low가 breakout_level 범위 내 (즉, breakout_level*(1 - retest_threshold) 이상)
            if current_row['low'] <= breakout_level and current_row['low'] >= breakout_level * (1 - retest_threshold):
                # 다음 캔들이 존재하는지 확인한 후, 그 캔들의 close가 breakout_level을 상회하면 리테스트 신호로 간주
                if pos + offset + 1 < len(data):
                    next_row = data.iloc[pos + offset + 1]
                    if next_row['close'] > breakout_level:
                        retest_idx = next_row.name
                        data.at[retest_idx, retest_signal_col] = True
                        break
    return data

def filter_long_trend(
    data: pd.DataFrame,
    price_column: str = 'close',
    sma_period: int = 200,
    macd_slow_period: int = 26,
    macd_fast_period: int = 12,
    macd_signal_period: int = 9,
    rsi_period: int = 14,
    rsi_threshold: float = 70.0,
    bb_period: int = 20,
    bb_std_multiplier: float = 2.0,
    fillna: bool = False,
    macd_diff_column: str = 'macd_diff',
    rsi_column: str = 'rsi',
    bb_upper_band_column: str = 'bb_hband'
) -> pd.DataFrame:
    """
    장기 추세 필터 함수.
    SMA, MACD, RSI, Bollinger Bands를 계산한 후, 여러 조건을 만족하면 long_filter_pass가 True가 됩니다.
    """
    # 1) SMA 계산
    sma_col = f"sma{sma_period}"
    data = compute_sma(
        data,
        price_column=price_column,
        period=sma_period,
        fillna=fillna,
        output_col=sma_col
    )
    # 2) MACD 계산
    data = compute_macd(
        data,
        price_column=price_column,
        slow_period=macd_slow_period,
        fast_period=macd_fast_period,
        signal_period=macd_signal_period,
        fillna=fillna,
        prefix='macd_'
    )
    # 3) RSI 계산
    data = compute_rsi(
        data,
        price_column=price_column,
        period=rsi_period,
        fillna=fillna,
        output_col=rsi_column
    )
    # 4) Bollinger Bands 계산
    data = compute_bollinger_bands(
        data,
        price_column=price_column,
        period=bb_period,
        std_multiplier=bb_std_multiplier,
        fillna=fillna,
        prefix='bb_'
    )
    # 5) 추세 필터 조건 적용
    data['long_filter_pass'] = (
        (data[price_column] >= data[sma_col]) &
        (data[macd_diff_column] > 0) &
        (data[rsi_column] < rsi_threshold) &
        (data[bb_upper_band_column] > data[price_column])
    )
    return data
