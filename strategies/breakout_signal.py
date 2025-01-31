# strategies/breakout_signal.py
# 특정 전략(돌파전략)에서 사용될 지표 계산 함수를 담은 파일입니다.

import pandas as pd

def calculate_breakout_signals(
    df: pd.DataFrame,
    window: int = 20,
    vol_factor: float = 1.5,
    confirm_bars: int = 2,
    use_high: bool = False,
    breakout_buffer: float = 0.0
) -> pd.DataFrame:
    """
    돌파 전략에 필요한 여러 시그널을 DataFrame에 추가해주는 함수.
    
    - 전고점(highest_xx) 계산
    - 종가/고가가 전고점보다 높은지 판단(breakout_signal)
    - 거래량이 과거 평균보다 큰지 판단(volume_condition)
    - 돌파 신호가 confirm_bars 연속으로 발생하면 확정 돌파(confirmed_breakout)로 표시
    """

    # 1) window 기간의 전고점(Highest High) 계산 (현재 봉 제외, shift(1) 사용)
    df[f'highest_{window}'] = (
        df['high'].shift(1)
        .rolling(window)
        .max()
    )

    # 2) 돌파 신호(breakout_signal) - 전고점 * (1 + breakout_buffer)보다 크면 돌파로 본다
    if use_high:
        df['breakout_signal'] = df['high'] > (df[f'highest_{window}'] * (1 + breakout_buffer))
    else:
        df['breakout_signal'] = df['close'] > (df[f'highest_{window}'] * (1 + breakout_buffer))

    # 3) 거래량 평균 구하기 (shift(1)로 현재 봉 제외)
    df[f'vol_ma_{window}'] = (
        df['volume'].shift(1)
        .rolling(window)
        .mean()
    )
    df['volume_condition'] = df['volume'] > (vol_factor * df[f'vol_ma_{window}'])

    # 4) 확정 돌파(confirmed_breakout) - 예: 2개 봉 연속 breakout_signal이 True이면 True
    df['confirmed_breakout'] = (
        df['breakout_signal'].rolling(confirm_bars).sum() == confirm_bars
    )
    df['confirmed_breakout'] = df['confirmed_breakout'].fillna(False)

    return df
