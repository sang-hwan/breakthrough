# strategies/signal_generator.py

import pandas as pd # Pandas 라이브러리: 데이터 분석 및 처리 도구

def calculate_breakout_signals(
    df: pd.DataFrame,
    window: int = 20,
    vol_factor: float = 1.5,
    confirm_bars: int = 2,
    use_high: bool = False,
    breakout_buffer: float = 0.0
) -> pd.DataFrame:
    """
    데이터에서 전고점 돌파, 거래량 조건, 확정 돌파 신호를 계산하는 함수입니다.

    ------------------------------------------------------------------------
    매개변수 (Parameters):
    - df (DataFrame): 데이터를 포함한 DataFrame (열: 시가, 고가, 저가, 종가, 거래량 등)
    - window (int): 전고점과 평균 거래량 계산 기준이 되는 과거 데이터 기간 (기본값: 20)
    - vol_factor (float): 거래량이 과거 평균 대비 얼마나 높은지를 결정하는 배수 (기본값: 1.5)
    - confirm_bars (int): 돌파 신호가 몇 개의 연속 봉에서 발생해야 확정 돌파로 간주할지 (기본값: 2)
    - use_high (bool): True이면 고가 기준, False이면 종가 기준으로 돌파 여부 판단 (기본값: False)
    - breakout_buffer (float): 돌파 기준에 추가 여유값 (비율) (기본값: 0.0)
    ------------------------------------------------------------------------
    반환 (Return):
    - 추가 컬럼들이 포함된 DataFrame:
      - highest_xx: window 기간 동안의 전고점
      - breakout_signal: 돌파 신호(True/False)
      - volume_condition: 거래량 조건 만족 여부(True/False)
      - confirmed_breakout: 확정 돌파 신호(True/False)
    """

    # 1. 전고점(highest_xx) 계산
    # - window 기간 동안 고가(high)의 최댓값(전고점)을 구합니다.
    # - shift(1)를 통해 현재 봉 데이터를 제외하고 계산.
    df[f'highest_{window}'] = (
        df['high'].shift(1)  # 현재 봉 제외
        .rolling(window)    # 지정한 window 기간 동안 계산
        .max()              # 해당 기간 내 최대값(전고점)
    )

    # 2. 돌파 신호(breakout_signal) 계산
    # - 주가가 (전고점 + breakout_buffer)보다 높은 경우 돌파로 간주.
    # - use_high가 True일 경우 고가(high), False일 경우 종가(close)를 기준으로 계산.
    if use_high:
        df['breakout_signal'] = df['high'] > (
            df[f'highest_{window}'] * (1 + breakout_buffer)
        )
    else:
        df['breakout_signal'] = df['close'] > (
            df[f'highest_{window}'] * (1 + breakout_buffer)
        )

    # 3. 거래량 조건(volume_condition) 계산
    # - window 기간 동안 평균 거래량(vol_ma_xx)을 계산.
    # - 현재 거래량이 평균 거래량 * vol_factor보다 큰 경우 조건 만족(True).
    df[f'vol_ma_{window}'] = (
        df['volume'].shift(1)  # 현재 봉 제외
        .rolling(window)      # 지정된 window 기간 동안 평균값 계산
        .mean()               # 평균 거래량
    )
    df['volume_condition'] = df['volume'] > (vol_factor * df[f'vol_ma_{window}'])

    # 4. 확정 돌파(confirmed_breakout) 계산
    # - 돌파 신호가 confirm_bars 기간 동안 연속 발생하면 확정 돌파(True).
    # - rolling(confirm_bars)을 사용해 연속 신호 수를 합산.
    df['confirmed_breakout'] = (
        df['breakout_signal'].rolling(confirm_bars).sum() == confirm_bars
    )

    # NaN(초기 값 없음) 처리: False로 채움
    df['confirmed_breakout'] = df['confirmed_breakout'].fillna(False)

    return df
