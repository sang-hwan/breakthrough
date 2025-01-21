# strategies/signal_generator.py

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
    전고점(rolling max) 돌파, 거래량 필터, 확정 돌파 시그널을 계산하는 함수입니다.
    ------------------------------------------------------------------------
    매개변수(Parameter)
    - df: 시가, 고가, 저가, 종가, 거래량 등의 정보를 담고 있는 DataFrame
    - window: '전고점' 및 '평균 거래량' 계산 시 사용할 이동 윈도우 크기 (기본값: 20)
    - vol_factor: 돌파 시 거래량이 얼마만큼(배수) 증가했는지 확인하기 위한 배수 (기본값: 1.5)
    - confirm_bars: 돌파 신호가 연속으로 몇 개의 봉에서 이어져야 '확정 돌파'로 볼지 설정 (기본값: 2)
    - use_high: True면 고가 기준, False면 종가 기준으로 돌파 판단 (기본값: False)
    - breakout_buffer: 돌파 시 (전고점 + 버퍼)의 형태로 추가 여유를 둘 때 사용 (기본값: 0.0)
    ------------------------------------------------------------------------
    반환(Return)
    - 원본 DataFrame(df)에 돌파 신호 컬럼들이 추가된 DataFrame.
      (highest_xx, breakout_signal, volume_condition, confirmed_breakout 등)
    """

    # 1) 전고점(rolling max) 계산: 과거 window개의 'high' 값 중 최댓값
    #    - 예: window=20이면 지난 20봉(캔들) 동안 가장 높았던 고가
    df[f'highest_{window}'] = df['high'].rolling(window).max()
    
    # 2) 돌파 시그널 계산: 종가 or 고가가 '전고점 + 버퍼'보다 높은지 확인
    if use_high:
        # use_high=True일 경우: 'high(고가)' 기준으로 돌파 여부 확인
        df['breakout_signal'] = df['high'] > (df[f'highest_{window}'] + df[f'highest_{window}'] * breakout_buffer)
    else:
        # use_high=False일 경우: 'close(종가)' 기준으로 돌파 여부 확인
        df['breakout_signal'] = df['close'] > (df[f'highest_{window}'] + df[f'highest_{window}'] * breakout_buffer)
    
    # 3) 거래량 조건: 과거 window봉의 평균 거래량 대비 vol_factor(배수) 이상인지 체크
    #    - 예: window=20, vol_factor=1.5 => 최근 20봉 평균 거래량의 1.5배 이상인지
    df[f'vol_ma_{window}'] = df['volume'].rolling(window).mean()  # 거래량 이동평균
    df['volume_condition'] = df['volume'] > (vol_factor * df[f'vol_ma_{window}'])
    
    # 4) '확정 돌파(confirmed_breakout)' 계산
    #    - confirm_bars: 연속적인 봉의 개수로, 돌파 신호(breakout_signal)가 해당 봉 수만큼 True일 때 확정 돌파로 간주
    #    - rolling(confirm_bars): 지정된 봉 수(confirm_bars)에 대한 rolling 합계를 계산하여 True의 연속성을 확인
    #    - sum() == confirm_bars: 지정된 연속 봉 동안 모든 값이 True일 경우 확정 돌파(True)로 설정
    df['confirmed_breakout'] = (
        df['breakout_signal'].rolling(confirm_bars).sum() == confirm_bars
    )
    # 초기 데이터 부족으로 인해 NaN 값이 포함된 구간을 False로 채움
    df['confirmed_breakout'] = df['confirmed_breakout'].fillna(False)
    
    return df