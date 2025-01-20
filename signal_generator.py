# signal_generator.py

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
    #    - 예: confirm_bars=2 => 돌파 신호(breakout_signal)가 연속 2봉 이상 True 이어야 확정으로 판정
    df['confirmed_breakout'] = (
        df['breakout_signal']
        .rolling(confirm_bars)               # 최근 confirm_bars 봉 범위로 rolling
        .apply(lambda x: all(x), raw=True)   # 해당 범위 내의 값이 모두 True인지 확인
        .fillna(False)                       # 첫 부분(rolling 불충분 구간)은 NaN이므로 False 처리
    )
    
    return df


def calculate_vcp_pattern(
    df: pd.DataFrame,
    window_list: list = [20, 10, 5]
) -> pd.DataFrame:
    """
    VCP(Volatility Contraction Pattern) 패턴을 단순 계산하는 함수입니다.
    ------------------------------------------------------------------------
    매개변수(Parameter)
    - df: 시가, 고가, 저가, 종가, 거래량 등의 정보를 담고 있는 DataFrame
    - window_list: 변동 폭(고가-저가)의 이동평균을 계산할 기간을 담은 리스트 (예: [20, 10, 5])
    ------------------------------------------------------------------------
    반환(Return)
    - 원본 DataFrame(df)에 VCP 관련 컬럼이 추가된 DataFrame.
      (예: range_ma_20, range_ma_10, range_ma_5, vcp_signal 등)
    """

    # 1) 각 window마다 (고가 - 저가)의 이동평균을 구함
    #    예: window=20 -> 최근 20봉의 (high - low)의 평균 (변동 폭 평균)
    for w in window_list:
        df[f'range_ma_{w}'] = (df['high'] - df['low']).rolling(w).mean()
    
    # 2) (단순한 예시)
    #    - ex) window_list = [20, 10, 5]
    #    - 세 구간의 변동 폭 평균이 큰 순서대로(20봉 > 10봉 > 5봉) '연속 감소'한다면 VCP로 판정
    w1, w2, w3 = window_list
    df['vcp_signal'] = (
        (df[f'range_ma_{w1}'] > df[f'range_ma_{w2}']) &
        (df[f'range_ma_{w2}'] > df[f'range_ma_{w3}'])
    )
    
    return df
