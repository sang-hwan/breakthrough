# strategies/stop_loss_take_profit.py

# 데이터 분석 및 수치 연산 라이브러리
import pandas as pd
import numpy as np
import ta  # 'ta'는 ATR, RSI 등 다양한 기술적 지표를 제공하는 라이브러리

def apply_stop_loss_atr(
    df: pd.DataFrame,
    atr_window: int = 14,
    atr_multiplier: float = 2.0,
    sl_colname: str = 'stop_loss_price',
    entry_price_col: str = 'entry_price'
) -> pd.DataFrame:
    """
    ATR(평균진폭)을 기반으로 손절가(stop_loss_price)를 계산하여 DataFrame에 추가합니다.
    
    ------------------------------------------------------------------------
    매개변수 (Parameters):
    - df (DataFrame): 매수 신호(long_entry)와 가격 정보(시가, 고가, 저가, 종가 등)가 포함된 데이터.
    - atr_window (int): ATR 계산에 사용할 과거 캔들 수. 기본값: 14
    - atr_multiplier (float): 손절가 계산 시 ATR에 곱할 배수. 기본값: 2.0
    - sl_colname (str): 계산된 손절가를 저장할 컬럼명. 기본값: 'stop_loss_price'
    - entry_price_col (str): 진입가(매수가)를 저장할 컬럼명. 기본값: 'entry_price'

    반환값 (Return):
    - DataFrame: 'atr', 'entry_price', 'stop_loss_price' 컬럼이 추가된 데이터프레임.
    """

    # 1) ATR(평균진폭) 계산
    # - AverageTrueRange: 고가(high), 저가(low), 종가(close)를 사용해 변동성을 계산.
    # - window: 계산 기준이 되는 캔들 수
    atr_indicator = ta.volatility.AverageTrueRange(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=atr_window,
        fillna=True  # 결측값 발생 방지
    )
    df['atr'] = atr_indicator.average_true_range()

    # 2) 매수 시점의 종가(entry_price) 기록
    # - 매수 신호(long_entry=True) 발생 시 종가(close)를 진입가로 설정.
    df[entry_price_col] = np.where(df['long_entry'], df['close'], np.nan)

    # 3) 진입가 유지 (forward fill)
    # - 진입 이후 익절/손절까지 동일한 진입가 유지.
    df[entry_price_col] = df[entry_price_col].ffill()

    # 4) 손절가(stop_loss_price) 계산
    # - 손절가 = 진입가 - (ATR × ATR 배수)
    df[sl_colname] = df[entry_price_col] - (df['atr'] * atr_multiplier)

    return df


def apply_take_profit_ratio(
    df: pd.DataFrame,
    profit_ratio: float = 0.05,
    tp_colname: str = 'take_profit_price',
    entry_price_col: str = 'entry_price'
) -> pd.DataFrame:
    """
    고정된 목표 수익률을 사용하여 익절가(take_profit_price)를 계산합니다.
    
    ------------------------------------------------------------------------
    매개변수 (Parameters):
    - df (DataFrame): 매수 신호와 진입가가 포함된 데이터프레임.
    - profit_ratio (float): 목표 수익률 (기본값: 0.05, 즉 5% 수익에 익절).
    - tp_colname (str): 계산된 익절가를 저장할 컬럼명.
    - entry_price_col (str): 진입가가 기록된 컬럼명.

    반환값 (Return):
    - DataFrame: 익절가(take_profit_price)가 추가된 데이터프레임.
    """

    # 익절가 계산: 진입가 × (1 + 목표 수익률)
    df[tp_colname] = df[entry_price_col] * (1 + profit_ratio)
    return df

def update_trailing_stop(current_stop_loss: float, current_price: float, 
                         highest_price: float, trailing_percent: float) -> float:
    """
    트레일링 스탑 로직:
      - 현재까지의 최고가(highest_price) 대비 trailing_percent 만큼 뒤로 따라오는 손절가를 설정.
      - 예) trailing_percent=0.05 → 5% 아래 위치로 손절가를 업데이트.
    Params
    ------
      - current_stop_loss: 기존 손절가
      - current_price: 현재가
      - highest_price: 지금까지의 최고가
      - trailing_percent: 트레일링 % (0.05 = 5%)
    Returns
    -------
      - 새로운 손절가
    """
    # 새로 계산된 트레일링 손절가
    new_stop = highest_price * (1.0 - trailing_percent)
    # 기존 손절가보다 더 높은 수준일 경우에만 갱신 (손절가가 낮아지지 않도록)
    if new_stop > current_stop_loss and new_stop < current_price:
        return new_stop
    else:
        return current_stop_loss

def check_trend_exit_condition(
    df_long: pd.DataFrame,
    current_time,
    sma_col: str = 'sma200'
) -> bool:
    """
    예시: 종가가 SMA(200) 아래로 떨어지면 추세 이탈로 보고 청산한다.
    실제론 MACD나 SuperTrend 등 다양한 지표로 수정 가능.
    """
    # 현재 시점의 장기봉 데이터(가장 최근) 가져오기
    if current_time not in df_long.index:
        # 만약 간격이 어긋나면, current_time 이전의 가장 가까운 시점으로 찾아볼 수도 있음
        df_sub = df_long.loc[:current_time]
        if df_sub.empty:
            return False
        row_l = df_sub.iloc[-1]
    else:
        row_l = df_long.loc[current_time]

    close_price = row_l['close']
    sma_val = row_l[sma_col] if sma_col in row_l else np.nan

    # 예) 종가 < sma(200)이면 추세 이탈 판단
    if pd.notna(sma_val) and close_price < sma_val:
        return True
    return False