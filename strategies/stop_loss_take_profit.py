# strategies/stop_loss_take_profit.py
# ATR 기반 손절가 설정, 고정익절가, 트레일링 스탑 로직 등을 담은 파일입니다.

import pandas as pd
import numpy as np
import ta  # 'ta' 라이브러리를 이용해 ATR 등을 계산


def apply_stop_loss_atr(
    df: pd.DataFrame,
    atr_window: int = 14,
    atr_multiplier: float = 2.0,
    sl_colname: str = 'stop_loss_price',
    entry_price_col: str = 'entry_price'
) -> pd.DataFrame:
    """
    ATR(평균진폭)을 활용하여, 매수 시점마다 손절가를 설정합니다.
    - 진입가 - (ATR * 배수) 식으로 계산
    """

    # ta 라이브러리로 ATR(최근 봉들의 변동폭 평균)을 구함
    atr_indicator = ta.volatility.AverageTrueRange(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=atr_window,
        fillna=True
    )
    df['atr'] = atr_indicator.average_true_range()

    # 매수 시점의 종가를 entry_price로 기록
    df[entry_price_col] = np.where(df['long_entry'], df['close'], np.nan)
    df[entry_price_col] = df[entry_price_col].ffill()  # 진입 후엔 계속 유지

    # 손절가 = 진입가 - (ATR * 배수)
    df[sl_colname] = df[entry_price_col] - (df['atr'] * atr_multiplier)

    return df

def update_trailing_stop(current_stop_loss: float, current_price: float, 
                         highest_price: float, trailing_percent: float) -> float:
    """
    트레일링 스탑: 상승장에서 최고가가 갱신될 때마다 손절가도 따라올림.
    - 예: 최고가의 5% 아래로 손절가를 계속 끌어올린다.
    """

    # highest_price 대비 trailing_percent만큼 내려온 지점
    new_stop = highest_price * (1.0 - trailing_percent)

    # 기존 손절가보다 높게만 업데이트(손절가가 내려가는 일은 없음), 그리고 현재가보단 낮아야 함
    if new_stop > current_stop_loss and new_stop < current_price:
        return new_stop
    else:
        return current_stop_loss


def apply_take_profit_ratio(
    df: pd.DataFrame,
    profit_ratio: float = 0.05,
    tp_colname: str = 'take_profit_price',
    entry_price_col: str = 'entry_price'
) -> pd.DataFrame:
    """
    고정된 이익률(예: 5%)을 목표로 익절가를 설정합니다.
    - 익절가 = 진입가 * (1 + profit_ratio)
    """

    df[tp_colname] = df[entry_price_col] * (1 + profit_ratio)
    return df


def check_trend_exit_condition(
    df_long: pd.DataFrame,
    current_time,
    sma_col: str = 'sma200'
) -> bool:
    """
    간단하게 '종가 < SMA(200)이면 추세 이탈' 로 가정하고 True/False를 반환하는 예시.
    """

    # current_time이 df_long 인덱스에 없을 수 있으므로 처리
    if current_time not in df_long.index:
        df_sub = df_long.loc[:current_time]
        if df_sub.empty:
            return False
        row_l = df_sub.iloc[-1]
    else:
        row_l = df_long.loc[current_time]

    close_price = row_l['close']
    sma_val = row_l[sma_col] if sma_col in row_l else np.nan

    if pd.notna(sma_val) and close_price < sma_val:
        return True
    return False


def create_sub_tps_for_partial_exit(
    entry_price: float,
    partial_ratio: float = 0.5,
    partial_tp_factor: float = 0.03,  # 3% 익절
    final_tp_factor: float = 0.06     # 6% 익절
):
    """
    예시:
      entry_price=10000,
      partial_tp_factor=0.03 => 10,300 달성 시 50%(partial_ratio=0.5) 익절
      final_tp_factor=0.06   => 10,600 달성 시 나머지 50% 익절
    """
    partial_tp_price = entry_price * (1.0 + partial_tp_factor)
    final_tp_price   = entry_price * (1.0 + final_tp_factor)

    return [
        (partial_tp_price, partial_ratio),  # 첫 익절
        (final_tp_price, 1.0)              # 나머지 전량 익절
    ]