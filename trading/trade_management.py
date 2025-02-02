# trading/trade_management.py
import pandas as pd
import numpy as np
import ta

def calculate_atr_stop_loss(
    data: pd.DataFrame,
    atr_period: int = 14,
    atr_factor: float = 2.0,
    stop_loss_col: str = 'stop_loss_price',
    entry_price_col: str = 'entry_price'
) -> pd.DataFrame:
    """
    ATR 기반 손절가 계산.
    손절가 = entry_price - (ATR * atr_factor)
    """
    atr_indicator = ta.volatility.AverageTrueRange(
        high=data['high'],
        low=data['low'],
        close=data['close'],
        window=atr_period,
        fillna=True
    )
    data['atr'] = atr_indicator.average_true_range()
    data[entry_price_col] = np.where(data.get('long_entry', False), data['close'], np.nan)
    data[entry_price_col] = data[entry_price_col].ffill()
    data[stop_loss_col] = data[entry_price_col] - (data['atr'] * atr_factor)
    return data

def adjust_trailing_stop(
    current_stop: float,
    current_price: float,
    highest_price: float,
    trailing_percentage: float
) -> float:
    """
    트레일링 스탑 업데이트:
      새로운 손절가는 최고가 대비 trailing_percentage만큼 낮은 가격.
      단, 업데이트된 손절가는 기존 손절가보다 높고 현재 가격보다는 낮아야 함.
    """
    new_stop = highest_price * (1.0 - trailing_percentage)
    if new_stop > current_stop and new_stop < current_price:
        return new_stop
    else:
        return current_stop

def set_fixed_take_profit(
    data: pd.DataFrame,
    profit_ratio: float = 0.05,
    take_profit_col: str = 'take_profit_price',
    entry_price_col: str = 'entry_price'
) -> pd.DataFrame:
    """
    고정 수익률 기반 익절가 설정:
      take_profit = entry_price * (1 + profit_ratio)
    """
    data[take_profit_col] = data[entry_price_col] * (1 + profit_ratio)
    return data

def should_exit_trend(
    data: pd.DataFrame,
    current_time,
    window_length: int = 20,
    price_column: str = 'close'
) -> bool:
    """
    추세 종료(청산) 신호 판단 함수.
    - 최근 window_length 봉의 최저가보다 현재 종가가 낮으면 추세가 꺾였다고 판단.
    """
    if current_time not in data.index:
        data_sub = data.loc[:current_time]
        if len(data_sub) < window_length:
            return False
        window_data = data_sub.iloc[-window_length:]
        current_row = data_sub.iloc[-1]
    else:
        idx = data.index.get_loc(current_time)
        if idx < window_length:
            return False
        window_data = data.iloc[idx - window_length + 1: idx + 1]
        current_row = data.loc[current_time]
    
    recent_min = window_data[price_column].min()
    current_price = current_row[price_column]
    
    return current_price < recent_min

def calculate_partial_exit_targets(
    entry_price: float,
    partial_exit_ratio: float = 0.5,
    partial_profit_ratio: float = 0.03,  # 예: 3% 상승 시 일부 청산
    final_profit_ratio: float = 0.06     # 예: 6% 상승 시 전량 청산
):
    """
    분할 청산 목표가 계산 함수.
    예) entry_price=10000 인 경우:
         - partial_profit_ratio=0.03 → 10,300 달성 시 50% 청산
         - final_profit_ratio=0.06   → 10,600 달성 시 나머지 전량 청산
    반환: [(target_price, exit_ratio), ...]
    """
    partial_target = entry_price * (1.0 + partial_profit_ratio)
    final_target = entry_price * (1.0 + final_profit_ratio)
    return [
        (partial_target, partial_exit_ratio),
        (final_target, 1.0)
    ]

def calculate_fibonacci_take_profit(
    entry_price: float,
    recent_high: float,
    recent_low: float,
    levels: list = [0.382, 0.618, 1.0]
):
    """
    피보나치 되돌림 기반으로 익절 목표가를 계산합니다.
    Parameters:
      - entry_price: 진입 가격
      - recent_high: 최근 고가 (지지/저항 분석에 활용)
      - recent_low: 최근 저가
      - levels: 사용할 피보나치 비율 리스트
    Returns:
      - [(target_price, exit_ratio), ...] 형식의 리스트.
        예를 들어, 첫 목표는 50% 청산, 이후 전량 청산 등으로 설정.
    """
    range_price = recent_high - recent_low
    targets = []
    for idx, level in enumerate(levels):
        target = recent_low + range_price * (1 + level)
        exit_ratio = 0.5 if idx < len(levels) - 1 else 1.0
        targets.append((target, exit_ratio))
    return targets
