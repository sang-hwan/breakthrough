# trade_logic.py

import pandas as pd
import numpy as np
import ta  # 'ta' 라이브러리는 여러 기술적 지표(ATR, RSI, MACD 등)를 제공

def apply_entry_signal(
    df: pd.DataFrame,
    entry_colname: str = 'long_entry'
) -> pd.DataFrame:
    """
    이전 단계에서 생성된 시그널(breakout_signal, volume_condition, confirmed_breakout)을 종합하여,
    최종 매수 여부(long_entry)를 확정하는 로직을 구현한 함수입니다.
    
    매개변수
    ----------
    df : pd.DataFrame
        시가, 고가, 저가, 종가, 거래량, 및 돌파 관련 시그널 등이 포함된 DataFrame
    entry_colname : str
        매수 여부를 저장할 컬럼명 (기본값: 'long_entry')
    
    반환값
    ----------
    pd.DataFrame
        원본 DataFrame에 'long_entry' 컬럼이 추가된 상태
    """
    # 예시 로직:
    # 전고점 돌파, 거래량 조건, 확정 돌파가 모두 True이면 매수 신호를 True로 설정
    df[entry_colname] = (
        (df['breakout_signal'] == True) &    # 돌파 여부
        (df['volume_condition'] == True) &   # 거래량 조건
        (df['confirmed_breakout'] == True)   # 확정 돌파
    )
    return df


def apply_stop_loss_atr(
    df: pd.DataFrame,
    atr_window: int = 14,
    atr_multiplier: float = 2.0,
    sl_colname: str = 'stop_loss_price',
    entry_price_col: str = 'entry_price'
) -> pd.DataFrame:
    """
    ATR(평균진폭)을 활용하여 손절가(stop_loss_price)를 DataFrame에 추가하는 함수입니다.
    실제 매매에서는 '진입 가격'에 기반해 손절가를 설정하는 경우가 많습니다.
    
    매개변수
    ----------
    df : pd.DataFrame
        시가, 고가, 저가, 종가 등의 기본 정보와 매수신호(long_entry) 등이 포함된 DataFrame
    atr_window : int
        ATR 계산에 사용하는 봉(캔들)의 개수 (기본값: 14)
    atr_multiplier : float
        ATR에 곱해줄 배수 (기본값: 2.0 -> '진입가 - 2*ATR' 형태의 손절가)
    sl_colname : str
        결과로 저장할 손절가 컬럼명 (기본값: 'stop_loss_price')
    entry_price_col : str
        진입가(매수가)를 저장/유지할 컬럼명 (기본값: 'entry_price')

    반환값
    ----------
    pd.DataFrame
        원본 DataFrame에 'atr', 'entry_price', 'stop_loss_price' 컬럼이 추가된 상태
    """
    # 1) ta 라이브러리의 AverageTrueRange 클래스를 이용해 ATR을 계산합니다.
    #    - high, low, close 컬럼과, atr_window(14)가 필요
    #    - fillna=True를 사용해 결측치가 생기지 않도록 처리
    atr_indicator = ta.volatility.AverageTrueRange(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=atr_window,
        fillna=True
    )
    
    # ATR 계산 결과를 'atr' 컬럼에 저장
    df['atr'] = atr_indicator.average_true_range()
    
    # 2) entry_price: 진입 시점(= long_entry가 True인 곳)의 종가를 저장
    #    - np.where를 이용해 매수신호(True)인 지점에서는 'close'를, 
    #      아니라면 np.nan을 저장
    df[entry_price_col] = np.where(df['long_entry'], df['close'], np.nan)
    
    # 3) forward fill: 한 번 진입한 뒤에는 별도 청산 시점이 오기 전까지 
    #    같은 entry_price를 유지(단순 예시).
    #    => 이렇게 해서 전체 구간에 걸쳐 진입가가 '계속' 기록됨
    df[entry_price_col] = df[entry_price_col].ffill()
    
    # 4) 손절가 계산: 
    #    예) 손절가 = entry_price - (atr * atr_multiplier)
    df[sl_colname] = df[entry_price_col] - (df['atr'] * atr_multiplier)
    
    return df


def apply_take_profit_ratio(
    df: pd.DataFrame,
    profit_ratio: float = 0.05,
    tp_colname: str = 'take_profit_price',
    entry_price_col: str = 'entry_price'
) -> pd.DataFrame:
    """
    고정된 이익률( profit_ratio )을 사용하여 익절가(take_profit_price)를 계산하는 함수입니다.
    
    매개변수
    ----------
    df : pd.DataFrame
        매수 시그널과 진입가가 포함된 DataFrame
    profit_ratio : float
        목표 수익률 (기본값: 0.05 => 5% 수익에 익절)
    tp_colname : str
        익절가를 저장할 컬럼명
    entry_price_col : str
        진입가가 기록된 컬럼명
    
    반환값
    ----------
    pd.DataFrame
        원본 DataFrame에 익절가(take_profit_price)가 추가된 상태
    """
    # 익절가 = 진입가 * (1 + 목표 수익률)
    df[tp_colname] = df[entry_price_col] * (1 + profit_ratio)
    return df


def generate_trade_signals(
    df: pd.DataFrame,
    atr_window: int = 14,
    atr_multiplier: float = 2.0,
    profit_ratio: float = 0.05,
) -> pd.DataFrame:
    """
    종합 매매 로직을 한 번에 수행하는 함수입니다.
    1) apply_entry_signal : 매수 신호(long_entry) 확정
    2) apply_stop_loss_atr: ATR 기반 손절가(stop_loss_price) 계산
    3) apply_take_profit_ratio: 고정 이익률 기반 익절가(take_profit_price) 계산
    
    매개변수
    ----------
    df : pd.DataFrame
        돌파, 거래량, 확정 돌파 등이 포함된 DataFrame
    atr_window : int
        ATR 계산 시 사용할 봉 개수 (기본값: 14)
    atr_multiplier : float
        ATR에 곱해줄 배수 (기본값: 2.0)
    profit_ratio : float
        익절 목표 수익률 (기본값: 0.05 => 5% 수익 시 익절)
    
    반환값
    ----------
    pd.DataFrame
        원본 DataFrame에 매수 신호, 손절가, 익절가 등이 추가된 상태
    """
    # 1) 매수 신호 확정
    df = apply_entry_signal(df, entry_colname='long_entry')
    
    # 2) ATR 손절가 계산
    df = apply_stop_loss_atr(
        df,
        atr_window=atr_window,
        atr_multiplier=atr_multiplier,
        sl_colname='stop_loss_price',
        entry_price_col='entry_price'
    )
    
    # 3) 고정 이익률 기반 익절가 계산
    df = apply_take_profit_ratio(
        df,
        profit_ratio=profit_ratio,
        tp_colname='take_profit_price',
        entry_price_col='entry_price'
    )
    
    return df
