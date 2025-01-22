# backtesting/verify_signals.py

import matplotlib.pyplot as plt # 시각화를 위한 matplotlib
import pandas as pd

# 데이터베이스 및 시그널 계산 함수
from data_collection.save_to_postgres import load_ohlcv_from_postgres
from strategies.signal_generator import calculate_breakout_signals

def verify_breakout_signals(
    symbol: str,
    timeframe: str,
    table_name: str,
    window: int = 20,
    vol_factor: float = 1.5,
    confirm_bars: int = 2,
    use_high: bool = False,
    breakout_buffer: float = 0.0,
    plot_chart: bool = False
):
    """
    돌파 시그널 검증 및 시각화 함수.

    주요 기능:
    ----------
    1. PostgreSQL에서 (symbol, timeframe)에 해당하는 데이터를 로드.
    2. 돌파 시그널(전고점, 거래량, 확정 돌파)을 계산.
    3. 시그널이 의도대로 생성되었는지 수치적으로 검증.
    4. (선택적) 차트를 통해 시그널을 시각적으로 확인.

    매개변수:
    ----------
    - symbol (str): 자산 심볼 (예: "BTC/USDT")
    - timeframe (str): 데이터의 시간 간격 (예: "4h")
    - table_name (str): 데이터베이스 테이블명
    - window (int): 전고점 계산 기간
    - vol_factor (float): 거래량 필터 배수
    - confirm_bars (int): 확정 돌파를 위한 봉 개수
    - use_high (bool): True면 고가 기준 돌파, False면 종가 기준 돌파
    - breakout_buffer (float): 돌파 기준에 추가할 버퍼 비율
    - plot_chart (bool): True면 차트로 결과 시각화

    반환값:
    ----------
    None
    """

    print(f"\n[Verification] symbol={symbol}, timeframe={timeframe}, table={table_name}")

    # 1. 데이터 로드
    df = load_ohlcv_from_postgres(table_name=table_name)
    if df.empty:
        print(f"  -> No data loaded from table: {table_name}")
        return

    # 2. 돌파 시그널 계산
    df = calculate_breakout_signals(
        df=df,
        window=window,
        vol_factor=vol_factor,
        confirm_bars=confirm_bars,
        use_high=use_high,
        breakout_buffer=breakout_buffer
    )

    # -------------------------------
    # 3. 수치적 검증
    # -------------------------------

    # (A) breakout_signal 검증
    condition_breakout_true = (df['breakout_signal'] == True)
    error_rows = df[condition_breakout_true & 
                    (df['close'] <= df[f'highest_{window}'] * (1 + breakout_buffer))]
    if not error_rows.empty:
        print("[Warning] breakout_signal=True 이지만 실제로 돌파되지 않은 봉이 발견됨!")
        print(error_rows[['close', f'highest_{window}', 'breakout_signal']].head(10))
    else:
        print(" -> breakout_signal과 실제 가격 돌파가 일치합니다. (수치상 오차 없음)")

    # (B) volume_condition 검증
    cond_vol_true = (df['volume_condition'] == True)
    error_vol = df[cond_vol_true & (df['volume'] <= df[f'vol_ma_{window}'] * vol_factor)]
    if not error_vol.empty:
        print("[Warning] volume_condition=True 이지만 실제 거래량 조건이 만족되지 않은 봉 발견!")
        print(error_vol[['volume', f'vol_ma_{window}', 'volume_condition']].head(10))
    else:
        print(" -> volume_condition과 실제 거래량 조건이 일치합니다. (수치상 오차 없음)")

    # -------------------------------
    # 4. 차트 시각화 (선택 사항)
    # -------------------------------
    if plot_chart:
        # 최근 300개 봉만 시각화 (데이터가 많을 경우 복잡해짐)
        df_plot = df.tail(300)

        plt.figure(figsize=(12, 6))
        plt.plot(df_plot.index, df_plot['close'], label='Close', color='blue')

        # 전고점(highest_xx) 표시
        plt.plot(df_plot.index, df_plot[f'highest_{window}'], label=f"Highest({window})", linestyle='--', color='orange')

        # breakout_signal이 True인 지점에 빨간 마커
        breakout_points = df_plot[df_plot['breakout_signal'] == True]
        plt.scatter(breakout_points.index, breakout_points['close'], color='red', marker='^', label='Breakout Signal')

        # confirmed_breakout이 True인 지점에 녹색 마커
        confirmed_points = df_plot[df_plot['confirmed_breakout'] == True]
        plt.scatter(confirmed_points.index, confirmed_points['close'], color='green', marker='o', label='Confirmed Breakout')

        plt.title(f"{symbol} {timeframe} - Breakout Signal Verification")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.show()