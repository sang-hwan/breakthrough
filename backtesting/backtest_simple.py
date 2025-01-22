# backtesting/backtest_simple.py

# 데이터 분석 및 수치 계산을 위한 라이브러리
import pandas as pd
import numpy as np

# 필요한 함수와 모듈 불러오기
from data_collection.save_to_postgres import load_ohlcv_from_postgres
from strategies.signal_generator import calculate_breakout_signals
from strategies.stop_loss_take_profit import apply_stop_loss_atr, apply_take_profit_ratio
from strategies.risk_management import calculate_position_size
from backtesting.performance_metrics import print_performance_report

def run_simple_backtest(
    symbol="BTC/USDT",
    timeframe="4h",
    window=20,
    volume_factor=1.5,
    confirm_bars=2,
    breakout_buffer=0.0,
    atr_window=14,
    atr_multiplier=2.0,
    profit_ratio=0.05,
    account_size=10_000.0,
    risk_per_trade=0.01,
    fee_rate=0.001
):
    """
    간단한 백테스트를 실행하는 함수입니다.

    주요 기능:
    ----------
    1. 과거 데이터를 로드하여 백테스트를 실행합니다.
    2. 돌파 신호와 거래량 조건을 통해 매수 시점을 결정합니다.
    3. ATR 기반 손절 및 고정 익절 전략을 적용합니다.
    4. 백테스트 결과를 계산하고, 성과를 출력합니다.

    매개변수:
    ----------
    - symbol (str): 테스트할 자산의 심볼 (예: "BTC/USDT")
    - timeframe (str): 데이터의 시간 간격 (예: "4h")
    - window (int): 전고점을 계산하는 기간
    - volume_factor (float): 거래량 필터 배수
    - confirm_bars (int): 돌파 확정을 위한 봉 개수
    - breakout_buffer (float): 돌파 기준에 추가할 버퍼
    - atr_window (int): ATR 계산 기간
    - atr_multiplier (float): ATR 기반 손절 배수
    - profit_ratio (float): 고정 익절 비율
    - account_size (float): 초기 계좌 자산
    - risk_per_trade (float): 매매당 최대 손실 비율
    - fee_rate (float): 매수 수수료 비율

    반환값:
    ----------
    - DataFrame: 각 매매에 대한 세부 정보가 담긴 DataFrame
    """

    # -------------------------------
    # 1) 과거 데이터 불러오기
    # -------------------------------
    table_name = f"ohlcv_{symbol.replace('/', '').lower()}_{timeframe}"
    df = load_ohlcv_from_postgres(table_name=table_name)
    df = df.sort_index()  # 시간순 정렬

    if df.empty:
        print("DataFrame is empty! 데이터가 없습니다.")
        return None

    print(f"Loaded data from {table_name}: {df.shape[0]} rows")

    # -------------------------------
    # 2) 돌파 시그널 계산
    # -------------------------------
    df = calculate_breakout_signals(
        df=df,
        window=window,
        vol_factor=volume_factor,
        confirm_bars=confirm_bars,
        use_high=False,
        breakout_buffer=breakout_buffer
    )

    # 돌파 신호와 관련된 통계를 출력
    print("\nSignal Stats:")
    print(f"Breakout signals: {df['breakout_signal'].sum()}")
    print(f"Confirmed breakouts: {df['confirmed_breakout'].sum()}")

    # -------------------------------
    # 3) ATR 손절 및 고정 익절
    # -------------------------------
    df['long_entry'] = df['confirmed_breakout'] & df['volume_condition']

    df = apply_stop_loss_atr(
        df=df,
        atr_window=atr_window,
        atr_multiplier=atr_multiplier,
        sl_colname='stop_loss_price',
        entry_price_col='entry_price'
    )

    df = apply_take_profit_ratio(
        df=df,
        profit_ratio=profit_ratio,
        tp_colname='take_profit_price',
        entry_price_col='entry_price'
    )

    # -------------------------------
    # 4) 백테스트 루프 실행
    # -------------------------------
    in_position = False
    trades = []

    for i in range(len(df)):
        row = df.iloc[i]
        if not in_position and row['long_entry']:
            # 포지션 진입
            entry_price = row['entry_price']
            stop_loss = row['stop_loss_price']
            take_profit = row['take_profit_price']
            size = calculate_position_size(
                account_balance=account_size,
                risk_per_trade=risk_per_trade,
                entry_price=entry_price,
                stop_loss_price=stop_loss,
                fee_rate=fee_rate
            )
            in_position = True
            trades.append({
                'entry_time': row.name,
                'entry_price': entry_price,
                'size': size,
                'exit_time': None,
                'exit_price': None,
                'pnl': None,
                'exit_reason': None
            })
        elif in_position:
            # 포지션 청산 조건
            if row['close'] <= stop_loss:
                in_position = False
                trades[-1].update({
                    'exit_time': row.name,
                    'exit_price': row['close'],
                    'pnl': (row['close'] - trades[-1]['entry_price']) * trades[-1]['size'],
                    'exit_reason': 'stop_loss'
                })
            elif row['close'] >= take_profit:
                in_position = False
                trades[-1].update({
                    'exit_time': row.name,
                    'exit_price': row['close'],
                    'pnl': (row['close'] - trades[-1]['entry_price']) * trades[-1]['size'],
                    'exit_reason': 'take_profit'
                })

    # -------------------------------
    # 5) 결과 정리 및 성과 출력
    # -------------------------------
    trades_df = pd.DataFrame(trades)
    trades_df.dropna(subset=['exit_time'], inplace=True)

    if trades_df.empty:
        print("No trades were completed.")
        return None

    total_pnl = trades_df['pnl'].sum()
    final_balance = account_size + total_pnl
    roi_percent = (final_balance - account_size) / account_size * 100.0

    print(f"Total Trades: {len(trades_df)}")
    print(f"Total PnL: {total_pnl:.2f} USDT")
    print(f"Final Balance: {final_balance:.2f} USDT")
    print(f"ROI: {roi_percent:.2f}%")

    print_performance_report(trades_df, initial_balance=account_size)

    return trades_df
