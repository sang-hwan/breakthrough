# backtesting/backtest_advanced.py

import pandas as pd
import numpy as np

from data_collection.postgres_ohlcv_handler import load_ohlcv_from_postgres
from strategies.breakout_signal import calculate_breakout_signals
from strategies.technical_indicators import (
    apply_sma, apply_macd, apply_rsi, apply_bollinger
)
from strategies.stop_loss_take_profit import (
    apply_stop_loss_atr,
    apply_take_profit_ratio,
    update_trailing_stop,
    check_trend_exit_condition
)
from strategies.risk_management import (
    calculate_position_size,
    split_position_sizes,
    add_position_sizes
)
from strategies.position import Position
from backtesting.performance_metrics import print_performance_report


def run_advanced_backtest(
    symbol="BTC/USDT",
    short_timeframe="4h",
    long_timeframe="1d",
    window=20,
    volume_factor=1.5,
    confirm_bars=2,
    breakout_buffer=0.0,
    atr_window=14,
    atr_multiplier=2.0,
    profit_ratio=0.05,
    account_size=10_000.0,
    risk_per_trade=0.01,
    fee_rate=0.001,
    slippage_rate=0.0005,
    taker_fee_rate=0.001,

    # 전체 분할매수(피라미딩) 횟수 (ex. 3 -> 총 3분할)
    total_splits=3,
    threshold_percent=0.02,  # +2% 마다 다음 분할 매수

    use_trailing_stop=False,
    trailing_percent=0.05,
    use_trend_exit=False,
    start_date=None,
    end_date=None
):
    """
    split_position_sizes + add_position_sizes를 활용한 예시
    """

    # 1) 데이터 로드
    short_table = f"ohlcv_{symbol.replace('/', '').lower()}_{short_timeframe}"
    long_table = f"ohlcv_{symbol.replace('/', '').lower()}_{long_timeframe}"

    df_short = load_ohlcv_from_postgres(short_table, start_date, end_date)
    df_long = load_ohlcv_from_postgres(long_table, start_date, end_date)

    if df_short.empty or df_long.empty:
        print("[ERROR] No data loaded.")
        return None

    df_short.sort_index(inplace=True)
    df_long.sort_index(inplace=True)

    # 2) 시그널 계산
    df_short = calculate_breakout_signals(
        df_short,
        window=window,
        vol_factor=volume_factor,
        confirm_bars=confirm_bars,
        use_high=False,
        breakout_buffer=breakout_buffer
    )
    df_short['long_entry'] = df_short['confirmed_breakout'] & df_short['volume_condition']

    df_short = apply_stop_loss_atr(df_short, atr_window, atr_multiplier,
                                   'stop_loss_price', 'entry_price')
    df_short = apply_take_profit_ratio(df_short, profit_ratio,
                                       'take_profit_price', 'entry_price')

    df_long = apply_sma(df_long, 'close', 200, 'sma200')
    df_long = apply_macd(df_long, 'close', 26, 12, 9)
    df_long = apply_rsi(df_long, 'close', 14)
    df_long = apply_bollinger(df_long, 'close', 20, 2.0)
    df_long['long_ok'] = (
        (df_long['close'] >= df_long['sma200']) &
        (df_long['macd_diff'] > 0) &
        (df_long['rsi'] < 70) &
        (df_long['bb_hband'] > df_long['close'])
    )

    # 3) 백테스트
    current_position = None
    trades = []
    highest_price_since_entry = None

    for i in range(len(df_short)):
        row = df_short.iloc[i]
        current_time = row.name
        close_price = row['close']
        high_price = row['high']
        low_price = row['low']

        # (A) 보유 포지션 처리
        if current_position is not None and not current_position.is_empty():
            # 손절/익절 처리
            sub_positions_to_close = []
            for idx, sp in enumerate(current_position.sub_positions):
                ep = sp['entry_price']
                sl = sp['stop_loss']
                tp = sp['take_profit']
                size = sp['size']

                triggered_stop = False
                triggered_take = False
                exit_price = None
                exit_reason = None

                if low_price <= sl:
                    triggered_stop = True
                    exit_price = sl * (1 - slippage_rate)
                    exit_reason = 'stop_loss'
                elif high_price >= tp:
                    triggered_take = True
                    exit_price = tp * (1 - slippage_rate)
                    exit_reason = 'take_profit'

                if triggered_stop or triggered_take:
                    fee = exit_price * size * taker_fee_rate
                    pnl = (exit_price - ep) * size - fee
                    trades.append({
                        'entry_time': sp['entry_time'],
                        'entry_price': ep,
                        'exit_time': current_time,
                        'exit_price': exit_price,
                        'size': size,
                        'pnl': pnl,
                        'reason': exit_reason
                    })
                    sub_positions_to_close.append(idx)

            for idx_close in reversed(sub_positions_to_close):
                current_position.close_sub_position(idx_close)

            # 트레일링 스탑
            if use_trailing_stop and current_position.side == "LONG":
                if highest_price_since_entry is None or high_price > highest_price_since_entry:
                    highest_price_since_entry = high_price
                for sp in current_position.sub_positions:
                    old_sl = sp['stop_loss']
                    new_sl = update_trailing_stop(
                        current_stop_loss=old_sl,
                        current_price=close_price,
                        highest_price=highest_price_since_entry,
                        trailing_percent=trailing_percent
                    )
                    sp['stop_loss'] = new_sl

            # 추세 이탈
            if use_trend_exit:
                if check_trend_exit_condition(df_long, current_time, 'sma200'):
                    # 모두 청산
                    for sp2 in current_position.sub_positions:
                        ep2 = sp2['entry_price']
                        sz2 = sp2['size']
                        exit_price = close_price * (1 - slippage_rate)
                        fee2 = exit_price * sz2 * taker_fee_rate
                        pnl2 = (exit_price - ep2) * sz2 - fee2
                        trades.append({
                            'entry_time': sp2['entry_time'],
                            'entry_price': ep2,
                            'exit_time': current_time,
                            'exit_price': exit_price,
                            'size': sz2,
                            'pnl': pnl2,
                            'reason': 'trend_exit'
                        })
                    current_position = None
                    highest_price_since_entry = None

            # 포지션 청산됐으면 끝
            if current_position and current_position.is_empty():
                current_position = None
                highest_price_since_entry = None

            # (A-1) 남은 분할 자동매수
            if current_position and not current_position.is_empty():
                add_position_sizes(
                    position=current_position,
                    current_price=close_price,
                    threshold_percent=threshold_percent,
                    slippage_rate=slippage_rate,
                    stop_loss_price=row['stop_loss_price'],
                    take_profit_price=row['take_profit_price'],
                    entry_time=current_time
                )

        # (B) 새 진입 시그널
        if row['long_entry']:
            # 장기 필터
            df_long_sub = df_long.loc[:current_time]
            if not df_long_sub.empty:
                row_l = df_long_sub.iloc[-1]
                if row_l['long_ok']:
                    # 포지션이 없으면 새로 생성
                    if current_position is None:
                        current_position = Position(
                            side="LONG",
                            initial_entry_price=close_price,  # 기준가
                            max_position_size=0.0,            # 뒤에서 할당
                            num_splits=total_splits,
                            split_plan=[]                     # 뒤에서 할당
                        )
                        highest_price_since_entry = None

                    # 리스크 기반으로 전체 수량 계산
                    entry_price_for_calc = close_price * (1 + slippage_rate)
                    total_size = calculate_position_size(
                        account_balance=account_size,
                        risk_per_trade=risk_per_trade,
                        entry_price=entry_price_for_calc,
                        stop_loss_price=row['stop_loss_price'],
                        fee_rate=taker_fee_rate
                    )
                    current_position.max_position_size = total_size

                    # split_position_sizes로 분할비중 리스트 생성
                    plan_list = split_position_sizes(
                        total_position_size=1.0,  # 비중(=1) 기준
                        split_count=total_splits,
                        scale_mode='equal'        # 자유롭게
                    )
                    current_position.split_plan = plan_list

                    # 첫 분할 체결( splits_filled=0 에서 0번 인덱스 매수 )
                    if total_splits > 0:
                        # size = total_size * plan_list[0]
                        buy_size = total_size * plan_list[0]
                        buy_price = close_price * (1 + slippage_rate)

                        current_position.add_sub_position(
                            entry_price=buy_price,
                            size=buy_size,
                            stop_loss=row['stop_loss_price'],
                            take_profit=row['take_profit_price'],
                            entry_time=current_time
                        )
                        # 첫 분할 체결 → splits_filled=1
                        current_position.splits_filled = 1

                    if highest_price_since_entry is None:
                        highest_price_since_entry = close_price

    # 4) 결과
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        print("[INFO] No trades executed.")
        return None

    total_pnl = trades_df['pnl'].sum()
    final_balance = account_size + total_pnl
    roi_percent = (final_balance - account_size) / account_size * 100.0

    print("\n=== 백테스트 결과 ===")
    print(f"총 거래 횟수(분할 포함): {len(trades_df)}")
    print(f"총 손익: {total_pnl:.2f} USDT")
    print(f"최종 잔고: {final_balance:.2f} USDT")
    print(f"수익률: {roi_percent:.2f}%")

    print_performance_report(trades_df, initial_balance=account_size)

    return trades_df
