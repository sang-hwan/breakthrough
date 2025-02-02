# backtesting/backtest_advanced.py

import pandas as pd
import numpy as np
from typing import Optional

# 기존 함수 임포트
from data_collection.db_ohlcv_manager import fetch_ohlcv_records
from trading.signals import (
    generate_breakout_signals,
    filter_long_trend,
    generate_retest_signals
)
from trading.trade_management import (
    calculate_atr_stop_loss,
    adjust_trailing_stop,
    set_fixed_take_profit,
    should_exit_trend,
    calculate_partial_exit_targets
)
from trading.risk import (
    compute_position_size,
    allocate_position_splits,
    attempt_scale_in_position
)
from trading.positions import TradePosition
from backtesting.performance_metrics import print_performance_report
from trading.indicators import compute_rsi

# --- 개선된 백테스트 로직 ---
def run_advanced_backtest(
    symbol: str = "BTC/USDT",
    short_timeframe: str = "4h",
    long_timeframe: str = "1d",
    window: int = 20,
    volume_factor: float = 1.5,
    confirm_bars: int = 2,
    breakout_buffer: float = 0.005,
    atr_window: int = 14,
    atr_multiplier: float = 2.5,
    profit_ratio: float = 0.05,
    account_size: float = 10000.0,
    risk_per_trade: float = 0.01,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    taker_fee_rate: float = 0.001,
    total_splits: int = 3,
    allocation_mode: str = "equal",
    threshold_percent: float = 0.02,
    use_trailing_stop: bool = False,
    trailing_percent: float = 0.05,
    use_trend_exit: bool = False,
    trend_exit_lookback: int = 30,
    trend_exit_price_column: str = "close",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_partial_take_profit: bool = False,
    partial_tp_factor: float = 0.03,
    final_tp_factor: float = 0.06,
    short_table_format: str = "ohlcv_{symbol}_{timeframe}",
    long_table_format: str = "ohlcv_{symbol}_{timeframe}",
    position_side: str = "LONG",
    stop_loss_col: str = "stop_loss_price",      # 미사용(각 거래별 계산)
    entry_price_col: str = "entry_price",         # 미사용(각 거래별 저장)
    take_profit_col: str = "take_profit_price",   # 미사용(각 거래별 계산)
    # 장기 추세 필터 관련 파라미터
    sma_period: int = 200,
    macd_slow_period: int = 26,
    macd_fast_period: int = 12,
    macd_signal_period: int = 9,
    rsi_period: int = 14,
    rsi_threshold: float = 70.0,
    bb_period: int = 20,
    bb_std_multiplier: float = 2.0,
    # 단기 보조지표 관련 파라미터
    use_short_term_indicators: bool = True,
    short_rsi_threshold: float = 40.0,
    short_rsi_period: int = 14,
    # 리테스트 관련 파라미터
    retest_threshold: float = 0.005,
    retest_confirmation_bars: int = 2,
    # 신호 컬럼명 (기본값)
    breakout_flag_col: str = "breakout_signal",
    confirmed_breakout_flag_col: str = "confirmed_breakout",
    retest_signal_col: str = "retest_signal",
    long_entry_col: str = "long_entry"
):
    """
    개선된 백테스트 함수
      - 거래 진입 전에 단기 데이터에 대해 ATR(및 관련 지표)를 미리 계산합니다.
      - 각 봉마다 포지션 상태(청산, 트레일링 스탑, 분할 매수)를 업데이트합니다.
      - 신호 조건(돌파/리테스트/단기 보조지표)와 장기 추세 필터를 모두 고려하여 거래 진입 여부를 결정합니다.
    """
    # 1. 데이터 로드 및 정렬
    short_table = short_table_format.format(symbol=symbol.replace('/', '').lower(), timeframe=short_timeframe)
    long_table = long_table_format.format(symbol=symbol.replace('/', '').lower(), timeframe=long_timeframe)
    df_short = fetch_ohlcv_records(table_name=short_table, start_date=start_date, end_date=end_date)
    df_long = fetch_ohlcv_records(table_name=long_table, start_date=start_date, end_date=end_date)
    if df_short.empty or df_long.empty:
        print("[ERROR] No data loaded.")
        return None
    df_short.sort_index(inplace=True)
    df_long.sort_index(inplace=True)
    
    # 2. 신호 생성 (단기: 돌파, 리테스트, 보조지표)
    df_short = generate_breakout_signals(
        data=df_short,
        lookback_window=window,
        volume_factor=volume_factor,
        confirmation_bars=confirm_bars,
        breakout_buffer=breakout_buffer,
        breakout_flag_col=breakout_flag_col,
        confirmed_breakout_flag_col=confirmed_breakout_flag_col
    )
    df_short = generate_retest_signals(
        data=df_short,
        retest_threshold=retest_threshold,
        confirmation_bars=retest_confirmation_bars,
        breakout_reference_col=f"highest_{window}",
        breakout_signal_col=breakout_flag_col,
        retest_signal_col=retest_signal_col
    )
    if use_short_term_indicators:
        df_short = compute_rsi(df_short, price_column="close", period=short_rsi_period, output_col="short_rsi")
        short_filter = df_short["short_rsi"] < short_rsi_threshold
    else:
        short_filter = True
    df_short[long_entry_col] = (df_short[confirmed_breakout_flag_col] | df_short[retest_signal_col]) & short_filter

    # 3. 장기 추세 필터 (동일 시간대의 데이터 사용)
    df_long = filter_long_trend(
        data=df_long,
        price_column='close',
        sma_period=sma_period,
        macd_slow_period=macd_slow_period,
        macd_fast_period=macd_fast_period,
        macd_signal_period=macd_signal_period,
        rsi_period=rsi_period,
        rsi_threshold=rsi_threshold,
        bb_period=bb_period,
        bb_std_multiplier=bb_std_multiplier,
        fillna=False
    )
    
    # 4. 단기 데이터에 대해 ATR를 미리 계산 (이후 각 거래 진입 시 사용)
    try:
        import ta
    except ImportError:
        print("[ERROR] 'ta' 라이브러리가 필요합니다.")
        return None

    atr_indicator = ta.volatility.AverageTrueRange(
        high=df_short['high'],
        low=df_short['low'],
        close=df_short['close'],
        window=atr_window,
        fillna=True
    )
    df_short['atr'] = atr_indicator.average_true_range()

    # 5. 백테스트 시뮬레이션
    current_position = None
    trades = []       # 개별 체결별 거래 내역 저장 리스트
    trade_logs = []   # 체결 상세 기록
    highest_price_since_entry = None  # 포지션 진입 후 최고가 기록

    # 각 봉마다 시뮬레이션 진행
    for current_time, row in df_short.iterrows():
        close_price = row['close']
        high_price = row['high']
        low_price = row['low']
        atr_value = row['atr']
        
        # 5-1. 열린 포지션 업데이트 (청산, 트레일링 스탑, 분할 매수)
        if current_position is not None:
            # (A) 트레일링 스탑 업데이트
            if use_trailing_stop:
                if highest_price_since_entry is None or high_price > highest_price_since_entry:
                    highest_price_since_entry = high_price
                for execution in current_position.executions:
                    if not execution.get("closed", False):
                        old_sl = execution['stop_loss']
                        new_sl = adjust_trailing_stop(
                            current_stop=old_sl,
                            current_price=close_price,
                            highest_price=highest_price_since_entry,
                            trailing_percentage=trailing_percent
                        )
                        execution['stop_loss'] = new_sl

            # (B) 각 체결에 대해 청산 조건 확인 (Stop Loss, Take Profit, Trend Exit)
            executions_to_close = []
            for i, exec_record in enumerate(current_position.executions):
                if exec_record.get("closed", False):
                    continue
                ep = exec_record['entry_price']
                sl = exec_record['stop_loss']
                tp = exec_record.get("take_profit")
                size = exec_record['size']
                exit_triggered = False
                exit_price = None
                exit_reason = None

                # Stop Loss 조건
                if low_price <= sl:
                    exit_triggered = True
                    exit_price = sl  # 슬리피지 적용 가능: sl * (1 - slippage_rate)
                    exit_reason = "stop_loss"
                # Take Profit 조건
                elif tp is not None and high_price >= tp:
                    exit_triggered = True
                    exit_price = tp  # 슬리피지 적용 가능: tp * (1 - slippage_rate)
                    exit_reason = "take_profit"
                # Trend Exit 조건 (장기 추세 필터 기반)
                elif use_trend_exit:
                    # 현재 시간에 대응하는 장기 데이터(또는 가장 최근 데이터) 사용
                    if current_time in df_long.index:
                        trend_row = df_long.loc[current_time]
                    else:
                        trend_row = df_long.iloc[-1]
                    df_long_sub = df_long.loc[:current_time]
                    if len(df_long_sub) >= trend_exit_lookback:
                        recent_window = df_long_sub.iloc[-trend_exit_lookback:]
                        recent_min = recent_window[trend_exit_price_column].min()
                        if close_price < recent_min:
                            exit_triggered = True
                            exit_price = close_price
                            exit_reason = "trend_exit"

                if exit_triggered:
                    # 체결 종료 처리
                    exec_record["closed"] = True
                    fee = exit_price * size * fee_rate
                    pnl = (exit_price - ep) * size - fee
                    trade_detail = {
                        'entry_time': exec_record['entry_time'],
                        'entry_price': ep,
                        'exit_time': current_time,
                        'exit_price': exit_price,
                        'size': size,
                        'pnl': pnl,
                        'reason': exit_reason,
                        'trade_type': exec_record.get("trade_type", "unknown"),
                        'position_id': current_position.position_id
                    }
                    trade_logs.append(trade_detail)
                    trades.append(trade_detail)
                    executions_to_close.append(i)
            
            # 제거: 종료된 체결은 리스트에서 제거하고, 포지션이 완전히 청산되었으면 None으로 처리
            for i in sorted(executions_to_close, reverse=True):
                current_position.executions.pop(i)
            if current_position.is_empty():
                current_position = None
                highest_price_since_entry = None

            # (C) 분할 매수 (Scale-In): 열린 포지션이 있고, 아직 전체 분할 횟수 미달이며, 현재 봉이 신규 진입 신호이면 추가 체결 시도
            if current_position is not None and current_position.executed_splits < current_position.total_splits and row[long_entry_col]:
                next_split = current_position.executed_splits
                # 다음 추가 체결 목표가: 초기 진입가의 (1 + threshold_percent * split_index)
                target_price = current_position.initial_price * (1.0 + threshold_percent * next_split)
                if close_price >= target_price:
                    total_size = current_position.maximum_size
                    plan_list = current_position.allocation_plan
                    if next_split < len(plan_list):
                        scale_in_size = total_size * plan_list[next_split]
                        scale_in_price = close_price  # 현재 가격 사용
                        scale_in_stop = scale_in_price - (atr_value * atr_multiplier)
                        scale_in_take_profit = scale_in_price * (1 + profit_ratio)
                        current_position.add_execution(
                            entry_price=scale_in_price,
                            size=scale_in_size,
                            stop_loss=scale_in_stop,
                            take_profit=scale_in_take_profit,
                            entry_time=current_time,
                            exit_targets=[],  # 분할 청산 목표 계산 함수 적용 가능
                            trade_type="scale_in"
                        )
                        current_position.executed_splits += 1

        # 5-2. 신규 진입: 열린 포지션이 없고, 단기 신호와 장기 추세 필터 조건 모두 만족할 때 진입
        if current_position is None and row[long_entry_col]:
            if current_time in df_long.index:
                trend_row = df_long.loc[current_time]
            else:
                trend_row = df_long.iloc[-1]
            if trend_row.get('long_filter_pass', False):
                current_position = TradePosition(
                    side=position_side,
                    initial_price=close_price,
                    maximum_size=0.0,  # 아래에서 계산
                    total_splits=total_splits,
                    allocation_plan=[]
                )
                # 진입 시 ATR을 기반으로 Stop Loss 계산
                entry_price_for_calc = close_price  # 슬리피지 미적용 가격
                initial_stop = close_price - (atr_value * atr_multiplier)
                total_size = compute_position_size(
                    account_balance=account_size,
                    risk_percentage=risk_per_trade,
                    entry_price=entry_price_for_calc,
                    stop_loss=initial_stop,
                    fee_rate=fee_rate
                )
                current_position.maximum_size = total_size
                plan_list = allocate_position_splits(
                    total_size=1.0,
                    splits_count=total_splits,
                    allocation_mode=allocation_mode
                )
                current_position.allocation_plan = plan_list
                # 첫 체결: 신규 진입
                buy_size = total_size * plan_list[0]
                buy_price = close_price
                initial_take_profit = buy_price * (1 + profit_ratio)
                current_position.add_execution(
                    entry_price=buy_price,
                    size=buy_size,
                    stop_loss=initial_stop,
                    take_profit=initial_take_profit,
                    entry_time=current_time,
                    exit_targets=[],  # 필요시 분할 청산 목표 계산 적용
                    trade_type="new_entry"
                )
                current_position.executed_splits = 1
                highest_price_since_entry = close_price

    # 6. 시뮬레이션 종료 후, 미체결 포지션은 마지막 봉의 종가로 청산 처리
    if current_position is not None and not current_position.is_empty():
        final_time = df_short.index[-1]
        final_close = df_short.iloc[-1]['close']
        for exec_record in current_position.executions:
            if not exec_record.get("closed", False):
                fee = final_close * exec_record['size'] * fee_rate
                pnl = (final_close - exec_record['entry_price']) * exec_record['size'] - fee
                exec_record["closed"] = True
                trade_detail = {
                    'entry_time': exec_record['entry_time'],
                    'entry_price': exec_record['entry_price'],
                    'exit_time': final_time,
                    'exit_price': final_close,
                    'size': exec_record['size'],
                    'pnl': pnl,
                    'reason': "final_exit",
                    'trade_type': exec_record.get("trade_type", "unknown"),
                    'position_id': current_position.position_id
                }
                trade_logs.append(trade_detail)
                trades.append(trade_detail)

    # 7. 결과 정리 및 성과 출력
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
    return trades_df, trade_logs
