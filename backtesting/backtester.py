# backtesting/backtester.py
import pandas as pd
import numpy as np
import logging

from trading.positions import TradePosition
from trading.signals import generate_breakout_signals, generate_retest_signals, filter_long_trend_relaxed
from trading.trade_management import (
    calculate_atr_stop_loss, adjust_trailing_stop, set_fixed_take_profit,
    should_exit_trend, calculate_partial_exit_targets
)
from trading.risk import compute_position_size, allocate_position_splits, attempt_scale_in_position
from backtesting.performance import print_performance_report
from dynamic_parameters.dynamic_param_manager import DynamicParamManager
from data_collection.db_manager import fetch_ohlcv_records

class Backtester:
    def __init__(self, symbol="BTC/USDT", account_size=10000.0, fee_rate=0.001, slippage_rate=0.0005, final_exit_slippage=0.0):
        self.symbol = symbol
        self.account_size = account_size
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.final_exit_slippage = final_exit_slippage
        self.positions = []
        self.trades = []
        self.trade_logs = []
        self.logger = logging.getLogger(self.__class__.__name__)
        # 동적 파라미터 관리를 위한 매니저 생성
        self.dynamic_param_manager = DynamicParamManager()

    def load_data(self, short_table_format, long_table_format, short_tf, long_tf, start_date=None, end_date=None):
        symbol_for_table = self.symbol.replace('/', '').lower()
        short_table = short_table_format.format(symbol=symbol_for_table, timeframe=short_tf)
        long_table = long_table_format.format(symbol=symbol_for_table, timeframe=long_tf)
        self.df_short = fetch_ohlcv_records(short_table, start_date, end_date)
        self.df_long = fetch_ohlcv_records(long_table, start_date, end_date)
        if self.df_short.empty or self.df_long.empty:
            self.logger.error("데이터 로드 실패: 데이터가 비어있습니다.")
            raise ValueError("No data loaded")
        self.df_short.sort_index(inplace=True)
        self.df_long.sort_index(inplace=True)
    
    def run_backtest(self, dynamic_params=None):
        # 동적 파라미터 미전달 시 기본값 사용
        if dynamic_params is None:
            dynamic_params = self.dynamic_param_manager.get_default_params()
        
        # 1. 신호 생성: 돌파, 리테스트, 장기 추세 필터 적용
        self.df_short = generate_breakout_signals(
            data=self.df_short,
            lookback_window=dynamic_params['lookback_window'],
            volume_factor=dynamic_params['volume_factor'],
            confirmation_bars=dynamic_params['confirmation_bars'],
            breakout_buffer=dynamic_params['breakout_buffer'],
            breakout_flag_col="breakout_signal",
            confirmed_breakout_flag_col="confirmed_breakout"
        )
        self.df_short = generate_retest_signals(
            data=self.df_short,
            retest_threshold=dynamic_params['retest_threshold'],
            confirmation_bars=dynamic_params['retest_confirmation_bars'],
            breakout_reference_col=f"highest_{dynamic_params['lookback_window']}",
            breakout_signal_col="breakout_signal",
            retest_signal_col="retest_signal"
        )
        self.df_long = filter_long_trend_relaxed(
            data=self.df_long,
            sma_period=dynamic_params['sma_period'],
            macd_slow_period=dynamic_params['macd_slow_period'],
            macd_fast_period=dynamic_params['macd_fast_period'],
            macd_signal_period=dynamic_params['macd_signal_period'],
            rsi_period=dynamic_params['rsi_period'],
            rsi_threshold=dynamic_params['rsi_threshold'],
            bb_period=dynamic_params['bb_period'],
            bb_std_multiplier=dynamic_params['bb_std_multiplier'],
            macd_diff_threshold=dynamic_params['macd_diff_threshold']
        )
        self.df_long = self.df_long.reindex(self.df_short.index, method='ffill')
        
        # 2. 신호 결합 (예: AND / OR 모드)
        entry_signal_mode = dynamic_params.get("entry_signal_mode", "AND")
        if entry_signal_mode == "AND":
            self.df_short['combined_entry'] = self.df_short["confirmed_breakout"] & self.df_long['long_filter_pass']
        elif entry_signal_mode == "OR":
            self.df_short['combined_entry'] = self.df_short["confirmed_breakout"] | self.df_long['long_filter_pass']
        else:
            self.df_short['combined_entry'] = self.df_short["confirmed_breakout"]
        
        # 3. 리스크 관리: ATR 기반 손절 및 고정 익절 설정
        self.df_short = calculate_atr_stop_loss(
            data=self.df_short,
            atr_period=dynamic_params['atr_period'],
            atr_multiplier=dynamic_params['atr_multiplier'],
            dynamic_sl_adjustment=dynamic_params['dynamic_sl_adjustment'],
            stop_loss_col="stop_loss_price",
            entry_price_col="entry_price",
            entry_signal_col="combined_entry"
        )
        self.df_short = set_fixed_take_profit(
            data=self.df_short,
            profit_ratio=dynamic_params['profit_ratio'],
            take_profit_col="take_profit_price",
            entry_price_col="entry_price"
        )
        
        # 4. 백테스트 메인 루프: 시간별 체결/청산 처리
        for current_time, row in self.df_short.iterrows():
            self.process_time_step(current_time, row, dynamic_params)
        
        # 5. 최종 미청산 포지션 강제 청산
        final_time = self.df_short.index[-1]
        final_close = self.df_short.iloc[-1]["close"]
        adjusted_final_close = final_close * (1 - self.final_exit_slippage) if self.final_exit_slippage else final_close
        for pos in self.positions:
            for exec_record in pos.executions:
                if not exec_record.get("closed", False):
                    exit_price = adjusted_final_close
                    fee = exit_price * exec_record["size"] * self.fee_rate
                    pnl = (exit_price - exec_record["entry_price"]) * exec_record["size"] - fee
                    exec_record["closed"] = True
                    trade_detail = {
                        "entry_time": exec_record["entry_time"],
                        "entry_price": exec_record["entry_price"],
                        "exit_time": final_time,
                        "exit_price": exit_price,
                        "size": exec_record["size"],
                        "pnl": pnl,
                        "reason": "final_exit",
                        "trade_type": exec_record.get("trade_type", "unknown"),
                        "position_id": pos.position_id
                    }
                    self.trade_logs.append(trade_detail)
                    self.trades.append(trade_detail)
        
        return self.trades, self.trade_logs

    def process_time_step(self, current_time, row, params):
        close_price = row["close"]
        high_price = row["high"]
        low_price = row["low"]
        computed_stop_loss = row["stop_loss_price"]
        computed_take_profit = row["take_profit_price"]

        # 기존 포지션 청산 및 조정 처리
        positions_to_remove = []
        for pos in self.positions:
            executions_to_close = []
            if params.get("use_trailing_stop", False):
                for execution in pos.executions:
                    if not execution.get("closed", False):
                        prev_high = execution.get("highest_price_since_entry", execution["entry_price"])
                        if high_price > prev_high:
                            execution["highest_price_since_entry"] = high_price
                        new_sl = adjust_trailing_stop(
                            current_stop=execution["stop_loss"],
                            current_price=close_price,
                            highest_price=execution["highest_price_since_entry"],
                            trailing_percentage=params.get("trailing_percent", 0.0)
                        )
                        execution["stop_loss"] = new_sl
            for i, exec_record in enumerate(pos.executions):
                if exec_record.get("closed", False):
                    continue
                ep = exec_record["entry_price"]
                size = exec_record["size"]
                exit_triggered = False
                exit_price = None
                exit_reason = None
                if low_price <= computed_stop_loss:
                    exit_triggered = True
                    exit_price = computed_stop_loss
                    exit_reason = "stop_loss"
                elif params.get("use_partial_take_profit", False) and "exit_targets" in exec_record:
                    for target in exec_record["exit_targets"]:
                        if not target.get("hit", False) and high_price >= target["price"]:
                            target["hit"] = True
                            partial_ratio = target["exit_ratio"]
                            closed_qty = size * partial_ratio
                            exec_record["size"] -= closed_qty
                            fee = target["price"] * closed_qty * self.fee_rate
                            pnl = (target["price"] - ep) * closed_qty - fee
                            trade_detail = {
                                "entry_time": exec_record["entry_time"],
                                "entry_price": ep,
                                "exit_time": current_time,
                                "exit_price": target["price"],
                                "size": closed_qty,
                                "pnl": pnl,
                                "reason": "partial_exit",
                                "trade_type": exec_record.get("trade_type", "unknown"),
                                "position_id": pos.position_id
                            }
                            self.trade_logs.append(trade_detail)
                            self.trades.append(trade_detail)
                            if exec_record["size"] < 1e-8:
                                exec_record["closed"] = True
                            break
                elif computed_take_profit and high_price >= computed_take_profit:
                    exit_triggered = True
                    exit_price = computed_take_profit
                    exit_reason = "take_profit"
                elif params.get("use_trend_exit", False) and should_exit_trend(self.df_long, current_time):
                    exit_triggered = True
                    exit_price = close_price
                    exit_reason = "trend_exit"
                if exit_triggered:
                    exec_record["closed"] = True
                    fee = exit_price * size * self.fee_rate
                    pnl = (exit_price - ep) * size - fee
                    trade_detail = {
                        "entry_time": exec_record["entry_time"],
                        "entry_price": ep,
                        "exit_time": current_time,
                        "exit_price": exit_price,
                        "size": size,
                        "pnl": pnl,
                        "reason": exit_reason,
                        "trade_type": exec_record.get("trade_type", "unknown"),
                        "position_id": pos.position_id
                    }
                    self.trade_logs.append(trade_detail)
                    self.trades.append(trade_detail)
                    executions_to_close.append(i)
            for i in sorted(executions_to_close, reverse=True):
                pos.executions.pop(i)
            if pos.is_empty():
                positions_to_remove.append(pos)
        for pos in positions_to_remove:
            self.positions.remove(pos)
        
        # 신규 진입 또는 스케일 인 (분할 매수)
        if row.get("combined_entry", False):
            scaled_in = False
            for pos in self.positions:
                if pos.side == "LONG":
                    attempt_scale_in_position(
                        position=pos,
                        current_price=close_price,
                        scale_in_threshold=params.get("scale_in_threshold", 0.02),
                        slippage_rate=self.slippage_rate,
                        stop_loss=computed_stop_loss,
                        take_profit=computed_take_profit,
                        entry_time=current_time,
                        trade_type="scale_in"
                    )
                    scaled_in = True
            if not scaled_in:
                from trading.risk import compute_position_size, allocate_position_splits
                new_position = TradePosition(
                    side="LONG",
                    initial_price=close_price,
                    maximum_size=0.0,
                    total_splits=params.get("total_splits", 3),
                    allocation_plan=[]
                )
                new_position.highest_price = close_price
                total_size = compute_position_size(
                    account_balance=self.account_size,
                    risk_percentage=params.get("risk_per_trade", 0.01),
                    entry_price=close_price,
                    stop_loss=computed_stop_loss,
                    fee_rate=self.fee_rate
                )
                new_position.maximum_size = total_size
                plan = allocate_position_splits(
                    total_size=1.0,
                    splits_count=params.get("total_splits", 3),
                    allocation_mode=params.get("allocation_mode", "equal")
                )
                new_position.allocation_plan = plan
                buy_size = total_size * plan[0]
                exit_targets = calculate_partial_exit_targets(
                    entry_price=close_price,
                    partial_exit_ratio=params.get("partial_tp_factor", 0.03),
                    final_profit_ratio=params.get("final_tp_factor", 0.06)
                )
                new_position.add_execution(
                    entry_price=close_price,
                    size=buy_size,
                    stop_loss=computed_stop_loss,
                    take_profit=computed_take_profit,
                    entry_time=current_time,
                    exit_targets=exit_targets,
                    trade_type="new_entry"
                )
                new_position.executed_splits = 1
                self.positions.append(new_position)
