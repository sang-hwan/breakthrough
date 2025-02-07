# backtesting/backtester.py
import pandas as pd
import numpy as np
from datetime import timedelta

from logs.logger_config import setup_logger
from trading.risk_manager import RiskManager
from trading.trade_manager import TradeManager
from dynamic_parameters.dynamic_param_manager import DynamicParamManager
from data_collection.db_manager import fetch_ohlcv_records
from trading.positions import TradePosition
from markets_analysis.hmm_model import MarketRegimeHMM
from markets_analysis.regime_filter import filter_by_confidence
from trading.strategies import TradingStrategies
from trading.account import Account
from trading.indicators import compute_sma, compute_rsi, compute_macd, compute_bollinger_bands
from trading.asset_manager import AssetManager
from trading.ensemble_manager import EnsembleManager

class Backtester:
    def __init__(self, symbol="BTC/USDT", account_size=10000.0, fee_rate=0.001, 
                 slippage_rate=0.0005, final_exit_slippage=0.0):
        self.symbol = symbol
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.final_exit_slippage = final_exit_slippage
        self.positions = []
        self.trades = []
        self.trade_logs = []
        self.logger = setup_logger("backtester")
        self.dynamic_param_manager = DynamicParamManager()
        self.account = Account(initial_balance=account_size, fee_rate=fee_rate)
        self.asset_manager = AssetManager(self.account)
        self.ensemble_manager = EnsembleManager()  # 신호 집계용
        self.strategy_manager = TradingStrategies()  # 개별 전략 관리
        self.hmm_model = None
        self.last_hmm_training_datetime = None
        self.df_extra = None
        self.last_signal_time = None
        self.last_rebalance_time = None

    def load_data(self, short_table_format, long_table_format, short_tf, long_tf, 
                  start_date=None, end_date=None, extra_tf=None):
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
        
        if extra_tf:
            extra_table = short_table_format.format(symbol=symbol_for_table, timeframe=extra_tf)
            self.df_extra = fetch_ohlcv_records(extra_table, start_date, end_date)
            if not self.df_extra.empty:
                self.df_extra.sort_index(inplace=True)
                self.df_extra = compute_bollinger_bands(self.df_extra, price_column='close', 
                                                        period=20, std_multiplier=2.0, fillna=True)

    def apply_indicators(self):
        self.df_long = compute_sma(self.df_long, price_column='close', period=200, fillna=True, output_col='sma')
        self.df_long = compute_rsi(self.df_long, price_column='close', period=14, fillna=True, output_col='rsi')
        self.df_long = compute_macd(self.df_long, price_column='close', slow_period=26, fast_period=12, 
                                     signal_period=9, fillna=True, prefix='macd_')

    def update_hmm_regime(self, dynamic_params):
        # 사용 피처 목록
        hmm_features = ['returns', 'volatility', 'sma', 'rsi', 'macd_macd', 'macd_signal', 'macd_diff']
        current_dt = self.df_long.index.max()
        retrain_interval = pd.Timedelta(minutes=dynamic_params.get('hmm_retrain_interval_minutes', 60))
        max_samples = dynamic_params.get('max_hmm_train_samples', 1000)
        # HMM 재학습: 시간 경과 및 데이터 변화(피처 평균 차이)는 이미 hmm_model 내부에서 세밀하게 처리됨
        if (self.hmm_model is None) or (self.last_hmm_training_datetime is None) or ((current_dt - self.last_hmm_training_datetime) >= retrain_interval):
            self.hmm_model = MarketRegimeHMM(n_components=3, retrain_interval_minutes=dynamic_params.get('hmm_retrain_interval_minutes', 60))
            training_data = self.df_long if len(self.df_long) <= max_samples else self.df_long.tail(max_samples)
            self.hmm_model.train(training_data, feature_columns=hmm_features)
            self.last_hmm_training_datetime = current_dt
            self.logger.info(f"HMM 모델 재학습 완료: {current_dt}")
        regime_predictions = self.hmm_model.predict(self.df_long, feature_columns=hmm_features)
        confidence_flags = filter_by_confidence(self.hmm_model, self.df_long, feature_columns=hmm_features, 
                                                  threshold=dynamic_params.get('hmm_confidence_threshold', 0.8))
        self.df_long['long_term_sma'] = self.df_long['close'].rolling(window=dynamic_params.get('sma_period', 200), min_periods=1).mean()
        regime_map = {0: "bullish", 1: "bearish", 2: "sideways"}
        adjusted_regimes = []
        for idx, (pred, conf) in enumerate(zip(regime_predictions, confidence_flags)):
            if not conf:
                regime = "bullish" if self.df_long['close'].iloc[idx] > self.df_long['long_term_sma'].iloc[idx] else "bearish"
            else:
                regime = regime_map.get(pred, "unknown")
            adjusted_regimes.append(regime)
        regime_series = pd.Series(adjusted_regimes, index=self.df_long.index)
        return regime_series

    def update_short_dataframe(self, regime_series, dynamic_params):
        self.df_short = self.df_short.join(self.df_long[['sma', 'rsi', 'volatility']], how='left').ffill()
        self.df_short['market_regime'] = regime_series.reindex(self.df_short.index).ffill()
        self.df_short = TradeManager.compute_atr(self.df_short, period=dynamic_params.get("atr_period", 14))

    def handle_walk_forward_window(self, current_time, row):
        # 워크‑포워드 기간 종료 시 모든 포지션 강제 청산
        for pos in self.positions:
            for exec_record in pos.executions:
                if not exec_record.get("closed", False):
                    final_close = row["close"]
                    adjusted_final_close = final_close * (1 - self.final_exit_slippage) if self.final_exit_slippage else final_close
                    exit_price = adjusted_final_close * (1 - self.slippage_rate)
                    fee = exit_price * exec_record["size"] * self.fee_rate
                    pnl = (exit_price - exec_record["entry_price"]) * exec_record["size"] - fee
                    exec_record["closed"] = True
                    trade_detail = {
                        "entry_time": exec_record["entry_time"],
                        "entry_price": exec_record["entry_price"],
                        "exit_time": current_time,
                        "exit_price": exit_price,
                        "size": exec_record["size"],
                        "pnl": pnl,
                        "reason": "walk_forward_window_close",
                        "trade_type": exec_record.get("trade_type", "unknown"),
                        "position_id": pos.position_id
                    }
                    self.trade_logs.append(trade_detail)
                    self.trades.append(trade_detail)
                    self.account.update_after_trade(trade_detail)
        self.positions = []

    def process_bullish_entry(self, current_time, row, risk_params, dynamic_params):
        close_price = row["close"]
        # 신호 쿨다운: 일정 시간 동안 신규 진입 신호 발생 억제
        signal_cooldown = pd.Timedelta(minutes=dynamic_params.get("signal_cooldown_minutes", 5))
        if self.last_signal_time is not None and (current_time - self.last_signal_time) < signal_cooldown:
            self.logger.debug(f"{current_time} - 신호 쿨다운 중 (최근 신호: {self.last_signal_time})")
            return
        scaled_in = False
        for pos in self.positions:
            if pos.side == "LONG":
                additional_size = RiskManager.compute_position_size(
                    available_balance=self.account.get_available_balance(),
                    risk_percentage=risk_params["risk_per_trade"],
                    entry_price=close_price,
                    stop_loss=row.get("stop_loss_price"),
                    fee_rate=self.fee_rate,
                    volatility=row.get("volatility", 0)
                )
                required_amount = close_price * additional_size * (1 + self.fee_rate)
                available_balance = self.account.get_available_balance()
                if available_balance >= required_amount:
                    RiskManager.attempt_scale_in_position(
                        position=pos,
                        current_price=close_price,
                        scale_in_threshold=dynamic_params.get("scale_in_threshold", 0.02),
                        slippage_rate=self.slippage_rate,
                        stop_loss=row.get("stop_loss_price"),
                        take_profit=row.get("take_profit_price"),
                        entry_time=current_time,
                        trade_type="scale_in"
                    )
                    scaled_in = True
                    self.last_signal_time = current_time
                else:
                    self.logger.info(f"{current_time} - 스케일인 불가: 가용 잔고 부족 (가용: {available_balance:.2f}, 필요: {required_amount:.2f})")
        if not scaled_in:
            total_size = RiskManager.compute_position_size(
                available_balance=self.account.get_available_balance(),
                risk_percentage=risk_params["risk_per_trade"],
                entry_price=close_price,
                stop_loss=row.get("stop_loss_price"),
                fee_rate=self.fee_rate,
                volatility=row.get("volatility", 0)
            )
            required_amount = close_price * total_size * (1 + self.fee_rate)
            available_balance = self.account.get_available_balance()
            if available_balance >= required_amount:
                new_position = TradePosition(
                    side="LONG",
                    initial_price=close_price,
                    maximum_size=total_size,
                    total_splits=dynamic_params.get("total_splits", 3),
                    allocation_plan=RiskManager.allocate_position_splits(
                        total_size=1.0,
                        splits_count=dynamic_params.get("total_splits", 3),
                        allocation_mode=dynamic_params.get("allocation_mode", "equal")
                    )
                )
                new_position.highest_price = close_price
                try:
                    atr_value = self.df_short.loc[current_time, "atr"]
                except KeyError:
                    atr_value = 0
                stop_loss_price, take_profit_price = TradeManager.calculate_dynamic_stop_and_take(close_price, atr_value, risk_params)
                new_position.allocation_plan = RiskManager.allocate_position_splits(
                    total_size=1.0,
                    splits_count=dynamic_params.get("total_splits", 3),
                    allocation_mode=dynamic_params.get("allocation_mode", "equal")
                )
                exit_targets = TradeManager.calculate_partial_exit_targets(
                    entry_price=close_price,
                    partial_exit_ratio=dynamic_params.get("partial_exit_ratio", 0.5),
                    partial_profit_ratio=dynamic_params.get("partial_profit_ratio", 0.03),
                    final_profit_ratio=dynamic_params.get("final_profit_ratio", 0.06)
                )
                executed_price = close_price * (1 + self.slippage_rate)
                new_position.add_execution(
                    entry_price=executed_price,
                    size=total_size * new_position.allocation_plan[0],
                    stop_loss=stop_loss_price,
                    take_profit=take_profit_price,
                    entry_time=current_time,
                    exit_targets=exit_targets,
                    trade_type="new_entry"
                )
                new_position.executed_splits = 1
                self.positions.append(new_position)
                self.account.add_position(new_position)
                self.last_signal_time = current_time
                self.logger.info(f"{current_time} - 신규 진입 실행: size={total_size:.4f}, entry_price={close_price:.2f}")
            else:
                self.logger.info(f"{current_time} - 신규 진입 불가: 가용 잔고 부족 (가용: {available_balance:.2f}, 필요: {required_amount:.2f})")

    def process_bearish_exit(self, current_time, row):
        close_price = row["close"]
        for pos in self.positions[:]:
            for exec_record in pos.executions:
                if not exec_record.get("closed", False):
                    exit_price = close_price * (1 - self.slippage_rate)
                    fee = exit_price * exec_record["size"] * self.fee_rate
                    pnl = (exit_price - exec_record["entry_price"]) * exec_record["size"] - fee
                    exec_record["closed"] = True
                    trade_detail = {
                        "entry_time": exec_record["entry_time"],
                        "entry_price": exec_record["entry_price"],
                        "exit_time": current_time,
                        "exit_price": exit_price,
                        "size": exec_record["size"],
                        "pnl": pnl,
                        "reason": "exit_regime_change",
                        "trade_type": exec_record.get("trade_type", "unknown"),
                        "position_id": pos.position_id
                    }
                    self.trade_logs.append(trade_detail)
                    self.trades.append(trade_detail)
                    self.account.update_after_trade(trade_detail)
            if pos.is_empty():
                self.positions.remove(pos)
                self.account.remove_position(pos)
        self.last_signal_time = current_time

    def process_sideways_trade(self, current_time, row, risk_params, dynamic_params):
        close_price = row["close"]
        liquidity = dynamic_params.get('liquidity_info', 'high').lower()
        if liquidity == "high":
            lower_bound = self.df_short['low'].rolling(window=20, min_periods=1).min().iloc[-1]
            upper_bound = self.df_short['high'].rolling(window=20, min_periods=1).max().iloc[-1]
            if close_price <= lower_bound:
                self.logger.info(f"{current_time} - Range Trade 진입 (하단 터치): {close_price}")
                self.process_bullish_entry(current_time, row, risk_params, dynamic_params)
            elif close_price >= upper_bound:
                self.logger.info(f"{current_time} - Range Trade 청산 (상단 터치): {close_price}")
                for pos in self.positions:
                    if pos.side == "LONG":
                        for i, exec_record in enumerate(pos.executions):
                            if not exec_record.get("closed", False) and "exit_targets" in exec_record:
                                for target in exec_record["exit_targets"]:
                                    if not target.get("hit", False) and close_price >= target["price"]:
                                        target["hit"] = True
                                        closed_qty = pos.partial_close_execution(i, target["exit_ratio"])
                                        fee = close_price * closed_qty * self.fee_rate
                                        pnl = (close_price - exec_record["entry_price"]) * closed_qty - fee
                                        trade_detail = {
                                            "entry_time": exec_record["entry_time"],
                                            "entry_price": exec_record["entry_price"],
                                            "exit_time": current_time,
                                            "exit_price": close_price,
                                            "size": closed_qty,
                                            "pnl": pnl,
                                            "reason": "range_trade_partial_exit",
                                            "trade_type": exec_record.get("trade_type", "unknown"),
                                            "position_id": pos.position_id
                                        }
                                        self.trade_logs.append(trade_detail)
                                        self.trades.append(trade_detail)
                                        self.account.update_after_trade(trade_detail)
                                        break
        else:
            mean_price = self.df_short['close'].rolling(window=20, min_periods=1).mean().iloc[-1]
            std_price = self.df_short['close'].rolling(window=20, min_periods=1).std().iloc[-1]
            if close_price < mean_price - std_price:
                self.logger.info(f"{current_time} - Mean Reversion 진입 (저평가): {close_price}")
                self.process_bullish_entry(current_time, row, risk_params, dynamic_params)
            elif close_price > mean_price + std_price:
                self.logger.info(f"{current_time} - Mean Reversion 청산 (고평가): {close_price}")
                self.process_bearish_exit(current_time, row)

    def update_positions(self, current_time, row):
        close_price = row["close"]
        for pos in self.positions:
            for exec_record in pos.executions:
                if not exec_record.get("closed", False):
                    pos.highest_price = max(pos.highest_price, close_price)
                    if abs(close_price - exec_record["entry_price"]) / exec_record["entry_price"] > 0.01:
                        new_stop = TradeManager.adjust_trailing_stop(
                            current_stop=row.get("stop_loss_price", 0),
                            current_price=close_price,
                            highest_price=pos.highest_price,
                            trailing_percentage=self.dynamic_param_manager.get_default_params().get("trailing_percent", 0.045)
                        )
                        exec_record["stop_loss"] = new_stop

    def finalize_all_positions(self):
        final_time = self.df_short.index[-1]
        final_close = self.df_short.iloc[-1]["close"]
        adjusted_final_close = final_close * (1 - self.final_exit_slippage) if self.final_exit_slippage else final_close
        for pos in self.positions:
            for exec_record in pos.executions:
                if not exec_record.get("closed", False):
                    exit_price = adjusted_final_close * (1 - self.slippage_rate)
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
                    self.account.update_after_trade(trade_detail)

    def monitor_orders(self, current_time, row):
        for pos in self.positions:
            for exec_record in pos.executions:
                if not exec_record.get("closed", False):
                    entry_price = exec_record.get("entry_price", 0)
                    current_price = row.get("close", entry_price)
                    if abs(current_price - entry_price) / entry_price > 0.05:
                        self.logger.debug(f"Order monitoring: {current_time} - Significant price move for position {pos.position_id}.")

    def run_backtest(self, dynamic_params=None, walk_forward_days: int = None, holdout_period: tuple = None):
        if dynamic_params is None:
            dynamic_params = self.dynamic_param_manager.get_default_params()
        self.df_long['returns'] = self.df_long['close'].pct_change().fillna(0)
        self.df_long['volatility'] = self.df_long['returns'].rolling(window=20).std().fillna(0)
        self.apply_indicators()
        regime_series = self.update_hmm_regime(dynamic_params)
        self.update_short_dataframe(regime_series, dynamic_params)
        
        if holdout_period:
            holdout_start, holdout_end = pd.to_datetime(holdout_period[0]), pd.to_datetime(holdout_period[1])
            df_train = self.df_short[self.df_short.index < holdout_start]
            df_holdout = self.df_short[(self.df_short.index >= holdout_start) & (self.df_short.index <= holdout_end)]
        else:
            df_train = self.df_short
            df_holdout = None
        
        # 워크‑포워드 윈도우 시작 설정 (walk_forward_days 지정 시)
        if walk_forward_days is not None:
            window_start = df_train.index[0]
            walk_forward_td = pd.Timedelta(days=walk_forward_days)
        else:
            window_start = None
        
        signal_cooldown = pd.Timedelta(minutes=dynamic_params.get("signal_cooldown_minutes", 5))
        rebalance_interval = pd.Timedelta(minutes=dynamic_params.get("rebalance_interval_minutes", 60))
        
        for current_time, row in df_train.iterrows():
            if walk_forward_days is not None and current_time - window_start >= walk_forward_td:
                self.handle_walk_forward_window(current_time, row)
                window_start = current_time
            if self.last_signal_time is not None and (current_time - self.last_signal_time) < signal_cooldown:
                self.logger.debug(f"{current_time} - 신호 쿨다운 적용: 최근 신호 시간 {self.last_signal_time}")
            else:
                action = self.ensemble_manager.get_final_signal(
                    row['market_regime'], 
                    dynamic_params.get('liquidity_info', 'high'), 
                    self.df_short, 
                    current_time
                )
                base_risk_params = {
                    "risk_per_trade": dynamic_params.get("risk_per_trade", 0.01),
                    "atr_multiplier": dynamic_params.get("atr_multiplier", 2.0),
                    "profit_ratio": dynamic_params.get("profit_ratio", 0.05),
                    "current_volatility": row.get("volatility", 0)
                }
                risk_params = RiskManager.compute_risk_parameters_by_regime(
                    base_risk_params,
                    regime=row['market_regime'],
                    liquidity=dynamic_params.get('liquidity_info', 'high')
                )
                if action == "enter_long":
                    self.process_bullish_entry(current_time, row, risk_params, dynamic_params)
                elif action == "exit_all":
                    self.process_bearish_exit(current_time, row)
                elif row['market_regime'] == "sideways":
                    self.process_sideways_trade(current_time, row, risk_params, dynamic_params)
            self.update_positions(current_time, row)
            if (self.last_rebalance_time is None) or ((current_time - self.last_rebalance_time) >= rebalance_interval):
                try:
                    self.asset_manager.rebalance(row['market_regime'])
                except Exception as e:
                    self.logger.error(f"Asset rebalancing failed: {e}")
                self.last_rebalance_time = current_time
        
        # 고빈도 거래 처리 (extra 데이터가 있을 경우)
        if self.df_extra is not None and not self.df_extra.empty:
            for current_time, row in self.df_extra.iterrows():
                hf_signal = self.strategy_manager.high_frequency_strategy(self.df_extra, current_time)
                regime = self.df_long.loc[self.df_long.index <= current_time].iloc[-1].get('market_regime', 'sideways')
                base_risk_params = {
                    "risk_per_trade": dynamic_params.get("risk_per_trade", 0.01),
                    "atr_multiplier": dynamic_params.get("atr_multiplier", 2.0),
                    "profit_ratio": dynamic_params.get("profit_ratio", 0.05),
                    "current_volatility": row.get("volatility", 0)
                }
                risk_params = RiskManager.compute_risk_parameters_by_regime(
                    base_risk_params,
                    regime=regime,
                    liquidity=dynamic_params.get('liquidity_info', 'high')
                )
                if hf_signal == "enter_long":
                    self.process_bullish_entry(current_time, row, risk_params, dynamic_params)
                elif hf_signal == "exit_all":
                    self.process_bearish_exit(current_time, row)
                self.monitor_orders(current_time, row)
        
        # 홀드아웃 구간 백테스트
        if df_holdout is not None:
            self.logger.info("홀드아웃 구간 백테스트 시작.")
            for current_time, row in df_holdout.iterrows():
                action = self.ensemble_manager.get_final_signal(
                    row['market_regime'], 
                    dynamic_params.get('liquidity_info', 'high'), 
                    self.df_short, 
                    current_time
                )
                base_risk_params = {
                    "risk_per_trade": dynamic_params.get("risk_per_trade", 0.01),
                    "atr_multiplier": dynamic_params.get("atr_multiplier", 2.0),
                    "profit_ratio": dynamic_params.get("profit_ratio", 0.05),
                    "current_volatility": row.get("volatility", 0)
                }
                risk_params = RiskManager.compute_risk_parameters_by_regime(
                    base_risk_params,
                    regime=row['market_regime'],
                    liquidity=dynamic_params.get('liquidity_info', 'high')
                )
                if action == "enter_long":
                    self.process_bullish_entry(current_time, row, risk_params, dynamic_params)
                elif action == "exit_all":
                    self.process_bearish_exit(current_time, row)
                elif row['market_regime'] == "sideways":
                    self.process_sideways_trade(current_time, row, risk_params, dynamic_params)
                self.update_positions(current_time, row)
        
        self.finalize_all_positions()
        return self.trades, self.trade_logs
