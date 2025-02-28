# backtesting/backtester.py
import pandas as pd
from logs.logger_config import setup_logger
from trading.risk_manager import RiskManager 
from trading.trade_executor import TradeExecutor
from core.account import Account
from core.position import Position
from trading.asset_manager import AssetManager
from trading.ensemble import Ensemble
from config.config_manager import ConfigManager

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
        self.logger = setup_logger(__name__)
        self.config_manager = ConfigManager()
        self.account = Account(initial_balance=account_size, fee_rate=fee_rate)
        self.asset_manager = AssetManager(self.account)
        self.ensemble_manager = Ensemble()
        self.risk_manager = RiskManager()
        self.last_signal_time = None
        self.bullish_entry_events = []
        self.hmm_model = None
        self.last_hmm_training_datetime = None
        self.df_extra = None
        self.df_weekly = None
        self.last_rebalance_time = None
        self.last_weekly_close_date = None
        self.clock = lambda: pd.Timestamp.now()

    def get_current_time(self):
        return self.clock()

    def load_data(self, short_table_format, long_table_format, short_tf, long_tf, 
                  start_date=None, end_date=None, extra_tf=None, use_weekly=False):
        from backtesting.steps.data_loader import load_data
        load_data(self, short_table_format, long_table_format, short_tf, long_tf, start_date, end_date, extra_tf, use_weekly)

    def apply_indicators(self):
        from backtesting.steps.indicator_applier import apply_indicators
        try:
            apply_indicators(self)
        except Exception as e:
            self.logger.error("Error applying indicators: " + str(e), exc_info=True)
            raise

    def update_hmm_regime(self, dynamic_params):
        try:
            hmm_features = ['returns', 'volatility', 'sma', 'rsi', 'macd_macd', 'macd_signal', 'macd_diff']
            current_dt = self.df_long.index.max()
            retrain_interval_minutes = dynamic_params.get('hmm_retrain_interval_minutes', 60)
            retrain_interval = pd.Timedelta(minutes=retrain_interval_minutes)
            max_samples = dynamic_params.get('max_hmm_train_samples', 1000)
            min_samples = dynamic_params.get('min_hmm_train_samples', 50)
            feature_change_threshold = dynamic_params.get('hmm_feature_change_threshold', 0.01)

            if len(self.df_long) < min_samples:
                self.logger.error(f"Not enough samples for HMM training (min required: {min_samples}).", exc_info=True)
                return pd.Series(["sideways"] * len(self.df_long), index=self.df_long.index)

            if self.hmm_model is None or self.last_hmm_training_datetime is None:
                from markets.regime_model import MarketRegimeHMM
                self.hmm_model = MarketRegimeHMM(n_components=3, retrain_interval_minutes=retrain_interval_minutes)
                training_data = self.df_long if len(self.df_long) <= max_samples else self.df_long.tail(max_samples)
                self.hmm_model.train(training_data, feature_columns=hmm_features)
                self.last_hmm_training_datetime = current_dt
                self.logger.debug(f"HMM 학습 완료: {current_dt}")
            else:
                training_data = self.df_long if len(self.df_long) <= max_samples else self.df_long.tail(max_samples)
                current_means = training_data[hmm_features].mean()
                diff = (abs(current_means - self.hmm_model.last_feature_stats).mean() 
                        if self.hmm_model.last_feature_stats is not None else float('inf'))
                if (current_dt - self.last_hmm_training_datetime) >= retrain_interval or diff >= feature_change_threshold:
                    self.hmm_model.train(training_data, feature_columns=hmm_features)
                    self.last_hmm_training_datetime = current_dt
                    self.logger.debug(f"HMM 재학습 완료: {current_dt} (피처 변화 diff: {diff:.6f})")
                else:
                    self.logger.debug(f"HMM 재학습 스킵: 시간 경과 및 피처 변화 미미 (diff: {diff:.6f}).")

            predicted_regimes = self.hmm_model.predict_regime_labels(self.df_long, feature_columns=hmm_features)
            sma_period = dynamic_params.get('sma_period', 200)
            self.df_long['long_term_sma'] = self.df_long['close'].rolling(window=sma_period, min_periods=1).mean()
            adjusted_regimes = []
            for idx, predicted in enumerate(predicted_regimes):
                close_price = self.df_long['close'].iloc[idx]
                sma_price = self.df_long['long_term_sma'].iloc[idx]
                threshold = 0.01 * sma_price
                if close_price < sma_price - threshold:
                    adjusted_regime = "bearish"
                elif close_price > sma_price + threshold:
                    adjusted_regime = "bullish"
                else:
                    adjusted_regime = predicted
                adjusted_regimes.append(adjusted_regime)
            return pd.Series(adjusted_regimes, index=self.df_long.index)
        except Exception as e:
            self.logger.error("Error updating HMM regime: " + str(e), exc_info=True)
            return pd.Series(["sideways"] * len(self.df_long), index=self.df_long.index)

    def update_short_dataframe(self, regime_series, dynamic_params):
        try:
            self.df_short = self.df_short.join(self.df_long[['sma', 'rsi', 'volatility']], how='left').ffill()
            self.df_short['market_regime'] = regime_series.reindex(self.df_short.index).ffill()
            self.df_short = TradeExecutor.compute_atr(self.df_short, period=dynamic_params.get("atr_period", 14))
            default_atr_multiplier = dynamic_params.get("default_atr_multiplier", 2.0)
            self.df_short["stop_loss_price"] = self.df_short["close"] - (self.df_short["atr"] * default_atr_multiplier)
        except Exception as e:
            self.logger.error("Error updating short dataframe: " + str(e), exc_info=True)
            raise

    def handle_walk_forward_window(self, current_time, row):
        try:
            for pos in self.positions:
                for exec_record in pos.executions:
                    if not exec_record.get("closed", False):
                        final_close = row["close"]
                        adjusted_final_close = final_close * (1 - self.final_exit_slippage) if self.final_exit_slippage else final_close
                        exit_price = adjusted_final_close * (1 - self.slippage_rate)
                        size_value = exec_record["size"]
                        if isinstance(size_value, list):
                            size_value = sum(size_value)
                        fee = exit_price * size_value * self.fee_rate
                        pnl = (exit_price - exec_record["entry_price"]) * size_value - fee
                        exec_record["closed"] = True
                        trade_detail = {
                            "entry_time": exec_record["entry_time"],
                            "entry_price": exec_record["entry_price"],
                            "exit_time": current_time,
                            "exit_price": exit_price,
                            "size": size_value,
                            "pnl": pnl,
                            "reason": "walk_forward_window_close",
                            "trade_type": exec_record.get("trade_type", "unknown"),
                            "position_id": pos.position_id
                        }
                        self.trade_logs.append(trade_detail)
                        self.trades.append(trade_detail)
                        self.account.update_after_trade(trade_detail)
            self.positions = []
        except Exception as e:
            self.logger.error("Error during walk-forward window handling: " + str(e), exc_info=True)
            raise

    def handle_weekly_end(self, current_time, row):
        try:
            final_close = row["close"]
            adjusted_final_close = final_close * (1 - self.final_exit_slippage) if self.final_exit_slippage else final_close
            for pos in self.positions:
                for exec_record in pos.executions:
                    if not exec_record.get("closed", False):
                        exit_price = adjusted_final_close * (1 - self.slippage_rate)
                        size_value = exec_record["size"]
                        if isinstance(size_value, list):
                            size_value = sum(size_value)
                        fee = exit_price * size_value * self.fee_rate
                        pnl = (exit_price - exec_record["entry_price"]) * size_value - fee
                        exec_record["closed"] = True
                        trade_detail = {
                            "entry_time": exec_record["entry_time"],
                            "entry_price": exec_record["entry_price"],
                            "exit_time": current_time,
                            "exit_price": exit_price,
                            "size": size_value,
                            "pnl": pnl,
                            "reason": "weekly_end_close",
                            "trade_type": exec_record.get("trade_type", "unknown"),
                            "position_id": pos.position_id
                        }
                        self.trade_logs.append(trade_detail)
                        self.trades.append(trade_detail)
                        self.account.update_after_trade(trade_detail)
            self.positions = []
        except Exception as e:
            self.logger.error("Error during weekly end handling: " + str(e), exc_info=True)
            raise

    def process_bullish_entry(self, current_time, row, risk_params, dynamic_params):
        try:
            close_price = row["close"]
            signal_cooldown = pd.Timedelta(minutes=dynamic_params.get("signal_cooldown_minutes", 5))
            if self.last_signal_time is not None and (current_time - self.last_signal_time) < signal_cooldown:
                return

            stop_loss_price = row.get("stop_loss_price")
            if stop_loss_price is None:
                stop_loss_price = close_price * 0.95
                self.logger.error(f"Missing stop_loss_price for bullish entry at {current_time}. Using default stop_loss_price={stop_loss_price:.2f}.", exc_info=True)

            for pos in self.positions:
                if pos.side == "LONG":
                    additional_size = self.risk_manager.compute_position_size(
                        available_balance=self.account.get_available_balance(),
                        risk_percentage=risk_params.get("risk_per_trade"),
                        entry_price=close_price,
                        stop_loss=stop_loss_price,
                        fee_rate=self.fee_rate,
                        volatility=row.get("volatility", 0)
                    )
                    required_amount = close_price * additional_size * (1 + self.fee_rate)
                    if self.account.get_available_balance() >= required_amount:
                        threshold = dynamic_params.get("scale_in_threshold", 0.02)
                        effective_threshold = threshold * (0.5 if close_price < 10 else 1)
                        self.risk_manager.attempt_scale_in_position(
                            position=pos,
                            current_price=close_price,
                            scale_in_threshold=effective_threshold,
                            slippage_rate=self.slippage_rate,
                            stop_loss=stop_loss_price,
                            take_profit=row.get("take_profit_price"),
                            entry_time=current_time,
                            trade_type="scale_in"
                        )
                        self.last_signal_time = current_time
                        return
            total_size = self.risk_manager.compute_position_size(
                available_balance=self.account.get_available_balance(),
                risk_percentage=risk_params.get("risk_per_trade"),
                entry_price=close_price,
                stop_loss=stop_loss_price,
                fee_rate=self.fee_rate,
                volatility=row.get("volatility", 0)
            )
            required_amount = close_price * total_size * (1 + self.fee_rate)
            if self.account.get_available_balance() >= required_amount:
                new_position = Position(
                    side="LONG",
                    initial_price=close_price,
                    maximum_size=total_size,
                    total_splits=dynamic_params.get("total_splits", 3),
                    allocation_plan=self.risk_manager.allocate_position_splits(
                        total_size=1.0,
                        splits_count=dynamic_params.get("total_splits", 3),
                        allocation_mode=dynamic_params.get("allocation_mode", "equal")
                    )
                )
                try:
                    atr_value = self.df_short.loc[current_time, "atr"]
                except KeyError:
                    atr_value = 0
                stop_loss_price_new, take_profit_price = TradeExecutor.calculate_dynamic_stop_and_take(
                    entry_price=close_price,
                    atr=atr_value,
                    risk_params=risk_params
                )
                new_position.add_execution(
                    entry_price=close_price * (1 + self.slippage_rate),
                    size=total_size * new_position.allocation_plan[0],
                    stop_loss=stop_loss_price_new,
                    take_profit=take_profit_price,
                    entry_time=current_time,
                    exit_targets=TradeExecutor.calculate_partial_exit_targets(
                        entry_price=close_price,
                        partial_exit_ratio=dynamic_params.get("partial_exit_ratio", 0.5),
                        partial_profit_ratio=dynamic_params.get("partial_profit_ratio", 0.03),
                        final_profit_ratio=dynamic_params.get("final_profit_ratio", 0.06)
                    ),
                    trade_type="new_entry"
                )
                new_position.executed_splits = 1
                self.positions.append(new_position)
                self.account.add_position(new_position)
                self.last_signal_time = current_time
        except Exception as e:
            self.logger.error("Error processing bullish entry: " + str(e), exc_info=True)
            raise

    def process_bearish_exit(self, current_time, row):
        try:
            close_price = row["close"]
            for pos in self.positions:
                for exec_record in pos.executions:
                    if not exec_record.get("closed", False):
                        exit_price = close_price * (1 - self.slippage_rate)
                        size_value = exec_record["size"]
                        if isinstance(size_value, list):
                            size_value = sum(size_value)
                        fee = exit_price * size_value * self.fee_rate
                        pnl = (exit_price - exec_record["entry_price"]) * size_value - fee
                        exec_record["closed"] = True
                        trade_detail = {
                            "entry_time": exec_record["entry_time"],
                            "entry_price": exec_record["entry_price"],
                            "exit_time": current_time,
                            "exit_price": exit_price,
                            "size": size_value,
                            "pnl": pnl,
                            "reason": "exit_regime_change",
                            "trade_type": exec_record.get("trade_type", "unknown"),
                            "position_id": pos.position_id
                        }
                        self.trade_logs.append(trade_detail)
                        self.trades.append(trade_detail)
                        self.account.update_after_trade(trade_detail)
            self.last_signal_time = current_time
        except Exception as e:
            self.logger.error("Error processing bearish exit: " + str(e), exc_info=True)
            raise

    def process_sideways_trade(self, current_time, row, risk_params, dynamic_params):
        try:
            close_price = row["close"]
            liquidity = dynamic_params.get('liquidity_info', 'high').lower()
            if liquidity == "high":
                lower_bound = self.df_short['low'].rolling(window=20, min_periods=1).min().iloc[-1]
                if close_price <= lower_bound:
                    self.process_bullish_entry(current_time, row, risk_params, dynamic_params)
            else:
                mean_price = self.df_short['close'].rolling(window=20, min_periods=1).mean().iloc[-1]
                std_price = self.df_short['close'].rolling(window=20, min_periods=1).std().iloc[-1]
                if close_price < mean_price - std_price:
                    self.process_bullish_entry(current_time, row, risk_params, dynamic_params)
                elif close_price > mean_price + std_price:
                    self.process_bearish_exit(current_time, row)
        except Exception as e:
            self.logger.error("Error processing sideways trade: " + str(e), exc_info=True)
            raise

    def update_positions(self, current_time, row):
        try:
            close_price = row["close"]
            for pos in self.positions:
                for exec_record in pos.executions:
                    if not exec_record.get("closed", False):
                        pos.highest_price = max(pos.highest_price, close_price)
                        new_stop = TradeExecutor.adjust_trailing_stop(
                            current_stop=row.get("stop_loss_price", 0),
                            current_price=close_price,
                            highest_price=pos.highest_price,
                            trailing_percentage=self.config_manager.get_defaults().get("trailing_percent", 0.045)
                        )
                        exec_record["stop_loss"] = new_stop
        except Exception as e:
            self.logger.error("Error updating positions: " + str(e), exc_info=True)
            raise

    def finalize_all_positions(self):
        try:
            final_time = self.df_short.index[-1]
            final_close = self.df_short.iloc[-1]["close"]
            adjusted_final_close = final_close * (1 - self.final_exit_slippage) if self.final_exit_slippage else final_close
            for pos in self.positions:
                for exec_record in pos.executions:
                    if not exec_record.get("closed", False):
                        exit_price = adjusted_final_close * (1 - self.slippage_rate)
                        size_value = exec_record["size"]
                        if isinstance(size_value, list):
                            size_value = sum(size_value)
                        fee = exit_price * size_value * self.fee_rate
                        pnl = (exit_price - exec_record["entry_price"]) * size_value - fee
                        exec_record["closed"] = True
                        trade_detail = {
                            "entry_time": exec_record["entry_time"],
                            "entry_price": exec_record["entry_price"],
                            "exit_time": final_time,
                            "exit_price": exit_price,
                            "size": size_value,
                            "pnl": pnl,
                            "reason": "final_exit",
                            "trade_type": exec_record.get("trade_type", "unknown"),
                            "position_id": pos.position_id
                        }
                        self.trade_logs.append(trade_detail)
                        self.trades.append(trade_detail)
                        self.account.update_after_trade(trade_detail)
            self.positions = []
        except Exception as e:
            self.logger.error("Error finalizing positions: " + str(e), exc_info=True)
            raise

    def monitor_orders(self, current_time, row):
        try:
            for pos in self.positions:
                for exec_record in pos.executions:
                    if not exec_record.get("closed", False):
                        entry_price = exec_record.get("entry_price", 0)
                        current_price = row.get("close", entry_price)
                        if entry_price > 0 and abs(current_price - entry_price) / entry_price > 0.05:
                            self.logger.debug(f"Significant price move for position {pos.position_id} at {current_time}.")
        except Exception as e:
            self.logger.error("Error monitoring orders: " + str(e), exc_info=True)
            raise

    def run_backtest(self, dynamic_params=None, walk_forward_days: int = None, holdout_period: tuple = None):
        try:
            if dynamic_params is None:
                dynamic_params = self.config_manager.get_dynamic_params()
            else:
                dynamic_params = self.config_manager.validate_params(dynamic_params)
            try:
                self.df_long['returns'] = self.df_long['close'].pct_change().fillna(0)
                self.df_long['volatility'] = self.df_long['returns'].rolling(window=20).std().fillna(0)
            except Exception as e:
                self.logger.error("Error computing returns/volatility: " + str(e), exc_info=True)
                raise

            try:
                self.apply_indicators()
                self.logger.debug("인디케이터 계산 완료.")
            except Exception as e:
                self.logger.error("Error during indicator application: " + str(e), exc_info=True)
                raise

            try:
                from backtesting.steps.hmm_manager import update_hmm
                regime_series = update_hmm(self, dynamic_params)
                self.logger.debug("HMM 업데이트 완료.")
            except Exception as e:
                self.logger.error("Error updating HMM: " + str(e), exc_info=True)
                regime_series = pd.Series(["sideways"] * len(self.df_long), index=self.df_long.index)

            try:
                self.update_short_dataframe(regime_series, dynamic_params)
            except Exception as e:
                self.logger.error("Error updating short dataframe: " + str(e), exc_info=True)
                raise

            if holdout_period:
                holdout_start, holdout_end = pd.to_datetime(holdout_period[0]), pd.to_datetime(holdout_period[1])
                df_train = self.df_short[self.df_short.index < holdout_start]
                df_holdout = self.df_short[(self.df_short.index >= holdout_start) & (self.df_short.index <= holdout_end)]
            else:
                df_train = self.df_short
                df_holdout = None

            if walk_forward_days is not None:
                self.window_start = df_train.index[0]
                self.walk_forward_td = pd.Timedelta(days=walk_forward_days)
                self.walk_forward_days = walk_forward_days
            else:
                self.window_start = None
                self.walk_forward_days = None

            signal_cooldown = pd.Timedelta(minutes=dynamic_params.get("signal_cooldown_minutes", 5))
            rebalance_interval = pd.Timedelta(minutes=dynamic_params.get("rebalance_interval_minutes", 60))

            self.df_train = df_train
            try:
                from backtesting.steps.order_manager import process_training_orders, process_extra_orders, process_holdout_orders, finalize_orders
                process_training_orders(self, dynamic_params, signal_cooldown, rebalance_interval)
                process_extra_orders(self, dynamic_params)
                process_holdout_orders(self, dynamic_params, df_holdout)
                finalize_orders(self)
                self.logger.debug("주문 처리 완료.")
            except Exception as e:
                self.logger.error("Error during order processing: " + str(e), exc_info=True)
                raise

            available_balance = self.account.get_available_balance()
            self.logger.debug(f"Account available balance after backtest: {available_balance:.2f}")

            total_pnl = sum(trade.get("pnl", 0) if not isinstance(trade.get("pnl", 0), list) 
                            else sum(trade.get("pnl", 0)) for trade in self.trades)
            roi = total_pnl / self.account.initial_balance * 100
            # 최종 성과 요약을 info 레벨로 출력
            self.logger.debug(
                f"백테스트 결과: 거래 건수={len(self.trades)}, 계좌 잔액={available_balance:.2f}, "
                f"총 PnL={total_pnl:.2f}, ROI={roi:.2f}%"
            )
            return self.trades, self.trade_logs

        except Exception as e:
            self.logger.error("Fatal error in run_backtest: " + str(e), exc_info=True)
            raise

    def run_backtest_pipeline(self, dynamic_params=None, walk_forward_days: int = None, holdout_period: tuple = None):
        trades, trade_logs = self.run_backtest(dynamic_params, walk_forward_days, holdout_period)
        available_balance = self.account.get_available_balance()
        self.logger.debug(f"최종 계좌 잔액: {available_balance:.2f}")
        if self.positions:
            self.logger.error("백테스트 종료 후에도 미체결 포지션이 남아있습니다.", exc_info=True)
        else:
            self.logger.debug("모든 포지션이 정상적으로 종료되었습니다.")
        return trades, trade_logs
