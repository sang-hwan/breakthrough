# backtesting/backtester.py
import pandas as pd
import numpy as np
from logs.logger_config import setup_logger

from trading.risk import (
    compute_position_size, allocate_position_splits, attempt_scale_in_position, 
    compute_risk_parameters_by_regime
)
from trading.trade_management import (
    calculate_partial_exit_targets, adjust_trailing_stop, compute_atr, calculate_dynamic_stop_and_take
)
from dynamic_parameters.dynamic_param_manager import DynamicParamManager
from data_collection.db_manager import fetch_ohlcv_records
from trading.positions import TradePosition

# HMM 및 보조 지표 레짐 판단 모듈 임포트
from markets_analysis.hmm_model import MarketRegimeHMM
from markets_analysis.regime_filter import filter_by_confidence

# 전략 선택 함수 (최종 액션은 backtester에서 세부 로직 구현)
from trading.strategies import select_strategy

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
        self.logger = setup_logger("backtester")
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
        if dynamic_params is None:
            dynamic_params = self.dynamic_param_manager.get_default_params()
        
        # === 시장 레짐 판단 (HMM 기반 및 보조 지표 적용) ===
        self.df_long['returns'] = self.df_long['close'].pct_change().fillna(0)
        self.df_long['volatility'] = self.df_long['returns'].rolling(window=20).std().fillna(0)
        
        hmm = MarketRegimeHMM(n_components=3)
        hmm.train(self.df_long, feature_columns=['returns', 'volatility'])
        regime_predictions = hmm.predict(self.df_long, feature_columns=['returns', 'volatility'])
        confidence_flags = filter_by_confidence(hmm, self.df_long, feature_columns=['returns', 'volatility'], 
                                                  threshold=dynamic_params.get('hmm_confidence_threshold', 0.8))
        
        sma_period = dynamic_params.get('sma_period', 200)
        self.df_long['long_term_sma'] = self.df_long['close'].rolling(window=sma_period, min_periods=1).mean()
        
        regime_map = {0: "bullish", 1: "bearish", 2: "sideways"}
        adjusted_regimes = []
        for idx, (pred, conf) in enumerate(zip(regime_predictions, confidence_flags)):
            if not conf:
                # HMM 신뢰도가 낮으면 보조 지표(SMA)를 사용하여 레짐 판단
                regime = "bullish" if self.df_long['close'].iloc[idx] > self.df_long['long_term_sma'].iloc[idx] else "bearish"
            else:
                regime = regime_map.get(pred, "unknown")
            adjusted_regimes.append(regime)
        
        regime_series = pd.Series(adjusted_regimes, index=self.df_long.index)
        self.df_short['market_regime'] = regime_series.reindex(self.df_short.index, method='ffill')
        
        # ATR 계산: short 데이터에 ATR 컬럼 추가 (atr_period는 dynamic_params에서 가져옴)
        self.df_short = compute_atr(self.df_short, period=dynamic_params.get("atr_period", 14))
        
        # === 레짐별 동적 리스크 파라미터 계산 ===
        base_risk_params = {
            "risk_per_trade": dynamic_params.get("risk_per_trade", 0.01),
            "atr_multiplier": dynamic_params.get("atr_multiplier", 2.0),
            "profit_ratio": dynamic_params.get("profit_ratio", 0.05)
        }
        
        # === 메인 백테스트 루프: 각 시점에서 레짐에 따른 전략 실행 ===
        for current_time, row in self.df_short.iterrows():
            market_regime = row['market_regime']
            risk_params = compute_risk_parameters_by_regime(
                base_risk_params,
                regime=market_regime,
                liquidity=dynamic_params.get('liquidity_info', 'high')
            )
            close_price = row["close"]
            
            # 전략 선택: 레짐 및 유동성 정보를 기반으로 결정
            action = select_strategy(
                market_regime=market_regime,
                liquidity_info=dynamic_params.get('liquidity_info', 'high'),
                data=self.df_short.loc[:current_time],
                current_time=current_time
            )
            
            if market_regime == "bullish" or action == "enter_long":
                # 롱 포지션 진입 및 스케일‑인 전략
                scaled_in = False
                for pos in self.positions:
                    if pos.side == "LONG":
                        attempt_scale_in_position(
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
                if not scaled_in:
                    new_position = TradePosition(
                        side="LONG",
                        initial_price=close_price,
                        maximum_size=0.0,
                        total_splits=dynamic_params.get("total_splits", 3),
                        allocation_plan=[]
                    )
                    new_position.highest_price = close_price
                    
                    # 동적 stop_loss, take_profit 계산
                    # 현재 진입가를 close_price로 가정하고, 해당 시점의 ATR 값을 가져옵니다.
                    entry_price = close_price
                    try:
                        atr_value = self.df_short.loc[current_time, "atr"]
                    except KeyError:
                        atr_value = 0  # 기본값 (예: 0) – 필요 시 추가 조치 가능
                    stop_loss_price, take_profit_price = calculate_dynamic_stop_and_take(entry_price, atr_value, risk_params)
                    
                    total_size = compute_position_size(
                        account_balance=self.account_size,
                        risk_percentage=risk_params["risk_per_trade"],
                        entry_price=close_price,
                        stop_loss=stop_loss_price,  # 동적 계산 값 사용
                        fee_rate=self.fee_rate
                    )
                    new_position.maximum_size = total_size
                    plan = allocate_position_splits(
                        total_size=1.0,
                        splits_count=dynamic_params.get("total_splits", 3),
                        allocation_mode=dynamic_params.get("allocation_mode", "equal")
                    )
                    new_position.allocation_plan = plan
                    exit_targets = calculate_partial_exit_targets(
                        entry_price=close_price,
                        partial_exit_ratio=dynamic_params.get("partial_exit_ratio", 0.5),
                        partial_profit_ratio=dynamic_params.get("partial_profit_ratio", 0.03),
                        final_profit_ratio=dynamic_params.get("final_profit_ratio", 0.06)
                    )
                    executed_price = close_price * (1 + self.slippage_rate)
                    new_position.add_execution(
                        entry_price=executed_price,
                        size=total_size * plan[0],
                        stop_loss=stop_loss_price,      # 동적 계산 값 사용
                        take_profit=take_profit_price,  # 동적 계산 값 사용
                        entry_time=current_time,
                        exit_targets=exit_targets,
                        trade_type="new_entry"
                    )
                    new_position.executed_splits = 1
                    self.positions.append(new_position)
            
            elif market_regime == "bearish" or action == "exit_all":
                # 베어리시 레짐: 전체 포지션 청산
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
                    if pos.is_empty():
                        self.positions.remove(pos)
            
            elif market_regime == "sideways":
                # 횡보장: 유동성에 따라 범위 트레이딩 또는 평균 회귀 전략 선택
                liquidity = dynamic_params.get('liquidity_info', 'high').lower()
                if liquidity == "high":
                    lower_bound = self.df_short['low'].rolling(window=20, min_periods=1).min().iloc[-1]
                    upper_bound = self.df_short['high'].rolling(window=20, min_periods=1).max().iloc[-1]
                    if close_price <= lower_bound:
                        self.logger.info(f"{current_time} - Range Trade 진입 (하단 터치): {close_price}")
                        scaled_in = False
                        for pos in self.positions:
                            if pos.side == "LONG":
                                attempt_scale_in_position(
                                    position=pos,
                                    current_price=close_price,
                                    scale_in_threshold=dynamic_params.get("scale_in_threshold", 0.02),
                                    slippage_rate=self.slippage_rate,
                                    stop_loss=row.get("stop_loss_price"),
                                    take_profit=row.get("take_profit_price"),
                                    entry_time=current_time,
                                    trade_type="range_trade_scale_in"
                                )
                                scaled_in = True
                        if not scaled_in:
                            new_position = TradePosition(
                                side="LONG",
                                initial_price=close_price,
                                maximum_size=0.0,
                                total_splits=dynamic_params.get("total_splits", 3),
                                allocation_plan=[]
                            )
                            new_position.highest_price = close_price
                            total_size = compute_position_size(
                                account_balance=self.account_size,
                                risk_percentage=risk_params["risk_per_trade"],
                                entry_price=close_price,
                                stop_loss=row.get("stop_loss_price"),
                                fee_rate=self.fee_rate
                            )
                            new_position.maximum_size = total_size
                            plan = allocate_position_splits(
                                total_size=1.0,
                                splits_count=dynamic_params.get("total_splits", 3),
                                allocation_mode=dynamic_params.get("allocation_mode", "equal")
                            )
                            new_position.allocation_plan = plan
                            exit_targets = calculate_partial_exit_targets(
                                entry_price=close_price,
                                partial_exit_ratio=dynamic_params.get("partial_exit_ratio", 0.5),
                                partial_profit_ratio=dynamic_params.get("partial_profit_ratio", 0.03),
                                final_profit_ratio=dynamic_params.get("final_profit_ratio", 0.06)
                            )
                            executed_price = close_price * (1 + self.slippage_rate)
                            new_position.add_execution(
                                entry_price=executed_price,
                                size=total_size * plan[0],
                                stop_loss=row.get("stop_loss_price"),
                                take_profit=row.get("take_profit_price"),
                                entry_time=current_time,
                                exit_targets=exit_targets,
                                trade_type="range_trade_entry"
                            )
                            new_position.executed_splits = 1
                            self.positions.append(new_position)
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
                                                break
                else:
                    mean_price = self.df_short['close'].rolling(window=20, min_periods=1).mean().iloc[-1]
                    std_price = self.df_short['close'].rolling(window=20, min_periods=1).std().iloc[-1]
                    if close_price < mean_price - std_price:
                        self.logger.info(f"{current_time} - Mean Reversion 진입 (가격 저평가): {close_price}")
                        scaled_in = False
                        for pos in self.positions:
                            if pos.side == "LONG":
                                attempt_scale_in_position(
                                    position=pos,
                                    current_price=close_price,
                                    scale_in_threshold=dynamic_params.get("scale_in_threshold", 0.02),
                                    slippage_rate=self.slippage_rate,
                                    stop_loss=row.get("stop_loss_price"),
                                    take_profit=row.get("take_profit_price"),
                                    entry_time=current_time,
                                    trade_type="mean_reversion_scale_in"
                                )
                                scaled_in = True
                        if not scaled_in:
                            new_position = TradePosition(
                                side="LONG",
                                initial_price=close_price,
                                maximum_size=0.0,
                                total_splits=dynamic_params.get("total_splits", 3),
                                allocation_plan=[]
                            )
                            new_position.highest_price = close_price
                            total_size = compute_position_size(
                                account_balance=self.account_size,
                                risk_percentage=risk_params["risk_per_trade"],
                                entry_price=close_price,
                                stop_loss=row.get("stop_loss_price"),
                                fee_rate=self.fee_rate
                            )
                            new_position.maximum_size = total_size
                            plan = allocate_position_splits(
                                total_size=1.0,
                                splits_count=dynamic_params.get("total_splits", 3),
                                allocation_mode=dynamic_params.get("allocation_mode", "equal")
                            )
                            new_position.allocation_plan = plan
                            exit_targets = calculate_partial_exit_targets(
                                entry_price=close_price,
                                partial_exit_ratio=dynamic_params.get("partial_exit_ratio", 0.5),
                                partial_profit_ratio=dynamic_params.get("partial_profit_ratio", 0.03),
                                final_profit_ratio=dynamic_params.get("final_profit_ratio", 0.06)
                            )
                            executed_price = close_price * (1 + self.slippage_rate)
                            new_position.add_execution(
                                entry_price=executed_price,
                                size=total_size * plan[0],
                                stop_loss=row.get("stop_loss_price"),
                                take_profit=row.get("take_profit_price"),
                                entry_time=current_time,
                                exit_targets=exit_targets,
                                trade_type="mean_reversion_entry"
                            )
                            new_position.executed_splits = 1
                            self.positions.append(new_position)
                    elif close_price > mean_price + std_price:
                        self.logger.info(f"{current_time} - Mean Reversion 청산 (가격 고평가): {close_price}")
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
                                                    "reason": "mean_reversion_partial_exit",
                                                    "trade_type": exec_record.get("trade_type", "unknown"),
                                                    "position_id": pos.position_id
                                                }
                                                self.trade_logs.append(trade_detail)
                                                self.trades.append(trade_detail)
                                                break

            # === 공통: 보유 포지션에 대해 트레일링 스탑 등 리스크 관리 업데이트 ===
            for pos in self.positions:
                for exec_record in pos.executions:
                    if not exec_record.get("closed", False):
                        pos.highest_price = max(pos.highest_price, close_price)
                        new_stop = adjust_trailing_stop(
                            current_stop=row.get("stop_loss_price", 0),
                            current_price=close_price,
                            highest_price=pos.highest_price,
                            trailing_percentage=dynamic_params.get("trailing_percent", 0.045)
                        )
                        exec_record["stop_loss"] = new_stop

        # === 최종 미청산 포지션 강제 청산 ===
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
        return self.trades, self.trade_logs
