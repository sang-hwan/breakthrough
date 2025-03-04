# backtesting/steps/order_manager.py
from logs.logger_config import setup_logger
from logs.logging_util import LoggingUtil  # 동적 상태 변화 로깅 유틸리티 추가

logger = setup_logger(__name__)
log_util = LoggingUtil(__name__)

def get_signal_with_weekly_override(backtester, row, current_time, dynamic_params):
    """
    Generates trading signal with weekly override based on perfect knowledge of weekly low and high.
    If the current close price is within a small tolerance of the weekly low, returns 'enter_long'.
    If it's near the weekly high, returns 'exit_all'.
    Otherwise, falls back to the ensemble manager's signal.
    """
    try:
        if hasattr(backtester, 'df_weekly') and backtester.df_weekly is not None and not backtester.df_weekly.empty:
            weekly_bar = backtester.df_weekly.loc[backtester.df_weekly.index <= current_time].iloc[-1]
            if "weekly_low" in weekly_bar and "weekly_high" in weekly_bar:
                tolerance = 0.002
                if abs(row["close"] - weekly_bar["weekly_low"]) / weekly_bar["weekly_low"] <= tolerance:
                    log_util.log_event("Weekly override: enter_long", state_key="order_signal")
                    return "enter_long"
                elif abs(row["close"] - weekly_bar["weekly_high"]) / weekly_bar["weekly_high"] <= tolerance:
                    log_util.log_event("Weekly override: exit_all", state_key="order_signal")
                    return "exit_all"
            else:
                backtester.logger.warning("Weekly override skipped: weekly_bar missing 'weekly_low' or 'weekly_high' keys.")
        return backtester.ensemble_manager.get_final_signal(
            row.get('market_regime', 'unknown'),
            dynamic_params.get('liquidity_info', 'high'),
            backtester.df_short,
            current_time,
            data_weekly=getattr(backtester, 'df_weekly', None)
        )
    except Exception as e:
        backtester.logger.error(f"Error in weekly override signal generation: {e}", exc_info=True)
        return backtester.ensemble_manager.get_final_signal(
            row.get('market_regime', 'unknown'),
            dynamic_params.get('liquidity_info', 'high'),
            backtester.df_short,
            current_time,
            data_weekly=getattr(backtester, 'df_weekly', None)
        )

def process_training_orders(backtester, dynamic_params, signal_cooldown, rebalance_interval):
    for current_time, row in backtester.df_train.iterrows():
        try:
            try:
                if current_time.weekday() == 4 and (
                    backtester.last_weekly_close_date is None or 
                    backtester.last_weekly_close_date != current_time.date()
                ):
                    try:
                        backtester.handle_weekly_end(current_time, row)
                    except Exception as e:
                        logger.error(f"Weekly end handling error {e}", exc_info=True)
                    backtester.last_weekly_close_date = current_time.date()
                    continue
            except Exception as e:
                logger.error(f"Error during weekly end check {e}", exc_info=True)
            
            try:
                if backtester.walk_forward_days is not None and (current_time - backtester.window_start) >= backtester.walk_forward_td:
                    try:
                        backtester.handle_walk_forward_window(current_time, row)
                    except Exception as e:
                        logger.error(f"Walk-forward window handling error {e}", exc_info=True)
                    backtester.window_start = current_time
            except Exception as e:
                logger.error(f"Error during walk-forward window check {e}", exc_info=True)
            
            if backtester.last_signal_time is None or (current_time - backtester.last_signal_time) >= signal_cooldown:
                action = get_signal_with_weekly_override(backtester, row, current_time, dynamic_params)
            else:
                action = "hold"
                
            base_risk_params = {
                "risk_per_trade": dynamic_params.get("risk_per_trade", 0.01),
                "atr_multiplier": dynamic_params.get("atr_multiplier", 2.0),
                "profit_ratio": dynamic_params.get("profit_ratio", 0.05),
                "current_volatility": row.get("volatility", 0)
            }
            risk_params = base_risk_params
            try:
                risk_params = backtester.risk_manager.compute_risk_parameters_by_regime(
                    base_risk_params,
                    row.get('market_regime', 'unknown'),
                    dynamic_params.get('liquidity_info', 'high')
                )
            except Exception as e:
                logger.error(f"Risk parameter computation error {e}", exc_info=True)
                risk_params = base_risk_params
            try:
                if action == "enter_long":
                    backtester.process_bullish_entry(current_time, row, risk_params, dynamic_params)
                    log_util.log_event("Order executed: enter_long", state_key="order_execution")
                elif action == "exit_all":
                    backtester.process_bearish_exit(current_time, row)
                    log_util.log_event("Order executed: exit_all", state_key="order_execution")
                elif row.get('market_regime', 'unknown') == "sideways":
                    backtester.process_sideways_trade(current_time, row, risk_params, dynamic_params)
                    log_util.log_event("Order executed: sideways", state_key="order_execution")
            except Exception as e:
                logger.error(f"Error processing order with action '{action}': {e}", exc_info=True)
            backtester.last_signal_time = current_time

            try:
                backtester.update_positions(current_time, row)
            except Exception as e:
                logger.error(f"Error updating positions {e}", exc_info=True)

            try:
                if backtester.last_rebalance_time is None or (current_time - backtester.last_rebalance_time) >= rebalance_interval:
                    try:
                        backtester.asset_manager.rebalance(row.get('market_regime', 'unknown'))
                    except Exception as e:
                        logger.error(f"Error during rebalance {e}", exc_info=True)
                    backtester.last_rebalance_time = current_time
                log_util.log_event("Rebalance executed", state_key="rebalance")
            except Exception as e:
                logger.error(f"Error in rebalance check {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Unexpected error during processing training orders {e}", exc_info=True)
            continue

def process_extra_orders(backtester, dynamic_params):
    if backtester.df_extra is not None and not backtester.df_extra.empty:
        for current_time, row in backtester.df_extra.iterrows():
            try:
                hf_signal = get_signal_with_weekly_override(backtester, row, current_time, dynamic_params)
                regime = "sideways"
                try:
                    regime = backtester.df_long.loc[backtester.df_long.index <= current_time].iloc[-1].get('market_regime', 'sideways')
                except Exception as e:
                    logger.error(f"Retrieving regime failed {e}", exc_info=True)
                    regime = "sideways"
                base_risk_params = {
                    "risk_per_trade": dynamic_params.get("risk_per_trade", 0.01),
                    "atr_multiplier": dynamic_params.get("atr_multiplier", 2.0),
                    "profit_ratio": dynamic_params.get("profit_ratio", 0.05),
                    "current_volatility": row.get("volatility", 0)
                }
                risk_params = base_risk_params
                try:
                    risk_params = backtester.risk_manager.compute_risk_parameters_by_regime(
                        base_risk_params,
                        regime,
                        dynamic_params.get('liquidity_info', 'high')
                    )
                except Exception as e:
                    logger.error(f"Risk params error (extra data) {e}", exc_info=True)
                    risk_params = base_risk_params
                try:
                    if hf_signal == "enter_long":
                        backtester.process_bullish_entry(current_time, row, risk_params, dynamic_params)
                        log_util.log_event("Extra: Order executed: enter_long", state_key="order_execution")
                    elif hf_signal == "exit_all":
                        backtester.process_bearish_exit(current_time, row)
                        log_util.log_event("Extra: Order executed: exit_all", state_key="order_execution")
                except Exception as e:
                    logger.error(f"Error processing extra order with hf_signal '{hf_signal}': {e}", exc_info=True)
                try:
                    backtester.monitor_orders(current_time, row)
                except Exception as e:
                    logger.error(f"Error monitoring orders {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Unexpected error in process_extra_orders {e}", exc_info=True)
                continue

def process_holdout_orders(backtester, dynamic_params, df_holdout):
    if df_holdout is not None:
        for current_time, row in df_holdout.iterrows():
            try:
                action = get_signal_with_weekly_override(backtester, row, current_time, dynamic_params)
                base_risk_params = {
                    "risk_per_trade": dynamic_params.get("risk_per_trade", 0.01),
                    "atr_multiplier": dynamic_params.get("atr_multiplier", 2.0),
                    "profit_ratio": dynamic_params.get("profit_ratio", 0.05),
                    "current_volatility": row.get("volatility", 0)
                }
                risk_params = base_risk_params
                try:
                    risk_params = backtester.risk_manager.compute_risk_parameters_by_regime(
                        base_risk_params,
                        row.get('market_regime', 'unknown'),
                        dynamic_params.get('liquidity_info', 'high')
                    )
                except Exception as e:
                    logger.error(f"Risk params error (holdout) {e}", exc_info=True)
                    risk_params = base_risk_params
                try:
                    if action == "enter_long":
                        backtester.process_bullish_entry(current_time, row, risk_params, dynamic_params)
                        log_util.log_event("Holdout: Order executed: enter_long", state_key="order_execution")
                    elif action == "exit_all":
                        backtester.process_bearish_exit(current_time, row)
                        log_util.log_event("Holdout: Order executed: exit_all", state_key="order_execution")
                    elif row.get('market_regime', 'unknown') == "sideways":
                        backtester.process_sideways_trade(current_time, row, risk_params, dynamic_params)
                        log_util.log_event("Holdout: Order executed: sideways", state_key="order_execution")
                except Exception as e:
                    logger.error(f"Error processing holdout order with action '{action}': {e}", exc_info=True)
                try:
                    backtester.update_positions(current_time, row)
                except Exception as e:
                    logger.error(f"Error updating positions in holdout {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Unexpected error in process_holdout_orders {e}", exc_info=True)
                continue

def finalize_orders(backtester):
    try:
        backtester.finalize_all_positions()
    except Exception as e:
        logger.error(f"Error finalizing orders: {e}", exc_info=True)
        raise
