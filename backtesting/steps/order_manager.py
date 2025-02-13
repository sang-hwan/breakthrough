# backtesting/steps/order_manager.py
from logs.logger_config import setup_logger

logger = setup_logger(__name__)

def process_training_orders(backtester, dynamic_params, signal_cooldown, rebalance_interval):
    for current_time, row in backtester.df_train.iterrows():
        if current_time.weekday() == 4:
            if backtester.last_weekly_close_date is None or backtester.last_weekly_close_date != current_time.date():
                try:
                    backtester.handle_weekly_end(current_time, row)
                except Exception as e:
                    logger.error(f"Weekly end handling failed at {current_time}: {e}", exc_info=True)
                backtester.last_weekly_close_date = current_time.date()
                continue
        if backtester.walk_forward_days is not None and current_time - backtester.window_start >= backtester.walk_forward_td:
            try:
                backtester.handle_walk_forward_window(current_time, row)
            except Exception as e:
                logger.error(f"Walk-forward window handling failed at {current_time}: {e}", exc_info=True)
            backtester.window_start = current_time
        if backtester.last_signal_time is None or (current_time - backtester.last_signal_time) >= signal_cooldown:
            try:
                action = backtester.ensemble_manager.get_final_signal(
                    row['market_regime'], 
                    dynamic_params.get('liquidity_info', 'high'), 
                    backtester.df_short, 
                    current_time,
                    data_weekly=backtester.df_weekly
                )
            except Exception as e:
                logger.error(f"Final signal generation failed at {current_time}: {e}", exc_info=True)
                action = "hold"
            base_risk_params = {
                "risk_per_trade": dynamic_params.get("risk_per_trade", 0.01),
                "atr_multiplier": dynamic_params.get("atr_multiplier", 2.0),
                "profit_ratio": dynamic_params.get("profit_ratio", 0.05),
                "current_volatility": row.get("volatility", 0)
            }
            try:
                risk_params = backtester.risk_manager.compute_risk_parameters_by_regime(
                    base_risk_params,
                    regime=row['market_regime'],
                    liquidity=dynamic_params.get('liquidity_info', 'high')
                )
            except Exception as e:
                logger.error(f"Risk parameter computation failed at {current_time}: {e}", exc_info=True)
                risk_params = base_risk_params
            if action == "enter_long":
                try:
                    backtester.process_bullish_entry(current_time, row, risk_params, dynamic_params)
                except Exception as e:
                    logger.error(f"Bullish entry processing failed at {current_time}: {e}", exc_info=True)
            elif action == "exit_all":
                try:
                    backtester.process_bearish_exit(current_time, row)
                except Exception as e:
                    logger.error(f"Bearish exit processing failed at {current_time}: {e}", exc_info=True)
            elif row['market_regime'] == "sideways":
                try:
                    backtester.process_sideways_trade(current_time, row, risk_params, dynamic_params)
                except Exception as e:
                    logger.error(f"Sideways trade processing failed at {current_time}: {e}", exc_info=True)
            backtester.last_signal_time = current_time
        try:
            backtester.update_positions(current_time, row)
        except Exception as e:
            logger.error(f"Position update failed at {current_time}: {e}", exc_info=True)
        if (backtester.last_rebalance_time is None) or ((current_time - backtester.last_rebalance_time) >= rebalance_interval):
            try:
                backtester.asset_manager.rebalance(row['market_regime'])
            except Exception as e:
                logger.error(f"Asset rebalancing failed at {current_time}: {e}", exc_info=True)
            backtester.last_rebalance_time = current_time

def process_extra_orders(backtester, dynamic_params):
    if backtester.df_extra is not None and not backtester.df_extra.empty:
        for current_time, row in backtester.df_extra.iterrows():
            try:
                hf_signal = backtester.ensemble_manager.get_final_signal(
                    row['market_regime'], 
                    dynamic_params.get('liquidity_info', 'high'), 
                    backtester.df_short, 
                    current_time,
                    data_weekly=backtester.df_weekly
                )
            except Exception as e:
                logger.error(f"High-frequency signal generation failed at {current_time}: {e}", exc_info=True)
                hf_signal = "hold"
            try:
                regime = backtester.df_long.loc[backtester.df_long.index <= current_time].iloc[-1].get('market_regime', 'sideways')
            except Exception as e:
                logger.error(f"Retrieving market regime failed at {current_time}: {e}", exc_info=True)
                regime = "sideways"
            base_risk_params = {
                "risk_per_trade": dynamic_params.get("risk_per_trade", 0.01),
                "atr_multiplier": dynamic_params.get("atr_multiplier", 2.0),
                "profit_ratio": dynamic_params.get("profit_ratio", 0.05),
                "current_volatility": row.get("volatility", 0)
            }
            try:
                risk_params = backtester.risk_manager.compute_risk_parameters_by_regime(
                    base_risk_params,
                    regime=regime,
                    liquidity=dynamic_params.get('liquidity_info', 'high')
                )
            except Exception as e:
                logger.error(f"Risk parameter computation (extra data) failed at {current_time}: {e}", exc_info=True)
                risk_params = base_risk_params
            if hf_signal == "enter_long":
                try:
                    backtester.process_bullish_entry(current_time, row, risk_params, dynamic_params)
                except Exception as e:
                    logger.error(f"High-frequency bullish entry failed at {current_time}: {e}", exc_info=True)
            elif hf_signal == "exit_all":
                try:
                    backtester.process_bearish_exit(current_time, row)
                except Exception as e:
                    logger.error(f"High-frequency bearish exit failed at {current_time}: {e}", exc_info=True)
            try:
                backtester.monitor_orders(current_time, row)
            except Exception as e:
                logger.error(f"Order monitoring failed at {current_time}: {e}", exc_info=True)

def process_holdout_orders(backtester, dynamic_params, df_holdout):
    if df_holdout is not None:
        for current_time, row in df_holdout.iterrows():
            try:
                action = backtester.ensemble_manager.get_final_signal(
                    row['market_regime'], 
                    dynamic_params.get('liquidity_info', 'high'), 
                    backtester.df_short, 
                    current_time,
                    data_weekly=backtester.df_weekly
                )
            except Exception as e:
                logger.error(f"Final signal generation (holdout) failed at {current_time}: {e}", exc_info=True)
                action = "hold"
            base_risk_params = {
                "risk_per_trade": dynamic_params.get("risk_per_trade", 0.01),
                "atr_multiplier": dynamic_params.get("atr_multiplier", 2.0),
                "profit_ratio": dynamic_params.get("profit_ratio", 0.05),
                "current_volatility": row.get("volatility", 0)
            }
            try:
                risk_params = backtester.risk_manager.compute_risk_parameters_by_regime(
                    base_risk_params,
                    regime=row['market_regime'],
                    liquidity=dynamic_params.get('liquidity_info', 'high')
                )
            except Exception as e:
                logger.error(f"Risk parameter computation (holdout) failed at {current_time}: {e}", exc_info=True)
                risk_params = base_risk_params
            if action == "enter_long":
                try:
                    backtester.process_bullish_entry(current_time, row, risk_params, dynamic_params)
                except Exception as e:
                    logger.error(f"Bullish entry (holdout) failed at {current_time}: {e}", exc_info=True)
            elif action == "exit_all":
                try:
                    backtester.process_bearish_exit(current_time, row)
                except Exception as e:
                    logger.error(f"Bearish exit (holdout) failed at {current_time}: {e}", exc_info=True)
            elif row['market_regime'] == "sideways":
                try:
                    backtester.process_sideways_trade(current_time, row, risk_params, dynamic_params)
                except Exception as e:
                    logger.error(f"Sideways trade (holdout) failed at {current_time}: {e}", exc_info=True)
            try:
                backtester.update_positions(current_time, row)
            except Exception as e:
                logger.error(f"Position update (holdout) failed at {current_time}: {e}", exc_info=True)

def finalize_orders(backtester):
    backtester.finalize_all_positions()
