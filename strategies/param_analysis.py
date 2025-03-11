# strategies/param_analysis.py
import numpy as np
from itertools import product
import random
from logs.logger_config import setup_logger
from backtesting.backtester import Backtester
from config.config_manager import ConfigManager

logger = setup_logger(__name__)

def run_sensitivity_analysis(param_settings,
                             assets,
                             short_tf="4h", long_tf="1d",
                             start_date="2018-06-01", end_date="2020-12-31",
                             periods=None,
                             base_dynamic_params=None,
                             max_combinations=20):
    if periods is None:
        periods = [(start_date, end_date)]
    if base_dynamic_params is None:
        base_dynamic_params = ConfigManager().get_defaults()

    logger.debug(f"Starting sensitivity analysis over assets: {assets}")
    results = {}
    param_names = list(param_settings.keys())
    combinations = list(product(*(param_settings[name] for name in param_names)))
    if len(combinations) > max_combinations:
        combinations = random.sample(combinations, max_combinations)

    for combo in combinations:
        dynamic_params = base_dynamic_params.copy()
        combo_key = tuple(sorted(zip(param_names, combo)))
        run_metrics = []
        for asset in assets:
            symbol_key = asset.replace("/", "").lower()
            for s_date, e_date in periods:
                try:
                    bt = Backtester(symbol=asset, account_size=10000)
                    bt.load_data(
                        short_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                        long_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                        short_tf=short_tf, long_tf=long_tf,
                        start_date=s_date, end_date=e_date
                    )
                    for name, val in combo_key:
                        dynamic_params[name] = val
                    trades, _ = bt.run_backtest_pipeline(dynamic_params=dynamic_params)
                    logger.info(f"{asset} {s_date}~{e_date}: {len(trades)} trades executed.")
                    if trades:
                        roi = sum(trade.get("pnl", 0) for trade in trades) / 10000 * 100
                        logger.info(f"{asset} {s_date}~{e_date}: ROI={roi:.2f}%")
                        from backtesting.performance import compute_performance
                        perf = compute_performance(trades)
                        run_metrics.append(perf)
                    else:
                        logger.warning(f"{asset} {s_date}~{e_date}: No trades executed.")
                except Exception as e:
                    logger.error(f"Error during sensitivity analysis for {asset} with combination {combo_key}: {e}", exc_info=True)
                    continue
        if run_metrics:
            aggregated = {}
            for key in ["roi", "sharpe_ratio", "max_drawdown", "trade_count", "cumulative_return", "total_pnl"]:
                values = [run.get(key, 0) for run in run_metrics]
                aggregated[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
            results[combo_key] = aggregated
        else:
            results[combo_key] = None
    return results
