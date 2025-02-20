# run_parameter_analysis.py
import argparse
import logging
import numpy as np
from logs.logger_config import setup_logger, initialize_root_logger, shutdown_logging
from logs.logging_util import LoggingUtil
from strategies.param_analysis import run_sensitivity_analysis
from logs.final_report import generate_parameter_sensitivity_report

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run parameter sensitivity analysis for trading strategies."
    )
    parser.add_argument("--param_names", type=str, 
                        default="profit_ratio,atr_multiplier,risk_per_trade,scale_in_threshold,weekly_breakout_threshold,weekly_momentum_threshold",
                        help="Comma-separated list of parameter names to analyze.")
    parser.add_argument("--param_steps", type=int, default=3, 
                        help="Number of steps for each parameter (default: 3)")
    parser.add_argument("--assets", type=str, default="BTC/USDT", 
                        help="Comma-separated list of assets (default: BTC/USDT)")
    parser.add_argument("--short_tf", type=str, default="4h", 
                        help="Short time frame (default: 4h)")
    parser.add_argument("--long_tf", type=str, default="1d", 
                        help="Long time frame (default: 1d)")
    parser.add_argument("--start_date", type=str, default="2018-06-01", 
                        help="Start date for data")
    parser.add_argument("--end_date", type=str, default="2020-12-31", 
                        help="End date for data")
    parser.add_argument("--periods", type=str, default="", 
                        help="Optional multiple periods in format start1:end1;start2:end2, etc.")
    return parser.parse_args()

def parse_assets(asset_str):
    return [asset.strip() for asset in asset_str.split(",") if asset.strip()]

def parse_periods(periods_str, default_start, default_end):
    if not periods_str:
        return [(default_start, default_end)]
    period_list = []
    for pair in periods_str.split(";"):
        if pair:
            try:
                s, e = pair.split(":")
                period_list.append((s.strip(), e.strip()))
            except Exception as e:
                logging.getLogger(__name__).error(f"Error parsing period pair '{pair}': {e}", exc_info=True)
    return period_list if period_list else [(default_start, default_end)]

def run_parameter_analysis():
    LoggingUtil.clear_log_files()
    initialize_root_logger()

    args = parse_args()
    logger = setup_logger(__name__)
    logger.info("Starting parameter sensitivity analysis.")

    assets = parse_assets(args.assets)
    periods = parse_periods(args.periods, args.start_date, args.end_date)

    from config.config_manager import ConfigManager
    cm = ConfigManager()
    defaults = cm.get_defaults()
    param_names = [p.strip() for p in args.param_names.split(",") if p.strip()]
    param_settings = {}
    for pname in param_names:
        if pname not in defaults:
            logger.warning(f"Parameter {pname} not found in default parameters. Skipping.")
            continue
        try:
            default_val = float(defaults[pname])
        except Exception:
            logger.warning(f"Parameter {pname} is not numeric. Skipping.")
            continue
        start_val = default_val * 0.9
        end_val = default_val * 1.1
        param_settings[pname] = np.linspace(start_val, end_val, args.param_steps)
        logger.info(f"Analyzing {pname} over range {start_val:.4f} to {end_val:.4f}")

    results_all = run_sensitivity_analysis(
        param_settings, assets, args.short_tf, args.long_tf, args.start_date, args.end_date, periods
    )
    report_title = "Multi-Parameter Analysis: " + ", ".join([str(k) for k in results_all.keys()])
    generate_parameter_sensitivity_report(report_title, results_all)
    shutdown_logging()

if __name__ == "__main__":
    run_parameter_analysis()
