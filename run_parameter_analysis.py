# run_parameter_analysis.py
import argparse
import logging
import numpy as np
from logs.logger_config import setup_logger, initialize_root_logger, shutdown_logging
from logs.logging_util import LoggingUtil
from strategies.param_analysis import run_sensitivity_analysis
from logs.final_report import generate_parameter_sensitivity_report
from data.db.db_manager import get_unique_symbol_list, get_date_range
from data.db.db_config import DATABASE

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run parameter sensitivity analysis for trading strategies."
    )
    parser.add_argument("--param_names", type=str, 
                        default="profit_ratio,atr_multiplier,risk_per_trade,scale_in_threshold,weekly_breakout_threshold,weekly_momentum_threshold",
                        help="Comma-separated list of parameter names to analyze.")
    parser.add_argument("--param_steps", type=int, default=3, 
                        help="Number of steps for each parameter (default: 3)")
    parser.add_argument("--assets", type=str, default="",
                        help="Optional: Comma-separated list of assets. If not provided, DB will be queried for unique symbols.")
    parser.add_argument("--short_tf", type=str, default="4h", 
                        help="Short time frame (default: 4h)")
    parser.add_argument("--long_tf", type=str, default="1d", 
                        help="Long time frame (default: 1d)")
    parser.add_argument("--periods", type=str, default="", 
                        help="Optional multiple periods in format start1:end1;start2:end2, etc.")
    # 새 인자: 분석 기간 축소를 위한 인자 (예: "2022-01-01:2023-01-01")
    parser.add_argument("--analysis_period", type=str, default="2022-01-01:2023-01-01",
                        help="Analysis period in format start_date:end_date to reduce overall analysis period.")
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

def get_default_date_range(symbol: str, timeframe: str = "1d") -> tuple:
    symbol_key = symbol.replace("/", "").lower()
    table_name = f"ohlcv_{symbol_key}_{timeframe}"
    start_date, end_date = get_date_range(table_name, DATABASE)
    # analysis_period 인자를 사용하므로 기본값을 무시합니다.
    if start_date is None or end_date is None:
        start_date, end_date = "2022-01-01 00:00:00", "2023-01-01 23:59:59"
    return start_date, end_date

def run_parameter_analysis():
    LoggingUtil.clear_log_files()
    initialize_root_logger()

    args = parse_args()
    logger = setup_logger(__name__)
    logger.info("Starting parameter sensitivity analysis.")

    if args.assets:
        assets = parse_assets(args.assets)
    else:
        assets = get_unique_symbol_list() or ["BTC/USDT"]
    logger.info(f"Assets for analysis: {assets}")

    # 분석 기간 인자 파싱 (예: "2022-01-01:2023-01-01")
    try:
        analysis_start, analysis_end = args.analysis_period.split(":")
        analysis_start = analysis_start.strip() + " 00:00:00"
        analysis_end = analysis_end.strip() + " 23:59:59"
    except Exception as e:
        logger.error(f"Error parsing analysis_period: {e}", exc_info=True)
        analysis_start, analysis_end = "2022-01-01 00:00:00", "2023-01-01 23:59:59"
    logger.info(f"Using analysis period: {analysis_start} to {analysis_end}")

    # DB에서 날짜 범위를 조회할 필요 없이, analysis_period을 기본 날짜 범위로 사용
    default_start, default_end = analysis_start, analysis_end
    logger.info(f"Default analysis date range: {default_start} to {default_end}")
    periods = parse_periods(args.periods, default_start, default_end)

    from config.config_manager import ConfigManager
    cm = ConfigManager()
    defaults = cm.get_defaults()
    logger.info(f"Default parameters: {defaults}")
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
        start_val = default_val * 0.8
        end_val = default_val * 1.2
        param_settings[pname] = np.linspace(start_val, end_val, args.param_steps)
        logger.info(f"Analyzing {pname} over range {start_val:.4f} to {end_val:.4f}")

    results_all = run_sensitivity_analysis(
        param_settings, assets, args.short_tf, args.long_tf, default_start, default_end, periods
    )
    report_title = "Multi-Parameter Analysis: " + ", ".join([str(k) for k in results_all.keys()])
    generate_parameter_sensitivity_report(report_title, results_all)
    shutdown_logging()

if __name__ == "__main__":
    run_parameter_analysis()
