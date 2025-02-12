# run_parameter_analysis.py
import argparse
import logging
import numpy as np
from logs.logger_config import setup_logger
from logs.logging_util import LoggingUtil
from strategy_tuning.parameter_analysis import run_sensitivity_analysis
from logs.final_report import generate_parameter_sensitivity_report

logger = setup_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run parameter sensitivity analysis for trading strategies."
    )
    # 다중 파라미터 분석용 인자 (기본값 지정)
    parser.add_argument("--param_names", type=str, 
                        default="profit_ratio,atr_multiplier,risk_per_trade,scale_in_threshold,weekly_breakout_threshold,weekly_momentum_threshold",
                        help="Comma-separated list of parameter names to analyze. Multi-parameter mode is activated.")
    parser.add_argument("--param_steps", type=int, default=10, 
                        help="Number of steps for each parameter (default: 10)")
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
    # sample_rate 인자는 더 이상 사용하지 않습니다.
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
                logger.error(f"Error parsing period pair '{pair}': {e}", exc_info=True)
    if not period_list:
        period_list = [(default_start, default_end)]
    return period_list

def run_parameter_analysis():
    # 1. 기존 로그 핸들러 종료 및 로그 파일 삭제
    logging.shutdown()
    LoggingUtil.clear_log_files()

    # 2. 인자 파싱
    args = parse_args()
    logger = setup_logger(__name__)
    logger.debug("Starting parameter sensitivity analysis with external configuration.")

    assets = parse_assets(args.assets)
    periods = parse_periods(args.periods, args.start_date, args.end_date)

    # 다중 파라미터 분석: --param_names 인자를 기반으로 분석할 파라미터와 그 범위를 dict로 구성합니다.
    from strategy_tuning.dynamic_param_manager import DynamicParamManager
    dpm = DynamicParamManager()
    defaults = dpm.get_default_params()
    param_names = [p.strip() for p in args.param_names.split(",") if p.strip()]
    param_settings = {}
    for pname in param_names:
        if pname not in defaults:
            logger.warning(f"Parameter {pname} not found in default parameters. Skipping.")
            continue
        try:
            default_val = float(defaults[pname])
        except Exception as e:
            logger.warning(f"Parameter {pname} is not numeric. Skipping.")
            continue
        # 기본값의 ±20% 범위로 분석 (필요에 따라 조정 가능)
        start_val = default_val * 0.8
        end_val = default_val * 1.2
        param_values = np.linspace(start_val, end_val, args.param_steps)
        logger.debug(f"Analyzing parameter {pname} with range {start_val:.4f} to {end_val:.4f} in {args.param_steps} steps.")
        param_settings[pname] = param_values

    # 3. 민감도 분석 실행 (다중 파라미터 모드)
    results_all = run_sensitivity_analysis(param_settings, assets, args.short_tf, args.long_tf, args.start_date, args.end_date, periods)

    # 4. 최종 결과 리포트 생성: 다중 파라미터 분석 결과 리포트를 생성합니다.
    report_title = "Multi-Parameter Analysis: " + ", ".join(results_all.keys())
    generate_parameter_sensitivity_report(report_title, results_all)

if __name__ == "__main__":
    run_parameter_analysis()
