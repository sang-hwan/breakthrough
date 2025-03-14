# run_parameter_analysis.py
"""
이 스크립트는 거래 전략의 파라미터 민감도 분석을 수행합니다.
주요 기능은 명령줄 인자를 통해 분석 설정을 받고, 
각 파라미터에 대해 기본값의 80%에서 120% 범위 내에서 값들을 생성하여
민감도 분석을 실행하고, 결과 보고서를 생성하는 것입니다.
"""

import argparse  # 명령줄 인자 파싱을 위한 모듈
import logging  # 로깅 처리를 위한 모듈
import numpy as np  # 수치 계산 및 배열 처리 모듈
from logs.logger_config import setup_logger, initialize_root_logger, shutdown_logging  # 로깅 설정 관련 함수
from logs.logging_util import LoggingUtil  # 추가 로깅 유틸리티 함수
from strategies.param_analysis import run_sensitivity_analysis  # 파라미터 민감도 분석 실행 함수
from logs.final_report import generate_parameter_sensitivity_report  # 분석 결과 보고서 생성 함수
from data.db.db_manager import get_unique_symbol_list, get_date_range  # 데이터베이스 관련 함수
from data.db.db_config import DATABASE  # 데이터베이스 접속 정보

def parse_args():
    """
    명령줄 인자를 파싱하여 분석에 필요한 옵션들을 반환합니다.
    
    Returns:
        argparse.Namespace: 파라미터 이름, 단계, 자산, 시간 프레임 등 분석에 필요한 인자들을 포함한 객체
    """
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
    """
    콤마로 구분된 자산 문자열을 개별 자산 리스트로 변환합니다.
    
    Parameters:
        asset_str (str): 콤마로 구분된 자산 문자열 (예: "BTC/USDT,ETH/USDT")
    
    Returns:
        list: 개별 자산 문자열의 리스트
    """
    return [asset.strip() for asset in asset_str.split(",") if asset.strip()]

def parse_periods(periods_str, default_start, default_end):
    """
    기간 문자열을 파싱하여 (시작 날짜, 종료 날짜) 쌍의 리스트를 생성합니다.
    
    Parameters:
        periods_str (str): 세미콜론(;)으로 구분된 기간 쌍 (예: "start1:end1;start2:end2")
        default_start (str): 기본 시작 날짜 (분석 기간 인자가 없을 경우 사용)
        default_end (str): 기본 종료 날짜 (분석 기간 인자가 없을 경우 사용)
    
    Returns:
        list: (시작 날짜, 종료 날짜) 튜플 리스트
    """
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
    """
    특정 심볼과 시간 프레임에 대해 데이터베이스에서 데이터 날짜 범위를 조회합니다.
    
    Parameters:
        symbol (str): 거래 심볼 (예: "BTC/USDT")
        timeframe (str): 데이터의 시간 간격 (기본값: "1d")
    
    Returns:
        tuple: (시작 날짜, 종료 날짜) 문자열. 데이터가 없으면 기본 날짜 범위("2022-01-01", "2023-01-01")를 반환합니다.
    """
    symbol_key = symbol.replace("/", "").lower()  # 테이블명 생성을 위한 심볼 변환
    table_name = f"ohlcv_{symbol_key}_{timeframe}"
    start_date, end_date = get_date_range(table_name, DATABASE)
    # 데이터가 없는 경우 기본 날짜 범위로 설정합니다.
    if start_date is None or end_date is None:
        start_date, end_date = "2022-01-01 00:00:00", "2023-01-01 23:59:59"
    return start_date, end_date

def run_parameter_analysis():
    """
    거래 전략 파라미터의 민감도를 분석합니다.
    
    주요 단계:
      1. 기존 로깅 파일 삭제 및 로거 초기화
      2. 명령줄 인자 파싱 및 자산, 분석 기간 설정
      3. 기본 파라미터값의 80% ~ 120% 범위 내에서 각 파라미터의 분석 값 생성
      4. 민감도 분석 실행 및 결과 보고서 생성
      5. 로깅 종료
      
    Parameters:
        없음
    
    Returns:
        None
    """
    # 기존 로깅 파일 초기화 및 로깅 설정 재설정
    LoggingUtil.clear_log_files()
    initialize_root_logger()

    args = parse_args()  # 명령줄 인자 파싱
    logger = setup_logger(__name__)
    logger.info("Starting parameter sensitivity analysis.")

    # 자산 목록 결정: 인자로 주어진 자산 사용, 없으면 DB에서 조회
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

    # 분석 기간을 기본 날짜 범위로 사용
    default_start, default_end = analysis_start, analysis_end
    logger.info(f"Default analysis date range: {default_start} to {default_end}")
    periods = parse_periods(args.periods, default_start, default_end)

    from config.config_manager import ConfigManager
    cm = ConfigManager()
    defaults = cm.get_defaults()  # 기본 파라미터 값을 조회합니다.
    logger.info(f"Default parameters: {defaults}")
    
    # 분석할 파라미터 이름 리스트 생성
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
        start_val = default_val * 0.8  # 기본값의 80%
        end_val = default_val * 1.2    # 기본값의 120%
        # 지정된 단계 수(param_steps) 만큼 균등 간격의 값을 생성합니다.
        param_settings[pname] = np.linspace(start_val, end_val, args.param_steps)
        logger.info(f"Analyzing {pname} over range {start_val:.4f} to {end_val:.4f}")

    # 파라미터 민감도 분석 실행
    results_all = run_sensitivity_analysis(
        param_settings, assets, args.short_tf, args.long_tf, default_start, default_end, periods
    )
    # 결과에 기반하여 보고서 제목 생성 후 최종 보고서 생성
    report_title = "Multi-Parameter Analysis: " + ", ".join([str(k) for k in results_all.keys()])
    generate_parameter_sensitivity_report(report_title, results_all)
    shutdown_logging()  # 로깅 시스템 종료

if __name__ == "__main__":
    run_parameter_analysis()
