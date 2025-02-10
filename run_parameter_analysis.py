# run_parameter_analysis.py
import logging
import numpy as np
from logs.logger_config import setup_logger
from logs.logging_util import LoggingUtil
from strategy_tuning.parameter_analysis import run_sensitivity_analysis
from logs.final_report import generate_parameter_sensitivity_report

def run_parameter_analysis():
    # 1. 기존 로그 핸들러 종료 및 글로벌 핸들러 초기화
    logging.shutdown()

    # 2. logs 폴더의 모든 로그 파일 삭제 (LoggingUtil의 정적 메서드 사용)
    LoggingUtil.clear_log_files()

    # 3. logger 재설정 및 시작 로그 출력
    logger = setup_logger(__name__)
    logger.info("프로젝트 전체 테스트 실행을 시작합니다. (기본 파라미터를 사용하여 자동 민감도 분석)")

    # 기본값 설정: 별도의 명령줄 인자 없이 자동 실행되도록 기본 파라미터를 사용
    default_param_name = "profit_ratio"
    default_start = 0.05
    default_end = 0.15
    default_steps = 10
    default_asset = "BTC/USDT"
    default_short_tf = "4h"
    default_long_tf = "1d"
    default_start_date = "2018-06-01"
    default_end_date = "2020-12-31"

    param_values = np.linspace(default_start, default_end, default_steps)
    logger.info("Starting parameter sensitivity analysis using default parameters...")
    results = run_sensitivity_analysis(
        param_name=default_param_name,
        param_range=param_values,
        asset=default_asset,
        short_tf=default_short_tf,
        long_tf=default_long_tf,
        start_date=default_start_date,
        end_date=default_end_date
    )
    
    # 최종 결과 리포트를 final_report 모듈의 함수를 이용해 생성 및 로그 출력
    generate_parameter_sensitivity_report(default_param_name, results)

if __name__ == "__main__":
    run_parameter_analysis()
