# run_strategy_performance.py
import os
import glob
import logging
import pandas as pd
from backtesting.optimizer import DynamicParameterOptimizer
from backtesting.backtester import Backtester
from backtesting.performance import compute_performance
from logs.final_report import generate_final_report
from logs.logger_config import setup_logger
from dynamic_parameters.dynamic_param_manager import DynamicParamManager

def clear_log_files():
    """
    실행 시 logs 폴더 내 모든 .log 파일을 삭제합니다.
    """
    # 현재 파일의 절대 경로를 기준으로 logs 폴더 경로 계산
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    for log_file in log_files:
        try:
            os.remove(log_file)
            print(f"Deleted log file: {log_file}")
        except Exception as e:
            print(f"Failed to remove log file {log_file}: {e}")

def run_strategy_performance():
    # 1. 기존 로그 핸들러 종료 및 글로벌 핸들러 초기화
    logging.shutdown()

    # 2. logs 폴더의 모든 로그 파일 삭제
    clear_log_files()

    logger = setup_logger(__name__)
    logger.info("프로젝트 전체 테스트 실행을 시작합니다.")
    
    # 3. 파라미터 최적화 (Walk-Forward 방식)
    logger.info("Walk-Forward 방식의 파라미터 최적화를 시작합니다...")
    optimizer = DynamicParameterOptimizer(n_trials=10)
    best_trial = optimizer.optimize()
    
    # 4. 기본 파라미터와 최적화된 파라미터 병합
    dynamic_manager = DynamicParamManager()
    best_params = dynamic_manager.merge_params(best_trial.params)
    logger.info("최적의 파라미터 도출 완료: %s", best_params)
    
    # 5. 각 종목별 백테스트 실행 및 성과 계산
    start_date = "2018-06-01"
    end_date = "2025-02-01"
    timeframes = {"short_tf": "4h", "long_tf": "1d"}
    symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT"]
    
    for symbol in symbols:
        logger.info("심볼 %s 백테스트 시작", symbol)
        try:
            symbol_key = symbol.replace("/", "").lower()
            backtester = Backtester(symbol=symbol, account_size=10000)
            backtester.load_data(
                short_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                long_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                short_tf=timeframes["short_tf"],
                long_tf=timeframes["long_tf"],
                start_date=start_date,
                end_date=end_date
            )
        except Exception as e:
            logger.error("심볼 %s 데이터 로드 실패: %s", symbol, e)
            continue
        
        try:
            trades, trade_logs = backtester.run_backtest(dynamic_params=best_params)
            logger.info("심볼 %s 백테스트 완료: 총 거래 횟수 = %d", symbol, len(trades))
        except Exception as e:
            logger.error("심볼 %s 백테스트 실행 중 에러: %s", symbol, e)
            continue
        
        if trades:
            performance_data = compute_performance(trades)
            logger.info("심볼 %s 성과 보고 생성", symbol)
            generate_final_report(performance_data, symbol=symbol)
        else:
            logger.info("심볼 %s 백테스트 결과, 생성된 거래 내역이 없습니다.", symbol)
    
    logger.info("전체 프로젝트 테스트 실행 완료.")

if __name__ == '__main__':
    run_strategy_performance()
