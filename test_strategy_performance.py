# test_strategy_performance.py
"""
실행 파일 개요:
1. Walk-Forward 방식으로 파라미터 최적화를 수행하여 최적의 동적 파라미터를 도출합니다.
   - 학습 구간과 테스트 구간을 겹치게 구성하여 다양한 시장 국면에 대한 과적합 위험을 줄입니다.
2. 최적화된 파라미터를 사용하여 BTC/USDT, ETH/USDT, XRP/USDT 종목에 대해
   1일(1d), 4시간(4h) 단위의 OHLCV 데이터를 사용하여 백테스트를 진행합니다.
3. 백테스트 결과로 도출된 각 종목별 거래 내역을 바탕으로 개별 성과 리포트를 로그에 기록합니다.
4. HMM의 신뢰도에 따라 보조지표를 통한 레짐 판단 및 전략 분기(상승장, 하락장, 횡보장에 따른 차별 전략)를 적용합니다.
"""

import os
import glob
import logging
import pandas as pd
from backtesting.optimizer import DynamicParameterOptimizer
from backtesting.backtester import Backtester
from backtesting.performance import print_performance_report
from logs.logger_config import setup_logger

def clear_logs():
    """
    실행 전에 logs 디렉토리 내 .log 확장자를 가진 파일들만 삭제합니다.
    """
    log_dir = "logs"
    pattern = os.path.join(log_dir, "*.log")
    log_files = glob.glob(pattern)
    for log_file in log_files:
        try:
            os.remove(log_file)
        except Exception as e:
            print(f"로그 파일 {log_file} 삭제 중 오류 발생: {e}")

def main():
    # 먼저 모든 로거 핸들러를 종료한 후 로그 삭제
    logging.shutdown()
    clear_logs()
    
    logger = setup_logger("test_strategy_performance")
    
    # 1. 파라미터 최적화 (Walk-Forward 방식)
    logger.info("Walk-Forward 방식의 파라미터 최적화를 시작합니다...")
    optimizer = DynamicParameterOptimizer(n_trials=10)
    best_trial = optimizer.optimize()
    best_params = best_trial.params
    logger.info("최적의 파라미터: %s", best_params)
    
    # 2. 각 종목별 백테스트 진행 및 개별 성과 리포트 생성
    start_date = "2018-06-01"
    end_date = "2025-02-01"
    timeframes = {"short_tf": "4h", "long_tf": "1d"}
    symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT"]
    
    for symbol in symbols:
        logger.info("심볼 %s에 대한 백테스트를 진행합니다.", symbol)
        backtester = Backtester(symbol=symbol, account_size=10000)
        try:
            backtester.load_data(
                short_table_format="ohlcv_{symbol}_{timeframe}",
                long_table_format="ohlcv_{symbol}_{timeframe}",
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
        
        # 종목별 거래 내역이 있으면 개별 성과 리포트 생성
        if trades:
            symbol_trades_df = pd.DataFrame(trades)
            if 'exit_time' in symbol_trades_df.columns:
                symbol_trades_df['exit_time'] = pd.to_datetime(symbol_trades_df['exit_time'])
            logger.info("종목 %s 백테스트 성과 보고:", symbol)
            print_performance_report(symbol_trades_df, initial_balance=10000, symbol=symbol)
        else:
            logger.info("심볼 %s 백테스트 결과, 생성된 거래 내역이 없습니다.", symbol)
    
if __name__ == "__main__":
    main()
