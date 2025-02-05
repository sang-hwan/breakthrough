# test_strategy_performance.py
"""
실행 파일 개요:
1. Walk-Forward 방식으로 파라미터 최적화를 수행하여 최적의 동적 파라미터를 도출합니다.
   - 학습 구간과 테스트 구간을 겹치게 구성하여 다양한 시장 국면에 대한 과적합 위험을 줄입니다.
2. 최적화된 파라미터를 사용하여 BTC/USDT, ETH/USDT, XRP/USDT 종목에 대해
   1일(1d), 4시간(4h) 단위의 OHLCV 데이터를 사용하여 백테스트를 진행합니다.
3. 백테스트 결과로 도출된 거래 내역을 바탕으로 성과(월별, 연도별, 전체 성과 등)를 콘솔 및 로그로 확인합니다.
4. HMM의 신뢰도에 따라 보조지표를 통한 레짐 판단 및 전략 분기(상승장, 하락장, 횡보장에 따른 차별 전략)를 적용합니다.
"""

import pandas as pd
from backtesting.optimizer import DynamicParameterOptimizer
from backtesting.backtester import Backtester
from backtesting.performance import print_performance_report
from logs.logger_config import setup_logger

def main():
    logger = setup_logger("test_strategy_performance")
    
    # 1. 파라미터 최적화 (Walk-Forward 방식)
    logger.info("Walk-Forward 방식의 파라미터 최적화를 시작합니다...")
    # (n_trials 값은 테스트 환경에 맞게 조정할 수 있습니다.)
    optimizer = DynamicParameterOptimizer(n_trials=10)
    best_trial = optimizer.optimize()
    best_params = best_trial.params
    logger.info("최적의 파라미터: %s", best_params)
    
    # 2. 최적의 파라미터를 사용하여 각 종목에 대해 백테스트 진행
    # 데이터 기간: 2018-06-01 ~ 2025-02-01
    start_date = "2018-06-01"
    end_date = "2025-02-01"
    # 사용될 타임프레임 (short: 4시간, long: 1일)
    timeframes = {"short_tf": "4h", "long_tf": "1d"}
    # 대상 심볼 목록
    symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT"]
    
    all_trades = []
    all_trade_logs = []
    
    for symbol in symbols:
        logger.info("심볼 %s에 대한 백테스트를 진행합니다.", symbol)
        backtester = Backtester(symbol=symbol, account_size=10000)
        try:
            # 데이터 테이블 이름 포맷: "ohlcv_{symbol}_{timeframe}"
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
            all_trades.extend(trades)
            all_trade_logs.extend(trade_logs)
            logger.info("심볼 %s 백테스트 완료: 총 거래 횟수 = %d", symbol, len(trades))
        except Exception as e:
            logger.error("심볼 %s 백테스트 실행 중 에러: %s", symbol, e)
            continue
    
    # 3. 백테스트 결과 성과 보고
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        # exit_time 컬럼이 문자열이면 datetime 변환
        if 'exit_time' in trades_df.columns:
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        logger.info("백테스트 성과 보고:")
        print_performance_report(trades_df, initial_balance=10000)
    else:
        logger.info("백테스트 결과, 생성된 거래 내역이 없습니다.")

if __name__ == "__main__":
    main()
