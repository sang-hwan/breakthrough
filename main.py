# main.py
import pandas as pd
from backtesting.backtester import Backtester
from backtesting.optimizer import DynamicParameterOptimizer
from backtesting.performance import print_performance_report

def main():
    # 동적 파라미터 최적화 실행
    optimizer = DynamicParameterOptimizer(n_trials=50)
    best_trial = optimizer.optimize()
    print("최적 동적 파라미터:", best_trial.params)
    
    # 최적 파라미터를 적용하여 백테스트 실행 (예: 2018-01-01 ~ 2021-01-01)
    backtester = Backtester(symbol="BTC/USDT", account_size=10000)
    backtester.load_data("ohlcv_{symbol}_{timeframe}", "ohlcv_{symbol}_{timeframe}", "4h", "1d", "2018-01-01", "2021-01-01")
    
    # 예시 시장 상황 데이터
    market_data = {"volatility": 0.07, "trend_strength": 0.5}
    dynamic_params = backtester.dynamic_param_manager.update_dynamic_params(market_data)
    dynamic_params.update(best_trial.params)
    
    trades, trade_logs = backtester.run_backtest(dynamic_params)
    print(f"총 거래 횟수: {len(trades)}")
    
    # 결과를 CSV 파일로 저장
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv("backtest_trades.csv", index=False)
        print("거래 내역이 backtest_trades.csv에 저장되었습니다.")
    else:
        print("거래 내역이 없습니다.")
    
    if trade_logs:
        logs_df = pd.DataFrame(trade_logs)
        logs_df.to_csv("trade_logs.csv", index=False)
        print("체결 기록이 trade_logs.csv에 저장되었습니다.")
    else:
        print("체결 기록이 없습니다.")
    
    # 성과 리포트 출력: trades 리스트를 DataFrame으로 변환 후 전달
    trades_df = pd.DataFrame(trades)
    print_performance_report(trades_df, initial_balance=10000)

if __name__ == "__main__":
    main()
