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
    
    # 최적 파라미터를 적용하여 백테스트 실행 (예: 2018-01-01 ~ 2025-01-01)
    backtester = Backtester(symbol="BNB/USDT", account_size=10000)
    backtester.load_data("ohlcv_{symbol}_{timeframe}", "ohlcv_{symbol}_{timeframe}", "4h", "1d", "2018-01-01", "2025-01-01")
    
    market_data = {"volatility": 0.07, "trend_strength": 0.5}
    dynamic_params = backtester.dynamic_param_manager.update_dynamic_params(market_data)
    dynamic_params.update(best_trial.params)
    
    trades, trade_logs = backtester.run_backtest(dynamic_params)
    print(f"총 거래 횟수: {len(trades)}")
    
    # 거래 결과가 있다면 DataFrame으로 변환 후 CSV 저장
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv("backtest_trades.csv", index=False)
        print("거래 내역이 backtest_trades.csv에 저장되었습니다.")
    if trade_logs:
        logs_df = pd.DataFrame(trade_logs)
        logs_df.to_csv("trade_logs.csv", index=False)
        print("체결 기록이 trade_logs.csv에 저장되었습니다.")
    
    # 성과 리포트를 출력할 때 DataFrame 형태로 전달
    if trades:
        trades_df = pd.DataFrame(trades)  # 또는 위에서 저장한 trades_df 사용
        print_performance_report(trades_df, initial_balance=10000)
    else:
        print("No trades to report.")

if __name__ == "__main__":
    main()
