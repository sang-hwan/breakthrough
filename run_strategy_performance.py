# run_strategy_performance.py
from logs.logger_config import setup_logger, initialize_root_logger, shutdown_logging
from logs.logging_util import LoggingUtil
from strategies.optimizer import DynamicParameterOptimizer
from backtesting.backtester import Backtester
from backtesting.performance import compute_performance
from logs.final_report import generate_final_report
from config.config_manager import ConfigManager

def run_strategy_performance():
    # 기존 로그 파일 삭제 및 루트 로거 초기화
    LoggingUtil.clear_log_files()
    initialize_root_logger()

    logger = setup_logger(__name__)
    logger.debug("Starting full project test.")

    # 파라미터 최적화 (Walk-Forward 방식)
    logger.debug("Starting Walk-Forward parameter optimization.")
    optimizer = DynamicParameterOptimizer(n_trials=10)
    best_trial = optimizer.optimize()

    # 기본 파라미터와 최적화된 파라미터 병합
    config_manager = ConfigManager()
    best_params = config_manager.merge_optimized(best_trial.params)
    logger.debug(f"Optimal parameters determined: {best_params}")

    # 각 심볼별 백테스트 실행
    start_date = "2018-06-01"
    end_date = "2025-02-01"
    timeframes = {"short_tf": "4h", "long_tf": "1d"}
    symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT"]

    for symbol in symbols:
        symbol_key = symbol.replace("/", "").lower()
        backtester = Backtester(symbol=symbol, account_size=10000)
        try:
            backtester.load_data(
                short_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                long_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                short_tf=timeframes["short_tf"],
                long_tf=timeframes["long_tf"],
                start_date=start_date,
                end_date=end_date,
                use_weekly=True
            )
        except Exception as e:
            logger.error(f"Data load failed for {symbol}: {e}")
            continue

        try:
            trades, _ = backtester.run_backtest(dynamic_params=best_params)
            logger.debug(f"Backtest complete for {symbol}: {len(trades)} trades executed.")
        except Exception as e:
            logger.error(f"Backtest error for {symbol}: {e}")
            continue

        if trades:
            performance_data = compute_performance(trades, weekly_data=backtester.df_weekly)
            generate_final_report(performance_data, symbol=symbol)
        else:
            logger.debug(f"No trades executed for {symbol}.")

    logger.debug("Project test complete.")
    shutdown_logging()

if __name__ == "__main__":
    run_strategy_performance()
