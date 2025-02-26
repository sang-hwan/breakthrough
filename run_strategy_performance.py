# run_strategy_performance.py
from logs.logger_config import setup_logger, initialize_root_logger, shutdown_logging
from logs.logging_util import LoggingUtil
from strategies.optimizer import DynamicParameterOptimizer
from backtesting.backtester import Backtester
from backtesting.performance import compute_performance
from logs.final_report import generate_final_report
from config.config_manager import ConfigManager
from data.db.db_manager import get_unique_symbol_list, get_date_range
from data.db.db_config import DATABASE

def get_default_date_range(symbol: str, timeframe: str = "1d") -> tuple:
    symbol_key = symbol.replace("/", "").lower()
    table_name = f"ohlcv_{symbol_key}_{timeframe}"
    start_date, end_date = get_date_range(table_name, DATABASE)
    if start_date is None or end_date is None:
        start_date, end_date = "2018-01-01 00:00:00", "2025-12-31 23:59:59"
    return start_date, end_date

def run_strategy_performance():
    LoggingUtil.clear_log_files()
    initialize_root_logger()

    logger = setup_logger(__name__)
    logger.info("Starting full project test.")

    assets = get_unique_symbol_list() or ["BTC/USDT", "ETH/USDT", "XRP/USDT"]
    logger.info(f"Assets for strategy performance: {assets}")

    logger.info("Starting Walk-Forward parameter optimization.")
    optimizer = DynamicParameterOptimizer(n_trials=10, assets=assets)
    best_trial = optimizer.optimize()

    config_manager = ConfigManager()
    best_params = config_manager.merge_optimized(best_trial.params)
    logger.info(f"Optimal parameters determined: {best_params}")

    # DB에서 대표 심볼의 1d 테이블을 통해 날짜 범위를 조회
    default_start, default_end = get_default_date_range(assets[0], "1d")
    logger.info(f"Using date range from DB: {default_start} to {default_end}")
    timeframes = {"short_tf": "4h", "long_tf": "1d"}

    for symbol in assets:
        symbol_key = symbol.replace("/", "").lower()
        backtester = Backtester(symbol=symbol, account_size=10000)
        try:
            backtester.load_data(
                short_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                long_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                short_tf=timeframes["short_tf"],
                long_tf=timeframes["long_tf"],
                start_date=default_start,
                end_date=default_end,
                use_weekly=True
            )
        except Exception as e:
            logger.error(f"Data load failed for {symbol}: {e}", exc_info=True)
            continue

        try:
            trades, _ = backtester.run_backtest(dynamic_params=best_params)
            logger.info(f"Backtest complete for {symbol}: {len(trades)} trades executed.")
        except Exception as e:
            logger.error(f"Backtest error for {symbol}: {e}", exc_info=True)
            continue

        if trades:
            performance_data = compute_performance(trades, weekly_data=backtester.df_weekly)
            generate_final_report(performance_data, symbol=symbol)
        else:
            logger.info(f"No trades executed for {symbol}.")

    logger.info("Project test complete.")
    shutdown_logging()

if __name__ == "__main__":
    run_strategy_performance()
