# backtesting/steps/indicator_applier.py
from logs.logger_config import setup_logger
from trading.indicators import compute_sma, compute_rsi, compute_macd

logger = setup_logger(__name__)

def apply_indicators(backtester):
    try:
        backtester.df_long = compute_sma(backtester.df_long, price_column='close', period=200, fillna=True, output_col='sma')
        backtester.df_long = compute_rsi(backtester.df_long, price_column='close', period=14, fillna=True, output_col='rsi')
        backtester.df_long = compute_macd(backtester.df_long, price_column='close', slow_period=26, fast_period=12, signal_period=9, fillna=True, prefix='macd_')
        logger.debug("인디케이터 적용 완료 (SMA, RSI, MACD)")
    except Exception as e:
        logger.error(f"인디케이터 적용 중 에러 발생: {e}", exc_info=True)
        raise
