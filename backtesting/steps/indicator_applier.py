# backtesting/steps/indicator_applier.py
from logs.logger_config import setup_logger
from trading.indicators import compute_sma, compute_rsi, compute_macd

logger = setup_logger(__name__)

def apply_indicators(backtester):
    backtester.df_long = compute_sma(backtester.df_long, price_column='close', period=200, fillna=True, output_col='sma')
    backtester.df_long = compute_rsi(backtester.df_long, price_column='close', period=14, fillna=True, output_col='rsi')
    backtester.df_long = compute_macd(backtester.df_long, price_column='close', slow_period=26, fast_period=12, signal_period=9, fillna=True, prefix='macd_')
    
    sma_min = backtester.df_long['sma'].min()
    sma_max = backtester.df_long['sma'].max()
    rsi_min = backtester.df_long['rsi'].min()
    rsi_max = backtester.df_long['rsi'].max()
    macd_diff_min = backtester.df_long['macd_diff'].min()
    macd_diff_max = backtester.df_long['macd_diff'].max()
    
    # 핵심 인디케이터 요약을 info 레벨로 출력
    logger.debug(
        f"인디케이터 적용 완료: SMA 범위=({sma_min:.2f}, {sma_max:.2f}), "
        f"RSI 범위=({rsi_min:.2f}, {rsi_max:.2f}), MACD diff 범위=({macd_diff_min:.2f}, {macd_diff_max:.2f})"
    )
