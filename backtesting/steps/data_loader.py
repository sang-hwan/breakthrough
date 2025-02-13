# backtesting/steps/data_loader.py
from logs.logger_config import setup_logger
from data.db.db_manager import fetch_ohlcv_records
from data.ohlcv.ohlcv_aggregator import aggregate_to_weekly
from trading.indicators import compute_bollinger_bands

logger = setup_logger(__name__)

def load_data(backtester, short_table_format, long_table_format, short_tf, long_tf, start_date=None, end_date=None, extra_tf=None, use_weekly=False):
    try:
        symbol_for_table = backtester.symbol.replace('/', '').lower()
        short_table = short_table_format.format(symbol=symbol_for_table, timeframe=short_tf)
        long_table = long_table_format.format(symbol=symbol_for_table, timeframe=long_tf)
        backtester.df_short = fetch_ohlcv_records(short_table, start_date, end_date)
        backtester.df_long = fetch_ohlcv_records(long_table, start_date, end_date)
        if backtester.df_short.empty or backtester.df_long.empty:
            logger.error("데이터 로드 실패: short 또는 long 데이터가 비어있습니다.")
            raise ValueError("No data loaded")
        backtester.df_short.sort_index(inplace=True)
        backtester.df_long.sort_index(inplace=True)
        logger.debug(f"데이터 로드 완료: short 데이터 {len(backtester.df_short)}행, long 데이터 {len(backtester.df_long)}행")
    except Exception as e:
        logger.error(f"데이터 로드 중 에러 발생: {e}", exc_info=True)
        raise

    if extra_tf:
        try:
            extra_table = short_table_format.format(symbol=symbol_for_table, timeframe=extra_tf)
            backtester.df_extra = fetch_ohlcv_records(extra_table, start_date, end_date)
            if not backtester.df_extra.empty:
                backtester.df_extra.sort_index(inplace=True)
                backtester.df_extra = compute_bollinger_bands(backtester.df_extra, price_column='close', period=20, std_multiplier=2.0, fillna=True)
                logger.debug(f"Extra 데이터 로드 완료: {len(backtester.df_extra)}행")
        except Exception as e:
            logger.error(f"Extra 데이터 로드 에러: {e}", exc_info=True)
    if use_weekly:
        try:
            backtester.df_weekly = aggregate_to_weekly(backtester.df_short, compute_indicators=True)
            if backtester.df_weekly.empty:
                logger.warning("주간 데이터 집계 결과가 비어있습니다.")
            else:
                logger.debug(f"주간 데이터 집계 완료: {len(backtester.df_weekly)}행")
        except Exception as e:
            logger.error(f"주간 데이터 집계 에러: {e}", exc_info=True)
