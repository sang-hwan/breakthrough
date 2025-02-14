# backtesting/steps/data_loader.py
from logs.logger_config import setup_logger
from data.db.db_manager import fetch_ohlcv_records
from data.ohlcv.ohlcv_aggregator import aggregate_to_weekly
from trading.indicators import compute_bollinger_bands
import threading

logger = setup_logger(__name__)

# 간단한 in-memory 캐시 (동일 테이블, 날짜 범위의 데이터 중복 호출 방지)
_cache_lock = threading.Lock()
_data_cache = {}

def _get_cached_ohlcv(table_name, start_date, end_date):
    key = (table_name, start_date, end_date)
    with _cache_lock:
        return _data_cache.get(key)

def _set_cached_ohlcv(table_name, start_date, end_date, df):
    key = (table_name, start_date, end_date)
    with _cache_lock:
        _data_cache[key] = df

def load_data(backtester, short_table_format, long_table_format, short_tf, long_tf, start_date=None, end_date=None, extra_tf=None, use_weekly=False):
    try:
        symbol_for_table = backtester.symbol.replace('/', '').lower()
        short_table = short_table_format.format(symbol=symbol_for_table, timeframe=short_tf)
        long_table = long_table_format.format(symbol=symbol_for_table, timeframe=long_tf)
        # 캐시에서 short 데이터 조회
        df_short = _get_cached_ohlcv(short_table, start_date, end_date)
        if df_short is None:
            df_short = fetch_ohlcv_records(short_table, start_date, end_date)
            _set_cached_ohlcv(short_table, start_date, end_date, df_short)
        # 캐시에서 long 데이터 조회
        df_long = _get_cached_ohlcv(long_table, start_date, end_date)
        if df_long is None:
            df_long = fetch_ohlcv_records(long_table, start_date, end_date)
            _set_cached_ohlcv(long_table, start_date, end_date, df_long)
        backtester.df_short = df_short
        backtester.df_long = df_long
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
            df_extra = _get_cached_ohlcv(extra_table, start_date, end_date)
            if df_extra is None:
                df_extra = fetch_ohlcv_records(extra_table, start_date, end_date)
                _set_cached_ohlcv(extra_table, start_date, end_date, df_extra)
            backtester.df_extra = df_extra
            if not backtester.df_extra.empty:
                backtester.df_extra.sort_index(inplace=True)
                backtester.df_extra = compute_bollinger_bands(
                    backtester.df_extra,
                    price_column='close',
                    period=20,
                    std_multiplier=2.0,
                    fillna=True
                )
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
