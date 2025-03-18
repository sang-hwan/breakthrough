# backtesting/steps/data_loader.py

# 모듈별 로깅 설정 및 데이터 관련 모듈 임포트
from logging.logger_config import setup_logger                   # 로깅 설정 함수
from data.db.db_manager import fetch_ohlcv_records              # OHLCV 데이터베이스 조회 함수
from data.ohlcv.ohlcv_aggregator import aggregate_to_weekly       # 주간 데이터 집계 함수
from signal_calculation.indicators import compute_bollinger_bands          # 볼린저 밴드 계산 함수
import threading                                                # 멀티스레드 동기화를 위한 모듈
import pandas as pd                                             # 데이터 프레임 처리를 위한 pandas
from logging.logging_util import LoggingUtil                       # 추가 로깅 유틸리티

# --- 전역 변수 및 객체 정의 ---
# 모듈 단위 로깅 인스턴스 설정: 로그를 남길 때 모듈명(__name__)을 이용
logger = setup_logger(__name__)
log_util = LoggingUtil(__name__)

# 멀티스레드 환경에서의 데이터 캐싱을 위한 전역 변수
_cache_lock = threading.Lock()    # 캐시 접근시 동기화를 위한 Lock 객체
_data_cache = {}                  # OHLCV 데이터 프레임을 메모리에 캐싱하기 위한 딕셔너리

# --- 내부 캐시 함수 정의 ---
def _get_cached_ohlcv(table_name, start_date, end_date):
    """
    캐시에서 특정 테이블과 날짜 범위에 해당하는 OHLCV 데이터를 조회합니다.
    
    Parameters:
        table_name (str): OHLCV 데이터가 저장된 테이블의 이름.
        start_date (str 또는 datetime): 데이터 조회의 시작 날짜.
        end_date (str 또는 datetime): 데이터 조회의 종료 날짜.
    
    Returns:
        pandas.DataFrame 또는 None: 캐시에 존재하는 경우 해당 DataFrame, 없으면 None.
    """
    # 캐시 키는 테이블 이름과 날짜 범위를 튜플로 결합하여 생성합니다.
    key = (table_name, start_date, end_date)
    # 멀티스레드 환경에서 동시 접근을 방지하기 위해 락을 사용합니다.
    with _cache_lock:
        return _data_cache.get(key)

def _set_cached_ohlcv(table_name, start_date, end_date, df):
    """
    주어진 OHLCV 데이터 프레임을 캐시에 저장합니다.
    
    Parameters:
        table_name (str): 데이터가 속한 테이블 이름.
        start_date (str 또는 datetime): 데이터 조회의 시작 날짜.
        end_date (str 또는 datetime): 데이터 조회의 종료 날짜.
        df (pandas.DataFrame): 저장할 OHLCV 데이터 프레임.
    
    Returns:
        None
    """
    # 캐시 키 생성
    key = (table_name, start_date, end_date)
    # 동기화를 위해 락 사용 후 캐시에 저장
    with _cache_lock:
        _data_cache[key] = df

def _validate_and_prepare_df(df, table_name):
    """
    불러온 OHLCV 데이터 프레임의 유효성을 검사하고, 필요한 전처리(시간 인덱스 변환, 정렬, 중복 제거 등)를 수행합니다.
    
    Parameters:
        df (pandas.DataFrame): 검증 및 전처리할 데이터 프레임.
        table_name (str): 데이터 프레임이 속한 테이블 이름(로그 메시지 용도).
    
    Returns:
        pandas.DataFrame: 전처리가 완료된 데이터 프레임.
    """
    # 데이터 프레임이 비어있는지 확인 후 에러 로그 출력
    if df.empty:
        logger.error(f"DataFrame for {table_name} is empty.", exc_info=True)
        return df

    # 인덱스가 datetime 형식인지 확인하고, 아니라면 변환 시도
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        try:
            df.index = pd.to_datetime(df.index, errors='coerce')
            # 변환 후 NaT (Not a Time)가 포함된 경우 해당 행 제거
            if df.index.isnull().any():
                logger.warning(f"Some index values in {table_name} could not be converted to datetime and will be dropped.")
                df = df[~df.index.isnull()]
        except Exception as e:
            logger.error(f"Error converting index to datetime for {table_name}: {e}", exc_info=True)
            raise

    # 인덱스를 오름차순으로 정렬 (시간 순 정렬)
    df.sort_index(inplace=True)

    # 중복된 인덱스가 존재하면 경고 로그 출력 후 중복 제거
    if df.index.duplicated().any():
        logger.warning(f"Duplicate datetime indices found in {table_name}; removing duplicates.")
        df = df[~df.index.duplicated(keep='first')]

    # 데이터 열이 존재할 경우, 고가(high), 저가(low), 종가(close)를 활용하여 평균 변동폭 계산
    if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
        avg_range = (df['high'] - df['low']).mean()
        avg_close = df['close'].mean()
        # 변동성이 비정상적으로 낮은 경우 경고 로그 출력 (정상 데이터 여부 점검)
        if avg_range < avg_close * 0.001:
            logger.warning(f"Data for {table_name} shows low volatility: avg_range={avg_range:.6f}, avg_close={avg_close:.6f}.")
    return df

def load_data(backtester, short_table_format, long_table_format, short_tf, long_tf, 
              start_date=None, end_date=None, extra_tf=None, use_weekly=False):
    """
    백테스터(backtester) 객체에 필요한 OHLCV 데이터와 인디케이터 데이터를 로드 및 전처리합니다.
    
    주요 기능:
      - 단기(short) 및 장기(long) 데이터 테이블 이름 생성 후 캐시에서 검색
      - 캐시에 없으면 데이터베이스에서 조회하고 캐시에 저장
      - 데이터 프레임의 유효성을 검사하고 전처리 수행
      - 추가 시간 프레임(extra_tf)이 제공되면 볼린저 밴드를 계산하여 추가 데이터 구성
      - use_weekly가 True면 단기 데이터를 주간 데이터로 집계하여 저장
    
    Parameters:
        backtester (object): 백테스트 실행 객체로, 로드된 데이터를 저장할 속성을 포함.
        short_table_format (str): 단기 데이터 테이블 이름 형식 (문자열 포맷).
        long_table_format (str): 장기 데이터 테이블 이름 형식 (문자열 포맷).
        short_tf (str): 단기 데이터의 시간 프레임 (예: '1m', '5m').
        long_tf (str): 장기 데이터의 시간 프레임 (예: '1h', '1d').
        start_date (str 또는 datetime, optional): 데이터 조회 시작 날짜.
        end_date (str 또는 datetime, optional): 데이터 조회 종료 날짜.
        extra_tf (str, optional): 추가 데이터 시간 프레임 (예: '15m'); 기본값은 None.
        use_weekly (bool, optional): 주간 데이터 집계를 사용할지 여부.
    
    Returns:
        None
    """
    try:
        # 심볼(symbol)을 포맷에 맞게 소문자 및 '/' 제거 처리
        symbol_for_table = backtester.symbol.replace('/', '').lower()
        # 단기 및 장기 테이블 이름 생성 (문자열 포맷 사용)
        short_table = short_table_format.format(symbol=symbol_for_table, timeframe=short_tf)
        long_table = long_table_format.format(symbol=symbol_for_table, timeframe=long_tf)
        
        # 단기 데이터 로드: 캐시에서 검색 후 없으면 DB에서 조회
        df_short = _get_cached_ohlcv(short_table, start_date, end_date)
        if df_short is None:
            df_short = fetch_ohlcv_records(short_table, start_date, end_date)
            _set_cached_ohlcv(short_table, start_date, end_date, df_short)
        # 데이터 프레임 유효성 검사 및 전처리 수행
        df_short = _validate_and_prepare_df(df_short, short_table)
        
        # 장기 데이터 로드: 캐시 검색 후 없으면 DB에서 조회
        df_long = _get_cached_ohlcv(long_table, start_date, end_date)
        if df_long is None:
            df_long = fetch_ohlcv_records(long_table, start_date, end_date)
            _set_cached_ohlcv(long_table, start_date, end_date, df_long)
        df_long = _validate_and_prepare_df(df_long, long_table)
        
        # 백테스터 객체에 로드된 데이터를 할당 (후속 전략 로직에서 사용)
        backtester.df_short = df_short
        backtester.df_long = df_long
        
        # 단기 또는 장기 데이터가 비어있으면 에러 로그 출력 후 예외 발생
        if backtester.df_short.empty or backtester.df_long.empty:
            logger.error("데이터 로드 실패: short 또는 long 데이터가 비어있습니다.", exc_info=True)
            raise ValueError("No data loaded")
        
        # 데이터 로드 성공 이벤트 로깅
        log_util.log_event("Data loaded successfully", state_key="data_load")
    except Exception as e:
        logger.error(f"데이터 로드 중 에러 발생: {e}", exc_info=True)
        raise

    # 추가 시간 프레임(extra_tf)이 지정된 경우 추가 데이터 로드 및 볼린저 밴드 계산 수행
    if extra_tf:
        try:
            extra_table = short_table_format.format(symbol=symbol_for_table, timeframe=extra_tf)
            df_extra = _get_cached_ohlcv(extra_table, start_date, end_date)
            if df_extra is None:
                df_extra = fetch_ohlcv_records(extra_table, start_date, end_date)
                _set_cached_ohlcv(extra_table, start_date, end_date, df_extra)
            df_extra = _validate_and_prepare_df(df_extra, extra_table)
            backtester.df_extra = df_extra
            if not backtester.df_extra.empty:
                # 볼린저 밴드를 계산하여 보조 지표 추가 (가격 열은 'close')
                backtester.df_extra = compute_bollinger_bands(
                    backtester.df_extra,
                    price_column='close',
                    period=20,
                    std_multiplier=2.0,
                    fillna=True
                )
                log_util.log_event("Extra data loaded", state_key="extra_load")
        except Exception as e:
            logger.error(f"Extra 데이터 로드 에러: {e}", exc_info=True)
    # 주간 데이터 집계 옵션이 True인 경우, 단기 데이터를 주간 단위로 집계하여 백테스터에 추가
    if use_weekly:
        try:
            backtester.df_weekly = aggregate_to_weekly(backtester.df_short, compute_indicators=True)
            if backtester.df_weekly.empty:
                logger.warning("주간 데이터 집계 결과가 비어있습니다.")
            else:
                log_util.log_event("Weekly data aggregated", state_key="weekly_load")
        except Exception as e:
            logger.error(f"주간 데이터 집계 에러: {e}", exc_info=True)
