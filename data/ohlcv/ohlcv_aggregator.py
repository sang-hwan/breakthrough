# data/ohlcv/ohlcv_aggregator.py

import pandas as pd
from ta.trend import SMAIndicator  # 기술적 지표 중 단순 이동평균(SMA)을 계산하기 위한 라이브러리
from logs.logger_config import setup_logger  # 로깅 설정을 위한 모듈

# 전역 변수: 로깅 설정 객체
# 현재 파일의 이름을 기반으로 로깅을 설정하여 디버그 및 에러 로그 기록에 활용됩니다.
logger = setup_logger(__name__)

def aggregate_to_weekly(
    df: pd.DataFrame,
    compute_indicators: bool = True,
    resample_rule: str = "W-MON",
    label: str = "left",
    closed: str = "left",
    timezone: str = None,
    sma_window: int = 5
) -> pd.DataFrame:
    """
    OHLCV 데이터를 주간 단위로 집계하고, 선택적으로 기술적 지표를 계산합니다.
    인덱스의 타임존 조정도 가능하도록 합니다.
    
    Parameters:
        df (pd.DataFrame): datetime 인덱스를 가진 OHLCV 데이터. (open, high, low, close, volume 컬럼 포함)
        compute_indicators (bool): 추가 지표(주간 SMA, 모멘텀, 변동성)를 계산할지 여부.
        resample_rule (str): pandas의 재샘플링 규칙 (기본값 "W-MON": 월요일 기준 주간).
        label (str): 재샘플링 결과의 인덱스 라벨 위치 (기본 "left").
        closed (str): 구간의 닫힌 쪽 지정 (기본 "left").
        timezone (str): 인덱스의 타임존 변환 옵션 (예: "UTC", "Asia/Seoul").
        sma_window (int): 주간 단순 이동평균(SMA) 계산 시 윈도우 크기 (기본 5).
    
    Returns:
        pd.DataFrame: 주간 단위로 집계된 데이터프레임.
                      기본 컬럼: open, weekly_high, weekly_low, close, volume.
                      compute_indicators가 True이면 weekly_sma, weekly_momentum, weekly_volatility 컬럼 추가.
    """
    
    # 필수 컬럼들 (OHLCV 데이터) 존재 여부 확인
    required_columns = {"open", "high", "low", "close", "volume"}
    missing = required_columns - set(df.columns)
    if missing:
        # 필수 컬럼이 누락된 경우 에러 로그 기록 후 빈 DataFrame 반환
        logger.error(f"Input DataFrame is missing required columns: {missing}", exc_info=True)
        return pd.DataFrame()

    # 인덱스가 datetime 형식인지 확인 후 변환
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            logger.error(f"Failed to convert index to datetime: {e}", exc_info=True)
            return pd.DataFrame()

    if df.empty:
        logger.error("Input DataFrame for aggregation is empty.", exc_info=True)
        return df

    try:
        if timezone:
            # 타임존이 지정된 경우: 인덱스가 naive이면 UTC로 로컬라이즈 후 지정된 타임존으로 변환
            if df.index.tz is None:
                df = df.tz_localize('UTC')
            df = df.tz_convert(timezone)
    except Exception as e:
        logger.error(f"Timezone conversion error: {e}", exc_info=True)

    try:
        # pandas의 resample 메소드를 이용하여 주간 단위로 OHLCV 데이터를 집계
        weekly = df.resample(rule=resample_rule, label=label, closed=closed).agg({
            'open': 'first',    # 해당 주의 첫 번째 가격을 open으로 사용
            'high': 'max',      # 주간 최고가
            'low': 'min',       # 주간 최저가
            'close': 'last',    # 해당 주의 마지막 가격을 close로 사용
            'volume': 'sum'     # 거래량은 주간 합계
        })
    except Exception as e:
        logger.error(f"Resampling error: {e}", exc_info=True)
        return pd.DataFrame()

    if weekly.empty:
        logger.error("Aggregated weekly DataFrame is empty after resampling.", exc_info=True)
        return weekly

    # 컬럼 이름 변경: 전략 개발 시 명확하게 식별할 수 있도록 high, low 컬럼을 주간 데이터로 재명명
    weekly.rename(columns={'high': 'weekly_high', 'low': 'weekly_low'}, inplace=True)

    if compute_indicators:
        try:
            # 주간 SMA 계산: SMAIndicator를 사용하여 close 가격의 단순 이동평균을 계산합니다.
            sma_indicator = SMAIndicator(close=weekly['close'], window=sma_window, fillna=True)
            weekly['weekly_sma'] = sma_indicator.sma_indicator()
            
            # 주간 모멘텀 계산: 주간 close 가격의 전 주 대비 백분율 변화
            weekly['weekly_momentum'] = weekly['close'].pct_change() * 100
            
            # 주간 변동성 계산: (주간 최고가 - 주간 최저가) / 주간 최저가,
            # 만약 주간 최저가가 0이면 0으로 설정하여 division by zero 방지
            weekly['weekly_volatility'] = weekly.apply(
                lambda row: (row['weekly_high'] - row['weekly_low']) / row['weekly_low']
                if row['weekly_low'] != 0 else 0.0, axis=1)
        except Exception as e:
            logger.error(f"Error computing weekly indicators: {e}", exc_info=True)
            
    # 최종 집계된 주간 데이터프레임 반환
    return weekly
