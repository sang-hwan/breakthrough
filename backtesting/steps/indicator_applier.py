# backtesting/steps/indicator_applier.py

from logs.logger_config import setup_logger
from trading.indicators import compute_sma, compute_rsi, compute_macd

# 모듈 로깅 인스턴스 설정
logger = setup_logger(__name__)

def apply_indicators(backtester):
    """
    백테스터 객체의 장기 데이터(df_long)에 SMA, RSI, MACD 등 다양한 트레이딩 인디케이터를 적용합니다.
    
    주요 동작:
      - 단순 이동평균(SMA) 계산 후 'sma' 열에 저장
      - 상대 강도 지수(RSI) 계산 후 'rsi' 열에 저장
      - MACD 및 시그널, 차이값 계산 후 'macd_' 접두사로 열 추가
      - 적용된 인디케이터 값들의 최소/최대 범위를 로그에 출력
    
    Parameters:
        backtester (object): 인디케이터를 적용할 데이터 프레임(df_long)을 포함하는 백테스터 객체.
    
    Returns:
        None
    """
    # SMA 계산: 종가('close') 기준, 200 기간, 결측값 채움 옵션 활성화, 결과는 'sma' 열에 저장
    backtester.df_long = compute_sma(backtester.df_long, price_column='close', period=200, fillna=True, output_col='sma')
    # RSI 계산: 종가('close') 기준, 14 기간, 결측값 채움, 결과는 'rsi' 열에 저장
    backtester.df_long = compute_rsi(backtester.df_long, price_column='close', period=14, fillna=True, output_col='rsi')
    # MACD 계산: 종가('close') 기준, 느린 기간=26, 빠른 기간=12, 시그널 기간=9, 결측값 채움, 결과 열은 'macd_' 접두사를 사용
    backtester.df_long = compute_macd(backtester.df_long, price_column='close', slow_period=26, fast_period=12, signal_period=9, fillna=True, prefix='macd_')
    
    # 인디케이터가 적용된 데이터의 값 범위를 계산하여 로그에 출력 (모든 값의 최소 및 최대값)
    sma_min = backtester.df_long['sma'].min()
    sma_max = backtester.df_long['sma'].max()
    rsi_min = backtester.df_long['rsi'].min()
    rsi_max = backtester.df_long['rsi'].max()
    macd_diff_min = backtester.df_long['macd_diff'].min()
    macd_diff_max = backtester.df_long['macd_diff'].max()
    
    logger.debug(
        f"인디케이터 적용 완료: SMA 범위=({sma_min:.2f}, {sma_max:.2f}), "
        f"RSI 범위=({rsi_min:.2f}, {rsi_max:.2f}), MACD diff 범위=({macd_diff_min:.2f}, {macd_diff_max:.2f})"
    )
