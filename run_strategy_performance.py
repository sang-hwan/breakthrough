# run_strategy_performance.py
"""
이 스크립트는 전체 거래 전략의 성능 평가를 위한 프로젝트 테스트를 수행합니다.
주요 작업:
  1. 워크포워드 방식의 파라미터 최적화 수행
  2. 최적의 파라미터를 적용하여 각 자산에 대해 백테스팅 실행
  3. 백테스트 결과를 기반으로 거래 성과 계산 및 최종 보고서 생성
"""

from logs.logger_config import setup_logger, initialize_root_logger, shutdown_logging  # 로깅 설정 함수들
from logs.logging_util import LoggingUtil  # 기존 로깅 파일 관리 함수
from strategies.optimizer import DynamicParameterOptimizer  # 동적 파라미터 최적화 클래스
from backtesting.backtester import Backtester  # 백테스팅 수행 클래스
from backtesting.performance import compute_performance  # 거래 성과 계산 함수
from logs.final_report import generate_final_report  # 최종 보고서 생성 함수
from config.config_manager import ConfigManager  # 설정 관리 클래스
from data.db.db_manager import get_unique_symbol_list, get_date_range  # DB 관련 함수
from data.db.db_config import DATABASE  # DB 접속 정보

def get_default_date_range(symbol: str, timeframe: str = "1d") -> tuple:
    """
    주어진 심볼과 시간 프레임에 대해 데이터베이스에서 날짜 범위를 조회합니다.
    
    Parameters:
        symbol (str): 거래 심볼 (예: "BTC/USDT")
        timeframe (str): 데이터의 시간 간격 (기본값: "1d")
    
    Returns:
        tuple: (시작 날짜, 종료 날짜) 문자열. 데이터가 없으면 기본 날짜 범위를 반환합니다.
    """
    symbol_key = symbol.replace("/", "").lower()  # 테이블명 생성을 위해 심볼 포맷 변환
    table_name = f"ohlcv_{symbol_key}_{timeframe}"
    start_date, end_date = get_date_range(table_name, DATABASE)
    if start_date is None or end_date is None:
        start_date, end_date = "2018-01-01 00:00:00", "2025-12-31 23:59:59"
    return start_date, end_date

def run_strategy_performance():
    """
    거래 전략의 성능을 평가하기 위한 전체 테스트를 수행합니다.
    
    주요 단계:
      1. 기존 로깅 파일 정리 및 로깅 시스템 초기화
      2. 워크포워드 파라미터 최적화를 통해 최적의 파라미터 탐색
      3. 각 자산별 백테스팅 실행 및 데이터 로드
      4. 백테스트 결과를 이용해 거래 성과 계산 후 최종 보고서 생성
      5. 로깅 종료
      
    Parameters:
        없음
    
    Returns:
        None
    """
    # 기존 로깅 파일 삭제 및 로깅 시스템 초기화
    LoggingUtil.clear_log_files()
    initialize_root_logger()

    logger = setup_logger(__name__)
    logger.info("Starting full project test.")

    # DB에서 분석 대상 자산 목록 조회, 없으면 기본 자산 리스트 사용
    assets = get_unique_symbol_list() or ["BTC/USDT", "ETH/USDT", "XRP/USDT"]
    logger.info(f"Assets for strategy performance: {assets}")

    logger.info("Starting Walk-Forward parameter optimization.")
    # 동적 파라미터 최적화를 위한 객체 생성 (최적화 반복 횟수: 10회)
    optimizer = DynamicParameterOptimizer(n_trials=10, assets=assets)
    best_trial = optimizer.optimize()  # 최적의 파라미터 탐색

    config_manager = ConfigManager()
    # 최적화된 파라미터와 기존 설정을 병합하여 최종 파라미터 결정
    best_params = config_manager.merge_optimized(best_trial.params)
    logger.info(f"Optimal parameters determined: {best_params}")

    # 대표 자산의 1일(1d) 데이터 테이블을 통해 기본 날짜 범위를 조회
    default_start, default_end = get_default_date_range(assets[0], "1d")
    logger.info(f"Using date range from DB: {default_start} to {default_end}")
    timeframes = {"short_tf": "4h", "long_tf": "1d"}

    # 각 자산에 대해 백테스팅 실행
    for symbol in assets:
        symbol_key = symbol.replace("/", "").lower()  # 테이블명 생성을 위해 심볼 형식 변환
        # 초기 계좌 크기를 10,000으로 설정하여 백테스터 인스턴스 생성
        backtester = Backtester(symbol=symbol, account_size=10000)
        try:
            # 단기 및 장기 데이터 테이블 형식 지정 후 데이터 로드
            backtester.load_data(
                short_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                long_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                short_tf=timeframes["short_tf"],
                long_tf=timeframes["long_tf"],
                start_date=default_start,
                end_date=default_end,
                use_weekly=True  # 주간 데이터도 함께 로드
            )
        except Exception as e:
            logger.error(f"Data load failed for {symbol}: {e}", exc_info=True)
            continue

        try:
            # 최적 파라미터를 적용하여 백테스트 실행; 실행 결과로 거래 내역 반환
            trades, _ = backtester.run_backtest(dynamic_params=best_params)
            logger.info(f"Backtest complete for {symbol}: {len(trades)} trades executed.")
        except Exception as e:
            logger.error(f"Backtest error for {symbol}: {e}", exc_info=True)
            continue

        if trades:
            # 거래 성과 계산 (예: 수익률, 승률 등) 후 최종 보고서 생성
            performance_data = compute_performance(trades, weekly_data=backtester.df_weekly)
            generate_final_report(performance_data, symbol=symbol)
        else:
            logger.info(f"No trades executed for {symbol}.")

    logger.info("Project test complete.")
    shutdown_logging()  # 로깅 시스템 종료

if __name__ == "__main__":
    run_strategy_performance()
