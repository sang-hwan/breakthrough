# strategies/param_analysis.py
import numpy as np
from itertools import product  # 여러 파라미터 값들의 모든 조합 생성을 위해 사용
import random
from logging.logger_config import setup_logger  # 로깅 설정 함수
from backtesting.backtester import Backtester  # 백테스트 수행 클래스
from parameter_management.config_manager import ConfigManager  # 기본 파라미터 관리 클래스

# 전역 logger 객체: 모듈 내 로그 기록에 사용
logger = setup_logger(__name__)

def run_sensitivity_analysis(param_settings,
                             assets,
                             short_tf="4h", long_tf="1d",
                             start_date="2018-06-01", end_date="2020-12-31",
                             periods=None,
                             base_dynamic_params=None,
                             max_combinations=20):
    """
    다양한 파라미터 조합에 대해 민감도 분석을 수행하고, 전략 성능의 변화를 평가합니다.
    
    Parameters:
        param_settings (dict): 각 파라미터 이름과 해당 파라미터가 가질 수 있는 값들의 리스트.
        assets (list): 분석에 사용할 자산 목록.
        short_tf (str): 단기 시간 프레임 (예: "4h").
        long_tf (str): 장기 시간 프레임 (예: "1d").
        start_date (str): 분석 시작 날짜 (YYYY-MM-DD).
        end_date (str): 분석 종료 날짜 (YYYY-MM-DD).
        periods (list of tuples, optional): (시작 날짜, 종료 날짜) 쌍의 리스트.
                                             지정하지 않으면 start_date ~ end_date 전체 기간 사용.
        base_dynamic_params (dict, optional): 기본 동적 파라미터 값.
                                              지정하지 않으면 ConfigManager에서 기본값을 불러옴.
        max_combinations (int): 시도할 최대 파라미터 조합 수.
                                모든 조합이 max_combinations보다 많으면 무작위로 샘플링.
    
    Returns:
        dict: 각 파라미터 조합에 대해 집계된 성능 지표 (ROI, Sharpe ratio, 최대 낙폭 등)를 담은 결과.
    """
    # periods가 지정되지 않으면 전체 기간을 하나의 기간으로 사용
    if periods is None:
        periods = [(start_date, end_date)]
    # base_dynamic_params가 없으면 기본값을 불러옴
    if base_dynamic_params is None:
        base_dynamic_params = ConfigManager().get_defaults()

    logger.debug(f"Starting sensitivity analysis over assets: {assets}")
    results = {}
    # 분석할 파라미터 이름 추출
    param_names = list(param_settings.keys())
    # 각 파라미터의 가능한 값들로 모든 조합 생성
    combinations = list(product(*(param_settings[name] for name in param_names)))
    # 조합 수가 max_combinations보다 많으면 무작위 샘플링하여 제한
    if len(combinations) > max_combinations:
        combinations = random.sample(combinations, max_combinations)

    # 각 파라미터 조합에 대해 백테스트를 수행하고 결과를 집계
    for combo in combinations:
        dynamic_params = base_dynamic_params.copy()
        # 파라미터 이름과 값 쌍을 튜플 형태로 정렬하여 결과 딕셔너리의 키로 사용
        combo_key = tuple(sorted(zip(param_names, combo)))
        run_metrics = []  # 해당 조합의 백테스트 성능 지표를 저장할 리스트
        for asset in assets:
            # 테이블 이름 구성에 사용할 심볼 키 생성 (예: "BTC/USDT" -> "btcusdt")
            symbol_key = asset.replace("/", "").lower()
            for s_date, e_date in periods:
                try:
                    # 백테스터 인스턴스 생성 (초기 계좌 잔액: 10000)
                    bt = Backtester(symbol=asset, account_size=10000)
                    bt.load_data(
                        short_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                        long_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                        short_tf=short_tf, long_tf=long_tf,
                        start_date=s_date, end_date=e_date
                    )
                    # 현재 조합의 파라미터로 덮어쓰기
                    for name, val in combo_key:
                        dynamic_params[name] = val
                    # 백테스트 파이프라인 실행 (전략 적용 및 거래 결과 산출)
                    trades, _ = bt.run_backtest_pipeline(dynamic_params=dynamic_params)
                    logger.info(f"{asset} {s_date}~{e_date}: {len(trades)} trades executed.")
                    if trades:
                        # ROI(수익률) 계산: 총 pnl을 초기 계좌 잔액으로 나누고 백분율로 환산
                        roi = sum(trade.get("pnl", 0) for trade in trades) / 10000 * 100
                        logger.info(f"{asset} {s_date}~{e_date}: ROI={roi:.2f}%")
                        from backtesting.performance import compute_performance
                        # 성능 지표(Sharpe, 최대 낙폭 등) 계산
                        perf = compute_performance(trades)
                        run_metrics.append(perf)
                    else:
                        logger.warning(f"{asset} {s_date}~{e_date}: No trades executed.")
                except Exception as e:
                    logger.error(f"Error during sensitivity analysis for {asset} with combination {combo_key}: {e}", exc_info=True)
                    continue
        # 각 조합별 성능 지표 집계: 평균, 표준편차, 최소, 최대 계산
        if run_metrics:
            aggregated = {}
            for key in ["roi", "sharpe_ratio", "max_drawdown", "trade_count", "cumulative_return", "total_pnl"]:
                values = [run.get(key, 0) for run in run_metrics]
                aggregated[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
            results[combo_key] = aggregated
        else:
            results[combo_key] = None
    return results
