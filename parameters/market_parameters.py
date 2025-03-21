# parameters/market_parameters.py
import numpy as np
from itertools import product
import random
from logs.log_config import setup_logger
from backtesting.backtester import Backtester
# ConfigManager를 trading_parameters.py에서 가져옴
from parameters.trading_parameters import ConfigManager

logger = setup_logger(__name__)

def run_sensitivity_analysis(param_settings: dict,
                             assets: list,
                             short_tf: str = "4h", long_tf: str = "1d",
                             start_date: str = "2018-06-01", end_date: str = "2020-12-31",
                             periods: list = None,
                             base_dynamic_params: dict = None,
                             max_combinations: int = 20) -> dict:
    """
    다양한 파라미터 조합에 대해 민감도 분석을 수행하고, 전략 성능의 변화를 평가합니다.
    
    Parameters:
        param_settings (dict): 각 파라미터 이름과 해당 파라미터가 가질 수 있는 값들의 리스트.
        assets (list): 분석에 사용할 자산 목록.
        short_tf (str): 단기 시간 프레임 (예: "4h").
        long_tf (str): 장기 시간 프레임 (예: "1d").
        start_date (str): 분석 시작 날짜 (YYYY-MM-DD).
        end_date (str): 분석 종료 날짜 (YYYY-MM-DD).
        periods (list, optional): (시작 날짜, 종료 날짜) 쌍의 리스트. 지정하지 않으면 전체 기간 사용.
        base_dynamic_params (dict, optional): 기본 동적 파라미터 값. 지정하지 않으면 ConfigManager에서 불러옴.
        max_combinations (int): 시도할 최대 파라미터 조합 수.
    
    Returns:
        dict: 각 파라미터 조합에 대해 집계된 성능 지표(ROI, Sharpe, 최대 낙폭 등)를 담은 결과.
    """
    if periods is None:
        periods = [(start_date, end_date)]
    if base_dynamic_params is None:
        base_dynamic_params = ConfigManager().get_defaults()

    logger.debug(f"Starting sensitivity analysis for assets: {assets}")
    results = {}
    param_names = list(param_settings.keys())
    combinations = list(product(*(param_settings[name] for name in param_names)))
    if len(combinations) > max_combinations:
        combinations = random.sample(combinations, max_combinations)

    for combo in combinations:
        dynamic_params = base_dynamic_params.copy()
        combo_key = tuple(sorted(zip(param_names, combo)))
        run_metrics = []
        for asset in assets:
            symbol_key = asset.replace("/", "").lower()
            for s_date, e_date in periods:
                try:
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
                    trades, _ = bt.run_backtest_pipeline(dynamic_params=dynamic_params)
                    logger.info(f"{asset} {s_date}~{e_date}: {len(trades)} trades executed.")
                    if trades:
                        roi = sum(trade.get("pnl", 0) for trade in trades) / 10000 * 100
                        logger.info(f"{asset} {s_date}~{e_date}: ROI={roi:.2f}%")
                        from backtesting.performance import compute_performance
                        perf = compute_performance(trades)
                        run_metrics.append(perf)
                    else:
                        logger.warning(f"{asset} {s_date}~{e_date}: No trades executed.")
                except Exception as e:
                    logger.error(f"Error during sensitivity analysis for {asset} with combination {combo_key}: {e}", exc_info=True)
                    continue
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
