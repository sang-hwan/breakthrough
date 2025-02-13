# strategies/param_analysis.py
import numpy as np
from itertools import product
from logs.logger_config import setup_logger
from backtesting.backtester import Backtester
from config.config_manager import ConfigManager  # 기존 DynamicParamManager → ConfigManager

logger = setup_logger(__name__)

def run_sensitivity_analysis(param_settings,
                             assets=None,
                             short_tf="4h", long_tf="1d",
                             start_date="2018-06-01", end_date="2020-12-31",
                             periods=None,
                             base_dynamic_params=None):
    """
    다중 파라미터 값 변화에 따른 백테스트 성과를 평가합니다.
    파라미터 간 상호작용을 고려하기 위해 모든 조합을 테스트하며,
    각 조합에 대해 각 성과 지표(ROI, Sharpe 등)의 평균, 표준편차, 최소, 최대를 산출합니다.
    """
    if assets is None:
        assets = ["BTC/USDT"]
    if periods is None:
        periods = [(start_date, end_date)]
    if base_dynamic_params is None:
        config_manager = ConfigManager()
        base_dynamic_params = config_manager.get_defaults()  # ConfigManager의 기본 설정 불러오기

    logger.debug(f"Starting multi-parameter sensitivity analysis over assets {assets} and periods {periods}")

    results = {}
    if not isinstance(param_settings, dict):
        raise ValueError("param_settings must be a dict of {parameter_name: [values]} for multi-parameter analysis.")

    param_names = list(param_settings.keys())
    # 모든 파라미터 조합 생성 (파라미터 간 상호작용 고려)
    combinations = list(product(*(param_settings[name] for name in param_names)))
    logger.debug(f"Total combinations to test: {len(combinations)}")

    for combo in combinations:
        # combo는 param_names 순서에 따른 값들의 튜플입니다.
        dynamic_params = base_dynamic_params.copy()
        # 키를 정렬된 튜플로 생성 (예: (("atr_multiplier", 2.0), ("profit_ratio", 0.08)))
        combo_key = tuple(sorted(zip(param_names, combo)))
        logger.debug(f"Testing combination: {combo_key}")
        run_metrics = []
        for asset in assets:
            for period in periods:
                s_date, e_date = period
                try:
                    bt = Backtester(symbol=asset, account_size=10000)
                    symbol_key = asset.replace("/", "").lower()
                    short_table_format = f"ohlcv_{symbol_key}_{{timeframe}}"
                    long_table_format = f"ohlcv_{symbol_key}_{{timeframe}}"
                    bt.load_data(short_table_format=short_table_format,
                                 long_table_format=long_table_format,
                                 short_tf=short_tf,
                                 long_tf=long_tf,
                                 start_date=s_date,
                                 end_date=e_date)
                    # 조합에 해당하는 파라미터 값 적용
                    for name, val in combo_key:
                        dynamic_params[name] = val
                    trades, _ = bt.run_backtest(dynamic_params=dynamic_params)
                    from backtesting.performance import compute_performance
                    perf = compute_performance(trades)
                    run_metrics.append(perf)
                    logger.debug(f"Combination {combo_key} | Asset: {asset} | Period: {s_date} ~ {e_date} => ROI: {perf.get('roi', 0):.2f}%, Sharpe: {perf.get('sharpe_ratio', 0):.2f}, Max Drawdown: {perf.get('max_drawdown', 0):.2f}, Trade Count: {perf.get('trade_count', 0)}")
                except Exception as e:
                    logger.error(f"Error during backtest for combination {combo_key}, Asset: {asset}, Period: {s_date} ~ {e_date}: {e}", exc_info=True)
        if run_metrics:
            aggregated = {}
            metric_keys = ["roi", "sharpe_ratio", "max_drawdown", "trade_count", "cumulative_return", "total_pnl"]
            for key in metric_keys:
                values = [run.get(key, 0) for run in run_metrics]
                aggregated[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
            results[combo_key] = aggregated
            logger.debug(f"Aggregated result for combination {combo_key}: {aggregated}")
        else:
            results[combo_key] = None
            logger.warning(f"No successful runs for combination {combo_key}")
    return results
