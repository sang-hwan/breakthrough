# strategy_tuning/parameter_analysis.py
import numpy as np
from logs.logger_config import setup_logger
from backtesting.backtester import Backtester
from backtesting.performance import compute_performance
from strategy_tuning.dynamic_param_manager import DynamicParamManager

logger = setup_logger(__name__)

def run_sensitivity_analysis(param_settings,
                             assets=None,
                             short_tf="4h", long_tf="1d",
                             start_date="2018-06-01", end_date="2020-12-31",
                             periods=None,
                             base_dynamic_params=None):
    """
    다중 파라미터 값 변화에 따른 백테스트 성과(ROI, Sharpe, 최대 낙폭, 거래 건수, 누적 수익률, 총 PnL 등)를
    평가합니다. 여러 자산 및 기간에 대해 반복 분석할 수 있도록 확장하였습니다.
    
    Args:
        param_settings (dict): 분석할 파라미터에 대한 설정 정보.
            예) {
                   "profit_ratio": [0.05, 0.06, 0.07, ...],
                   "atr_multiplier": [1.8, 1.9, 2.0, ...],
                   ...
                 }
        assets (list of str): 분석에 사용할 자산 심볼 리스트 (기본: ["BTC/USDT"])
        short_tf (str): 단기 타임프레임 (예: "4h")
        long_tf (str): 장기 타임프레임 (예: "1d")
        start_date (str): 기본 데이터 로드 시작일 (periods가 제공되지 않을 경우 사용)
        end_date (str): 기본 데이터 로드 종료일 (periods가 제공되지 않을 경우 사용)
        periods (list of tuples): 각 튜플이 (start_date, end_date)를 나타내며, 여러 기간에 대해 분석할 수 있음.
                                  기본값이 None인 경우 [(start_date, end_date)]로 설정됨.
        base_dynamic_params (dict): 기본 동적 파라미터 사전. None인 경우 DynamicParamManager에서 기본값 사용.
        
    Returns:
        dict: { param_name: {value: {metric_key: value, ...}, ...}, ... }
    """
    # 기본 값 설정
    if assets is None:
        assets = ["BTC/USDT"]
    if periods is None:
        periods = [(start_date, end_date)]
    if base_dynamic_params is None:
        dpm = DynamicParamManager()
        base_dynamic_params = dpm.get_default_params()

    logger.info(f"Starting sensitivity analysis over assets {assets} and periods {periods}")
    
    results = {}
    # param_settings는 반드시 dict 형식이어야 합니다.
    if not isinstance(param_settings, dict):
        raise ValueError("param_settings must be a dict of {parameter_name: [values]} for multi-parameter analysis.")

    for param_name, param_range in param_settings.items():
        logger.info(f"Analyzing parameter: {param_name}")
        results[param_name] = {}
        for val in param_range:
            dynamic_params = base_dynamic_params.copy()
            dynamic_params[param_name] = val
            run_metrics = []  # 각 run의 성과 지표를 저장할 리스트

            # 여러 자산, 여러 기간에 대해 반복
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
                        trades, _ = bt.run_backtest(dynamic_params=dynamic_params)
                        perf = compute_performance(trades)
                        run_metrics.append(perf)
                        logger.info(f"{param_name} = {val} | Asset: {asset} | Period: {s_date} ~ {e_date} => "
                                    f"ROI: {perf.get('roi', 0):.2f}%, Sharpe: {perf.get('sharpe_ratio', 0):.2f}, "
                                    f"Max Drawdown: {perf.get('max_drawdown', 0):.2f}, Trade Count: {perf.get('trade_count', 0)}")
                    except Exception as e:
                        logger.error(f"Error during backtest for {param_name}={val}, Asset: {asset}, Period: {s_date} ~ {e_date}: {e}", exc_info=True)
                        # 실패한 run은 건너뛰고 다음 run 진행
            if run_metrics:
                aggregated = {}
                metric_keys = ["roi", "sharpe_ratio", "max_drawdown", "trade_count", "cumulative_return", "total_pnl"]
                for key in metric_keys:
                    aggregated[key] = np.mean([run.get(key, 0) for run in run_metrics])
                results[param_name][val] = aggregated
                logger.info(f"Aggregated result for {param_name} = {val}: {aggregated}")
            else:
                results[param_name][val] = None
                logger.warning(f"No successful runs for {param_name} = {val}")
    return results
