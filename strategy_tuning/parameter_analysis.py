# strategy_tuning/parameter_analysis.py
from logs.logger_config import setup_logger
from backtesting.backtester import Backtester
from backtesting.performance import compute_performance
from strategy_tuning.dynamic_param_manager import DynamicParamManager

logger = setup_logger(__name__)

def run_sensitivity_analysis(param_name, param_range, asset="BTC/USDT",
                             short_tf="4h", long_tf="1d", start_date="2018-06-01", end_date="2020-12-31",
                             base_dynamic_params=None):
    """
    특정 파라미터의 값 변화에 따른 백테스트 성과(ROI)를 평가합니다.

    Args:
        param_name (str): 분석할 파라미터 이름 (예: "profit_ratio")
        param_range (list or np.array): 파라미터 값의 리스트 또는 배열
        asset (str): 자산 심볼 (예: "BTC/USDT")
        short_tf (str): 단기 타임프레임 (예: "4h")
        long_tf (str): 장기 타임프레임 (예: "1d")
        start_date (str): 데이터 로드 시작일 (예: "2018-06-01")
        end_date (str): 데이터 로드 종료일 (예: "2020-12-31")
        base_dynamic_params (dict): 기본 동적 파라미터 사전. None인 경우, DynamicParamManager에서 기본값을 사용

    Returns:
        dict: 파라미터 값과 해당 ROI를 매핑한 결과 딕셔너리
    """
    results = {}
    if base_dynamic_params is None:
        dpm = DynamicParamManager()
        base_dynamic_params = dpm.get_default_params()

    logger.info(f"Starting sensitivity analysis for parameter: {param_name}")
    for val in param_range:
        dynamic_params = base_dynamic_params.copy()
        dynamic_params[param_name] = val

        logger.info(f"Testing {param_name} = {val}")
        try:
            bt = Backtester(symbol=asset, account_size=10000)
            symbol_key = asset.replace("/", "").lower()
            short_table_format = f"ohlcv_{symbol_key}_{{timeframe}}"
            long_table_format = f"ohlcv_{symbol_key}_{{timeframe}}"
            bt.load_data(short_table_format=short_table_format,
                         long_table_format=long_table_format,
                         short_tf=short_tf,
                         long_tf=long_tf,
                         start_date=start_date,
                         end_date=end_date)
            trades, _ = bt.run_backtest(dynamic_params=dynamic_params)
            perf = compute_performance(trades)
            roi = perf.get("roi", 0)
            logger.info(f"{param_name} = {val} => ROI: {roi:.2f}%")
            results[val] = roi
        except Exception as e:
            logger.error(f"Error during backtest for {param_name}={val}: {e}", exc_info=True)
            results[val] = None

    return results
