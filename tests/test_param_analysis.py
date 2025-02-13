# tests/test_param_analysis.py
import pytest
import numpy as np
from strategies.param_analysis import run_sensitivity_analysis
from config.config_manager import ConfigManager

def test_run_sensitivity_analysis():
    param_settings = {
        "profit_ratio": np.linspace(0.05, 0.15, 3)
    }
    cm = ConfigManager()
    base_dynamic_params = cm.get_defaults()
    results = run_sensitivity_analysis(param_settings, assets=["BTC/USDT"], short_tf="4h", long_tf="1d",
                                       start_date="2023-01-01", end_date="2023-01-31",
                                       periods=[("2023-01-01", "2023-01-31")],
                                       base_dynamic_params=base_dynamic_params)
    # 결과가 딕셔너리 형태이며, profit_ratio 값에 대한 결과가 존재해야 함
    assert "profit_ratio" in results
    for val, metrics in results["profit_ratio"].items():
        # ROI 등의 지표가 포함되어 있으면 통과
        assert metrics is None or "roi" in metrics
