# tests/optimizer/test_param_analysis.py
import numpy as np
from strategies.param_analysis import run_sensitivity_analysis

def test_sensitivity_analysis():
    param_settings = {
        "profit_ratio": np.linspace(0.07, 0.09, 3),
        "atr_multiplier": np.linspace(2.0, 2.2, 3)
    }
    results = run_sensitivity_analysis(
        param_settings,
        assets=["BTC/USDT"],
        short_tf="4h",
        long_tf="1d",
        start_date="2023-01-01",
        end_date="2023-01-10",
        periods=[("2023-01-01", "2023-01-10")]
    )
    assert isinstance(results, dict)
