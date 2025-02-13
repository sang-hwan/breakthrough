# tests/test_trade_executor.py
import pandas as pd
from trading.trade_executor import TradeExecutor

def test_compute_atr_via_trade_executor():
    # 간단한 데이터프레임
    df = pd.DataFrame({
        "high": [110, 115, 120],
        "low": [100, 105, 110],
        "close": [105, 110, 115]
    })
    df_atr = TradeExecutor.compute_atr(df, period=2)
    assert "atr" in df_atr.columns

def test_calculate_partial_exit_targets():
    targets = TradeExecutor.calculate_partial_exit_targets(100, partial_exit_ratio=0.5, partial_profit_ratio=0.03, final_profit_ratio=0.06)
    assert isinstance(targets, list)
    assert len(targets) == 2
