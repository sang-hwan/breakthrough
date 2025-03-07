# tests/config/test_config_manager.py
from config.config_manager import ConfigManager

def test_get_defaults():
    cm = ConfigManager()
    defaults = cm.get_defaults()
    # 기본 파라미터에 몇 가지 주요 키가 있는지 확인
    for key in ["sma_period", "atr_period", "risk_per_trade"]:
        assert key in defaults

def test_update_with_market_data():
    cm = ConfigManager()
    base_defaults = cm.get_defaults()
    market_data = {
        "volatility": 0.08,
        "trend": "bullish",
        "trend_strength": 0.1,
        "volume": 800,
        "weekly_volatility": 0.09
    }
    updated = cm.update_with_market_data(market_data)
    # atr_multiplier와 risk_per_trade가 변경되었는지 확인
    assert updated["atr_multiplier"] != base_defaults["atr_multiplier"]
    assert updated["risk_per_trade"] != base_defaults["risk_per_trade"]

def test_merge_optimized():
    cm = ConfigManager()
    defaults = cm.get_defaults()
    optimized = {"profit_ratio": defaults["profit_ratio"] * 1.1, "new_param": 123}
    merged = cm.merge_optimized(optimized)
    # 민감도 대상은 평균값, 그 외는 최적화 값 적용됨
    assert merged["profit_ratio"] == (defaults["profit_ratio"] + defaults["profit_ratio"] * 1.1) / 2
    assert merged["new_param"] == 123
