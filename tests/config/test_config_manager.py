# tests/config/test_config_manager.py

# ConfigManager 클래스를 임포트합니다.
from parameter_management.config_manager import ConfigManager

def test_get_defaults():
    """
    ConfigManager의 get_defaults 메서드가 기본 파라미터를 올바르게 반환하는지 테스트합니다.
    
    - 반환된 기본 파라미터(defaults) 내에 'sma_period', 'atr_period', 'risk_per_trade' 키가 포함되어 있는지 확인합니다.
    
    Parameters:
        없음
        
    Returns:
        없음 (assertion을 통해 테스트 결과를 검증)
    """
    cm = ConfigManager()
    defaults = cm.get_defaults()
    # 주요 파라미터 키들이 defaults에 포함되어 있는지 확인
    for key in ["sma_period", "atr_period", "risk_per_trade"]:
        assert key in defaults

def test_update_with_market_data():
    """
    ConfigManager의 update_with_market_data 메서드가 시장 데이터에 따라 파라미터를 업데이트하는지 테스트합니다.
    
    - 기본 파라미터와 비교하여, atr_multiplier와 risk_per_trade가 변경되었음을 검증합니다.
    
    Parameters:
        없음
        
    Returns:
        없음 (assertion을 통해 테스트 결과를 검증)
    """
    cm = ConfigManager()
    base_defaults = cm.get_defaults()
    # 시장 데이터를 모의한 딕셔너리 생성
    market_data = {
        "volatility": 0.08,
        "trend": "bullish",
        "trend_strength": 0.1,
        "volume": 800,
        "weekly_volatility": 0.09
    }
    updated = cm.update_with_market_data(market_data)
    # atr_multiplier와 risk_per_trade 값이 기본값과 다르게 업데이트 되었는지 확인
    assert updated["atr_multiplier"] != base_defaults["atr_multiplier"]
    assert updated["risk_per_trade"] != base_defaults["risk_per_trade"]

def test_merge_optimized():
    """
    ConfigManager의 merge_optimized 메서드가 최적화된 파라미터와 기본 파라미터를 올바르게 병합하는지 테스트합니다.
    
    - profit_ratio는 두 값의 평균으로 계산되며, 새로운 파라미터(new_param)는 최적화 값이 그대로 적용되어야 합니다.
    
    Parameters:
        없음
        
    Returns:
        없음 (assertion을 통해 테스트 결과를 검증)
    """
    cm = ConfigManager()
    defaults = cm.get_defaults()
    # 최적화 값으로 profit_ratio를 1.1배, 그리고 새로운 파라미터 추가
    optimized = {"profit_ratio": defaults["profit_ratio"] * 1.1, "new_param": 123}
    merged = cm.merge_optimized(optimized)
    # profit_ratio는 기본값과 최적화 값의 평균값이어야 함
    assert merged["profit_ratio"] == (defaults["profit_ratio"] + defaults["profit_ratio"] * 1.1) / 2
    # 새로운 파라미터는 최적화 값 그대로 반영되어야 함
    assert merged["new_param"] == 123
