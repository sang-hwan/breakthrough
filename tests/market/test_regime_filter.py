# tests/market/test_regime_filter.py
# 이 파일은 시장의 가격 데이터를 기반으로 시장 상황(레짐)을 판단하는 함수들
# determine_market_regime과 filter_regime의 올바른 동작을 검증하기 위한 테스트를 포함합니다.

from markets.regime_filter import determine_market_regime, filter_regime

def test_determine_market_regime_bullish():
    """
    상승장(bullish) 상황 판단 함수 테스트

    목적:
      - 현재 가격이 이전 가격보다 높을 경우 시장을 'bullish'로 판단하는지 확인.
    
    Parameters:
      없음

    Returns:
      없음 (assert 구문으로 결과 검증)
    """
    # 테스트 데이터: 현재 가격이 105, 이전 가격이 100
    price_data = {"current_price": 105, "previous_price": 100}
    regime = determine_market_regime(price_data)
    # 상승장이 예상되므로 'bullish'이어야 함
    assert regime == "bullish"

def test_determine_market_regime_bearish():
    """
    하락장(bearish) 상황 판단 함수 테스트

    목적:
      - 현재 가격이 이전 가격보다 낮을 경우 시장을 'bearish'로 판단하는지 확인.
    
    Parameters:
      없음

    Returns:
      없음 (assert 구문으로 결과 검증)
    """
    # 테스트 데이터: 현재 가격이 95, 이전 가격이 100
    price_data = {"current_price": 95, "previous_price": 100}
    regime = determine_market_regime(price_data)
    # 하락장이 예상되므로 'bearish'이어야 함
    assert regime == "bearish"

def test_filter_regime():
    """
    filter_regime 함수 테스트

    목적:
      - 주어진 가격 데이터를 대상으로 특정 시장 레짐(target_regime)과의 일치 여부를 판별.
    
    Parameters:
      없음

    Returns:
      없음 (assert 구문으로 결과 검증)
    """
    price_data = {"current_price": 105, "previous_price": 100}
    # bullish 조건에 대해 True 반환해야 함
    assert filter_regime(price_data, target_regime="bullish")
    # bearish 조건에 대해 False 반환해야 함
    assert not filter_regime(price_data, target_regime="bearish")
