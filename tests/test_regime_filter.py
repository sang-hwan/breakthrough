# tests/test_regime_filter.py
from markets.regime_filter import determine_market_regime, filter_regime

def test_determine_market_regime_bullish():
    price_data = {"current_price": 105, "previous_price": 100}
    regime = determine_market_regime(price_data)
    assert regime == "bullish"

def test_determine_market_regime_bearish():
    price_data = {"current_price": 95, "previous_price": 100}
    regime = determine_market_regime(price_data)
    assert regime == "bearish"

def test_filter_regime():
    price_data = {"current_price": 105, "previous_price": 100}
    assert filter_regime(price_data, target_regime="bullish")
    assert not filter_regime(price_data, target_regime="bearish")
