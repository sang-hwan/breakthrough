# tests/unit_tests/market_analysis/test_analyze_market.py
import pytest
from market_analysis.sentiment_analysis import clean_text, simple_sentiment_score, aggregate_sentiment

def test_clean_text():
    text = "This   is   a    test."
    cleaned = clean_text(text)
    # 연속 공백이 제거되었는지 확인
    assert "  " not in cleaned
    assert cleaned == "This is a test."

def test_simple_sentiment_score():
    positive_text = "The market is bullish and good."
    score = simple_sentiment_score(positive_text)
    assert score > 0

    negative_text = "The market is bearish and bad."
    score_neg = simple_sentiment_score(negative_text)
    assert score_neg < 0

def test_aggregate_sentiment():
    texts = ["The market is bullish.", "Investors are positive."]
    avg_score = aggregate_sentiment(texts)
    assert avg_score > 0
