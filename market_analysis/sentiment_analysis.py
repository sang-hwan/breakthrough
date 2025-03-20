# market_analysis/sentiment_analysis.py
from logs.log_config import setup_logger
import re

logger = setup_logger(__name__)


def clean_text(text: str) -> str:
    """
    감성 분석을 위한 입력 텍스트를 정제합니다.
    
    Parameters:
        text (str): 원본 텍스트.
        
    Returns:
        str: 정제된 텍스트.
    """
    try:
        cleaned = re.sub(r'\s+', ' ', text)
        logger.debug("Text cleaned for sentiment analysis.")
        return cleaned.strip()
    except Exception as e:
        logger.error(f"Error cleaning text: {e}", exc_info=True)
        raise


def simple_sentiment_score(text: str) -> float:
    """
    키워드 매칭 방식을 사용하여 간단한 감성 점수를 계산합니다.
    
    Parameters:
        text (str): 입력 텍스트.
        
    Returns:
        float: -1 (매우 부정)에서 1 (매우 긍정) 사이의 감성 점수.
    """
    try:
        text = clean_text(text).lower()
        positive_words = ['good', 'bullish', 'up', 'positive', 'gain']
        negative_words = ['bad', 'bearish', 'down', 'negative', 'loss']
        
        score = 0
        for word in positive_words:
            score += text.count(word)
        for word in negative_words:
            score -= text.count(word)
        
        # 간단한 정규화 (기본 구현)
        normalized_score = max(min(score / 5.0, 1), -1)
        logger.debug(f"Computed sentiment score: {normalized_score} for text: {text}")
        return normalized_score
    except Exception as e:
        logger.error(f"Error computing sentiment score: {e}", exc_info=True)
        raise

def aggregate_sentiment(texts: list) -> float:
    """
    여러 텍스트의 감성 점수를 집계하여 평균 감성 점수를 계산합니다.
    
    Parameters:
        texts (list): 텍스트 문자열 리스트.
        
    Returns:
        float: 평균 감성 점수.
    """
    try:
        if not texts:
            logger.warning("No texts provided for sentiment aggregation.")
            return 0.0
        scores = [simple_sentiment_score(text) for text in texts]
        avg_score = sum(scores) / len(scores)
        logger.debug(f"Aggregated sentiment score: {avg_score}")
        return avg_score
    except Exception as e:
        logger.error(f"Error aggregating sentiment: {e}", exc_info=True)
        raise
