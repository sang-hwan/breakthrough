# markets_analysis/regime_filter.py
from logs.logger_config import setup_logger

logger = setup_logger(__name__)

def determine_market_regime(price_data):
    """
    주어진 가격 데이터를 바탕으로 시장 레짐을 결정합니다.
    
    파라미터:
      - price_data (dict): 'current_price'와 'previous_price' 키를 포함하는 가격 데이터 딕셔너리.
    
    반환값:
      - str: 'bullish', 'bearish', 'sideways' 또는 에러 발생 시 'unknown'
    """
    try:
        current_price = price_data.get("current_price")
        previous_price = price_data.get("previous_price")
        if current_price is None or previous_price is None:
            logger.error("필수 가격 데이터 누락: 'current_price' 또는 'previous_price'가 제공되지 않음.")
            return "unknown"
        
        change_percent = (current_price - previous_price) / previous_price
        
        # 단순 기준: 2% 이상 상승이면 bullish, 2% 이상 하락이면 bearish, 그 외에는 sideways
        if change_percent > 0.02:
            regime = "bullish"
        elif change_percent < -0.02:
            regime = "bearish"
        else:
            regime = "sideways"
        
        logger.info(f"시장 레짐 결정: {regime} (변화율: {change_percent:.2%})")
        return regime
    except Exception as e:
        logger.error(f"시장 레짐 결정 중 에러 발생: {e}", exc_info=True)
        return "unknown"

def filter_regime(price_data, target_regime="bullish"):
    """
    주어진 가격 데이터를 바탕으로 결정된 시장 레짐이 target_regime과 일치하는지 확인합니다.
    
    파라미터:
      - price_data (dict): 가격 데이터 딕셔너리.
      - target_regime (str): 확인할 목표 레짐 (기본값: 'bullish')
    
    반환값:
      - bool: 결정된 레짐이 target_regime과 일치하면 True, 아니면 False.
    """
    regime = determine_market_regime(price_data)
    match = (regime == target_regime)
    logger.info(f"레짐 필터링: 목표={target_regime}, 결정={regime}, 일치 여부={match}")
    return match

def filter_by_confidence(hmm_model, df, feature_columns, threshold=0.8):
    """
    HMM 모델의 예측 확률을 이용해 각 행의 예측 신뢰도를 평가합니다.
    
    파라미터:
      - hmm_model: 학습된 HMM 모델 (predict_proba 메서드를 제공해야 합니다).
      - df (pd.DataFrame): 평가할 데이터 프레임.
      - feature_columns (list): 특징으로 사용할 컬럼명 리스트.
      - threshold (float): 신뢰도 임계치 (기본값: 0.8).
      
    반환값:
      - List[bool]: 각 행에 대해 최대 확률이 임계치 이상이면 True, 아니면 False.
    """
    try:
        # 새로 추가한 predict_proba 메서드를 호출합니다.
        probabilities = hmm_model.predict_proba(df, feature_columns=feature_columns)
        confidence_flags = [max(probs) >= threshold for probs in probabilities]
        logger.info(f"Computed confidence flags with threshold {threshold}.")
        return confidence_flags
    except Exception as e:
        logger.error(f"Error computing confidence flags: {e}", exc_info=True)
        return [False] * len(df)
