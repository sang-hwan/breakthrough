# markets_analysis/regime_filter.py
import numpy as np

def filter_by_confidence(hmm_model, data, feature_columns, threshold=0.8):
    """
    HMM 모델의 posterior 확률을 이용해 각 샘플의 예측 신뢰도를 평가합니다.
    
    :param hmm_model: MarketRegimeHMM 객체
    :param data: 예측에 사용할 pandas DataFrame
    :param feature_columns: 예측에 사용할 피처 컬럼 리스트 (예: ["returns", "volatility"])
    :param threshold: 신뢰도 임계치 (0.0 ~ 1.0)
    :return: numpy array (각 샘플이 threshold 이상이면 True, 아니면 False)
    """
    X = data[feature_columns].values
    # score_samples()를 사용해 각 샘플의 posterior 확률 분포를 구합니다.
    logprob, posteriors = hmm_model.model.score_samples(X)
    max_probs = posteriors.max(axis=1)
    return max_probs >= threshold

def adjust_regime(prediction, technical_indicators):
    """
    HMM 예측 결과와 보조 기술적 지표를 바탕으로 최종 시장 레짐을 조정합니다.
    
    :param prediction: HMM 예측 결과 (예: 정수형 상태값)
    :param technical_indicators: dict, 보조 지표 분석 결과 (예: {"trend": "bullish"})
    :return: 최종 조정된 시장 레짐 (문자열)
    """
    # HMM의 상태를 미리 정의된 매핑으로 변환합니다.
    state_mapping = {0: "bullish", 1: "bearish", 2: "sideways"}
    hmm_regime = state_mapping.get(prediction, "unknown") if isinstance(prediction, int) else prediction
    
    # 보조 지표에서 분석한 추세(trend) 값을 가져옵니다.
    tech_trend = technical_indicators.get("trend", None)
    
    # 간단한 조정 로직:
    # 보조 지표가 유효한 경우, 그 값을 우선적으로 사용합니다.
    if tech_trend in ["bullish", "bearish", "sideways"]:
        return tech_trend
    else:
        return hmm_regime
