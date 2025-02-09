# markets_analysis/regime_filter.py
import numpy as np
from logs.logger_config import setup_logger

logger = setup_logger(__name__)

def filter_by_confidence(hmm_model, data, feature_columns, threshold=0.8):
    """
    HMM 모델의 posterior 확률을 이용해 각 샘플의 예측 신뢰도를 평가합니다.
    
    :param hmm_model: MarketRegimeHMM 객체
    :param data: 예측에 사용할 pandas DataFrame
    :param feature_columns: 예측에 사용할 피처 컬럼 리스트 (예: ["returns", "volatility", ...])
    :param threshold: 신뢰도 임계치 (0.0 ~ 1.0)
    :return: numpy array (각 샘플이 threshold 이상이면 True, 아니면 False)
    """
    try:
        X = data[feature_columns].values
        logger.debug(f"Score samples input shape: {X.shape}")
        # score_samples()를 사용해 각 샘플의 posterior 확률 분포를 구합니다.
        logprob, posteriors = hmm_model.model.score_samples(X)
        logger.debug(f"Posterior sample stats: logprob_mean={np.mean(logprob):.4f}, max_posteriors_mean={np.mean(posteriors.max(axis=1)):.4f}")
        max_probs = posteriors.max(axis=1)
        confidence_flags = max_probs >= threshold
        logger.info(f"Filter by confidence applied on {len(confidence_flags)} samples with threshold {threshold}.")
        return confidence_flags
    except Exception as e:
        logger.error(f"filter_by_confidence 에러: {e}", exc_info=True)
        return np.array([])

def adjust_regime(prediction, technical_indicators):
    """
    HMM 예측 결과와 보조 기술적 지표를 바탕으로 최종 시장 레짐을 조정합니다.
    
    :param prediction: HMM 예측 결과 (예: 정수형 상태값)
    :param technical_indicators: dict, 보조 지표 분석 결과 (예: {"trend": "bullish"})
    :return: 최종 조정된 시장 레짐 (문자열)
    """
    # HMM 상태를 미리 정의된 매핑으로 변환합니다.
    state_mapping = {0: "bullish", 1: "bearish", 2: "sideways"}
    hmm_regime = state_mapping.get(prediction, "unknown") if isinstance(prediction, int) else prediction
    
    # 보조 지표에서 분석한 추세(trend) 값을 가져옵니다.
    tech_trend = technical_indicators.get("trend", None)
    
    # 보조 지표가 유효하면 우선 적용합니다.
    if tech_trend in ["bullish", "bearish", "sideways"]:
        adjusted_regime = tech_trend
    else:
        adjusted_regime = hmm_regime

    logger.info(f"Adjusted regime: HMM prediction={hmm_regime}, technical trend={tech_trend} -> final regime={adjusted_regime}")
    logger.debug(f"Technical indicators detail: {technical_indicators}")
    return adjusted_regime

def get_regime_intervals(regime_series):
    """
    regime_series: pandas Series, 인덱스는 날짜, 값은 레짐 (예: bullish, bearish, sideways)
    반환: 각 레짐별 (레짐, 시작일, 종료일) 튜플 리스트
    """
    intervals = []
    if regime_series.empty:
        logger.warning("get_regime_intervals: Empty regime series provided.")
        return intervals
    current_regime = regime_series.iloc[0]
    start_date = regime_series.index[0]
    logger.debug(f"Initial regime: {current_regime} starting at {start_date}")
    for dt, regime in regime_series.iteritems():
        if regime != current_regime:
            end_date = dt
            intervals.append((current_regime, start_date, end_date))
            logger.debug(f"Regime change detected: {current_regime} from {start_date} to {end_date}")
            current_regime = regime
            start_date = dt
    intervals.append((current_regime, start_date, regime_series.index[-1]))
    logger.info(f"Calculated {len(intervals)} regime intervals.")
    return intervals
