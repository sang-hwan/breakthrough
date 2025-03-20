# market_analysis/analyze_market.py
from logs.log_config import setup_logger
import pandas as pd
import numpy as np

# 다른 분석 모듈에서 필요한 함수 임포트 (필요 시 실제 모듈에서 호출)
from market_analysis.technical_analysis import compute_rsi
from market_analysis.onchain_analysis import calculate_mvrv_ratio, analyze_exchange_flow
from market_analysis.sentiment_analysis import aggregate_sentiment
from market_analysis.ml_market_analysis import MarketLSTMAnalyzer

logger = setup_logger(__name__)

def get_technical_signal(price_series: pd.Series, window: int = 14) -> str:
    """
    기술적 분석 신호를 산출합니다.
    - RSI가 50 이상이면 bullish, 50 미만이면 bearish로 판단하며,
      단, 극단 값(예: RSI가 70 이상 또는 30 이하)일 경우 추가 고려할 수 있습니다.
      
    Parameters:
        price_series (pd.Series): 가격 시계열 데이터.
        window (int): RSI 계산 기간 (기본: 14).
    
    Returns:
        str: 'bullish', 'bearish' 또는 'sideways'.
    """
    try:
        rsi = compute_rsi(price_series, window)
        current_rsi = rsi.iloc[-1]
        logger.debug(f"Computed RSI: {current_rsi:.2f}")
        if current_rsi >= 50:
            return "bullish"
        elif current_rsi < 50:
            return "bearish"
        else:
            return "sideways"
    except Exception as e:
        logger.error(f"Error in technical signal computation: {e}", exc_info=True)
        return "sideways"

def get_onchain_signal(market_cap: float, realized_cap: float,
                       exchange_inflow: float, exchange_outflow: float,
                       mvrv_threshold: float = 2.0) -> str:
    """
    온체인 데이터를 바탕으로 시장 신호를 산출합니다.
    
    - MVRV 비율이 높으면(예: mvrv > threshold) bearish, 낮으면 bullish로 판단합니다.
    - 거래소 유입/유출 데이터에 따라 'accumulation'은 bullish, 'distribution'은 bearish 신호로 해석합니다.
    - 두 신호가 상반될 경우 'sideways'로 판단합니다.
    
    Parameters:
        market_cap (float): 시가총액.
        realized_cap (float): 실현 시가총액.
        exchange_inflow (float): 거래소 유입량.
        exchange_outflow (float): 거래소 유출량.
        mvrv_threshold (float): MVRV 비율 임계값 (기본: 2.0).
        
    Returns:
        str: 'bullish', 'bearish' 또는 'sideways'.
    """
    try:
        mvrv = calculate_mvrv_ratio(market_cap, realized_cap)
        flow_signal = analyze_exchange_flow(exchange_inflow, exchange_outflow)
        logger.debug(f"MVRV ratio: {mvrv:.2f}, Flow signal: {flow_signal}")

        # 단순 판단: MVRV가 임계값보다 높으면 bearish, 낮으면 bullish
        onchain_signal_mvrv = "bearish" if mvrv > mvrv_threshold else "bullish"
        
        # 거래소 흐름 신호: accumulation은 bullish, distribution은 bearish
        onchain_signal_flow = "bullish" if flow_signal == "accumulation" else "bearish"
        
        # 두 신호가 일치하면 해당 신호, 다르면 중립적 판단
        if onchain_signal_mvrv == onchain_signal_flow:
            return onchain_signal_mvrv
        else:
            return "sideways"
    except Exception as e:
        logger.error(f"Error in onchain signal computation: {e}", exc_info=True)
        return "sideways"

def get_sentiment_signal(texts: list, bullish_threshold: float = 0.1, bearish_threshold: float = -0.1) -> str:
    """
    감성 분석 결과를 기반으로 시장 신호를 산출합니다.
    
    Parameters:
        texts (list): 분석할 텍스트 문자열 리스트.
        bullish_threshold (float): 긍정적 신호 기준 (기본: 0.1).
        bearish_threshold (float): 부정적 신호 기준 (기본: -0.1).
        
    Returns:
        str: 'bullish', 'bearish' 또는 'sideways'.
    """
    try:
        sentiment = aggregate_sentiment(texts)
        logger.debug(f"Aggregated sentiment score: {sentiment:.2f}")
        if sentiment >= bullish_threshold:
            return "bullish"
        elif sentiment <= bearish_threshold:
            return "bearish"
        else:
            return "sideways"
    except Exception as e:
        logger.error(f"Error in sentiment signal computation: {e}", exc_info=True)
        return "sideways"

def get_ml_signal(ml_model: MarketLSTMAnalyzer, ml_input: pd.DataFrame, current_price: float) -> str:
    """
    머신러닝 모델의 예측 결과를 기반으로 시장 신호를 산출합니다.
    
    Parameters:
        ml_model (MarketLSTMAnalyzer): 학습된 LSTM 분석 모델.
        ml_input (pd.DataFrame): 예측에 사용할 데이터.
        current_price (float): 현재 가격.
    
    Returns:
        str: 'bullish' (예측 가격이 현재보다 높음), 'bearish' (낮음), 또는 'sideways' (변화 미미).
    """
    try:
        predictions = ml_model.predict(ml_input)
        # 예측 결과는 (n_samples, 1) 배열로 가정
        predicted_price = predictions[-1, 0]
        logger.debug(f"ML predicted price: {predicted_price:.2f}, current price: {current_price:.2f}")
        if predicted_price > current_price * 1.01:  # 1% 이상 상승 예측 시 bullish
            return "bullish"
        elif predicted_price < current_price * 0.99:  # 1% 이상 하락 예측 시 bearish
            return "bearish"
        else:
            return "sideways"
    except Exception as e:
        logger.error(f"Error in ML signal computation: {e}", exc_info=True)
        return "sideways"
      
def determine_final_market_state(technical_signal: str,
                                 onchain_signal: str,
                                 sentiment_signal: str,
                                 ml_signal: str) -> str:
    """
    개별 분석 신호(기술적, 온체인, 감성, 머신러닝)를 종합하여 최종 시장 상태를 결정합니다.
    
    단순 다수결(majority vote) 방식으로 최종 상태를 산출하며,
    신호 간 상반되는 경우 'sideways'로 판단합니다.
    
    Parameters:
        technical_signal (str): 기술적 분석 신호.
        onchain_signal (str): 온체인 분석 신호.
        sentiment_signal (str): 감성 분석 신호.
        ml_signal (str): 머신러닝 분석 신호.
    
    Returns:
        str: 최종 시장 상태 ('bullish', 'bearish', 'sideways').
    """
    signals = [technical_signal, onchain_signal, sentiment_signal, ml_signal]
    logger.info(f"Individual signals: {signals}")
    counts = {"bullish": signals.count("bullish"),
              "bearish": signals.count("bearish"),
              "sideways": signals.count("sideways")}
    
    logger.debug(f"Signal counts: {counts}")
    
    # 단순 다수결: 가장 많은 신호를 최종 상태로 결정
    final_state = max(counts, key=counts.get)
    # 만약 두 신호 이상이 동일한 경우(예: bullish와 bearish가 같으면) 보수적으로 sideways 반환
    if list(counts.values()).count(counts[final_state]) > 1:
        final_state = "sideways"
    
    logger.info(f"Final market state determined: {final_state}")
    return final_state

def aggregate_market_analysis(technical_data: dict,
                              onchain_data: dict,
                              sentiment_texts: list,
                              ml_data: dict) -> str:
    """
    개별 분석 모듈에서 산출한 데이터를 입력받아 최종 시장 상태를 판단합니다.
    
    Parameters:
        technical_data (dict): {'price_series': pd.Series, 'rsi_window': int (선택)}
        onchain_data (dict): {'market_cap': float, 'realized_cap': float,
                              'exchange_inflow': float, 'exchange_outflow': float}
        sentiment_texts (list): 감성 분석 대상 텍스트 리스트.
        ml_data (dict): {'ml_model': MarketLSTMAnalyzer, 'ml_input': pd.DataFrame,
                          'current_price': float}
    
    Returns:
        str: 최종 시장 상태 ('bullish', 'bearish', 'sideways').
    """
    try:
        # 기술적 분석 신호 산출
        technical_signal = get_technical_signal(technical_data.get("price_series"),
                                                  technical_data.get("rsi_window", 14))
        
        # 온체인 분석 신호 산출
        onchain_signal = get_onchain_signal(onchain_data.get("market_cap"),
                                            onchain_data.get("realized_cap"),
                                            onchain_data.get("exchange_inflow"),
                                            onchain_data.get("exchange_outflow"))
        
        # 감성 분석 신호 산출
        sentiment_signal = get_sentiment_signal(sentiment_texts)
        
        # 머신러닝 분석 신호 산출
        ml_signal = get_ml_signal(ml_data.get("ml_model"),
                                  ml_data.get("ml_input"),
                                  ml_data.get("current_price"))
        
        # 최종 신호 종합
        final_state = determine_final_market_state(technical_signal,
                                                   onchain_signal,
                                                   sentiment_signal,
                                                   ml_signal)
        return final_state
    except Exception as e:
        logger.error(f"Error aggregating market analysis: {e}", exc_info=True)
        return "sideways"
