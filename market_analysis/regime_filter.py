# markets/regime_filter.py

# 로그 설정 모듈에서 설정 함수 import
from logging.logger_config import setup_logger

# 글로벌 로거 객체 정의:
#  - 이 모듈에서 발생하는 이벤트(디버그, 에러 등)를 기록하기 위해 사용됩니다.
logger = setup_logger(__name__)

def determine_market_regime(price_data: dict) -> str:
    """
    주어진 가격 데이터를 기반으로 시장 레짐(시장 상태)을 결정하는 함수입니다.
    
    이 함수는 'current_price'와 'previous_price'를 포함하는 가격 데이터를 입력받아,
    가격 변화율에 따라 'bullish' (상승장), 'bearish' (하락장), 'sideways' (횡보장) 중 하나를 반환합니다.
    
    Parameters:
      - price_data (dict): 현재 가격과 이전 가격 정보를 포함하는 딕셔너리.
        필수 키: 'current_price', 'previous_price'
    
    Returns:
      - str: 결정된 시장 레짐. ('bullish', 'bearish', 'sideways', 에러 시 'unknown')
    """
    try:
        # price_data 딕셔너리에서 현재 가격과 이전 가격을 추출합니다.
        current_price = price_data.get("current_price")
        previous_price = price_data.get("previous_price")
        
        # 필수 데이터가 누락되었을 경우, 에러 로그를 남기고 'unknown'을 반환합니다.
        if current_price is None or previous_price is None:
            logger.error("필수 가격 데이터 누락: 'current_price' 또는 'previous_price'가 제공되지 않음.", exc_info=True)
            return "unknown"
        
        # 가격 변화율 계산: (현재가격 - 이전가격) / 이전가격
        change_percent = (current_price - previous_price) / previous_price
        
        # 변화율 기준에 따라 시장 상태 결정
        if change_percent > 0.02:
            # 변화율이 2% 이상이면 상승장(bullish)으로 간주
            regime = "bullish"
        elif change_percent < -0.02:
            # 변화율이 -2% 이하이면 하락장(bearish)으로 간주
            regime = "bearish"
        else:
            # 2% 이내의 변화는 횡보장(sideways)으로 간주
            regime = "sideways"
        
        # 결정된 상태와 계산된 변화율을 디버깅 로그에 기록합니다.
        logger.debug(f"시장 레짐 결정: {regime} (변화율: {change_percent:.2%})")
        return regime
    except Exception as e:
        # 예외 발생 시, 에러 로그를 기록하고 'unknown'을 반환합니다.
        logger.error(f"시장 레짐 결정 중 에러 발생: {e}", exc_info=True)
        return "unknown"

def filter_regime(price_data: dict, target_regime: str = "bullish") -> bool:
    """
    결정된 시장 레짐이 목표 레짐과 일치하는지 확인하는 함수입니다.
    
    주어진 가격 데이터를 사용하여 시장 상태를 판단한 뒤,
    목표 레짐과 비교하여 일치하면 True, 그렇지 않으면 False를 반환합니다.
    
    Parameters:
      - price_data (dict): 가격 데이터를 포함하는 딕셔너리.
      - target_regime (str): 목표로 하는 시장 레짐 (기본값은 'bullish').
    
    Returns:
      - bool: 목표 레짐과 일치하면 True, 아니면 False.
    """
    # 위에서 정의한 determine_market_regime 함수를 호출하여 현재 시장 상태를 파악합니다.
    regime = determine_market_regime(price_data)
    
    # 결정된 상태가 목표 상태와 동일한지 여부를 비교합니다.
    match = (regime == target_regime)
    
    # 비교 결과(일치 여부)를 디버깅 로그에 기록합니다.
    logger.debug(f"레짐 필터링: 목표={target_regime}, 결정={regime}, 일치 여부={match}")
    return match

def determine_weekly_extreme_signal(price_data: dict, weekly_data: dict, threshold: float = 0.002) -> str:
    """
    주간 가격 데이터와 현재 가격 데이터를 바탕으로 극단적인 가격 신호(매수 또는 매도)를 판단하는 함수입니다.
    
    현재 가격이 주간 최저치에 근접하면 'enter_long' (매수 신호),
    주간 최고치에 근접하면 'exit_all' (매도 신호)를 반환하며,
    두 조건 모두 만족하지 않으면 빈 문자열("")을 반환합니다.
    
    Parameters:
      - price_data (dict): 'current_price' 키를 포함하는 현재 가격 데이터.
      - weekly_data (dict): 'weekly_low'와 'weekly_high' 키를 포함하는 주간 가격 데이터.
      - threshold (float): 가격이 극값에 근접한 것으로 판단하는 임계값 (기본 0.002, 즉 0.2%).
    
    Returns:
      - str: 주간 신호. 'enter_long'이면 매수 신호, 'exit_all'이면 매도 신호, 조건 미충족 시 빈 문자열.
    """
    try:
        # 현재 가격과 주간 최저/최고 가격 정보를 추출합니다.
        current_price = price_data.get("current_price")
        weekly_low = weekly_data.get("weekly_low")
        weekly_high = weekly_data.get("weekly_high")
        
        # 필수 데이터가 누락된 경우, 에러 로그를 남기고 빈 문자열을 반환합니다.
        if current_price is None or weekly_low is None or weekly_high is None:
            logger.error("필수 데이터 누락: 'current_price', 'weekly_low', 'weekly_high' 모두 제공되어야 합니다.", exc_info=True)
            return ""
        
        # 현재 가격이 주간 최저치에 얼마나 가까운지 비율로 계산 후 임계값과 비교
        if abs(current_price - weekly_low) / weekly_low <= threshold:
            logger.debug(f"주간 저점 신호 감지: current_price={current_price}, weekly_low={weekly_low}")
            return "enter_long"  # 매수 신호: 주간 저점 근접
        # 현재 가격이 주간 최고치에 얼마나 가까운지 비율로 계산 후 임계값과 비교
        elif abs(weekly_high - current_price) / weekly_high <= threshold:
            logger.debug(f"주간 고점 신호 감지: current_price={current_price}, weekly_high={weekly_high}")
            return "exit_all"   # 매도 신호: 주간 고점 근접
        # 신호 조건에 부합하지 않으면 빈 문자열 반환
        return ""
    except Exception as e:
        # 예외 발생 시, 에러 로그를 기록하고 빈 문자열 반환
        logger.error(f"주간 극값 신호 결정 중 에러 발생: {e}", exc_info=True)
        return ""
