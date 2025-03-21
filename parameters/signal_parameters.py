# parameters/signal_parameters.py
from pydantic import BaseModel, Field
from logs.log_config import setup_logger

logger = setup_logger(__name__)

class SignalConfig(BaseModel):
    """
    신호 계산 관련 기본 파라미터를 정의하는 모델.
    """
    rsi_period: int = Field(default=14, ge=1, description="RSI 계산 기간")
    macd_fast: int = Field(default=12, ge=1, description="MACD 계산을 위한 빠른 EMA 기간")
    macd_slow: int = Field(default=26, ge=1, description="MACD 계산을 위한 느린 EMA 기간")
    macd_signal: int = Field(default=9, ge=1, description="MACD 시그널 라인 기간")
    bollinger_period: int = Field(default=20, ge=1, description="볼린저 밴드 계산 기간")
    bollinger_std: float = Field(default=2.0, gt=0, description="볼린저 밴드 표준편차 배수")

def get_default_signal_config() -> dict:
    """
    기본 신호 계산 파라미터를 반환합니다.
    
    Returns:
        dict: 기본 SignalConfig 파라미터 딕셔너리.
    """
    config = SignalConfig()
    logger.debug("Default SignalConfig loaded: %s", config.model_dump())
    return config.model_dump()
