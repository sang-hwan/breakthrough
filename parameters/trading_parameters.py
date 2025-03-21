# parameters/trading_parameters.py
from logs.log_config import setup_logger
from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import Literal

logger = setup_logger(__name__)

class TradingConfig(BaseModel):
    """
    거래 전략에 사용되는 파라미터를 정의 및 검증하는 모델.
    """
    model_config = ConfigDict(extra="allow")
    
    sma_period: int = Field(default=200, ge=1, description="단순 이동평균 계산 기간")
    atr_period: int = Field(default=14, ge=1, description="ATR 계산 기간")
    atr_multiplier: float = Field(default=2.07, gt=0, le=10, description="ATR에 곱해지는 계수")
    dynamic_sl_adjustment: float = Field(default=1.18, gt=0, le=5, description="동적 손절 조정 계수")
    profit_ratio: float = Field(default=0.098, gt=0, le=1.0, description="목표 이익 비율")
    use_trailing_stop: bool = Field(default=True, description="트레일링 스탑 사용 여부")
    trailing_percent: float = Field(default=0.045, gt=0, le=0.5, description="트레일링 스탑 기준 퍼센트")
    partial_exit_ratio: float = Field(default=0.5, gt=0, lt=1, description="부분 청산 비율")
    partial_profit_ratio: float = Field(default=0.03, gt=0, le=1.0, description="부분 이익 목표 비율")
    final_profit_ratio: float = Field(default=0.06, gt=0, le=1.0, description="최종 이익 목표 비율")
    risk_per_trade: float = Field(default=0.0162, gt=0, le=1.0, description="거래당 위험 비율")
    total_splits: int = Field(default=3, ge=1, description="분할 거래 횟수")
    allocation_mode: Literal['equal', 'pyramid_up', 'pyramid_down'] = Field(default="equal", description="자금 배분 모드")
    scale_in_threshold: float = Field(default=0.0153, gt=0, le=1.0, description="추가 진입 임계값")
    hmm_confidence_threshold: float = Field(default=0.8, ge=0, le=1, description="HMM 신뢰도 임계값")
    liquidity_info: Literal['high', 'low'] = Field(default="high", description="유동성 정보")
    volatility_multiplier: float = Field(default=1.0, gt=0, le=5, description="변동성 조정 계수")
    use_candle_pattern: bool = Field(default=True, description="캔들 패턴 사용 여부")
    weekly_breakout_threshold: float = Field(default=0.01, gt=0, le=0.1, description="주간 돌파 임계값")
    weekly_momentum_threshold: float = Field(default=0.5, gt=0, le=1.0, description="주간 모멘텀 임계값")
    weekly_risk_coefficient: float = Field(default=1.0, gt=0, le=5, description="주간 위험 계수")
    weight_vol_threshold: float = Field(default=0.05, gt=0, le=1, description="거래량 가중치 임계값")
    vol_weight_factor: float = Field(default=0.9, gt=0, le=2, description="거래량 가중치 계수")
    liquidity_weight_high: float = Field(default=0.8, gt=0, lt=1, description="높은 유동성에 적용할 가중치")
    liquidity_weight_low: float = Field(default=0.6, gt=0, lt=1, description="낮은 유동성에 적용할 가중치")

    @model_validator(mode="after")
    def check_profit_ratios(cls, model: "TradingConfig") -> "TradingConfig":
        """
        TradingConfig 객체 생성 후, 부분 이익 목표 비율이 최종 이익 목표 비율보다 작은지 검증합니다.
        
        Raises:
            ValueError: partial_profit_ratio가 final_profit_ratio보다 크거나 같을 경우.
        """
        if model.partial_profit_ratio >= model.final_profit_ratio:
            raise ValueError("partial_profit_ratio must be less than final_profit_ratio")
        return model

class ConfigManager:
    """
    거래 파라미터의 기본값 관리 및 동적 업데이트를 위한 싱글턴 매니저 클래스.
    
    과적합 방지를 위해 기본값과 최적화 결과(민감도 분석)를 병합하거나, 시장 데이터 기반으로 파라미터를
    안정적으로 업데이트하는 전략을 포함합니다.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.defaults = TradingConfig()
        self.dynamic_params = self.defaults.model_dump()
        self.sensitivity_keys = [
            "profit_ratio",
            "atr_multiplier",
            "risk_per_trade",
            "scale_in_threshold",
            "weekly_breakout_threshold",
            "weekly_momentum_threshold"
        ]
        self.logger = logger
        self.logger.debug("ConfigManager initialized with default trading parameters.")
        self._initialized = True

    def get_defaults(self) -> dict:
        """
        기본 거래 설정값을 딕셔너리로 반환합니다.
        
        Returns:
            dict: 기본 TradingConfig 파라미터.
        """
        return self.defaults.model_dump()

    def get_dynamic_params(self) -> dict:
        """
        현재 동적 파라미터의 사본을 반환합니다.
        
        Returns:
            dict: 동적 파라미터.
        """
        return self.dynamic_params.copy()

    def update_with_market_data(self, market_data: dict) -> dict:
        """
        시장 데이터를 기반으로 거래 파라미터를 동적으로 업데이트합니다.
        
        Parameters:
            market_data (dict): 시장 데이터를 포함하는 딕셔너리.
                예상 키:
                    - volatility: 현재 시장 변동성 (float)
                    - trend: 시장 추세 (예: "bullish", "bearish", "neutral")
                    - trend_strength: 추세 강도 (float, 선택적)
                    - volume: 거래량 (float, 선택적)
                    - weekly_volatility: 주간 변동성 (float, 선택적)
                    - weekly_low: 주간 최저가 (float, 선택적)
                    - weekly_high: 주간 최고가 (float, 선택적)
        
        Returns:
            dict: 시장 데이터 반영 후 업데이트된 동적 파라미터.
        """
        config_dict = self.get_defaults()
        volatility = market_data.get("volatility", 0.0)
        trend = market_data.get("trend", "neutral").lower()
        trend_strength = market_data.get("trend_strength")
        volume = market_data.get("volume")
        weekly_volatility = market_data.get("weekly_volatility")
        base_volatility = 0.05
        if volatility > base_volatility:
            factor = 1 + 0.5 * ((volatility - base_volatility) / base_volatility)
            config_dict["atr_multiplier"] *= factor
            config_dict["volatility_multiplier"] = 1.2
        else:
            factor = 1 - 0.3 * ((base_volatility - volatility) / base_volatility)
            config_dict["atr_multiplier"] *= factor
            config_dict["volatility_multiplier"] = 1.0

        if weekly_volatility is not None:
            config_dict["weekly_risk_coefficient"] *= 1.2 if weekly_volatility > 0.07 else 0.9

        if trend_strength is not None:
            config_dict["profit_ratio"] *= (1 + trend_strength)
        else:
            if trend == "bullish":
                config_dict["profit_ratio"] *= 1.05
            elif trend == "bearish":
                config_dict["profit_ratio"] *= 0.95

        if volume is not None:
            volume_threshold = 1000
            config_dict["risk_per_trade"] *= (volume / volume_threshold) if volume < volume_threshold else 1.05

        weekly_low = market_data.get("weekly_low")
        weekly_high = market_data.get("weekly_high")
        if weekly_low is not None and weekly_high is not None and weekly_low > 0:
            spread_ratio = (weekly_high - weekly_low) / weekly_low
            if spread_ratio > 0.05:
                config_dict["weekly_breakout_threshold"] = max(config_dict["weekly_breakout_threshold"] * 0.8, 0.005)
                config_dict["weekly_momentum_threshold"] = min(config_dict["weekly_momentum_threshold"] * 1.05, 0.7)
                self.logger.debug("Weekly strategy parameters adjusted due to high spread ratio: %.2f", spread_ratio)

        try:
            updated_config = TradingConfig(**config_dict)
        except Exception as e:
            self.logger.error("Validation error in updated configuration: %s", e, exc_info=True)
            raise

        self.dynamic_params = updated_config.model_dump()
        self.logger.debug("Updated config with market data: %s", self.dynamic_params)
        return self.dynamic_params

    def merge_optimized(self, optimized: dict) -> dict:
        """
        최적화된 파라미터와 기본 설정값을 병합하여 업데이트된 파라미터를 생성합니다.
        
        Parameters:
            optimized (dict): 최적화된 파라미터 딕셔너리.
        
        Returns:
            dict: 병합 후 업데이트된 동적 파라미터.
        """
        merged_dict = self.get_defaults()
        for key in self.sensitivity_keys:
            opt_value = optimized.get(key)
            default_value = merged_dict.get(key)
            if opt_value is not None and isinstance(default_value, (int, float)) and isinstance(opt_value, (int, float)):
                merged_dict[key] = (default_value + opt_value) / 2
            elif opt_value is not None:
                merged_dict[key] = opt_value
        for key, opt_value in optimized.items():
            if key not in self.sensitivity_keys:
                merged_dict[key] = opt_value

        try:
            merged_config = TradingConfig(**merged_dict)
        except Exception as e:
            self.logger.error("Validation error in merged configuration: %s", e, exc_info=True)
            raise

        self.logger.debug("Merged configuration: %s", merged_config.model_dump())
        self.dynamic_params = merged_config.model_dump()
        return self.dynamic_params

    def validate_params(self, params: dict) -> dict:
        """
        입력된 파라미터를 TradingConfig 모델로 검증하며, 유효하지 않으면 기본 설정값을 반환합니다.
        
        Parameters:
            params (dict): 검증 대상 파라미터 딕셔너리.
        
        Returns:
            dict: 검증된 파라미터. 검증 실패 시 기본 설정값 반환.
        """
        try:
            validated = TradingConfig(**params)
            return validated.model_dump()
        except Exception as e:
            self.logger.error("Dynamic parameter validation failed: %s", e, exc_info=True)
            return self.get_defaults()
