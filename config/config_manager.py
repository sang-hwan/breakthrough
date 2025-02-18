# config/config_manager.py
from logs.logger_config import setup_logger
from pydantic import BaseModel, Field, ValidationError
from typing import Literal

class TradingConfig(BaseModel):
    sma_period: int = Field(default=200, ge=1)
    atr_period: int = Field(default=14, ge=1)
    atr_multiplier: float = Field(default=2.07, gt=0)
    dynamic_sl_adjustment: float = Field(default=1.18, gt=0)
    profit_ratio: float = Field(default=0.098, gt=0)
    use_trailing_stop: bool = True
    trailing_percent: float = Field(default=0.045, gt=0)
    partial_exit_ratio: float = Field(default=0.5, gt=0, lt=1)
    partial_profit_ratio: float = Field(default=0.03, gt=0)
    final_profit_ratio: float = Field(default=0.06, gt=0)
    risk_per_trade: float = Field(default=0.0162, gt=0)
    total_splits: int = Field(default=3, ge=1)
    allocation_mode: Literal['equal', 'pyramid_up', 'pyramid_down'] = "equal"
    scale_in_threshold: float = Field(default=0.0153, gt=0)
    hmm_confidence_threshold: float = Field(default=0.8, ge=0, le=1)
    liquidity_info: Literal['high', 'low'] = "high"
    volatility_multiplier: float = Field(default=1.0, gt=0)
    use_candle_pattern: bool = True
    weekly_breakout_threshold: float = Field(default=0.01, gt=0)
    weekly_momentum_threshold: float = Field(default=0.5, gt=0)
    weekly_risk_coefficient: float = Field(default=1.0, gt=0)
    # 추가: 동적 가중치 산출 관련 파라미터 (optional)
    weight_vol_threshold: float = Field(default=0.05, gt=0)
    vol_weight_factor: float = Field(default=0.9, gt=0)
    liquidity_weight_high: float = Field(default=0.8, gt=0, lt=1)
    liquidity_weight_low: float = Field(default=0.6, gt=0, lt=1)
    
    class Config:
        extra = "allow"

class ConfigManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return

        # 기본 설정 파라미터를 Pydantic 모델로 관리
        self.defaults = TradingConfig()
        # 동적 파라미터 저장 변수 (업데이트 시 사용)
        self.dynamic_params = self.defaults.model_dump()
        
        # 민감도 분석 대상 파라미터 (최적화 병합 시 평균 적용)
        self.sensitivity_keys = [
            "profit_ratio",
            "atr_multiplier",
            "risk_per_trade",
            "scale_in_threshold",
            "weekly_breakout_threshold",
            "weekly_momentum_threshold"
        ]
        self.logger = setup_logger(__name__)
        self.logger.debug("ConfigManager initialized with default parameters.")
        self._initialized = True

    def get_defaults(self) -> dict:
        """기본 설정 파라미터를 사전 형태로 복사하여 반환."""
        return self.defaults.model_dump()

    def get_dynamic_params(self) -> dict:
        """
        동적 업데이트가 반영된 현재 설정 파라미터를 반환합니다.
        (업데이트가 없으면 기본 설정을 반환)
        """
        return self.dynamic_params.copy()

    def update_with_market_data(self, market_data: dict) -> dict:
        """
        시장 데이터를 반영하여 설정 파라미터를 동적으로 업데이트합니다.
        (업데이트가 없으면 기본 설정을 반환)
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

        # 주간 전략 파라미터 업데이트: weekly_low 및 weekly_high 사용
        weekly_low = market_data.get("weekly_low")
        weekly_high = market_data.get("weekly_high")
        if weekly_low is not None and weekly_high is not None and weekly_low > 0:
            spread_ratio = (weekly_high - weekly_low) / weekly_low
            if spread_ratio > 0.05:
                config_dict["weekly_breakout_threshold"] = max(config_dict["weekly_breakout_threshold"] * 0.8, 0.005)
                config_dict["weekly_momentum_threshold"] = min(config_dict["weekly_momentum_threshold"] * 1.05, 0.7)
                self.logger.debug(f"Weekly strategy parameters adjusted due to high spread ratio: {spread_ratio:.2f}")

        try:
            # 업데이트된 파라미터를 Pydantic을 통해 검증
            updated_config = TradingConfig(**config_dict)
        except ValidationError as e:
            self.logger.error(f"Validation error in updated configuration: {e}", exc_info=True)
            raise

        self.dynamic_params = updated_config.model_dump()
        self.logger.debug(f"Updated config with market data: {self.dynamic_params}")
        return self.dynamic_params

    def merge_optimized(self, optimized: dict) -> dict:
        """
        최적화된 파라미터와 기본 설정을 병합하여 반환합니다.
        민감도 대상 파라미터는 기본값과 최적화값의 평균(50:50)을 사용하며,
        그 외는 최적화값을 우선 적용합니다.
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
        except ValidationError as e:
            self.logger.error(f"Validation error in merged configuration: {e}", exc_info=True)
            raise

        self.logger.debug(f"Merged configuration: {merged_config.model_dump()}")
        # 업데이트된 동적 파라미터에도 반영
        self.dynamic_params = merged_config.model_dump()
        return self.dynamic_params
