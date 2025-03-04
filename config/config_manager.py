# config/config_manager.py
from logs.logger_config import setup_logger
from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import Literal

class TradingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    
    sma_period: int = Field(default=200, ge=1)
    atr_period: int = Field(default=14, ge=1)
    atr_multiplier: float = Field(default=2.07, gt=0, le=10)
    dynamic_sl_adjustment: float = Field(default=1.18, gt=0, le=5)
    profit_ratio: float = Field(default=0.098, gt=0, le=1.0)
    use_trailing_stop: bool = True
    trailing_percent: float = Field(default=0.045, gt=0, le=0.5)
    partial_exit_ratio: float = Field(default=0.5, gt=0, lt=1)
    partial_profit_ratio: float = Field(default=0.03, gt=0, le=1.0)
    final_profit_ratio: float = Field(default=0.06, gt=0, le=1.0)
    risk_per_trade: float = Field(default=0.0162, gt=0, le=1.0)
    total_splits: int = Field(default=3, ge=1)
    allocation_mode: Literal['equal', 'pyramid_up', 'pyramid_down'] = "equal"
    scale_in_threshold: float = Field(default=0.0153, gt=0, le=1.0)
    hmm_confidence_threshold: float = Field(default=0.8, ge=0, le=1)
    liquidity_info: Literal['high', 'low'] = "high"
    volatility_multiplier: float = Field(default=1.0, gt=0, le=5)
    use_candle_pattern: bool = True
    weekly_breakout_threshold: float = Field(default=0.01, gt=0, le=0.1)
    weekly_momentum_threshold: float = Field(default=0.5, gt=0, le=1.0)
    weekly_risk_coefficient: float = Field(default=1.0, gt=0, le=5)
    weight_vol_threshold: float = Field(default=0.05, gt=0, le=1)
    vol_weight_factor: float = Field(default=0.9, gt=0, le=2)
    liquidity_weight_high: float = Field(default=0.8, gt=0, lt=1)
    liquidity_weight_low: float = Field(default=0.6, gt=0, lt=1)

    @model_validator(mode="after")
    def check_profit_ratios(cls, model: "TradingConfig") -> "TradingConfig":
        if model.partial_profit_ratio >= model.final_profit_ratio:
            raise ValueError("partial_profit_ratio must be less than final_profit_ratio")
        return model

class ConfigManager:
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
        self.logger = setup_logger(__name__)
        self.logger.debug("ConfigManager initialized with default parameters.")
        self._initialized = True

    def get_defaults(self) -> dict:
        return self.defaults.model_dump()

    def get_dynamic_params(self) -> dict:
        return self.dynamic_params.copy()

    def update_with_market_data(self, market_data: dict) -> dict:
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
                self.logger.debug(f"Weekly strategy parameters adjusted due to high spread ratio: {spread_ratio:.2f}")

        try:
            updated_config = TradingConfig(**config_dict)
        except Exception as e:
            self.logger.error("Validation error in updated configuration: " + str(e), exc_info=True)
            raise

        self.dynamic_params = updated_config.model_dump()
        self.logger.debug(f"Updated config with market data: {self.dynamic_params}")
        return self.dynamic_params

    def merge_optimized(self, optimized: dict) -> dict:
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
            self.logger.error("Validation error in merged configuration: " + str(e), exc_info=True)
            raise

        self.logger.debug(f"Merged configuration: {merged_config.model_dump()}")
        self.dynamic_params = merged_config.model_dump()
        return self.dynamic_params

    def validate_params(self, params: dict) -> dict:
        try:
            validated = TradingConfig(**params)
            return validated.model_dump()
        except Exception as e:
            self.logger.error("Dynamic parameter validation failed: " + str(e), exc_info=True)
            return self.get_defaults()
