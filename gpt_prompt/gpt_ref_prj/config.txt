# config/config_manager.py
from logs.logger_config import setup_logger

class ConfigManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return

        # 기본 설정 파라미터
        self.defaults = {
            "sma_period": 200,
            "atr_period": 14,
            "atr_multiplier": 2.07,
            "dynamic_sl_adjustment": 1.18,
            "profit_ratio": 0.098,
            "use_trailing_stop": True,
            "trailing_percent": 0.045,
            "partial_exit_ratio": 0.5,
            "partial_profit_ratio": 0.03,
            "final_profit_ratio": 0.06,
            "risk_per_trade": 0.0162,
            "total_splits": 3,
            "allocation_mode": "equal",
            "scale_in_threshold": 0.0153,
            "hmm_confidence_threshold": 0.8,
            "liquidity_info": "high",
            "volatility_multiplier": 1.0,
            "use_candle_pattern": True,
            "weekly_breakout_threshold": 0.01,
            "weekly_momentum_threshold": 0.5,
            "weekly_risk_coefficient": 1.0
        }

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
        """기본 설정 파라미터 사전을 복사하여 반환."""
        return self.defaults.copy()

    def update_with_market_data(self, market_data: dict) -> dict:
        """
        시장 데이터를 반영하여 설정 파라미터를 동적으로 업데이트합니다.
        (주요 입력: volatility, trend, trend_strength, volume, weekly_volatility)
        """
        config = self.get_defaults()
        volatility = market_data.get("volatility", 0.0)
        trend = market_data.get("trend", "neutral").lower()
        trend_strength = market_data.get("trend_strength")
        volume = market_data.get("volume")
        weekly_volatility = market_data.get("weekly_volatility")

        base_volatility = 0.05
        if volatility > base_volatility:
            factor = 1 + 0.5 * ((volatility - base_volatility) / base_volatility)
            config["atr_multiplier"] *= factor
            config["volatility_multiplier"] = 1.2
        else:
            factor = 1 - 0.3 * ((base_volatility - volatility) / base_volatility)
            config["atr_multiplier"] *= factor
            config["volatility_multiplier"] = 1.0

        if weekly_volatility is not None:
            config["weekly_risk_coefficient"] *= 1.2 if weekly_volatility > 0.07 else 0.9

        if trend_strength is not None:
            config["profit_ratio"] *= (1 + trend_strength)
        else:
            if trend == "bullish":
                config["profit_ratio"] *= 1.05
            elif trend == "bearish":
                config["profit_ratio"] *= 0.95

        if volume is not None:
            volume_threshold = 1000
            config["risk_per_trade"] *= (volume / volume_threshold) if volume < volume_threshold else 1.05

        self.logger.debug(f"Updated config with market data: {config}")
        return config

    def merge_optimized(self, optimized: dict) -> dict:
        """
        최적화된 파라미터와 기본 설정을 병합하여 반환합니다.
        민감도 대상 파라미터는 기본값과 최적화값의 평균(50:50)을 사용하며,
        그 외는 최적화값을 우선 적용합니다.
        """
        merged = self.get_defaults()
        for key in self.sensitivity_keys:
            opt_value = optimized.get(key)
            default_value = merged.get(key)
            if opt_value is not None and isinstance(default_value, (int, float)) and isinstance(opt_value, (int, float)):
                merged[key] = (default_value + opt_value) / 2
            elif opt_value is not None:
                merged[key] = opt_value
        for key, opt_value in optimized.items():
            if key not in self.sensitivity_keys:
                merged[key] = opt_value
        self.logger.debug(f"Merged configuration: {merged}")
        return merged
