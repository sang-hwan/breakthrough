# config/config_manager.py
from logs.logger_config import setup_logger

class ConfigManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # 초기화가 이미 이루어진 경우 재초기화를 방지
        if hasattr(self, '_initialized') and self._initialized:
            return

        # 기본 설정 파라미터 정의 (원본 dynamic_param_manager.py의 default_params를 재구성)
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
            # 주간 전략 관련 파라미터
            "weekly_breakout_threshold": 0.01,
            "weekly_momentum_threshold": 0.5,
            "weekly_risk_coefficient": 1.0
        }

        # 민감도 분석 대상 파라미터 목록 (나중에 merge할 때 평균을 적용)
        self.sensitivity_keys = [
            "profit_ratio",
            "atr_multiplier",
            "risk_per_trade",
            "scale_in_threshold",
            "weekly_breakout_threshold",
            "weekly_momentum_threshold"
        ]

        self.logger = setup_logger(__name__)
        self.logger.debug("ConfigManager 초기화 완료: 기본 설정 파라미터 로드됨.")
        self._initialized = True

    def get_defaults(self) -> dict:
        """
        기본 설정 파라미터 사전을 복사하여 반환합니다.
        """
        return self.defaults.copy()

    def update_with_market_data(self, market_data: dict) -> dict:
        """
        시장 데이터를 반영하여 설정 파라미터를 동적으로 업데이트합니다.
        - volatility, trend, trend_strength, volume, weekly_volatility 등을 활용
        """
        config = self.get_defaults()
        volatility = market_data.get("volatility", 0.0)
        trend = market_data.get("trend", "neutral")
        trend_strength = market_data.get("trend_strength", None)
        volume = market_data.get("volume", None)
        weekly_volatility = market_data.get("weekly_volatility", None)

        base_volatility = 0.05
        if volatility > base_volatility:
            factor = 1 + 0.5 * ((volatility - base_volatility) / base_volatility)
            config["atr_multiplier"] *= factor
            config["volatility_multiplier"] = 1.2
            self.logger.debug(f"높은 단기 변동성 감지 (volatility={volatility}): atr_multiplier 조정 계수={factor:.2f}")
        else:
            factor = 1 - 0.3 * ((base_volatility - volatility) / base_volatility)
            config["atr_multiplier"] *= factor
            config["volatility_multiplier"] = 1.0
            self.logger.debug(f"낮은 단기 변동성 감지 (volatility={volatility}): atr_multiplier 조정 계수={factor:.2f}")

        if weekly_volatility is not None:
            if weekly_volatility > 0.07:
                config["weekly_risk_coefficient"] *= 1.2
                self.logger.debug(f"높은 주간 변동성 감지 (weekly_volatility={weekly_volatility}): weekly_risk_coefficient 상향 조정.")
            else:
                config["weekly_risk_coefficient"] *= 0.9
                self.logger.debug(f"낮은 주간 변동성 감지 (weekly_volatility={weekly_volatility}): weekly_risk_coefficient 하향 조정.")

        if trend_strength is not None:
            config["profit_ratio"] *= (1 + trend_strength)
            self.logger.debug(f"추세 강도 반영: trend_strength={trend_strength}, profit_ratio 조정됨.")
        else:
            if trend.lower() == "bullish":
                config["profit_ratio"] *= 1.05
                self.logger.debug("Bullish 추세 감지: profit_ratio 상향 조정.")
            elif trend.lower() == "bearish":
                config["profit_ratio"] *= 0.95
                self.logger.debug("Bearish 추세 감지: profit_ratio 하향 조정.")
            else:
                self.logger.debug("중립 추세: profit_ratio 변경 없음.")

        volume_threshold = 1000
        if volume is not None:
            if volume < volume_threshold:
                factor = volume / volume_threshold
                config["risk_per_trade"] *= factor
                self.logger.debug(f"낮은 거래량 감지 (volume={volume}): risk_per_trade 축소, 계수={factor:.2f}.")
            else:
                config["risk_per_trade"] *= 1.05
                self.logger.debug(f"높은 거래량 감지 (volume={volume}): risk_per_trade 소폭 상향 조정.")

        self.logger.debug(f"Market data: {market_data}")
        self.logger.debug(f"업데이트된 설정 파라미터: {config}")
        return config

    def merge_optimized(self, optimized: dict) -> dict:
        """
        최적화 결과로 도출된 파라미터와 기본 설정 파라미터를 병합하여 반환합니다.
        민감도 분석 대상 파라미터는 기본값과 최적화값의 평균(50:50)을 사용하고,
        그 외의 파라미터는 최적화값을 우선 적용합니다.
        """
        merged = self.get_defaults()
        for key in self.sensitivity_keys:
            opt_value = optimized.get(key, None)
            default_value = merged.get(key)
            if opt_value is not None and isinstance(default_value, (int, float)) and isinstance(opt_value, (int, float)):
                merged[key] = (default_value + opt_value) / 2
            elif opt_value is not None:
                merged[key] = opt_value
        for key, opt_value in optimized.items():
            if key not in self.sensitivity_keys:
                merged[key] = opt_value
        self.logger.debug(f"병합된 설정 파라미터: {merged}")
        return merged
