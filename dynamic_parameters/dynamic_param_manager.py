# dynamic_parameters/dynamic_param_manager.py
from logs.logger_config import setup_logger

class DynamicParamManager:
    _instance = None  # 싱글턴 인스턴스 저장

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DynamicParamManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # 이미 초기화된 경우 재초기화 방지
        if hasattr(self, '_initialized') and self._initialized:
            return

        # 기본 파라미터 설정 (다른 모듈과 연계되는 주요 값들)
        self.default_params = {
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
            "volatility_multiplier": 1.0,  # 변동성 반영 인자
            "use_candle_pattern": True
        }
        self.logger = setup_logger(__name__)
        self.logger.info("DynamicParamManager 초기화 완료 (레짐 기반 전략 적용).")
        self._initialized = True

    def get_default_params(self) -> dict:
        """
        기본 파라미터 사전을 복사하여 반환합니다.
        """
        return self.default_params.copy()

    def update_dynamic_params(self, market_data: dict) -> dict:
        """
        시장 데이터(예: 변동성, 추세, 거래량 등)에 따라 파라미터를 동적으로 업데이트합니다.
        
        예시:
          - 변동성이 높으면 atr_multiplier 및 volatility_multiplier를 상향 조정.
          - bullish 추세이면 profit_ratio를 소폭 상향, bearish 추세이면 하향.
          - 거래량(volume)이 낮으면 risk_per_trade를 축소하는 방식 적용.
        """
        dynamic_params = self.get_default_params()
        volatility = market_data.get("volatility", 0.0)
        trend = market_data.get("trend", "neutral")
        volume = market_data.get("volume", None)

        # 변동성에 따른 조정
        if volatility > 0.05:
            dynamic_params["atr_multiplier"] *= 1.1
            dynamic_params["volatility_multiplier"] = 1.2
            self.logger.info("높은 변동성 감지: atr_multiplier 및 volatility_multiplier 상향 조정.")
        else:
            dynamic_params["atr_multiplier"] *= 0.95
            dynamic_params["volatility_multiplier"] = 1.0
            self.logger.info("낮거나 보통의 변동성: atr_multiplier 소폭 하향 조정.")

        # 추세에 따른 profit_ratio 조정
        if trend == "bullish":
            dynamic_params["profit_ratio"] *= 1.05
            self.logger.info("Bullish 추세 감지: profit_ratio 상향 조정.")
        elif trend == "bearish":
            dynamic_params["profit_ratio"] *= 0.95
            self.logger.info("Bearish 추세 감지: profit_ratio 하향 조정.")
        else:
            self.logger.info("중립 추세: profit_ratio 변경 없음.")

        # 거래량(volume)에 따른 risk_per_trade 조정
        if volume is not None:
            if volume < 1000:
                dynamic_params["risk_per_trade"] *= 0.9
                self.logger.info("낮은 거래량 감지: risk_per_trade 하향 조정.")
            else:
                dynamic_params["risk_per_trade"] *= 1.05
                self.logger.info("높은 거래량 감지: risk_per_trade 소폭 상향 조정.")

        self.logger.info(f"Market data: {market_data}")
        self.logger.info(f"업데이트된 동적 파라미터: {dynamic_params}")
        self.logger.info("동적 파라미터 업데이트 완료.")
        return dynamic_params

    def merge_params(self, optimized_params: dict) -> dict:
        """
        최적화 결과로 도출된 파라미터와 기본 파라미터를 병합하여 반환합니다.
        숫자형 값의 경우 기본값과 최적화값의 가중 평균(여기서는 단순 평균, 50:50)을 사용하고,
        그 외의 타입은 최적화값을 우선 적용합니다.
        """
        merged = self.get_default_params()
        for key, opt_value in optimized_params.items():
            default_value = merged.get(key)
            if isinstance(default_value, (int, float)) and isinstance(opt_value, (int, float)):
                merged[key] = (default_value + opt_value) / 2
            else:
                merged[key] = opt_value
        self.logger.info(f"병합된 동적 파라미터: {merged}")
        return merged
