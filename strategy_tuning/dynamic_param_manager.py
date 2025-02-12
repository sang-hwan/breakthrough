# strategy_tuning/dynamic_param_manager.py
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

        # 기본 파라미터 설정 (주요 값 및 민감도 분석 대상 파라미터 목록 포함)
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
            "use_candle_pattern": True,
            # 주간 전략 관련 파라미터
            "weekly_breakout_threshold": 0.01,    # 주간 돌파 임계치 (예: 1% 이상)
            "weekly_momentum_threshold": 0.5,       # 주간 모멘텀 기준
            "weekly_risk_coefficient": 1.0          # 주간 리스크 계수 (시장 변동성이 크면 값 상승)
        }
        # 민감도 분석 대상 파라미터 목록 (이 목록은 민감도 분석 시 대상 파라미터를 명확히 함)
        self.sensitivity_params = [
            "profit_ratio",
            "atr_multiplier",
            "risk_per_trade",
            "scale_in_threshold",
            "weekly_breakout_threshold",
            "weekly_momentum_threshold"
        ]
        self.logger = setup_logger(__name__)
        self.logger.debug("DynamicParamManager 초기화 완료 (레짐 및 주간 전략 적용).")
        self._initialized = True

    def get_default_params(self) -> dict:
        """
        기본 파라미터 사전을 복사하여 반환합니다.
        """
        return self.default_params.copy()

    def update_dynamic_params(self, market_data: dict) -> dict:
        """
        시장 데이터(예: 변동성, 추세, 거래량, 추세 강도 등)를 반영하여 파라미터를 동적으로 업데이트합니다.
        
        개선 사항:
          - 단기 변동성(volatility)이 기본 임계치(0.05)보다 높으면, 초과분에 비례하여 atr_multiplier를 증가시킵니다.
          - 시장 추세(trend)는 'bullish'/'bearish' 뿐 아니라, 숫자형 추세 강도(trend_strength)가 있다면 이를 반영합니다.
          - 거래량(volume)은 일정 임계치(예: 1000)보다 낮을 경우 risk_per_trade를 부드럽게 축소합니다.
          - 주간 변동성(weekly_volatility)에 따라 weekly_risk_coefficient도 조정합니다.
        """
        dynamic_params = self.get_default_params()
        volatility = market_data.get("volatility", 0.0)  # 예: 0.08
        trend = market_data.get("trend", "neutral")       # 'bullish', 'bearish', or 'neutral'
        trend_strength = market_data.get("trend_strength", None)  # 숫자형, 예: 0.1 (양수이면 강한 상승)
        volume = market_data.get("volume", None)
        weekly_volatility = market_data.get("weekly_volatility", None)

        # 단기 변동성 조정 (기본 기준: 0.05)
        base_volatility = 0.05
        if volatility > base_volatility:
            # 예: vol 0.08 → 초과분 0.03, 0.03/0.05 = 0.6,  atr_multiplier를 기본 대비 (1 + 0.6*0.5)=1.3배 증가
            factor = 1 + 0.5 * ((volatility - base_volatility) / base_volatility)
            dynamic_params["atr_multiplier"] *= factor
            dynamic_params["volatility_multiplier"] = 1.2  # 고변동 시장에서는 보수적 접근
            self.logger.debug(f"높은 단기 변동성 감지 (volatility={volatility}): atr_multiplier 조정 계수={factor:.2f}")
        else:
            factor = 1 - 0.3 * ((base_volatility - volatility) / base_volatility)
            dynamic_params["atr_multiplier"] *= factor
            dynamic_params["volatility_multiplier"] = 1.0
            self.logger.debug(f"낮은 단기 변동성 감지 (volatility={volatility}): atr_multiplier 조정 계수={factor:.2f}")

        # 주간 변동성 조정
        if weekly_volatility is not None:
            if weekly_volatility > 0.07:
                dynamic_params["weekly_risk_coefficient"] *= 1.2
                self.logger.debug(f"높은 주간 변동성 감지 (weekly_volatility={weekly_volatility}): weekly_risk_coefficient 상향 조정.")
            else:
                dynamic_params["weekly_risk_coefficient"] *= 0.9
                self.logger.debug(f"낮은 주간 변동성 감지 (weekly_volatility={weekly_volatility}): weekly_risk_coefficient 하향 조정.")

        # 추세 조정: trend와 trend_strength를 모두 고려
        if trend_strength is not None:
            # trend_strength가 양수이면 상승 추세, 음수이면 하락 추세
            dynamic_params["profit_ratio"] *= (1 + trend_strength)
            self.logger.debug(f"추세 강도 반영: trend_strength={trend_strength}, profit_ratio 조정됨.")
        else:
            if trend == "bullish":
                dynamic_params["profit_ratio"] *= 1.05
                self.logger.debug("Bullish 추세 감지: profit_ratio 상향 조정.")
            elif trend == "bearish":
                dynamic_params["profit_ratio"] *= 0.95
                self.logger.debug("Bearish 추세 감지: profit_ratio 하향 조정.")
            else:
                self.logger.debug("중립 추세: profit_ratio 변경 없음.")

        # 거래량(volume)에 따른 risk_per_trade 조정
        volume_threshold = 1000  # 기준 거래량
        if volume is not None:
            if volume < volume_threshold:
                # 예: volume 500이면 risk_per_trade를 500/1000 = 0.5배 적용
                factor = volume / volume_threshold
                dynamic_params["risk_per_trade"] *= factor
                self.logger.debug(f"낮은 거래량 감지 (volume={volume}): risk_per_trade 축소, 계수={factor:.2f}.")
            else:
                # volume이 높으면 소폭 상향
                dynamic_params["risk_per_trade"] *= 1.05
                self.logger.debug(f"높은 거래량 감지 (volume={volume}): risk_per_trade 소폭 상향 조정.")

        self.logger.debug(f"Market data: {market_data}")
        self.logger.debug(f"업데이트된 동적 파라미터: {dynamic_params}")
        self.logger.debug("동적 파라미터 업데이트 완료.")
        return dynamic_params

    def merge_params(self, optimized_params: dict) -> dict:
        """
        최적화 결과로 도출된 파라미터와 기본 파라미터를 병합하여 반환합니다.
        숫자형 값의 경우 기본값과 최적화값의 가중 평균(현재는 50:50)을 사용하고,
        그 외의 타입은 최적화값을 우선 적용합니다.
        
        개선 사항:
          - 민감도 분석 대상 파라미터 목록(self.sensitivity_params)에 해당하는 값만 병합 대상으로 고려합니다.
          - 향후 사용자 정의 가중치를 적용할 수 있도록 확장 가능하도록 구조를 개선합니다.
        """
        merged = self.get_default_params()
        for key in self.sensitivity_params:
            opt_value = optimized_params.get(key, None)
            default_value = merged.get(key)
            if opt_value is not None and isinstance(default_value, (int, float)) and isinstance(opt_value, (int, float)):
                # 단순 평균(50:50) 대신 향후 weight 인자를 추가할 수 있도록 구조 개선
                merged[key] = (default_value + opt_value) / 2
            elif opt_value is not None:
                merged[key] = opt_value
        # 최적화 결과에 민감도 분석 대상이 아닌 항목이 있다면, 사용자 정의 값으로 덮어씁니다.
        for key, opt_value in optimized_params.items():
            if key not in self.sensitivity_params:
                merged[key] = opt_value
        self.logger.debug(f"병합된 동적 파라미터: {merged}")
        return merged
