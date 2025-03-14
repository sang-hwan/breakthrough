# config/config_manager.py

# 로그 설정을 위한 모듈 임포트: 로거 설정 함수(setup_logger)를 통해 디버그 및 오류 로그를 기록합니다.
from logs.logger_config import setup_logger

# Pydantic의 BaseModel, Field, ConfigDict, model_validator를 사용하여 데이터 검증 및 모델 생성에 활용
from pydantic import BaseModel, Field, ConfigDict, model_validator

# 특정 문자열 값만 허용하도록 Literal을 사용 (예: 자금 배분 모드 등)
from typing import Literal


# TradingConfig 클래스
# 이 클래스는 거래 전략에 필요한 설정 값들을 Pydantic 모델을 통해 정의 및 검증합니다.
class TradingConfig(BaseModel):
    # 모델 설정: 추가 필드를 허용하여 모델 외의 필드가 입력되어도 예외를 발생시키지 않음
    model_config = ConfigDict(extra="allow")
    
    # 거래 전략에 사용되는 다양한 파라미터를 정의
    sma_period: int = Field(default=200, ge=1)  # 단순 이동평균 계산 기간, 최소값은 1
    atr_period: int = Field(default=14, ge=1)  # ATR 계산 기간, 최소값은 1
    atr_multiplier: float = Field(default=2.07, gt=0, le=10)  # ATR에 곱해지는 계수 (0보다 크고 최대 10)
    dynamic_sl_adjustment: float = Field(default=1.18, gt=0, le=5)  # 동적 손절 조정 계수 (0보다 크고 최대 5)
    profit_ratio: float = Field(default=0.098, gt=0, le=1.0)  # 목표 이익 비율 (0보다 크고 1 이하)
    use_trailing_stop: bool = True  # 트레일링 스탑 사용 여부 (True: 사용, False: 미사용)
    trailing_percent: float = Field(default=0.045, gt=0, le=0.5)  # 트레일링 스탑 기준 퍼센트 (0보다 크고 0.5 이하)
    partial_exit_ratio: float = Field(default=0.5, gt=0, lt=1)  # 부분 청산 비율 (0보다 크고 1 미만)
    partial_profit_ratio: float = Field(default=0.03, gt=0, le=1.0)  # 부분 이익 목표 비율 (0보다 크고 1 이하)
    final_profit_ratio: float = Field(default=0.06, gt=0, le=1.0)  # 최종 이익 목표 비율 (0보다 크고 1 이하)
    risk_per_trade: float = Field(default=0.0162, gt=0, le=1.0)  # 거래당 위험 비율 (0보다 크고 1 이하)
    total_splits: int = Field(default=3, ge=1)  # 분할 거래 횟수, 최소값은 1
    allocation_mode: Literal['equal', 'pyramid_up', 'pyramid_down'] = "equal"  # 자금 배분 모드: 균등(equal), 피라미드 상승(pyramid_up), 피라미드 하락(pyramid_down)
    scale_in_threshold: float = Field(default=0.0153, gt=0, le=1.0)  # 추가 진입 임계값 (스케일 인)
    hmm_confidence_threshold: float = Field(default=0.8, ge=0, le=1)  # HMM(은닉 마르코프 모델) 신뢰도 임계값 (0과 1 사이)
    liquidity_info: Literal['high', 'low'] = "high"  # 유동성 정보: 높음(high) 또는 낮음(low)
    volatility_multiplier: float = Field(default=1.0, gt=0, le=5)  # 변동성 조정 계수 (0보다 크고 최대 5)
    use_candle_pattern: bool = True  # 캔들 패턴 사용 여부
    weekly_breakout_threshold: float = Field(default=0.01, gt=0, le=0.1)  # 주간 돌파 임계값 (0보다 크고 0.1 이하)
    weekly_momentum_threshold: float = Field(default=0.5, gt=0, le=1.0)  # 주간 모멘텀 임계값 (0보다 크고 1 이하)
    weekly_risk_coefficient: float = Field(default=1.0, gt=0, le=5)  # 주간 위험 계수 (0보다 크고 최대 5)
    weight_vol_threshold: float = Field(default=0.05, gt=0, le=1)  # 거래량 가중치 임계값 (0보다 크고 1 이하)
    vol_weight_factor: float = Field(default=0.9, gt=0, le=2)  # 거래량 가중치 계수 (0보다 크고 최대 2)
    liquidity_weight_high: float = Field(default=0.8, gt=0, lt=1)  # 높은 유동성에 적용할 가중치 (0보다 크고 1 미만)
    liquidity_weight_low: float = Field(default=0.6, gt=0, lt=1)  # 낮은 유동성에 적용할 가중치 (0보다 크고 1 미만)

    @model_validator(mode="after")
    def check_profit_ratios(cls, model: "TradingConfig") -> "TradingConfig":
        """
        TradingConfig 객체 생성 후 검증을 통해 부분 이익 목표 비율이 최종 이익 목표 비율보다 작은지 확인합니다.
        
        Parameters:
            model (TradingConfig): 검증 대상 TradingConfig 객체
            
        Returns:
            TradingConfig: 검증된 TradingConfig 객체
            
        Raises:
            ValueError: partial_profit_ratio가 final_profit_ratio보다 크거나 같으면 예외 발생
        """
        if model.partial_profit_ratio >= model.final_profit_ratio:
            raise ValueError("partial_profit_ratio must be less than final_profit_ratio")
        return model


# ConfigManager 클래스: 설정값 관리 및 업데이트를 위한 싱글턴 매니저 클래스
class ConfigManager:
    _instance = None  # 클래스 전역 변수: 단일 인스턴스를 저장 (싱글턴 패턴)

    def __new__(cls, *args, **kwargs):
        """
        싱글턴 패턴 구현: 이미 인스턴스가 존재하면 기존 인스턴스를 반환하고, 그렇지 않으면 새 인스턴스를 생성합니다.
        
        Returns:
            ConfigManager: 단일 인스턴스 객체
        """
        if not cls._instance:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """
        ConfigManager의 초기화 메서드.
        - 기본 거래 설정(TradingConfig)을 생성하고, 동적 파라미터를 초기화합니다.
        - 최적화 과정에서 민감하게 조정해야 할 파라미터 목록을 설정합니다.
        - 로깅 설정을 초기화하여 디버깅 메시지를 기록합니다.
        """
        if hasattr(self, '_initialized') and self._initialized:
            # 이미 초기화된 경우, 재초기화를 방지
            return

        # 기본 거래 설정값 생성 (TradingConfig 인스턴스)
        self.defaults = TradingConfig()
        # 기본 설정을 딕셔너리 형태로 저장 (동적 파라미터 초기 상태)
        self.dynamic_params = self.defaults.model_dump()
        
        # 민감한 파라미터 키 목록: 최적화 병합 시 평균을 내어 조정할 파라미터들
        self.sensitivity_keys = [
            "profit_ratio",
            "atr_multiplier",
            "risk_per_trade",
            "scale_in_threshold",
            "weekly_breakout_threshold",
            "weekly_momentum_threshold"
        ]
        # 로깅 설정 초기화: 현재 모듈 이름을 사용하여 로거 생성
        self.logger = setup_logger(__name__)
        self.logger.debug("ConfigManager initialized with default parameters.")
        self._initialized = True  # 초기화 완료 플래그 설정

    def get_defaults(self) -> dict:
        """
        기본 설정값을 반환하는 함수.
        
        Returns:
            dict: TradingConfig 기본 설정을 딕셔너리 형태로 반환
        """
        return self.defaults.model_dump()

    def get_dynamic_params(self) -> dict:
        """
        현재 동적 파라미터 값을 반환하는 함수.
        
        Returns:
            dict: 현재 저장된 동적 파라미터의 사본을 반환
        """
        return self.dynamic_params.copy()

    def update_with_market_data(self, market_data: dict) -> dict:
        """
        시장 데이터를 기반으로 거래 설정값을 동적으로 업데이트합니다.
        
        Parameters:
            market_data (dict): 시장 데이터를 포함하는 딕셔너리. 예상 키:
                - volatility: 현재 시장 변동성 (float)
                - trend: 시장 추세 (예: "bullish", "bearish", "neutral")
                - trend_strength: 추세 강도 (float, 선택적)
                - volume: 거래량 (float, 선택적)
                - weekly_volatility: 주간 변동성 (float, 선택적)
                - weekly_low: 주간 최저가 (float, 선택적)
                - weekly_high: 주간 최고가 (float, 선택적)
        
        Returns:
            dict: 시장 데이터 반영 후 업데이트된 설정값(동적 파라미터) 딕셔너리
        """
        # 기본 설정값을 딕셔너리로 가져옴
        config_dict = self.get_defaults()

        # 시장 데이터에서 필요한 값들을 추출 (기본값 제공)
        volatility = market_data.get("volatility", 0.0)
        trend = market_data.get("trend", "neutral").lower()
        trend_strength = market_data.get("trend_strength")
        volume = market_data.get("volume")
        weekly_volatility = market_data.get("weekly_volatility")

        # 기준 변동성 값 설정 (예: 0.05)
        base_volatility = 0.05
        if volatility > base_volatility:
            # 변동성이 기준보다 높으면 ATR multiplier를 증가시키고, 변동성 계수를 높게 설정
            factor = 1 + 0.5 * ((volatility - base_volatility) / base_volatility)
            config_dict["atr_multiplier"] *= factor
            config_dict["volatility_multiplier"] = 1.2
        else:
            # 변동성이 기준 이하이면 ATR multiplier를 감소시키고, 변동성 계수를 기본 값으로 설정
            factor = 1 - 0.3 * ((base_volatility - volatility) / base_volatility)
            config_dict["atr_multiplier"] *= factor
            config_dict["volatility_multiplier"] = 1.0

        # 주간 변동성 정보가 있을 경우 주간 위험 계수를 조정
        if weekly_volatility is not None:
            config_dict["weekly_risk_coefficient"] *= 1.2 if weekly_volatility > 0.07 else 0.9

        # 추세 강도가 제공된 경우 profit_ratio를 조정, 그렇지 않으면 추세 문자열에 따라 조정
        if trend_strength is not None:
            config_dict["profit_ratio"] *= (1 + trend_strength)
        else:
            if trend == "bullish":
                config_dict["profit_ratio"] *= 1.05
            elif trend == "bearish":
                config_dict["profit_ratio"] *= 0.95

        # 거래량(volume)이 제공된 경우 위험 비율(risk_per_trade) 조정: 거래량이 임계값 미만이면 비례 조정
        if volume is not None:
            volume_threshold = 1000  # 거래량 임계값 설정
            config_dict["risk_per_trade"] *= (volume / volume_threshold) if volume < volume_threshold else 1.05

        # 주간 최저가와 최고가 정보를 사용하여 스프레드 비율 계산 후, 스프레드가 기준을 초과하면 일부 주간 전략 파라미터 조정
        weekly_low = market_data.get("weekly_low")
        weekly_high = market_data.get("weekly_high")
        if weekly_low is not None and weekly_high is not None and weekly_low > 0:
            spread_ratio = (weekly_high - weekly_low) / weekly_low
            if spread_ratio > 0.05:
                config_dict["weekly_breakout_threshold"] = max(config_dict["weekly_breakout_threshold"] * 0.8, 0.005)
                config_dict["weekly_momentum_threshold"] = min(config_dict["weekly_momentum_threshold"] * 1.05, 0.7)
                self.logger.debug(f"Weekly strategy parameters adjusted due to high spread ratio: {spread_ratio:.2f}")

        # 업데이트된 설정값을 TradingConfig 객체를 사용하여 검증
        try:
            updated_config = TradingConfig(**config_dict)
        except Exception as e:
            self.logger.error("Validation error in updated configuration: " + str(e), exc_info=True)
            raise

        # 동적 파라미터 업데이트 및 로깅
        self.dynamic_params = updated_config.model_dump()
        self.logger.debug(f"Updated config with market data: {self.dynamic_params}")
        return self.dynamic_params

    def merge_optimized(self, optimized: dict) -> dict:
        """
        최적화된 파라미터와 기존 기본 설정값을 병합하여 업데이트된 설정값을 생성합니다.
        
        Parameters:
            optimized (dict): 최적화된 파라미터를 포함하는 딕셔너리
        
        Returns:
            dict: 병합 후 업데이트된 설정값(동적 파라미터) 딕셔너리
        """
        # 기본 설정값을 기준으로 병합 시작
        merged_dict = self.get_defaults()
        # 민감한 파라미터에 대해서는 기본값과 최적화 값의 평균을 계산하여 적용
        for key in self.sensitivity_keys:
            opt_value = optimized.get(key)
            default_value = merged_dict.get(key)
            if opt_value is not None and isinstance(default_value, (int, float)) and isinstance(opt_value, (int, float)):
                merged_dict[key] = (default_value + opt_value) / 2
            elif opt_value is not None:
                merged_dict[key] = opt_value
        # 민감하지 않은 파라미터들은 최적화된 값으로 직접 대체
        for key, opt_value in optimized.items():
            if key not in self.sensitivity_keys:
                merged_dict[key] = opt_value

        # 병합된 설정값을 TradingConfig 객체를 사용하여 검증
        try:
            merged_config = TradingConfig(**merged_dict)
        except Exception as e:
            self.logger.error("Validation error in merged configuration: " + str(e), exc_info=True)
            raise

        self.logger.debug(f"Merged configuration: {merged_config.model_dump()}")
        self.dynamic_params = merged_config.model_dump()
        return self.dynamic_params

    def validate_params(self, params: dict) -> dict:
        """
        입력된 파라미터를 TradingConfig 모델을 통해 검증하고, 유효하지 않은 경우 기본 설정값을 반환합니다.
        
        Parameters:
            params (dict): 검증 대상 파라미터를 포함하는 딕셔너리
        
        Returns:
            dict: 검증된 파라미터 딕셔너리. 검증 실패 시 기본 설정값을 반환
        """
        try:
            # TradingConfig 인스턴스 생성 시도를 통해 파라미터 유효성 검증
            validated = TradingConfig(**params)
            return validated.model_dump()
        except Exception as e:
            self.logger.error("Dynamic parameter validation failed: " + str(e), exc_info=True)
            # 검증 실패 시 기본 설정값 반환
            return self.get_defaults()
