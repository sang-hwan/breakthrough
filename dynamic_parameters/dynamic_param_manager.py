# dynamic_parameters/dynamic_param_manager.py
from logs.logger_config import setup_logger

class DynamicParamManager:
    def __init__(self):
        # 기본 동적 파라미터: 불필요한 추세 관련 파라미터 제거 및 레짐 기반 파라미터 추가
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
            "liquidity_info": "high"
        }

        self.logger = setup_logger(__name__)
        self.logger.info("DynamicParamManager 초기화 완료 (레짐 기반 전략 적용).")

    def get_default_params(self):
        return self.default_params.copy()
    
    def update_dynamic_params(self, market_data):
        dynamic_params = self.get_default_params()
        volatility = market_data.get("volatility", 0.0)
        
        if volatility > 0.05:
            dynamic_params["atr_multiplier"] *= 1.1
        else:
            dynamic_params["atr_multiplier"] *= 0.9
        
        # 기존 trend_strength 기반 조정 로직은 제거됨 (레짐 기반 전략 적용)
        self.logger.info(f"Market data: {market_data}")
        self.logger.info(f"Updated dynamic parameters: {dynamic_params}")
        
        return dynamic_params
