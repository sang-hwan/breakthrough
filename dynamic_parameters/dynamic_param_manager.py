# dynamic_parameters/dynamic_param_manager.py
import logging

class DynamicParamManager:
    def __init__(self):
        self.default_params = {
            "lookback_window": 17,
            "volume_factor": 1.38,
            "confirmation_bars": 1,
            "breakout_buffer": 0.00014,
            "retest_threshold": 0.0194,
            "retest_confirmation_bars": 1,
            "sma_period": 200,
            "macd_slow_period": 23,
            "macd_fast_period": 13,
            "macd_signal_period": 10,
            "rsi_period": 16,
            "rsi_threshold": 83.5,
            "bb_period": 19,
            "bb_std_multiplier": 2.05,
            "macd_diff_threshold": -0.91,
            "atr_period": 14,
            "atr_multiplier": 2.07,
            "dynamic_sl_adjustment": 1.18,
            "profit_ratio": 0.098,
            "use_trailing_stop": True,
            "trailing_percent": 0.045,
            "use_partial_take_profit": False,
            "partial_tp_factor": 0.05,
            "final_tp_factor": 0.07,
            "use_trend_exit": True,
            "risk_per_trade": 0.0162,
            "total_splits": 3,
            "allocation_mode": "equal",
            "scale_in_threshold": 0.0153,
            "entry_signal_mode": "AND"
        }

        # 로깅 설정
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 이미 핸들러가 등록되어 있지 않은 경우에만 설정
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            
            # 포맷 설정
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s:%(name)s:%(message)s')
            
            # 콘솔(Stream) 핸들러
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)
            
            # 파일(File) 핸들러
            file_handler = logging.FileHandler('dynamic_param_manager.log')  # 원하는 경로/파일명 지정 가능
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def get_default_params(self):
        return self.default_params.copy()
    
    def update_dynamic_params(self, market_data):
        dynamic_params = self.get_default_params()
        volatility = market_data.get("volatility", 0.0)
        trend_strength = market_data.get("trend_strength", 0.0)
        
        # 예: 변동성이 5% 이상이면 ATR multiplier를 10% 증가, 아니면 10% 감소
        if volatility > 0.05:
            dynamic_params["atr_multiplier"] *= 1.1
        else:
            dynamic_params["atr_multiplier"] *= 0.9
        
        # 트렌드 강도가 낮으면 RSI threshold를 낮추고, 높으면 높임
        if trend_strength < 0.3:
            dynamic_params["rsi_threshold"] -= 5
        else:
            dynamic_params["rsi_threshold"] += 5

        # 현재 적용된 동적 파라미터와 시장 데이터를 로깅
        self.logger.info(f"Market data: {market_data}")
        self.logger.info(f"Updated dynamic parameters: {dynamic_params}")
        
        return dynamic_params
