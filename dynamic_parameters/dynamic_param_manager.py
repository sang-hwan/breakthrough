# dynamic_parameters/dynamic_param_manager.py
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

    def get_default_params(self):
        return self.default_params.copy()
    
    def update_dynamic_params(self, market_data):
        dynamic_params = self.get_default_params()
        volatility = market_data.get("volatility", 0.0)
        trend_strength = market_data.get("trend_strength", 0.0)
        
        if volatility > 0.05:
            dynamic_params["atr_multiplier"] *= 1.1
        else:
            dynamic_params["atr_multiplier"] *= 0.9
        
        if trend_strength < 0.3:
            dynamic_params["rsi_threshold"] -= 5
        else:
            dynamic_params["rsi_threshold"] += 5
        
        return dynamic_params
