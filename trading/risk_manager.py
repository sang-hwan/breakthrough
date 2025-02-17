# trading/risk_manager.py
from logs.logger_config import setup_logger

class RiskManager:
    def __init__(self):
        self.logger = setup_logger(__name__)

    def compute_position_size(self, available_balance: float, risk_percentage: float, entry_price: float,
                              stop_loss: float, fee_rate: float = 0.001, min_order_size: float = 1e-8,
                              volatility: float = 0.0, weekly_volatility: float = None, weekly_risk_coefficient: float = 1.0) -> float:
        """
        Computes the position size based on available balance, risk parameters, and market conditions.
        Incorporates volatility adjustments including weekly volatility if available.
        """
        if entry_price <= 0 or stop_loss <= 0:
            self.logger.error(f"Invalid entry_price ({entry_price}) or stop_loss ({stop_loss})")
            return 0.0
        price_diff = abs(entry_price - stop_loss)
        if price_diff == 0:
            self.logger.warning("Zero price difference between entry and stop_loss; assigning minimal epsilon to price_diff.")
            price_diff = entry_price * 1e-4
        max_risk = available_balance * risk_percentage
        fee_amount = entry_price * fee_rate
        loss_per_unit = price_diff + fee_amount
        if loss_per_unit <= 0:
            self.logger.error("Non-positive loss per unit computed.")
            return 0.0
        computed_size = max_risk / loss_per_unit
        self.logger.debug(f"Initial computed size: {computed_size:.8f} (max_risk={max_risk}, loss_per_unit={loss_per_unit})")
        if volatility > 0:
            computed_size /= ((1 + volatility) ** 2)
            self.logger.debug(f"Size adjusted for volatility {volatility} with square factor: {computed_size:.8f}")
        if weekly_volatility is not None:
            computed_size /= (1 + weekly_risk_coefficient * weekly_volatility)
            self.logger.debug(f"Size adjusted for weekly_volatility {weekly_volatility} with coefficient {weekly_risk_coefficient}: {computed_size:.8f}")
        final_size = computed_size if computed_size >= min_order_size else 0.0
        self.logger.debug(f"Final computed position size: {final_size:.8f} (min_order_size={min_order_size})")
        return final_size

    def allocate_position_splits(self, total_size: float, splits_count: int = 3, allocation_mode: str = 'equal', min_order_size: float = 1e-8) -> list:
        """
        Allocates the total position size into splits based on the specified allocation mode.
        """
        if splits_count < 1:
            raise ValueError("splits_count must be at least 1")
        if allocation_mode not in ['equal', 'pyramid_up', 'pyramid_down']:
            raise ValueError("allocation_mode must be 'equal', 'pyramid_up', or 'pyramid_down'")
        if allocation_mode == 'equal':
            return [1.0 / splits_count] * splits_count
        ratio_sum = splits_count * (splits_count + 1) / 2
        if allocation_mode == 'pyramid_up':
            return [i / ratio_sum for i in range(1, splits_count + 1)]
        else:
            return [i / ratio_sum for i in range(splits_count, 0, -1)]

    def attempt_scale_in_position(self, position, current_price: float, scale_in_threshold: float = 0.02, slippage_rate: float = 0.0,
                                  stop_loss: float = None, take_profit: float = None, entry_time=None, trade_type: str = "scale_in",
                                  dynamic_volatility: float = 1.0):
        """
        Attempts to scale in additional positions based on the current market price.
        """
        if not position or position.is_empty():
            return
        while position.executed_splits < position.total_splits:
            next_split = position.executed_splits
            target_price = position.initial_price * (1 + scale_in_threshold * (next_split + 1)) * dynamic_volatility
            if current_price < target_price:
                break
            if next_split < len(position.allocation_plan):
                portion = position.allocation_plan[next_split]
            else:
                break
            chunk_size = position.maximum_size * portion
            executed_price = current_price * (1 + slippage_rate)
            position.add_execution(entry_price=executed_price, size=chunk_size, stop_loss=stop_loss,
                                   take_profit=take_profit, entry_time=entry_time, trade_type=trade_type)
            position.executed_splits += 1

    def compute_risk_parameters_by_regime(self, base_params: dict, regime: str, liquidity: str = None,
                                          bullish_risk_multiplier: float = 1.1, bullish_atr_multiplier_factor: float = 0.9, bullish_profit_ratio_multiplier: float = 1.1,
                                          bearish_risk_multiplier: float = 0.8, bearish_atr_multiplier_factor: float = 1.1, bearish_profit_ratio_multiplier: float = 0.9,
                                          high_liquidity_risk_multiplier: float = 1.0, low_liquidity_risk_multiplier: float = 0.8, high_atr_multiplier_factor: float = 1.0, low_atr_multiplier_factor: float = 1.1,
                                          high_profit_ratio_multiplier: float = 1.0, low_profit_ratio_multiplier: float = 0.9) -> dict:
        """
        Computes risk parameters adjustments based on market regime and liquidity information.
        """
        regime = regime.lower()
        risk_params = {}
        try:
            if regime == "bullish":
                risk_params['risk_per_trade'] = base_params['risk_per_trade'] * bullish_risk_multiplier
                risk_params['atr_multiplier'] = base_params['atr_multiplier'] * bullish_atr_multiplier_factor
                risk_params['profit_ratio'] = base_params['profit_ratio'] * bullish_profit_ratio_multiplier
            elif regime == "bearish":
                risk_params['risk_per_trade'] = base_params['risk_per_trade'] * bearish_risk_multiplier
                risk_params['atr_multiplier'] = base_params['atr_multiplier'] * bearish_atr_multiplier_factor
                risk_params['profit_ratio'] = base_params['profit_ratio'] * bearish_profit_ratio_multiplier
            elif regime == "sideways":
                if liquidity is None:
                    self.logger.error("Liquidity info required for sideways regime")
                    raise ValueError("Liquidity info required for sideways regime")
                liquidity = liquidity.lower()
                if liquidity == "high":
                    risk_params['risk_per_trade'] = base_params['risk_per_trade'] * high_liquidity_risk_multiplier
                    risk_params['atr_multiplier'] = base_params['atr_multiplier'] * high_atr_multiplier_factor
                    risk_params['profit_ratio'] = base_params['profit_ratio'] * high_profit_ratio_multiplier
                else:
                    risk_params['risk_per_trade'] = base_params['risk_per_trade'] * low_liquidity_risk_multiplier
                    risk_params['atr_multiplier'] = base_params['atr_multiplier'] * low_atr_multiplier_factor
                    risk_params['profit_ratio'] = base_params['profit_ratio'] * low_profit_ratio_multiplier
            else:
                self.logger.error(f"Invalid market regime: {regime}")
                raise ValueError("Invalid market regime. Must be 'bullish', 'bearish', or 'sideways'.")
            current_volatility = base_params.get("current_volatility", None)
            if current_volatility is not None:
                if current_volatility > 0.05:
                    risk_params['risk_per_trade'] *= 0.8
                    self.logger.debug(f"Adjusted risk_per_trade for high volatility {current_volatility}")
                else:
                    risk_params['risk_per_trade'] *= 1.1
                    self.logger.debug(f"Adjusted risk_per_trade for low volatility {current_volatility}")
            self.logger.debug(f"Computed risk parameters: {risk_params}")
            return risk_params
        except Exception as e:
            self.logger.error(f"Error computing risk parameters: {e}", exc_info=True)
            raise

    def adjust_trailing_stop(self, current_stop: float, current_price: float, highest_price: float, trailing_percentage: float,
                               volatility: float = 0.0, weekly_high: float = None, weekly_volatility: float = None) -> float:
        """
        Adjusts the trailing stop based on market conditions, considering both intraday and weekly data.
        """
        if current_stop is None:
            current_stop = highest_price * (1 - trailing_percentage * (1 + volatility))
        new_stop_intraday = highest_price * (1 - trailing_percentage * (1 + volatility))
        if weekly_high is not None:
            w_vol = weekly_volatility if weekly_volatility is not None else 0.0
            new_stop_weekly = weekly_high * (1 - trailing_percentage * (1 + w_vol))
            candidate_stop = max(new_stop_intraday, new_stop_weekly)
        else:
            candidate_stop = new_stop_intraday
        return candidate_stop if candidate_stop > current_stop and candidate_stop < current_price else current_stop

    def calculate_partial_exit_targets(self, entry_price: float, partial_exit_ratio: float = 0.5,
                                         partial_profit_ratio: float = 0.03, final_profit_ratio: float = 0.06,
                                         final_exit_ratio: float = 1.0, use_weekly_target: bool = False,
                                         weekly_momentum: float = None, weekly_adjustment_factor: float = 0.5):
        """
        Calculates partial and final exit targets based on entry price and profit ratios,
        with optional adjustments using weekly momentum.
        """
        if use_weekly_target and weekly_momentum is not None:
            adjusted_partial = partial_profit_ratio + weekly_adjustment_factor * weekly_momentum
            adjusted_final = final_profit_ratio + weekly_adjustment_factor * weekly_momentum
        else:
            adjusted_partial = partial_profit_ratio
            adjusted_final = final_profit_ratio
        partial_target = entry_price * (1 + adjusted_partial)
        final_target = entry_price * (1 + adjusted_final)
        return [(partial_target, partial_exit_ratio), (final_target, final_exit_ratio)]
