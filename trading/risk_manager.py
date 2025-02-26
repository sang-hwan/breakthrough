# trading/risk_manager.py
from logs.logger_config import setup_logger

logger = setup_logger(__name__)

class RiskManager:
    def __init__(self):
        self.logger = setup_logger(__name__)

    def compute_position_size(
        self,
        available_balance: float,
        risk_percentage: float,
        entry_price: float,
        stop_loss: float,
        fee_rate: float = 0.001,
        min_order_size: float = 1e-8,
        volatility: float = 0.0,
        weekly_volatility: float = None,
        weekly_risk_coefficient: float = 1.0
    ) -> float:
        if available_balance <= 0:
            self.logger.error(f"Available balance is {available_balance}; no funds available for trading.", exc_info=True)
            return 0.0
        if not (0 < risk_percentage <= 1):
            self.logger.error(f"Invalid risk_percentage ({risk_percentage}). Must be between 0 and 1.", exc_info=True)
            return 0.0
        if fee_rate < 0:
            self.logger.error(f"Invalid fee_rate ({fee_rate}). Must be non-negative.", exc_info=True)
            return 0.0
        if volatility < 0 or (weekly_volatility is not None and weekly_volatility < 0):
            self.logger.error("Volatility values must be non-negative.", exc_info=True)
            return 0.0

        if entry_price <= 0 or stop_loss <= 0:
            self.logger.error(f"Invalid entry_price ({entry_price}) or stop_loss ({stop_loss}). Must be positive.", exc_info=True)
            return 0.0

        price_diff = abs(entry_price - stop_loss)
        if price_diff == 0:
            self.logger.error("Zero price difference between entry and stop_loss; assigning minimal epsilon to price_diff.", exc_info=True)
            price_diff = entry_price * 1e-4

        max_risk = available_balance * risk_percentage
        fee_amount = entry_price * fee_rate
        loss_per_unit = price_diff + fee_amount

        if loss_per_unit <= 0:
            self.logger.error("Non-positive loss per unit computed.", exc_info=True)
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

    def allocate_position_splits(
        self,
        total_size: float,
        splits_count: int = 3,
        allocation_mode: str = 'equal',
        min_order_size: float = 1e-8
    ) -> list:
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

    def attempt_scale_in_position(
        self,
        position,
        current_price: float,
        scale_in_threshold: float = 0.02,
        slippage_rate: float = 0.0,
        stop_loss: float = None,
        take_profit: float = None,
        entry_time=None,
        trade_type: str = "scale_in",
        dynamic_volatility: float = 1.0
    ):
        try:
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
                position.add_execution(
                    entry_price=executed_price,
                    size=chunk_size,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    entry_time=entry_time,
                    trade_type=trade_type
                )
                position.executed_splits += 1
                self.logger.debug(f"Scaled in: split {next_split+1}, executed_price={executed_price:.2f}, chunk_size={chunk_size:.8f}")
        except Exception as e:
            self.logger.error("Error in attempt_scale_in_position: " + str(e), exc_info=True)

    def compute_risk_parameters_by_regime(
        self,
        base_params: dict,
        regime,  # 문자열이 아닐 경우 숫자형 매핑 시도
        liquidity: str = None,
        bullish_risk_multiplier: float = 1.1,
        bullish_atr_multiplier_factor: float = 0.9,
        bullish_profit_ratio_multiplier: float = 1.1,
        bearish_risk_multiplier: float = 0.8,
        bearish_atr_multiplier_factor: float = 1.1,
        bearish_profit_ratio_multiplier: float = 0.9,
        high_liquidity_risk_multiplier: float = 1.0,
        low_liquidity_risk_multiplier: float = 0.8,
        high_atr_multiplier_factor: float = 1.0,
        low_atr_multiplier_factor: float = 1.1,
        high_profit_ratio_multiplier: float = 1.0,
        low_profit_ratio_multiplier: float = 0.9
    ) -> dict:
        # regime이 문자열이 아니라면 숫자형 매핑을 시도합니다.
        if not isinstance(regime, str):
            try:
                regime = {0.0: "bullish", 1.0: "bearish", 2.0: "sideways"}.get(float(regime), "unknown")
            except Exception:
                regime = "unknown"
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
                    self.logger.error("Liquidity info required for sideways regime; using default risk parameters.", exc_info=True)
                    return base_params
                liquidity = liquidity.lower()
                if liquidity == "high":
                    risk_params['risk_per_trade'] = base_params['risk_per_trade'] * high_liquidity_risk_multiplier
                    risk_params['atr_multiplier'] = base_params['atr_multiplier'] * high_atr_multiplier_factor
                    risk_params['profit_ratio'] = base_params['profit_ratio'] * high_profit_ratio_multiplier
                else:
                    risk_params['risk_per_trade'] = base_params['risk_per_trade'] * low_liquidity_risk_multiplier
                    risk_params['atr_multiplier'] = base_params['atr_multiplier'] * low_atr_multiplier_factor
                    risk_params['profit_ratio'] = base_params['profit_ratio'] * low_profit_ratio_multiplier
            elif regime == "unknown":
                # 개선: unknown regime일 경우, 보수적인 위험 조정을 적용합니다.
                self.logger.warning("Market regime is unknown; applying conservative risk adjustments.")
                # liquidity 정보가 있다면 sideways 기준으로 조정, 없으면 기본 파라미터에 소폭 감쇄 적용
                if liquidity is not None:
                    liquidity = liquidity.lower()
                    if liquidity == "high":
                        risk_params['risk_per_trade'] = base_params['risk_per_trade'] * 0.95
                        risk_params['atr_multiplier'] = base_params['atr_multiplier']
                        risk_params['profit_ratio'] = base_params['profit_ratio']
                    else:
                        risk_params['risk_per_trade'] = base_params['risk_per_trade'] * 0.90
                        risk_params['atr_multiplier'] = base_params['atr_multiplier']
                        risk_params['profit_ratio'] = base_params['profit_ratio']
                else:
                    risk_params = {
                        'risk_per_trade': base_params['risk_per_trade'] * 0.95,
                        'atr_multiplier': base_params['atr_multiplier'],
                        'profit_ratio': base_params['profit_ratio']
                    }
            else:
                self.logger.error(f"Invalid market regime: {regime}; using default risk parameters.", exc_info=True)
                return base_params

            # 추가: 현재 변동성이 있다면 risk_per_trade에 보정 적용
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
            self.logger.error("Error computing risk parameters: " + str(e), exc_info=True)
            raise

    def adjust_trailing_stop(
        self,
        current_stop: float,
        current_price: float,
        highest_price: float,
        trailing_percentage: float,
        volatility: float = 0.0,
        weekly_high: float = None,
        weekly_volatility: float = None
    ) -> float:
        if current_price <= 0 or highest_price <= 0:
            self.logger.error(f"Invalid current_price ({current_price}) or highest_price ({highest_price}).", exc_info=True)
            raise ValueError("current_price and highest_price must be positive.")
        if trailing_percentage < 0:
            self.logger.error(f"Invalid trailing_percentage ({trailing_percentage}). Must be non-negative.", exc_info=True)
            raise ValueError("trailing_percentage must be non-negative.")
        try:
            if current_stop is None or current_stop <= 0:
                current_stop = highest_price * (1 - trailing_percentage * (1 + volatility))
            new_stop_intraday = highest_price * (1 - trailing_percentage * (1 + volatility))
            if weekly_high is not None:
                w_vol = weekly_volatility if weekly_volatility is not None else 0.0
                new_stop_weekly = weekly_high * (1 - trailing_percentage * (1 + w_vol))
                candidate_stop = max(new_stop_intraday, new_stop_weekly)
            else:
                candidate_stop = new_stop_intraday
            adjusted_stop = candidate_stop if candidate_stop > current_stop and candidate_stop < current_price else current_stop
            self.logger.debug(f"Adjusted trailing stop: {adjusted_stop:.2f} (current_stop={current_stop}, candidate_stop={candidate_stop}, current_price={current_price})")
            return adjusted_stop
        except Exception as e:
            self.logger.error("adjust_trailing_stop error: " + str(e), exc_info=True)
            raise

    def calculate_partial_exit_targets(
        self,
        entry_price: float,
        partial_exit_ratio: float = 0.5,
        partial_profit_ratio: float = 0.03,
        final_profit_ratio: float = 0.06,
        final_exit_ratio: float = 1.0,
        use_weekly_target: bool = False,
        weekly_momentum: float = None,
        weekly_adjustment_factor: float = 0.5
    ):
        if entry_price <= 0:
            self.logger.error(f"Invalid entry_price: {entry_price}; must be positive.", exc_info=True)
            raise ValueError(f"Invalid entry_price: {entry_price}. Must be positive.")
        try:
            if use_weekly_target and weekly_momentum is not None:
                adjusted_partial = partial_profit_ratio + weekly_adjustment_factor * weekly_momentum
                adjusted_final = final_profit_ratio + weekly_adjustment_factor * weekly_momentum
            else:
                adjusted_partial = partial_profit_ratio
                adjusted_final = final_profit_ratio
            partial_target = round(entry_price * (1 + adjusted_partial), 2)
            final_target = round(entry_price * (1 + adjusted_final), 2)
            self.logger.debug(f"Partial targets: partial={partial_target:.2f}, final={final_target:.2f} (entry_price={entry_price}, adjusted_partial={adjusted_partial}, adjusted_final={adjusted_final})")
            return [(partial_target, partial_exit_ratio), (final_target, final_exit_ratio)]
        except Exception as e:
            self.logger.error("calculate_partial_exit_targets error: " + str(e), exc_info=True)
            raise
