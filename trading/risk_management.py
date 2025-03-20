# trading/risk_management.py
from logs.log_config import setup_logger

logger = setup_logger(__name__)

class RiskManager:
    """
    리스크 관리 기능을 제공합니다.
    """

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
        """
        포지션 크기를 계산합니다.

        Parameters:
            available_balance (float): 사용 가능한 잔고.
            risk_percentage (float): 노출할 위험 비율 (0~1).
            entry_price (float): 진입 가격.
            stop_loss (float): 손절 가격.
            fee_rate (float): 수수료율.
            min_order_size (float): 최소 주문 크기.
            volatility (float): 현재 변동성.
            weekly_volatility (float): 주간 변동성 (옵션).
            weekly_risk_coefficient (float): 주간 변동성 조정 인자.
        Returns:
            float: 계산된 포지션 사이즈.
        """
        if available_balance <= 0:
            self.logger.error(f"Available balance is {available_balance}; no funds available.", exc_info=True)
            return 0.0
        if not (0 < risk_percentage <= 1):
            self.logger.error(f"Invalid risk_percentage ({risk_percentage}).", exc_info=True)
            return 0.0
        if fee_rate < 0:
            self.logger.error(f"Invalid fee_rate ({fee_rate}).", exc_info=True)
            return 0.0
        if volatility < 0 or (weekly_volatility is not None and weekly_volatility < 0):
            self.logger.error("Volatility must be non-negative.", exc_info=True)
            return 0.0
        if entry_price <= 0 or stop_loss <= 0:
            self.logger.error("Entry price and stop loss must be positive.", exc_info=True)
            return 0.0

        price_diff = abs(entry_price - stop_loss)
        if price_diff == 0:
            self.logger.error("Zero price difference between entry and stop_loss.", exc_info=True)
            price_diff = entry_price * 1e-4

        max_risk = available_balance * risk_percentage
        fee_amount = entry_price * fee_rate
        loss_per_unit = price_diff + fee_amount

        if loss_per_unit <= 0:
            self.logger.error("Non-positive loss per unit computed.", exc_info=True)
            return 0.0

        computed_size = max_risk / loss_per_unit
        self.logger.debug(f"Initial computed size: {computed_size:.8f}")

        if volatility > 0:
            computed_size /= ((1 + volatility) ** 2)
            self.logger.debug(f"Size adjusted for volatility: {computed_size:.8f}")

        if weekly_volatility is not None:
            computed_size /= (1 + weekly_risk_coefficient * weekly_volatility)
            self.logger.debug(f"Size adjusted for weekly_volatility: {computed_size:.8f}")

        final_size = computed_size if computed_size >= min_order_size else 0.0
        self.logger.debug(f"Final computed position size: {final_size:.8f}")
        return final_size

    def allocate_position_splits(
        self,
        total_size: float,
        splits_count: int = 3,
        allocation_mode: str = 'equal',
        min_order_size: float = 1e-8
    ) -> list:
        """
        각 분할의 할당 비율을 계산합니다.

        Parameters:
            total_size (float): 전체 포지션 크기.
            splits_count (int): 분할 횟수.
            allocation_mode (str): 'equal', 'pyramid_up', 'pyramid_down'.
            min_order_size (float): 최소 주문 크기.
        Returns:
            list: 각 분할의 할당 비율 리스트.
        """
        if splits_count < 1:
            raise ValueError("splits_count must be at least 1")
        if allocation_mode not in ['equal', 'pyramid_up', 'pyramid_down']:
            raise ValueError("Invalid allocation_mode.")
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
        """
        포지션에 단계적으로 추가 진입(scale-in)을 시도합니다.

        Parameters:
            position: 포지션 객체.
            current_price (float): 현재 가격.
            scale_in_threshold (float): 추가 진입 임계값.
            slippage_rate (float): 슬리피지 비율.
            stop_loss (float): (옵션) 손절 가격.
            take_profit (float): (옵션) 익절 가격.
            entry_time: (옵션) 진입 시각.
            trade_type (str): 거래 유형.
            dynamic_volatility (float): 동적 변동성 인자.
        """
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
        regime,
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
        """
        시장 레짐에 따라 기본 위험 파라미터를 조정합니다.

        Parameters:
            base_params (dict): 기본 위험 파라미터.
            regime: 시장 레짐 ("bullish", "bearish", "sideways") 또는 숫자형 매핑.
            liquidity (str): "high" 또는 "low" (sideways인 경우 필수).
        Returns:
            dict: 조정된 위험 파라미터.
        """
        if not isinstance(regime, str):
            try:
                regime = {0.0: "bullish", 1.0: "bearish", 2.0: "sideways"}.get(float(regime), "unknown")
            except Exception:
                regime = "unknown"
        regime = regime.lower()
        if regime not in ["bullish", "bearish", "sideways"]:
            self.logger.error(f"Invalid market regime: {regime}")
            raise ValueError(f"Invalid market regime: {regime}")

        if regime == "sideways" and liquidity is None:
            self.logger.error("Liquidity info required for sideways regime")
            raise ValueError("Liquidity info required for sideways regime")

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
                if liquidity.lower() == "high":
                    risk_params['risk_per_trade'] = base_params['risk_per_trade'] * high_liquidity_risk_multiplier
                    risk_params['atr_multiplier'] = base_params['atr_multiplier'] * high_atr_multiplier_factor
                    risk_params['profit_ratio'] = base_params['profit_ratio'] * high_profit_ratio_multiplier
                else:
                    risk_params['risk_per_trade'] = base_params['risk_per_trade'] * low_liquidity_risk_multiplier
                    risk_params['atr_multiplier'] = base_params['atr_multiplier'] * low_atr_multiplier_factor
                    risk_params['profit_ratio'] = base_params['profit_ratio'] * low_profit_ratio_multiplier
            elif regime == "unknown":
                self.logger.warning("Market regime is unknown; applying conservative risk adjustments.")
                if liquidity is not None:
                    if liquidity.lower() == "high":
                        risk_params['risk_per_trade'] = base_params['risk_per_trade'] * 0.95
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
            if base_params.get("current_volatility") is not None:
                current_volatility = base_params["current_volatility"]
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
        """
        후행 손절(trailing stop)을 조정합니다.

        Parameters:
            current_stop (float): 현재 손절 가격.
            current_price (float): 현재 가격.
            highest_price (float): 최고 가격.
            trailing_percentage (float): 백분율 임계값.
            volatility (float): 변동성 (옵션).
            weekly_high (float): 주간 최고 가격 (옵션).
            weekly_volatility (float): 주간 변동성 (옵션).
        Returns:
            float: 조정된 trailing stop 가격.
        """
        if current_price <= 0 or highest_price <= 0:
            self.logger.error("Invalid current_price or highest_price.", exc_info=True)
            raise ValueError("current_price and highest_price must be positive.")
        if trailing_percentage < 0:
            self.logger.error("Invalid trailing_percentage.", exc_info=True)
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
            self.logger.debug(f"Adjusted trailing stop: {adjusted_stop:.2f}")
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
        """
        부분 및 최종 청산 목표 가격들을 계산합니다.

        Parameters:
            entry_price (float): 진입 가격.
            partial_exit_ratio (float): 부분 청산 비율.
            partial_profit_ratio (float): 부분 이익 목표.
            final_profit_ratio (float): 최종 이익 목표.
            final_exit_ratio (float): 최종 청산 비율.
            use_weekly_target (bool): 주간 모멘텀 적용 여부.
            weekly_momentum (float): 주간 모멘텀 값 (옵션).
            weekly_adjustment_factor (float): 주간 조정 인자.
        Returns:
            list of tuples: [(목표가격, 청산비율), ...]
        """
        if entry_price <= 0:
            self.logger.error("Invalid entry_price.", exc_info=True)
            raise ValueError("Entry price must be positive.")
        try:
            if use_weekly_target and weekly_momentum is not None:
                adjusted_partial = partial_profit_ratio + weekly_adjustment_factor * weekly_momentum
                adjusted_final = final_profit_ratio + weekly_adjustment_factor * weekly_momentum
            else:
                adjusted_partial = partial_profit_ratio
                adjusted_final = final_profit_ratio
            partial_target = round(entry_price * (1 + adjusted_partial), 2)
            final_target = round(entry_price * (1 + adjusted_final), 2)
            self.logger.debug(f"Partial targets computed: partial={partial_target}, final={final_target}")
            return [(partial_target, partial_exit_ratio), (final_target, final_exit_ratio)]
        except Exception as e:
            self.logger.error("calculate_partial_exit_targets error: " + str(e), exc_info=True)
            raise
