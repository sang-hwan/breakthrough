# trading/risk_manager.py

from logging.logger_config import setup_logger

# 전역 변수 및 객체 정의
# logger: 이 모듈에서 발생하는 디버그 및 에러 메시지를 기록하는 로깅 객체입니다.
logger = setup_logger(__name__)


class RiskManager:
    def __init__(self):
        # 인스턴스 별로 로거 객체를 초기화합니다.
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
        주어진 위험 관리 파라미터를 바탕으로 적절한 포지션 크기를 계산합니다.
        
        Parameters:
          - available_balance (float): 사용 가능한 총 잔고.
          - risk_percentage (float): 거래에 노출할 위험 비율 (0과 1 사이).
          - entry_price (float): 진입 가격.
          - stop_loss (float): 손절 가격.
          - fee_rate (float): 거래 수수료 비율 (예: 0.001).
          - min_order_size (float): 최소 주문 크기.
          - volatility (float): 현재 변동성 지표.
          - weekly_volatility (float): 주간 변동성 (옵션).
          - weekly_risk_coefficient (float): 주간 변동성에 적용할 위험 계수.
        
        Returns:
          - float: 계산된 포지션 사이즈. 계산 조건에 맞지 않을 경우 0.0 반환.
        
        동작 방식:
          - 사용 가능한 잔고와 위험 비율을 곱해 최대 위험 금액(max_risk)을 산출합니다.
          - 진입 가격과 손절 가격의 차이 및 수수료를 반영해 단위당 손실(loss_per_unit)을 계산합니다.
          - 변동성(현재 및 주간)이 있을 경우 추가 조정하여 최종 포지션 사이즈를 산출합니다.
          - 최종 포지션 사이즈가 최소 주문 크기를 만족하지 않으면 0.0을 반환합니다.
        """
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

        # 진입 가격과 손절 가격 간의 차이를 계산합니다.
        price_diff = abs(entry_price - stop_loss)
        if price_diff == 0:
            self.logger.error("Zero price difference between entry and stop_loss; assigning minimal epsilon to price_diff.", exc_info=True)
            price_diff = entry_price * 1e-4

        # 최대 위험 금액 계산: available_balance * risk_percentage
        max_risk = available_balance * risk_percentage
        fee_amount = entry_price * fee_rate
        loss_per_unit = price_diff + fee_amount

        if loss_per_unit <= 0:
            self.logger.error("Non-positive loss per unit computed.", exc_info=True)
            return 0.0

        # 초기 포지션 사이즈 계산
        computed_size = max_risk / loss_per_unit
        self.logger.debug(f"Initial computed size: {computed_size:.8f} (max_risk={max_risk}, loss_per_unit={loss_per_unit})")

        # 변동성이 존재하면 포지션 사이즈를 추가 조정합니다.
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
        """
        포지션 분할 시, 각 분할의 할당 비율을 계산합니다.
        
        Parameters:
          - total_size (float): 전체 포지션 크기 (참고용; 내부 계산에는 직접 사용하지 않음).
          - splits_count (int): 포지션을 분할할 횟수.
          - allocation_mode (str): 할당 방식 ('equal', 'pyramid_up', 'pyramid_down').
          - min_order_size (float): 최소 주문 크기.
        
        Returns:
          - list: 각 분할에 해당하는 할당 비율의 리스트.
        
        동작 방식:
          - 'equal' 모드인 경우 동일한 비율로 분할합니다.
          - 'pyramid_up' 또는 'pyramid_down' 모드인 경우, 분할 순서에 따라 비율을 점진적으로 증가(또는 감소)시킵니다.
          - 잘못된 입력 값에 대해서는 예외를 발생시킵니다.
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
        포지션에 대해 단계적으로 추가 진입(scale-in)을 시도합니다.
        
        Parameters:
          - position: 현재 포지션 객체로, 포지션의 상태와 할당 계획 정보를 포함해야 합니다.
          - current_price (float): 현재 시장 가격.
          - scale_in_threshold (float): 추가 진입을 위한 가격 변동 임계값 (예: 0.02는 2%).
          - slippage_rate (float): 슬리피지(미끄러짐) 비율.
          - stop_loss (float): 손절 가격 (옵션).
          - take_profit (float): 이익 실현 가격 (옵션).
          - entry_time: 진입 시간 (옵션).
          - trade_type (str): 거래 유형 식별자.
          - dynamic_volatility (float): 동적 변동성 조정 인자.
        
        Returns:
          - None: 포지션 객체에 직접 실행 결과를 추가합니다.
        
        동작 방식:
          - 포지션 객체가 유효하고 비어있지 않은지 확인합니다.
          - 아직 실행되지 않은 분할에 대해 목표 가격을 계산하고, 조건이 충족되면 해당 분할을 실행하여 포지션에 기록합니다.
        """
        try:
            if not position or position.is_empty():
                return

            while position.executed_splits < position.total_splits:
                next_split = position.executed_splits
                # 목표 가격은 초기 가격에 scale_in_threshold와 동적 변동성을 적용하여 계산합니다.
                target_price = position.initial_price * (1 + scale_in_threshold * (next_split + 1)) * dynamic_volatility
                if current_price < target_price:
                    break
                if next_split < len(position.allocation_plan):
                    portion = position.allocation_plan[next_split]
                else:
                    break
                # 각 분할 주문 크기는 전체 포지션 크기에 해당 비율을 곱하여 계산합니다.
                chunk_size = position.maximum_size * portion
                # 슬리피지 적용: 현재 가격에 slippage_rate 만큼 조정합니다.
                executed_price = current_price * (1 + slippage_rate)
                # 포지션 객체에 실행 결과를 추가합니다.
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
        """
        시장 상황(레짐)에 따라 기본 위험 파라미터를 조정합니다.
        
        Parameters:
          - base_params (dict): 기본 위험 파라미터 (예: risk_per_trade, atr_multiplier, profit_ratio).
          - regime: 시장 레짐, 문자열 또는 숫자형 매핑 (예: "bullish", "bearish", "sideways").
          - liquidity (str): 유동성 상태 ("high" 또는 "low"); sideways 레짐에서 필수.
          - bullish_risk_multiplier (float): 상승장에서의 위험 배수.
          - bullish_atr_multiplier_factor (float): 상승장에서의 ATR 조정 인자.
          - bullish_profit_ratio_multiplier (float): 상승장에서의 이익 비율 배수.
          - bearish_risk_multiplier (float): 하락장에서의 위험 배수.
          - bearish_atr_multiplier_factor (float): 하락장에서의 ATR 조정 인자.
          - bearish_profit_ratio_multiplier (float): 하락장에서의 이익 비율 배수.
          - high_liquidity_risk_multiplier (float): 고유동성 조건에서의 위험 배수.
          - low_liquidity_risk_multiplier (float): 저유동성 조건에서의 위험 배수.
          - high_atr_multiplier_factor (float): 고유동성 조건에서의 ATR 조정 인자.
          - low_atr_multiplier_factor (float): 저유동성 조건에서의 ATR 조정 인자.
          - high_profit_ratio_multiplier (float): 고유동성 조건에서의 이익 비율 배수.
          - low_profit_ratio_multiplier (float): 저유동성 조건에서의 이익 비율 배수.
        
        Returns:
          - dict: 조정된 위험 파라미터 딕셔너리.
        
        동작 방식:
          - regime 값을 문자열로 변환(또는 숫자형 매핑)한 후, 각 시장 상황에 따라 기본 파라미터에 특정 배수를 적용합니다.
          - sideways 레짐인 경우 유동성 정보가 필수이며, 추가적으로 현재 변동성이 제공되면 위험 비율을 보정합니다.
        """
        if not isinstance(regime, str):
            try:
                regime = {0.0: "bullish", 1.0: "bearish", 2.0: "sideways"}.get(float(regime))
            except Exception:
                regime = "unknown"
        regime = regime.lower()
        if regime not in ["bullish", "bearish", "sideways"]:
            self.logger.error(f"Invalid market regime: {regime}")
            raise ValueError(f"Invalid market regime: {regime}")
        
        if regime == "sideways":
            if liquidity is None:
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

            # 현재 변동성이 제공된 경우, 위험 비율에 추가 보정을 적용합니다.
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
        """
        주어진 매개변수를 기반으로 trailing stop(후행 손절)을 조정합니다.
        
        Parameters:
          - current_stop (float): 현재 설정된 stop 가격.
          - current_price (float): 현재 시장 가격.
          - highest_price (float): 최근 최고 가격.
          - trailing_percentage (float): trailing stop에 적용할 백분율 (예: 0.02는 2%).
          - volatility (float): 현재 변동성 (옵션).
          - weekly_high (float): 주간 최고 가격 (옵션).
          - weekly_volatility (float): 주간 변동성 (옵션).
        
        Returns:
          - float: 조정된 trailing stop 가격.
        
        동작 방식:
          - 현재 stop, 최고 가격 및 trailing 비율을 이용해 새로운 stop 후보를 계산합니다.
          - 주간 정보가 제공된 경우 이를 고려하여 후보 값을 결정하고,
            조건에 따라 현재 stop 값을 갱신할지 결정합니다.
        """
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
        """
        진입 가격을 기준으로 부분 청산 및 최종 청산 목표 가격을 계산합니다.
        
        Parameters:
          - entry_price (float): 진입 가격.
          - partial_exit_ratio (float): 부분 청산 시 청산 비율.
          - partial_profit_ratio (float): 부분 이익 목표 비율.
          - final_profit_ratio (float): 최종 이익 목표 비율.
          - final_exit_ratio (float): 최종 청산 시 청산 비율.
          - use_weekly_target (bool): 주간 모멘텀을 적용할지 여부.
          - weekly_momentum (float): 주간 모멘텀 값 (옵션).
          - weekly_adjustment_factor (float): 주간 모멘텀에 적용할 조정 인자.
        
        Returns:
          - list of tuples: 각 청산 단계에 대해 (목표 가격, 청산 비율)을 담은 튜플 리스트.
        
        동작 방식:
          - 주간 모멘텀이 사용되는 경우, 이를 반영하여 목표 이익 비율을 조정합니다.
          - 진입 가격에 목표 이익 비율을 적용해 부분 및 최종 청산 가격을 계산합니다.
        """
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
