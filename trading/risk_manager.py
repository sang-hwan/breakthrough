# trading/risk_manager.py
from logs.logger_config import setup_logger

logger = setup_logger(__name__)

class RiskManager:
    # 마지막으로 기록된 리스크 파라미터를 저장
    last_computed_risk_params = None
    # risk 파라미터 로그 집계를 위한 호출 카운터
    risk_params_call_count = 0
    RISK_PARAMS_AGG_THRESHOLD = 1000  # 예: 매 1000번째 호출마다 요약 로그 남김

    @staticmethod
    def compute_position_size(available_balance: float, risk_percentage: float, entry_price: float, stop_loss: float, fee_rate: float = 0.001, min_order_size: float = 1e-8, volatility: float = 0.0) -> float:
        if stop_loss is None:
            stop_loss = entry_price * 0.98
        price_diff = abs(entry_price - stop_loss)
        max_risk = available_balance * risk_percentage
        fee_amount = entry_price * fee_rate
        loss_per_unit = price_diff + fee_amount
        computed_size = max_risk / loss_per_unit if loss_per_unit > 0 else 0.0
        if volatility > 0:
            computed_size /= (1 + volatility)
        computed_size = computed_size if computed_size >= min_order_size else 0.0
        logger.debug(f"포지션 사이즈 계산: available_balance={available_balance}, risk_percentage={risk_percentage}, entry_price={entry_price}, stop_loss={stop_loss}, computed_size={computed_size}")
        return computed_size

    @staticmethod
    def allocate_position_splits(total_size: float, splits_count: int = 3, allocation_mode: str = 'equal', min_order_size: float = 1e-8) -> list:
        if splits_count < 1:
            raise ValueError("splits_count는 1 이상이어야 합니다.")
        if allocation_mode not in ['equal', 'pyramid_up', 'pyramid_down']:
            raise ValueError("allocation_mode는 'equal', 'pyramid_up', 'pyramid_down' 중 하나여야 합니다.")
        if total_size < min_order_size:
            return [1.0]
        if allocation_mode == 'equal':
            allocation = [1.0 / splits_count] * splits_count
        elif allocation_mode == 'pyramid_up':
            ratio_sum = splits_count * (splits_count + 1) / 2
            allocation = [i / ratio_sum for i in range(1, splits_count + 1)]
        elif allocation_mode == 'pyramid_down':
            ratio_sum = splits_count * (splits_count + 1) / 2
            allocation = [i / ratio_sum for i in range(splits_count, 0, -1)]
        logger.debug(f"포지션 분할 할당: total_size={total_size}, splits_count={splits_count}, allocation_mode={allocation_mode}, allocation={allocation}")
        return allocation

    @staticmethod
    def attempt_scale_in_position(position, current_price: float, scale_in_threshold: float = 0.02, slippage_rate: float = 0.0, stop_loss: float = None, take_profit: float = None, entry_time=None, trade_type: str = "scale_in", base_multiplier: float = 1.0, dynamic_volatility: float = 1.0):
        if position is None or position.is_empty():
            logger.debug("스케일인 시도: 포지션이 없거나 비어있음")
            return
        scale_in_count = 0
        while position.executed_splits < position.total_splits:
            next_split = position.executed_splits
            target_price = position.initial_price * (1.0 + scale_in_threshold * (next_split + 1)) * dynamic_volatility
            logger.debug(f"스케일인 타겟 가격 계산: next_split={next_split}, target_price={target_price:.2f}, current_price={current_price:.2f}")
            if current_price < target_price:
                break
            if next_split < len(position.allocation_plan):
                portion = position.allocation_plan[next_split]
            else:
                break
            chunk_size = position.maximum_size * portion
            executed_price = current_price * (1.0 + slippage_rate)
            position.add_execution(entry_price=executed_price, size=chunk_size, stop_loss=stop_loss, take_profit=take_profit, entry_time=entry_time, trade_type=trade_type)
            position.executed_splits += 1
            scale_in_count += 1
            logger.debug(f"스케일인 실행: 실행 가격={executed_price:.2f}, 크기={chunk_size:.4f}, 새로운 실행 횟수={position.executed_splits}")

        if scale_in_count > 0:
            if not hasattr(position, 'scale_in_attempt_count'):
                position.scale_in_attempt_count = 0
            position.scale_in_attempt_count += scale_in_count
            logger.debug(f"포지션 {position.position_id} 스케일인 시도 누적: {position.scale_in_attempt_count}")
            if position.scale_in_attempt_count >= 1000:
                logger.info(f"포지션 {position.position_id} 집계: 총 {position.scale_in_attempt_count}회 스케일인 시도 (최근 가격: {current_price})")
                position.scale_in_attempt_count = 0

    @staticmethod
    def is_significant_change(new_params: dict, old_params: dict, threshold: float = 0.10) -> bool:
        for k in new_params:
            if k in old_params:
                old_val = old_params[k]
                new_val = new_params[k]
                if old_val == 0:
                    if new_val != 0:
                        logger.debug(f"파라미터 {k} 변화: old=0, new={new_val} (변화 있음)")
                        return True
                else:
                    rel_diff = abs(new_val - old_val) / abs(old_val)
                    if rel_diff > threshold:
                        logger.debug(f"파라미터 {k} 변화: old={old_val}, new={new_val}, rel_diff={rel_diff:.2f} (임계치 초과)")
                        return True
            else:
                logger.debug(f"새로운 파라미터 {k} 발견됨: {new_params[k]}")
                return True
        return False

    @staticmethod
    def compute_risk_parameters_by_regime(base_params: dict, regime: str, liquidity: str = None, bullish_risk_multiplier: float = 1.1, bullish_atr_multiplier_factor: float = 0.9, bullish_profit_ratio_multiplier: float = 1.1, bearish_risk_multiplier: float = 0.8, bearish_atr_multiplier_factor: float = 1.1, bearish_profit_ratio_multiplier: float = 0.9, high_liquidity_risk_multiplier: float = 1.0, low_liquidity_risk_multiplier: float = 0.8, high_atr_multiplier_factor: float = 1.0, low_atr_multiplier_factor: float = 1.1, high_profit_ratio_multiplier: float = 1.0, low_profit_ratio_multiplier: float = 0.9) -> dict:
        regime = regime.lower()
        risk_params = {}
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
                raise ValueError("횡보장에서는 'liquidity' 정보를 반드시 제공해야 합니다.")
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
            raise ValueError("유효하지 않은 시장 레짐입니다. ('bullish', 'bearish', 'sideways' 중 하나여야 합니다.)")

        current_volatility = base_params.get("current_volatility", None)
        if current_volatility is not None:
            if current_volatility > 0.05:
                risk_params['risk_per_trade'] *= 0.8
                logger.debug(f"현재 변동성이 높음({current_volatility}), risk_per_trade 조정됨")
            else:
                risk_params['risk_per_trade'] *= 1.1
                logger.debug(f"현재 변동성이 낮음({current_volatility}), risk_per_trade 조정됨")

        # 호출 횟수를 집계하여, 일정 횟수마다 요약 로그를 남깁니다.
        RiskManager.risk_params_call_count += 1
        if RiskManager.risk_params_call_count >= RiskManager.RISK_PARAMS_AGG_THRESHOLD:
            if (RiskManager.last_computed_risk_params is None or
                RiskManager.is_significant_change(risk_params, RiskManager.last_computed_risk_params)):
                logger.info(f"[요약] 최종 리스크 파라미터 (최근 {RiskManager.RISK_PARAMS_AGG_THRESHOLD}회 호출): {risk_params}")
            else:
                logger.debug(f"[요약] 최종 리스크 파라미터 (미세 변화, 최근 {RiskManager.RISK_PARAMS_AGG_THRESHOLD}회 호출): {risk_params}")
            RiskManager.risk_params_call_count = 0

        # 개별 호출에서 의미 있는 변화가 있으면 DEBUG로, 미세 변화면 DEBUG로 모두 기록하여 추적
        if (RiskManager.last_computed_risk_params is None or
            RiskManager.is_significant_change(risk_params, RiskManager.last_computed_risk_params)):
            logger.debug(f"최종 리스크 파라미터 (의미 있는 변화): {risk_params}")
        else:
            logger.debug(f"최종 리스크 파라미터 (미세 변화): {risk_params}")
        RiskManager.last_computed_risk_params = risk_params.copy()

        return risk_params
