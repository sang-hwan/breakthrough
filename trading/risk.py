# trading/risk.py
def compute_position_size(
    account_balance: float,
    risk_percentage: float,
    entry_price: float,
    stop_loss: float,
    fee_rate: float = 0.001,
    min_order_size: float = 1e-8
) -> float:
    """
    포지션 사이즈를 계산합니다.
    stop_loss가 None인 경우, 기본적으로 entry_price의 2% 하락값을 사용합니다.
    """
    if stop_loss is None:
        stop_loss = entry_price * 0.98  # 기본값: entry_price의 2% 하락
    price_diff = abs(entry_price - stop_loss)
    max_risk = account_balance * risk_percentage
    fee_amount = entry_price * fee_rate
    loss_per_unit = price_diff + fee_amount
    computed_size = max_risk / loss_per_unit if loss_per_unit > 0 else 0.0
    return computed_size if computed_size >= min_order_size else 0.0

def allocate_position_splits(
    total_size: float,
    splits_count: int = 3,
    allocation_mode: str = 'equal',
    min_order_size: float = 1e-8
) -> list:
    """
    포지션 분할 시 각 분할의 비율을 결정합니다.
    """
    if splits_count < 1:
        raise ValueError("splits_count는 1 이상이어야 합니다.")
    if allocation_mode not in ['equal', 'pyramid_up', 'pyramid_down']:
        raise ValueError("allocation_mode는 'equal', 'pyramid_up', 'pyramid_down' 중 하나여야 합니다.")
    if total_size < min_order_size:
        return [1.0]
    if allocation_mode == 'equal':
        split_amount = 1.0 / splits_count
        return [split_amount] * splits_count
    elif allocation_mode == 'pyramid_up':
        ratio_sum = splits_count * (splits_count + 1) / 2
        return [(i / ratio_sum) for i in range(1, splits_count + 1)]
    elif allocation_mode == 'pyramid_down':
        ratio_sum = splits_count * (splits_count + 1) / 2
        return [(i / ratio_sum) for i in range(splits_count, 0, -1)]

def attempt_scale_in_position(
    position,
    current_price: float,
    scale_in_threshold: float = 0.02,
    slippage_rate: float = 0.0,
    stop_loss: float = None,
    take_profit: float = None,
    entry_time = None,
    trade_type: str = "scale_in",
    base_multiplier: float = 1.0,
    dynamic_volatility: float = 1.0
):
    """
    레짐 기반 전략에 따른 스케일‑인 포지션 진입 시도.
    """
    if position is None or position.is_empty():
        return
    while position.executed_splits < position.total_splits:
        next_split = position.executed_splits
        target_price = position.initial_price * (1.0 + scale_in_threshold * (next_split + 1)) * dynamic_volatility
        if current_price < target_price:
            break
        if next_split < len(position.allocation_plan):
            portion = position.allocation_plan[next_split]
        else:
            break
        chunk_size = position.maximum_size * portion
        executed_price = current_price * (1.0 + slippage_rate)
        position.add_execution(
            entry_price=executed_price,
            size=chunk_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=entry_time,
            trade_type=trade_type
        )
        position.executed_splits += 1

def compute_risk_parameters_by_regime(
    base_params: dict,
    regime: str,
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
    시장 레짐에 따라 리스크 관리 파라미터를 조정합니다.
    레짐 기반 전략에 맞게 각 파라미터를 동적으로 변경합니다.
    """
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
            raise ValueError("횡보장(regime='sideways')에서는 'liquidity' 정보를 반드시 제공해야 합니다.")
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

    return risk_params
