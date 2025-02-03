# trading/risk.py
import math

def compute_position_size(
    account_balance: float,
    risk_percentage: float,
    entry_price: float,
    stop_loss: float,
    fee_rate: float = 0.001,
    min_order_size: float = 1e-8
) -> float:
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
    dynamic_volatility: float = 1.0  # 추가: 시장 변동성에 따른 동적 조정
):
    if position is None or position.is_empty():
        return
    # 가격이 충분히 상승하고 동적 변동성 조건 만족 시 scale‑in 시도
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
