# strategies/risk_management.py
# 매수/매도할 물량 크기를 결정하거나 분할매수 로직을 구현하는 등 '위험관리' 관련 함수.

import math

def calculate_position_size(
    account_balance: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss_price: float,
    fee_rate: float = 0.001
) -> float:
    """
    계좌에서 1% 손실만 허용한다면, 그에 맞춰 실제 몇 코인을 살 것인지 계산하는 예시 함수.
    
    계산 과정:
    1) (entry_price - stop_loss_price) = 1코인당 손실
    2) account_balance * risk_per_trade = 전체 허용 손실금
    3) 수수료(fee_rate)까지 고려해 1코인당 총손실을 구하고
    4) 허용 손실금 / 1코인당 손실 => 매수할 수 있는 최대 코인수
    """

    price_diff = abs(entry_price - stop_loss_price)
    max_risk_amount = account_balance * risk_per_trade
    fee_amount = entry_price * fee_rate
    per_unit_loss = price_diff + fee_amount

    if per_unit_loss > 0:
        position_size = max_risk_amount / per_unit_loss
    else:
        position_size = 0.0

    return position_size


def split_position_sizes(
    total_position_size: float,
    split_count: int = 3,
    scale_mode: str = 'equal'
) -> list:
    """
    전체 매수 물량을 여러 번에 걸쳐 나누기 위한 분할 비중을 계산하는 함수.
    
    - scale_mode='equal': 같은 비중씩 나눔
    - scale_mode='pyramid_up': 뒤로 갈수록 더 많이 매수
    - scale_mode='pyramid_down': 앞으로 갈수록 더 많이 매수
    """

    if split_count < 1:
        raise ValueError("split_count는 최소 1 이상이어야 합니다.")
    if scale_mode not in ['equal', 'pyramid_up', 'pyramid_down']:
        raise ValueError("scale_mode는 'equal', 'pyramid_up', 'pyramid_down' 중 하나여야 합니다.")

    if scale_mode == 'equal':
        split_size = total_position_size / split_count
        return [split_size] * split_count

    elif scale_mode == 'pyramid_up':
        # ex) split_count=3 => (1 + 2 + 3) = 6 을 분모로 하여 비중을 계산
        ratio_sum = split_count * (split_count + 1) / 2
        return [(i / ratio_sum) * total_position_size for i in range(1, split_count + 1)]

    elif scale_mode == 'pyramid_down':
        # ex) split_count=3 => (3 + 2 + 1) = 6
        ratio_sum = split_count * (split_count + 1) / 2
        return [(i / ratio_sum) * total_position_size for i in range(split_count, 0, -1)]


def add_position_sizes(
    position,
    current_price: float,
    threshold_percent: float,
    slippage_rate: float,
    stop_loss_price: float,
    take_profit_price: float,
    entry_time
):
    """
    다음 분할 매수가 체결될 조건을 검사하고, 조건 충족 시 sub_position을 추가하는 로직.
    예: threshold_percent=0.02 => 진입가 대비 +2% 오르면 다음 분할 체결
    """

    if position is None or position.is_empty():
        return

    while position.splits_filled < position.num_splits:
        next_split_index = position.splits_filled

        # 목표 가격(needed_price)을 계산. (예: initial_entry_price * (1 + 0.02 * next_split_index))
        needed_price = position.initial_entry_price * (1.0 + threshold_percent * next_split_index)

        # 아직 현재가격이 충분히 오르지 않았다면(needed_price 넘지 않았으면) 분할매수 중단
        if current_price < needed_price:
            break

        # split_plan에 따라 해당 분할의 비중을 가져옴
        if next_split_index < len(position.split_plan):
            portion_rate = position.split_plan[next_split_index]
        else:
            break

        # 실제 매수할 수량
        chunk_size = position.max_position_size * portion_rate

        # 슬리피지를 고려해 약간 높게 체결된다고 가정
        buy_price = current_price * (1.0 + slippage_rate)

        # sub_position에 추가 (실제 체결 내역 기록)
        position.add_sub_position(
            entry_price=buy_price,
            size=chunk_size,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price,
            entry_time=entry_time
        )

        position.splits_filled += 1
