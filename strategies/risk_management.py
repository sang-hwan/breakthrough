# strategies/risk_management.py

import math # 수학 함수 사용을 위한 math 모듈

def calculate_position_size(
    account_balance: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss_price: float,
    fee_rate: float = 0.001
) -> float:
    """
    한 번의 매매에서 최대 손실을 제한하도록 포지션 크기를 계산하는 함수입니다.

    매개변수:
    ----------
    - account_balance (float): 계좌 잔고 (예: 10,000 USDT)
    - risk_per_trade (float): 허용 가능한 손실 비율 (예: 0.01 = 1%)
    - entry_price (float): 매수 가격
    - stop_loss_price (float): 손절 가격
    - fee_rate (float): 매수 수수료 비율 (기본값: 0.001 = 0.1%)

    반환값:
    ----------
    - float: 계산된 매수 가능 코인(또는 계약) 수량

    계산 절차:
    1. 코인 1개당 손실 금액 계산:
       `price_diff = abs(entry_price - stop_loss_price)`
    2. 최대 감당 가능 손실 금액 계산:
       `max_risk_amount = account_balance * risk_per_trade`
    3. 수수료 계산:
       `fee_amount = entry_price * fee_rate`
    4. 코인 1개당 총 손실:
       `per_unit_loss = price_diff + fee_amount`
    5. 허용 손실 금액 내에서 매수 가능한 최대 코인 수 계산:
       `position_size = max_risk_amount / per_unit_loss`
    """

    # 코인 1개당 손실 계산 (진입가와 손절가의 차이)
    price_diff = abs(entry_price - stop_loss_price)

    # 감당 가능한 최대 손실 금액
    max_risk_amount = account_balance * risk_per_trade

    # 매수 시 발생할 수수료
    fee_amount = entry_price * fee_rate

    # 코인 1개당 총 손실 계산
    per_unit_loss = price_diff + fee_amount

    # 최대 매수 가능 코인 수 계산
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
    전체 매수 물량을 여러 단계로 나누는 분할 매매 함수입니다.

    매개변수:
    ----------
    - total_position_size (float): 매수할 총 코인 수량
    - split_count (int): 매수 단계를 몇 번으로 나눌지 (기본값: 3)
    - scale_mode (str): 분할 비율 설정
        * 'equal': 균등 분할
        * 'pyramid_up': 뒤로 갈수록 매수 물량 증가
        * 'pyramid_down': 앞으로 갈수록 매수 물량 감소

    반환값:
    ----------
    - list: 각 단계별 매수 물량

    예시:
    ------
    >>> split_position_sizes(9, split_count=3, scale_mode='equal')
    [3.0, 3.0, 3.0]

    >>> split_position_sizes(9, split_count=3, scale_mode='pyramid_up')
    [1.5, 3.0, 4.5]
    """

    # 최소 1회 이상 매수를 나눌 수 있어야 함
    if split_count < 1:
        raise ValueError("split_count는 최소 1 이상이어야 합니다.")

    # 지원하지 않는 모드는 예외 처리
    if scale_mode not in ['equal', 'pyramid_up', 'pyramid_down']:
        raise ValueError("scale_mode는 'equal', 'pyramid_up', 'pyramid_down' 중 하나여야 합니다.")

    if scale_mode == 'equal':
        # 균등 분할
        split_size = total_position_size / split_count
        return [split_size] * split_count

    elif scale_mode == 'pyramid_up':
        # 피라미드 업: 뒤로 갈수록 더 많은 매수 비중
        ratio_sum = split_count * (split_count + 1) / 2  # 합계 계산 (1+2+3...)
        return [(i / ratio_sum) * total_position_size for i in range(1, split_count + 1)]

    elif scale_mode == 'pyramid_down':
        # 피라미드 다운: 앞으로 갈수록 더 많은 매수 비중
        ratio_sum = split_count * (split_count + 1) / 2
        return [(i / ratio_sum) * total_position_size for i in range(split_count, 0, -1)]
