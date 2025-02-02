# trading/risk.py
import math

def compute_position_size(
    account_balance: float,
    risk_percentage: float,
    entry_price: float,
    stop_loss: float,
    fee_rate: float = 0.001
) -> float:
    """
    계좌 자산 대비 지정 위험 비율에 따라 매수할 수 있는 포지션 사이즈 계산.
    
    계산:
      1) 1코인당 손실 = |entry_price - stop_loss|
      2) 허용 손실 금액 = account_balance * risk_percentage
      3) 수수료(fee_rate) 고려: 1코인당 총 손실 = 가격 차이 + (entry_price * fee_rate)
      4) 포지션 사이즈 = 허용 손실 금액 / (1코인당 총 손실)
    """
    price_diff = abs(entry_price - stop_loss)
    max_risk = account_balance * risk_percentage
    fee_amount = entry_price * fee_rate
    loss_per_unit = price_diff + fee_amount

    return max_risk / loss_per_unit if loss_per_unit > 0 else 0.0

def allocate_position_splits(
    total_size: float,
    splits_count: int = 3,
    allocation_mode: str = 'equal'
) -> list:
    """
    전체 포지션 사이즈를 여러 분할 체결하기 위한 비중 리스트 계산.
    
    - allocation_mode:
        'equal'        : 같은 비중
        'pyramid_up'   : 후속 분할일수록 비중 증가
        'pyramid_down' : 초기 분할일수록 비중 증가
    """
    if splits_count < 1:
        raise ValueError("splits_count는 1 이상이어야 합니다.")
    if allocation_mode not in ['equal', 'pyramid_up', 'pyramid_down']:
        raise ValueError("allocation_mode는 'equal', 'pyramid_up', 'pyramid_down' 중 하나여야 합니다.")
    
    if allocation_mode == 'equal':
        split_amount = total_size / splits_count
        return [split_amount] * splits_count
    elif allocation_mode == 'pyramid_up':
        ratio_sum = splits_count * (splits_count + 1) / 2
        return [(i / ratio_sum) * total_size for i in range(1, splits_count + 1)]
    elif allocation_mode == 'pyramid_down':
        ratio_sum = splits_count * (splits_count + 1) / 2
        return [(i / ratio_sum) * total_size for i in range(splits_count, 0, -1)]

def attempt_scale_in_position(
    position,
    current_price: float,
    scale_in_threshold: float = 0.02,
    slippage_rate: float = 0.0,
    stop_loss: float = None,
    take_profit: float = None,
    entry_time = None,
    trade_type: str = "scale_in"  # 기본적으로 "scale_in"으로 설정
):
    """
    분할 매수(스케일 인) 조건 확인 및 실행.
    - scale_in_threshold: 초기 진입가 대비 몇 퍼센트 상승하면 추가 체결
    - slippage_rate: 체결시 발생하는 슬리피지 (가격 상승 비율)
    - trade_type: 추가 체결에 대한 거래 유형 (기본값 "scale_in")
    """
    if position is None or position.is_empty():
        return
    
    # 분할 매수 실행: 아직 체결되지 않은 분할이 남은 경우
    while position.executed_splits < position.total_splits:
        next_split = position.executed_splits
        # 목표 가격 계산: initial_price * (1 + scale_in_threshold * split_index)
        target_price = position.initial_price * (1.0 + scale_in_threshold * next_split)
        if current_price < target_price:
            break  # 아직 추가 체결 조건 미충족
        if next_split < len(position.allocation_plan):
            portion = position.allocation_plan[next_split]
        else:
            break
        
        # 매수 수량 계산: 전체 최대 수량의 해당 비중
        chunk_size = position.maximum_size * portion
        # 슬리피지 적용: 체결가는 current_price * (1 + slippage_rate)
        executed_price = current_price * (1.0 + slippage_rate)
        # 체결 시 trade_type 인수를 명시적으로 전달
        position.add_execution(
            entry_price=executed_price,
            size=chunk_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=entry_time,
            trade_type=trade_type
        )
        position.executed_splits += 1
