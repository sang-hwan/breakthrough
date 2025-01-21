# strategies/risk_management.py

import math

def calculate_position_size(
    account_balance: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss_price: float,
    fee_rate: float = 0.001
) -> float:
    """
    --------------------------------------------------------------------------------
    1) 돌파매매 전략에서 강조하는 '1회 손실 최소화' 개념을 반영한 포지션 크기 계산 함수
    --------------------------------------------------------------------------------
    - '돌파매매 전략' 책(저자 systrader79 & 김대현)에서 가장 중요한 리스크 관리 개념:
        "한 번의 트레이드에서 계좌의 (n)% 이상 잃지 말자."
      예) 계좌 10,000 USDT, n=1% -> 이번 매매로 최대로 잃을 수 있는 돈은 100 USDT
    - 손절 가격(stop_loss_price)이 정해진 상태에서,
      (진입가(entry_price) ~ 손절가(stop_loss_price))까지의 가격 차이를 이용해
      1코인(또는 1계약)당 '최대 예상 손실액'을 구한 뒤,
      '계좌에서 허용하는 총 손실 한도'와 비교하여 구매 가능한 최대 코인 수를 구합니다.

    파라미터 설명
    -------------
    1) account_balance : float
       - 계좌에 있는 총 금액 (예: 10,000 USDT)
    2) risk_per_trade : float
       - 한 번의 매매로 잃어도 괜찮다고 생각하는 비율(계좌 대비)
       - 예) 0.01 = 1%, 0.02 = 2%
    3) entry_price : float
       - 내가 매수(진입)하려는 가격 (돌파 지점 등)
    4) stop_loss_price : float
       - 손절가격 (이 가격까지 떨어지면 손실을 확정하고 청산)
    5) fee_rate : float
       - 매수 시 발생하는 수수료 비율 (기본 0.1% = 0.001)
       - 실제 매도 시 수수료, 슬리피지 등은 별도 고려 필요

    반환값
    -------
    float
        - '한 번의 매매에서 계좌 (n)% 손실 한도'를 초과하지 않는 범위 내에서
          매수(혹은 매수 계약 진입)할 수 있는 코인(또는 계약) 수

    작동 흐름 (간단히)
    ------------------
    1) price_diff = |entry_price - stop_loss_price|
       => 1코인당 가격 하락 시 발생할 수 있는 손실액(최악의 경우)
    2) max_risk_amount = account_balance * risk_per_trade
       => 이번 트레이드로 감수할 수 있는 손실 총액 (ex. 10,000 * 0.01 = 100 USDT)
    3) fee_amount = entry_price * fee_rate
       => 매수 시 수수료 (ex. 100 USDT * 0.001 = 0.1 USDT)
    4) per_unit_loss = price_diff + fee_amount
       => 코인 1개당 발생 가능한 최대 손실액
    5) position_size = max_risk_amount / per_unit_loss
       => 허용손실 / (1코인당 손실) = 매수 가능 최대 코인 수

    주의사항
    --------
    - (entry_price - stop_loss_price)가 0이면, 계산에 문제가 생길 수 있으므로 체크 필요.
    - 실제 매매에서는 매도 수수료, 슬리피지(체결시 가격차), 펀딩피(선물) 등을 함께 고려하면 더 정확합니다.
    - 레버리지를 사용하는 경우 마진(증거금)과 청산가격(유지증거금) 계산 등 추가 로직이 필요합니다.
    """
    # 1. 코인 1개당 (진입~손절) 구간에서 발생 가능한 최대 손실금액 계산
    price_diff = abs(entry_price - stop_loss_price)

    # 2. 계좌에서 이번 매매에 감당할 수 있는 손실 한도(계좌잔고 × 리스크 비율)
    max_risk_amount = account_balance * risk_per_trade

    # 3. 매수 시 발생할 수수료 (매도 시 수수료는 별도)
    fee_amount = entry_price * fee_rate

    # 4. 코인 1개당 실제 손실: (진입-손절) + 매수 수수료
    per_unit_loss = price_diff + fee_amount

    # 5. 최대 손실 한도를 넘지 않는 선에서 매수할 수 있는 코인 수(또는 계약 수)
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
    ------------------------------------------------------------------------------------------------
    2) 돌파매매 전략에서 자주 활용되는 '피라미딩(분할매매)' 기법을 예시로 구현한 함수
    ------------------------------------------------------------------------------------------------
    - "돌파 성공 후 추세가 이어질 때, 매수 물량을 추가"하는 기법을 '피라미딩'이라고 합니다.
      예) 돌파 매매 후 상승 추세가 유효하면 2차 매수를 더 넣어 수익을 극대화.
    - 여기서는 '전체 매수할 물량'이 정해져 있다고 할 때, 이를 몇 번에 나누어 살지,
      그리고 그 분할 비율을 어떻게 조정할지(균등/피라미드 업/피라미드 다운)를 간단히 보여줍니다.

    파라미터 설명
    -------------
    1) total_position_size : float
       - 최종적으로 매수하고자 하는 전체 코인 수(또는 계약 수)
         (예: calculate_position_size()로부터 받은 결과)
    2) split_count : int, optional
       - 몇 번에 나누어 매수할지 (기본값 3번)
    3) scale_mode : str, optional
       - 'equal'       : 균등 분할 (예: 총 9개, 3번 -> 각 3개씩)
       - 'pyramid_up'   : 뒤로 갈수록 더 많은 비중으로 매수 (1 : 2 : 3)
                          => 추세가 확실해질수록 매수량을 늘려 수익 극대화
       - 'pyramid_down' : 반대로 앞에서 많이 매수하고, 뒤로 갈수록 줄임 (3 : 2 : 1)
                          => 돌파 직후 초기포지션을 크게 잡는 전략과 유사

    반환값
    -------
    list
        각 분할 단계(예: 3단계)에 할당될 매수 물량(코인 수) 목록

    예시
    ----
    >>> split_position_sizes(9, split_count=3, scale_mode='equal')
    [3.0, 3.0, 3.0]

    >>> split_position_sizes(9, split_count=3, scale_mode='pyramid_up')
    [1.5, 3.0, 4.5]   # 합계=9

    관련 개념 (돌파매매 전략과의 연결)
    --------------------------------
    - 책에서는 '돌파 지점'을 여러 구간으로 나눠, 돌파 후 일정 % 더 오르면 2차 매수,
      또 일정 % 더 오르면 3차 매수… 이런 식으로 '피라미딩'을 제안합니다.
    - 본 함수는 '몇 번에 나누고', '어떤 비율로 나눌지'만 계산하는 예시입니다.
      실제로 "언제 2차 매수를 실행하느냐?"(가격 조건 등)는 별도 로직 필요.
    """
    # 1) 매수 횟수(split_count)는 1 이상이어야 한다.
    if split_count < 1:
        raise ValueError("split_count는 최소 1 이상의 정수여야 합니다.")

    # 2) 지원하는 모드(세 가지) 확인
    if scale_mode not in ['equal', 'pyramid_up', 'pyramid_down']:
        raise ValueError("scale_mode는 'equal', 'pyramid_up', 'pyramid_down' 중 하나를 사용하세요.")

    # 3) 분할매매 계산 로직
    if scale_mode == 'equal':
        # (A) 균등 분할
        # 예: 총 9개를 3회 -> 각 3개씩
        split_size = total_position_size / split_count
        return [split_size] * split_count

    elif scale_mode == 'pyramid_up':
        # (B) 피라미드 업
        # 예: split_count=3 이면 비율은 (1 : 2 : 3) = 합계 6
        # => 첫 번째 분할은 총량의 1/6, 두 번째 분할은 2/6, 세 번째 분할은 3/6
        ratio_sum = split_count * (split_count + 1) / 2  # n(n+1)/2 공식
        sizes = []
        for i in range(1, split_count + 1):
            portion = (i / ratio_sum) * total_position_size
            sizes.append(portion)
        return sizes

    elif scale_mode == 'pyramid_down':
        # (C) 피라미드 다운
        # 예: split_count=3 이면 비율은 (3 : 2 : 1) = 합계 6
        # => 첫 번째 분할은 총량의 3/6, 두 번째 2/6, 세 번째 1/6
        ratio_sum = split_count * (split_count + 1) / 2
        sizes = []
        # range(3, 0, -1) -> 3, 2, 1
        for i in range(split_count, 0, -1):
            portion = (i / ratio_sum) * total_position_size
            sizes.append(portion)
        return sizes
