# tests/trading/test_risk_manager.py
# 이 모듈은 RiskManager 클래스의 포지션 사이즈 계산, 포지션 분할 할당, 리스크 파라미터 계산 기능을 검증하는 테스트 케이스들을 포함합니다.

import pytest  # 테스트 프레임워크용
from trading.risk_manager import RiskManager  # 테스트할 RiskManager 클래스

def test_compute_position_size():
    """
    RiskManager의 compute_position_size 메서드가 사용 가능한 자본, 리스크 비율, 진입가, 스탑로스를 기반으로
    올바른 포지션 사이즈(음수가 아닌 값)를 계산하는지 테스트합니다.
    """
    rm = RiskManager()  # RiskManager 인스턴스 생성
    size = rm.compute_position_size(
        available_balance=10000,  # 총 자본
        risk_percentage=0.01,     # 거래당 리스크 1%
        entry_price=100,          # 진입 가격
        stop_loss=90,             # 스탑로스 가격
        fee_rate=0.001            # 거래 수수료율
    )
    assert size >= 0

def test_allocate_position_splits():
    """
    RiskManager의 allocate_position_splits 메서드가 총 포지션을 지정한 분할 수로 균등하게 분할하고,
    합계가 1(100%)이 되는지 테스트합니다.
    """
    rm = RiskManager()  # RiskManager 인스턴스 생성
    allocation = rm.allocate_position_splits(total_size=1.0, splits_count=3, allocation_mode="equal")
    assert len(allocation) == 3  # 분할 수가 3개인지 확인
    assert abs(sum(allocation) - 1.0) < 1e-6  # 분할 합계가 1에 근접하는지 검증

def test_compute_position_size_invalid_entry():
    """
    진입 가격이 0인 경우, compute_position_size 메서드가 올바르게 0.0의 포지션 사이즈를 반환하는지 테스트합니다.
    """
    rm = RiskManager()
    size = rm.compute_position_size(
        available_balance=10000,
        risk_percentage=0.01,
        entry_price=0,  # 잘못된 진입 가격
        stop_loss=90,
        fee_rate=0.001
    )
    assert size == 0.0

def test_compute_position_size_invalid_stop_loss():
    """
    스탑로스 가격이 0인 경우, compute_position_size 메서드가 포지션 사이즈로 0.0을 반환하는지 테스트합니다.
    """
    rm = RiskManager()
    size = rm.compute_position_size(
        available_balance=10000,
        risk_percentage=0.01,
        entry_price=100,
        stop_loss=0,  # 잘못된 스탑로스 가격
        fee_rate=0.001
    )
    assert size == 0.0

def test_compute_position_size_extreme_volatility():
    """
    극단적인 변동성 조건에서 compute_position_size 메서드가 포지션 사이즈를 크게 축소하는지 테스트합니다.
    """
    rm = RiskManager()
    size = rm.compute_position_size(
        available_balance=10000,
        risk_percentage=0.02,
        entry_price=100,
        stop_loss=90,
        fee_rate=0.001,
        volatility=100  # 매우 높은 변동성
    )
    assert size < 0.01

def test_compute_risk_parameters_invalid_regime():
    """
    존재하지 않는 시장 상태(regime)가 입력될 경우, compute_risk_parameters_by_regime 메서드가 ValueError를 발생시키는지 테스트합니다.
    """
    rm = RiskManager()
    with pytest.raises(ValueError):
        rm.compute_risk_parameters_by_regime(
            base_params={"risk_per_trade": 0.01, "atr_multiplier": 2.0, "profit_ratio": 0.05},
            regime="invalid"  # 잘못된 시장 상태
        )

def test_compute_risk_parameters_missing_liquidity():
    """
    'sideways'와 같이 유동성 정보가 필요한 시장 상태에서, 필요한 파라미터가 없을 경우 ValueError가 발생하는지 테스트합니다.
    """
    rm = RiskManager()
    with pytest.raises(ValueError):
        rm.compute_risk_parameters_by_regime(
            base_params={"risk_per_trade": 0.01, "atr_multiplier": 2.0, "profit_ratio": 0.05},
            regime="sideways"  # 유동성 데이터가 필요하지만 제공되지 않음
        )
