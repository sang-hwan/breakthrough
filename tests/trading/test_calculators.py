# tests/trading/test_calculators.py
# 이 모듈은 거래 실행 및 리스크 관리와 관련된 계산 함수들(ATR, 동적 스탑/테이크, 부분 청산 목표, 후행 스탑 조정)의 정확성을 검증합니다.

import pandas as pd  # DataFrame 조작용
import numpy as np  # 수치 계산용
import pytest  # 테스트 케이스 작성을 위한 패키지
from trading.calculators import (
    calculate_atr,
    calculate_dynamic_stop_and_take,
    calculate_partial_exit_targets,
    adjust_trailing_stop
)  # 테스트할 계산 함수들

def test_calculate_atr():
    """
    calculate_atr 함수가 ATR(평균 진폭)을 올바르게 계산하여 'atr' 컬럼을 추가하는지 테스트합니다.
    
    생성된 데이터프레임에 대해 고가, 저가, 종가 데이터를 사용하여 ATR을 계산합니다.
    """
    data = pd.DataFrame({
        "high": np.linspace(110, 120, 15),  # 고가 데이터
        "low": np.linspace(100, 110, 15),     # 저가 데이터
        "close": np.linspace(105, 115, 15)    # 종가 데이터
    })
    data = calculate_atr(data, period=14)  # ATR 계산 (기간: 14)
    assert "atr" in data.columns  # 'atr' 컬럼 존재 여부 확인

def test_dynamic_stop_and_take():
    """
    calculate_dynamic_stop_and_take 함수가 주어진 진입가, ATR 및 리스크 파라미터를 바탕으로 동적 스탑로스와 테이크프로핏을 올바르게 계산하는지 테스트합니다.
    
    검증 조건: 스탑로스는 진입가보다 낮고, 테이크프로핏은 진입가보다 높아야 합니다.
    """
    entry_price = 100
    atr = 5
    risk_params = {"atr_multiplier": 2.0, "profit_ratio": 0.05, "volatility_multiplier": 1.0}
    stop_loss, take_profit = calculate_dynamic_stop_and_take(entry_price, atr, risk_params)
    assert stop_loss < entry_price
    assert take_profit > entry_price

def test_partial_exit_targets():
    """
    calculate_partial_exit_targets 함수가 부분 청산 목표를 올바르게 계산하여 리스트 형태로 반환하는지 테스트합니다.
    
    반환된 리스트의 길이가 2인지 확인합니다.
    """
    targets = calculate_partial_exit_targets(100, partial_exit_ratio=0.5, partial_profit_ratio=0.03, final_profit_ratio=0.06)
    assert isinstance(targets, list)
    assert len(targets) == 2

def test_dynamic_stop_and_take_invalid_entry_price():
    """
    잘못된 진입가(0)가 입력될 경우, calculate_dynamic_stop_and_take 함수가 ValueError를 발생시키는지 테스트합니다.
    """
    with pytest.raises(ValueError):
        calculate_dynamic_stop_and_take(0, 5, {"atr_multiplier": 2.0, "profit_ratio": 0.05})

def test_dynamic_stop_and_take_invalid_atr():
    """
    ATR 값이 0일 때, calculate_dynamic_stop_and_take 함수가 기본 조정값을 사용하여 스탑로스와 테이크프로핏을 계산하는지 테스트합니다.
    """
    stop_loss, take_profit = calculate_dynamic_stop_and_take(100, 0, {"atr_multiplier": 2.0, "profit_ratio": 0.05})
    assert stop_loss == 98.00, f"Expected stop_loss 98.00, got {stop_loss}"
    assert take_profit == 104.00, f"Expected take_profit 104.00, got {take_profit}"

def test_adjust_trailing_stop_extreme():
    """
    극단적인 변동성 상황에서 adjust_trailing_stop 함수가 올바른 후행 스탑 값을 산출하는지 테스트합니다.
    
    검증 조건: 새로 계산된 스탑 값은 현재 가격보다 낮아야 합니다.
    """
    current_stop = 80
    current_price = 120
    highest_price = 130
    trailing_percentage = 0.05
    extreme_volatility = 10.0
    new_stop = adjust_trailing_stop(current_stop, current_price, highest_price, trailing_percentage, volatility=extreme_volatility)
    assert new_stop < current_price

def test_calculate_partial_exit_targets_invalid():
    """
    잘못된 진입가(0)가 입력될 경우, calculate_partial_exit_targets 함수가 ValueError를 발생시키는지 테스트합니다.
    """
    with pytest.raises(ValueError):
        calculate_partial_exit_targets(0, partial_exit_ratio=0.5, partial_profit_ratio=0.03, final_profit_ratio=0.06)

def test_calculate_dynamic_stop_and_take_extreme_profit_ratio():
    """
    극단적으로 높은 profit_ratio를 사용한 경우, calculate_dynamic_stop_and_take 함수가 올바른 스탑로스와 테이크프로핏을 계산하는지 테스트합니다.
    """
    entry_price = 100
    atr = 5
    stop_loss, take_profit = calculate_dynamic_stop_and_take(entry_price, atr, {"atr_multiplier": 2.0, "profit_ratio": 1.5})
    assert take_profit == pytest.approx(120, rel=1e-3)
    assert stop_loss == pytest.approx(90, rel=1e-3)
