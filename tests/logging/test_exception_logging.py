# tests/logging/test_exception_logging.py
# 이 파일은 calculate_dynamic_stop_and_take 및 adjust_trailing_stop 함수에서
# 잘못된 입력에 대해 예외가 발생하고 로그에 에러 메시지가 기록되는지 테스트합니다.

import logging
import pytest
from trading.calculators import calculate_dynamic_stop_and_take, adjust_trailing_stop

def test_exception_logging_dynamic_stop_and_take(caplog):
    """
    calculate_dynamic_stop_and_take 함수의 예외 로깅 테스트

    목적:
      - entry_price가 0인 경우 함수가 ValueError를 발생시키고,
        "Invalid entry_price"라는 에러 메시지가 로그에 남는지 확인.

    Parameters:
      caplog: pytest의 캡처된 로그 객체 (로그 내용 확인에 사용)

    Returns:
      없음 (assert 구문으로 예외 및 로그 메시지 검증)
    """
    # ERROR 레벨 이상의 로그를 캡처하도록 설정
    caplog.set_level(logging.ERROR)
    # entry_price가 0인 경우 ValueError가 발생함을 테스트
    with pytest.raises(ValueError):
        calculate_dynamic_stop_and_take(0, 5, {"atr_multiplier": 2.0, "profit_ratio": 0.05})
    # 로그에 'Invalid entry_price' 메시지가 포함되어 있는지 확인
    assert "Invalid entry_price" in caplog.text

def test_exception_logging_adjust_trailing_stop(caplog):
    """
    adjust_trailing_stop 함수의 예외 로깅 테스트

    목적:
      - current_price 또는 highest_price가 0 이하인 경우 함수가 ValueError를 발생시키고,
        관련 에러 메시지가 로그에 기록되는지 확인.

    Parameters:
      caplog: pytest의 캡처된 로그 객체

    Returns:
      없음 (assert 구문으로 예외 발생 및 로그 메시지 검증)
    """
    caplog.set_level(logging.ERROR)
    # 잘못된 가격 입력으로 인해 ValueError 발생 확인
    with pytest.raises(ValueError):
        adjust_trailing_stop(0, -100, -100, 0.05)
    # 로그 메시지에 "Invalid current_price" 또는 "highest_price" 관련 문구가 있는지 확인
    assert "Invalid current_price" in caplog.text or "highest_price" in caplog.text
