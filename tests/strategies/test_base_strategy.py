# tests/strategies/test_base_strategy.py
# 이 파일은 BaseStrategy를 상속받아 DummyStrategy를 구현한 후,
# get_signal 메서드가 올바르게 동작하는지 테스트합니다.
# BaseStrategy는 모든 트레이딩 전략이 구현해야 하는 기본 인터페이스를 제공합니다.

from strategies.base_strategy import BaseStrategy

class DummyStrategy(BaseStrategy):
    """
    DummyStrategy 클래스

    목적:
      - BaseStrategy를 상속받아 get_signal 메서드를 단순히 "dummy_signal"을 반환하도록 구현.
      - 테스트를 위해 기본 전략 인터페이스가 올바르게 상속 및 동작하는지 확인.
    
    Methods:
      get_signal(data, current_time, **kwargs): 항상 "dummy_signal" 반환.
    """
    def get_signal(self, data, current_time, **kwargs):
        # data: 시장 데이터 (예시로 빈 dict 사용)
        # current_time: 현재 시간 (예시로 문자열 사용)
        # kwargs: 추가 인자 (필요에 따라 사용)
        return "dummy_signal"

def test_dummy_strategy():
    """
    DummyStrategy의 get_signal 메서드 테스트

    목적:
      - DummyStrategy 인스턴스 생성 후, get_signal 메서드가 "dummy_signal"을 반환하는지 확인.
    
    Parameters:
      없음

    Returns:
      없음 (assert 구문을 통해 반환된 신호 검증)
    """
    strat = DummyStrategy()
    # 테스트용으로 빈 데이터와 단순 문자열 형태의 current_time 전달
    signal = strat.get_signal({}, "2023-01-01")
    assert signal == "dummy_signal"
