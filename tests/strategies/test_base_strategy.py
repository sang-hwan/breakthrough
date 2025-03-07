# tests/strategies/test_base_strategy.py
from strategies.base_strategy import BaseStrategy

class DummyStrategy(BaseStrategy):
    def get_signal(self, data, current_time, **kwargs):
        return "dummy_signal"

def test_dummy_strategy():
    strat = DummyStrategy()
    # data와 current_time은 dummy 값 사용
    signal = strat.get_signal({}, "2023-01-01")
    assert signal == "dummy_signal"
