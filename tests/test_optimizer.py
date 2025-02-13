# tests/test_optimizer.py
import pytest
from strategies.optimizer import DynamicParameterOptimizer

class DummyTrial:
    # 간단한 dummy trial를 만들어 objective가 실행되도록 함
    def suggest_float(self, name, low, high):
        return (low + high) / 2
    def suggest_categorical(self, name, choices):
        return choices[0]

def test_optimizer_objective():
    optimizer = DynamicParameterOptimizer(n_trials=1)
    dummy_trial = DummyTrial()
    score = optimizer.objective(dummy_trial)
    assert isinstance(score, float)
