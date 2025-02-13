# tests/test_optimizer.py
from strategies.optimizer import DynamicParameterOptimizer

def test_optimizer_returns_trial():
    optimizer = DynamicParameterOptimizer(n_trials=2)
    best_trial = optimizer.optimize()
    assert best_trial is not None
    assert isinstance(best_trial.params, dict)
