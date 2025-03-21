# tests/unit_tests/optimization/test_optimization_unit.py
import optuna
import pytest

# 최적화 모듈에서 정의된 클래스 임포트
from optimization.market_optimize import MarketParameterOptimizer
from optimization.signal_optimize import SignalParameterOptimizer
from optimization.trade_optimize import DynamicParameterOptimizer


def test_market_optimizer_objective():
    """
    MarketParameterOptimizer의 objective 함수가 올바른 float 값을 반환하는지 확인합니다.
    """
    optimizer = MarketParameterOptimizer(n_trials=1)
    # FixedTrial을 사용하여 dummy 파라미터를 전달
    fixed_params = {
        "ma_period": 20,
        "bollinger_std": 2.0
    }
    trial = optuna.trial.FixedTrial(fixed_params)
    score = optimizer.objective(trial)
    assert isinstance(score, float)
    # 예외 발생 시 큰 페널티 값(1e6)보다 낮은 값이어야 함
    assert score < 1e6


def test_market_optimizer_optimize():
    """
    MarketParameterOptimizer의 optimize 메서드가 최적의 파라미터 trial을 반환하는지 확인합니다.
    """
    optimizer = MarketParameterOptimizer(n_trials=2)
    best_trial = optimizer.optimize()
    assert best_trial is not None
    assert isinstance(best_trial.params, dict)
    # 예상 파라미터 키가 포함되어 있는지 확인
    assert "ma_period" in best_trial.params
    assert "bollinger_std" in best_trial.params


def test_signal_optimizer_objective():
    """
    SignalParameterOptimizer의 objective 함수가 올바른 float 값을 반환하는지 확인합니다.
    """
    optimizer = SignalParameterOptimizer(n_trials=1)
    fixed_params = {
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26
    }
    trial = optuna.trial.FixedTrial(fixed_params)
    score = optimizer.objective(trial)
    assert isinstance(score, float)
    assert score < 1e6


def test_signal_optimizer_optimize():
    """
    SignalParameterOptimizer의 optimize 메서드가 최적의 파라미터 trial을 반환하는지 확인합니다.
    """
    optimizer = SignalParameterOptimizer(n_trials=2)
    best_trial = optimizer.optimize()
    assert best_trial is not None
    assert isinstance(best_trial.params, dict)
    for key in ["rsi_period", "macd_fast", "macd_slow"]:
        assert key in best_trial.params


def test_trade_optimizer_objective():
    """
    DynamicParameterOptimizer의 objective 함수가 올바른 float 값을 반환하는지 확인합니다.
    """
    optimizer = DynamicParameterOptimizer(n_trials=1, assets=["BTC/USDT"])
    fixed_params = {
        "hmm_confidence_threshold": 0.8,
        "liquidity_info": "high",
        "atr_multiplier": 2.0,
        "profit_ratio": 0.1,
        "risk_per_trade": 0.01,
        "scale_in_threshold": 0.02,
        "partial_exit_ratio": 0.5,
        "partial_profit_ratio": 0.03,
        "final_profit_ratio": 0.07,
        "weekly_breakout_threshold": 0.01,
        "weekly_momentum_threshold": 0.5,
        "risk_reward_ratio": 2.0
    }
    trial = optuna.trial.FixedTrial(fixed_params)
    score = optimizer.objective(trial)
    assert isinstance(score, float)
    assert score < 1e6


def test_trade_optimizer_optimize():
    """
    DynamicParameterOptimizer의 optimize 메서드가 최적의 파라미터 trial을 반환하는지 확인합니다.
    """
    optimizer = DynamicParameterOptimizer(n_trials=2, assets=["BTC/USDT"])
    best_trial = optimizer.optimize()
    assert best_trial is not None
    assert isinstance(best_trial.params, dict)
    expected_keys = [
        "hmm_confidence_threshold", "liquidity_info", "atr_multiplier", "profit_ratio",
        "risk_per_trade", "scale_in_threshold", "partial_exit_ratio", "partial_profit_ratio",
        "final_profit_ratio", "weekly_breakout_threshold", "weekly_momentum_threshold",
        "risk_reward_ratio"
    ]
    for key in expected_keys:
        assert key in best_trial.params
