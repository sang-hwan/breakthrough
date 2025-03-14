# tests/optimizer/test_optimizer.py
# 이 파일은 DynamicParameterOptimizer를 사용하여 최적의 파라미터 탐색 기능을 테스트합니다.
# 최적화 과정에서 반환되는 trial 객체가 올바른 형식(dict 형태의 params 포함)인지 검증합니다.

from strategies.optimizer import DynamicParameterOptimizer

def test_optimizer_returns_trial():
    """
    DynamicParameterOptimizer의 최적화 결과(trial) 반환 테스트

    목적:
      - 주어진 n_trials(여기서는 2회)로 최적화 실행 시, 반환되는 최적 trial 객체가
        None이 아니며, trial 객체 내부에 params가 dict 형태로 존재하는지 확인.
    
    Parameters:
      없음

    Returns:
      없음 (assert 구문으로 반환 객체의 유효성 검증)
    """
    # 최적화 인스턴스 생성: 2번의 trial 실행
    optimizer = DynamicParameterOptimizer(n_trials=2)
    best_trial = optimizer.optimize()
    # 반환된 trial 객체가 None이 아니고, params 속성이 dict임을 검증
    assert best_trial is not None
    assert isinstance(best_trial.params, dict)
