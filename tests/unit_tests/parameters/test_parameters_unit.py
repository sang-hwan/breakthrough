# tests/unit_tests/parameters/test_parameters_unit.py
import pytest
import numpy as np

# --- Dummy classes and functions for monkeypatching ---

class DummyBacktester:
    """
    Dummy Backtester 클래스.
    - load_data: 아무 작업도 수행하지 않음.
    - run_backtest_pipeline: 단일 거래 결과(dummy trade)를 반환.
    """
    def __init__(self, symbol, account_size):
        self.symbol = symbol
        self.account_size = account_size

    def load_data(self, short_table_format, long_table_format, short_tf, long_tf, start_date, end_date):
        pass

    def run_backtest_pipeline(self, dynamic_params):
        # 단일 거래 결과(dummy trade)를 반환 (pnl=100)
        dummy_trade = {"pnl": 100}
        return ([dummy_trade], None)

def dummy_compute_performance(trades):
    """
    Dummy 성능 지표 계산 함수.
    """
    return {
        "roi": sum(trade.get("pnl", 0) for trade in trades) / 10000 * 100,
        "sharpe_ratio": 1.0,
        "max_drawdown": -5.0,
        "trade_count": len(trades),
        "cumulative_return": 0.1,
        "total_pnl": sum(trade.get("pnl", 0) for trade in trades)
    }

# --- Unit tests ---

def test_run_sensitivity_analysis(monkeypatch):
    """
    market_parameters.run_sensitivity_analysis의 기본 동작 검증.
    Dummy Backtester와 성능 계산 함수를 사용하여, 
    단일 자산에 대해 거래가 수행되고 결과가 집계되는지 확인합니다.
    """
    from parameters.market_parameters import run_sensitivity_analysis
    from parameters.trading_parameters import ConfigManager

    # monkeypatch Backtester 사용
    monkeypatch.setattr("parameters.market_parameters.Backtester", DummyBacktester)
    # monkeypatch compute_performance 함수 사용 (모듈 내부에서 동적 임포트됨)
    monkeypatch.setattr("backtesting.performance.compute_performance", dummy_compute_performance)

    dummy_param_settings = {
        "profit_ratio": [0.09, 0.098],
        "atr_multiplier": [2.0, 2.07]
    }
    assets = ["BTC/USDT"]

    # 실행 시 에러 없이 결과가 반환되는지 확인
    results = run_sensitivity_analysis(dummy_param_settings, assets, max_combinations=2)
    assert isinstance(results, dict)
    # 각 조합별 결과가 집계되어 있거나 None이어야 함
    for combo_key, metrics in results.items():
        if metrics is not None:
            for key in ["roi", "sharpe_ratio", "max_drawdown", "trade_count",
                        "cumulative_return", "total_pnl"]:
                assert key in metrics
                # 평균, std, min, max 값이 모두 float 타입인지 확인
                for stat in metrics[key].values():
                    assert isinstance(stat, (float, np.floating, int))

def test_get_default_signal_config():
    """
    signal_parameters.get_default_signal_config의 반환값이 기본값 딕셔너리임을 검증합니다.
    """
    from parameters.signal_parameters import get_default_signal_config

    config = get_default_signal_config()
    assert isinstance(config, dict)
    # 기본적으로 정의한 파라미터 값들이 존재하는지 확인 (예: rsi_period)
    assert "rsi_period" in config
    assert config["rsi_period"] >= 1

def test_config_manager_update(monkeypatch):
    """
    trading_parameters.ConfigManager의 update_with_market_data 기능 테스트.
    Dummy 데이터를 사용해 업데이트 후, 파라미터 값이 변경되는지 확인합니다.
    """
    from parameters.trading_parameters import ConfigManager

    cm = ConfigManager()
    default_params = cm.get_defaults()

    dummy_market_data = {
        "volatility": 0.06,
        "trend": "bullish",
        "volume": 800,
        "weekly_volatility": 0.08,
        "weekly_low": 100,
        "weekly_high": 110
    }
    updated_params = cm.update_with_market_data(dummy_market_data)
    # 변경 사항이 반영되어 기본값과 다르거나, 일부 값이 조정되었음을 확인
    assert updated_params != default_params
    # 예를 들어 profit_ratio가 상승(bullish) 방향으로 조정되었을 가능성 확인
    assert updated_params["profit_ratio"] >= default_params["profit_ratio"]
