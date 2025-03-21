# tests/integration_tests/parameters/test_parameters_integration.py
import pytest

def test_parameters_integration(monkeypatch):
    """
    ConfigManager, market_parameters, signal_parameters 모듈이 함께 동작하는지 확인합니다.
    - 시장 데이터 기반 업데이트
    - 민감도 분석 실행
    - 기본 신호 계산 파라미터 확인
    """
    # monkeypatch: Dummy Backtester와 성능 계산 함수 적용 (market_parameters)
    from parameters.market_parameters import run_sensitivity_analysis
    from parameters.trading_parameters import ConfigManager
    from parameters.signal_parameters import get_default_signal_config

    # Dummy Backtester 설정
    class DummyBacktester:
        def __init__(self, symbol, account_size):
            self.symbol = symbol
            self.account_size = account_size

        def load_data(self, short_table_format, long_table_format, short_tf, long_tf, start_date, end_date):
            pass

        def run_backtest_pipeline(self, dynamic_params):
            dummy_trade = {"pnl": 100}
            return ([dummy_trade], None)

    def dummy_compute_performance(trades):
        return {
            "roi": sum(trade.get("pnl", 0) for trade in trades) / 10000 * 100,
            "sharpe_ratio": 1.0,
            "max_drawdown": -5.0,
            "trade_count": len(trades),
            "cumulative_return": 0.1,
            "total_pnl": sum(trade.get("pnl", 0) for trade in trades)
        }

    monkeypatch.setattr("parameters.market_parameters.Backtester", DummyBacktester)
    monkeypatch.setattr("backtesting.performance.compute_performance", dummy_compute_performance)

    # 1. 기본 신호 파라미터 확인
    signal_config = get_default_signal_config()
    assert isinstance(signal_config, dict)
    assert "rsi_period" in signal_config

    # 2. ConfigManager의 기본값 및 업데이트 확인
    cm = ConfigManager()
    defaults = cm.get_defaults()
    dummy_market_data = {
        "volatility": 0.07,
        "trend": "bearish",
        "volume": 500,
        "weekly_volatility": 0.09,
        "weekly_low": 90,
        "weekly_high": 100
    }
    updated_params = cm.update_with_market_data(dummy_market_data)
    assert updated_params != defaults

    # 3. 민감도 분석 실행
    dummy_param_settings = {
        "profit_ratio": [0.09, 0.095],
        "atr_multiplier": [2.0, 2.05]
    }
    assets = ["BTC/USDT"]
    sensitivity_results = run_sensitivity_analysis(dummy_param_settings, assets, max_combinations=2)
    assert isinstance(sensitivity_results, dict)
    # 최소 하나 이상의 조합 결과가 존재함을 확인
    assert len(sensitivity_results) > 0
