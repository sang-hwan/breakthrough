# tests/integration_tests/optimization/test_optimization_integration.py
import datetime
import pandas as pd
import numpy as np
import pytest

# trade_optimize 모듈의 함수 임포트
from optimization.trade_optimize import (
    generate_final_report,
    generate_parameter_sensitivity_report,
    generate_weekly_signal_report,
    calculate_overall_performance,
    calculate_monthly_performance,
    compute_performance
)


def dummy_trades():
    """
    테스트용 dummy 거래 데이터를 생성합니다.
    """
    now = datetime.datetime.now()
    trades = []
    for i in range(5):
        trade = {
            "pnl": 100 * (-1) ** i,  # 양수/음수 번갈아 발생
            "entry_time": now - datetime.timedelta(days=5 - i),
            "exit_time": now - datetime.timedelta(days=4 - i)
        }
        trades.append(trade)
    return trades


def dummy_weekly_data():
    """
    테스트용 dummy 주간 데이터를 DataFrame으로 생성합니다.
    """
    dates = pd.date_range(start="2023-01-01", periods=10, freq="W")
    data = {"close": np.linspace(100, 110, num=10)}
    return pd.DataFrame(data, index=dates)


def test_calculate_overall_performance():
    """
    전체 성과 계산 함수가 필수 지표들을 포함하는 결과를 반환하는지 확인합니다.
    """
    trades = dummy_trades()
    overall = calculate_overall_performance(trades)
    required_keys = [
        "roi", "cumulative_return", "total_pnl", "trade_count", "annualized_return",
        "annualized_volatility", "sharpe_ratio", "sortino_ratio", "calmar_ratio",
        "max_drawdown", "win_rate", "avg_win", "avg_loss", "profit_factor",
        "trades_per_year", "max_consecutive_wins", "max_consecutive_losses"
    ]
    for key in required_keys:
        assert key in overall


def test_calculate_monthly_performance():
    """
    월별 및 주간 성과 지표 계산 함수가 올바른 결과를 반환하는지 확인합니다.
    """
    trades = dummy_trades()
    weekly_data = dummy_weekly_data()
    performance = calculate_monthly_performance(trades, weekly_data)
    assert "monthly" in performance
    assert "weekly" in performance
    # 월별 데이터가 하나 이상 존재해야 합니다.
    assert len(performance["monthly"]) > 0


def test_compute_performance():
    """
    전체, 월별, 주간 성과 지표를 포함하는 종합 성과 계산 함수의 결과를 검증합니다.
    """
    trades = dummy_trades()
    weekly_data = dummy_weekly_data()
    performance = compute_performance(trades, weekly_data)
    assert "overall" in performance
    assert "monthly" in performance
    assert "weekly" in performance


def test_generate_final_report(caplog):
    """
    최종 성과 보고서 생성 함수가 로그에 보고서를 기록하는지 확인합니다.
    """
    performance = {
        "overall": {
            "roi": 10.0, "cumulative_return": 0.1, "total_pnl": 1000,
            "trade_count": 5, "annualized_return": 12.0, "annualized_volatility": 15.0,
            "sharpe_ratio": 0.8, "sortino_ratio": 0.7, "calmar_ratio": 1.2, "max_drawdown": 5.0,
            "win_rate": 60.0, "avg_win": 150.0, "avg_loss": -100.0, "profit_factor": 1.5,
            "trades_per_year": 50, "max_consecutive_wins": 3, "max_consecutive_losses": 2
        },
        "monthly": {
            "2023-01": {"roi": 2.5, "trade_count": 2, "total_pnl": 200},
            "2023-02": {"roi": 3.0, "trade_count": 3, "total_pnl": 300}
        },
        "weekly": {"weekly_roi": 1.0, "weekly_max_drawdown": -0.5}
    }
    generate_final_report(performance, symbol="BTC/USDT")
    logs = [record.message for record in caplog.records]
    assert any("FINAL BACKTEST PERFORMANCE REPORT" in log for log in logs)


def test_generate_parameter_sensitivity_report(caplog):
    """
    파라미터 민감도 보고서 생성 함수가 로그에 보고서를 기록하는지 확인합니다.
    """
    results = {
        14: {"roi": 5.0},
        15: {"roi": 6.0},
        16: {"roi": 4.5}
    }
    generate_parameter_sensitivity_report("rsi_period", results)
    logs = [record.message for record in caplog.records]
    assert any("FINAL PARAMETER SENSITIVITY REPORT" in log for log in logs)


def test_generate_weekly_signal_report(caplog):
    """
    주간 신호 보고서 생성 함수가 로그에 보고서를 기록하는지 확인합니다.
    """
    weekly_signal_counts = {
        ("logger_name", "file.py", "func"): 5,
        ("other_logger", "other_file.py", "other_func"): 3
    }
    generate_weekly_signal_report(weekly_signal_counts)
    logs = [record.message for record in caplog.records]
    assert any("WEEKLY SIGNAL REPORT" in log for log in logs)
