# tests/test_performance_report.py
import io
import logging
from logs.final_report import generate_final_report

def test_final_report_output():
    sample_performance = {
        "overall": {
            "roi": 1.5,
            "cumulative_return": -0.015,
            "total_pnl": -150.0,
            "trade_count": 10,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "trades_per_year": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0
        },
        "monthly": {
            "2023-01": {"roi": 1.8, "trade_count": 5},
            "2023-02": {"roi": 2.2, "trade_count": 7},
            "2023-03": {"roi": 1.0, "trade_count": 4},
        },
        "weekly": {
            "weekly_roi": 0.0,
            "weekly_max_drawdown": 0.0
        }
    }
    
    log_stream = io.StringIO()
    logger = logging.getLogger("logs.final_report")
    stream_handler = logging.StreamHandler(log_stream)
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    generate_final_report(sample_performance)
    
    logger.removeHandler(stream_handler)
    output = log_stream.getvalue()
    # 핵심 지표들이 출력되는지 확인
    assert "ROI" in output
    assert "Trade Count" in output or "거래 횟수" in output
    for month in sample_performance["monthly"]:
        assert month in output
