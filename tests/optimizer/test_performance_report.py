# tests/optimizer/test_performance_report.py
# 이 파일은 generate_final_report 함수를 사용하여
# 최종 성과 보고서에 주요 성과 지표(ROI, 거래 횟수 등)가 올바르게 출력되는지 테스트합니다.

import io
import logging
from logs.final_report import generate_final_report

def test_final_report_output():
    """
    최종 성과 보고서 출력 테스트

    목적:
      - 샘플 성과 데이터를 기반으로 generate_final_report 함수를 실행한 후,
        로그 스트림에 ROI, Trade Count(또는 거래 횟수) 및 월별 지표들이 포함되어 있는지 확인.
    
    Parameters:
      없음

    Returns:
      없음 (assert 구문을 통해 출력된 로그 내용 검증)
    """
    # 샘플 성과 데이터: 전체 성과(overall), 월별(monthly), 주별(weekly) 지표 포함
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
    
    # 메모리 내 스트림을 로그 출력 대상으로 설정
    log_stream = io.StringIO()
    logger = logging.getLogger("logs.final_report")
    stream_handler = logging.StreamHandler(log_stream)
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # 성과 보고서 생성 함수 호출
    generate_final_report(sample_performance)
    
    # 테스트 후 핸들러 제거하여 정리
    logger.removeHandler(stream_handler)
    stream_handler.flush()
    output = log_stream.getvalue()
    
    # 출력 로그에 핵심 지표(ROI, Trade Count 또는 거래 횟수, 월별 날짜)가 포함되었는지 검증
    assert "ROI" in output
    assert "Trade Count" in output or "거래 횟수" in output
    for month in sample_performance["monthly"]:
        assert month in output
