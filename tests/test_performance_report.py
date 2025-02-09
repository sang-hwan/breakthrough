# tests/test_performance_report.py

import io
import sys
from logs.final_report import generate_final_report

def test_final_report_output():
    sample_performance = {
        "roi": 1.5,
        "pnl": -150.0,
        "trade_count": 10,
        "monthly_roi": {
            "2023-01": 1.8,
            "2023-02": 2.2,
            "2023-03": 1.0,
        }
    }
    # stdout 캡처
    captured_output = io.StringIO()
    sys.stdout = captured_output

    generate_final_report(sample_performance)

    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    # 핵심 지표들이 출력되는지 확인 (예: ROI, 거래 횟수, 월별 데이터 등)
    assert "ROI" in output
    # 출력된 리포트에서 거래 횟수를 "거래 횟수:"로 표기하도록 변경했으므로 이를 확인합니다.
    assert "trade_count" in output or "거래 횟수" in output
    for month in sample_performance["monthly_roi"]:
        assert month in output
