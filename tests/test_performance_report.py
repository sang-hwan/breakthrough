# tests/test_performance_report.py

import io
import logging
from logs.final_report import generate_final_report

def test_final_report_output():
    sample_performance = {
        "roi": 1.5,
        "pnl": -150.0,
        "trade_count": 10,
        "monthly_performance": {
            "2023-01": {"roi": 1.8, "trade_count": 5},
            "2023-02": {"roi": 2.2, "trade_count": 7},
            "2023-03": {"roi": 1.0, "trade_count": 4},
        }
    }
    
    # logger 출력 캡처를 위한 스트림 핸들러 설정
    log_stream = io.StringIO()
    # generate_final_report()에서 사용되는 logger의 이름은 보통 "logs.final_report" 입니다.
    logger = logging.getLogger("logs.final_report")
    stream_handler = logging.StreamHandler(log_stream)
    stream_handler.setLevel(logging.debug)
    formatter = logging.Formatter('%(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # 리포트 생성 (logger를 통해 출력됨)
    generate_final_report(sample_performance)
    
    # 테스트 후 핸들러 제거
    logger.removeHandler(stream_handler)
    output = log_stream.getvalue()
    
    # 핵심 지표들이 출력되는지 확인 (예: ROI, Trade Count, 월별 데이터 등)
    assert "ROI" in output
    assert "Trade Count" in output or "거래 횟수" in output
    for month in sample_performance["monthly_performance"]:
        assert month in output
