# tests/test_aggregating_handler.py
import logging
import io
from logs.logging_util import LoggingUtil
from logs.aggregating_handler import AggregatingHandler  # AggregatingHandler를 명시적으로 추가

def test_logging_summary_output():
    # 메모리 내 로그 스트림 설정
    log_stream = io.StringIO()
    
    test_logger = logging.getLogger("test_logging_summary")
    test_logger.setLevel(logging.DEBUG)
    
    # 기존 핸들러 제거 후 새 스트림 핸들러 추가
    for h in test_logger.handlers[:]:
        test_logger.removeHandler(h)
    
    stream_handler = logging.StreamHandler(log_stream)
    formatter = logging.Formatter('%(levelname)s:%(message)s')
    stream_handler.setFormatter(formatter)
    
    # test_logger와 root logger 모두에 스트림 핸들러 추가
    test_logger.addHandler(stream_handler)
    root_logger = logging.getLogger()
    root_logger.addHandler(stream_handler)
    
    # AggregatingHandler 추가 (테스트 시에 집계 로그를 생성하기 위함)
    agg_handler = AggregatingHandler(level=logging.DEBUG)
    test_logger.addHandler(agg_handler)
    
    # LoggingUtil 인스턴스 생성 후 테스트용 logger 주입
    logging_util = LoggingUtil("test_logging_summary")
    logging_util.logger = test_logger  # 테스트용 logger 주입
    
    # 임계치 전까지 이벤트 기록 (예: 2000회 미만)
    for i in range(1999):
        logging_util.log_event(f"Test event {i}")
    # 2000번째 이벤트 – 이 시점에서 집계 로그가 찍혀야 함
    logging_util.log_event("Test event 1999")
    
    # AggregatingHandler의 flush를 호출하여 집계 로그를 강제로 출력
    agg_handler.flush_aggregation_summary()
    
    stream_handler.flush()
    output = log_stream.getvalue()
    
    # 테스트 후 root logger에서 스트림 핸들러 제거 (선택 사항)
    root_logger.removeHandler(stream_handler)
    
    # 집계 메시지(예: "전체 누적 로그 집계:" 또는 "집계:" 문자열)가 출력에 포함되었는지 확인
    assert "집계:" in output
