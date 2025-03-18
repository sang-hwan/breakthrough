# tests/logging/test_aggregating_handler.py
# 이 파일은 AggregatingHandler와 LoggingUtil을 사용하여 로그 집계 기능이 올바르게 동작하는지 테스트합니다.
# AggregatingHandler는 로그 이벤트들을 집계하여 일정 임계치에 도달했을 때 요약 로그를 출력합니다.

import logging
import io
from logging.logging_util import LoggingUtil
from logging.aggregating_handler import AggregatingHandler  # AggregatingHandler 클래스를 명시적으로 가져옴

def test_logging_summary_output():
    """
    로그 집계 요약 출력 기능 테스트

    목적:
      - 다수의 로그 이벤트를 기록한 후 임계치(여기서는 2000번째 이벤트)에 도달하면
        AggregatingHandler가 집계 로그를 출력하는지 확인합니다.

    Parameters:
      없음

    Returns:
      없음 (assert를 통해 집계 메시지 출력 여부 검증)
    """
    # 메모리 내 로그 스트림 생성: 로그 결과를 문자열로 저장
    log_stream = io.StringIO()
    
    # 테스트 전용 로거 생성 (이름: test_logging_summary)
    test_logger = logging.getLogger("test_logging_summary")
    test_logger.setLevel(logging.DEBUG)
    
    # 기존 핸들러를 모두 제거하여 깨끗한 상태로 만듦
    for h in test_logger.handlers[:]:
        test_logger.removeHandler(h)
    
    # StringIO 스트림을 출력 대상으로 하는 핸들러 생성 및 포맷 지정
    stream_handler = logging.StreamHandler(log_stream)
    formatter = logging.Formatter('%(levelname)s:%(message)s')
    stream_handler.setFormatter(formatter)
    
    # test_logger와 root logger에 스트림 핸들러 추가 (모든 로그가 동일 스트림으로 출력되도록 함)
    test_logger.addHandler(stream_handler)
    root_logger = logging.getLogger()
    root_logger.addHandler(stream_handler)
    
    # AggregatingHandler 인스턴스 생성: DEBUG 레벨 이상의 로그를 집계하도록 설정
    agg_handler = AggregatingHandler(level=logging.DEBUG)
    test_logger.addHandler(agg_handler)
    
    # LoggingUtil 인스턴스 생성 및 테스트용 로거 주입
    logging_util = LoggingUtil("test_logging_summary")
    logging_util.logger = test_logger  # 테스트용 로거 주입으로 로그 기록 제어
    
    # 임계치(2000 이벤트) 미만까지 이벤트 로그 기록
    for i in range(1999):
        logging_util.log_event(f"Test event {i}")
    # 2000번째 이벤트 기록 – 이 시점에서 AggregatingHandler가 집계 로그를 생성해야 함
    logging_util.log_event("Test event 1999")
    
    # 집계된 로그를 강제로 출력하도록 flush 호출
    agg_handler.flush_aggregation_summary()
    
    stream_handler.flush()
    output = log_stream.getvalue()
    
    # 테스트 후, root logger에서 스트림 핸들러 제거 (테스트 후 정리)
    root_logger.removeHandler(stream_handler)
    
    # 출력된 로그에 집계 메시지(예: "집계:" 문자열)가 포함되어 있는지 검증
    assert "집계:" in output
