# tests/test_logging_summary.py
import logging
import io
from logs.logging_util import LoggingUtil

def test_logging_summary_output():
    # 메모리 내 로그 스트림 설정
    log_stream = io.StringIO()
    
    # 대상 로거 생성 및 기존 핸들러 제거
    test_logger = logging.getLogger("test_logging_summary")
    test_logger.setLevel(logging.DEBUG)
    for h in test_logger.handlers[:]:
        test_logger.removeHandler(h)
    test_logger.propagate = False

    # 새 스트림 핸들러 추가
    stream_handler = logging.StreamHandler(log_stream)
    formatter = logging.Formatter('%(levelname)s:%(message)s')
    stream_handler.setFormatter(formatter)
    test_logger.addHandler(stream_handler)

    # LoggingUtil 인스턴스 생성 (테스트용 logger로 교체)
    logging_util = LoggingUtil("test_logging_summary")
    logging_util.logger = test_logger  # 테스트용 logger 주입

    # 요약 로그 임계치(예: 2000회) 전까지 이벤트 기록
    for i in range(1999):
        logging_util.log_event(f"Test event {i}", level=logging.DEBUG)

    # 2000번째 이벤트 – 이 시점에서 요약 로그(INFO 레벨)가 찍혀야 함
    logging_util.log_event("Test event 1999", level=logging.DEBUG)

    # 강제로 핸들러 flush
    stream_handler.flush()

    # 로그 출력값을 가져와서 요약 로그 메시지가 포함되었는지 확인
    output = log_stream.getvalue()
    assert "요약 로그:" in output
