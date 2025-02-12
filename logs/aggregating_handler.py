# logs/aggregating_handler.py
import logging
import os
import atexit

class AggregatingHandler(logging.Handler):
    """
    AggregatingHandler는 각 로그 레코드를 (logger 이름, 파일명, 함수명) 단위로 집계합니다.
    
    실행 시작부터 종료 시까지 누적한 모든 (파일명: 함수명) 단위의 로그 발생 건수를
    마지막에 한 번에 출력합니다.
    
    (주의: 로그 파일은 회전(backupCount=7)되어 중간 로그는 사라질 수 있으나,
     이 집계는 메모리 내에서 실행 전체의 로그 발생 건수를 누적하므로 실행 종료 시점의
     누적 결과를 확인할 수 있습니다.)
    """
    def __init__(self, level=logging.DEBUG):  # 기본 레벨을 DEBUG로 설정
        super().__init__(level)
        # 누적 집계: 실행 시작부터 종료까지의 전체 로그 발생 건수를 누적
        self.total_aggregation = {}
        # 프로그램 종료 시 누적 집계 요약을 출력하도록 flush_aggregation_summary()를 atexit에 등록
        atexit.register(self.flush_aggregation_summary)

    def emit(self, record):
        try:
            filename = os.path.basename(record.pathname)
            key = (record.name, filename, record.funcName)
            if key not in self.total_aggregation:
                self.total_aggregation[key] = {"count": 0}
            self.total_aggregation[key]["count"] += 1
        except Exception:
            self.handleError(record)

    def flush_aggregation_summary(self):
        """
        실행 시작부터 종료 시까지 누적된 (파일명: 함수명) 단위의 로그 발생 건수를
        한 번에 출력합니다.
        """
        if not self.total_aggregation:
            return
        summary_lines = []
        for key, agg in self.total_aggregation.items():
            logger_name, filename, funcname = key
            count = agg.get("count", 0)
            summary_lines.append(f"{filename}:{funcname} (logger: {logger_name}) - 총 {count}회 발생")
        summary = "\n".join(summary_lines)
        # extra 플래그를 추가하여 최종 집계 메시지임을 표시함으로써 콘솔 출력 등에서 제외할 수 있습니다.
        logging.getLogger().info("전체 누적 로그 집계:\n" + summary, extra={'_is_summary': True})
