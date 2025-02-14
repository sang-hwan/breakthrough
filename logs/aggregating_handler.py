# logs/aggregating_handler.py
import logging
import os

class AggregatingHandler(logging.Handler):
    """
    AggregatingHandler는 (logger 이름, 파일명, 함수명) 단위로 로그 발생 건수를 집계하여,
    실행 종료 시 누적 결과를 한 번에 출력합니다.
    """
    def __init__(self, level=logging.DEBUG):
        super().__init__(level)
        self.total_aggregation = {}

    def emit(self, record):
        try:
            key = (record.name, os.path.basename(record.pathname), record.funcName)
            self.total_aggregation[key] = self.total_aggregation.get(key, 0) + 1
        except Exception:
            self.handleError(record)

    def flush_aggregation_summary(self):
        if not self.total_aggregation:
            return
        summary_lines = [
            f"{filename}:{funcname} (logger: {logger_name}) - 총 {count}회 발생"
            for (logger_name, filename, funcname), count in self.total_aggregation.items()
        ]
        summary = "\n".join(summary_lines)
        try:
            logging.getLogger().info("전체 누적 로그 집계:\n" + summary)
        except Exception:
            pass
