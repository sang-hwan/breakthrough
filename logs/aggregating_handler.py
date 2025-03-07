# logs/aggregating_handler.py
import logging
import os
import threading

class AggregatingHandler(logging.Handler):
    """
    AggregatingHandler는 (logger 이름, 파일명, 함수명)별로 이벤트 발생 횟수를 집계합니다.
    플러시 시 집계된 결과를 최종 로그로 출력하고, 집계를 초기화합니다.
    """

    def __init__(self, level=logging.DEBUG):
        super().__init__(level)
        self.total_aggregation = {}
        self.weekly_signal_aggregation = {}
        self.lock = threading.RLock()

    def emit(self, record):
        try:
            key = (record.name, os.path.basename(record.pathname), record.funcName)
            with self.lock:
                self.total_aggregation[key] = self.total_aggregation.get(key, 0) + 1
                if getattr(record, 'is_weekly_signal', False):
                    self.weekly_signal_aggregation[key] = self.weekly_signal_aggregation.get(key, 0) + 1
        except Exception:
            self.handleError(record)

    def flush_aggregation_summary(self):
        with self.lock:
            if self.total_aggregation:
                summary_lines = [
                    f"{filename}:{funcname} (logger: {logger_name}) - 총 {count}회 발생"
                    for (logger_name, filename, funcname), count in self.total_aggregation.items()
                ]
                summary = "\n".join(summary_lines)
                try:
                    logging.getLogger().info("전체 누적 로그 집계:\n" + summary)
                except Exception:
                    pass
                self.total_aggregation.clear()
            if self.weekly_signal_aggregation:
                weekly_summary_lines = [
                    f"{filename}:{funcname} (logger: {logger_name}) - 주간 신호 {count}회 발생"
                    for (logger_name, filename, funcname), count in self.weekly_signal_aggregation.items()
                ]
                weekly_summary = "\n".join(weekly_summary_lines)
                try:
                    logging.getLogger().info("전체 주간 신호 로그 집계:\n" + weekly_summary)
                except Exception:
                    pass
                self.weekly_signal_aggregation.clear()
