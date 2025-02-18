# logs/aggregating_handler.py
import logging
import os
import threading

class AggregatingHandler(logging.Handler):
    """
    AggregatingHandler aggregates log occurrence counts by (logger name, filename, function name).
    Additionally, logs with the 'is_weekly_signal' flag are aggregated separately to track weekly strategy signal events.
    Upon flush, it outputs summaries of the total aggregated logs and the weekly signal logs.
    """
    def __init__(self, level=logging.DEBUG):
        super().__init__(level)
        self.total_aggregation = {}
        self.weekly_signal_aggregation = {}
        # 재진입(Reentrant) Lock을 사용하여 flush_aggregation_summary 중 재호출 시 데드락을 방지합니다.
        self.lock = threading.RLock()

    def emit(self, record):
        try:
            key = (record.name, os.path.basename(record.pathname), record.funcName)
            with self.lock:
                self.total_aggregation[key] = self.total_aggregation.get(key, 0) + 1

                # 만약 record에 'is_weekly_signal' 플래그가 있으면 별도로 집계
                if getattr(record, 'is_weekly_signal', False):
                    self.weekly_signal_aggregation[key] = self.weekly_signal_aggregation.get(key, 0) + 1
        except Exception:
            self.handleError(record)

    def flush_aggregation_summary(self):
        with self.lock:
            # 전체 누적 로그 집계 출력
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

            # 주간 신호 로그 집계 출력 (존재하는 경우)
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
