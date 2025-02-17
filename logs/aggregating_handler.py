# logs/aggregating_handler.py
import logging
import os

class AggregatingHandler(logging.Handler):
    """
    AggregatingHandler는 (logger 이름, 파일명, 함수명) 단위로 로그 발생 건수를 집계합니다.
    추가로, 'is_weekly_signal' 플래그가 설정된 로그를 별도로 집계하여 주간 전략 신호 발생 건수를 파악할 수 있습니다.
    flush 시 전체 로그 집계와 주간 신호 로그 집계를 각각 출력합니다.
    """
    def __init__(self, level=logging.DEBUG):
        super().__init__(level)
        self.total_aggregation = {}
        self.weekly_signal_aggregation = {}

    def emit(self, record):
        try:
            key = (record.name, os.path.basename(record.pathname), record.funcName)
            self.total_aggregation[key] = self.total_aggregation.get(key, 0) + 1

            # 주간 전략 신호 이벤트 플래그가 있으면 별도 집계
            if getattr(record, 'is_weekly_signal', False):
                self.weekly_signal_aggregation[key] = self.weekly_signal_aggregation.get(key, 0) + 1
        except Exception:
            self.handleError(record)

    def flush_aggregation_summary(self):
        # 전체 로그 집계 요약 출력
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

        # 주간 신호 로그 집계 요약 출력 (있을 경우)
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
