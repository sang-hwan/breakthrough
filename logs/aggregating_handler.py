# logs/aggregating_handler.py
import logging
import os
from datetime import datetime

class AggregatingHandler(logging.Handler):
    """
    AggregatingHandler는 각 로그 레코드를 (logger 이름, 파일명, 함수명) 별로 집계합니다.
    지정된 임계치(threshold)만큼 로그가 누적되면, 해당 모듈/함수에서 발생한 로그의 특성을 요약하여
    다음과 같은 형식으로 출력합니다.
    
    예시)
      INFO:trading.strategies:strategies.py:high_frequency_strategy: high_frequency_strategy 집계: 최근 2000회 로그 발생, 마지막 메시지: hold at 2021-04-02 04:00:00
     
    모듈마다 로그 발생량이 다를 수 있으므로, 생성 시 module_name을 전달하면 
    환경변수 AGG_THRESHOLD_<MODULE_NAME> (대문자) 값을 임계치로 사용하며,
    만약 로그 레코드에 sensitivity_analysis 속성이 True로 설정되면, 
    환경변수 AGG_THRESHOLD_SENSITIVITY 값을 별도로 적용할 수 있습니다.
    """
    def __init__(self, threshold=None, level=logging.INFO, module_name=None, sensitivity_threshold=None):
        # 먼저, threshold가 제공되지 않은 경우 AGG_THRESHOLD_GLOBAL를 사용
        if threshold is None:
            global_threshold_str = os.getenv("AGG_THRESHOLD_GLOBAL")
            if global_threshold_str and global_threshold_str.isdigit():
                threshold = int(global_threshold_str)
            else:
                threshold = 2000  # fallback 기본값
        # module_name이 제공되면 해당 모듈의 전용 임계치 확인 (우선 적용)
        if module_name:
            env_var = f"AGG_THRESHOLD_{module_name.upper()}"
            module_threshold_str = os.getenv(env_var)
            if module_threshold_str and module_threshold_str.isdigit():
                threshold = int(module_threshold_str)
        super().__init__(level)
        self.threshold = threshold

        # sensitivity_threshold: sensitivity_analysis 플래그가 있는 로그에 사용할 임계치
        if sensitivity_threshold is None:
            sens_str = os.getenv("AGG_THRESHOLD_SENSITIVITY")
            if sens_str and sens_str.isdigit():
                sensitivity_threshold = int(sens_str)
            else:
                sensitivity_threshold = threshold
        self.sensitivity_threshold = sensitivity_threshold

        self.aggregation = {}

    def emit(self, record):
        try:
            import os
            filename = os.path.basename(record.pathname)
            key = (record.name, filename, record.funcName)
            if getattr(record, "sensitivity_analysis", False):
                current_threshold = self.sensitivity_threshold
            else:
                current_threshold = self.threshold

            if key not in self.aggregation:
                self.aggregation[key] = {"count": 0, "last_message": None, "last_time": None}
            agg = self.aggregation[key]
            agg["count"] += 1
            agg["last_message"] = record.getMessage()
            agg["last_time"] = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
            
            if agg["count"] >= current_threshold:
                summary_msg = (
                    f"{key[2]} 집계: 최근 {agg['count']}회 로그 발생, 마지막 메시지: {agg['last_message']} at {agg['last_time']}"
                )
                summary_record = self.make_summary_record(record, summary_msg)
                logging.getLogger(record.name).handle(summary_record)
                self.aggregation[key]["count"] = 0
        except Exception:
            self.handleError(record)

    def make_summary_record(self, original_record, summary):
        summary_record = logging.LogRecord(
            name=original_record.name,
            level=original_record.levelno,
            pathname=original_record.pathname,
            lineno=original_record.lineno,
            msg=summary,
            args=original_record.args,
            exc_info=original_record.exc_info
        )
        return summary_record
