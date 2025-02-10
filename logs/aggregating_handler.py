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
      [2025-02-10 14:25:23,659] INFO:backtesting.backtester:backtester.py:process_bullish_entry: 2021-04-12 08:00:00 - Bullish Entry Summary: 5000 events; 신규 진입 불가 (가용 잔고 부족): 3, 신규 진입 실행됨: 1, 스케일인 실행됨: 4996; 평균 진입가: 418.86
     
    모듈마다 로그 발생량이 다를 수 있으므로, 생성 시 module_name을 전달하면 
    환경변수 AGG_THRESHOLD_<MODULE_NAME> (대문자) 값이 있으면 이를 임계치로 사용합니다.
    """
    def __init__(self, threshold=None, level=logging.INFO, module_name=None):
        # 만약 threshold가 명시되지 않았다면, module_name을 기준으로 환경변수를 확인
        if threshold is None and module_name:
            env_var = f"AGG_THRESHOLD_{module_name.upper()}"
            threshold_str = os.getenv(env_var)
            if threshold_str and threshold_str.isdigit():
                threshold = int(threshold_str)
            else:
                threshold = 2000  # 기본값
        elif threshold is None:
            threshold = 2000

        super().__init__(level)
        self.threshold = threshold
        # 집계 딕셔너리: key = (logger 이름, 파일명, 함수명)
        self.aggregation = {}

    def emit(self, record):
        try:
            # 파일 경로에서 파일명만 추출
            filename = os.path.basename(record.pathname)
            key = (record.name, filename, record.funcName)
            
            # 해당 key에 대한 집계 정보 초기화
            if key not in self.aggregation:
                self.aggregation[key] = {
                    "count": 0,
                    "last_message": None,
                    "last_time": None,
                }
            agg = self.aggregation[key]
            agg["count"] += 1
            agg["last_message"] = record.getMessage()
            agg["last_time"] = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
            
            # 임계치(threshold)에 도달하면 요약 로그 생성
            if agg["count"] >= self.threshold:
                summary_msg = (
                    f"{key[2]} 집계: 최근 {agg['count']}회 로그 발생, 마지막 메시지: {agg['last_message']} at {agg['last_time']}"
                )
                summary_record = self.make_summary_record(record, summary_msg)
                # 현재 record의 logger를 사용하여 summary record를 처리
                logging.getLogger(record.name).handle(summary_record)
                # 집계 카운터 초기화
                self.aggregation[key]["count"] = 0
        except Exception:
            self.handleError(record)

    def make_summary_record(self, original_record, summary):
        """
        원본 로그 레코드의 정보를 유지하면서 msg만 요약 메시지로 대체한 새로운 LogRecord를 생성합니다.
        """
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
