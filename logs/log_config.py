# logs/log_config.py
import logging  # 로그 기록을 위한 표준 모듈
import os       # 파일 경로 및 디렉토리 생성 관련 모듈
import queue    # 로그 메시지 큐를 다루기 위한 모듈
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener  # 회전 파일 핸들러 및 큐 관련 핸들러 임포트
from logs.aggregating_handler import AggregatingHandler  # 사용자 정의 AggregatingHandler 임포트

# 전역 변수: 파일 로그는 INFO 레벨 이상의 메시지만 기록
FILE_LOG_LEVEL = logging.INFO
# 전역 변수: 콘솔 로그는 DEBUG 레벨 이상의 메시지를 기록
detail_level = logging.DEBUG

# 로그 파일을 저장할 디렉토리 설정 (logs/log_files)
LOG_FILES_DIR = os.path.join("logs", "log_files")
if not os.path.exists(LOG_FILES_DIR):
    os.makedirs(LOG_FILES_DIR)  # 디렉토리가 없으면 생성
# 기본 로그 파일 경로 설정
BASE_LOG_FILE = os.path.join(LOG_FILES_DIR, "project.log")

class OneLineFormatter(logging.Formatter):
    """
    OneLineFormatter는 로그 메시지 내의 줄바꿈을 파이프(|) 문자로 대체하여 한 줄로 출력하도록 포맷팅합니다.
    """
    def format(self, record):
        formatted = super().format(record)  # 기본 포맷팅 수행
        return formatted.replace("\n", " | ")  # 줄바꿈을 ' | '로 대체

class LineRotatingFileHandler(RotatingFileHandler):
    """
    LineRotatingFileHandler는 로그 파일의 최대 라인 수를 초과하면 새로운 파일로 롤오버하는 기능을 제공합니다.
    
    주요 기능:
      - 파일 크기 대신 로그 라인 수를 기준으로 롤오버
      - 롤오버 시 새로운 파일 이름 생성 및 라인 카운트 초기화
    """
    def __init__(self, base_filename, mode='a', max_lines=500, encoding=None, delay=False):
        """
        초기화 메서드
        
        Parameters:
            base_filename (str): 기본 로그 파일 이름
            mode (str): 파일 열기 모드 (기본 'a' - append)
            max_lines (int): 하나의 로그 파일에 기록할 최대 라인 수
            encoding (str): 파일 인코딩 (기본값 None)
            delay (bool): 파일 열기를 지연할지 여부 (기본 False)
        """
        self.base_filename = base_filename
        self.current_index = 0  # 현재 파일 인덱스, 0이면 기본 파일 사용
        self._set_current_filename()  # 현재 파일 이름 설정
        super().__init__(self.current_filename, mode, maxBytes=0, encoding=encoding, delay=delay)
        self.max_lines = max_lines  # 최대 라인 수 설정
        self.current_line_count = 0  # 현재까지 기록한 라인 수

    def _set_current_filename(self):
        """
        현재 인덱스에 따라 현재 로그 파일 이름을 설정합니다.
        """
        base, ext = os.path.splitext(self.base_filename)
        # 인덱스 0이면 기본 파일명, 그 외에는 인덱스 번호를 포함한 파일명 사용
        self.current_filename = self.base_filename if self.current_index == 0 else f"{base}{self.current_index}{ext}"
        self.baseFilename = os.path.abspath(self.current_filename)

    def doRollover(self):
        """
        로그 파일 롤오버를 수행합니다.
        
        주요 동작:
          - 기존 스트림을 닫고, 파일 인덱스를 증가시킨 후 새로운 파일로 전환
          - 라인 카운트를 초기화하여 새 파일에서 새로 기록 시작
        """
        if self.stream:
            self.stream.close()
            self.stream = None
        self.current_index += 1  # 파일 인덱스 증가
        self._set_current_filename()  # 새로운 파일 이름 설정
        self.mode = 'w'  # 새 파일은 쓰기 모드로 열림
        self.stream = self._open()  # 새로운 파일 스트림 열기
        self.current_line_count = 0  # 라인 카운트 초기화

    def emit(self, record):
        """
        로그 레코드를 기록하기 전에 메시지의 라인 수를 체크하고,
        최대 라인 수를 초과하면 롤오버를 수행한 후 기록합니다.
        
        Parameters:
            record (logging.LogRecord): 기록할 로그 이벤트
        
        주요 동작:
          - 로그 메시지를 포맷팅하여 몇 줄인지 확인
          - 현재 기록된 라인 수와 합산해 최대치를 초과하면 doRollover() 호출
          - 업데이트된 라인 수를 반영 후 상위 emit() 메서드로 메시지 기록
        """
        try:
            if record.levelno < FILE_LOG_LEVEL:
                return  # 설정된 레벨보다 낮은 메시지는 무시
            msg = self.format(record)  # 로그 메시지 포맷팅
            lines_in_msg = msg.count("\n") or 1  # 메시지 내 줄바꿈 수 계산 (없으면 1라인)
            if self.current_line_count + lines_in_msg > self.max_lines:
                self.doRollover()  # 최대 라인 수 초과 시 롤오버 수행
            self.current_line_count += lines_in_msg  # 현재 라인 수 업데이트
            super().emit(record)  # 상위 클래스의 emit() 호출하여 로그 기록
        except Exception:
            self.handleError(record)

# 전역 변수: 로그 메시지 큐 (크기 무제한)
log_queue = queue.Queue(-1)
queue_listener = None  # 전역 변수: QueueListener 객체 (추후 초기화)

def initialize_root_logger():
    """
    루트 로거를 초기화하여 파일 핸들러, 콘솔 핸들러, 큐 핸들러 및 AggregatingHandler를 추가합니다.
    
    주요 동작:
      - 루트 로거의 레벨 설정 및 기존 핸들러 제거
      - 로그 파일용 핸들러와 콘솔용 핸들러 생성 및 설정
      - 큐 핸들러를 통해 비동기 로깅 구성
      - 외부 라이브러리(ccxt, urllib3)의 로그 레벨을 경고(WARNING)로 설정
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(detail_level)

    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 파일 핸들러 생성: 로그 파일은 LOG_FILES_DIR 내에 생성됨
    file_handler = LineRotatingFileHandler(
        base_filename=BASE_LOG_FILE,
        max_lines=500,
        encoding="utf-8",
        delay=True
    )
    file_handler.setLevel(FILE_LOG_LEVEL)
    formatter = OneLineFormatter('[%(asctime)s] %(levelname)s:%(name)s:%(filename)s:%(funcName)s: %(message)s')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 콘솔 핸들러 생성
    console_handler = logging.StreamHandler()
    console_handler.setLevel(detail_level)
    console_formatter = OneLineFormatter('[%(asctime)s] %(levelname)s:%(name)s:%(filename)s:%(funcName)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    # _is_summary 속성이 있는 레코드는 필터링(제외)
    console_handler.addFilter(lambda record: not getattr(record, '_is_summary', False))

    # 큐 핸들러 추가: 로그 메시지를 큐에 넣어 비동기 처리 지원
    q_handler = QueueHandler(log_queue)
    root_logger.addHandler(q_handler)

    global queue_listener
    queue_listener = QueueListener(log_queue, console_handler)
    queue_listener.start()

    # AggregatingHandler 추가: 로그 이벤트 집계를 위해 설정
    try:
        aggregator_handler = AggregatingHandler(level=detail_level)
        aggregator_handler.addFilter(lambda record: not getattr(record, '_is_summary', False))
        aggregator_formatter = OneLineFormatter('[%(asctime)s] %(levelname)s:%(name)s:%(filename)s:%(funcName)s: %(message)s')
        aggregator_handler.setFormatter(aggregator_formatter)
        root_logger.addHandler(aggregator_handler)
    except Exception as e:
        logging.getLogger().error("Failed to add AggregatingHandler: " + str(e), exc_info=True)

    # 외부 라이브러리 로그 레벨 조정 (불필요한 디버그 메시지 제거)
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

def setup_logger(module_name: str) -> logging.Logger:
    """
    모듈별 로거를 설정하여 반환합니다.
    
    Parameters:
        module_name (str): 해당 모듈을 식별하기 위한 이름.
    
    Returns:
        logging.Logger: 설정된 로거 객체.
    
    주요 동작:
      - 모듈 이름에 따라 로거 생성 후, 모듈 전용 AggregatingHandler를 추가하여 집계 기능 제공.
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(detail_level)
    logger.propagate = True  # 상위 로거로 메시지 전파
    try:
        agg_handler = AggregatingHandler(level=detail_level)
        agg_handler.addFilter(lambda record: not getattr(record, '_is_summary', False))
        formatter = OneLineFormatter('[%(asctime)s] %(levelname)s:%(name)s:%(filename)s:%(funcName)s: %(message)s')
        agg_handler.setFormatter(formatter)
        logger.addHandler(agg_handler)
    except Exception as e:
        logger.error("Failed to add module-specific AggregatingHandler: " + str(e), exc_info=True)
    return logger

def shutdown_logging():
    """
    로깅 시스템을 종료합니다.
    
    주요 동작:
      - 루트 로거에 추가된 핸들러 중 flush_aggregation_summary() 메서드가 있으면 호출하여 집계 결과 플러시
      - 큐 리스너를 중지한 후 logging.shutdown()을 호출하여 모든 로깅 자원 정리
    """
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        try:
            if hasattr(handler, 'flush_aggregation_summary'):
                handler.flush_aggregation_summary()
        except Exception:
            pass
    global queue_listener
    if queue_listener is not None:
        queue_listener.stop()
        queue_listener = None
    logging.shutdown()

import logging  # 로그 기록을 위한 표준 모듈
import os       # 파일 및 경로 관련 작업 지원
import threading  # 스레드 동기화를 위한 모듈

class AggregatingHandler(logging.Handler):
    """
    AggregatingHandler는 (logger 이름, 파일명, 함수명)별로 이벤트 발생 횟수를 집계합니다.
    플러시 시 집계된 결과를 최종 로그로 출력하고, 집계를 초기화합니다.
    
    주요 기능:
      - 로그 이벤트가 발생할 때마다 해당 키의 발생 횟수를 카운트합니다.
      - 'is_weekly_signal' 속성이 있는 경우 별도로 주간 신호 집계합니다.
    """
    def __init__(self, level=logging.DEBUG):
        """
        초기화 메서드
        
        Parameters:
            level (int): 로깅 레벨 (기본값: logging.DEBUG)
        
        주요 동작:
          - 전체 로그 집계와 주간 신호 집계를 위한 딕셔너리 초기화
          - 스레드 안전성을 보장하기 위해 RLock 객체 생성
        """
        super().__init__(level)
        self.total_aggregation = {}       # 전체 로그 집계를 위한 딕셔너리, key: (logger name, filename, funcName)
        self.weekly_signal_aggregation = {}  # 주간 신호 로그 집계를 위한 딕셔너리
        self.lock = threading.RLock()       # 스레드 간 동시 접근 제어를 위한 재진입 락

    def emit(self, record):
        """
        로그 레코드가 발생할 때마다 호출되며, 해당 이벤트를 집계합니다.
        
        Parameters:
            record (logging.LogRecord): 발생한 로그 이벤트에 대한 정보를 담은 객체
        
        주요 동작:
          - record에서 (logger name, 파일명, 함수명)을 추출하여 집계의 key로 사용
          - 전체 집계와 'is_weekly_signal' 속성에 따른 주간 신호 집계를 각각 업데이트
          - 예외 발생 시 handleError() 호출하여 오류 처리
        """
        try:
            # 파일 경로에서 파일 이름만 추출하여 key 생성
            key = (record.name, os.path.basename(record.pathname), record.funcName)
            with self.lock:  # 스레드 안전하게 집계 딕셔너리 업데이트
                self.total_aggregation[key] = self.total_aggregation.get(key, 0) + 1
                # 'is_weekly_signal' 속성이 True면 주간 신호 집계 업데이트
                if getattr(record, 'is_weekly_signal', False):
                    self.weekly_signal_aggregation[key] = self.weekly_signal_aggregation.get(key, 0) + 1
        except Exception:
            self.handleError(record)

    def flush_aggregation_summary(self):
        """
        집계된 로그 결과를 포맷팅하여 최종 로그로 출력한 후, 집계 데이터를 초기화합니다.
        
        주요 동작:
          - 전체 집계와 주간 신호 집계에 대해 각각 문자열 리스트를 생성 후 하나의 메시지로 결합
          - 생성된 메시지를 로그에 기록하고, 집계 딕셔너리를 초기화
        """
        with self.lock:  # 집계 데이터에 대한 동시 접근 제어
            if self.total_aggregation:
                summary_lines = [
                    f"{filename}:{funcname} (logger: {logger_name}) - 총 {count}회 발생"
                    for (logger_name, filename, funcname), count in self.total_aggregation.items()
                ]
                summary = "\n".join(summary_lines)
                try:
                    # 전체 집계 결과를 info 레벨로 로그 출력
                    logging.getLogger().info("전체 누적 로그 집계:\n" + summary)
                except Exception:
                    pass
                self.total_aggregation.clear()  # 전체 집계 초기화
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
                self.weekly_signal_aggregation.clear()  # 주간 신호 집계 초기화

import threading  # 동시성 제어를 위한 모듈
import time       # 시간 관련 함수 제공
from logs.log_config import setup_logger, LOG_FILES_DIR  # 로거 설정 및 로그 파일 경로 임포트
from logs.state_change_manager import StateChangeManager  # 상태 변화 관리 클래스 임포트

class DynamicLogTracker:
    """
    각 state_key별로 최근 이벤트 발생 빈도를 EMA(지수이동평균)로 계산하여 저장합니다.
    
    주요 기능:
      - 특정 이벤트의 발생 간격을 측정하여 빈도를 산출
      - EMA를 통해 이벤트 빈도의 평활화(smoothing) 수행
    """
    def __init__(self, alpha=0.1, baseline=1.0):
        """
        초기화 메서드
        
        Parameters:
            alpha (float): EMA 계산 시 가중치 계수 (기본값: 0.1)
            baseline (float): 기본 빈도 기준값 (기본값: 1.0)
        
        주요 동작:
          - 각 state_key의 이벤트 빈도 계산을 위한 초기 데이터 구조(self.data) 설정
        """
        self.alpha = alpha
        self.baseline = baseline
        self.data = {}  # 각 state_key에 대해 {'last_time': timestamp, 'ema_freq': float} 저장

    def update(self, state_key, current_time):
        """
        주어진 state_key에 대해 현재 시간 기준으로 EMA 기반 이벤트 빈도를 업데이트합니다.
        
        Parameters:
            state_key (str): 이벤트를 식별하는 키
            current_time (float): 현재 시간 (타임스탬프)
        
        Returns:
            float: 업데이트된 EMA 빈도 값
        
        주요 동작:
          - 해당 state_key에 대한 이전 기록이 없으면 초기값 설정 후 0 반환
          - 이전 이벤트 발생 시간과의 간격(dt)을 계산하여 현재 빈도를 도출
          - EMA 공식을 적용하여 평활화된 빈도를 계산, 업데이트 후 반환
        """
        if state_key not in self.data:
            self.data[state_key] = {'last_time': current_time, 'ema_freq': 0.0}
            return 0.0
        else:
            last_time = self.data[state_key]['last_time']
            dt = current_time - last_time  # 이전 이벤트와의 시간 간격
            freq = 1.0 / dt if dt > 0 else 100.0  # dt가 0이면 매우 높은 빈도로 가정
            ema_old = self.data[state_key]['ema_freq']
            # EMA 계산: 새로운 빈도 = alpha * 현재 빈도 + (1 - alpha) * 이전 EMA
            ema_new = self.alpha * freq + (1 - self.alpha) * ema_old
            self.data[state_key]['ema_freq'] = ema_new
            self.data[state_key]['last_time'] = current_time  # 마지막 시간 업데이트
            return ema_new

    def get_ema(self, state_key):
        """
        지정된 state_key의 현재 EMA 빈도 값을 반환합니다.
        
        Parameters:
            state_key (str): 이벤트 식별 키
        
        Returns:
            float: 해당 state_key의 EMA 빈도 (기본값 0.0)
        """
        return self.data.get(state_key, {}).get('ema_freq', 0.0)

class LoggingUtil:
    """
    이벤트 로그 기록 시 동적 필터링을 적용합니다.
    동일 이벤트(state_key)의 중복 기록을 방지하고, EMA 기반 필터링을 통해
    중요도에 따라 로그 레벨을 조절하거나 생략합니다.
    """
    def __init__(self, module_name: str):
        """
        초기화 메서드
        
        Parameters:
            module_name (str): 로거를 식별하기 위한 모듈 이름
        
        주요 동작:
          - 지정 모듈의 로거 객체를 생성 및 초기화
          - 상태 변화 관리를 위한 StateChangeManager와 동적 로그 추적을 위한 DynamicLogTracker 초기화
        """
        self.module_name = module_name
        self.lock = threading.RLock()  # 동시 접근 제어를 위한 락
        self.logger = setup_logger(module_name)  # 모듈 전용 로거 생성
        self.state_manager = StateChangeManager()  # 상태 변화 관리 객체 생성
        self.log_tracker = DynamicLogTracker(alpha=0.1, baseline=1.0)  # 동적 로그 빈도 추적 객체 생성
    
    def log_event(self, event_message: str, state_key: str = None, importance: str = 'MEDIUM'):
        """
        이벤트 메시지를 기록하는 메서드.
        동일 state_key의 이벤트가 자주 발생하면 EMA를 통해 로그 레벨을 조절하거나 생략합니다.
        
        Parameters:
            event_message (str): 기록할 이벤트 메시지
            state_key (str, optional): 이벤트를 식별하는 키 (기본값: None)
            importance (str, optional): 이벤트의 중요도 ('LOW', 'MEDIUM', 'HIGH'; 기본 'MEDIUM')
        
        Returns:
            None: 결과는 로그로 기록되며 반환값은 없습니다.
        
        주요 동작:
          - 현재 시간을 기준으로 해당 state_key의 이벤트 빈도를 업데이트
          - 이전 상태와 동일한 경우 기록을 생략
          - 계산된 EMA 값에 따라 로그 레벨(INFO 또는 DEBUG) 결정 후 메시지 기록
        """
        with self.lock:
            current_time = time.time()
            ema = 0.0
            if state_key:
                ema = self.log_tracker.update(state_key, current_time)
            
            # 동일 state_key에 대해 상태 변화가 없으면 로그 기록 생략
            if state_key and not self.state_manager.has_changed(state_key, event_message):
                return
            
            effective_level = 'INFO'  # 기본 로그 레벨
            if ema > self.log_tracker.baseline * 2:
                # EMA가 기준치의 2배 이상이면, 중요도에 따라 로그 레벨 조정 또는 생략
                if importance.upper() == 'LOW':
                    return  # 낮은 중요도 이벤트는 생략
                elif importance.upper() == 'MEDIUM':
                    effective_level = 'DEBUG'
                else:
                    effective_level = 'INFO'
            
            msg = f"[{self.module_name}] Event: {event_message}"
            if effective_level == 'DEBUG':
                self.logger.debug(msg)
            else:
                self.logger.info(msg)
    
    @staticmethod
    def clear_log_files():
        """
        로그 파일들을 삭제하여 초기화하는 정적 메서드.
        
        주요 동작:
          - LOG_FILES_DIR 내 모든 .log 파일을 찾아 삭제 시도
          - 삭제 성공/실패 결과를 출력
        """
        import os, glob
        log_files = glob.glob(os.path.join(LOG_FILES_DIR, "*.log"))
        for log_file in log_files:
            try:
                os.remove(log_file)
                print(f"Deleted log file: {log_file}")
            except Exception as e:
                print(f"Failed to remove log file {log_file}: {e}")

class StateChangeManager:
    """
    StateChangeManager는 상태 값을 추적하며,
    이전 상태와 비교하여 의미 있는 변화가 있는지 확인합니다.
    숫자형 상태의 경우, 상대 변화율이 일정 임계값 이상일 때만 변화를 감지합니다.
    """
    def __init__(self, numeric_threshold: float = 0.01):
        """
        초기화 메서드
        
        Parameters:
            numeric_threshold (float): 숫자형 상태 값의 상대 변화 임계값 (기본값 0.01, 즉 1%)
        
        주요 동작:
          - 상태 값을 저장할 내부 딕셔너리 초기화
          - 숫자형 상태 변화 감지를 위한 임계값 설정
        """
        self._state_dict = {}  # 상태 값을 저장할 딕셔너리, key: 상태 키, value: 상태 값
        self.numeric_threshold = numeric_threshold

    def has_changed(self, key: str, new_value) -> bool:
        """
        기존 상태와 비교하여 주어진 key의 상태가 의미 있게 변경되었는지 확인합니다.
        - 숫자형 상태의 경우, 상대 변화율(또는 절대값 비교)을 통해 판단.
        - 숫자형이 아닌 경우, 단순 불일치를 확인.
        
        Parameters:
            key (str): 상태를 식별하는 키
            new_value: 새로운 상태 값 (숫자형 또는 기타)
        
        Returns:
            bool: 상태가 변경되었으면 True, 그렇지 않으면 False
        
        주요 동작:
          - 초기 상태라면 저장 후 True 반환
          - 숫자형 값이면, old_value가 0인 경우 절대 변화량 비교, 아니면 상대 변화율 비교
          - 숫자형이 아닌 경우 단순 비교 수행
        """
        old_value = self._state_dict.get(key)
        # 처음 상태인 경우
        if old_value is None:
            self._state_dict[key] = new_value
            return True

        # 숫자형 값인 경우: 상대 변화율 비교 (old_value가 0인 경우 절대 변화량으로 판단)
        if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
            if old_value == 0:
                if abs(new_value) >= self.numeric_threshold:
                    self._state_dict[key] = new_value
                    return True
                else:
                    return False
            else:
                relative_change = abs(new_value - old_value) / abs(old_value)
                if relative_change >= self.numeric_threshold:
                    self._state_dict[key] = new_value
                    return True
                else:
                    return False
        else:
            # 숫자형이 아닌 경우 단순 비교
            if old_value != new_value:
                self._state_dict[key] = new_value
                return True
            else:
                return False

    def get_state(self, key: str):
        """
        현재 저장된 상태 값을 반환합니다.
        
        Parameters:
            key (str): 상태를 식별하는 키
        
        Returns:
            해당 key에 대한 상태 값 (존재하지 않으면 None)
        """
        return self._state_dict.get(key)

    def reset_state(self, key: str = None):
        """
        상태 값을 리셋합니다.
        
        Parameters:
            key (str, optional): 특정 상태만 리셋할 경우 해당 키 지정. (지정하지 않으면 전체 상태 초기화)
        
        Returns:
            None
        """
        if key:
            if key in self._state_dict:
                del self._state_dict[key]
        else:
            self._state_dict.clear()
