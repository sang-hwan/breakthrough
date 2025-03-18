# logs/logging_util.py
import threading  # 동시성 제어를 위한 모듈
import time       # 시간 관련 함수 제공
from logging.logger_config import setup_logger, LOG_FILES_DIR  # 로거 설정 및 로그 파일 경로 임포트
from logging.state_change_manager import StateChangeManager  # 상태 변화 관리 클래스 임포트

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
