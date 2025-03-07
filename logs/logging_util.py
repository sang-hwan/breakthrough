# logs/logging_util.py
import threading
import time
from logs.logger_config import setup_logger, LOG_FILES_DIR
from logs.state_change_manager import StateChangeManager

class DynamicLogTracker:
    """
    각 state_key별로 최근 이벤트 발생 빈도를 EMA(지수이동평균)로 계산하여 저장합니다.
    """
    def __init__(self, alpha=0.1, baseline=1.0):
        self.alpha = alpha
        self.baseline = baseline
        self.data = {}  # state_key -> {'last_time': timestamp, 'ema_freq': float}

    def update(self, state_key, current_time):
        if state_key not in self.data:
            self.data[state_key] = {'last_time': current_time, 'ema_freq': 0.0}
            return 0.0
        else:
            last_time = self.data[state_key]['last_time']
            dt = current_time - last_time
            freq = 1.0 / dt if dt > 0 else 100.0
            ema_old = self.data[state_key]['ema_freq']
            ema_new = self.alpha * freq + (1 - self.alpha) * ema_old
            self.data[state_key]['ema_freq'] = ema_new
            self.data[state_key]['last_time'] = current_time
            return ema_new

    def get_ema(self, state_key):
        return self.data.get(state_key, {}).get('ema_freq', 0.0)

class LoggingUtil:
    """
    이벤트 로그를 기록할 때 동적 필터링을 적용합니다.
    동일 이벤트(state_key)의 중복 기록을 방지하고, EMA 기반 필터링을 통해
    중요도에 따라 로그 레벨을 조절하거나 생략합니다.
    """
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.lock = threading.RLock()
        self.logger = setup_logger(module_name)
        self.state_manager = StateChangeManager()
        self.log_tracker = DynamicLogTracker(alpha=0.1, baseline=1.0)
    
    def log_event(self, event_message: str, state_key: str = None, importance: str = 'MEDIUM'):
        with self.lock:
            current_time = time.time()
            ema = 0.0
            if state_key:
                ema = self.log_tracker.update(state_key, current_time)
            
            if state_key and not self.state_manager.has_changed(state_key, event_message):
                return
            
            effective_level = 'INFO'
            if ema > self.log_tracker.baseline * 2:
                if importance.upper() == 'LOW':
                    return  # LOW 중요도 이벤트는 생략
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
        import os, glob
        log_files = glob.glob(os.path.join(LOG_FILES_DIR, "*.log"))
        for log_file in log_files:
            try:
                os.remove(log_file)
                print(f"Deleted log file: {log_file}")
            except Exception as e:
                print(f"Failed to remove log file {log_file}: {e}")
