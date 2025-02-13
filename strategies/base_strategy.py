# strategies/base_strategy.py
from logs.logger_config import setup_logger

class BaseStrategy:
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
    
    def get_signal(self, data, current_time, **kwargs):
        """
        각 전략이 구현해야 하는 신호 산출 메서드.
        """
        raise NotImplementedError("Subclasses must implement get_signal()")
