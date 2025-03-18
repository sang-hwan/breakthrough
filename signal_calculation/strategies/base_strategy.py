# strategies/base_strategy.py
# 이 파일은 모든 트레이딩 전략 클래스들이 상속할 기본 전략(BaseStrategy)을 정의합니다.
# 각 전략 클래스는 get_signal() 메서드를 구현하여 거래 신호를 생성해야 합니다.

from logging.logger_config import setup_logger  # 로깅 설정을 위한 함수 임포트

class BaseStrategy:
    def __init__(self):
        """
        기본 전략 클래스의 생성자.
        
        목적:
          - 자식 클래스에서 사용할 로거(logger) 객체를 초기화합니다.
        
        동작:
          - 클래스 이름을 이용해 로거를 설정함으로써, 로그 메시지에 전략 이름을 포함시킵니다.
        """
        self.logger = setup_logger(self.__class__.__name__)
    
    def get_signal(self, data, current_time, **kwargs):
        """
        거래 신호를 생성하기 위한 추상 메서드.
        
        Parameters:
            data (pandas.DataFrame): 거래 데이터 (예: OHLCV 데이터 등).
            current_time (datetime): 거래 신호를 생성할 기준 시점.
            **kwargs: 추가 인자들.
        
        Returns:
            str: 거래 신호 (예: "enter_long", "exit_all", "hold").
        
        주의:
            - 이 메서드는 구현되어 있지 않으므로 반드시 자식 클래스에서 오버라이딩 해야 합니다.
        """
        raise NotImplementedError("Subclasses must implement get_signal()")
