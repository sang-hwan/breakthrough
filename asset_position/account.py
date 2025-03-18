# core/account.py

# 로그 설정 함수 임포트: 로그 설정을 위한 모듈에서 setup_logger 함수를 가져옴
from logging.logger_config import setup_logger

# 모듈 이름(__name__)을 사용하여 현재 파일에 대한 로거 인스턴스 생성
logger = setup_logger(__name__)  # 전역 로거: 계좌 관련 이벤트 로깅에 사용됨


class Account:
    """
    Account 클래스는 트레이딩 계좌의 잔고 및 포지션을 관리하는 역할을 합니다.
    
    Attributes:
        initial_balance (float): 계좌 초기 잔고.
        spot_balance (float): 현재 사용 가능한 현물(Spot) 잔고.
        stablecoin_balance (float): 스테이블 코인으로 변환된 잔고.
        fee_rate (float): 거래 수수료율 (기본값 0.001, 즉 0.1%).
        positions (list): 계좌에 등록된 거래 포지션 목록.
    """
    
    def __init__(self, initial_balance: float, fee_rate: float = 0.001) -> None:
        """
        계좌를 초기화합니다.
        
        Parameters:
            initial_balance (float): 계좌 시작 금액 (음수가 아니어야 함).
            fee_rate (float): 거래 시 적용할 수수료율 (기본값 0.1%).
            
        Raises:
            ValueError: 초기 잔고가 음수일 경우.
        """
        if initial_balance < 0:
            raise ValueError("Initial balance must be non-negative.")
        # 초기 잔고와 수수료율을 객체 속성으로 설정
        self.initial_balance: float = initial_balance  # 계좌 생성 시 초기 잔고 기록
        self.spot_balance: float = initial_balance  # 현물 잔고: 거래에 즉시 사용 가능한 자금
        self.stablecoin_balance: float = 0.0  # 스테이블 코인 잔고 (초기에는 0)
        self.fee_rate: float = fee_rate  # 거래 시 적용되는 수수료율
        self.positions: list = []  # 현재 열려있는 포지션들을 저장하는 리스트
        logger.debug(f"Account initialized with balance: {initial_balance:.2f}")

    def add_position(self, position) -> None:
        """
        새로운 거래 포지션을 계좌에 추가합니다.
        
        Parameters:
            position: 포지션 객체 (반드시 position_id 속성이 있어야 함).
            
        Returns:
            None
        """
        self.positions.append(position)
        logger.debug(f"Position added: ID={position.position_id}")

    def remove_position(self, position) -> None:
        """
        계좌에서 특정 거래 포지션을 제거합니다.
        
        Parameters:
            position: 제거할 포지션 객체.
            
        Returns:
            None
        """
        if position in self.positions:
            self.positions.remove(position)
            logger.debug(f"Position removed: ID={position.position_id}")
        else:
            logger.warning(f"Failed to remove position: ID={position.position_id}")

    def get_used_balance(self) -> float:
        """
        사용 중인(잠긴) 잔고를 계산합니다.
        
        각 포지션의 미체결(미종료) 거래 기록을 순회하며, 
        '진입 가격 × 거래 수량 × (1 + fee_rate)' 값을 합산하여 사용된 자금을 산출합니다.
        
        Returns:
            float: 사용된 총 잔고.
        """
        used: float = 0.0
        # 각 포지션에 대해 열려있는(execution이 닫히지 않은) 거래의 자금을 계산
        for pos in self.positions:
            for record in pos.executions:
                if not record.get("closed", False):
                    used += record["entry_price"] * record["size"] * (1 + self.fee_rate)
        return used

    def get_available_balance(self) -> float:
        """
        새 거래를 위한 사용 가능한 잔고(실제 잔고에서 사용 중인 금액을 제외한 금액)를 계산합니다.
        
        Returns:
            float: 사용 가능한 잔고 (음수가 되지 않도록 0.0 이상).
        """
        available = self.spot_balance - self.get_used_balance()
        return available if available >= 0 else 0.0

    def update_after_trade(self, trade: dict) -> None:
        """
        거래 체결 후 계좌의 잔고를 업데이트합니다.
        
        Parameters:
            trade (dict): 거래 세부 정보를 담은 딕셔너리. 'pnl' (손익) 항목을 포함해야 함.
            
        Returns:
            None
        """
        pnl = trade.get("pnl", 0.0)
        # 거래 손익(PnL)에 따라 현물 잔고를 증가 또는 감소시킴
        self.spot_balance += pnl
        logger.debug(f"Trade executed: PnL={pnl:.2f}, Updated spot balance={self.spot_balance:.2f}")

    def convert_to_stablecoin(self, amount: float, conversion_fee: float = 0.001) -> float:
        """
        현물 잔고의 일부를 스테이블 코인으로 변환합니다.
        
        변환 시 수수료를 차감하며, 변환 요청 금액이 사용 가능한 잔고를 초과하면 사용 가능한 최대 금액으로 변환합니다.
        
        Parameters:
            amount (float): 변환할 금액.
            conversion_fee (float): 변환에 적용되는 수수료율 (기본값 0.1%).
            
        Returns:
            float: 수수료 차감 후 스테이블 코인으로 변환된 순 금액.
        """
        if amount <= 0:
            logger.error("Conversion amount must be positive.", exc_info=True)
            return 0.0
        available = self.get_available_balance()
        # 요청 금액이 사용 가능한 잔고보다 크면, 사용 가능한 최대 금액으로 대체
        if amount > available:
            amount = available
        fee = amount * conversion_fee  # 변환 수수료 계산
        net_amount = amount - fee  # 수수료 차감 후 순 금액
        # 잔고 업데이트: 현물 잔고에서 차감하고 스테이블 코인 잔고에 추가
        self.spot_balance -= amount
        self.stablecoin_balance += net_amount
        logger.debug(f"Converted {amount:.2f} from spot to stablecoin (fee {fee:.2f}, net {net_amount:.2f}).")
        return net_amount

    def convert_to_spot(self, amount: float, conversion_fee: float = 0.001) -> float:
        """
        스테이블 코인 잔고의 일부를 다시 현물 잔고로 변환합니다.
        
        변환 시 수수료를 차감하며, 요청 금액이 스테이블 코인 잔고를 초과할 경우 잔고 전체를 변환합니다.
        
        Parameters:
            amount (float): 변환할 금액.
            conversion_fee (float): 변환에 적용되는 수수료율 (기본값 0.1%).
            
        Returns:
            float: 수수료 차감 후 현물 잔고로 변환된 순 금액.
        """
        if amount <= 0:
            logger.error("Conversion amount must be positive.", exc_info=True)
            return 0.0
        # 요청 금액이 스테이블 코인 잔고보다 클 경우 잔고 전체로 조정
        if amount > self.stablecoin_balance:
            amount = self.stablecoin_balance
        fee = amount * conversion_fee  # 변환 수수료 계산
        net_amount = amount - fee  # 수수료 차감 후 순 금액
        # 잔고 업데이트: 스테이블 코인 잔고에서 차감하고 현물 잔고에 추가
        self.stablecoin_balance -= amount
        self.spot_balance += net_amount
        logger.debug(f"Converted {amount:.2f} from stablecoin to spot (fee {fee:.2f}, net {net_amount:.2f}).")
        return net_amount

    def __str__(self) -> str:
        """
        계좌 상태를 문자열로 표현합니다.
        
        Returns:
            str: 현물 잔고, 스테이블 코인 잔고, 사용 가능한 잔고 정보를 포함한 포맷 문자열.
        """
        return (
            f"Account(spot_balance={self.spot_balance:.2f}, "
            f"stablecoin_balance={self.stablecoin_balance:.2f}, "
            f"available_balance={self.get_available_balance():.2f})"
        )
