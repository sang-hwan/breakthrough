# trading/position_management.py

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

# 모듈 및 라이브러리 임포트:
# - setup_logger: 로깅 설정을 위한 함수
# - timedelta: 시간 간격 계산을 위한 datetime 모듈의 클래스
# - pandas: 데이터 처리 및 시간 관련 기능 제공
# - LoggingUtil: 추가 로깅 유틸리티
from logging.logger_config import setup_logger
from datetime import timedelta
import pandas as pd
from logging.logging_util import LoggingUtil

# AssetManager 클래스는 계좌의 자산 배분(스팟 vs. 스테이블코인)을 재조정(리밸런싱)하는 역할을 합니다.
class AssetManager:
    # _instances: 특정 파라미터 조합에 대해 단일 인스턴스를 유지하기 위한 클래스 변수 (싱글톤 패턴)
    _instances = {}

    def __new__(cls, account, min_rebalance_threshold=0.05, min_rebalance_interval_minutes=60):
        """
        객체 생성 시, account와 재조정 임계치, 최소 재조정 간격에 따라 고유한 인스턴스를 반환합니다.
        
        Parameters:
            account (object): 자산 정보를 포함한 계좌 객체.
            min_rebalance_threshold (float): 재조정을 위한 최소 비율 차이.
            min_rebalance_interval_minutes (int): 재조정 간 최소 시간 간격(분).
        
        Returns:
            AssetManager 인스턴스 (싱글톤 패턴 적용).
        """
        # 계좌의 고유 id와 임계치, 간격을 기준으로 고유 키 생성
        key = (id(account), min_rebalance_threshold, min_rebalance_interval_minutes)
        if key not in cls._instances:
            # 아직 인스턴스가 없으면 새로 생성 후 저장
            instance = super(AssetManager, cls).__new__(cls)
            cls._instances[key] = instance
        return cls._instances[key]

    def __init__(self, account, min_rebalance_threshold=0.05, min_rebalance_interval_minutes=60):
        """
        AssetManager 인스턴스를 초기화합니다.
        
        Parameters:
            account (object): 거래 계좌 객체로, 스팟 및 스테이블코인 잔고와 변환 메소드를 포함.
            min_rebalance_threshold (float): 리밸런싱을 실행하기 위한 최소 잔고 차이 비율.
            min_rebalance_interval_minutes (int): 연속 리밸런싱 간 최소 시간 간격(분).
        
        Returns:
            None
        """
        # 이미 초기화된 인스턴스는 재초기화를 방지 (싱글톤 패턴)
        if hasattr(self, '_initialized') and self._initialized:
            return

        # 계좌 객체가 None이면 에러 발생
        if account is None:
            raise ValueError("Account must not be None.")
        self.account = account
        # 모듈 전반에 걸쳐 로깅을 사용하기 위한 로거 설정
        self.logger = setup_logger(__name__)
        # 추가적인 로깅 유틸리티 객체 생성
        self.log_util = LoggingUtil(__name__)
        # 재조정 임계치와 최소 재조정 간격 설정
        self.min_rebalance_threshold = min_rebalance_threshold
        self.min_rebalance_interval = timedelta(minutes=min_rebalance_interval_minutes)
        # 마지막 재조정 시간과 마지막 계좌 상태를 추적하기 위한 변수 초기화
        self.last_rebalance_time = None
        self.last_account_state = None
        # 초기화 정보 디버그 로그 기록
        self.logger.debug(
            f"AssetManager initialized with threshold {min_rebalance_threshold} and interval {min_rebalance_interval_minutes} min"
        )
        self._initialized = True

    def _get_account_state(self):
        """
        현재 계좌의 상태(스팟 잔고와 스테이블코인 잔고)를 소수점 4자리로 반올림하여 반환합니다.
        
        Returns:
            tuple: (spot_balance, stablecoin_balance)
        """
        return (round(self.account.spot_balance, 4), round(self.account.stablecoin_balance, 4))

    def rebalance(self, market_regime):
        """
        계좌의 자산 배분을 현재 시장 상황(market_regime)에 따라 재조정합니다.
        
        1. 최근 재조정 시간과 최소 재조정 간격을 비교하여 재조정 실행 여부를 결정.
        2. 총 자산(스팟+스테이블코인)과 현재 스팟 비중을 기준으로 목표 스팟 비중(desired_spot)을 산정.
        3. 목표와 실제 간의 차이가 임계치(min_rebalance_threshold)를 초과하면 자산 변환 실행.
           - 스팟 잔고가 부족하면 스테이블코인을 스팟으로 변환.
           - 과다하면 스팟을 스테이블코인으로 변환.
        4. 변환 후 계좌 상태를 업데이트하고 이벤트 로그를 남깁니다.
        
        Parameters:
            market_regime (str or numeric): 현재 시장 상태를 나타내며 "bullish", "bearish", "sideways" 중 하나를 기대.
        
        Returns:
            None
        """
        # 현재 시간을 타임스탬프로 기록
        current_time = pd.Timestamp.now()
        # 마지막 재조정 시간과의 간격이 최소 재조정 간격보다 짧으면 재조정 스킵
        if self.last_rebalance_time and (current_time - self.last_rebalance_time < self.min_rebalance_interval):
            self.logger.debug("Rebalance skipped due to interval constraint.")
            return

        # 총 자산 계산: 스팟 잔고와 스테이블코인 잔고의 합
        total_assets = self.account.spot_balance + self.account.stablecoin_balance
        if total_assets <= 0:
            self.logger.warning("Total assets <= 0. Skipping rebalance.")
            return

        # market_regime이 문자열이 아닌 경우, 숫자를 문자열로 매핑 (예: 0.0 -> "bullish")
        if not isinstance(market_regime, str):
            try:
                market_regime = {0.0: "bullish", 1.0: "bearish", 2.0: "sideways"}.get(float(market_regime), "unknown")
            except Exception:
                market_regime = "unknown"
        regime = market_regime.lower()
        if regime not in ["bullish", "bearish", "sideways"]:
            self.logger.warning(f"Market regime '{market_regime}' is unknown; treating as 'sideways'.")
            regime = "sideways"

        # 시장 상태에 따라 목표 스팟 비중 계산:
        # - bullish (또는 enter_long): 스팟 비중 90% 또는 100%
        # - bearish (또는 exit_all): 스팟 비중 10% 또는 0%
        # - sideways: 스팟 비중 60%
        if regime in ["bullish", "enter_long"]:
            desired_spot = total_assets * (1.0 if regime == "enter_long" else 0.90)
        elif regime in ["bearish", "exit_all"]:
            desired_spot = total_assets * (0.0 if regime == "exit_all" else 0.10)
        elif regime == "sideways":
            desired_spot = total_assets * 0.60

        # 현재 스팟 잔고와 목표 스팟 잔고의 차이를 계산
        current_spot = self.account.spot_balance
        diff_ratio = abs(current_spot - desired_spot) / total_assets
        # 차이가 임계치보다 작으면 재조정 불필요
        if diff_ratio < self.min_rebalance_threshold:
            self.logger.debug("No significant imbalance detected; skipping rebalance.")
            return

        try:
            if current_spot < desired_spot:
                # 스팟 잔고 부족 시: 스테이블코인을 스팟으로 변환할 금액 산정 후 변환 실행
                amount_to_convert = desired_spot - current_spot
                converted = self.account.convert_to_spot(amount_to_convert)
                self.logger.debug(f"Rebalance ({regime.capitalize()}): Converted {converted:.2f} from stablecoin to spot.")
            else:
                # 과잉 시: 스팟 잔고 일부를 스테이블코인으로 변환
                amount_to_convert = current_spot - desired_spot
                converted = self.account.convert_to_stablecoin(amount_to_convert)
                self.logger.debug(f"Rebalance ({regime.capitalize()}): Converted {converted:.2f} from spot to stablecoin.")
        except Exception as e:
            # 변환 중 예외 발생 시 에러 로그 기록
            self.logger.error(f"Rebalance conversion failed: {e}", exc_info=True)
            return

        # 재조정 완료 후 마지막 재조정 시간 업데이트
        self.last_rebalance_time = current_time
        # 계좌 상태 업데이트 후, 이전 상태와 다르면 이벤트 로그 기록
        new_state = self._get_account_state()
        if new_state != self.last_account_state:
            self.last_account_state = new_state
            self.log_util.log_event("Rebalance complete", state_key="asset_state")

import uuid  # 포지션 식별자 생성을 위한 uuid 모듈 임포트
from logging.logger_config import setup_logger  # 로그 설정 함수를 임포트하여 로깅 설정

# 모듈 이름(__name__)을 기반으로 로거 인스턴스 생성
logger = setup_logger(__name__)  # 전역 로거: 포지션 관련 이벤트 로깅에 사용됨


class Position:
    """
    Position 클래스는 개별 거래 포지션을 관리합니다.
    
    Attributes:
        position_id (str): 포지션의 고유 식별자.
        side (str): 포지션의 방향 ("LONG" 또는 "SHORT").
        executions (list): 해당 포지션에서 발생한 거래 실행 기록 목록.
        initial_price (float): 포지션이 시작될 때의 진입 가격.
        maximum_size (float): 포지션의 최대 허용 크기.
        total_splits (int): 포지션 진입 시 계획된 분할 수.
        executed_splits (int): 실제 실행된 분할 수.
        allocation_plan (list): 자금 배분 계획.
        highest_price (float) 또는 lowest_price (float): 포지션의 진행 상황에 따른 최고 또는 최저 가격.
    """
    
    def __init__(self, side: str = "LONG", initial_price: float = None, maximum_size: float = 0.0,
                 total_splits: int = 1, allocation_plan: list = None) -> None:
        """
        새로운 거래 포지션을 초기화합니다.
        
        Parameters:
            side (str): 거래 방향 ("LONG"은 매수, "SHORT"는 매도). 기본값은 "LONG".
            initial_price (float): 포지션 진입 시 가격. 반드시 양수여야 함.
            maximum_size (float): 포지션이 가질 수 있는 최대 크기.
            total_splits (int): 포지션을 분할해 진입할 총 횟수.
            allocation_plan (list): 자금 배분 전략을 담은 리스트 (선택 사항).
            
        Raises:
            ValueError: initial_price가 제공되지 않았거나 0 이하인 경우.
        """
        if initial_price is None or initial_price <= 0:
            raise ValueError("Initial price must be positive.")
        # 고유 포지션 식별자 생성 (UUID 사용)
        self.position_id: str = str(uuid.uuid4())
        # 거래 방향을 대문자로 변환하여 저장 (예: "LONG", "SHORT")
        self.side: str = side.upper()
        # 거래 실행 기록을 저장할 리스트 초기화
        self.executions: list = []
        self.initial_price: float = initial_price
        self.maximum_size: float = maximum_size
        self.total_splits: int = total_splits
        self.executed_splits: int = 0
        # 배분 계획이 제공되지 않으면 빈 리스트로 초기화
        self.allocation_plan: list = allocation_plan if allocation_plan is not None else []
        # LONG 포지션은 진입 후 최고가를, SHORT 포지션은 최저가를 추적
        if self.side == "SHORT":
            self.lowest_price: float = initial_price
        else:
            self.highest_price: float = initial_price
        logger.debug(f"New position created: ID={self.position_id}, side={self.side}, entry price={self.initial_price}")

    def add_execution(self, entry_price: float, size: float, stop_loss: float = None,
                      take_profit: float = None, entry_time=None, exit_targets: list = None,
                      trade_type: str = "unknown", min_order_size: float = 1e-8) -> None:
        """
        포지션에 새로운 거래 실행 기록을 추가합니다.
        
        각 실행은 포지션 내에서 개별 거래를 나타내며, 진입 가격, 거래 수량, 
        손절/익절 가격, 거래 유형 등 다양한 거래 정보를 포함합니다.
        
        Parameters:
            entry_price (float): 거래 실행 시의 진입 가격.
            size (float): 거래 수량.
            stop_loss (float): (선택) 손절 가격.
            take_profit (float): (선택) 익절 가격.
            entry_time: (선택) 거래 실행 시각.
            exit_targets (list): (선택) 부분 청산 목표를 나타내는 (목표 가격, 청산 비율) 튜플의 리스트.
            trade_type (str): 거래 유형을 설명하는 문자열 (기본값 "unknown").
            min_order_size (float): 최소 허용 거래 수량 (거래 실행 검증용, 기본값 매우 작은 수).
            
        Returns:
            None
        
        Notes:
            - 거래 수량이 최소 주문 크기보다 작으면 실행이 추가되지 않습니다.
            - exit_targets는 리스트 형식이어야 하며, 각 목표는 딕셔너리 형태로 변환되어 저장됩니다.
            - LONG 포지션은 진입 이후 최고가, SHORT 포지션은 최저가를 추적합니다.
        """
        # 최소 주문 수량 이하일 경우 실행 추가하지 않고 경고 로그 출력
        if size < min_order_size:
            logger.warning("Execution size below minimum order size; execution not added.")
            return
        # exit_targets가 제공되었을 때 리스트 타입이 아니면 오류 로그 출력 후 종료
        if exit_targets and not isinstance(exit_targets, list):
            logger.error("exit_targets must be a list.", exc_info=True)
            return

        targets = []
        # exit_targets가 존재하면, 각 목표 가격과 청산 비율을 포함하는 딕셔너리 생성
        if exit_targets:
            for target_price, exit_ratio in exit_targets:
                targets.append({'price': target_price, 'exit_ratio': exit_ratio, 'hit': False})
        # 실행 기록을 딕셔너리 형태로 생성하여 필요한 모든 정보를 포함
        execution = {
            'entry_price': entry_price,
            'size': size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': entry_time,
            'exit_targets': targets,
            'trade_type': trade_type,
            'closed': False
        }
        # 포지션 방향에 따라 실행 후 추적할 극단값(최고가 또는 최저가) 초기화
        if self.side == "SHORT":
            execution["lowest_price_since_entry"] = entry_price
        else:
            execution["highest_price_since_entry"] = entry_price

        # 생성된 실행 기록을 포지션의 executions 리스트에 추가
        self.executions.append(execution)
        logger.debug(f"Execution added: entry_price={entry_price}, size={size}, type={trade_type}")

    def update_extremum(self, current_price: float) -> None:
        """
        모든 열려 있는 거래 실행에 대해, 현재 시장 가격을 반영하여 최고가 또는 최저가 값을 업데이트합니다.
        
        Parameters:
            current_price (float): 최신 시장 가격.
            
        Returns:
            None
        
        Notes:
            - LONG 포지션의 경우, 현재 가격이 이전 최고가보다 높으면 최고가를 갱신합니다.
            - SHORT 포지션의 경우, 현재 가격이 이전 최저가보다 낮으면 최저가를 갱신합니다.
        """
        for record in self.executions:
            if record.get("closed", False):
                continue  # 이미 종료된 실행은 건너뜁니다.
            if self.side == "LONG":
                # LONG 포지션: 이전 최고가와 비교하여 갱신
                prev = record.get("highest_price_since_entry", record["entry_price"])
                if current_price > prev:
                    record["highest_price_since_entry"] = current_price
                    logger.debug(f"Updated highest price: {prev} -> {current_price} for execution at entry {record['entry_price']}")
            elif self.side == "SHORT":
                # SHORT 포지션: 이전 최저가와 비교하여 갱신
                prev = record.get("lowest_price_since_entry", record["entry_price"])
                if current_price < prev:
                    record["lowest_price_since_entry"] = current_price
                    logger.debug(f"Updated lowest price: {prev} -> {current_price} for execution at entry {record['entry_price']}")
        logger.debug(f"Extremum values updated with current_price={current_price}")

    def get_total_size(self) -> float:
        """
        포지션 내 모든 열려 있는 실행의 총 거래 수량을 계산합니다.
        
        Returns:
            float: 미종료 실행들의 수량 합계.
        """
        return sum(record['size'] for record in self.executions if not record.get("closed", False))

    def get_average_entry_price(self) -> float:
        """
        포지션 내 모든 열려 있는 실행의 가중 평균 진입 가격을 계산합니다.
        
        가중치는 각 실행의 거래 수량에 비례합니다.
        
        Returns:
            float: 가중 평균 진입 가격. 열려 있는 실행이 없으면 0.0을 반환.
        """
        total_cost = sum(record['entry_price'] * record['size'] for record in self.executions if not record.get("closed", False))
        total_qty = self.get_total_size()
        return total_cost / total_qty if total_qty > 0 else 0.0

    def remove_execution(self, index: int) -> None:
        """
        지정한 인덱스의 실행 기록을 포지션에서 제거합니다.
        
        Parameters:
            index (int): 제거할 실행 기록의 인덱스.
            
        Returns:
            None
        
        Notes:
            - 인덱스가 유효하지 않으면 경고 로그를 남깁니다.
        """
        if 0 <= index < len(self.executions):
            self.executions.pop(index)
            logger.debug(f"Execution removed at index {index}")
        else:
            logger.warning(f"Failed to remove execution: invalid index {index}")

    def is_empty(self) -> bool:
        """
        포지션 내 모든 실행이 종료되었는지 여부를 확인합니다.
        
        Returns:
            bool: 모든 실행이 종료되었다면 True, 그렇지 않으면 False.
        """
        return all(record.get("closed", False) for record in self.executions)

    def partial_close_execution(self, index: int, close_ratio: float, min_order_size: float = 1e-8) -> float:
        """
        포지션의 특정 실행을 부분적으로 청산하여 거래 수량을 줄입니다.
        
        Parameters:
            index (int): 청산할 실행 기록의 인덱스.
            close_ratio (float): 청산할 비율 (0보다 크고 1 이하).
            min_order_size (float): 남은 거래 수량이 이 값보다 작으면 해당 실행을 종료 처리합니다.
            
        Returns:
            float: 청산된 거래 수량. 청산에 실패하면 0.0 반환.
        
        Notes:
            - close_ratio가 유효한 범위(0,1]가 아니면 오류 로그를 남깁니다.
            - 실행 후 남은 거래 수량이 최소 주문 크기보다 작으면 해당 실행을 종료 상태로 표시합니다.
        """
        if not (0 < close_ratio <= 1):
            logger.error("close_ratio must be between 0 and 1.", exc_info=True)
            return 0.0
        if 0 <= index < len(self.executions):
            record = self.executions[index]
            # 청산할 거래 수량 계산 (현재 수량의 close_ratio 비율)
            qty_to_close = record['size'] * close_ratio
            # 거래 수량 감소 처리
            record['size'] -= qty_to_close
            # exit_targets가 있다면, 청산 처리된 타겟은 제거
            if record.get('exit_targets'):
                record['exit_targets'] = [t for t in record['exit_targets'] if not t.get('hit', False)]
            # 남은 거래 수량이 최소 주문 크기보다 작으면 해당 실행을 종료로 처리
            if record['size'] < min_order_size:
                record['closed'] = True
                logger.debug(f"Execution at index {index} closed due to size below minimum order size.")
            logger.debug(f"Partial close executed: index={index}, ratio={close_ratio}, closed qty={qty_to_close}")
            return qty_to_close
        logger.warning(f"Partial close failed: invalid index {index}")
        return 0.0
