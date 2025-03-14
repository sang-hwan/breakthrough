# trading/asset_manager.py

# 모듈 및 라이브러리 임포트:
# - setup_logger: 로깅 설정을 위한 함수
# - timedelta: 시간 간격 계산을 위한 datetime 모듈의 클래스
# - pandas: 데이터 처리 및 시간 관련 기능 제공
# - LoggingUtil: 추가 로깅 유틸리티
from logs.logger_config import setup_logger
from datetime import timedelta
import pandas as pd
from logs.logging_util import LoggingUtil

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
