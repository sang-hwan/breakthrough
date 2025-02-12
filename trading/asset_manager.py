# trading/asset_manager.py
from logs.logger_config import setup_logger
from datetime import datetime, timedelta

class AssetManager:
    def __init__(self, account, min_rebalance_threshold=0.05, min_rebalance_interval_minutes=60):
        """
        자산 배분 및 리밸런싱을 관리합니다.
        - min_rebalance_threshold: 리밸런싱 실행을 위한 최소 배분 차이 (전체 자산 대비)
        - min_rebalance_interval_minutes: 리밸런싱 최소 간격 (분)
        """
        self.account = account
        self.logger = setup_logger(__name__)
        self.min_rebalance_threshold = min_rebalance_threshold
        self.min_rebalance_interval = timedelta(minutes=min_rebalance_interval_minutes)
        self.last_rebalance_time = None
        self.last_account_state = None  # (spot_balance, stablecoin_balance)
        # INFO 레벨 로그로 남겨 AggregatingHandler가 집계하도록 함.
        self.logger.debug(f"AssetManager 초기화: 최소 임계치={min_rebalance_threshold}, 최소 간격={min_rebalance_interval_minutes}분")

    def _get_account_state(self):
        state = (round(self.account.spot_balance, 4), round(self.account.stablecoin_balance, 4))
        self.logger.debug(f"현재 계좌 상태: spot_balance={state[0]}, stablecoin_balance={state[1]}")
        return state

    def rebalance(self, market_regime):
        """
        시장 레짐에 따라 자산 배분을 재조정합니다.
        """
        current_time = datetime.now()
        self.logger.debug(f"리밸런싱 시작 시각: {current_time}")
        
        # 리밸런싱 전 계좌 상태 로깅 추가
        pre_state = self._get_account_state()
        self.logger.debug(f"리밸런싱 전 계좌 상태: spot_balance={pre_state[0]}, stablecoin_balance={pre_state[1]}")

        if self.last_rebalance_time is not None:
            elapsed = current_time - self.last_rebalance_time
            if elapsed < self.min_rebalance_interval:
                self.logger.debug(f"리밸런싱 최소 간격 미충족: 경과 시간 {elapsed} (최소 간격: {self.min_rebalance_interval}).")
                return

        total_assets = self.account.spot_balance + self.account.stablecoin_balance
        self.logger.debug(
            f"총 자산 계산: spot_balance={self.account.spot_balance:.2f}, "
            f"stablecoin_balance={self.account.stablecoin_balance:.2f}, total_assets={total_assets:.2f}"
        )
        if total_assets <= 0:
            self.logger.warning("총 자산이 0 이하입니다. 리밸런싱을 건너뜁니다.")
            return

        regime = market_regime.lower()
        self.logger.debug(f"시장 레짐: {regime}")
        if regime == "bullish":
            desired_spot = total_assets * 0.90
        elif regime == "bearish":
            desired_spot = total_assets * 0.10
        elif regime == "sideways":
            desired_spot = total_assets * 0.60
        else:
            self.logger.warning(f"알 수 없는 시장 레짐: {market_regime}. 리밸런싱을 건너뜁니다.")
            return

        self.logger.debug(f"목표 현물 자산(desired_spot): {desired_spot:.2f}")
        current_spot = self.account.spot_balance
        self.logger.debug(f"현재 현물 자산(current_spot): {current_spot:.2f}")
        diff_ratio = abs(current_spot - desired_spot) / total_assets
        self.logger.debug(f"배분 차이 비율: {diff_ratio:.4f} (임계치: {self.min_rebalance_threshold})")

        if diff_ratio < self.min_rebalance_threshold:
            self.logger.debug("자산 배분 차이가 임계치 미만입니다. 리밸런싱을 실행하지 않습니다.")
            return

        # 리밸런싱 실행
        if current_spot < desired_spot:
            amount_to_convert = desired_spot - current_spot
            self.logger.debug(f"스테이블코인 -> 현물 전환 필요량: {amount_to_convert:.2f}")
            converted = self.account.convert_to_spot(amount_to_convert)
            self.logger.debug(f"[{market_regime.capitalize()}] 리밸런싱 실행: 스테이블코인 -> 현물 전환 {converted:.2f}")
        else:
            amount_to_convert = current_spot - desired_spot
            self.logger.debug(f"현물 -> 스테이블코인 전환 필요량: {amount_to_convert:.2f}")
            converted = self.account.convert_to_stablecoin(amount_to_convert)
            self.logger.debug(f"[{market_regime.capitalize()}] 리밸런싱 실행: 현물 -> 스테이블코인 전환 {converted:.2f}")

        self.last_rebalance_time = current_time

        # 리밸런싱 완료 후 계좌 상태 로깅
        current_state = self._get_account_state()
        if current_state != self.last_account_state:
            self.last_account_state = current_state
            self.logger.debug(f"리밸런싱 완료 후 계좌 상태: {self.account}")
        else:
            self.logger.debug("리밸런싱 완료: 계좌 상태 변화 없음.")
