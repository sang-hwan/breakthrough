# trading/asset_manager.py
from logs.logger_config import setup_logger
from datetime import datetime, timedelta

class AssetManager:
    def __init__(self, account, min_rebalance_threshold=0.05, min_rebalance_interval_minutes=60):
        """
        자산 배분 및 리밸런싱을 관리합니다.
        - min_rebalance_threshold: 리밸런싱 실행을 위한 최소 배분 차이 (전체 자산 대비)
        - min_rebalance_interval_minutes: 리밸런싱 최소 간격
        """
        self.account = account
        self.logger = setup_logger("asset_manager")
        self.min_rebalance_threshold = min_rebalance_threshold
        self.min_rebalance_interval = timedelta(minutes=min_rebalance_interval_minutes)
        self.last_rebalance_time = None
        self.last_account_state = None  # (spot_balance, stablecoin_balance)

    def _get_account_state(self):
        return (round(self.account.spot_balance, 4), round(self.account.stablecoin_balance, 4))

    def rebalance(self, market_regime):
        """
        시장 레짐에 따라 자산 배분을 재조정합니다.
          - bullish: 목표 배분 90% spot
          - bearish: 목표 배분 10% spot
          - sideways: 목표 배분 60% spot
        현재 배분과 목표 배분의 차이가 전체 자산 대비 최소 임계치 이상일 경우에만 전환을 실행합니다.
        """
        current_time = datetime.now()

        if self.last_rebalance_time is not None:
            elapsed = current_time - self.last_rebalance_time
            if elapsed < self.min_rebalance_interval:
                self.logger.debug(
                    f"리밸런싱 건너뜀: 마지막 리밸런싱 후 {elapsed.total_seconds()/60:.2f}분 경과 (최소 {self.min_rebalance_interval.total_seconds()/60:.2f}분 필요)."
                )
                return

        total_assets = self.account.spot_balance + self.account.stablecoin_balance
        if total_assets <= 0:
            self.logger.warning("총 자산이 0 이하입니다. 리밸런싱을 건너뜁니다.")
            return

        regime = market_regime.lower()
        if regime == "bullish":
            desired_spot = total_assets * 0.90
        elif regime == "bearish":
            desired_spot = total_assets * 0.10
        elif regime == "sideways":
            desired_spot = total_assets * 0.60
        else:
            self.logger.warning(f"알 수 없는 시장 레짐: {market_regime}. 리밸런싱 건너뜀.")
            return

        current_spot = self.account.spot_balance
        diff_ratio = abs(current_spot - desired_spot) / total_assets

        if diff_ratio < self.min_rebalance_threshold:
            self.logger.info("자산 배분 차이가 임계치 미만입니다. 리밸런싱 건너뜁니다.")
            return

        # 자산 전환 수행
        if current_spot < desired_spot:
            amount_to_convert = desired_spot - current_spot
            converted = self.account.convert_to_spot(amount_to_convert)
            self.logger.info(f"[{market_regime.capitalize()}] {converted:.2f} 만큼 스테이블코인에서 현물로 전환됨.")
        else:
            amount_to_convert = current_spot - desired_spot
            converted = self.account.convert_to_stablecoin(amount_to_convert)
            self.logger.info(f"[{market_regime.capitalize()}] {converted:.2f} 만큼 현물에서 스테이블코인으로 전환됨.")

        self.last_rebalance_time = current_time

        current_state = self._get_account_state()
        if current_state != self.last_account_state:
            self.last_account_state = current_state
            self.logger.info(f"리밸런싱 후 계좌 상태: {self.account}")
        else:
            self.logger.debug("리밸런싱 후 계좌 상태 변화 없음.")
