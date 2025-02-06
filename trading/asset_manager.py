# trading/asset_manager.py
from logs.logger_config import setup_logger

class AssetManager:
    def __init__(self, account, target_allocation=None):
        """
        target_allocation: dict, 예) {"spot": 0.7, "stablecoin": 0.3}
        """
        self.account = account
        self.logger = setup_logger(__name__)
        # 기본 목표 배분: 상승장 시 현물 비중 100%, 불안정 시 스테이블코인 비중 증가
        self.target_allocation = target_allocation or {"spot": 1.0, "stablecoin": 0.0}

    def rebalance(self, market_regime):
        """
        시장 레짐에 따라 자산 배분 목표를 조정하고, 실제 계좌 자산을 재분배합니다.
        - bullish: 현물 비중 100%
        - bearish: 스테이블코인 비중 100%
        - sideways: 50:50 혹은 동적 조정 (예시로 60% 현물, 40% 스테이블)
        """
        if market_regime.lower() == "bullish":
            target = {"spot": 1.0, "stablecoin": 0.0}
        elif market_regime.lower() == "bearish":
            target = {"spot": 0.0, "stablecoin": 1.0}
        elif market_regime.lower() == "sideways":
            target = {"spot": 0.6, "stablecoin": 0.4}
        else:
            target = self.target_allocation

        self.target_allocation = target
        total_assets = self.account.spot_balance + self.account.stablecoin_balance
        desired_spot = total_assets * target["spot"]
        desired_stable = total_assets * target["stablecoin"]

        # 만약 현물 잔고가 과도하면 스테이블코인으로 전환
        if self.account.spot_balance > desired_spot:
            amount_to_convert = self.account.spot_balance - desired_spot
            converted = self.account.convert_to_stablecoin(amount_to_convert)
            self.logger.info(f"Rebalancing: Converted {converted:.2f} from spot to stablecoin.")
        # 반대로 스테이블코인이 과도하면 현물로 전환
        elif self.account.spot_balance < desired_spot:
            amount_to_convert = desired_spot - self.account.spot_balance
            converted = self.account.convert_to_spot(amount_to_convert)
            self.logger.info(f"Rebalancing: Converted {converted:.2f} from stablecoin to spot.")
        else:
            self.logger.info("Rebalancing: No conversion needed.")

        self.logger.info(f"Post-rebalance Account: {self.account}")
