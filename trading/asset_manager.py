# trading/asset_manager.py
from logs.logger_config import setup_logger
from datetime import timedelta
import pandas as pd

class AssetManager:
    _instances = {}

    def __new__(cls, account, min_rebalance_threshold=0.05, min_rebalance_interval_minutes=60):
        key = (id(account), min_rebalance_threshold, min_rebalance_interval_minutes)
        if key not in cls._instances:
            instance = super(AssetManager, cls).__new__(cls)
            cls._instances[key] = instance
        return cls._instances[key]

    def __init__(self, account, min_rebalance_threshold=0.05, min_rebalance_interval_minutes=60):
        if hasattr(self, '_initialized') and self._initialized:
            return

        if account is None:
            raise ValueError("Account must not be None.")
        self.account = account
        self.logger = setup_logger(__name__)
        self.min_rebalance_threshold = min_rebalance_threshold
        self.min_rebalance_interval = timedelta(minutes=min_rebalance_interval_minutes)
        self.last_rebalance_time = None
        self.last_account_state = None
        self.logger.debug(
            f"AssetManager initialized with threshold {min_rebalance_threshold} and interval {min_rebalance_interval_minutes} min"
        )
        self._initialized = True

    def _get_account_state(self):
        return (round(self.account.spot_balance, 4), round(self.account.stablecoin_balance, 4))

    def rebalance(self, market_regime):
        # 현재 시간: Asia/Seoul 기준
        current_time = pd.Timestamp.now()
        if self.last_rebalance_time and (current_time - self.last_rebalance_time < self.min_rebalance_interval):
            self.logger.debug("Rebalance skipped due to interval constraint.")
            return

        total_assets = self.account.spot_balance + self.account.stablecoin_balance
        if total_assets <= 0:
            self.logger.warning("Total assets <= 0. Skipping rebalance.")
            return

        regime = market_regime.lower()
        if regime in ["bullish", "enter_long"]:
            desired_spot = total_assets * (1.0 if regime == "enter_long" else 0.90)
        elif regime in ["bearish", "exit_all"]:
            desired_spot = total_assets * (0.0 if regime == "exit_all" else 0.10)
        elif regime == "sideways":
            desired_spot = total_assets * 0.60
        else:
            self.logger.warning(f"Unknown market regime: {market_regime}. Skipping rebalance.")
            return

        current_spot = self.account.spot_balance
        diff_ratio = abs(current_spot - desired_spot) / total_assets
        if diff_ratio < self.min_rebalance_threshold:
            self.logger.debug("No significant imbalance detected; skipping rebalance.")
            return

        try:
            if current_spot < desired_spot:
                amount_to_convert = desired_spot - current_spot
                converted = self.account.convert_to_spot(amount_to_convert)
                self.logger.debug(f"Rebalance ({market_regime.capitalize()}): Converted {converted:.2f} from stablecoin to spot.")
            else:
                amount_to_convert = current_spot - desired_spot
                converted = self.account.convert_to_stablecoin(amount_to_convert)
                self.logger.debug(f"Rebalance ({market_regime.capitalize()}): Converted {converted:.2f} from spot to stablecoin.")
        except Exception as e:
            self.logger.error(f"Rebalance conversion failed: {e}", exc_info=True)
            return

        self.last_rebalance_time = current_time
        new_state = self._get_account_state()
        if new_state != self.last_account_state:
            self.last_account_state = new_state
            self.logger.debug(f"Rebalance complete. New account state: {self.account}")
