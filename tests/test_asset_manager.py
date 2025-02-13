# tests/test_asset_manager.py
import pytest
from core.account import Account
from trading.asset_manager import AssetManager

def test_rebalance_bullish():
    account = Account(initial_balance=10000)
    # 가상 계좌에 임의의 현물, 스테이블코인 할당
    account.spot_balance = 4000
    account.stablecoin_balance = 6000
    am = AssetManager(account, min_rebalance_threshold=0.01, min_rebalance_interval_minutes=0)
    # bullish 시장에서는 목표 현물이 총 자산의 90%
    am.rebalance("bullish")
    # 현물 잔고가 상승했을 것으로 예상
    assert account.spot_balance > 4000

def test_rebalance_bearish():
    account = Account(initial_balance=10000)
    account.spot_balance = 8000
    account.stablecoin_balance = 2000
    am = AssetManager(account, min_rebalance_threshold=0.01, min_rebalance_interval_minutes=0)
    am.rebalance("bearish")
    # 현물 잔고가 하락했을 것으로 예상
    assert account.spot_balance < 8000
