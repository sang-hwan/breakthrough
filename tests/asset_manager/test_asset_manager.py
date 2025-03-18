# tests/asset_manager/test_asset_manager.py

# core.account 모듈에서 Account 클래스를 가져옵니다.
from asset_position.account import Account
# trading.asset_manager 모듈에서 AssetManager 클래스를 가져옵니다.
from asset_position.asset_manager import AssetManager

def test_rebalance_bullish():
    """
    bullish(상승장) 상황에서 자산 리밸런싱이 정상적으로 이루어지는지 테스트합니다.
    
    - 초기 잔고를 설정하고, 현물과 스테이블코인 잔고를 임의로 할당합니다.
    - bullish 시장에서는 목표 현물 비중이 총 자산의 90%가 되도록 리밸런싱을 수행합니다.
    - 리밸런싱 후 현물 잔고가 증가해야 하므로 이를 검증합니다.
    
    Parameters:
        없음
        
    Returns:
        없음 (assertion을 통해 테스트 결과를 검증)
    """
    # 초기 잔고 10,000달러로 Account 객체 생성
    account = Account(initial_balance=10000)
    # 가상 계좌에 현물 잔고 4,000달러, 스테이블코인 잔고 6,000달러 할당
    account.spot_balance = 4000
    account.stablecoin_balance = 6000
    # AssetManager 객체 생성. 리밸런싱 최소 임계치 1% 및 간격 0분으로 설정
    am = AssetManager(account, min_rebalance_threshold=0.01, min_rebalance_interval_minutes=0)
    # bullish 시장에서는 목표로 하는 현물 비중이 높아지도록 리밸런싱 수행
    am.rebalance("bullish")
    # 리밸런싱 후 현물 잔고가 4,000달러보다 커졌는지 확인
    assert account.spot_balance > 4000

def test_rebalance_bearish():
    """
    bearish(하락장) 상황에서 자산 리밸런싱이 정상적으로 이루어지는지 테스트합니다.
    
    - 초기 잔고를 설정하고, 현물과 스테이블코인 잔고를 임의로 할당합니다.
    - bearish 시장에서는 리스크 관리를 위해 현물 보유 비중을 낮추도록 리밸런싱을 수행합니다.
    - 리밸런싱 후 현물 잔고가 감소했는지 확인합니다.
    
    Parameters:
        없음
        
    Returns:
        없음 (assertion을 통해 테스트 결과를 검증)
    """
    # 초기 잔고 10,000달러로 Account 객체 생성
    account = Account(initial_balance=10000)
    # 가상 계좌에 현물 잔고 8,000달러, 스테이블코인 잔고 2,000달러 할당
    account.spot_balance = 8000
    account.stablecoin_balance = 2000
    # AssetManager 객체 생성 (리밸런싱 최소 임계치 1%, 간격 0분)
    am = AssetManager(account, min_rebalance_threshold=0.01, min_rebalance_interval_minutes=0)
    # bearish 시장 상황에 맞게 리밸런싱 수행
    am.rebalance("bearish")
    # 리밸런싱 후 현물 잔고가 8,000달러보다 작아졌는지 확인
    assert account.spot_balance < 8000
