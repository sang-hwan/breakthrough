# trading/trade_decision.py
from logs.log_config import setup_logger

logger = setup_logger(__name__)

class TradeDecision:
    """
    거래 판단 관련 기능을 제공합니다.
    """

    def __init__(self):
        self.logger = setup_logger(__name__)

    def decide_trade(self, signal: str, current_price: float, account) -> str:
        """
        매수/매도/홀드를 판단합니다.

        Parameters:
            signal (str): 전략 신호 ("buy", "sell", "hold" 등).
            current_price (float): 현재 시장 가격.
            account: 계좌 객체 (잔고, 포지션 관리).
        Returns:
            str: "buy", "sell", "hold" 중 하나.
        """
        self.logger.debug(f"Deciding trade for signal: {signal}, price: {current_price}")
        # 간단 예시: 신호에 따라 판단
        if signal.lower() == "buy":
            if account.get_available_balance() > current_price:
                decision = "buy"
            else:
                decision = "hold"
        elif signal.lower() == "sell":
            decision = "sell"
        else:
            decision = "hold"
        self.logger.debug(f"Trade decision: {decision}")
        return decision
