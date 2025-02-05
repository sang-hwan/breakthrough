# trading/strategies.py
def select_strategy(market_regime: str, liquidity_info: str, data, current_time, market_type: str = "crypto") -> str:
    """
    시장 레짐 및 유동성 상태에 따라 실행할 전략을 결정합니다.
    
    - bullish: 신규 매수 진입 ("enter_long")
    - bearish: 모든 포지션 청산 ("exit_all")
    - sideways: 유동성이 높으면 범위 트레이딩, 낮으면 평균 회귀 전략 선택
    - 그 외: "hold"
    
    이 전략 선택 로직은 기존 추세 추종이 아닌 레짐 기반으로 결정됩니다.
    """
    regime = market_regime.lower()
    if regime == "bullish":
        return "enter_long"
    elif regime == "bearish":
        return "exit_all"
    elif regime == "sideways":
        if liquidity_info.lower() == "high":
            return "range_trade"
        else:
            return "mean_reversion"
    else:
        return "hold"
