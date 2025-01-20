# test.py

from fetch_binance_data import fetch_binance_ohlcv
from signal_generator import calculate_breakout_signals, calculate_vcp_pattern
from trade_logic import generate_trade_signals

# 1) 데이터 수집
df = fetch_binance_ohlcv('BTC/USDT', '4h', 500)

# 2) 시그널 계산 (전고점 돌파 + 거래량 + 확정 돌파), 고가 기준 + 0.5% 버퍼
df = calculate_breakout_signals(
    df,
    window=20,
    vol_factor=1.5,
    confirm_bars=2,
    use_high=True,
    breakout_buffer=0.005  # 예: 전고점 대비 +0.5% 초과해야 돌파로 인정
)

# 3) VCP 로직 추가 (3단계 수축 체크)
df = calculate_vcp_pattern(df, window_list=[20, 10, 5])

# 4) 최종 매매 로직 (ATR 손절, 고정익절) - 진입가 기준
df = generate_trade_signals(
    df,
    atr_window=14,
    atr_multiplier=1.5,
    profit_ratio=0.07,
)

print(df.tail(20))  # 결과 확인
