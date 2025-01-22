# backtesting/param_tuning.py

import itertools  # 파라미터 조합을 만들기 위한 라이브러리
import pandas as pd  # 데이터 처리를 위한 라이브러리

# 단순 백테스트 함수 임포트
from backtesting.backtest_simple import run_simple_backtest
# 성과 지표 계산 함수 임포트
from backtesting.performance_metrics import calculate_mdd

def param_sweep_test():
    """
    여러 가지 전략 파라미터 조합(window, atr_multiplier, profit_ratio)을 테스트하여,
    각 조합의 백테스트 결과를 요약한 데이터프레임을 반환합니다.
    """
    # -------------------------------
    # (A) 테스트할 파라미터 정의
    # -------------------------------
    # 전략에서 조정할 파라미터 리스트
    window_list = [10, 20, 30]  # 돌파 신호를 위한 기간
    atr_list = [1.5, 2.0]  # ATR(평균 진폭) 배수: 손절 기준 설정
    profit_ratio_list = [0.03, 0.05]  # 고정 익절 비율

    # 결과를 저장할 리스트
    results = []

    # itertools.product를 사용해 모든 파라미터 조합 생성
    for window, atr_mult, pr in itertools.product(window_list, atr_list, profit_ratio_list):
        print(f"\n[Running] window={window}, atr_multiplier={atr_mult}, profit_ratio={pr}")

        # -------------------------------
        # (B) 각 조합에 대해 백테스트 실행
        # -------------------------------
        trades_df = run_simple_backtest(
            symbol="BTC/USDT",  # 거래 종목 (비트코인/테더)
            short_timeframe="4h",  # 단기 봉 주기 (4시간)
            long_timeframe="1d",  # 장기 봉 주기 (1일)
            window=window,  # 돌파 신호를 위한 기간
            volume_factor=1.5,  # 거래량 필터
            confirm_bars=2,  # 추가 확인 봉 수
            breakout_buffer=0.0,  # 돌파 신호 버퍼
            atr_window=14,  # ATR 계산 기간
            atr_multiplier=atr_mult,  # 손절 기준 ATR 배수
            profit_ratio=pr,  # 익절 비율
            account_size=10_000.0,  # 초기 계좌 잔액
            risk_per_trade=0.01,  # 1회 거래 시 총 자산 대비 위험 비율
            fee_rate=0.001  # 거래 수수료 비율
        )

        # 백테스트 결과가 없거나 거래가 발생하지 않은 경우, 다음 조합으로 넘어감
        if trades_df is None or trades_df.empty:
            continue

        # -------------------------------
        # (C) 성과 지표 계산
        # -------------------------------
        initial_balance = 10_000.0  # 초기 자산

        # 총 손익 (PnL)
        total_pnl = trades_df['pnl'].sum()
        final_balance = initial_balance + total_pnl  # 최종 잔액 계산
        roi_percent = (final_balance - initial_balance) / initial_balance * 100.0  # ROI 계산

        # 최대 낙폭(MDD)
        mdd_percent = calculate_mdd(trades_df, initial_balance=initial_balance)

        # 승률 계산
        num_trades = len(trades_df)  # 총 거래 수
        wins = (trades_df['pnl'] > 0).sum()  # 수익 거래 수
        win_rate = wins / num_trades * 100.0 if num_trades > 0 else 0.0  # 승률 (%)

        # -------------------------------
        # (D) 결과를 정리해 리스트에 저장
        # -------------------------------
        results.append({
            'window'        : window,  # 돌파 신호 기간
            'atr_multiplier': atr_mult,  # ATR 배수
            'profit_ratio'  : pr,  # 익절 비율
            'num_trades'    : num_trades,  # 총 거래 수
            'win_rate(%)'   : round(win_rate, 2),  # 승률 (%)
            'final_balance' : round(final_balance, 2),  # 최종 잔액
            'ROI(%)'        : round(roi_percent, 2),  # 수익률 (%)
            'MDD(%)'        : round(mdd_percent, 2),  # 최대 낙폭 (%)
        })

    # -------------------------------
    # (F) 결과를 데이터프레임으로 변환 후 반환
    # -------------------------------
    results_df = pd.DataFrame(results)
    return results_df  # 최종 결과를 호출부로 반환
