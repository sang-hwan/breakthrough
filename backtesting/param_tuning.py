# backtesting/param_tuning.py

import itertools
import pandas as pd

from backtesting.backtest_simple import run_simple_backtest
# performance_metrics 모듈에서 필요한 계산 함수를 임포트
from backtesting.performance_metrics import (
    calculate_mdd,
    calculate_monthly_performance,
    calculate_yearly_performance
)

def param_sweep_test():
    """
    여러 파라미터(window, atr_multiplier, profit_ratio)의 조합을 테스트하고,
    각 조합별 성과 지표를 데이터프레임으로 만들어 반환합니다.
    (CSV 저장이나 상위 결과 출력 등은 이 함수를 호출하는 쪽에서 처리)
    """

    # (A) 테스트할 파라미터 후보 정의
    window_list = [10, 20, 30]
    atr_list = [1.5, 2.0]
    profit_ratio_list = [0.03, 0.05]

    results = []

    for window, atr_mult, pr in itertools.product(window_list, atr_list, profit_ratio_list):
        print(f"\n[Running] window={window}, atr_multiplier={atr_mult}, profit_ratio={pr}")

        # (B) 백테스트 실행
        trades_df = run_simple_backtest(
            symbol="BTC/USDT",
            timeframe="4h",
            window=window,
            volume_factor=1.5,
            confirm_bars=2,
            breakout_buffer=0.0,
            atr_window=14,
            atr_multiplier=atr_mult,
            profit_ratio=pr,
            account_size=10_000.0,
            risk_per_trade=0.01,
            fee_rate=0.001
        )

        if trades_df is None or trades_df.empty:
            # 트레이드가 없거나 비정상 -> 기록X
            continue

        # (C) 성과 지표 추가 계산
        initial_balance = 10_000.0

        # 총 손익 (pnl)
        total_pnl = trades_df['pnl'].sum()
        final_balance = initial_balance + total_pnl
        roi_percent = (final_balance - initial_balance) / initial_balance * 100.0

        # MDD 계산
        mdd_percent = calculate_mdd(trades_df, initial_balance=initial_balance)

        # 승률
        num_trades = len(trades_df)
        wins = (trades_df['pnl'] > 0).sum()
        win_rate = wins / num_trades * 100.0 if num_trades > 0 else 0.0

        # (D) 필요한 항목만 모아서 results에 저장
        results.append({
            'window'        : window,
            'atr_multiplier': atr_mult,
            'profit_ratio'  : pr,
            'num_trades'    : num_trades,
            'win_rate(%)'   : round(win_rate, 2),
            'final_balance' : round(final_balance, 2),
            'ROI(%)'        : round(roi_percent, 2),
            'MDD(%)'        : round(mdd_percent, 2),  # 보통 음수
        })

        # (E) (옵션) 월별·연도별 DF CSV 출력은 호출부에서 처리할 수도 있음
        # monthly_df = calculate_monthly_performance(trades_df)
        # yearly_df = calculate_yearly_performance(trades_df)
        # ...
        # (호출부에서 필요 시 저장)

    # (F) 결과 취합
    results_df = pd.DataFrame(results)
    return results_df  # ===> 호출부에서 이 결과를 받아 처리
