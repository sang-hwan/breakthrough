# backtesting/param_sweep.py
# 다양한 파라미터 조합(예: window=10,20 / atr_multiplier=1.5,2.0 / profit_ratio=0.03,0.05 등)을
# 전수 테스트하고, 각 결과를 비교해주는 코드입니다.

import itertools
import pandas as pd

from backtesting.backtest_advanced import run_advanced_backtest
from backtesting.performance_metrics import calculate_mdd, calculate_sharpe_ratio


def run_param_sweep_advanced(
    symbol="BTC/USDT",
    short_timeframe="4h",
    long_timeframe="1d",
    account_size=10_000.0,
    # 테스트할 파라미터 리스트들(None이면 디폴트 목록 사용)
    window_list=None,
    atr_multiplier_list=None,
    profit_ratio_list=None,
    use_partial_tp_list=None,
    # 부분 익절 관련 파라미터
    partial_tp_factor=0.02,
    final_tp_factor=0.05,
    # 날짜 범위
    start_date=None,
    end_date=None
) -> pd.DataFrame:
    """
    (1) 여러 파라미터 조합(window, atr_multiplier, profit_ratio, use_partial_tp 등)을 전수 조사.
    (2) 각 조합에 대해 run_advanced_backtest() 함수를 실행.
    (3) 트레이드 결과(Trades DataFrame)로부터 ROI, MDD, Sharpe 등을 계산.
    (4) 최종 결과를 DataFrame으로 묶어서 반환.

    Args:
        symbol (str): 거래 종목 예) "BTC/USDT"
        short_timeframe (str): 짧은 주기의 타임프레임 예) "4h"
        long_timeframe (str): 긴 주기의 타임프레임 예) "1d"
        account_size (float): 초기 투자금(기본: 10000)
        window_list (list[int] or None): 예) [10, 20, 30]
        atr_multiplier_list (list[float] or None): 예) [1.5, 2.0]
        profit_ratio_list (list[float] or None): 예) [0.03, 0.05]
        use_partial_tp_list (list[bool] or None): 예) [False, True]
        partial_tp_factor (float): 부분 익절 퍼센트(예: 0.02 → 2% 익절 지점)
        final_tp_factor (float): 최종 익절 퍼센트(예: 0.05 → 5% 익절 지점)
        start_date (str or None): 백테스트 시작일 (YYYY-MM-DD)
        end_date (str or None): 백테스트 종료일 (YYYY-MM-DD)

    Returns:
        pd.DataFrame:
            각 파라미터 조합별 결과가 행(row) 하나씩으로 구성된 DataFrame
            (columns 예시: ['window', 'atr_multiplier', 'profit_ratio', 'use_partial_tp', 'num_trades', ...])
    """

    # (A) 파라미터가 None으로 넘어온 경우 기본값 지정
    if window_list is None:
        window_list = [10, 20, 30]
    if atr_multiplier_list is None:
        atr_multiplier_list = [1.5, 2.0]
    if profit_ratio_list is None:
        profit_ratio_list = [0.03, 0.05]
    if use_partial_tp_list is None:
        use_partial_tp_list = [False, True]

    results = []

    # (B) 모든 파라미터 조합을 itertools.product()로 생성
    for window, atr_mult, pr, partial_flag in itertools.product(
        window_list, atr_multiplier_list, profit_ratio_list, use_partial_tp_list
    ):

        # (C) 백테스트 실행
        trades_df = run_advanced_backtest(
            symbol=symbol,
            short_timeframe=short_timeframe,
            long_timeframe=long_timeframe,
            window=window,
            atr_multiplier=atr_mult,
            profit_ratio=pr,
            use_partial_take_profit=partial_flag,
            partial_tp_factor=partial_tp_factor,
            final_tp_factor=final_tp_factor,
            account_size=account_size,
            start_date=start_date,
            end_date=end_date
        )

        # (D) 결과 집계
        if trades_df is None or trades_df.empty:
            # 매매가 하나도 없을 경우
            results.append({
                'window': window,
                'atr_multiplier': atr_mult,
                'profit_ratio': pr,
                'use_partial_tp': partial_flag,
                'num_trades': 0,
                'final_balance': account_size,
                'ROI(%)': 0.0,
                'MDD(%)': 0.0,
                'Sharpe': 0.0,
            })
        else:
            # 총 손익
            total_pnl = trades_df['pnl'].sum()
            final_balance = account_size + total_pnl
            roi_percent = (final_balance - account_size) / account_size * 100.0

            # MDD, Sharpe 계산
            mdd_percent = calculate_mdd(trades_df, initial_balance=account_size)
            sharpe_val = calculate_sharpe_ratio(trades_df, initial_balance=account_size)

            results.append({
                'window': window,
                'atr_multiplier': atr_mult,
                'profit_ratio': pr,
                'use_partial_tp': partial_flag,
                'num_trades': len(trades_df),
                'final_balance': round(final_balance, 2),
                'ROI(%)': round(roi_percent, 2),
                'MDD(%)': round(mdd_percent, 2),
                'Sharpe': round(sharpe_val, 3),
            })

    # (E) DataFrame으로 변환 후 ROI 기준 내림차순 정렬
    results_df = pd.DataFrame(results)
    results_df.sort_values(by='ROI(%)', ascending=False, inplace=True)
    return results_df