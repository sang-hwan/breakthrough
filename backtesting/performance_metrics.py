# backtesting/performance_metrics.py
# 백테스트 결과를 평가하기 위해 MDD, 월별 성과, 연간 성과 등을 계산하는 함수들.

import pandas as pd
import numpy as np

def calculate_monthly_performance(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    월별로(YYYY-MM) 손익, 매매 횟수, 승률을 계산하여 요약한 DataFrame을 반환.
    """

    if 'exit_time' not in trades_df.columns:
        return pd.DataFrame()

    trades_df['year_month'] = trades_df['exit_time'].dt.to_period('M')
    grouped = trades_df.groupby('year_month')

    results = []
    for ym, grp in grouped:
        total_pnl = grp['pnl'].sum()
        num_trades = len(grp)
        win_trades = (grp['pnl'] > 0).sum()
        win_rate = (win_trades / num_trades * 100.0) if num_trades > 0 else 0.0

        results.append({
            'year_month': str(ym),
            'total_pnl': total_pnl,
            'num_trades': num_trades,
            'win_rate(%)': win_rate
        })

    return pd.DataFrame(results)


def calculate_yearly_performance(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    연도별(YYYY)로 손익, 매매 횟수, 승률을 계산.
    """

    if 'exit_time' not in trades_df.columns:
        return pd.DataFrame()

    trades_df['year'] = trades_df['exit_time'].dt.year
    grouped = trades_df.groupby('year')

    results = []
    for y, grp in grouped:
        total_pnl = grp['pnl'].sum()
        num_trades = len(grp)
        win_trades = (grp['pnl'] > 0).sum()
        win_rate = (win_trades / num_trades * 100.0) if num_trades > 0 else 0.0

        results.append({
            'year': y,
            'total_pnl': total_pnl,
            'num_trades': num_trades,
            'win_rate(%)': win_rate
        })

    return pd.DataFrame(results)


def calculate_mdd(trades_df: pd.DataFrame, initial_balance: float) -> float:
    """
    MDD(최대낙폭) %를 계산하는 함수.
    - 거래 순서대로 잔고를 추적하며, 최고점(peak)과 비교해 얼마나 내려갔나를 계산
    """

    if 'exit_time' not in trades_df.columns:
        return 0.0

    trades_df = trades_df.sort_values(by='exit_time')

    equity_list = []
    current_balance = initial_balance

    for _, row in trades_df.iterrows():
        current_balance += row['pnl']
        equity_list.append(current_balance)

    equity_arr = np.array(equity_list)
    peak_arr = np.maximum.accumulate(equity_arr)
    drawdown_arr = (equity_arr - peak_arr) / peak_arr
    mdd = drawdown_arr.min() * 100.0
    return mdd

def calculate_sharpe_ratio(trades_df: pd.DataFrame, initial_balance: float, risk_free_rate=0.0) -> float:
    """
    각 트레이드마다 발생한 이익을 기반으로 '단순 Sharpe' 지수를 추정.
    - 실제론 일/주별로 Equity Curve를 만들어서 변동성 분석하는 게 더 정확함.
    """
    if trades_df.empty:
        return 0.0

    if len(trades_df) < 2:
        return 0.0

    trades_df = trades_df.sort_values(by='exit_time')
    current_balance = initial_balance
    returns_list = []

    for _, row in trades_df.iterrows():
        pnl = row['pnl']
        ret = pnl / current_balance
        returns_list.append(ret)
        current_balance += pnl

    returns_arr = np.array(returns_list)

    if len(returns_arr) < 2:
        return 0.0

    mean_return = returns_arr.mean()
    std_return  = returns_arr.std(ddof=1)

    if std_return == 0:
        return 0.0

    sharpe = (mean_return - risk_free_rate) / std_return
    return sharpe

def print_performance_report(trades_df: pd.DataFrame, initial_balance: float) -> None:
    """
    월별 성과, 연도별 성과, 전체 성과(ROI, MDD, 총손익, 승률 등)를 콘솔에 출력.
    """

    if trades_df.empty:
        print("No trades to report.")
        return

    monthly_df = calculate_monthly_performance(trades_df)
    yearly_df = calculate_yearly_performance(trades_df)
    total_pnl = trades_df['pnl'].sum()
    final_balance = initial_balance + total_pnl
    mdd = calculate_mdd(trades_df, initial_balance=initial_balance)
    sharpe = calculate_sharpe_ratio(trades_df, initial_balance=initial_balance)

    print("=== (A) 월별 성과 ===")
    print(monthly_df)

    print("\n=== (B) 연도별 성과 ===")
    print(yearly_df)

    print("\n=== (C) 전체 성과 ===")
    print(f"  - 초기 잔고       : {initial_balance:.2f}")
    print(f"  - 최종 잔고       : {final_balance:.2f}")
    print(f"  - 총 손익         : {total_pnl:.2f}")
    print(f"  - ROI(%)          : {(final_balance - initial_balance) / initial_balance * 100:.2f}%")
    print(f"  - 최대낙폭(MDD)   : {mdd:.2f}%")
    print(f"  - 샤프 지수(단순) : {sharpe:.3f}")

    num_trades = len(trades_df)
    wins = (trades_df['pnl'] > 0).sum()
    win_rate = (wins / num_trades * 100.0) if num_trades > 0 else 0.0
    print(f"  - 총 매매 횟수    : {num_trades}")
    print(f"  - 승률(%)         : {win_rate:.2f}%")
