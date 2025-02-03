# backtesting/performance.py
import pandas as pd
import numpy as np

def calculate_monthly_performance(
    trades_df: pd.DataFrame,
    exit_time_col: str = "exit_time",
    pnl_col: str = "pnl",
    period_freq: str = 'M',
    period_col: str = "year_month"
) -> pd.DataFrame:
    if exit_time_col not in trades_df.columns:
        return pd.DataFrame()
    trades_df[period_col] = trades_df[exit_time_col].dt.to_period(period_freq)
    grouped = trades_df.groupby(period_col)
    results = []
    for period_val, grp in grouped:
        total_pnl = grp[pnl_col].sum()
        num_trades = len(grp)
        win_trades = (grp[pnl_col] > 0).sum()
        win_rate = (win_trades / num_trades * 100.0) if num_trades > 0 else 0.0
        results.append({
            period_col: str(period_val),
            'total_pnl': total_pnl,
            'num_trades': num_trades,
            'win_rate(%)': win_rate
        })
    return pd.DataFrame(results)

def calculate_yearly_performance(
    trades_df: pd.DataFrame,
    exit_time_col: str = "exit_time",
    pnl_col: str = "pnl",
    year_col: str = "year"
) -> pd.DataFrame:
    if exit_time_col not in trades_df.columns:
        return pd.DataFrame()
    trades_df[year_col] = trades_df[exit_time_col].dt.year
    grouped = trades_df.groupby(year_col)
    results = []
    for yr, grp in grouped:
        total_pnl = grp[pnl_col].sum()
        num_trades = len(grp)
        win_trades = (grp[pnl_col] > 0).sum()
        win_rate = (win_trades / num_trades * 100.0) if num_trades > 0 else 0.0
        results.append({
            year_col: yr,
            'total_pnl': total_pnl,
            'num_trades': num_trades,
            'win_rate(%)': win_rate
        })
    return pd.DataFrame(results)

def calculate_mdd(
    trades_df: pd.DataFrame,
    initial_balance: float,
    exit_time_col: str = "exit_time",
    pnl_col: str = "pnl",
    mdd_factor: float = 100.0
) -> float:
    if exit_time_col not in trades_df.columns:
        return 0.0
    trades_df = trades_df.sort_values(by=exit_time_col)
    equity_list = []
    current_balance = initial_balance
    for _, row in trades_df.iterrows():
        current_balance += row[pnl_col]
        equity_list.append(current_balance)
    equity_arr = np.array(equity_list)
    peak_arr = np.maximum.accumulate(equity_arr)
    drawdown_arr = (equity_arr - peak_arr) / peak_arr
    mdd = drawdown_arr.min() * mdd_factor
    return mdd

def calculate_sharpe_ratio(
    trades_df: pd.DataFrame,
    initial_balance: float,
    risk_free_rate: float = 0.0,
    exit_time_col: str = "exit_time",
    pnl_col: str = "pnl"
) -> float:
    if trades_df.empty or len(trades_df) < 2:
        return 0.0
    trades_df = trades_df.sort_values(by=exit_time_col)
    current_balance = initial_balance
    returns_list = []
    for _, row in trades_df.iterrows():
        pnl = row[pnl_col]
        ret = pnl / current_balance
        returns_list.append(ret)
        current_balance += pnl
    returns_arr = np.array(returns_list)
    if len(returns_arr) < 2:
        return 0.0
    mean_return = returns_arr.mean()
    std_return = returns_arr.std(ddof=1)
    if std_return == 0:
        return 0.0
    sharpe = (mean_return - risk_free_rate) / std_return
    return sharpe

def print_performance_report(
    trades_df: pd.DataFrame,
    initial_balance: float,
    exit_time_col: str = "exit_time",
    pnl_col: str = "pnl",
    monthly_header: str = "=== (A) 월별 성과 ===",
    yearly_header: str = "=== (B) 연도별 성과 ===",
    overall_header: str = "=== (C) 전체 성과 ===",
    no_trades_message: str = "No trades to report."
) -> None:
    if trades_df.empty:
        print(no_trades_message)
        return
    monthly_df = calculate_monthly_performance(trades_df, exit_time_col, pnl_col)
    yearly_df = calculate_yearly_performance(trades_df, exit_time_col, pnl_col)
    total_pnl = trades_df[pnl_col].sum()
    final_balance = initial_balance + total_pnl
    mdd = calculate_mdd(trades_df, initial_balance=initial_balance, exit_time_col=exit_time_col, pnl_col=pnl_col)
    sharpe = calculate_sharpe_ratio(trades_df, initial_balance=initial_balance, exit_time_col=exit_time_col, pnl_col=pnl_col)
    print(monthly_header)
    print(monthly_df)
    print("\n" + yearly_header)
    print(yearly_df)
    print("\n" + overall_header)
    print(f"  - 초기 잔고       : {initial_balance:.2f}")
    print(f"  - 최종 잔고       : {final_balance:.2f}")
    print(f"  - 총 손익         : {total_pnl:.2f}")
    print(f"  - ROI(%)          : {(final_balance - initial_balance) / initial_balance * 100:.2f}%")
    print(f"  - 최대낙폭(MDD)   : {mdd:.2f}%")
    print(f"  - 샤프 지수(단순) : {sharpe:.3f}")
    num_trades = len(trades_df)
    wins = (trades_df[pnl_col] > 0).sum()
    win_rate = (wins / num_trades * 100.0) if num_trades > 0 else 0.0
    print(f"  - 총 매매 횟수    : {num_trades}")
    print(f"  - 승률(%)         : {win_rate:.2f}%")
