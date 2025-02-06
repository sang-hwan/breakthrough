# backtesting/performance.py
import pandas as pd
import numpy as np
from logs.logger_config import setup_logger

logger = setup_logger("performance")

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
    symbol: str = "",  # 신규 파라미터: 종목 이름
    exit_time_col: str = "exit_time",
    pnl_col: str = "pnl",
    monthly_header: str = "=== (A) 월별 성과 ===",
    yearly_header: str = "=== (B) 연도별 성과 ===",
    overall_header: str = "=== (C) 전체 성과 ===",
    no_trades_message: str = "No trades to report."
) -> None:
    if trades_df.empty:
        logger.info(no_trades_message)
        return
    monthly_df = calculate_monthly_performance(trades_df, exit_time_col, pnl_col)
    yearly_df = calculate_yearly_performance(trades_df, exit_time_col, pnl_col)
    total_pnl = trades_df[pnl_col].sum()
    final_balance = initial_balance + total_pnl
    mdd = calculate_mdd(trades_df, initial_balance=initial_balance, exit_time_col=exit_time_col, pnl_col=pnl_col)
    sharpe = calculate_sharpe_ratio(trades_df, initial_balance=initial_balance, exit_time_col=exit_time_col, pnl_col=pnl_col)
    
    report_lines = []
    if symbol:
        report_lines.append(f"종목: {symbol}")
    report_lines.append(monthly_header)
    report_lines.append(str(monthly_df))
    report_lines.append("\n" + yearly_header)
    report_lines.append(str(yearly_df))
    report_lines.append("\n" + overall_header)
    report_lines.append(f"  - 초기 잔고       : {initial_balance:.2f}")
    report_lines.append(f"  - 최종 잔고       : {final_balance:.2f}")
    report_lines.append(f"  - 총 손익         : {total_pnl:.2f}")
    report_lines.append(f"  - ROI(%)          : {(final_balance - initial_balance) / initial_balance * 100:.2f}%")
    report_lines.append(f"  - 최대낙폭(MDD)   : {mdd:.2f}%")
    report_lines.append(f"  - 샤프 지수(단순) : {sharpe:.3f}")
    num_trades = len(trades_df)
    wins = (trades_df[pnl_col] > 0).sum()
    win_rate = (wins / num_trades * 100.0) if num_trades > 0 else 0.0
    report_lines.append(f"  - 총 매매 횟수    : {num_trades}")
    report_lines.append(f"  - 승률(%)         : {win_rate:.2f}%")
    
    report = "\n".join(report_lines)
    logger.info("백테스트 성과 보고:\n" + report)

def monte_carlo_simulation(trades_df: pd.DataFrame, initial_balance: float, n_simulations: int = 1000, perturbation_std: float = 0.002):
    """
    거래 내역에 무작위 슬리피지 및 거래 비용 변동을 적용하여 Monte Carlo 시뮬레이션을 수행.
    perturbation_std: 각 거래의 pnl에 적용할 노이즈 표준편차.
    반환: ROI 분포, MDD 분포 등의 통계.
    """
    rois = []
    mdds = []
    for _ in range(n_simulations):
        simulated_trades = trades_df.copy()
        # 각 trade의 pnl에 랜덤 노이즈 적용 (정규분포)
        noise = np.random.normal(loc=0, scale=perturbation_std, size=len(simulated_trades))
        simulated_trades["pnl"] = simulated_trades["pnl"] * (1 + noise)
        total_pnl = simulated_trades["pnl"].sum()
        final_balance = initial_balance + total_pnl
        roi = (final_balance - initial_balance) / initial_balance * 100
        mdd = calculate_mdd(simulated_trades, initial_balance)
        rois.append(roi)
        mdds.append(mdd)
    return {
        "roi_mean": np.mean(rois),
        "roi_std": np.std(rois),
        "mdd_mean": np.mean(mdds),
        "mdd_std": np.std(mdds)
    }