# backtesting/performance.py
import pandas as pd
import numpy as np
from logs.logger_config import setup_logger

logger = setup_logger("performance")

def calculate_monthly_performance(trades_df: pd.DataFrame, exit_time_col: str = "exit_time", pnl_col: str = "pnl", period_freq: str = 'M', period_col: str = "year_month") -> pd.DataFrame:
    if trades_df.empty:
        logger.warning("Monthly performance: No trades data provided.")
        return pd.DataFrame()
    trades_df = trades_df.copy()
    if exit_time_col not in trades_df.columns:
        logger.error(f"Monthly performance: '{exit_time_col}' 컬럼이 없습니다.")
        return pd.DataFrame()
    trades_df[exit_time_col] = pd.to_datetime(trades_df[exit_time_col])
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
    monthly_df = pd.DataFrame(results)
    logger.info("월별 성과 계산 완료.")
    return monthly_df

def calculate_yearly_performance(trades_df: pd.DataFrame, exit_time_col: str = "exit_time", pnl_col: str = "pnl", year_col: str = "year") -> pd.DataFrame:
    if trades_df.empty:
        logger.warning("Yearly performance: No trades data provided.")
        return pd.DataFrame()
    trades_df = trades_df.copy()
    if exit_time_col not in trades_df.columns:
        logger.error(f"Yearly performance: '{exit_time_col}' 컬럼이 없습니다.")
        return pd.DataFrame()
    trades_df[exit_time_col] = pd.to_datetime(trades_df[exit_time_col])
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
    yearly_df = pd.DataFrame(results)
    logger.info("연도별 성과 계산 완료.")
    return yearly_df

def calculate_mdd(trades_df: pd.DataFrame, initial_balance: float, exit_time_col: str = "exit_time", pnl_col: str = "pnl", mdd_factor: float = 100.0) -> float:
    if trades_df.empty:
        logger.warning("MDD 계산: No trades data provided.")
        return 0.0
    trades_df = trades_df.copy().sort_values(by=exit_time_col)
    equity = initial_balance
    peak = initial_balance
    mdd = 0.0
    for _, row in trades_df.iterrows():
        equity += row[pnl_col]
        if equity > peak:
            peak = equity
        dd = (equity - peak) / peak
        if dd < mdd:
            mdd = dd
    mdd_percentage = mdd * mdd_factor
    logger.info("MDD 계산 완료.")
    return mdd_percentage

def calculate_sharpe_ratio(trades_df: pd.DataFrame, initial_balance: float, risk_free_rate: float = 0.0, exit_time_col: str = "exit_time", pnl_col: str = "pnl") -> float:
    if trades_df.empty or len(trades_df) < 2:
        logger.warning("Sharpe Ratio 계산: 거래 데이터가 부족합니다.")
        return 0.0
    trades_df = trades_df.copy().sort_values(by=exit_time_col)
    equity = initial_balance
    returns = []
    for _, row in trades_df.iterrows():
        ret = row[pnl_col] / equity
        returns.append(ret)
        equity += row[pnl_col]
    returns = np.array(returns)
    if returns.std() == 0:
        logger.warning("Sharpe Ratio 계산: 수익률 표준편차가 0입니다.")
        return 0.0
    sharpe = (returns.mean() - risk_free_rate) / returns.std()
    logger.info("Sharpe Ratio 계산 완료.")
    return sharpe

def print_performance_report(trades_df: pd.DataFrame, initial_balance: float, symbol: str = "", exit_time_col: str = "exit_time", pnl_col: str = "pnl", monthly_header: str = "=== (A) 월별 성과 ===", yearly_header: str = "=== (B) 연도별 성과 ===", overall_header: str = "=== (C) 전체 성과 ===", no_trades_message: str = "No trades to report.") -> None:
    if trades_df.empty:
        logger.info(no_trades_message)
        return
    monthly_df = calculate_monthly_performance(trades_df, exit_time_col, pnl_col)
    yearly_df = calculate_yearly_performance(trades_df, exit_time_col, pnl_col)
    total_pnl = trades_df[pnl_col].sum()
    final_balance = initial_balance + total_pnl
    mdd = calculate_mdd(trades_df, initial_balance, exit_time_col, pnl_col)
    sharpe = calculate_sharpe_ratio(trades_df, initial_balance, exit_time_col=exit_time_col, pnl_col=pnl_col)
    
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
    print(report)

def monte_carlo_simulation(trades_df: pd.DataFrame, initial_balance: float, n_simulations: int = 1000, perturbation_std: float = 0.002):
    rois = []
    mdds = []
    for _ in range(n_simulations):
        simulated_trades = trades_df.copy()
        noise = np.random.normal(loc=0, scale=perturbation_std, size=len(simulated_trades))
        simulated_trades["pnl"] = simulated_trades["pnl"] * (1 + noise)
        total_pnl = simulated_trades["pnl"].sum()
        final_balance_sim = initial_balance + total_pnl
        roi = (final_balance_sim - initial_balance) / initial_balance * 100
        mdd = calculate_mdd(simulated_trades, initial_balance)
        rois.append(roi)
        mdds.append(mdd)
    stats = {
        "roi_mean": np.mean(rois),
        "roi_std": np.std(rois),
        "mdd_mean": np.mean(mdds),
        "mdd_std": np.std(mdds)
    }
    logger.info("Monte Carlo 시뮬레이션 완료.")
    return stats
