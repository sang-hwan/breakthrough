# backtesting/performance.py
import pandas as pd
import numpy as np
from logs.logger_config import setup_logger

logger = setup_logger(__name__)

def calculate_monthly_performance(trades, weekly_data=None):
    monthly_data = {}
    for trade in trades:
        exit_time = trade.get("exit_time") or trade.get("entry_time")
        if not exit_time:
            continue
        month = exit_time.strftime("%Y-%m") if hasattr(exit_time, "strftime") else exit_time[:7]
        monthly_data.setdefault(month, []).append(trade.get("pnl", 0))
    
    monthly_perf = {month: {
        "roi": (sum(pnls) / 10000.0) * 100,
        "trade_count": len(pnls),
        "total_pnl": sum(pnls)
    } for month, pnls in monthly_data.items()}
    
    weekly_metrics = {}
    if weekly_data is not None and not weekly_data.empty:
        weekly_returns = weekly_data['close'].pct_change().dropna()
        if not weekly_returns.empty:
            cumulative = (weekly_returns + 1).cumprod()
            weekly_roi = cumulative.iloc[-1] - 1
            drawdowns = (cumulative - cumulative.cummax()) / cumulative.cummax()
            weekly_metrics = {
                "weekly_roi": weekly_roi * 100,
                "weekly_max_drawdown": drawdowns.min() * 100
            }
        else:
            weekly_metrics = {"weekly_roi": 0.0, "weekly_max_drawdown": 0.0}
    
    logger.debug("Monthly performance calculated.")
    return {"monthly": monthly_perf, "weekly": weekly_metrics}

def calculate_overall_performance(trades):
    initial_capital = 10000.0
    total_pnl = sum(trade.get("pnl", 0) for trade in trades)
    trade_count = len(trades)
    cumulative_return = (initial_capital + total_pnl) / initial_capital - 1
    roi = cumulative_return * 100
    sorted_trades = sorted(trades, key=lambda t: t.get("exit_time") or t.get("entry_time"))
    
    dates, equity_list, trade_pnls = [], [], []
    equity = initial_capital
    for trade in sorted_trades:
        dt = trade.get("exit_time") or trade.get("entry_time")
        if not dt:
            continue
        pnl = trade.get("pnl", 0)
        equity += pnl
        dates.append(pd.to_datetime(dt))
        equity_list.append(equity)
        trade_pnls.append(pnl)
    
    if not dates:
        return {
            "roi": roi, "cumulative_return": cumulative_return, "total_pnl": total_pnl,
            "trade_count": trade_count, "annualized_return": 0.0, "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "calmar_ratio": 0.0, "max_drawdown": 0.0,
            "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "profit_factor": 0.0,
            "trades_per_year": 0.0, "max_consecutive_wins": 0, "max_consecutive_losses": 0
        }
    
    df_eq = pd.DataFrame({"equity": equity_list}, index=pd.to_datetime(dates)).groupby(level=0).last().asfreq("D", method="ffill")
    daily_returns = df_eq["equity"].pct_change().dropna()
    annualized_vol = daily_returns.std() * np.sqrt(252) if not daily_returns.empty else 0.0
    total_days = (df_eq.index.max() - df_eq.index.min()).days
    annualized_return = (df_eq["equity"].iloc[-1] / initial_capital) ** (365 / total_days) - 1 if total_days > 0 else 0.0
    max_drawdown = (df_eq["equity"].cummax() - df_eq["equity"]).max()
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0.0
    downside = daily_returns[daily_returns < 0]
    sortino_ratio = annualized_return / (downside.std() * np.sqrt(252)) if not downside.empty and downside.std() != 0 else 0.0
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0
    wins = [pnl for pnl in trade_pnls if pnl > 0]
    losses = [pnl for pnl in trade_pnls if pnl <= 0]
    win_rate = (len(wins) / trade_count * 100) if trade_count else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else 0.0
    max_consec_wins = max((len(list(g)) for k, g in pd.Series(trade_pnls > 0).groupby((trade_pnls > 0).ne((trade_pnls > 0).shift()).cumsum())), default=0)
    max_consec_losses = max((len(list(g)) for k, g in pd.Series(trade_pnls <= 0).groupby((trade_pnls <= 0).ne((trade_pnls <= 0).shift()).cumsum())), default=0)
    years = total_days / 365 if total_days > 0 else 1
    trades_per_year = trade_count / years
    overall = {
        "roi": roi, "cumulative_return": cumulative_return, "total_pnl": total_pnl,
        "trade_count": trade_count, "annualized_return": annualized_return * 100,
        "annualized_volatility": annualized_vol * 100, "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio, "calmar_ratio": calmar_ratio, "max_drawdown": max_drawdown,
        "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss, "profit_factor": profit_factor,
        "trades_per_year": trades_per_year, "max_consecutive_wins": max_consec_wins,
        "max_consecutive_losses": max_consec_losses
    }
    logger.debug("Overall performance calculated.")
    return overall

def compute_performance(trades, weekly_data=None):
    overall = calculate_overall_performance(trades)
    monthly = calculate_monthly_performance(trades, weekly_data=weekly_data)
    logger.debug("Performance report generated.")
    return {"overall": overall, "monthly": monthly["monthly"], "weekly": monthly["weekly"]}
