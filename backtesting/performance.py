# backtesting/performance.py
import pandas as pd
import numpy as np
from datetime import datetime
import math

def calculate_monthly_performance(trades):
    monthly_data = {}
    for trade in trades:
        exit_time = trade.get("exit_time") or trade.get("entry_time")
        if exit_time is None:
            continue
        if hasattr(exit_time, "strftime"):
            month = exit_time.strftime("%Y-%m")
        else:
            month = exit_time[:7]
        monthly_data.setdefault(month, []).append(trade.get("pnl", 0))
    
    monthly_performance = {}
    for month, pnl_list in monthly_data.items():
        total_pnl = sum(pnl_list)
        roi = (total_pnl / 10000.0) * 100  # 초기 자본 10000 기준
        monthly_performance[month] = {
            "roi": roi,
            "trade_count": len(pnl_list)
        }
    return monthly_performance

def calculate_overall_performance(trades):
    initial_capital = 10000.0
    total_pnl = sum(trade.get("pnl", 0) for trade in trades)
    trade_count = len(trades)
    cumulative_return = (initial_capital + total_pnl) / initial_capital - 1
    roi = cumulative_return * 100

    # 정렬: exit_time (없으면 entry_time)
    sorted_trades = sorted(trades, key=lambda t: t.get("exit_time") or t.get("entry_time"))
    
    # Equity curve 생성
    dates = []
    equity = initial_capital
    equity_list = []
    trade_pnls = []
    for trade in sorted_trades:
        dt = trade.get("exit_time") or trade.get("entry_time")
        if dt is None:
            continue
        pnl = trade.get("pnl", 0)
        equity += pnl
        dates.append(pd.to_datetime(dt))
        equity_list.append(equity)
        trade_pnls.append(pnl)
    
    if not dates:
        # 거래가 하나도 없으면 기본값 반환
        return {
            "roi": roi,
            "cumulative_return": cumulative_return,
            "total_pnl": total_pnl,
            "trade_count": trade_count,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "trades_per_year": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0
        }
    
    # Equity DataFrame를 일별로 재구성
    df_equity = pd.DataFrame({"equity": equity_list}, index=pd.to_datetime(dates))
    df_equity = df_equity.asfreq("D", method="ffill")
    daily_returns = df_equity["equity"].pct_change().dropna()
    
    # 연간화 변동성
    annualized_vol = daily_returns.std() * np.sqrt(252) if not daily_returns.empty else 0.0

    # 연간화 수익률 (기간 보정)
    start_date = df_equity.index.min()
    end_date = df_equity.index.max()
    total_days = (end_date - start_date).days
    if total_days > 0:
        annualized_return = (df_equity["equity"].iloc[-1] / initial_capital) ** (365 / total_days) - 1
    else:
        annualized_return = 0.0

    # 최대 낙폭 계산
    roll_max = df_equity["equity"].cummax()
    drawdown = roll_max - df_equity["equity"]
    max_drawdown = drawdown.max()

    # 샤프 지수 (무위험 수익률 0으로 가정)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0.0

    # 소르티노 지수: 음의 수익률만 사용
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0.0
    sortino_ratio = annualized_return / downside_std if downside_std != 0 else 0.0

    # 칼마 지수: 연간화 수익률 / 최대 낙폭 (낙폭 0이면 0)
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0

    # 매매 통계 계산
    wins = [pnl for pnl in trade_pnls if pnl > 0]
    losses = [pnl for pnl in trade_pnls if pnl <= 0]
    win_rate = (len(wins) / trade_count * 100) if trade_count > 0 else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else 0.0

    # 연속 승/패 계산
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_wins = 0
    current_losses = 0
    for pnl in trade_pnls:
        if pnl > 0:
            current_wins += 1
            current_losses = 0
        else:
            current_losses += 1
            current_wins = 0
        max_consecutive_wins = max(max_consecutive_wins, current_wins)
        max_consecutive_losses = max(max_consecutive_losses, current_losses)

    # 거래 빈도: 연간 거래 건수
    years = total_days / 365 if total_days > 0 else 1
    trades_per_year = trade_count / years

    return {
        "roi": roi,
        "cumulative_return": cumulative_return,
        "total_pnl": total_pnl,
        "trade_count": trade_count,
        "annualized_return": annualized_return * 100,  # %로 환산
        "annualized_volatility": annualized_vol * 100,  # %로 환산
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "trades_per_year": trades_per_year,
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses
    }

def compute_performance(trades):
    overall = calculate_overall_performance(trades)
    monthly = calculate_monthly_performance(trades)
    overall["monthly_performance"] = monthly
    return overall
