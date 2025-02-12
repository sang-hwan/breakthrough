# backtesting/performance.py
import pandas as pd
import numpy as np
from logs.logger_config import setup_logger

logger = setup_logger(__name__)

def calculate_monthly_performance(trades, weekly_data=None):
    """
    거래 내역(trades)을 월별로 그룹화하여, 월별 ROI, 거래 건수, 총 PnL 등을 산출합니다.
    또한, 주간 캔들 데이터(weekly_data)가 제공되면 주간 ROI와 최대 낙폭도 계산합니다.
    """
    # 월별 pnl 리스트 집계
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
    
    # 월별 성과 계산
    monthly_perf = {}
    for month, pnl_list in monthly_data.items():
        total_pnl = sum(pnl_list)
        roi = (total_pnl / 10000.0) * 100  # 초기 자본 10000 기준 ROI(%)
        monthly_perf[month] = {
            "roi": roi,
            "trade_count": len(pnl_list),
            "total_pnl": total_pnl
        }
    
    # 주간 전략 성과 계산 (주간 캔들 데이터가 제공되는 경우)
    weekly_metrics = {}
    if weekly_data is not None and not weekly_data.empty:
        weekly_returns = weekly_data['close'].pct_change().dropna()
        if not weekly_returns.empty:
            cumulative_weekly = (weekly_returns + 1).cumprod()
            weekly_roi = cumulative_weekly.iloc[-1] - 1  # 소수점 ROI
            weekly_metrics['weekly_roi'] = weekly_roi * 100  # % 단위
            running_max = cumulative_weekly.cummax()
            drawdowns = (cumulative_weekly - running_max) / running_max
            weekly_max_drawdown = drawdowns.min()
            weekly_metrics['weekly_max_drawdown'] = weekly_max_drawdown * 100  # % 단위
        else:
            weekly_metrics['weekly_roi'] = 0.0
            weekly_metrics['weekly_max_drawdown'] = 0.0

    result = {
        "monthly": monthly_perf,
        "weekly": weekly_metrics
    }
    logger.debug(f"[Performance] 월별 성과 계산 완료: {result}")
    return result

def calculate_overall_performance(trades):
    """
    거래 내역(trades)을 기반으로 전체 성과 지표를 계산합니다.
    산출 지표: ROI, 누적 수익률, 총 PnL, 거래 건수, 연간화 수익률, 연간화 변동성,
             샤프 지수, 소르티노 지수, 칼마 지수, 최대 낙폭, 승률, 평균 승/패, 프로핏 팩터,
             연간 거래 건수, 최대 연속 승/패 등.
    """
    initial_capital = 10000.0
    total_pnl = sum(trade.get("pnl", 0) for trade in trades)
    trade_count = len(trades)
    cumulative_return = (initial_capital + total_pnl) / initial_capital - 1
    roi = cumulative_return * 100

    # 거래 시각 기준 정렬
    sorted_trades = sorted(trades, key=lambda t: t.get("exit_time") or t.get("entry_time"))
    
    # Equity curve 계산
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
        logger.debug("[Performance] 거래 데이터 없음: 기본 성과 반환")
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
    
    # 일별 Equity DataFrame 구성
    df_equity = pd.DataFrame({"equity": equity_list}, index=pd.to_datetime(dates))
    df_equity = df_equity.groupby(df_equity.index).last()
    df_equity = df_equity.asfreq("D", method="ffill")
    daily_returns = df_equity["equity"].pct_change().dropna()
    
    annualized_vol = daily_returns.std() * np.sqrt(252) if not daily_returns.empty else 0.0
    start_date = df_equity.index.min()
    end_date = df_equity.index.max()
    total_days = (end_date - start_date).days
    if total_days > 0:
        annualized_return = (df_equity["equity"].iloc[-1] / initial_capital) ** (365 / total_days) - 1
    else:
        annualized_return = 0.0

    roll_max = df_equity["equity"].cummax()
    drawdown = roll_max - df_equity["equity"]
    max_drawdown = drawdown.max()

    sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0.0
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0.0
    sortino_ratio = annualized_return / downside_std if downside_std != 0 else 0.0
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0

    wins = [pnl for pnl in trade_pnls if pnl > 0]
    losses = [pnl for pnl in trade_pnls if pnl <= 0]
    win_rate = (len(wins) / trade_count * 100) if trade_count > 0 else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else 0.0

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

    years = total_days / 365 if total_days > 0 else 1
    trades_per_year = trade_count / years

    overall_performance = {
        "roi": roi,
        "cumulative_return": cumulative_return,
        "total_pnl": total_pnl,
        "trade_count": trade_count,
        "annualized_return": annualized_return * 100,  # %
        "annualized_volatility": annualized_vol * 100,  # %
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
    
    logger.debug(f"[Performance] 전체 성과 계산 완료: ROI={roi:.2f}%, Trade Count={trade_count}, Annualized Return={annualized_return*100:.2f}%, Sharpe Ratio={sharpe_ratio:.2f}, Max Drawdown={max_drawdown:.2f}")
    return overall_performance

def compute_performance(trades, weekly_data=None):
    """
    거래 내역(trades)과 주간 데이터(weekly_data)를 기반으로 전체 성과를 계층적으로 산출합니다.
    결과는 'overall' (전체 성과 지표), 'monthly' (월별 성과), 'weekly' (주간 전략 성과)로 구성됩니다.
    """
    overall = calculate_overall_performance(trades)
    monthly = calculate_monthly_performance(trades, weekly_data=weekly_data)
    performance_report = {
        "overall": overall,
        "monthly": monthly["monthly"],
        "weekly": monthly["weekly"]
    }
    logger.debug(f"[Performance] 최종 성과 종합 계산 완료: {performance_report}")
    return performance_report
