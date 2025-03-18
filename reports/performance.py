# backtesting/performance.py

import pandas as pd
import numpy as np
# 로깅 설정을 위한 모듈 임포트
from logging.logger_config import setup_logger

# 모듈 전용 로거 객체 생성
logger = setup_logger(__name__)

def calculate_overall_performance(trades):
    """
    전체 성과 계산 함수
    -------------------
    거래 내역(trades)을 바탕으로 총 수익률, 연간 수익률, 변동성, 샤프비율 등 전체 백테스트 성과 지표를 계산합니다.
    
    Parameters:
      trades (list): 거래 내역 리스트. 각 거래는 딕셔너리 형식으로, entry_time, exit_time, pnl 등 포함
      
    Returns:
      dict: ROI, 총 pnl, 거래 수, 연간 수익률, 변동성, 샤프비율, 소르티노 비율, 칼마 비율, 최대 낙폭, 승률, 평균 이익/손실, 
            프로핏 팩터, 연간 거래 수, 최대 연속 승/패 횟수 등의 성과 지표를 포함하는 딕셔너리
    """
    initial_capital = 10000.0
    total_pnl = 0.0
    trade_count = len(trades)
    cumulative_return = 0.0
    # 거래 내역을 시간순으로 정렬 (exit_time 또는 entry_time 기준)
    sorted_trades = sorted(trades, key=lambda t: t.get("exit_time") or t.get("entry_time"))
    
    dates, equity_list, trade_pnls = [], [], []
    equity = initial_capital
    # 거래별 손익을 누적하여 계좌 잔액을 계산
    for trade in sorted_trades:
        dt = trade.get("exit_time") or trade.get("entry_time")
        if not dt:
            continue
        pnl = trade.get("pnl", 0)
        if isinstance(pnl, list):
            pnl = sum(pnl)
        equity += pnl
        dates.append(pd.to_datetime(dt))
        equity_list.append(equity)
        trade_pnls.append(pnl)
    
    # 거래 내역이 없으면 모든 성과 지표를 0으로 반환
    if not dates:
        return {
            "roi": 0.0, "cumulative_return": 0.0, "total_pnl": 0.0,
            "trade_count": trade_count, "annualized_return": 0.0, "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "calmar_ratio": 0.0, "max_drawdown": 0.0,
            "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "profit_factor": 0.0,
            "trades_per_year": 0.0, "max_consecutive_wins": 0, "max_consecutive_losses": 0
        }
    
    # 계좌 잔액 변화 데이터를 일별 데이터프레임으로 변환 (forward fill)
    df_eq = pd.DataFrame({"equity": equity_list}, index=pd.to_datetime(dates)).groupby(level=0).last().asfreq("D", method="ffill")
    # 일별 수익률 계산
    daily_returns = df_eq["equity"].pct_change().dropna()
    annualized_vol = daily_returns.std() * np.sqrt(252) if not daily_returns.empty else 0.0
    total_days = (df_eq.index.max() - df_eq.index.min()).days
    annualized_return = (df_eq["equity"].iloc[-1] / initial_capital) ** (365 / total_days) - 1 if total_days > 0 else 0.0
    # 최대 낙폭 계산: 누적 최고치 대비 하락 폭의 최대값
    max_drawdown = (df_eq["equity"].cummax() - df_eq["equity"]).max()
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0.0
    # 하락 수익률만 모아 소르티노 비율 계산
    downside = daily_returns[daily_returns < 0]
    sortino_ratio = annualized_return / (downside.std() * np.sqrt(252)) if not downside.empty and downside.std() != 0 else 0.0
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0
    # 승리 및 패배 거래 분리하여 평균 이익/손실 계산
    wins = [pnl for pnl in trade_pnls if pnl > 0]
    losses = [pnl for pnl in trade_pnls if pnl <= 0]
    win_rate = (len(wins) / trade_count * 100) if trade_count else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else 0.0

    # 최대 연속 승리 및 패배 계산
    wins_series = pd.Series(np.array(trade_pnls) > 0)
    max_consec_wins = max((len(list(g)) for k, g in wins_series.groupby(wins_series.ne(wins_series.shift()).cumsum())), default=0)
    losses_series = pd.Series(np.array(trade_pnls) <= 0)
    max_consec_losses = max((len(list(g)) for k, g in losses_series.groupby(losses_series.ne(losses_series.shift()).cumsum())), default=0)
    
    years = total_days / 365 if total_days > 0 else 1
    trades_per_year = trade_count / years
    overall = {
        "roi": (roi := (total_pnl / initial_capital * 100)),
        "cumulative_return": (initial_capital + total_pnl) / initial_capital - 1,
        "total_pnl": total_pnl,
        "trade_count": trade_count,
        "annualized_return": annualized_return * 100,
        "annualized_volatility": annualized_vol * 100,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "trades_per_year": trades_per_year,
        "max_consecutive_wins": max_consec_wins,
        "max_consecutive_losses": max_consec_losses
    }
    logger.debug("Overall performance calculated.")
    return overall

def calculate_monthly_performance(trades, weekly_data=None):
    """
    월별 성과 계산 함수
    --------------------
    거래 내역을 바탕으로 월별 ROI, 거래 횟수, 총 손익 등을 계산하며, 
    추가로 주간 데이터가 있으면 주간 성과 지표(ROI, 최대 낙폭)도 계산합니다.
    
    Parameters:
      trades (list): 거래 내역 리스트
      weekly_data (pd.DataFrame, optional): 주간 가격 데이터 (종가 등 포함)
      
    Returns:
      dict: {"monthly": {월별 성과 지표}, "weekly": {주간 성과 지표}}
    """
    monthly_data = {}
    # 거래별로 종료 시각을 기준으로 월 추출하여 월별 거래 손익 리스트 구성
    for trade in trades:
        exit_time = trade.get("exit_time") or trade.get("entry_time")
        if not exit_time:
            continue
        month = exit_time.strftime("%Y-%m") if hasattr(exit_time, "strftime") else exit_time[:7]
        monthly_data.setdefault(month, []).append(trade.get("pnl", 0))
    
    # 월별 ROI, 거래 횟수, 총 손익 계산
    monthly_perf = {month: {
        "roi": (sum(pnls) / 10000.0) * 100,
        "trade_count": len(pnls),
        "total_pnl": sum(pnls)
    } for month, pnls in monthly_data.items()}
    
    weekly_metrics = {}
    if weekly_data is not None and not weekly_data.empty:
        # 주간 수익률 계산 및 누적 수익률, 최대 낙폭 산출
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

def compute_performance(trades, weekly_data=None):
    """
    성과 보고서 생성 함수
    -----------------------
    전체 거래 내역(trades)과 주간 데이터(있는 경우)를 바탕으로 전체 및 월별, 주간 성과 지표를 계산하여 보고서를 생성합니다.
    
    Parameters:
      trades (list): 거래 내역 리스트
      weekly_data (pd.DataFrame, optional): 주간 가격 데이터
      
    Returns:
      dict: {"overall": 전체 성과 지표, "monthly": 월별 성과 지표, "weekly": 주간 성과 지표}
    """
    overall = calculate_overall_performance(trades)
    monthly = calculate_monthly_performance(trades, weekly_data=weekly_data)
    logger.debug("Performance report generated.")
    return {"overall": overall, "monthly": monthly["monthly"], "weekly": monthly["weekly"]}
