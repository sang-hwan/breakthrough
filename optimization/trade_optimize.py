# optimization/trade_optimize.py
import optuna
import logging
import pandas as pd
import numpy as np
from logs.log_config import setup_logger
from backtesting.backtester import Backtester  # 백테스트 수행 클래스
from parameters.trading_parameters import ConfigManager, TradingConfig  # 거래 파라미터 관리
from data.data_utils import get_unique_symbol_list  # 유니크 심볼 목록 조회

logger = setup_logger(__name__)

class DynamicParameterOptimizer:
    """
    DynamicParameterOptimizer는 거래 모듈의 파라미터 최적화를 수행합니다.
    
    백테스트 결과에 따라 제안된 파라미터 조합의 성능을 평가하며,
    과적합 방지를 위해 학습/테스트 ROI 차이, 홀드아웃 ROI 부족, 그리고 정규화 패널티를 적용합니다.
    
    Attributes:
        n_trials (int): 최적화 시도 횟수.
        study (optuna.study.Study): 최적화 결과를 저장하는 study 객체.
        config_manager (ConfigManager): 거래 파라미터의 기본값 및 검증 객체.
        assets (list): 최적화에 사용할 자산 목록.
    """

    def __init__(self, n_trials=10, assets=None):
        """
        초기화 함수.
        
        Parameters:
            n_trials (int): 최적화 시도 횟수 (기본값: 10).
            assets (list, optional): 최적화 대상 자산 목록. 제공되지 않으면 데이터베이스 또는 기본값 사용.
        """
        self.n_trials = n_trials
        self.study = None
        self.config_manager = ConfigManager()
        self.assets = assets if assets is not None else (get_unique_symbol_list() or ["BTC/USDT", "ETH/USDT", "XRP/USDT"])

    def objective(self, trial):
        """
        optuna의 objective 함수.
        
        각 자산과 기간에 대해 백테스트를 실행하여 성능 점수를 산출하고,
        학습과 테스트, 홀드아웃 단계의 결과 차이에 따른 과적합 패널티를 적용합니다.
        
        Parameters:
            trial (optuna.trial.Trial): 현재 최적화 시도의 trial 객체.
        
        Returns:
            float: 최종 성능 점수 (낮을수록 우수).
        """
        try:
            base_params = self.config_manager.get_defaults()
            suggested_params = {
                "hmm_confidence_threshold": trial.suggest_float("hmm_confidence_threshold", 0.7, 0.95),
                "liquidity_info": trial.suggest_categorical("liquidity_info", ["high", "low"]),
                "atr_multiplier": trial.suggest_float("atr_multiplier", 1.5, 3.0),
                "profit_ratio": trial.suggest_float("profit_ratio", 0.05, 0.15),
                "risk_per_trade": trial.suggest_float("risk_per_trade", 0.005, 0.02),
                "scale_in_threshold": trial.suggest_float("scale_in_threshold", 0.01, 0.03),
                "partial_exit_ratio": trial.suggest_float("partial_exit_ratio", 0.4, 0.6),
                "partial_profit_ratio": trial.suggest_float("partial_profit_ratio", 0.02, 0.04),
                "final_profit_ratio": trial.suggest_float("final_profit_ratio", 0.05, 0.1),
                "weekly_breakout_threshold": trial.suggest_float("weekly_breakout_threshold", 0.005, 0.02),
                "weekly_momentum_threshold": trial.suggest_float("weekly_momentum_threshold", 0.3, 0.7),
                "risk_reward_ratio": trial.suggest_float("risk_reward_ratio", 1.0, 5.0)
            }
            dynamic_params = {**base_params, **suggested_params}
            dynamic_params = self.config_manager.validate_params(dynamic_params)

            try:
                _ = TradingConfig(**dynamic_params)
            except Exception as e:
                logger.error("Validation error in dynamic_params: " + str(e), exc_info=True)
                return 1e6

            splits = [{
                "train_start": "2018-06-01",
                "train_end": "2020-12-31",
                "test_start": "2021-01-01",
                "test_end": "2023-12-31"
            }]
            holdout = {"holdout_start": "2024-01-01", "holdout_end": "2025-02-01"}

            total_score, num_evaluations = 0.0, 0

            for split in splits:
                for asset in self.assets:
                    symbol_key = asset.replace("/", "").lower()
                    try:
                        # 학습 단계 백테스트
                        bt_train = Backtester(symbol=asset, account_size=10000)
                        bt_train.load_data(
                            short_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                            long_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                            short_tf="4h", long_tf="1d",
                            start_date=split["train_start"], end_date=split["train_end"]
                        )
                        trades_train, _ = bt_train.run_backtest(dynamic_params)
                        roi_train = sum(trade.get("pnl", 0) for trade in trades_train) / 10000 * 100

                        # 테스트 단계 백테스트
                        bt_test = Backtester(symbol=asset, account_size=10000)
                        bt_test.load_data(
                            short_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                            long_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                            short_tf="4h", long_tf="1d",
                            start_date=split["test_start"], end_date=split["test_end"]
                        )
                        trades_test, _ = bt_test.run_backtest(dynamic_params)
                        roi_test = sum(trade.get("pnl", 0) for trade in trades_test) / 10000 * 100

                        # 홀드아웃 단계 백테스트
                        bt_holdout = Backtester(symbol=asset, account_size=10000)
                        bt_holdout.load_data(
                            short_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                            long_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                            short_tf="4h", long_tf="1d",
                            start_date=holdout["holdout_start"], end_date=holdout["holdout_end"]
                        )
                        trades_holdout, _ = bt_holdout.run_backtest(dynamic_params)
                        roi_holdout = sum(trade.get("pnl", 0) for trade in trades_holdout) / 10000 * 100

                        # 과적합 패널티 적용
                        overfit_penalty = abs(roi_train - roi_test)
                        holdout_penalty = 0 if roi_holdout >= 2.0 else (2.0 - roi_holdout) * 20
                        score = -roi_test + overfit_penalty + holdout_penalty
                        total_score += score
                        num_evaluations += 1
                    except Exception as e:
                        logger.error(f"Backtest error for {asset} in split {split}: {e}", exc_info=True)
                        continue

            if num_evaluations == 0:
                return 1e6
            avg_score = total_score / num_evaluations
            reg_penalty = 0.1 * sum(
                (dynamic_params.get(key, base_params.get(key, 1.0)) - base_params.get(key, 1.0)) ** 2
                for key in ["atr_multiplier", "profit_ratio", "risk_per_trade", "scale_in_threshold",
                            "weekly_breakout_threshold", "weekly_momentum_threshold"]
            )
            return avg_score + reg_penalty
        except Exception as e:
            logger.error("Unexpected error in objective: " + str(e), exc_info=True)
            return 1e6

    def optimize(self):
        """
        최적화 실행 함수.
        
        n_trials 횟수 동안 objective 함수를 최적화하여 최적의 거래 파라미터 조합을 도출합니다.
        
        Returns:
            optuna.trial.FrozenTrial: 최적의 파라미터 조합과 해당 점수를 포함하는 trial 객체.
        """
        sampler = optuna.samplers.TPESampler(seed=42)
        self.study = optuna.create_study(direction="minimize", sampler=sampler)
        self.study.optimize(self.objective, n_trials=self.n_trials)
        best_trial = self.study.best_trial
        logger.debug(f"Best trial: {best_trial.number}, Value: {best_trial.value:.2f}")
        logger.debug(f"Best parameters: {best_trial.params}")
        return best_trial

# =============================================================================
# 보고서 및 성과 계산 관련 함수들
# =============================================================================

def generate_final_report(performance_data, symbol=None):
    """
    백테스트 최종 성과 보고서를 생성하여 로깅합니다.
    
    Parameters:
        performance_data (dict): 전체, 주간, 월별 성과 지표를 담은 딕셔너리.
        symbol (str, optional): 특정 심볼이 주어지면 헤더에 포함.
    """
    overall = performance_data.get("overall", {})
    report_lines = []
    header = f"=== FINAL BACKTEST PERFORMANCE REPORT for {symbol} ===" if symbol else "=== FINAL BACKTEST PERFORMANCE REPORT ==="
    report_lines.append(header)
    report_lines.append(f"Overall ROI: {overall.get('roi', 0):.2f}%")
    report_lines.append(f"Cumulative Return: {overall.get('cumulative_return', 0):.2f}")
    report_lines.append(f"Total PnL: {overall.get('total_pnl', 0):.2f}")
    report_lines.append(f"Trade Count: {overall.get('trade_count', 0)}")
    report_lines.append("")
    report_lines.append("Performance Overview:")
    report_lines.append(f"  Annualized Return: {overall.get('annualized_return', 0):.2f}%")
    report_lines.append(f"  Annualized Volatility: {overall.get('annualized_volatility', 0):.2f}%")
    report_lines.append(f"  Sharpe Ratio: {overall.get('sharpe_ratio', 0):.2f}")
    report_lines.append(f"  Sortino Ratio: {overall.get('sortino_ratio', 0):.2f}")
    report_lines.append(f"  Calmar Ratio: {overall.get('calmar_ratio', 0):.2f}")
    report_lines.append(f"  Maximum Drawdown: {overall.get('max_drawdown', 0):.2f}")
    report_lines.append("")
    report_lines.append("Weekly Strategy Metrics:")
    weekly = performance_data.get("weekly", {})
    report_lines.append(f"  Weekly ROI: {weekly.get('weekly_roi', 0):.2f}%")
    report_lines.append(f"  Weekly Max Drawdown: {weekly.get('weekly_max_drawdown', 0):.2f}%")
    report_lines.append("")
    report_lines.append("Trading Stats:")
    report_lines.append(f"  Win Rate: {overall.get('win_rate', 0):.2f}%")
    report_lines.append(f"  Average Win: {overall.get('avg_win', 0):.2f}")
    report_lines.append(f"  Average Loss: {overall.get('avg_loss', 0):.2f}")
    report_lines.append(f"  Profit Factor: {overall.get('profit_factor', 0):.2f}")
    report_lines.append(f"  Trades per Year: {overall.get('trades_per_year', 0):.2f}")
    report_lines.append(f"  Max Consecutive Wins: {overall.get('max_consecutive_wins', 0)}")
    report_lines.append(f"  Max Consecutive Losses: {overall.get('max_consecutive_losses', 0)}")
    report_lines.append("")
    report_lines.append("Monthly Performance:")
    monthly = performance_data.get("monthly", {})
    for month in sorted(monthly.keys()):
        data = monthly[month]
        status = "TARGET MET" if data["roi"] >= 2.0 else "TARGET NOT MET"
        report_lines.append(f"  {month}: ROI {data['roi']:.2f}% (Trades: {data['trade_count']}) --> {status}")
    report_lines.append("=========================================")
    
    report_str = "\n".join(report_lines)
    logger.info(report_str)

def generate_parameter_sensitivity_report(param_name, results):
    """
    파라미터 민감도 분석 결과 보고서를 생성하여 로깅합니다.
    
    Parameters:
        param_name (str): 분석 대상 파라미터 이름.
        results (dict): 파라미터 값과 성과 지표 결과의 매핑.
    """
    report_lines = []
    report_lines.append("=== FINAL PARAMETER SENSITIVITY REPORT ===")
    
    if all(isinstance(k, tuple) for k in results.keys()):
        report_lines.append("Multi-Parameter Analysis Results:")
        for combo_key, metrics in results.items():
            combo_str = ", ".join([f"{p}={v:.4f}" for p, v in combo_key])
            report_lines.append(f"Combination: {combo_str}")
            if metrics is not None:
                for metric_name, stats in metrics.items():
                    report_lines.append(f"  {metric_name}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")
            else:
                report_lines.append("  Error during backtesting for this combination.")
            report_lines.append("")
    else:
        report_lines.append(f"Analyzed Parameter: {param_name}")
        report_lines.append("Results:")
        for val in sorted(results.keys()):
            result = results[val]
            if result is not None:
                roi = result.get("roi", 0)
                report_lines.append(f"{param_name} = {val:.4f} -> ROI: {roi:.2f}%")
            else:
                report_lines.append(f"{param_name} = {val:.4f} -> ROI: Error")
    report_lines.append("==========================================")
    
    report_str = "\n".join(report_lines)
    logger.info(report_str)

def generate_weekly_signal_report(weekly_signal_counts):
    """
    주간 신호 보고서를 생성하여 로깅합니다.
    
    Parameters:
        weekly_signal_counts (dict): (logger 이름, 파일명, 함수명)별 주간 신호 발생 횟수를 담은 딕셔너리.
    """
    report_lines = []
    report_lines.append("=== WEEKLY SIGNAL REPORT ===")
    for (logger_name, filename, funcname), count in weekly_signal_counts.items():
        report_lines.append(f"{filename}:{funcname} (logger: {logger_name}) - 주간 신호 {count}회 발생")
    report_lines.append("==========================================")
    
    report_str = "\n".join(report_lines)
    logger.info(report_str)

def calculate_overall_performance(trades):
    """
    전체 성과 계산 함수.
    
    거래 내역을 바탕으로 총 수익률, 연간 수익률, 변동성, 샤프비율 등 성과 지표를 계산합니다.
    
    Parameters:
        trades (list): 거래 내역 리스트. 각 거래는 딕셔너리 형식.
        
    Returns:
        dict: ROI, 총 pnl, 거래 횟수, 연간 수익률, 변동성, 샤프비율 등 성과 지표를 포함.
    """
    initial_capital = 10000.0
    total_pnl = 0.0
    trade_count = len(trades)
    sorted_trades = sorted(trades, key=lambda t: t.get("exit_time") or t.get("entry_time"))
    dates, equity_list, trade_pnls = [], [], []
    equity = initial_capital
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
    if not dates:
        return {
            "roi": 0.0, "cumulative_return": 0.0, "total_pnl": 0.0,
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
    월별 및 주간 성과 지표를 계산합니다.
    
    Parameters:
        trades (list): 거래 내역 리스트.
        weekly_data (pd.DataFrame, optional): 주간 가격 데이터.
    
    Returns:
        dict: {"monthly": 월별 성과 지표, "weekly": 주간 성과 지표}
    """
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

def compute_performance(trades, weekly_data=None):
    """
    전체, 월별, 주간 성과 지표를 계산하여 반환합니다.
    
    Parameters:
        trades (list): 거래 내역 리스트.
        weekly_data (pd.DataFrame, optional): 주간 가격 데이터.
    
    Returns:
        dict: {"overall": 전체 성과, "monthly": 월별 성과, "weekly": 주간 성과}
    """
    overall = calculate_overall_performance(trades)
    monthly = calculate_monthly_performance(trades, weekly_data=weekly_data)
    logger.debug("Performance report generated.")
    return {"overall": overall, "monthly": monthly["monthly"], "weekly": monthly["weekly"]}
