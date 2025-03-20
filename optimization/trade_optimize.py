# optimization/trade_optimize.py
import optuna
from logs.log_config import setup_logger  # 로거 설정 함수
from backtesting.backtester import Backtester  # 백테스트 수행 클래스
from parameters.config_manager import ConfigManager, TradingConfig  # 기본 파라미터 관리 및 검증을 위한 클래스
from data.db.db_manager import get_unique_symbol_list  # 데이터베이스에서 유니크 심볼 목록을 가져오는 함수

# 전역 logger 객체: 모듈 내에서 로깅에 사용
logger = setup_logger(__name__)

class DynamicParameterOptimizer:
    def __init__(self, n_trials=10, assets=None):
        """
        동적 파라미터 최적화 클래스의 생성자.
        
        목적:
          - 최적화 시도 횟수와 대상 자산 목록을 초기화합니다.
        
        Parameters:
            n_trials (int): 최적화 시도 횟수 (기본값: 10).
            assets (list, optional): 최적화에 사용할 자산 목록.
                                    주어지지 않을 경우 데이터베이스에서 유니크 심볼을 가져오거나 기본 목록 사용.
        """
        self.n_trials = n_trials  # 최적화 시도 횟수 저장
        self.study = None         # 최적화 결과를 저장할 optuna study 객체 (초기에는 None)
        self.config_manager = ConfigManager()  # 파라미터 기본값 및 유효성 검증을 위한 객체 생성
        # 자산 목록이 주어지지 않으면 데이터베이스에서 가져오거나, 없으면 기본 자산 사용
        self.assets = assets if assets is not None else (get_unique_symbol_list() or ["BTC/USDT", "ETH/USDT", "XRP/USDT"])

    def objective(self, trial):
        """
        optuna 최적화 objective 함수.
        
        목적:
          - 제안된 파라미터 조합에 대해 백테스트를 수행하고, 성능 점수를 계산합니다.
          - 점수가 낮을수록 해당 파라미터 조합이 더 우수하다는 의미입니다.
        
        Parameters:
            trial (optuna.trial.Trial): 현재 최적화 시도의 trial 객체.
        
        Returns:
            float: 계산된 성능 점수 (낮을수록 좋은 파라미터).
        """
        try:
            # 기본 파라미터 값 불러오기
            base_params = self.config_manager.get_defaults()
            
            # optuna를 사용하여 동적으로 파라미터 값을 추천받음
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
            # 기본 파라미터와 추천받은 파라미터를 병합하여 최종 파라미터 집합 생성
            dynamic_params = {**base_params, **suggested_params}
            # 구성 매니저를 통해 파라미터 유효성 검사 수행
            dynamic_params = self.config_manager.validate_params(dynamic_params)

            try:
                # TradingConfig 객체를 생성해 파라미터의 추가 검증을 진행
                _ = TradingConfig(**dynamic_params)
            except Exception as e:
                logger.error("Validation error in dynamic_params: " + str(e), exc_info=True)
                return 1e6  # 유효하지 않은 경우 큰 패널티 값 반환

            # 백테스트 기간 설정 (학습, 테스트, 홀드아웃 단계로 구분)
            splits = [{
                "train_start": "2018-06-01",
                "train_end": "2020-12-31",
                "test_start": "2021-01-01",
                "test_end": "2023-12-31"
            }]
            holdout = {"holdout_start": "2024-01-01", "holdout_end": "2025-02-01"}

            total_score, num_evaluations = 0.0, 0

            # 각 기간 및 자산에 대해 백테스트를 수행하여 점수를 계산
            for split in splits:
                for asset in self.assets:
                    # 심볼 키 생성 (예: "BTC/USDT" -> "btcusdt") => 테이블 이름 생성에 사용
                    symbol_key = asset.replace("/", "").lower()
                    try:
                        # 학습 단계: 지정된 기간 동안 백테스트 실행
                        bt_train = Backtester(symbol=asset, account_size=10000)
                        bt_train.load_data(
                            short_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                            long_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                            short_tf="4h", long_tf="1d",
                            start_date=split["train_start"], end_date=split["train_end"]
                        )
                        trades_train, _ = bt_train.run_backtest(dynamic_params)
                        # 학습 단계 수익률(ROI) 계산
                        roi_train = sum(trade["pnl"] for trade in trades_train) / 10000 * 100

                        # 테스트 단계: 다른 기간에 대해 백테스트 실행
                        bt_test = Backtester(symbol=asset, account_size=10000)
                        bt_test.load_data(
                            short_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                            long_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                            short_tf="4h", long_tf="1d",
                            start_date=split["test_start"], end_date=split["test_end"]
                        )
                        trades_test, _ = bt_test.run_backtest(dynamic_params)
                        roi_test = sum(trade["pnl"] for trade in trades_test) / 10000 * 100

                        # 홀드아웃 단계: 최종 검증을 위한 별도 기간
                        bt_holdout = Backtester(symbol=asset, account_size=10000)
                        bt_holdout.load_data(
                            short_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                            long_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                            short_tf="4h", long_tf="1d",
                            start_date=holdout["holdout_start"], end_date=holdout["holdout_end"]
                        )
                        trades_holdout, _ = bt_holdout.run_backtest(dynamic_params)
                        roi_holdout = sum(trade["pnl"] for trade in trades_holdout) / 10000 * 100

                        # 과적합(overfit) 패널티: 학습과 테스트 단계의 수익률 차이를 절대값으로 계산
                        overfit_penalty = abs(roi_train - roi_test)
                        # 홀드아웃 패널티: 홀드아웃 수익률이 최소 2% 미만이면 부족분에 비례해 패널티 부과
                        holdout_penalty = 0 if roi_holdout >= 2.0 else (2.0 - roi_holdout) * 20
                        # 최종 점수 계산: 테스트 수익률의 음수에 패널티들을 더함 (낮을수록 우수)
                        score = -roi_test + overfit_penalty + holdout_penalty
                        total_score += score
                        num_evaluations += 1
                    except Exception as e:
                        logger.error(f"Backtest error for {asset} in split {split}: {e}", exc_info=True)
                        # 백테스트 중 오류가 발생하면 해당 평가 건너뜀
                        continue

            if num_evaluations == 0:
                return 1e6  # 평가가 하나도 진행되지 않았을 경우 큰 페널티 반환
            avg_score = total_score / num_evaluations
            # 정규화 패널티: 주요 파라미터가 기본값에서 벗어난 정도에 따른 패널티 부과
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
        
        목적:
          - n_trials 횟수 동안 objective 함수를 최적화하여 최적의 동적 파라미터 조합을 도출합니다.
        
        Returns:
            optuna.trial.FrozenTrial: 최적의 파라미터 조합과 해당 점수를 포함하는 trial 객체.
        """
        # TPESampler: 효율적인 파라미터 탐색을 위해 Tree-structured Parzen Estimator 사용 (시드 42로 고정)
        sampler = optuna.samplers.TPESampler(seed=42)
        # 최적화 study 생성 (방향은 최소화)
        self.study = optuna.create_study(direction="minimize", sampler=sampler)
        # n_trials 횟수 만큼 objective 함수를 최적화 수행
        self.study.optimize(self.objective, n_trials=self.n_trials)
        best_trial = self.study.best_trial
        # 최적 결과를 디버그 로그로 기록
        logger.debug(f"Best trial: {best_trial.number}, Value: {best_trial.value:.2f}")
        logger.debug(f"Best parameters: {best_trial.params}")
        return best_trial

from logs.log_config import setup_logger  # 모듈별 로거 설정 함수 임포트

# 모듈 전체에서 사용할 로거 객체 생성 (현재 모듈 이름을 사용)
logger = setup_logger(__name__)

def generate_final_report(performance_data, symbol=None):
    """
    백테스트 최종 성과 보고서를 생성하여 로깅합니다.
    
    Parameters:
        performance_data (dict): 성과 데이터를 담은 딕셔너리. "overall", "weekly", "monthly" 등의 키 포함.
        symbol (str, optional): 특정 심볼(예: 종목 코드)이 주어지면 보고서 제목에 포함 (기본값: None).
    
    Returns:
        None: 보고서는 로깅을 통해 출력되며 별도의 반환값은 없습니다.
    
    주요 동작:
        - 전체 성과, 주간 성과, 거래 통계 및 월별 성과 정보를 포맷팅하여 보고서 문자열을 구성
        - 구성된 보고서를 로거를 통해 info 레벨로 기록
    """
    overall = performance_data.get("overall", {})  # 전체 성과 데이터 추출 (없으면 빈 딕셔너리 사용)
    report_lines = []  # 보고서 각 줄을 저장할 리스트
    # 심볼 제공 여부에 따라 헤더 설정
    header = f"=== FINAL BACKTEST PERFORMANCE REPORT for {symbol} ===" if symbol else "=== FINAL BACKTEST PERFORMANCE REPORT ==="
    report_lines.append(header)
    # 전체 성과 데이터 항목 추가 (ROI, 누적 수익, 총 PnL, 거래 횟수)
    report_lines.append(f"Overall ROI: {overall.get('roi', 0):.2f}%")
    report_lines.append(f"Cumulative Return: {overall.get('cumulative_return', 0):.2f}")
    report_lines.append(f"Total PnL: {overall.get('total_pnl', 0):.2f}")
    report_lines.append(f"Trade Count: {overall.get('trade_count', 0)}")
    report_lines.append("")  # 빈 줄 추가
    report_lines.append("Performance Overview:")
    report_lines.append(f"  Annualized Return: {overall.get('annualized_return', 0):.2f}%")
    report_lines.append(f"  Annualized Volatility: {overall.get('annualized_volatility', 0):.2f}%")
    report_lines.append(f"  Sharpe Ratio: {overall.get('sharpe_ratio', 0):.2f}")
    report_lines.append(f"  Sortino Ratio: {overall.get('sortino_ratio', 0):.2f}")
    report_lines.append(f"  Calmar Ratio: {overall.get('calmar_ratio', 0):.2f}")
    report_lines.append(f"  Maximum Drawdown: {overall.get('max_drawdown', 0):.2f}")
    report_lines.append("")
    report_lines.append("Weekly Strategy Metrics:")
    weekly = performance_data.get("weekly", {})  # 주간 성과 데이터 추출
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
    monthly = performance_data.get("monthly", {})  # 월별 성과 데이터 추출
    # 월별 데이터는 정렬된 키 순으로 처리
    for month in sorted(monthly.keys()):
        data = monthly[month]
        status = "TARGET MET" if data["roi"] >= 2.0 else "TARGET NOT MET"
        report_lines.append(f"  {month}: ROI {data['roi']:.2f}% (Trades: {data['trade_count']}) --> {status}")
    report_lines.append("=========================================")
    
    # 최종 보고서 문자열을 생성한 후 로거에 기록
    report_str = "\n".join(report_lines)
    logger.info(report_str)

def generate_parameter_sensitivity_report(param_name, results):
    """
    파라미터 민감도 분석 결과 보고서를 생성하여 로깅합니다.
    
    Parameters:
        param_name (str): 분석 대상 파라미터의 이름.
        results (dict): 파라미터 값과 해당 결과(ROI 등)가 매핑된 딕셔너리 또는 다중 파라미터 조합 결과.
    
    Returns:
        None: 결과는 로깅되어 출력되며 반환값은 없습니다.
    
    주요 동작:
        - 단일 또는 다중 파라미터 분석 결과에 따라 보고서 형식이 달라짐.
        - 각 파라미터(또는 조합)의 성과 지표(평균, 표준편차, 최솟값, 최댓값)를 포맷팅하여 보고서에 포함.
    """
    report_lines = []
    report_lines.append("=== FINAL PARAMETER SENSITIVITY REPORT ===")
    
    # 결과의 키가 튜플이면 다중 파라미터 분석으로 판단
    if all(isinstance(k, tuple) for k in results.keys()):
        report_lines.append("Multi-Parameter Analysis Results:")
        for combo_key, metrics in results.items():
            # 각 파라미터 조합을 "파라미터=값" 형태의 문자열로 생성
            combo_str = ", ".join([f"{p}={v:.4f}" for p, v in combo_key])
            report_lines.append(f"Combination: {combo_str}")
            if metrics is not None:
                for metric_name, stats in metrics.items():
                    report_lines.append(f"  {metric_name}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")
            else:
                report_lines.append("  Error during backtesting for this combination.")
            report_lines.append("")
    else:
        # 단일 파라미터 분석 결과 처리
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
    
    # 최종 보고서 문자열 생성 후 로깅
    report_str = "\n".join(report_lines)
    logger.info(report_str)

def generate_weekly_signal_report(weekly_signal_counts):
    """
    주간 신호 보고서를 생성하여 로깅합니다.
    
    Parameters:
        weekly_signal_counts (dict): (logger 이름, 파일명, 함수명)별 주간 신호 발생 횟수를 담은 딕셔너리.
    
    Returns:
        None: 결과는 로깅되어 출력되며 반환값은 없습니다.
    
    주요 동작:
        - 각 집계 항목을 "파일명:함수명 (logger: logger_name) - 주간 신호 count회 발생" 형태로 포맷팅
        - 최종 보고서를 로거에 기록
    """
    report_lines = []
    report_lines.append("=== WEEKLY SIGNAL REPORT ===")
    for (logger_name, filename, funcname), count in weekly_signal_counts.items():
        report_lines.append(f"{filename}:{funcname} (logger: {logger_name}) - 주간 신호 {count}회 발생")
    report_lines.append("==========================================")
    
    report_str = "\n".join(report_lines)
    logger.info(report_str)

import pandas as pd
import numpy as np
# 로깅 설정을 위한 모듈 임포트
from logs.log_config import setup_logger

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
