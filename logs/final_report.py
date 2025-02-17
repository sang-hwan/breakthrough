# logs/final_report.py
from logs.logger_config import setup_logger

logger = setup_logger(__name__)

def generate_final_report(performance_data, symbol=None):
    """
    종목별 백테스트 최종 성과 리포트를 생성합니다.
    (symbol이 전달되면 헤더에 포함)
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
    최종 파라미터 민감도 리포트를 생성합니다.
    (다중 파라미터 분석 시, 각 조합에 대한 평균/표준편차/최소/최대 지표를 출력)
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
    주간 신호 발생 건수를 요약하는 리포트를 생성합니다.
    weekly_signal_counts는 (logger_name, filename, funcName)를 key로 하는 딕셔너리입니다.
    """
    report_lines = []
    report_lines.append("=== WEEKLY SIGNAL REPORT ===")
    for (logger_name, filename, funcname), count in weekly_signal_counts.items():
        report_lines.append(f"{filename}:{funcname} (logger: {logger_name}) - 주간 신호 {count}회 발생")
    report_lines.append("==========================================")
    
    report_str = "\n".join(report_lines)
    logger.info(report_str)
