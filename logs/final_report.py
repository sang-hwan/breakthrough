# logs/final_report.py
from logs.logger_config import setup_logger

logger = setup_logger(__name__)

def generate_final_report(performance_data, symbol=None):
    """
    종목별 성과 리포트를 생성합니다.
    symbol 인자가 전달되면 헤더에 심볼명을 포함합니다.
    """
    report_lines = []
    header = f"=== FINAL BACKTEST PERFORMANCE REPORT for {symbol} ===" if symbol else "=== FINAL BACKTEST PERFORMANCE REPORT ==="
    report_lines.append(header)
    report_lines.append(f"Overall ROI: {performance_data.get('roi', 0):.2f}%")
    report_lines.append(f"Cumulative Return: {performance_data.get('cumulative_return', 0):.2f}")
    report_lines.append(f"Total PnL: {performance_data.get('total_pnl', 0):.2f}")
    report_lines.append(f"Trade Count: {performance_data.get('trade_count', 0)}")
    report_lines.append("")
    report_lines.append("Performance Overview:")
    report_lines.append(f"  Annualized Return: {performance_data.get('annualized_return', 0):.2f}%")
    report_lines.append(f"  Annualized Volatility: {performance_data.get('annualized_volatility', 0):.2f}%")
    report_lines.append(f"  Sharpe Ratio: {performance_data.get('sharpe_ratio', 0):.2f}")
    report_lines.append(f"  Sortino Ratio: {performance_data.get('sortino_ratio', 0):.2f}")
    report_lines.append(f"  Calmar Ratio: {performance_data.get('calmar_ratio', 0):.2f}")
    report_lines.append(f"  Maximum Drawdown: {performance_data.get('max_drawdown', 0):.2f}")
    report_lines.append("")
    report_lines.append("Trading Stats:")
    report_lines.append(f"  Win Rate: {performance_data.get('win_rate', 0):.2f}%")
    report_lines.append(f"  Average Win: {performance_data.get('avg_win', 0):.2f}")
    report_lines.append(f"  Average Loss: {performance_data.get('avg_loss', 0):.2f}")
    report_lines.append(f"  Profit Factor: {performance_data.get('profit_factor', 0):.2f}")
    report_lines.append(f"  Trades per Year: {performance_data.get('trades_per_year', 0):.2f}")
    report_lines.append(f"  Max Consecutive Wins: {performance_data.get('max_consecutive_wins', 0)}")
    report_lines.append(f"  Max Consecutive Losses: {performance_data.get('max_consecutive_losses', 0)}")
    report_lines.append("")
    report_lines.append("Monthly Performance:")
    monthly = performance_data.get("monthly_performance", {})
    for month in sorted(monthly.keys()):
        data = monthly[month]
        status = "TARGET MET" if data["roi"] >= 2.0 else "TARGET NOT MET"
        report_lines.append(f"  {month}: ROI {data['roi']:.2f}% (Trades: {data['trade_count']}) --> {status}")
    report_lines.append("=========================================")
    
    report_str = "\n".join(report_lines)
    logger.info(report_str)

def generate_parameter_sensitivity_report(param_name, results):
    """
    Parameter Sensitivity Report 생성 (최종 로그용).
    """
    report_lines = []
    report_lines.append("=== FINAL PARAMETER SENSITIVITY REPORT ===")
    report_lines.append(f"Analyzed Parameter: {param_name}")
    report_lines.append("Results:")
    for val in sorted(results.keys()):
        roi = results[val]
        if roi is not None:
            report_lines.append(f"{param_name} = {val:.4f} -> ROI: {roi:.2f}%")
        else:
            report_lines.append(f"{param_name} = {val:.4f} -> ROI: Error")
    report_lines.append("==========================================")
    report_str = "\n".join(report_lines)
    logger.info(report_str)
