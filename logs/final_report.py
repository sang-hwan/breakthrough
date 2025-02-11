# logs/final_report.py
from logs.logger_config import setup_logger

logger = setup_logger(__name__)

def generate_final_report(performance_data, symbol=None):
    """
    종목별 성과 리포트를 생성합니다.
    symbol 인자가 전달되면 헤더에 심볼명을 포함합니다.
    최종 성과 지표는 performance_data["overall"]에 저장되어 있으므로,
    해당 서브 딕셔너리에서 값을 추출하도록 수정합니다.
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
    Parameter Sensitivity Report 생성 (최종 로그용).
    다중 파라미터 분석의 경우, results는 {param1: {value: metrics, ...}, param2: {value: metrics, ...}, ...} 형식입니다.
    """
    report_lines = []
    report_lines.append("=== FINAL PARAMETER SENSITIVITY REPORT ===")
    
    if isinstance(results, dict) and all(isinstance(v, dict) for v in results.values()):
        # 다중 파라미터 모드
        for p, res in results.items():
            report_lines.append(f"Parameter: {p}")
            for val in sorted(res.keys()):
                result = res[val]
                if result is not None:
                    roi = result.get("roi", 0)
                    report_lines.append(f"  {p} = {val:.4f} -> ROI: {roi:.2f}%")
                else:
                    report_lines.append(f"  {p} = {val:.4f} -> ROI: Error")
            report_lines.append("")  # 파라미터 간 빈 줄 추가
    else:
        # 단일 파라미터 모드 (지원하지 않음)
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
