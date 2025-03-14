# logs/final_report.py
from logs.logger_config import setup_logger  # 모듈별 로거 설정 함수 임포트

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
