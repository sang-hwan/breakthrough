# backtesting/performance_metrics.py

# 데이터 분석과 수치 연산을 위한 라이브러리
import pandas as pd
import numpy as np

def calculate_monthly_performance(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    월별 손익(PnL), 매매 횟수, 승률을 계산하는 함수.

    주요 기능:
    ----------
    - 월(YYYY-MM) 단위로 매매 데이터를 그룹화.
    - 각 월별 총 손익, 매매 횟수, 승률을 계산.

    매개변수:
    ----------
    - trades_df (DataFrame): 매매 기록 데이터프레임.
      필요한 컬럼: exit_time(datetime), pnl(float).

    반환값:
    ----------
    - DataFrame: 월별 성과를 요약한 데이터프레임.
    """
    # 연월(YYYY-MM) 기준으로 그룹화
    trades_df['year_month'] = trades_df['exit_time'].dt.to_period('M')

    # 그룹별 성과 계산
    grouped = trades_df.groupby('year_month')
    results = []

    for ym, grp in grouped:
        total_pnl = grp['pnl'].sum()  # 총 손익
        num_trades = len(grp)        # 매매 횟수
        win_trades = (grp['pnl'] > 0).sum()  # 이긴 매매 수
        win_rate = win_trades / num_trades * 100.0 if num_trades > 0 else 0.0  # 승률 계산

        results.append({
            'year_month': str(ym),
            'total_pnl': total_pnl,
            'num_trades': num_trades,
            'win_rate(%)': win_rate
        })

    return pd.DataFrame(results)


def calculate_yearly_performance(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    연도별 손익(PnL), 매매 횟수, 승률을 계산하는 함수.

    매개변수:
    ----------
    - trades_df (DataFrame): 매매 기록 데이터프레임.
      필요한 컬럼: exit_time(datetime), pnl(float).

    반환값:
    ----------
    - DataFrame: 연도별 성과를 요약한 데이터프레임.
    """
    # 연도별 그룹화
    trades_df['year'] = trades_df['exit_time'].dt.year
    grouped = trades_df.groupby('year')

    results = []
    for y, grp in grouped:
        total_pnl = grp['pnl'].sum()
        num_trades = len(grp)
        win_trades = (grp['pnl'] > 0).sum()
        win_rate = win_trades / num_trades * 100.0 if num_trades > 0 else 0.0

        results.append({
            'year': y,
            'total_pnl': total_pnl,
            'num_trades': num_trades,
            'win_rate(%)': win_rate
        })

    return pd.DataFrame(results)


def calculate_mdd(trades_df: pd.DataFrame, initial_balance: float) -> float:
    """
    최대낙폭(MDD)을 계산하는 함수.

    주요 기능:
    ----------
    - 매매 기록을 순서대로 처리하여 최대낙폭(MDD)을 계산.

    매개변수:
    ----------
    - trades_df (DataFrame): 매매 기록 데이터프레임.
      필요한 컬럼: exit_time(datetime), pnl(float).
    - initial_balance (float): 초기 계좌 잔고.

    반환값:
    ----------
    - float: 최대낙폭(MDD) 값(음수, % 단위).
    """
    # 시간 순 정렬
    trades_df = trades_df.sort_values(by='exit_time')

    # 잔고 추적
    equity_list = []
    current_balance = initial_balance

    for _, row in trades_df.iterrows():
        current_balance += row['pnl']
        equity_list.append(current_balance)

    # MDD 계산
    equity_arr = np.array(equity_list)
    peak_arr = np.maximum.accumulate(equity_arr)  # 최고점 추적
    drawdown_arr = (equity_arr - peak_arr) / peak_arr  # 낙폭 계산
    mdd = drawdown_arr.min() * 100.0  # %로 변환
    return mdd


def print_performance_report(trades_df: pd.DataFrame, initial_balance: float) -> None:
    """
    전체 성과를 요약 출력하는 함수.

    주요 기능:
    ----------
    - 월별, 연도별 성과와 MDD(최대낙폭), ROI, 승률 등을 출력.

    매개변수:
    ----------
    - trades_df (DataFrame): 매매 기록 데이터프레임.
    - initial_balance (float): 초기 계좌 잔고.

    반환값:
    ----------
    - None
    """
    if trades_df.empty:
        print("No trades to report.")
        return

    # 성과 계산
    monthly_df = calculate_monthly_performance(trades_df)
    yearly_df = calculate_yearly_performance(trades_df)
    total_pnl = trades_df['pnl'].sum()
    final_balance = initial_balance + total_pnl
    mdd = calculate_mdd(trades_df, initial_balance=initial_balance)

    # 출력
    print("=== (A) 월별 성과 ===")
    print(monthly_df)

    print("\n=== (B) 연도별 성과 ===")
    print(yearly_df)

    print("\n=== (C) 전체 성과 ===")
    print(f"  - 초기 잔고       : {initial_balance:.2f}")
    print(f"  - 최종 잔고       : {final_balance:.2f}")
    print(f"  - 총 손익         : {total_pnl:.2f}")
    print(f"  - ROI(%)          : {(final_balance - initial_balance) / initial_balance * 100:.2f}%")
    print(f"  - 최대낙폭(MDD)   : {mdd:.2f}%")

    # 매매 통계
    num_trades = len(trades_df)
    wins = (trades_df['pnl'] > 0).sum()
    win_rate = wins / num_trades * 100.0 if num_trades > 0 else 0.0
    print(f"  - 총 매매 횟수    : {num_trades}")
    print(f"  - 승률(%)         : {win_rate:.2f}%")
