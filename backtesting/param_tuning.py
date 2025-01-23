# backtesting/param_tuning.py

import itertools  # 여러 파라미터 조합을 생성하기 위해 사용됩니다.
import pandas as pd  # 데이터 저장 및 처리에 유용한 라이브러리입니다.

from backtesting.backtest_simple import run_simple_backtest  # 단순 백테스트 실행 함수
from backtesting.performance_metrics import calculate_mdd  # 최대 낙폭(MDD)을 계산하는 함수

def param_sweep_test(
    symbol: str = "BTC/USDT",  # 거래 종목 (기본값: 비트코인/테더)
    short_timeframe: str = "4h",  # 단기 봉의 주기 (기본값: 4시간)
    long_timeframe: str = "1d",  # 장기 봉의 주기 (기본값: 1일)
    account_size: float = 10_000.0,  # 초기 계좌 잔액 (기본값: 10,000 USDT)
    start_date: str = None,  # 백테스트 시작 날짜 (예: '2021-01-01')
    end_date: str = None  # 백테스트 종료 날짜 (예: '2021-12-31')
) -> pd.DataFrame:
    """
    다양한 파라미터 조합으로 주어진 기간 동안 백테스트를 실행하고 결과를 반환합니다.

    Parameters:
    ----------
    symbol : str
        거래할 종목 이름 (기본값: "BTC/USDT").
    short_timeframe : str
        단기 봉 주기 (기본값: "4h").
    long_timeframe : str
        장기 봉 주기 (기본값: "1d").
    account_size : float
        초기 계좌 잔액 (기본값: 10,000.0 USDT).
    start_date : str
        백테스트 시작 날짜 (예: "2021-01-01").
    end_date : str
        백테스트 종료 날짜 (예: "2021-12-31").

    Returns:
    -------
    pd.DataFrame
        각 파라미터 조합에 대한 성과 지표를 담은 DataFrame.
    """

    # (A) 백테스트에서 사용할 파라미터의 범위를 설정합니다.
    window_list = [10, 20, 30]  # 돌파 신호를 위한 기간
    atr_multiplier_list = [1.5, 2.0]  # 손절 기준이 되는 ATR(평균 진폭)의 배수
    profit_ratio_list = [0.03, 0.05]  # 고정된 익절 비율

    # 결과를 저장할 리스트를 초기화합니다.
    results = []

    # (B) itertools.product를 사용해 모든 파라미터 조합을 생성하고 순회합니다.
    for window, atr_mult, pr in itertools.product(window_list, atr_multiplier_list, profit_ratio_list):
        print(f"\n[Param Test] window={window}, atr_multiplier={atr_mult}, profit_ratio={pr}")

        # 각 파라미터 조합으로 백테스트를 실행합니다.
        trades_df = run_simple_backtest(
            symbol=symbol,  # 거래 종목
            short_timeframe=short_timeframe,  # 단기 봉 주기
            long_timeframe=long_timeframe,  # 장기 봉 주기
            window=window,  # 돌파 신호를 위한 기간
            atr_multiplier=atr_mult,  # 손절 기준 ATR 배수
            profit_ratio=pr,  # 고정 익절 비율
            account_size=account_size,  # 초기 계좌 잔액
            start_date=start_date,  # 시작 날짜
            end_date=end_date  # 종료 날짜
        )

        # 거래 데이터가 없거나 비어 있으면 다음 조합으로 넘어갑니다.
        if trades_df is None or trades_df.empty:
            continue

        # (C) 각 조합의 성과 지표를 계산합니다.
        initial_balance = account_size  # 초기 잔액
        total_pnl = trades_df['pnl'].sum()  # 총 수익 (profit and loss)
        final_balance = initial_balance + total_pnl  # 최종 계좌 잔액
        roi_percent = (final_balance - initial_balance) / initial_balance * 100.0  # 총 수익률(ROI)
        mdd_percent = calculate_mdd(trades_df, initial_balance=initial_balance)  # 최대 낙폭(MDD)

        num_trades = len(trades_df)  # 총 거래 횟수
        wins = (trades_df['pnl'] > 0).sum()  # 수익이 난 거래의 개수
        win_rate = (wins / num_trades * 100.0) if num_trades > 0 else 0.0  # 승률(%)

        # (D) 계산된 결과를 리스트에 추가합니다.
        results.append({
            'window': window,  # 돌파 신호를 위한 기간
            'atr_multiplier': atr_mult,  # ATR 배수
            'profit_ratio': pr,  # 고정 익절 비율
            'num_trades': num_trades,  # 총 거래 횟수
            'win_rate(%)': round(win_rate, 2),  # 승률 (%)
            'final_balance': round(final_balance, 2),  # 최종 잔액
            'ROI(%)': round(roi_percent, 2),  # 총 수익률 (%)
            'MDD(%)': round(mdd_percent, 2),  # 최대 낙폭 (%)
        })

    # (E) 모든 결과를 DataFrame으로 변환하여 반환합니다.
    results_df = pd.DataFrame(results)
    return results_df
