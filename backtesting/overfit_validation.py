# backtesting/overfit_validation.py
# 오버피팅(과적합)을 막기 위해 '학습 구간'과 '테스트 구간'을 나눠서
# 최적 파라미터를 찾고, 그것을 별도 기간에 적용해 검증하는 예시.

import pandas as pd
from datetime import timedelta

# param_sweep.py 에서 전수 조사 함수 불러오기
from backtesting.param_sweep import run_param_sweep_advanced
from backtesting.backtest_advanced import run_advanced_backtest


def train_test_validation(
    symbol: str = "BTC/USDT",
    short_timeframe: str = "4h",
    long_timeframe: str = "1d",
    train_start: str = "2021-01-01 00:00:00",
    train_end: str = "2021-06-30 23:59:59",
    test_start: str = "2021-07-01 00:00:00",
    test_end: str = "2021-12-31 23:59:59",
    account_size: float = 10_000.0
) -> dict:
    """
    (1) 훈련(Train) 구간에서 여러 파라미터를 시도해 최적값(ROI가 최고인 파라미터) 찾기
    (2) 찾은 파라미터로 테스트(Test) 구간을 백테스트
    (3) 결과를 dict 형태로 반환
    """

    # [A] 훈련 구간 파라미터 스윕
    train_results_df = run_param_sweep_advanced(
        symbol=symbol,
        short_timeframe=short_timeframe,
        long_timeframe=long_timeframe,
        account_size=account_size,
        start_date=train_start,
        end_date=train_end
    )

    if train_results_df.empty:
        print("[train_test_validation] 훈련 구간에서 매매 결과가 없습니다.")
        return {}

    # ROI(%)가 가장 높은 행(파라미터) 추출
    best_row = train_results_df.iloc[0]  # 이미 ROI 내림차순으로 sort되어 있을 것
    best_params = {
        'window': int(best_row['window']),
        'atr_multiplier': float(best_row['atr_multiplier']),
        'profit_ratio': float(best_row['profit_ratio']),
        'use_partial_tp': bool(best_row['use_partial_tp'])
    }

    # [B] 테스트 구간 백테스트 (최적 파라미터 적용)
    test_trades_df = run_advanced_backtest(
        symbol=symbol,
        short_timeframe=short_timeframe,
        long_timeframe=long_timeframe,
        window=best_params['window'],
        atr_multiplier=best_params['atr_multiplier'],
        profit_ratio=best_params['profit_ratio'],
        use_partial_take_profit=best_params['use_partial_tp'],
        account_size=account_size,
        start_date=test_start,
        end_date=test_end
    )

    return {
        'best_params': best_params,
        'train_results': train_results_df,
        'test_trades': test_trades_df
    }


def walk_forward_analysis(
    symbol: str = "BTC/USDT",
    short_timeframe: str = "4h",
    long_timeframe: str = "1d",
    overall_start: str = "2021-01-01 00:00:00",
    overall_end: str = "2022-12-31 23:59:59",
    n_splits: int = 3,
    account_size: float = 10_000.0,
    train_ratio: float = 0.5
) -> list:
    """
    전체 기간(overall_start ~ overall_end)을 n_splits 구간으로 나누어:
      - 각 구간에서 train_ratio 만큼의 기간을 '훈련(Train)'으로 사용,
      - 나머지 기간을 '테스트(Test)'로 사용,
      - 이를 순차적으로 반복(워크포워드)하며 성능을 측정.

    Returns:
        list: 각 split마다의 결과(훈련 결과, 테스트 결과, 최적 파라미터 등)가 들어 있는 dict들의 리스트
    """

    results_list = []

    start_dt = pd.to_datetime(overall_start)
    end_dt = pd.to_datetime(overall_end)
    total_days = (end_dt - start_dt).days

    if n_splits < 1 or total_days < 1:
        print("[walk_forward_analysis] 기간 또는 n_splits가 유효하지 않습니다.")
        return results_list

    # 각 split마다 사용할 일수
    split_days = total_days // n_splits

    for i in range(n_splits):
        split_start_dt = start_dt + pd.to_timedelta(split_days * i, unit='D')
        split_end_dt = start_dt + pd.to_timedelta(split_days * (i + 1), unit='D')

        # 마지막 split은 end_dt까지
        if i == n_splits - 1:
            split_end_dt = end_dt

        days_in_this_split = (split_end_dt - split_start_dt).days
        if days_in_this_split <= 1:
            continue

        # 훈련 기간 vs. 테스트 기간 분할
        train_days = int(days_in_this_split * train_ratio)

        train_start = split_start_dt
        train_end = train_start + pd.to_timedelta(train_days, unit='D')
        test_start = train_end + pd.to_timedelta(1, unit='D')
        test_end = split_end_dt

        # str 변환
        t_start_str = train_start.strftime("%Y-%m-%d %H:%M:%S")
        t_end_str = train_end.strftime("%Y-%m-%d %H:%M:%S")
        v_start_str = test_start.strftime("%Y-%m-%d %H:%M:%S")
        v_end_str = test_end.strftime("%Y-%m-%d %H:%M:%S")

        # 훈련/테스트 진행
        result_dict = train_test_validation(
            symbol=symbol,
            short_timeframe=short_timeframe,
            long_timeframe=long_timeframe,
            train_start=t_start_str,
            train_end=t_end_str,
            test_start=v_start_str,
            test_end=v_end_str,
            account_size=account_size
        )

        if not result_dict:
            # 비어있으면 넘어감
            continue

        # 결과 저장
        results_list.append({
            'split_index': i + 1,
            'train_start': t_start_str,
            'train_end': t_end_str,
            'test_start': v_start_str,
            'test_end': v_end_str,
            'best_params': result_dict.get('best_params'),
            'train_results': result_dict.get('train_results'),
            'test_trades': result_dict.get('test_trades')
        })

    return results_list
