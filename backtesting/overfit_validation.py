# backtesting/overfit_validation.py

import pandas as pd  # 데이터 처리와 날짜 계산에 사용
import numpy as np  # 수학 연산에 사용
from datetime import timedelta  # 날짜 간격 계산에 사용

# 백테스팅 관련 함수 가져오기
from backtesting.param_tuning import param_sweep_test  # 최적의 파라미터를 찾기 위한 함수
from backtesting.backtest_simple import run_simple_backtest  # 간단한 백테스트 실행 함수

def train_test_validation(
    symbol: str = "BTC/USDT",  # 거래 대상 (예: 비트코인/테더)
    short_timeframe: str = "4h",  # 단기 봉 간격 (기본값: 4시간)
    long_timeframe: str = "1d",  # 장기 봉 간격 (기본값: 1일)
    train_start: str = "2021-01-01 00:00:00",  # 훈련 시작 날짜
    train_end: str = "2021-06-30 23:59:59",  # 훈련 종료 날짜
    test_start: str = "2021-07-01 00:00:00",  # 테스트 시작 날짜
    test_end: str = "2021-12-31 23:59:59",  # 테스트 종료 날짜
    account_size: float = 10_000.0  # 초기 계좌 잔액 (기본값: 10,000 USDT)
) -> dict:
    """
    주어진 훈련 기간에서 최적의 거래 전략 파라미터를 찾아내고,
    이를 테스트 기간에 적용하여 검증합니다.
    
    Returns:
        dict: 훈련 및 테스트 결과
            - best_params: 최적의 파라미터 (예: 이동 평균, 손익비 등)
            - train_performance: 훈련 구간의 성과 데이터
            - test_trades: 테스트 구간의 거래 데이터
    """

    print("\n=== [Train/Test Validation] ===")
    print(f"Train Period: {train_start} ~ {train_end}")  # 훈련 구간 출력
    print(f"Test Period : {test_start} ~ {test_end}")    # 테스트 구간 출력

    # (A) 훈련 단계: 다양한 파라미터 조합으로 테스트를 수행
    train_results_df = param_sweep_test(
        symbol=symbol,
        short_timeframe=short_timeframe,
        long_timeframe=long_timeframe,
        account_size=account_size,
        start_date=train_start,
        end_date=train_end
    )

    # 훈련 결과가 비어 있으면 중단
    if train_results_df.empty:
        print("No train results found.")  # 훈련 결과 없음
        return {}

    # (B) 훈련 결과 정렬: ROI(%)가 가장 높은 파라미터 선택
    best_row = train_results_df.sort_values(by='ROI(%)', ascending=False).iloc[0]
    best_params = {
        'window': best_row['window'],  # 신호 생성 기간
        'atr_multiplier': best_row['atr_multiplier'],  # 변동성 계산 배수
        'profit_ratio': best_row['profit_ratio']  # 목표 수익 비율
    }
    print("\n[Train Result] Best Params:", best_params)  # 최적 파라미터 출력

    # (C) 테스트 단계: 최적 파라미터로 거래 전략 검증
    test_trades = run_simple_backtest(
        symbol=symbol,
        short_timeframe=short_timeframe,
        long_timeframe=long_timeframe,
        window=int(best_params['window']),
        atr_multiplier=float(best_params['atr_multiplier']),
        profit_ratio=float(best_params['profit_ratio']),
        account_size=account_size,
        start_date=test_start,
        end_date=test_end
    )

    # 테스트 결과 출력
    print("=== [Test Result] ===")
    if test_trades is None or test_trades.empty:
        print("No trades in test period.")  # 거래 없음
    else:
        print(f"Test Trades: {len(test_trades)}")  # 거래 수 출력

    # 훈련 및 테스트 결과 반환
    return {
        'best_params': best_params,  # 최적 파라미터
        'train_performance': train_results_df,  # 훈련 구간 결과
        'test_trades': test_trades  # 테스트 구간 결과
    }


def walk_forward_analysis(
    symbol: str = "BTC/USDT",  # 거래 대상
    short_timeframe: str = "4h",  # 단기 봉 간격
    long_timeframe: str = "1d",  # 장기 봉 간격
    overall_start: str = "2021-01-01 00:00:00",  # 전체 분석 시작 날짜
    overall_end: str = "2022-12-31 23:59:59",  # 전체 분석 종료 날짜
    n_splits: int = 3,  # 구간 나누기 개수
    account_size: float = 10_000.0,  # 초기 계좌 잔액
    train_ratio: float = 0.5  # 각 구간의 훈련/테스트 비율
) -> list:
    """
    전체 분석 기간을 여러 구간으로 나누어 순차적으로 훈련 및 테스트를 진행합니다.
    각 구간마다:
      1. 훈련 구간에서 최적 파라미터 선정
      2. 테스트 구간에서 성능 검증
      3. 결과 저장
    
    Returns:
        list: 구간별 결과 리스트
    """
    print("\n=== [Walk-Forward Analysis] ===")
    print(f"Period: {overall_start} ~ {overall_end}")  # 전체 분석 기간 출력
    print(f"Splits: {n_splits}, Train Ratio: {train_ratio}")  # 구간 및 훈련 비율 출력

    # 전체 기간의 날짜 계산
    start_dt = pd.to_datetime(overall_start)
    end_dt = pd.to_datetime(overall_end)
    total_days = (end_dt - start_dt).days

    # 각 구간의 기간 계산
    split_days = total_days // n_splits
    results_list = []  # 결과 저장용 리스트

    for i in range(n_splits):
        # 각 구간의 시작 및 종료 날짜 계산
        split_start_dt = start_dt + pd.to_timedelta(split_days * i, unit='D')
        split_end_dt = start_dt + pd.to_timedelta(split_days * (i + 1), unit='D')
        if i == n_splits - 1:  # 마지막 구간의 종료 날짜 처리
            split_end_dt = end_dt

        # 훈련/테스트 기간 설정
        days_in_this_split = (split_end_dt - split_start_dt).days
        train_days = int(days_in_this_split * train_ratio)

        train_start = split_start_dt
        train_end = train_start + pd.to_timedelta(train_days, unit='D')

        test_start = train_end + pd.to_timedelta(1, unit='D')
        test_end = split_end_dt

        # 날짜를 문자열로 변환
        t_start_str = train_start.strftime("%Y-%m-%d %H:%M:%S")
        t_end_str = train_end.strftime("%Y-%m-%d %H:%M:%S")
        v_start_str = test_start.strftime("%Y-%m-%d %H:%M:%S")
        v_end_str = test_end.strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n--- Split {i + 1}/{n_splits} ---")
        print(f"Train: {t_start_str} ~ {t_end_str}")  # 훈련 기간 출력
        print(f"Test : {v_start_str} ~ {v_end_str}")  # 테스트 기간 출력

        # 훈련 및 테스트 실행
        tt_result = train_test_validation(
            symbol=symbol,
            short_timeframe=short_timeframe,
            long_timeframe=long_timeframe,
            train_start=t_start_str,
            train_end=t_end_str,
            test_start=v_start_str,
            test_end=v_end_str,
            account_size=account_size
        )

        # 결과 저장
        if not tt_result:
            print("Empty result in this split.")  # 결과 없음 처리
            continue

        results_list.append({
            'split_index': i + 1,
            'train_start': t_start_str,
            'train_end': t_end_str,
            'test_start': v_start_str,
            'test_end': v_end_str,
            'best_params': tt_result.get('best_params'),
            'train_performance': tt_result.get('train_performance'),
            'test_trades': tt_result.get('test_trades')
        })

    return results_list  # 구간별 결과 반환
