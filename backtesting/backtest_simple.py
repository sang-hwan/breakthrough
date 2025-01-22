# backtesting/backtest_simple.py

import pandas as pd
import numpy as np

# 데이터베이스에서 시세 데이터를 불러오는 함수
from data_collection.postgres_ohlcv_handler import load_ohlcv_from_postgres

# 단기 돌파 전략을 위한 신호 계산 함수
from strategies.breakout_signal import calculate_breakout_signals

# 장기 보조 지표 계산을 위한 함수 (SMA, MACD, RSI, Bollinger Bands)
from strategies.technical_indicators import (
    apply_sma,  # 단순 이동평균
    apply_macd,  # 이동평균 수렴·확산
    apply_rsi,  # 상대 강도 지수
    apply_bollinger  # 볼린저 밴드
)

# 손절/익절 전략과 리스크 관리를 위한 함수
from strategies.stop_loss_take_profit import apply_stop_loss_atr, apply_take_profit_ratio
from strategies.risk_management import calculate_position_size

# 백테스트 결과를 출력하는 함수
from backtesting.performance_metrics import print_performance_report


def run_simple_backtest(
    symbol="BTC/USDT",  # 거래할 종목 (예: 비트코인/테더)
    short_timeframe="4h",  # 단기 봉 주기 (4시간)
    long_timeframe="1d",  # 장기 봉 주기 (1일)
    window=20,  # 돌파 전략에 사용할 기간
    volume_factor=1.5,  # 돌파 신호의 거래량 필터 기준
    confirm_bars=2,  # 돌파 신호 확인을 위한 추가 봉 수
    breakout_buffer=0.0,  # 돌파 신호 버퍼 (추가 여유 범위)
    atr_window=14,  # ATR(평균 진폭) 계산 기간
    atr_multiplier=2.0,  # 손절 기준 ATR 배수
    profit_ratio=0.05,  # 고정 익절 비율
    account_size=10_000.0,  # 초기 계좌 잔액
    risk_per_trade=0.01,  # 1회 매매 시 총 자산 대비 위험 비율
    fee_rate=0.001  # 거래 수수료 비율
):
    """
    다중 타임프레임 백테스트 실행 함수.

    단기 전략(4시간 봉)을 사용해 돌파 신호를 확인하고,
    장기 전략(1일 봉)으로 보조 지표를 통해 신호를 검증합니다.

    1. 단기 봉 데이터를 사용해 돌파 신호 계산
    2. 장기 봉 데이터를 통해 SMA, MACD, RSI, Bollinger Bands 조건 확인
    3. 매매 시 ATR 기반 손절과 고정 비율 익절 적용
    4. 결과를 출력하여 총 수익률, 성과 등을 확인
    """

    # -------------------------------
    # 1) 데이터 불러오기
    # -------------------------------
    # 데이터베이스에서 단기/장기 데이터를 불러옵니다.
    short_table = f"ohlcv_{symbol.replace('/', '').lower()}_{short_timeframe}"  # 단기 데이터 테이블 이름
    long_table = f"ohlcv_{symbol.replace('/', '').lower()}_{long_timeframe}"  # 장기 데이터 테이블 이름

    # 단기/장기 데이터를 로드
    df_short = load_ohlcv_from_postgres(table_name=short_table)
    df_long = load_ohlcv_from_postgres(table_name=long_table)

    # 데이터 정렬
    df_short.sort_index(inplace=True)
    df_long.sort_index(inplace=True)

    # 데이터 유효성 검사
    if df_short.empty:
        print(f"[ERROR] 단기 데이터가 비어 있습니다: {short_table}")
        return None
    if df_long.empty:
        print(f"[ERROR] 장기 데이터가 비어 있습니다: {long_table}")
        return None

    print(f"[INFO] 단기 데이터 로드 완료: {df_short.shape[0]} 행")
    print(f"[INFO] 장기 데이터 로드 완료: {df_long.shape[0]} 행\n")

    # -------------------------------
    # 2) 돌파 신호 계산 (단기 데이터)
    # -------------------------------
    # 단기 봉 데이터에서 돌파 신호를 계산합니다.
    df_short = calculate_breakout_signals(
        df=df_short,
        window=window,  # 돌파 기간
        vol_factor=volume_factor,  # 거래량 필터
        confirm_bars=confirm_bars,  # 추가 확인 봉 수
        use_high=False,  # 고점 대신 종가 기준 사용
        breakout_buffer=breakout_buffer  # 돌파 신호 버퍼
    )

    # 돌파 신호 통계 출력
    print("[단기 봉] 신호 통계:")
    print(f"  -> 돌파 신호: {df_short['breakout_signal'].sum()} 개")
    print(f"  -> 확인된 돌파: {df_short['confirmed_breakout'].sum()} 개")

    # -------------------------------
    # 3) 보조 지표 계산 (장기 데이터)
    # -------------------------------

    # (a) SMA 200 계산
    df_long = apply_sma(
        df=df_long,
        price_col='close',  # 종가를 기준으로 계산
        window=200,  # 200일 이동평균
        colname='sma200'  # 결과를 저장할 컬럼 이름
    )

    # (b) MACD 계산
    df_long = apply_macd(
        df=df_long,
        price_col='close',  # 종가 사용
        window_slow=26,  # 느린 이동평균
        window_fast=12,  # 빠른 이동평균
        window_sign=9,  # 시그널 이동평균
        prefix='macd_'  # 컬럼 접두사
    )

    # (c) RSI(14) 계산
    df_long = apply_rsi(
        df=df_long,
        price_col='close',  # 종가 기준
        window=14,  # RSI 기간
        colname='rsi14'  # 결과 저장 컬럼 이름
    )

    # (d) 볼린저 밴드 계산
    df_long = apply_bollinger(
        df=df_long,
        price_col='close',  # 종가 기준
        window=20,  # 이동평균 기간
        window_dev=2.0,  # 표준편차 배수
        prefix='bb_'  # 컬럼 접두사
    )

    # 장기 필터 조건 정의 (예시)
    df_long['long_ok'] = (
        (df_long['close'] >= df_long['sma200']) &  # SMA200 이상
        (df_long['rsi14'] < 70) &  # RSI가 과매수 상태가 아님
        (df_long['macd_diff'] > 0) &  # MACD 상승
        (df_long['close'] < df_long['bb_hband'])  # 볼린저 상단 밴드 미돌파
    )

    # -------------------------------
    # 4) 손절 및 익절 계산
    # -------------------------------
    # 단기 신호에서 손절/익절 가격 설정
    df_short['long_entry'] = df_short['confirmed_breakout'] & df_short['volume_condition']

    # ATR 기반 손절 설정
    df_short = apply_stop_loss_atr(
        df=df_short,
        atr_window=atr_window,  # ATR 기간
        atr_multiplier=atr_multiplier,  # 손절 배수
        sl_colname='stop_loss_price',  # 손절 가격 컬럼
        entry_price_col='entry_price'  # 진입 가격
    )

    # 고정 익절 비율 설정
    df_short = apply_take_profit_ratio(
        df=df_short,
        profit_ratio=profit_ratio,  # 익절 비율
        tp_colname='take_profit_price',  # 익절 가격 컬럼
        entry_price_col='entry_price'  # 진입 가격
    )

    # -------------------------------
    # 5) 백테스트 수행
    # -------------------------------
    # 거래 내역 기록용 리스트
    trades = []
    # 현재 포지션 보유 여부
    in_position = False

    # 단기 데이터를 순회하며 매매 전략 실행
    for i in range(len(df_short)):
        row_s = df_short.iloc[i]
        current_time = row_s.name  # 현재 시간(봉 기준)

        # (A) 포지션 진입
        if not in_position and row_s['long_entry']:
            # 장기 데이터에서 현재 시간까지의 데이터 가져오기
            df_long_sub = df_long.loc[:current_time]
            if df_long_sub.empty:
                continue
            row_l = df_long_sub.iloc[-1]

            # 장기 필터 조건 통과 시 매수
            if row_l['long_ok']:
                # 진입 가격, 손절/익절 설정
                entry_price = row_s['entry_price']
                stop_loss = row_s['stop_loss_price']
                take_profit = row_s['take_profit_price']

                # 포지션 크기 계산
                size = calculate_position_size(
                    account_balance=account_size,
                    risk_per_trade=risk_per_trade,
                    entry_price=entry_price,
                    stop_loss_price=stop_loss,
                    fee_rate=fee_rate
                )

                # 포지션 활성화 및 거래 기록 추가
                in_position = True
                trades.append({
                    'entry_time': current_time,
                    'entry_price': entry_price,
                    'size': size,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'exit_time': None,
                    'exit_price': None,
                    'pnl': None,
                    'exit_reason': None
                })

        # (B) 포지션 청산
        elif in_position:
            # 손절 조건 확인
            if row_s['close'] <= trades[-1]['stop_loss']:
                in_position = False
                exit_price = row_s['close']
                trades[-1].update({
                    'exit_time': current_time,
                    'exit_price': exit_price,
                    'pnl': (exit_price - trades[-1]['entry_price']) * trades[-1]['size'],
                    'exit_reason': 'stop_loss'
                })
            # 익절 조건 확인
            elif row_s['close'] >= trades[-1]['take_profit']:
                in_position = False
                exit_price = row_s['close']
                trades[-1].update({
                    'exit_time': current_time,
                    'exit_price': exit_price,
                    'pnl': (exit_price - trades[-1]['entry_price']) * trades[-1]['size'],
                    'exit_reason': 'take_profit'
                })

    # -------------------------------
    # 6) 결과 출력
    # -------------------------------
    # 거래 내역을 DataFrame으로 변환
    trades_df = pd.DataFrame(trades)
    trades_df.dropna(subset=['exit_time'], inplace=True)

    # 결과 출력
    if trades_df.empty:
        print("매매가 실행되지 않았습니다.")
        return None

    total_pnl = trades_df['pnl'].sum()  # 총 수익 계산
    final_balance = account_size + total_pnl  # 최종 계좌 잔액
    roi_percent = (final_balance - account_size) / account_size * 100.0  # 총 수익률

    # 성과 요약 출력
    print("\n=== 백테스트 결과 ===")
    print(f"총 거래 수: {len(trades_df)}")
    print(f"총 수익: {total_pnl:.2f} USDT")
    print(f"최종 계좌 잔액: {final_balance:.2f} USDT")
    print(f"수익률: {roi_percent:.2f}%")

    # 성과 세부 지표 출력 (월별 성과 등)
    print_performance_report(trades_df, initial_balance=account_size)

    return trades_df
