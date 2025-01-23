# backtesting/backtest_simple.py

import pandas as pd  # 데이터 프레임 처리
import numpy as np  # 수학 연산

from data_collection.postgres_ohlcv_handler import load_ohlcv_from_postgres  # 데이터 로드
from strategies.breakout_signal import calculate_breakout_signals  # 돌파 신호 계산
from strategies.technical_indicators import (  # 기술적 지표 계산
    apply_sma,  # 단순 이동 평균
    apply_macd,  # MACD (이동 평균 수렴·발산 지표)
    apply_rsi,  # RSI (상대 강도 지수)
    apply_bollinger  # 볼린저 밴드
)
from strategies.stop_loss_take_profit import (  # 손절 및 익절 계산
    apply_stop_loss_atr,  # ATR 기반 손절 설정
    apply_take_profit_ratio  # 고정 비율 익절 설정
)
from strategies.risk_management import (  # 리스크 관리
    calculate_position_size,  # 포지션 크기 계산
    split_position_sizes  # 포지션 분할 계산
)
from backtesting.performance_metrics import print_performance_report  # 성과 보고서 출력


def run_simple_backtest(
    symbol="BTC/USDT",  # 거래 종목 (기본값: 비트코인/테더)
    short_timeframe="4h",  # 단기 봉 간격 (4시간)
    long_timeframe="1d",  # 장기 봉 간격 (1일)
    window=20,  # 돌파 신호 기간
    volume_factor=1.5,  # 거래량 기준 배수
    confirm_bars=2,  # 돌파 신호 확인을 위한 봉 수
    breakout_buffer=0.0,  # 돌파 여유 범위
    atr_window=14,  # ATR 계산 기간
    atr_multiplier=2.0,  # ATR 기반 손절 배수
    profit_ratio=0.05,  # 고정 비율 익절 기준
    account_size=10_000.0,  # 초기 계좌 잔액 (기본값: 10,000 USDT)
    risk_per_trade=0.01,  # 거래당 리스크 비율
    fee_rate=0.001,  # 거래 수수료 비율
    split_count=3,  # 분할 매매 개수
    split_scale_mode='equal',  # 분할 매매 크기 분배 방식 (균등 분배)
    start_date: str = None,  # 데이터 로드 시작 날짜
    end_date: str = None  # 데이터 로드 종료 날짜
):
    """
    단순 백테스트 실행 함수 (분할매매 포함)
    1. 데이터를 로드하고 전략을 적용합니다.
    2. 돌파 신호, 기술적 지표, 손절/익절 조건을 계산합니다.
    3. 분할 매매를 포함한 백테스트를 실행합니다.
    Returns:
        pd.DataFrame: 매매 기록 데이터프레임
    """

    # -------------------------------
    # 1) 데이터 로드
    # -------------------------------
    short_table = f"ohlcv_{symbol.replace('/', '').lower()}_{short_timeframe}"  # 단기 데이터 테이블 이름
    long_table = f"ohlcv_{symbol.replace('/', '').lower()}_{long_timeframe}"  # 장기 데이터 테이블 이름

    # 단기 및 장기 데이터를 PostgreSQL에서 로드
    df_short = load_ohlcv_from_postgres(
        table_name=short_table,
        start_date=start_date,
        end_date=end_date
    )

    df_long = load_ohlcv_from_postgres(
        table_name=long_table,
        start_date=start_date,
        end_date=end_date
    )

    # 데이터가 비어 있는 경우 에러 메시지 출력 및 종료
    if df_short.empty:
        print(f"[ERROR] 단기 데이터가 없습니다: {short_table}")
        return None
    if df_long.empty:
        print(f"[ERROR] 장기 데이터가 없습니다: {long_table}")
        return None

    # 데이터 정렬
    df_short.sort_index(inplace=True)
    df_long.sort_index(inplace=True)

    print(f"[INFO] 단기 데이터 로드 완료: {df_short.shape[0]} 행")
    print(f"[INFO] 장기 데이터 로드 완료: {df_long.shape[0]} 행\n")

    # -------------------------------
    # 2) 돌파 신호 계산
    # -------------------------------
    # 단기 봉 데이터에서 돌파 신호 계산
    df_short = calculate_breakout_signals(
        df=df_short,
        window=window,
        vol_factor=volume_factor,
        confirm_bars=confirm_bars,
        use_high=False,
        breakout_buffer=breakout_buffer
    )

    print("[단기 봉] 신호 통계:")
    print(f"  -> 돌파 신호: {df_short['breakout_signal'].sum()} 개")
    print(f"  -> 확인된 돌파: {df_short['confirmed_breakout'].sum()} 개\n")

    # -------------------------------
    # 3) 보조 지표 계산
    # -------------------------------
    # 장기 봉 데이터에서 SMA, MACD, RSI, 볼린저 밴드 계산
    df_long = apply_sma(df_long, price_col='close', window=200, colname='sma200')
    df_long = apply_macd(df_long, price_col='close', window_slow=26, window_fast=12, window_sign=9, prefix='macd_')
    df_long = apply_rsi(df_long, price_col='close', window=14, colname='rsi14')
    df_long = apply_bollinger(df_long, price_col='close', window=20, window_dev=2.0, prefix='bb_')

    # 롱 포지션 조건 설정
    cond_sma = (df_long['close'] >= df_long['sma200'])  # 200일 SMA 이상
    cond_macd = (df_long['macd_diff'] > 0)  # MACD 양수
    cond_rsi = (df_long['rsi14'] < 70)  # RSI 과매수 미만
    cond_bb = (df_long['close'] < df_long['bb_hband'])  # 볼린저 밴드 상단 미만

    df_long['long_ok'] = cond_sma & cond_macd & cond_rsi & cond_bb  # 모든 조건 충족 시 롱 가능
    print(f"  -> 장기 long_ok True: {df_long['long_ok'].sum()} / {len(df_long)}")

    # -------------------------------
    # 4) 손절 및 익절 계산
    # -------------------------------
    df_short['long_entry'] = df_short['confirmed_breakout'] & df_short['volume_condition']

    # 손절 가격 계산 (ATR 기반)
    df_short = apply_stop_loss_atr(
        df=df_short,
        atr_window=atr_window,
        atr_multiplier=atr_multiplier,
        sl_colname='stop_loss_price',
        entry_price_col='entry_price'
    )
    
    # 익절 가격 계산 (고정 비율 기반)
    df_short = apply_take_profit_ratio(
        df=df_short,
        profit_ratio=profit_ratio,
        tp_colname='take_profit_price',
        entry_price_col='entry_price'
    )

    # -------------------------------
    # 5) 백테스트 수행 (분할매매 적용)
    # -------------------------------
    trades = []  # 매매 기록
    in_position = False  # 포지션 보유 여부

    for i in range(len(df_short)):  # 단기 봉 데이터 순회
        row_s = df_short.iloc[i]
        current_time = row_s.name  # 현재 시간

        # (A) 포지션 진입 조건 확인
        if not in_position and row_s['long_entry']:
            # 장기 봉 조건 검토
            df_long_sub = df_long.loc[:current_time]
            if df_long_sub.empty:
                continue

            row_l = df_long_sub.iloc[-1]
            if row_l['long_ok']:
                # 포지션 크기 및 분할 매매 크기 계산
                entry_price = row_s['entry_price']
                stop_loss   = row_s['stop_loss_price']
                total_size  = calculate_position_size(
                    account_balance=account_size,
                    risk_per_trade=risk_per_trade,
                    entry_price=entry_price,
                    stop_loss_price=stop_loss,
                    fee_rate=fee_rate
                )

                if total_size > 0:
                    partial_sizes = split_position_sizes(
                        total_position_size=total_size,
                        split_count=split_count,
                        scale_mode=split_scale_mode
                    )

                    # 분할 매매 기록 추가
                    for ps in partial_sizes:
                        trades.append({
                            'entry_time': current_time,
                            'entry_price': entry_price,
                            'size': ps,
                            'stop_loss': stop_loss,
                            'take_profit': row_s['take_profit_price'],
                            'exit_time': None,
                            'exit_price': None,
                            'pnl': None,
                            'exit_reason': None
                        })
                    in_position = True

        # (B) 포지션 청산 조건 확인 (손절/익절)
        if in_position:
            open_trades = [t for t in trades if t['exit_time'] is None]
            if not open_trades:
                in_position = False
                continue

            last_trade = open_trades[-1]
            current_close = row_s['close']

            # 손절 청산
            if current_close <= last_trade['stop_loss']:
                for t in open_trades:
                    t['exit_time'] = current_time
                    t['exit_price'] = current_close
                    t['pnl'] = (current_close - t['entry_price']) * t['size']
                    t['exit_reason'] = 'stop_loss'
                in_position = False

            # 익절 청산
            elif current_close >= last_trade['take_profit']:
                for t in open_trades:
                    t['exit_time'] = current_time
                    t['exit_price'] = current_close
                    t['pnl'] = (current_close - t['entry_price']) * t['size']
                    t['exit_reason'] = 'take_profit'
                in_position = False

    # -------------------------------
    # 6) 결과 출력
    # -------------------------------
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        print("\n[INFO] 매매가 발생하지 않았습니다.")
        return None

    # 최종 성과 계산 및 출력
    total_pnl = trades_df['pnl'].sum()
    final_balance = account_size + total_pnl
    roi_percent = (final_balance - account_size) / account_size * 100.0

    print("\n=== 백테스트 결과 ===")
    print(f"총 거래 수(분할 포함): {len(trades_df)}")
    print(f"총 손익: {total_pnl:.2f} USDT")
    print(f"최종 계좌 잔액: {final_balance:.2f} USDT")
    print(f"수익률: {roi_percent:.2f}%")

    # 필요 시 상세 보고서
    # print_performance_report(trades_df, initial_balance=account_size)

    return trades_df
