# backtesting/backtest_simple.py

import pandas as pd
import numpy as np

# (1) 우리가 이미 만들어 둔 모듈들 임포트
#     - 경로는 예시입니다. 실제 디렉토리 구조에 맞춰 수정하세요.
from data_collection.save_to_postgres import load_ohlcv_from_postgres
from strategies.signal_generator import calculate_breakout_signals
from strategies.stop_loss_take_profit import (
    apply_stop_loss_atr,
    apply_take_profit_ratio
)
from strategies.risk_management import calculate_position_size


def run_simple_backtest():
    """
    가장 단순한 버전의 백테스트를 시연하는 함수입니다.
    
    대상: BTC/USDT, 4시간봉
    기간: 2018년 ~ 2025년 (DB에 저장된 구간)
    전략: 
      1) 전고점(20봉) 돌파 + 거래량(1.5배) 필터 + 2봉 연속 확정돌파
      2) ATR(14) x 2.0 손절
      3) 5% 익절
      4) 계좌 1% 리스크로 포지션 사이즈 결정
      5) 포지션은 1회 진입 후 손절 or 익절 시 청산 (가장 단순 버전)
    """
    # --------------------------------------------------------------------
    # 0) 백테스트에 사용할 기본 변수 설정
    # --------------------------------------------------------------------
    symbol        = "BTC/USDT"
    timeframe     = "4h"
    table_name    = "ohlcv_btcusdt_4h"  # ohlcv_{심볼소문자}_{타임프레임}
    account_size  = 10_000.0  # 계좌자산 (예: 10,000 USDT)
    risk_per_trade= 0.01      # 매매 1회당 계좌의 1%까지 손실 허용
    atr_window    = 14
    atr_mult      = 2.0
    profit_ratio  = 0.05      # 5% 익절
    
    # 전고점(rolling) 돌파 파라미터
    breakout_window      = 20
    volume_factor        = 1.5
    breakout_confirm_bars= 2
    breakout_buffer      = 0.0  # 버퍼 없는 단순 돌파

    # --------------------------------------------------------------------
    # 1) 과거 데이터 불러오기 (PostgreSQL → DataFrame)
    # --------------------------------------------------------------------
    df = load_ohlcv_from_postgres(table_name=table_name)
    # 필요에 따라 기간 필터 (2018년 ~ 2025년). 여기선 전체 그대로 사용.
    df = df.sort_index()  # 혹시 인덱스(타임스탬프)가 뒤섞여 있으면 정렬
    
    
    # ================== 데이터 자체를 확인 (1) ===================
    print("=== (A) Data Check ===")
    print("DataFrame shape:", df.shape)
    if not df.empty:
        # 앞뒤 3줄씩만 간단 확인
        print(df.head(3))
        print(df.tail(3))
        
        print("\nData Stats:")
        print(df[['open','high','low','close','volume']].describe())
    else:
        print("DataFrame is empty! Check if the DB table has data.")

    # --------------------------------------------------------------------
    # 2) 돌파 시그널/거래량 필터 계산 (strategies/signal_generator)
    # --------------------------------------------------------------------
    df = calculate_breakout_signals(
        df=df,
        window=breakout_window,
        vol_factor=volume_factor,
        confirm_bars=breakout_confirm_bars,
        use_high=False,
        breakout_buffer=breakout_buffer
    )
    
    # ================== 시그널 계산 중간결과 확인 (2) ===================
    if not df.empty:
        # breakout_signal True 개수
        bs_count = df['breakout_signal'].sum() if 'breakout_signal' in df.columns else 0
        # volume_condition True 개수
        vc_count = df['volume_condition'].sum() if 'volume_condition' in df.columns else 0
        # confirmed_breakout True 개수
        cb_count = df['confirmed_breakout'].sum() if 'confirmed_breakout' in df.columns else 0

        print("\n=== (B) Signal Columns Check ===")
        print(f"breakout_signal True count    = {bs_count}")
        print(f"volume_condition True count   = {vc_count}")
        print(f"confirmed_breakout True count = {cb_count}")
    else:
        print("\nNo data to check for signals (DataFrame is empty).")
    
    # --------------------------------------------------------------------
    # 3) ATR 손절가 + 5% 익절가 계산
    # --------------------------------------------------------------------
    # 우선, “매수 시그널”을 판단하기 위한 컬럼('long_entry')을 단순 생성:
    # 전고점 돌파가 확정이면서 거래량 조건이 True면 진입 시그널이라고 가정.
    df['long_entry'] = (df['confirmed_breakout'] & df['volume_condition'])

    df = apply_stop_loss_atr(
        df=df,
        atr_window=atr_window,
        atr_multiplier=atr_mult,
        sl_colname='stop_loss_price',
        entry_price_col='entry_price'  # 이 컬럼에 매수 가격이 ffill로 기록
    )
    df = apply_take_profit_ratio(
        df=df,
        profit_ratio=profit_ratio,
        tp_colname='take_profit_price',
        entry_price_col='entry_price'
    )

    # --------------------------------------------------------------------
    # 4) 백테스트 루프 준비
    #    - 간단하게 한 번에 1회 포지션만 보유하는 구조
    #    - 매 봉마다 포지션 상태 확인 → 진입 or 청산
    # --------------------------------------------------------------------
    in_position = False
    entry_price = np.nan
    stop_loss   = np.nan
    take_profit = np.nan
    trades      = []  # 매매 기록 로그 (dict 형태로 append → 최종 DataFrame 변환 가능)

    for i in range(len(df)):
        row = df.iloc[i]
        current_time = row.name  # timestamp 가 인덱스
        close_price  = row['close']

        if not in_position:
            # (a) 포지션 미보유 상태 -> 매수 신호가 뜨면 진입
            if row['long_entry']:
                # 손절가, 익절가, 진입가 정보를 미리 DataFrame에서 가져온다.
                # apply_stop_loss_atr / apply_take_profit_ratio 에서 ffill된 값 사용
                entry_price = row['entry_price']
                stop_loss   = row['stop_loss_price']
                take_profit = row['take_profit_price']
                
                # 포지션 사이즈 계산 (리스크 1%)
                size = calculate_position_size(
                    account_balance=account_size,
                    risk_per_trade=risk_per_trade,
                    entry_price=entry_price,
                    stop_loss_price=stop_loss,
                    fee_rate=0.001  # 예시로 0.1% 수수료
                )
                
                in_position = True
                trades.append({
                    'entry_time' : current_time,
                    'entry_price': entry_price,
                    'size'       : size,
                    'exit_time'  : None,
                    'exit_price' : None,
                    'pnl'        : None,
                    'exit_reason': None
                })

        else:
            # (b) 포지션 보유 중 -> 손절 or 익절 조건 체크
            # 간단히 'close'가 stop_loss 밑으로 내려가면 손절, take_profit 위로 올라가면 익절
            if close_price <= stop_loss:
                # 손절 청산
                in_position = False
                trades[-1]['exit_time']  = current_time
                trades[-1]['exit_price'] = close_price
                trades[-1]['pnl']        = (close_price - trades[-1]['entry_price']) * trades[-1]['size']
                trades[-1]['exit_reason']= 'stop_loss'

            elif close_price >= take_profit:
                # 익절 청산
                in_position = False
                trades[-1]['exit_time']  = current_time
                trades[-1]['exit_price'] = close_price
                trades[-1]['pnl']        = (close_price - trades[-1]['entry_price']) * trades[-1]['size']
                trades[-1]['exit_reason']= 'take_profit'
    
    # --------------------------------------------------------------------
    # 5) 백테스트 결과 정리
    # --------------------------------------------------------------------
    # columns를 미리 지정해서 빈 경우에도 exit_time 등이 생성되도록 처리
    trades_df = pd.DataFrame(trades, columns=[
        'entry_time', 'entry_price', 'size',
        'exit_time', 'exit_price', 'pnl', 'exit_reason'
    ])

    # 만약 trades_df가 비어있다면 -> 매매가 없었던 경우
    if trades_df.empty:
        print("No trades were triggered.")
        return None

    # exit_time이 없는(= 아직 미청산) 행은 제외
    trades_df = trades_df.dropna(subset=['exit_time'])

    if trades_df.empty:
        print("No closed trades. (All positions might still be open or no trades at all.)")
        return None

    # 손익 합계
    total_pnl = trades_df['pnl'].sum()
    final_balance = account_size + total_pnl
    num_trades = len(trades_df)

    roi_percent = ((final_balance - account_size) / account_size) * 100.0

    print("=== Simple Backtest Result ===")
    print(f"Total Trades : {num_trades}")
    print(f"Total PnL    : {total_pnl:.2f} USDT")
    print(f"Final Balance: {final_balance:.2f} USDT")
    print(f"ROI          : {roi_percent:.2f}%")
    print("\nTrade Details:")
    print(trades_df.tail(10))  # 최근 10건만 표시

    return trades_df


if __name__ == "__main__":
    run_simple_backtest()
