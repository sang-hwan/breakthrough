# backtest.py

import pandas as pd
import numpy as np

def backtest_breakout_strategy(
    df: pd.DataFrame,
    initial_capital: float = 10000.0,  # 초기 투자금 (USDT 등)
    risk_per_trade: float = 1.0,       # 한 번 진입 시 전체 자금 중 몇 %를 사용할지 (0~100)
    fee_rate: float = 0.0004           # 매매 수수료 비율(예: 0.04%)
):
    """
    단순 Breakout+ATR손절+TP 전략에 대한 백테스트 함수.
    ------------------------------------------------------------
    Parameters
    ----------
    df : pd.DataFrame
        - 필요한 컬럼:
          ['close', 'high', 'low', 'long_entry', 'stop_loss_price', 'take_profit_price']
    initial_capital : float
        백테스트 시작 시점의 가상 자본
    risk_per_trade : float
        거래 1회당 자본의 몇 %를 리스크로 사용할지 (0.01 => 1%, 1.0 => 100%)
    fee_rate : float
        매수/매도시 부과할 수수료 비율 (예: 0.0004 => 0.04%)

    Returns
    -------
    df : pd.DataFrame
        원본 DF에 'position', 'pnl', 'cum_pnl' 등이 추가된 결과
    trade_history : list of dict
        각 거래(진입~청산)의 상세 내역
    summary : dict
        최종 결과 요약(총 손익, 승률 등)
    """

    # --- 결과 저장용 컬럼들 초기화 ---
    df = df.copy()
    df['position'] = 0   # 1: 롱 포지션 진입 중, 0: 청산 상태
    df['pnl'] = 0.0      # 각 봉 마감시 확정된 손익
    df['cum_pnl'] = 0.0  # 누적 손익

    # --- 상태 변수들 ---
    in_position = False
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0

    # 1회 거래당 몇 개 코인을 살지 계산 (예: 100% 자본으로 'close' 매수가정)
    #  - 다만 시뮬레이션에서는 “고정 개수” 혹은 “고정 USD 분” 등 방식을 자유롭게 선택 가능
    #  - 예: risk_per_trade=1.0 => 전체 자본으로 진입
    #    => position_size = (initial_capital * risk_per_trade) / 현재가격
    #  - 실전 백테스트는 거래 반복마다 잔고가 변하므로, 그때그때 position_size 재계산이 일반적입니다.
    capital = initial_capital
    trade_risk_capital = capital * risk_per_trade

    # 실제 체결 이력 저장 (진입 시점, 청산 시점, 가격, 손익 등)
    trade_history = []

    # --- 백테스트 메인 루프 ---
    for i in range(len(df)):
        row = df.iloc[i]

        # 현재 봉의 정보
        current_close = row['close']
        current_high = row['high']
        current_low = row['low']
        current_time = row.name  # timestamp (인덱스)

        # 1) 포지션이 없는 상태라면 (in_position=False)
        if in_position is False:
            # 매수 신호가 True인가?
            if row.get('long_entry', False) is True:
                # --- 진입 실행 ---
                in_position = True
                entry_price = current_close
                sl_price = row['stop_loss_price']
                tp_price = row['take_profit_price']

                # 매수 수량 계산 (여기서는 단순 '진입 시점의 자본 risk_per_trade%로 매수' 가정)
                position_size = (trade_risk_capital / entry_price)

                # 매수 수수료 (대략 시장가 가정)
                buy_fee = entry_price * position_size * fee_rate
                # 체결 시점 자본에서 수수료 차감
                capital -= buy_fee

                # Trade history에 기록(진입 중)
                trade_history.append({
                    'entry_time': current_time,
                    'entry_price': entry_price,
                    'position_size': position_size,
                    'exit_time': None,
                    'exit_price': None,
                    'pnl': None
                })
        else:
            # 2) 이미 포지션을 들고 있는 상태
            # 손절 or 익절 조건 확인
            position_size = trade_history[-1]['position_size']  # 마지막 진입 기록의 수량
            
            exit_price = None
            exit_reason = None

            # (a) 손절 조건: 현재 봉의 저가(low)가 손절가 이하인 경우
            if current_low <= sl_price:
                exit_price = sl_price
                exit_reason = 'StopLoss'
            # (b) 익절 조건: 현재 봉의 고가(high)가 익절가 이상인 경우
            elif current_high >= tp_price:
                exit_price = tp_price
                exit_reason = 'TakeProfit'

            # (c) exit_price가 결정됐다면 (StopLoss or TakeProfit)
            if exit_price is not None:
                # 포지션 청산
                in_position = False

                # 매도 수수료
                sell_fee = exit_price * position_size * fee_rate

                # PnL 계산
                trade_pnl = (exit_price - entry_price) * position_size
                # 수수료 반영
                trade_pnl -= sell_fee

                # 자본 업데이트
                capital += trade_pnl

                # 확정 손익을 df['pnl']에 반영
                df.at[df.index[i], 'pnl'] = trade_pnl

                # trade_history 업데이트
                trade_history[-1]['exit_time'] = current_time
                trade_history[-1]['exit_price'] = exit_price
                trade_history[-1]['pnl'] = trade_pnl
                trade_history[-1]['exit_reason'] = exit_reason

        # position 여부(1 or 0)를 기록
        df.at[df.index[i], 'position'] = 1 if in_position else 0

        # 누적 손익 갱신 (이전까지의 cum_pnl + 이번 봉 pnl)
        if i == 0:
            df.at[df.index[i], 'cum_pnl'] = df.at[df.index[i], 'pnl']
        else:
            df.at[df.index[i], 'cum_pnl'] = df.at[df.index[i-1], 'cum_pnl'] + df.at[df.index[i], 'pnl']

    # --- 모든 봉 순회 후, trade_history와 요약 정보 정리 ---
    total_net_profit = df['pnl'].sum()
    total_trades = sum(1 for t in trade_history if t['pnl'] is not None)
    wins = sum(1 for t in trade_history if t['pnl'] is not None and t['pnl'] > 0)
    losses = total_trades - wins
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    summary = {
        'Initial Capital': initial_capital,
        'Final Capital': capital,
        'Total Net Profit': total_net_profit,
        'Total Trades': total_trades,
        'Win Rate (%)': win_rate,
        'Wins': wins,
        'Losses': losses,
    }

    return df, trade_history, summary


if __name__ == "__main__":
    # -----------------------------------------------------------
    # 간단 테스트 (실제로는 signal_generator + trade_logic 적용한 df를 넣으면 됨)
    # -----------------------------------------------------------
    # 가상의 예시 DataFrame 생성
    # (실제로는 fetch_binance_ohlcv(), calculate_breakout_signals() 등을 거쳐 df를 만들었다고 가정)
    data = {
        'close': [100, 102, 103, 105, 104, 106, 107, 103, 101, 110],
        'high':  [101, 103, 105, 106, 105, 107, 108, 106, 102, 115],
        'low':   [99,  99,  101, 103, 103, 104, 105, 102, 99,  100],
        'long_entry': [False, True, False, False, False, False, False, False, True, False],  # 2번째, 9번째 봉에 진입 시도
        'stop_loss_price':   [np.nan, 98, 98, 98, 98, 98, 98, 98,  99,  99],
        'take_profit_price': [np.nan, 110,110,110,110,110,110,110, 120, 120],
    }
    test_df = pd.DataFrame(data)
    test_df.index = pd.date_range(start='2025-01-01', periods=len(test_df), freq='4h')  # 예시 Timestamp

    # 백테스트 실행
    bt_df, trades, result = backtest_breakout_strategy(test_df)
    print(bt_df)
    print("\nTrade History:", trades)
    print("\nSummary:", result)
