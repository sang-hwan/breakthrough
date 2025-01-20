import pandas as pd
import numpy as np

def backtest_breakout_strategy(
    df: pd.DataFrame,
    initial_capital: float = 10000.0,  # 투자 시작 시점의 가상 자본(예: 10,000 USDT)
    risk_per_trade: float = 1.0,       # 한 번 매수할 때 전체 자본 중 몇 %를 쓰는지 (0.0 ~ 1.0)
    fee_rate: float = 0.0004           # 매수/매도시 부과되는 수수료 비율 (예: 0.0004 => 0.04%)
):
    """
    이 함수는 단순 '돌파 매매 전략 + ATR 기반 손절가 + 고정 익절가'를 이용해
    과거 데이터로 전략을 시뮬레이션(백테스트)하는 예시입니다.

    (1) 백테스트에 필요한 주요 컬럼:
        - 'close'  : 각 시점의 종가(가격)
        - 'high'   : 각 시점의 고가(해당 봉/캔들에서의 최고가)
        - 'low'    : 각 시점의 저가(해당 봉/캔들에서의 최저가)
        - 'long_entry'       : 매수 신호(True/False)
        - 'stop_loss_price'  : 계산된 손절가(Stop Loss Price)
        - 'take_profit_price': 계산된 익절가(Take Profit Price)

    (2) 파라미터 설명:
        - initial_capital : 백테스트 시작 시점에 갖고 있는 가상의 자본 (ex: 10,000 USD)
        - risk_per_trade  : 매수 시점마다 전체 자본 중 어느 정도를 매수에 쓸지 결정 (0~1)
                            예) 1.0 => 100% 자본 사용, 0.5 => 50% 자본 사용
        - fee_rate        : 매매 수수료 비율. 예) 0.0004 => 0.04%

    (3) 함수가 반환하는 세 가지 결과:
        1) df : 원본 데이터프레임에다 'position', 'pnl', 'cum_pnl' 등을 추가한 결과
        2) trade_history : 매매 내역(진입 시점, 청산 시점, 손익 등)을 담은 리스트
        3) summary       : 최종 요약(총 손익, 최종자본, 승률 등) 정보가 담긴 딕셔너리
    """

    # -----------------------------------------------------------
    # (A) 백테스트에 필요한 새로운 컬럼을 초기화
    # -----------------------------------------------------------
    df = df.copy()  # 원본 df를 건드리지 않기 위해 복사본을 만듦
    df['position'] = 0   # 포지션 상태(1: 매수 상태, 0: 청산 상태)를 기록
    df['pnl'] = 0.0      # 각 시점(봉/캔들)에서 실현된 손익(이익 또는 손실)
    df['cum_pnl'] = 0.0  # 누적 손익(이전까지의 손익을 계속 합산)

    # -----------------------------------------------------------
    # (B) 백테스트에 필요한 상태 변수들 정의
    # -----------------------------------------------------------
    in_position = False     # 현재 포지션을 가지고 있는지 여부 (True/False)
    entry_price = 0.0       # 매수 진입 가격
    sl_price = 0.0          # 손절가
    tp_price = 0.0          # 익절가

    # 초기 자본 설정
    capital = initial_capital

    # 예) risk_per_trade=1.0 이고 자본이 10,000 USDT이면
    #     한 번에 10,000 USDT로 매수 진입한다고 가정
    trade_risk_capital = capital * risk_per_trade

    # 실제 매매 내역을 저장하기 위한 리스트 (매 수/청산별로 딕셔너리를 기록)
    trade_history = []

    # -----------------------------------------------------------
    # (C) 메인 루프: 각 시점(각 봉/캔들)에 대해 매수/청산 로직을 처리
    # -----------------------------------------------------------
    for i in range(len(df)):
        # df.iloc[i]: 현재 봉(캔들)의 정보 (가격, 시그널, 손절/익절가 등)
        row = df.iloc[i]

        # 고유 식별 정보 (예: 2025-01-01 00:00:00 등) - 시계열 인덱스
        current_time = row.name
        # 현재 시점의 종가, 고가, 저가를 가져옴
        current_close = row['close']
        current_high = row['high']
        current_low = row['low']

        # -------------------------------------------------------
        # (1) 만약 현재 포지션이 없다면(in_position=False) => 매수 신호를 확인
        # -------------------------------------------------------
        if in_position is False:
            # df의 'long_entry'가 True이면 매수 신호 발생
            if row.get('long_entry', False) == True:
                # --- 진입(매수) 실행 ---
                in_position = True
                entry_price = current_close       # 매수 진입가격을 현재 종가로 가정
                sl_price = row['stop_loss_price'] # 이 시점에서 계산된 손절가
                tp_price = row['take_profit_price'] # 이 시점에서 계산된 익절가

                # (가정) 한 번 매수할 때 전체 자본*risk_per_trade 만큼 매수한다고 함
                #       => 매수 가능한 '코인 개수'를 계산
                position_size = trade_risk_capital / entry_price

                # 매수 시, 수수료도 차감(시장가 매수 시 가정)
                buy_fee = entry_price * position_size * fee_rate
                capital -= buy_fee  # 수수료만큼 자본 감소

                # 매수(진입) 정보를 trade_history에 추가
                trade_history.append({
                    'entry_time': current_time,   # 진입한 시점
                    'entry_price': entry_price,   # 진입 가격
                    'position_size': position_size,  # 코인 몇 개 샀나
                    'exit_time': None,            # 아직 청산 전이므로 None
                    'exit_price': None,
                    'pnl': None                   # 손익도 아직 None
                })

        # -------------------------------------------------------
        # (2) 이미 포지션을 들고 있는 경우(in_position=True)
        #     => 손절/익절 조건을 확인하여 청산 여부 결정
        # -------------------------------------------------------
        else:
            # 마지막으로 기록된(방금 매수했던) position_size를 불러옴
            position_size = trade_history[-1]['position_size']

            exit_price = None    # 청산가격(손절 or 익절)이 발생했을 때 기록
            exit_reason = None   # 청산 사유 (StopLoss / TakeProfit)

            # (a) 손절 조건: 현재 저가가 손절가 이하인 경우 => 손절가에서 청산
            if current_low <= sl_price:
                exit_price = sl_price
                exit_reason = 'StopLoss'

            # (b) 익절 조건: 현재 고가가 익절가 이상인 경우 => 익절가에서 청산
            elif current_high >= tp_price:
                exit_price = tp_price
                exit_reason = 'TakeProfit'

            # (c) 위 두 조건 중 하나라도 충족하면, 포지션 청산
            if exit_price is not None:
                # 포지션 해제
                in_position = False

                # 매도 시 수수료 계산
                sell_fee = exit_price * position_size * fee_rate

                # 실제 손익 계산: (청산가격 - 매수가격) * 코인 수량 - 매도 수수료
                trade_pnl = (exit_price - entry_price) * position_size
                trade_pnl -= sell_fee

                # 실현된 손익을 자본(capital)에 반영
                capital += trade_pnl

                # 이번 봉(행)에서 확정된 손익을 df['pnl']에 기록
                df.at[df.index[i], 'pnl'] = trade_pnl

                # 매매 이력 업데이트 (나가던 trade_history 마지막 기록에 청산 정보 기입)
                trade_history[-1]['exit_time'] = current_time
                trade_history[-1]['exit_price'] = exit_price
                trade_history[-1]['pnl'] = trade_pnl
                trade_history[-1]['exit_reason'] = exit_reason

        # -------------------------------------------------------
        # (3) 현재 시점(position, 누적손익) 기록
        # -------------------------------------------------------
        # in_position에 따라 1(포지션 보유) / 0(포지션 없음)으로 기록
        df.at[df.index[i], 'position'] = 1 if in_position else 0

        # (4) 누적 손익(cum_pnl) 계산
        if i == 0:
            # 첫 봉이라면, 현재 pnl값 그대로 사용
            df.at[df.index[i], 'cum_pnl'] = df.at[df.index[i], 'pnl']
        else:
            # 이전 봉의 cum_pnl에 현재 pnl을 더해 업데이트
            df.at[df.index[i], 'cum_pnl'] = df.at[df.index[i-1], 'cum_pnl'] + df.at[df.index[i], 'pnl']

    # -----------------------------------------------------------
    # (D) 모든 데이터(봉)를 순회한 후, 요약 결과 계산
    # -----------------------------------------------------------
    # 1) 전체 순회가 끝나면 df['pnl']의 합이 총 손익이 됨
    total_net_profit = df['pnl'].sum()

    # 2) total_trades: 실제로 체결된 거래(진입 후 청산) 횟수
    #    trade_history에는 진입 기록이 모두 들어있지만, exit_price가 None이 아니어야
    #    실제로 청산된 거래이므로 pnl이 존재하는 거래만 세어줌
    total_trades = sum(1 for t in trade_history if t['pnl'] is not None)

    # 3) 승리(수익실현)한 트레이드 개수
    wins = sum(1 for t in trade_history if t['pnl'] is not None and t['pnl'] > 0)
    losses = total_trades - wins

    # 4) 승률(Win Rate)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    # 5) 요약 정보 딕셔너리
    summary = {
        'Initial Capital': initial_capital,  # 초기 자본
        'Final Capital': capital,            # 최종 자본 (실현손익 반영)
        'Total Net Profit': total_net_profit,# 총 손익 (df['pnl'].sum())
        'Total Trades': total_trades,        # 총 거래 횟수(진입 & 청산이 완료된 것)
        'Win Rate (%)': win_rate,            # 승률
        'Wins': wins,                        # 수익 낸 거래 수
        'Losses': losses,                    # 손실 낸 거래 수
    }

    # -----------------------------------------------------------
    # (E) 결과 반환
    # -----------------------------------------------------------
    # 1) df: 각 시점의 포지션/손익 정보가 추가된 데이터
    # 2) trade_history: 개별 거래마다의 상세 내역
    # 3) summary: 최종 결과 요약
    return df, trade_history, summary
