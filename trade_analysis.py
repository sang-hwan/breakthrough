# trade_analysis.py

import pandas as pd

def analyze_trade_data(trades_csv='backtest_trades.csv', logs_csv='trade_logs.csv'):
    # 거래 내역 데이터 로드
    trades_df = pd.read_csv(trades_csv, parse_dates=['entry_time', 'exit_time'])
    logs_df = pd.read_csv(logs_csv, parse_dates=['entry_time', 'exit_time'])
    
    # 전체 거래 건수, 총 pnl, 승률 계산
    total_trades = len(trades_df)
    total_pnl = trades_df['pnl'].sum()
    wins = (trades_df['pnl'] > 0).sum()
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0.0

    print(f"전체 거래 건수: {total_trades}")
    print(f"총 PnL: {total_pnl:.2f}")
    print(f"승률: {win_rate:.2f}%")
    
    # 거래 사유별 건수 및 평균 pnl 분석 (예: stop_loss, take_profit, scale_in, final_exit 등)
    reason_group = trades_df.groupby('reason')
    for reason, group in reason_group:
        count = len(group)
        avg_pnl = group['pnl'].mean()
        win_rate_reason = (group['pnl'] > 0).mean() * 100
        print(f"[{reason}] 거래 건수: {count}, 평균 PnL: {avg_pnl:.2f}, 승률: {win_rate_reason:.2f}%")
    
    # 거래별 pnl 분포 및 손실 원인 예시
    trades_df['pnl_category'] = trades_df['pnl'].apply(lambda x: 'win' if x > 0 else 'loss')
    print("\n거래별 pnl 요약 (승/패 기준):")
    print(trades_df.groupby('pnl_category')['pnl'].describe())

if __name__ == "__main__":
    analyze_trade_data()
