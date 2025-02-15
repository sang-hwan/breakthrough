# tests/test_data_validation.py
import pandas as pd
import pytest
from backtesting.steps.data_loader import load_data

# 더미 백테스터 클래스 정의
class DummyBacktester:
    def __init__(self, symbol="BTC/USDT"):
        self.symbol = symbol
        self.df_short = None
        self.df_long = None

# fetch_ohlcv_records를 더미 함수로 대체 (유효한 날짜 범위에서는 non-empty DataFrame, 그렇지 않으면 empty DataFrame 반환)
def dummy_fetch_ohlcv_records(table_name, start_date, end_date):
    if start_date and end_date and start_date < end_date:
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        df = pd.DataFrame({
            "open": [100] * len(dates),
            "high": [110] * len(dates),
            "low": [90] * len(dates),
            "close": [105] * len(dates),
            "volume": [1000] * len(dates)
        }, index=dates)
        return df
    else:
        return pd.DataFrame()

# 모든 테스트에서 fetch_ohlcv_records를 패치
@pytest.fixture(autouse=True)
def patch_fetch_ohlcv(monkeypatch):
    monkeypatch.setattr("backtesting.steps.data_loader.fetch_ohlcv_records", dummy_fetch_ohlcv_records)

def test_load_data_valid():
    backtester = DummyBacktester(symbol="BTC/USDT")
    short_table_format = "ohlcv_{symbol}_{timeframe}"
    long_table_format = "ohlcv_{symbol}_{timeframe}"
    short_tf = "1d"
    long_tf = "1d"
    start_date = "2023-01-01"
    end_date = "2023-01-10"
    # load_data 호출 시 정상적인 데이터를 반환해야 함.
    load_data(backtester, short_table_format, long_table_format, short_tf, long_tf, start_date, end_date)
    assert backtester.df_short is not None and not backtester.df_short.empty
    assert backtester.df_long is not None and not backtester.df_long.empty

def test_load_data_invalid():
    backtester = DummyBacktester(symbol="BTC/USDT")
    short_table_format = "ohlcv_{symbol}_{timeframe}"
    long_table_format = "ohlcv_{symbol}_{timeframe}"
    short_tf = "1d"
    long_tf = "1d"
    start_date = "2023-01-10"
    end_date = "2023-01-01"  # 잘못된 날짜 범위: 시작일 > 종료일
    with pytest.raises(ValueError):
        load_data(backtester, short_table_format, long_table_format, short_tf, long_tf, start_date, end_date)
