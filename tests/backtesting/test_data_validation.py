# tests/backtesting/test_data_validation.py

import pandas as pd
import pytest
# 데이터 로딩 및 검증 함수 load_data를 임포트합니다.
from backtesting.steps.data_loader import load_data

# 더미 백테스터 클래스를 정의합니다.
class DummyBacktester:
    """
    백테스트 로직에서 사용하기 위한 최소한의 속성을 가진 더미 클래스.
    
    Attributes:
        symbol (str): 거래할 자산 심볼 (예: "BTC/USDT")
        df_short (pd.DataFrame): 짧은 주기 데이터 (초기값 None)
        df_long (pd.DataFrame): 긴 주기 데이터 (초기값 None)
    """
    def __init__(self, symbol="BTC/USDT"):
        self.symbol = symbol
        self.df_short = None
        self.df_long = None

def dummy_fetch_ohlcv_records(table_name, start_date, end_date):
    """
    fetch_ohlcv_records 함수의 더미 구현.
    
    - 유효한 날짜 범위(start_date < end_date)인 경우, non-empty DataFrame 반환.
    - 그렇지 않으면 빈 DataFrame 반환.
    
    Parameters:
        table_name (str): 데이터베이스 테이블 이름 (사용되지만 실제로는 dummy 데이터 생성)
        start_date (str): 데이터 시작 날짜
        end_date (str): 데이터 종료 날짜
        
    Returns:
        pd.DataFrame: 생성된 OHLCV 데이터 또는 빈 DataFrame
    """
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

# 모든 테스트에서 fetch_ohlcv_records를 dummy_fetch_ohlcv_records로 패치합니다.
@pytest.fixture(autouse=True)
def patch_fetch_ohlcv(monkeypatch):
    """
    pytest의 monkeypatch를 사용하여 데이터 로더에서 fetch_ohlcv_records 함수를
    dummy_fetch_ohlcv_records로 교체하는 fixture.
    
    Parameters:
        monkeypatch: pytest fixture
        
    Returns:
        없음
    """
    monkeypatch.setattr("backtesting.steps.data_loader.fetch_ohlcv_records", dummy_fetch_ohlcv_records)

def test_load_data_valid():
    """
    올바른 날짜 범위를 사용하여 load_data 함수가 올바른 데이터를 로드하는지 테스트합니다.
    
    - DummyBacktester 객체를 생성하고, 테이블 포맷 및 타임프레임, 시작일, 종료일을 설정합니다.
    - load_data를 호출한 후, df_short와 df_long 속성이 None이 아니고 빈 DataFrame이 아님을 확인합니다.
    
    Parameters:
        없음
        
    Returns:
        없음 (assertion을 통해 테스트 결과를 검증)
    """
    backtester = DummyBacktester(symbol="BTC/USDT")
    short_table_format = "ohlcv_{symbol}_{timeframe}"
    long_table_format = "ohlcv_{symbol}_{timeframe}"
    short_tf = "1d"
    long_tf = "1d"
    start_date = "2023-01-01"
    end_date = "2023-01-10"
    # load_data 호출 시 정상적인 데이터가 로드되어야 함.
    load_data(backtester, short_table_format, long_table_format, short_tf, long_tf, start_date, end_date)
    assert backtester.df_short is not None and not backtester.df_short.empty
    assert backtester.df_long is not None and not backtester.df_long.empty

def test_load_data_invalid():
    """
    잘못된 날짜 범위 (시작일이 종료일보다 늦은 경우)를 사용하여 load_data 함수가 ValueError를 발생시키는지 테스트합니다.
    
    Parameters:
        없음
        
    Returns:
        없음 (ValueError 발생 여부로 검증)
    """
    backtester = DummyBacktester(symbol="BTC/USDT")
    short_table_format = "ohlcv_{symbol}_{timeframe}"
    long_table_format = "ohlcv_{symbol}_{timeframe}"
    short_tf = "1d"
    long_tf = "1d"
    # 시작일이 종료일보다 늦은 잘못된 날짜 범위 설정
    start_date = "2023-01-10"
    end_date = "2023-01-01"
    with pytest.raises(ValueError):
        load_data(backtester, short_table_format, long_table_format, short_tf, long_tf, start_date, end_date)
