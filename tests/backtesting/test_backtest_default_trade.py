# tests/backtesting/test_backtest_default_trade.py

import pytest
# 백테스트 로직이 구현된 Backtester 클래스를 임포트
from backtesting.backtester import Backtester
# 기본 파라미터 관리를 위한 ConfigManager 클래스 임포트
from config.config_manager import ConfigManager
import numpy as np
import pandas as pd

@pytest.fixture
def sample_ohlcv_data():
    """
    간단한 테스트용 OHLCV 데이터를 생성하는 fixture.
    
    - 시작일: 2022-01-01, 기간: 60일, 빈도: 일간(D)
    - open, high, low, close 가격은 선형 보간(linear space)로 생성
    - volume은 모든 값이 1000인 리스트
    
    Returns:
        pd.DataFrame: 생성된 OHLCV 데이터 (인덱스는 날짜)
    """
    # 날짜 범위를 생성합니다.
    dates = pd.date_range(start="2022-01-01", periods=60, freq="D")
    # OHLCV 데이터를 생성 (np.linspace로 선형 분포 생성)
    df = pd.DataFrame({
        "open": np.linspace(100, 160, 60),   # 시작값 100에서 160까지 60단계 분포
        "high": np.linspace(105, 165, 60),
        "low": np.linspace(95, 155, 60),
        "close": np.linspace(100, 160, 60),
        "volume": [1000] * 60
    }, index=dates)
    return df

def test_default_config_backtest(sample_ohlcv_data, monkeypatch):
    """
    기본 파라미터를 사용하여 백테스트 파이프라인이 정상적으로 실행되고 거래가 발생하는지 검증합니다.
    
    - Backtester 인스턴스를 생성한 후, 테스트용 데이터를 각 데이터프레임에 할당합니다.
    - 외부 의존성을 제거하기 위해 HMM, 지표 적용, short 데이터 업데이트 및 시그널 결정 함수를 dummy lambda 함수로 오버라이드합니다.
    - ConfigManager를 통해 기본 파라미터를 가져와 백테스트를 실행합니다.
    - 거래(trades) 리스트에 최소 1건 이상의 거래가 발생했는지 확인합니다.
    
    Parameters:
        sample_ohlcv_data (pd.DataFrame): 위에서 생성한 테스트용 OHLCV 데이터
        monkeypatch: pytest의 monkeypatch fixture, 외부 의존성을 오버라이드하기 위해 사용
        
    Returns:
        없음 (assertion을 통해 테스트 결과를 검증)
    """
    # 테스트할 자산 심볼 설정
    asset = "BTC/USDT"
    symbol_key = asset.replace("/", "").lower()  # 예: btcusdt
    # 백테스터 객체 생성, 초기 계좌 사이즈 10,000달러
    bt = Backtester(symbol=asset, account_size=10000)
    
    # 테스트를 위해 sample_ohlcv_data를 백테스터의 각 데이터프레임에 할당
    bt.df_long = sample_ohlcv_data.copy()
    bt.df_short = sample_ohlcv_data.copy()
    bt.df_train = sample_ohlcv_data.copy()
    
    # 외부 의존성 제거: 각 메서드를 dummy 함수(lambda)로 오버라이드
    bt.apply_indicators = lambda: None
    bt.update_hmm_regime = lambda dynamic_params: pd.Series(["bullish"] * len(bt.df_long), index=bt.df_long.index)
    bt.update_short_dataframe = lambda regime_series, dynamic_params: None
    bt.ensemble_manager.get_final_signal = lambda regime, liquidity, data, current_time, data_weekly=None: "enter_long"
    
    # ConfigManager를 사용하여 기본 파라미터(defaults) 가져오기
    cm = ConfigManager()
    default_params = cm.get_defaults()
    
    # 백테스트 파이프라인 실행: dynamic_params를 기본 파라미터로 사용
    trades, trade_logs = bt.run_backtest_pipeline(dynamic_params=default_params)
    
    # trades가 리스트 타입인지 확인
    assert isinstance(trades, list)
    # 거래가 최소 1건 이상 발생했는지 확인 (테스트 환경에 따라 임계값 조정 가능)
    assert len(trades) > 0, "기본 파라미터로 실행 시 거래가 체결되어야 합니다."
