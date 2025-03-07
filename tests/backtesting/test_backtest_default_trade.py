# tests/backtesting/test_backtest_default_trade.py
import pytest
from backtesting.backtester import Backtester
from config.config_manager import ConfigManager
import pandas as pd

@pytest.fixture
def sample_ohlcv_data():
    # 간단한 테스트용 OHLCV 데이터 생성 (예: 60일치 데이터)
    dates = pd.date_range(start="2022-01-01", periods=60, freq="D")
    df = pd.DataFrame({
        "open": pd.np.linspace(100, 160, 60),
        "high": pd.np.linspace(105, 165, 60),
        "low": pd.np.linspace(95, 155, 60),
        "close": pd.np.linspace(100, 160, 60),
        "volume": [1000] * 60
    }, index=dates)
    return df

def test_default_config_backtest(sample_ohlcv_data, monkeypatch):
    # 백테스터 인스턴스 생성 (간단히 기본 파라미터 사용)
    asset = "BTC/USDT"
    symbol_key = asset.replace("/", "").lower()
    bt = Backtester(symbol=asset, account_size=10000)
    
    # 데이터 로드 부분: 실제 DB 대신 sample 데이터 사용
    bt.df_long = sample_ohlcv_data.copy()
    bt.df_short = sample_ohlcv_data.copy()
    bt.df_train = sample_ohlcv_data.copy()
    
    # HMM, 지표 적용, short DF 업데이트 등의 외부 의존성 제거를 위해 dummy 함수 삽입
    bt.apply_indicators = lambda: None
    bt.update_hmm_regime = lambda dynamic_params: pd.Series(["bullish"] * len(bt.df_long), index=bt.df_long.index)
    bt.update_short_dataframe = lambda regime_series, dynamic_params: None
    bt.ensemble_manager.get_final_signal = lambda regime, liquidity, data, current_time, data_weekly=None: "enter_long"
    
    # ConfigManager 기본 파라미터 사용
    cm = ConfigManager()
    default_params = cm.get_defaults()
    
    # 백테스트 실행 (주문이 발생하면 trades 리스트에 값이 채워져야 함)
    trades, trade_logs = bt.run_backtest_pipeline(dynamic_params=default_params)
    
    # 최소 1건 이상의 거래가 발생했는지 검증
    assert isinstance(trades, list)
    # 거래 건수가 0이면 문제로 간주 (테스트 환경에 따라 다를 수 있으므로 임계값은 조정 가능)
    assert len(trades) > 0, "기본 파라미터로 실행 시 거래가 체결되어야 합니다."
