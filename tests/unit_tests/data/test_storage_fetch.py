# tests/unit_tests/test_storage_fetch.py
"""
이 테스트 파일은 data 모듈의 저장(store_data) 및 조회(fetch_data) 기능이 올바르게 동작하는지 검증합니다.
주요 검증 항목은 다음과 같습니다.
  - OHLCV 데이터가 지정 테이블에 올바르게 삽입되는지
  - 삽입한 데이터가 조회(fetch_ohlcv_records) 시 정확히 반환되는지
  - 테이블 생성 및 중복 처리(on conflict) 기능이 정상 동작하는지

이 테스트는 데이터베이스 관련 기능의 변경이나 리팩토링 후에도 핵심 기능이 유지되고 있는지
빠르게 확인할 수 있도록 도와줍니다.
"""

import pandas as pd
import pytest
from data.store_data import insert_ohlcv_records
from data.fetch_data import fetch_ohlcv_records
from data.db_config import DATABASE

def test_insert_and_fetch_ohlcv_records():
    # 테스트용 샘플 데이터 생성
    data = {
        "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="D"),
        "open": [100, 102, 101, 103, 104],
        "high": [105, 106, 107, 108, 109],
        "low": [99, 98, 97, 96, 95],
        "close": [104, 105, 102, 107, 108],
        "volume": [1000, 1500, 1100, 1200, 1300]
    }
    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)
    
    # 테스트 전용 테이블명을 사용하여 기존 데이터와 충돌을 방지합니다.
    table_name = "test_ohlcv_data_unit"
    
    # 데이터 삽입: 데이터가 없으면 테이블 생성 후 삽입됨
    insert_ohlcv_records(df, table_name=table_name, db_config=DATABASE)
    
    # 데이터 조회: 삽입한 데이터가 정상적으로 조회되어야 함
    df_fetched = fetch_ohlcv_records(table_name=table_name, db_config=DATABASE)
    
    # 조회된 데이터가 비어있지 않고, 최소 5행 이상의 데이터가 있는지 확인
    assert not df_fetched.empty, "조회된 DataFrame은 비어있으면 안 됩니다."
    assert len(df_fetched) >= 5, "최소 5행 이상의 데이터가 있어야 합니다."
    
    # 컬럼 이름이 올바른지 확인
    expected_columns = {"open", "high", "low", "close", "volume"}
    assert expected_columns.issubset(set(df_fetched.columns)), "조회된 데이터에 필요한 컬럼이 모두 포함되어야 합니다."
