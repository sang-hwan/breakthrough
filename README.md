# 트레이딩 봇 프로젝트

이 프로젝트는 기술적 트레이딩 봇 개발을 위한 통합 코드베이스로, 암호화폐 시장의 데이터 수집, 백테스팅, 전략 파라미터 최적화, 거래 실행, 리스크 관리, 로깅 및 성과 분석을 포함합니다.  
모든 모듈은 상호 연계되어 있으며, 전체 프로세스는 개발 환경에서 철저한 백테스트와 최적화를 통해 월간 ROI 2% 이상의 목표 달성을 위해 설계되었습니다.

---

## 디렉토리 및 파일 구성

### **backtesting 폴더**
- **backtester.py**  
  - **클래스: `Backtester`**  
    - **역할:**  
      백테스트 실행의 핵심 클래스로, 지정 심볼 및 계좌 환경에서 데이터 로드, 기술적 인디케이터 적용, 시장 레짐 업데이트(HMM 기반), 포지션 관리 및 최종 거래 청산을 포함한 전체 백테스트 프로세스를 수행합니다.
    - **주요 메서드:**  
      - `__init__`: 기본 설정(심볼, 계좌, 거래 수수료, 슬리피지 등) 초기화  
      - `load_data`: 단기 및 장기 OHLCV 데이터를 DB에서 로드  
      - `apply_indicators`: SMA, RSI, MACD, Bollinger Bands 등 주요 인디케이터 계산 적용  
      - `update_hmm_regime`: HMM 모델을 통한 시장 레짐(예: bullish, bearish, sideways) 업데이트  
      - `update_positions`: 포지션별 스탑로스 조정 및 관리  
      - `finalize_all_positions`: 백테스트 종료 시 미체결 포지션 청산 및 거래 내역 업데이트  
      - `run_backtest`: 전체 백테스트 워크플로우 실행 및 성과 데이터 반환

- **performance.py**  
  - **역할:**  
      백테스트 결과 거래 내역을 기반으로 월별 및 전체 성과(ROI, 누적 수익률, 연간 수익률, 변동성, 샤프/소르티노/칼마 지수, 최대 낙폭 등)를 계산하는 함수를 제공합니다.
  - **주요 함수:**  
      - `calculate_monthly_performance`: 월별 PnL 및 ROI 계산  
      - `calculate_overall_performance`: 전체 성과 지표 산출  
      - `compute_performance`: 종합 성과 리포트 생성

---

### **strategy_tuning 폴더**
- **dynamic_param_manager.py**  
  - **클래스: `DynamicParamManager`**  
    - **역할:**  
      프로젝트 전반에 적용할 기본 파라미터를 관리하고, 시장 데이터(변동성, 추세, 거래량 등)에 따라 동적으로 업데이트하며, 최적화 결과와 병합하여 가중 평균 방식으로 반환합니다.
- **parameter_analysis.py**  
  - **역할:**  
      파라미터 민감도 분석을 위한 함수(`run_sensitivity_analysis`)를 제공하여, 특정 파라미터 범위 내에서 백테스트 성과(ROI 등)를 평가합니다.
- **optimizer.py**  
  - **클래스: `DynamicParameterOptimizer`**  
    - **역할:**  
      Optuna 라이브러리를 이용하여 동적 파라미터(예: risk_per_trade, atr_multiplier, profit_ratio 등)를 최적화합니다.
    - **주요 메서드:**  
      - `objective`: 각 트라이얼마다 백테스트를 실행하여 평가 점수를 산출  
      - `optimize`: 지정한 트라이얼 수 만큼 최적화 후 최적 파라미터 도출

---

### **data_collection 폴더**
- **db_config.py**  
  - **역할:**  
      환경변수를 통해 데이터베이스 접속 정보를 설정합니다.
- **db_manager.py**  
  - **역할:**  
      SQLAlchemy 및 psycopg2를 사용해 데이터베이스와 인터랙션하며, OHLCV 데이터의 삽입 및 조회 기능을 제공합니다.  
  - **주요 함수:**  
      - `insert_on_conflict`: 중복 timestamp 발생 시 삽입 건너뜀  
      - `insert_ohlcv_records`: 대용량 데이터를 chunk 단위로 저장  
      - `fetch_ohlcv_records`: 지정 기간의 OHLCV 데이터를 조회
- **ohlcv_fetcher.py**  
  - **역할:**  
      ccxt 라이브러리를 이용해 암호화폐 거래소에서 OHLCV 데이터를 수집합니다.  
  - **주요 함수:**  
      - `fetch_historical_ohlcv_data`: 지정 시작일 이후 과거 데이터 수집  
      - `fetch_latest_ohlcv_data`: 최신 데이터 수집  
      - `get_top_market_cap_symbols`: 거래량 기준 상위 심볼 및 데이터 가용성 확인
- **ohlcv_pipeline.py**  
  - **역할:**  
      여러 심볼 및 타임프레임에 대해 수집한 OHLCV 데이터를 DB에 저장하는 파이프라인을 구성합니다.

---

### **logs 폴더**
- **aggregating_handler.py**  
  - **클래스: `AggregatingHandler`**  
    - **역할:**  
      동일 모듈/함수에서 발생하는 반복 로그를 집계하여, 설정 임계치 도달 시 요약 로그를 출력합니다.
- **final_report.py**  
  - **역할:**  
      백테스트 및 거래 성과 데이터를 기반으로 최종 성과 리포트를 생성하고, 이를 로그로 출력합니다.
- **logger_config.py**  
  - **역할:**  
      전역 로깅 설정(회전 파일 핸들러, 콘솔 핸들러, AggregatingHandler)을 초기화하며, 모듈별 로거를 반환합니다.
- **logging_util.py**  
  - **역할:**  
      이벤트 로깅 및 로그 파일 관리를 위한 유틸리티 클래스를 제공합니다.  
  - **주요 클래스:**  
      - `LoggingUtil` (alias: `EventLogger`): 지정 모듈 내 이벤트를 INFO 레벨로 기록하여, AggregatingHandler를 통한 로그 요약을 지원

---

### **markets_analysis 폴더**
- **hmm_model.py**  
  - **클래스: `MarketRegimeHMM`**  
    - **역할:**  
      Gaussian HMM을 활용해 시장 레짐(예: bullish, bearish, sideways)을 학습 및 예측하며, 재학습 조건(시간 간격, 피처 변화 등)을 고려합니다.
    - **주요 메서드:**  
      - `train`, `predict`, `predict_proba`, `update`
- **regime_filter.py**  
  - **역할:**  
      가격 데이터 기반으로 시장 레짐을 결정하거나, 목표 레짐과 일치하는지 필터링하는 로직을 제공합니다.
  - **주요 함수:**  
      - `determine_market_regime`, `filter_regime`, `filter_by_confidence`

---

### **tests 폴더**
- **conftest.py**  
  - **역할:**  
      Pytest 실행 전 로그 파일 삭제 및 루트 로거 재초기화를 수행하는 설정 스크립트
- **test_auto_optimization_trigger.py**  
  - **역할:**  
      월간 ROI가 2% 미만인 경우 자동 최적화 트리거를 활성화하는 로직을 테스트하는 더미 함수를 포함
- **test_logging_summary.py**  
  - **역할:**  
      LoggingUtil(또는 EventLogger)를 사용하여 반복 로그가 일정 임계치 도달 시 요약 로그가 생성되는지 검증
- **test_performance_report.py**  
  - **역할:**  
      최종 성과 리포트 생성 함수(`generate_final_report`)가 올바른 지표(ROI, 거래 횟수, 월별 데이터 등)를 출력하는지 테스트

---

### **trading 폴더**
- **account.py**  
  - **클래스: `Account`**  
    - **역할:**  
      거래 계좌의 잔고 관리 및 포지션 관리를 담당하며, 현물 및 스테이블코인 간 전환 기능을 제공합니다.
- **asset_manager.py**  
  - **클래스: `AssetManager`**  
    - **역할:**  
      시장 레짐(예: bullish, bearish, sideways)에 따라 자산 배분 및 리밸런싱을 수행합니다.
- **ensemble_manager.py**  
  - **클래스: `EnsembleManager`**  
    - **역할:**  
      여러 거래 전략에서 산출된 신호들을 가중치 기반으로 집계하여 최종 거래 신호(enter_long, exit_all, hold 등)를 결정합니다.
- **indicators.py**  
  - **역할:**  
      TA 라이브러리를 활용하여 SMA, MACD, RSI, Bollinger Bands 등 주요 기술적 인디케이터를 계산합니다.
- **positions.py**  
  - **클래스: `TradePosition`**  
    - **역할:**  
      개별 포지션과 그 내의 체결(실행) 내역을 관리하며, 부분 청산 및 실행 추가 기능을 제공합니다.
- **risk_manager.py**  
  - **클래스: `RiskManager`** (정적 메서드 집합)  
    - **역할:**  
      포지션 사이즈 계산, 분할 진입, 스케일인(scale-in) 시도, 및 시장 레짐에 따른 리스크 파라미터 조정을 지원합니다.
- **strategies.py**  
  - **클래스: `TradingStrategies`**  
    - **역할:**  
      여러 개별 거래 전략(예: 트렌드 팔로잉, 브레이크아웃, 반대 추세, 고빈도 등)을 구현하며, 다양한 보조 신호를 종합해 최종 거래 신호를 산출합니다.
- **trade_manager.py**  
  - **클래스: `TradeManager`** (정적 메서드 집합)  
    - **역할:**  
      ATR 기반 스탑로스, 트레일링 스탑, 고정 테이크 프로핏 설정, 추세 종료 판단, 부분 청산 목표 계산 등 거래 실행 관련 계산 로직을 제공합니다.

---

### **프로젝트 루트 파일**
- **.env**  
  - 데이터베이스 연결 정보 및 로깅 설정 등 환경변수를 정의합니다.
- **requirements.txt**  
  - 프로젝트에 필요한 Python 패키지(예: pandas, numpy, SQLAlchemy, ccxt, optuna, ta, hmmlearn, python-dotenv, pytest 등)를 나열합니다.
- **run_parameter_analysis.py**  
  - **역할:**  
      파라미터 민감도 분석을 수행하는 스크립트로, 지정 범위 내 파라미터에 대해 백테스트를 실행하고 최종 결과를 리포트합니다.
- **run_strategy_performance.py**  
  - **역할:**  
      전체 테스트 실행 스크립트로, 기존 로그 파일 초기화, DynamicParameterOptimizer를 통한 최적 파라미터 도출, 각 심볼(BTC/USDT, ETH/USDT, XRP/USDT)에 대한 백테스트 실행 및 성과 리포트 생성을 순차적으로 수행합니다.

---

## 실행 방법

1. **환경설정:**  
   - 프로젝트 루트에 위치한 `.env` 파일을 열어 데이터베이스 접속 정보 및 로깅 관련 환경변수를 확인하고 수정합니다.
2. **패키지 설치:**  
   - `pip install -r requirements.txt` 명령어를 통해 필요한 Python 패키지를 설치합니다.
3. **백테스트 및 성과 분석 실행:**  
   - `python run_strategy_performance.py` 명령어를 실행하여, 각 심볼에 대한 백테스트를 수행하고 최종 성과 리포트를 로그로 출력합니다.
4. **파라미터 민감도 분석 (선택 사항):**  
   - `python run_parameter_analysis.py --param_name profit_ratio --start 0.05 --end 0.15 --steps 10` 와 같이 실행하여 특정 파라미터 범위에 따른 백테스트 성과를 분석할 수 있습니다.

---

## 결론

이 프로젝트는 최소한의 리소스와 효율적인 코드 개선을 통해 월간 ROI 2% 이상의 성과 달성을 목표로 합니다. 각 모듈은 기존 코드 구조를 최대한 유지하면서, 필수 기능과 로깅, 리스크 관리, 전략 최적화를 포함하도록 설계되었습니다.  
백테스트 환경에서 충분히 검증된 후, 실제 운영 환경으로의 전환을 고려할 수 있습니다.
