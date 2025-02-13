# Quantitative Trading System

본 프로젝트는 알고리즘 트레이딩을 위한 백테스팅, 전략 평가, 파라미터 최적화 및 리스크 관리를 통합적으로 지원하는 시스템입니다.  
주요 기능으로는 데이터 수집 및 DB 관리, 다양한 거래 전략 구현, HMM 기반 시장 레짐 분석, 포지션 및 자산 관리, 종합 전략 신호 산출, 그리고 상세한 로그 및 성과 리포트 생성이 포함됩니다.

---

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [프로젝트 구조](#프로젝트-구조)
  - [backtesting](#backtesting)
  - [config](#config)
  - [core](#core)
  - [data](#data)
  - [logs](#logs)
  - [markets](#markets)
  - [strategies](#strategies)
  - [trading](#trading)
  - [tests](#tests)
- [주요 실행 파일](#주요-실행-파일)

---

## 프로젝트 개요

이 시스템은 다음과 같은 주요 기능을 제공합니다:

- **데이터 수집 및 저장**:  
  `data/ohlcv/ohlcv_fetcher.py` 및 `data/ohlcv/ohlcv_pipeline.py`를 통해 외부 API (ccxt 등)로부터 OHLCV 데이터를 수집하여, `data/db/db_manager.py`를 이용해 PostgreSQL 데이터베이스에 저장합니다.

- **백테스팅 및 전략 평가**:  
  `backtesting/backtester.py`와 그 하위 단계(`steps/data_loader.py`, `steps/indicator_applier.py`, `steps/hmm_manager.py`, `steps/order_manager.py`)를 통해 백테스트를 수행하며, 성과 지표는 `backtesting/performance.py`에서 계산됩니다.

- **전략 및 파라미터 최적화**:  
  `strategies/optimizer.py`와 `strategies/param_analysis.py`를 통해 다양한 전략 파라미터에 대한 민감도 분석 및 최적화를 지원합니다.

- **거래 전략 구현**:  
  개별 전략은 `strategies/base_strategy.py`를 상속받아 구현되며, `strategies/trading_strategies.py`에서 여러 전략 (Select, TrendFollowing, Breakout, CounterTrend, HighFrequency, Weekly 전략 등)을 제공하고, `trading/ensemble.py`에서 종합 신호를 산출합니다.

- **리스크 및 자산 관리**:  
  `trading/risk_manager.py`에서 포지션 사이즈 산출, 분할 진입, 스케일인, 리스크 파라미터 계산 등을 지원하며, `trading/asset_manager.py`에서 시장 레짐에 따른 자산 배분 및 리밸런싱을 관리합니다.

- **로그 관리 및 리포트 생성**:  
  `logs/logger_config.py`를 통해 세부 로그와 집계 로그(AggregatingHandler)를 구성하며, `logs/final_report.py`에서 최종 성과 및 파라미터 민감도 리포트를 생성합니다.

---

## 프로젝트 구조

### backtesting
- **backtester.py**  
  백테스팅 전반을 관리하는 메인 클래스. 데이터 로드, 인디케이터 적용, HMM 레짐 업데이트, 주문 및 포지션 관리를 수행하며, 백테스트 실행 결과(거래 내역 및 로그)를 산출합니다.
  
- **steps/**  
  - **data_loader.py**: DB에서 OHLCV 데이터를 불러와 정렬 및 추가 지표 (Bollinger Bands 등) 계산을 수행합니다.
  - **hmm_manager.py**: HMM 모델을 사용하여 시장 레짐을 업데이트합니다.
  - **indicator_applier.py**: SMA, RSI, MACD 등 인디케이터를 적용합니다.
  - **order_manager.py**: 학습, extra, holdout 데이터에 대한 주문 실행 및 포지션 관리 로직을 포함합니다.
  
- **performance.py**  
  거래 내역을 기반으로 월별 및 전체 성과(ROI, 누적 수익률, 샤프/소르티노/칼마 지수 등)를 계산합니다.

---

### config
- **config_manager.py**  
  기본 설정 파라미터를 정의하고, 시장 데이터에 따라 동적으로 업데이트하거나, 최적화 결과와 병합하는 기능을 제공합니다.

---

### core
- **account.py**  
  거래 계좌를 관리하며, 현물 및 스테이블코인 잔고, 포지션 추가/제거, 거래 후 잔고 업데이트, 자산 전환 기능을 제공합니다.
- **position.py**  
  개별 포지션(진입/청산 내역)을 관리하며, 부분 청산, 평균 진입 가격 계산 등의 기능을 지원합니다.

---

### data
- **db/**
  - **db_config.py**: 데이터베이스 연결 정보 (환경변수 기반)를 설정합니다.
  - **db_manager.py**: SQLAlchemy를 사용하여 OHLCV 데이터를 데이터베이스에 저장하거나 불러옵니다.
- **ohlcv/**
  - **ohlcv_aggregator.py**: 일별 OHLCV 데이터를 주간 데이터로 집계하고, 주간 인디케이터(SMA, 모멘텀 등)를 계산합니다.
  - **ohlcv_fetcher.py**: ccxt를 사용하여 외부 거래소로부터 OHLCV 데이터를 수집합니다.
  - **ohlcv_pipeline.py**: 지정된 심볼과 타임프레임에 대해 데이터를 수집 후 DB에 저장하는 파이프라인을 구현합니다.

---

### logs
- **logger_config.py**  
  로깅 설정 (파일 및 콘솔 핸들러, QueueHandler, RotatingFileHandler, AggregatingHandler 등)을 구성합니다.
- **aggregating_handler.py**  
  로그 레코드를 집계하여 실행 종료 시 전체 로그 발생 건수를 요약하여 출력합니다.
- **final_report.py**  
  백테스트 및 파라미터 민감도 분석 결과를 기반으로 최종 성과 리포트 및 민감도 리포트를 생성합니다.
- **logging_util.py**  
  로그 이벤트 기록 및 로그 파일 관리를 위한 유틸리티 기능을 제공합니다.
- **logging_util.py** 외에도 여러 로깅 관련 모듈이 프로젝트 전반에 걸쳐 사용됩니다.

---

### markets
- **regime_filter.py**  
  단순 가격 변화율을 기반으로 시장 레짐(상승, 하락, 횡보)을 결정하고 필터링하는 기능을 제공합니다.
- **regime_model.py**  
  HMM (GaussianHMM)을 사용하여 보다 정교하게 시장 레짐을 예측하고, 재학습 조건(시간, 피처 변화)을 확인하는 기능을 포함합니다.

---

### strategies
- **base_strategy.py**  
  모든 거래 전략이 상속받을 기본 인터페이스를 정의합니다.
- **trading_strategies.py**  
  다양한 단기 및 주간 거래 전략(Select, TrendFollowing, Breakout, CounterTrend, HighFrequency, WeeklyBreakout, WeeklyMomentum)을 구현합니다.
- **optimizer.py**  
  Optuna를 사용한 파라미터 최적화 로직을 제공하며, 백테스트 결과를 기반으로 최적의 동적 파라미터를 산출합니다.
- **param_analysis.py**  
  지정된 파라미터 범위에 대해 백테스트 성과를 평가하고, 민감도 분석 결과를 산출합니다.

---

### trading
- **asset_manager.py**  
  계좌의 현물 및 스테이블코인 자산을 시장 레짐에 따라 리밸런싱하며, 자산 배분을 관리합니다.
- **calculators.py**  
  ATR, 스탑로스/테이크프로핏, 부분 청산 목표 등 거래 관련 수치 계산 기능을 제공합니다.
- **ensemble.py**  
  여러 전략의 신호를 종합하여 최종 거래 신호를 산출하는 앙상블 로직을 구현합니다.
- **indicators.py**  
  SMA, MACD, RSI, Bollinger Bands 등 기술적 지표를 계산하여 데이터프레임에 추가합니다.
- **risk_manager.py**  
  포지션 사이즈 계산, 분할 진입, 스케일인, 리스크 파라미터 조정 등 리스크 관리를 위한 다양한 기능을 제공합니다.
- **trade_executor.py**  
  거래 실행과 관련된 계산 로직(ATR, 동적 스탑로스/테이크프로핏, 부분 청산 목표 산출 등)을 캘큘레이터 모듈을 통해 제공합니다.

---

### tests
테스트 폴더는 pytest 기반의 단위 테스트 및 통합 테스트를 포함합니다.
- **conftest.py**: 테스트 실행 전 로그 파일 초기화 및 로거 재설정을 수행합니다.
- 각 모듈별 테스트 파일 (예: `test_aggregating_handler.py`, `test_asset_manager.py`, `test_base_strategy.py`, `test_calculators.py`, `test_config_manager.py`, `test_core_account.py`, `test_core_position.py`, `test_ensemble.py`, `test_indicators.py`, `test_ohlcv_aggregator.py`, `test_optimizer.py`, `test_param_analysis.py`, `test_performance_report.py`, `test_regime_filter.py`, `test_regime_model.py`, `test_risk_manager.py`, `test_trade_executor.py`, `test_weekly_strategies.py`)를 통해 각 기능별 정확성 및 안정성을 검증합니다.

---

## 주요 실행 파일

- **run_parameter_analysis.py**  
  파라미터 민감도 분석을 실행하여, 다양한 파라미터 설정에 따른 백테스트 성과를 평가하고, 최종 민감도 리포트를 생성합니다.

- **run_strategy_performance.py**  
  Walk-Forward 방식의 파라미터 최적화를 진행한 후, 여러 자산에 대해 백테스트를 실행하고, 최종 전략 성과 리포트를 생성합니다.
