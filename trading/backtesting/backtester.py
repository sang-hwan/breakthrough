# backtesting/backtester.py

# pandas 라이브러리: 데이터프레임 조작 및 시간 관련 처리를 위해 사용
import pandas as pd

# 로그 설정을 위한 모듈 (로깅 설정 및 유틸리티 제공)
from logging.logger_config import setup_logger
# 리스크 관리 관련 모듈: 포지션 사이즈 계산, 리스크 제어 기능 제공
from trading.risk_manager import RiskManager 
# 거래 실행 관련 모듈: 거래 체결, 슬리피지 및 수수료 적용 기능 제공
from trading.trade_executor import TradeExecutor
# 계좌 관리 관련 모듈: 초기 자본, 잔액 업데이트 등 관리
from asset_position.account import Account
# 포지션(거래 단위) 관련 모듈
from asset_position.position import Position
# 자산 배분 및 관리 모듈
from asset_position.asset_manager import AssetManager
# 앙상블 전략 관련 모듈 (여러 전략 결합 시 사용)
from signal_calculation.ensemble import Ensemble
# 환경설정 관리 모듈
from parameter_management.config_manager import ConfigManager
# 추가 로깅 유틸리티
from logging.logging_util import LoggingUtil

# 전역 로깅 유틸 객체: 현재 모듈의 이름으로 로그 기록
log_util = LoggingUtil(__name__)

class Backtester:
    """
    백테스터 클래스
    -----------------
    여러 거래 전략 및 리스크 관리, 데이터 로딩, 인디케이터 적용, 거래 신호 처리 등을 포함하는 
    백테스팅 전체 파이프라인을 관리하는 클래스입니다.
    
    전역 변수 및 객체:
      - symbol: 거래할 종목 심볼 (예: "BTC/USDT")
      - fee_rate: 거래 수수료 비율
      - slippage_rate: 슬리피지 비율
      - final_exit_slippage: 최종 청산 시 적용할 슬리피지
      - positions: 현재 보유한 포지션 리스트
      - trades: 실행된 거래 내역 리스트
      - trade_logs: 상세 거래 로그 리스트
      - logger: 모듈 전용 로깅 객체
      - config_manager: 설정 및 파라미터 관리 객체
      - account: 거래 계좌 관리 객체
      - asset_manager: 자산 배분 관리 객체
      - ensemble_manager: 앙상블 전략 관리 객체
      - risk_manager: 리스크 관리 객체
      - clock: 현재 시간을 반환하는 람다 함수 (기본: pd.Timestamp.now)
      - state: 상태 정보를 저장하는 딕셔너리
      - 그 외 데이터프레임 및 시간 관련 변수들
    """
    def __init__(self, symbol="BTC/USDT", account_size=10000.0, fee_rate=0.001, 
                 slippage_rate=0.0005, final_exit_slippage=0.0):
        # 거래할 종목과 관련 파라미터 설정
        self.symbol = symbol
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.final_exit_slippage = final_exit_slippage
        # 포지션, 거래, 거래 로그를 저장할 리스트 초기화
        self.positions = []
        self.trades = []
        self.trade_logs = []
        # 모듈 전용 로거 객체 초기화
        self.logger = setup_logger(__name__)
        # 설정 관리 객체 초기화 (동적 파라미터, 환경설정 관리)
        self.config_manager = ConfigManager()
        # 거래 계좌 객체 생성 (초기 잔액, 수수료 적용)
        self.account = Account(initial_balance=account_size, fee_rate=fee_rate)
        # 자산 배분 관리자 초기화
        self.asset_manager = AssetManager(self.account)
        # 앙상블 전략 관리자 초기화 (여러 전략 결합 시 활용)
        self.ensemble_manager = Ensemble()
        # 리스크 관리 객체 초기화
        self.risk_manager = RiskManager()
        # 마지막 신호 발생 시각 초기화
        self.last_signal_time = None
        # 상승 진입 이벤트 로그 저장 리스트
        self.bullish_entry_events = []
        # HMM(은닉 마르코프 모델) 관련 변수 초기화
        self.hmm_model = None
        self.last_hmm_training_datetime = None
        # 추가 데이터프레임 변수: 추가 데이터, 주간 데이터 등
        self.df_extra = None
        self.df_weekly = None
        # 리밸런싱 관련 시각 변수 초기화
        self.last_rebalance_time = None
        self.last_weekly_close_date = None
        # 현재 시간을 반환하는 함수 (pandas Timestamp)
        self.clock = lambda: pd.Timestamp.now()
        # 임의 상태 정보를 저장할 딕셔너리
        self.state = {}

    def get_current_time(self):
        """
        현재 시각 반환 함수
        --------------------
        현재 백테스트에서 사용 중인 시간(타임스탬프)을 반환합니다.
        
        Returns:
          pd.Timestamp: 현재 시각
        """
        return self.clock()

    def load_data(self, short_table_format, long_table_format, short_tf, long_tf, 
                  start_date=None, end_date=None, extra_tf=None, use_weekly=False):
        """
        데이터 로딩 함수
        ----------------
        지정한 포맷과 타임프레임에 따라 백테스트용 데이터 로딩을 수행합니다.
        
        Parameters:
          short_table_format (str): 짧은 데이터 테이블 포맷
          long_table_format (str): 긴 데이터 테이블 포맷
          short_tf (str): 짧은 타임프레임 (예: '1m', '5m')
          long_tf (str): 긴 타임프레임 (예: '1h', '1d')
          start_date (str, optional): 시작 날짜 (YYYY-MM-DD 등)
          end_date (str, optional): 종료 날짜
          extra_tf (str, optional): 추가 타임프레임 데이터
          use_weekly (bool): 주간 데이터 사용 여부
        
        Returns:
          None
        """
        # 데이터 로딩 모듈을 임포트하고 호출
        from backtesting.steps.data_loader import load_data
        load_data(self, short_table_format, long_table_format, short_tf, long_tf, start_date, end_date, extra_tf, use_weekly)

    def apply_indicators(self):
        """
        인디케이터 적용 함수
        ---------------------
        데이터프레임에 기술적 인디케이터(예: SMA, RSI 등)를 적용합니다.
        
        Returns:
          None
          
        Raises:
          Exception: 인디케이터 적용 중 오류 발생 시 예외 전달
        """
        from backtesting.steps.indicator_applier import apply_indicators
        try:
            apply_indicators(self)
        except Exception as e:
            # 인디케이터 적용 중 오류 발생 시 로그 기록 및 예외 재발생
            self.logger.error("Error applying indicators: " + str(e), exc_info=True)
            raise

    def update_hmm_regime(self, dynamic_params):
        """
        HMM 시장 상태 업데이트 함수
        -----------------------------
        HMM(은닉 마르코프 모델)을 사용해 시장 상태(예: bullish, bearish, sideways)를 예측 및 업데이트합니다.
        또한 장기 SMA와 현재 가격을 비교해 상태를 조정합니다.
        
        Parameters:
          dynamic_params (dict): HMM 관련 재학습 주기, 샘플 개수, 피처 변화 임계치 등 동적 파라미터
        
        Returns:
          pd.Series: 시장 상태 레이블 시리즈 (인덱스: 날짜, 값: 상태 문자열)
        """
        try:
            # HMM 학습에 사용할 피처 목록 정의
            hmm_features = ['returns', 'volatility', 'sma', 'rsi', 'macd_macd', 'macd_signal', 'macd_diff']
            # 데이터프레임의 마지막 인덱스를 현재 시각으로 사용
            current_dt = self.df_long.index.max()
            # 재학습 간격 및 샘플 수 관련 파라미터 추출
            retrain_interval_minutes = dynamic_params.get('hmm_retrain_interval_minutes', 60)
            retrain_interval = pd.Timedelta(minutes=retrain_interval_minutes)
            max_samples = dynamic_params.get('max_hmm_train_samples', 1000)
            min_samples = dynamic_params.get('min_hmm_train_samples', 50)
            feature_change_threshold = dynamic_params.get('hmm_feature_change_threshold', 0.01)

            # 최소 샘플 수 부족 시 에러 로그 후 'sideways' 상태 반환
            if len(self.df_long) < min_samples:
                self.logger.error(f"Not enough samples for HMM training (min required: {min_samples}).", exc_info=True)
                return pd.Series(["sideways"] * len(self.df_long), index=self.df_long.index)

            # HMM 모델이 없거나 최초 학습인 경우 초기화 및 학습 수행
            if self.hmm_model is None or self.last_hmm_training_datetime is None:
                from market_analysis.regime_model import MarketRegimeHMM
                self.hmm_model = MarketRegimeHMM(n_components=3, retrain_interval_minutes=retrain_interval_minutes)
                training_data = self.df_long if len(self.df_long) <= max_samples else self.df_long.tail(max_samples)
                self.hmm_model.train(training_data, feature_columns=hmm_features)
                self.last_hmm_training_datetime = current_dt
                log_util.log_event("HMM updated", state_key="hmm_update")
            else:
                # 기존 HMM 모델이 존재하는 경우 현재 데이터의 피처 평균과 이전 통계 비교
                training_data = self.df_long if len(self.df_long) <= max_samples else self.df_long.tail(max_samples)
                current_means = training_data[hmm_features].mean()
                diff = (abs(current_means - self.hmm_model.last_feature_stats).mean() 
                        if self.hmm_model.last_feature_stats is not None else float('inf'))
                # 재학습 주기 경과 또는 피처 변화가 임계치를 초과하면 재학습 수행
                if (current_dt - self.last_hmm_training_datetime) >= retrain_interval or diff >= feature_change_threshold:
                    self.hmm_model.train(training_data, feature_columns=hmm_features)
                    self.last_hmm_training_datetime = current_dt
                    self.logger.debug(f"HMM 재학습 완료 (피처 변화 diff: {diff:.6f})")
                else:
                    self.logger.debug(f"HMM 재학습 스킵 (피처 변화 diff: {diff:.6f})")

            # HMM 모델을 이용해 시장 상태 예측
            predicted_regimes = self.hmm_model.predict_regime_labels(self.df_long, feature_columns=hmm_features)
            # 장기 SMA 계산 (예: 200일 이동평균)
            sma_period = dynamic_params.get('sma_period', 200)
            self.df_long['long_term_sma'] = self.df_long['close'].rolling(window=sma_period, min_periods=1).mean()
            adjusted_regimes = []
            # 각 시점별로 현재 가격과 SMA 비교 후 상태 조정
            for idx, predicted in enumerate(predicted_regimes):
                close_price = self.df_long['close'].iloc[idx]
                sma_price = self.df_long['long_term_sma'].iloc[idx]
                threshold = 0.01 * sma_price
                if close_price < sma_price - threshold:
                    adjusted_regime = "bearish"
                elif close_price > sma_price + threshold:
                    adjusted_regime = "bullish"
                else:
                    adjusted_regime = predicted
                adjusted_regimes.append(adjusted_regime)
            return pd.Series(adjusted_regimes, index=self.df_long.index)
        except Exception as e:
            self.logger.error("Error updating HMM regime: " + str(e), exc_info=True)
            return pd.Series(["sideways"] * len(self.df_long), index=self.df_long.index)

    def update_short_dataframe(self, regime_series, dynamic_params):
        """
        짧은 타임프레임 데이터프레임 업데이트 함수
        -------------------------------------------
        장기 데이터(df_long)의 기술적 인디케이터와 시장 상태를 짧은 타임프레임 데이터(df_short)에 병합하고,
        ATR(평균 진폭 범위) 기반의 스톱로스 가격을 계산합니다.
        
        Parameters:
          regime_series (pd.Series): 시장 상태 시리즈 (인덱스: 날짜, 값: 상태 문자열)
          dynamic_params (dict): 동적 파라미터 (ATR 기간 등 포함)
        
        Returns:
          None
        """
        try:
            # df_long에서 sma, rsi, volatility 컬럼을 df_short에 결합 (좌측 조인 후 결측값 전파)
            self.df_short = self.df_short.join(self.df_long[['sma', 'rsi', 'volatility']], how='left').ffill()
            # 시장 상태 정보를 df_short에 추가 (인덱스 기준 재배열 및 전파)
            self.df_short['market_regime'] = regime_series.reindex(self.df_short.index).ffill()
            # ATR 계산 함수 호출: 주어진 기간 내 변동성을 반영해 ATR 산출
            self.df_short = TradeExecutor.compute_atr(self.df_short, period=dynamic_params.get("atr_period", 14))
            # 스톱로스 가격 계산: 현재 가격에서 ATR에 배수를 곱한 값 차감
            default_atr_multiplier = dynamic_params.get("default_atr_multiplier", 2.0)
            self.df_short["stop_loss_price"] = self.df_short["close"] - (self.df_short["atr"] * default_atr_multiplier)
        except Exception as e:
            self.logger.error("Error updating short dataframe: " + str(e), exc_info=True)
            raise

    def handle_walk_forward_window(self, current_time, row):
        """
        워크포워드 윈도우 처리 함수
        ----------------------------
        워크포워드 기간 종료 시 모든 미체결 포지션에 대해 청산 처리를 수행합니다.
        
        Parameters:
          current_time (pd.Timestamp): 현재 시각
          row (pd.Series): 현재 데이터 행 (예: 가격 정보 포함)
        
        Returns:
          None
        """
        try:
            # 보유 포지션 각각에 대해 실행 기록을 확인하며 청산 처리
            for pos in self.positions:
                for exec_record in pos.executions:
                    # 아직 청산되지 않은 실행 기록에 대해 처리
                    if not exec_record.get("closed", False):
                        final_close = row["close"]
                        # 최종 슬리피지 적용 (없으면 그대로 사용)
                        adjusted_final_close = final_close * (1 - self.final_exit_slippage) if self.final_exit_slippage else final_close
                        # 슬리피지 비율 적용하여 최종 청산 가격 계산
                        exit_price = adjusted_final_close * (1 - self.slippage_rate)
                        size_value = exec_record["size"]
                        if isinstance(size_value, list):
                            size_value = sum(size_value)
                        # 거래 수수료 계산
                        fee = exit_price * size_value * self.fee_rate
                        # 손익 계산: (청산 가격 - 진입 가격) * 사이즈 - 수수료
                        pnl = (exit_price - exec_record["entry_price"]) * size_value - fee
                        exec_record["closed"] = True
                        # 거래 상세 내역 구성
                        trade_detail = {
                            "entry_time": exec_record["entry_time"],
                            "entry_price": exec_record["entry_price"],
                            "exit_time": current_time,
                            "exit_price": exit_price,
                            "size": size_value,
                            "pnl": pnl,
                            "reason": "walk_forward_window_close",
                            "trade_type": exec_record.get("trade_type", "unknown"),
                            "position_id": pos.position_id
                        }
                        # 로그 및 거래 내역 업데이트
                        self.trade_logs.append(trade_detail)
                        self.trades.append(trade_detail)
                        self.account.update_after_trade(trade_detail)
            # 모든 포지션을 청산 후 초기화
            self.positions = []
        except Exception as e:
            self.logger.error("Error during walk-forward window handling: " + str(e), exc_info=True)
            raise

    def handle_weekly_end(self, current_time, row):
        """
        주간 종료 처리 함수
        --------------------
        주간 종료 시 모든 미체결 포지션에 대해 청산 처리를 수행하여 주간 거래를 마감합니다.
        
        Parameters:
          current_time (pd.Timestamp): 현재 시각
          row (pd.Series): 현재 데이터 행 (종가 등 포함)
        
        Returns:
          None
        """
        try:
            final_close = row["close"]
            adjusted_final_close = final_close * (1 - self.final_exit_slippage) if self.final_exit_slippage else final_close
            for pos in self.positions:
                for exec_record in pos.executions:
                    if not exec_record.get("closed", False):
                        exit_price = adjusted_final_close * (1 - self.slippage_rate)
                        size_value = exec_record["size"]
                        if isinstance(size_value, list):
                            size_value = sum(size_value)
                        fee = exit_price * size_value * self.fee_rate
                        pnl = (exit_price - exec_record["entry_price"]) * size_value - fee
                        exec_record["closed"] = True
                        trade_detail = {
                            "entry_time": exec_record["entry_time"],
                            "entry_price": exec_record["entry_price"],
                            "exit_time": current_time,
                            "exit_price": exit_price,
                            "size": size_value,
                            "pnl": pnl,
                            "reason": "weekly_end_close",
                            "trade_type": exec_record.get("trade_type", "unknown"),
                            "position_id": pos.position_id
                        }
                        self.trade_logs.append(trade_detail)
                        self.trades.append(trade_detail)
                        self.account.update_after_trade(trade_detail)
            self.positions = []
        except Exception as e:
            self.logger.error("Error during weekly end handling: " + str(e), exc_info=True)
            raise

    def process_bullish_entry(self, current_time, row, risk_params, dynamic_params):
        """
        강세 진입(롱) 신호 처리 함수
        ----------------------------
        주어진 조건 하에서 강세 신호가 발생하면 기존 포지션에 추가 진입(scale-in)하거나 신규 포지션을 생성합니다.
        
        Parameters:
          current_time (pd.Timestamp): 현재 시각
          row (pd.Series): 현재 데이터 행 (가격, 스톱로스 등 정보 포함)
          risk_params (dict): 리스크 관리 관련 파라미터 (예: 리스크 비율 등)
          dynamic_params (dict): 동적 파라미터 (예: 신호 쿨다운, 분할 매수 기준 등)
        
        Returns:
          None
        """
        try:
            close_price = row["close"]
            # 신호 재발생 방지를 위해 일정 시간 동안 추가 신호를 무시
            signal_cooldown = pd.Timedelta(minutes=dynamic_params.get("signal_cooldown_minutes", 5))
            if self.last_signal_time is not None and (current_time - self.last_signal_time) < signal_cooldown:
                return

            # 스톱로스 가격이 없으면 기본값(현재 가격의 95%) 사용
            stop_loss_price = row.get("stop_loss_price")
            if stop_loss_price is None:
                stop_loss_price = close_price * 0.95
                self.logger.error(
                    f"Missing stop_loss_price for bullish entry. 기본값 사용: {stop_loss_price:.2f}.", exc_info=True
                )

            # 이미 롱 포지션이 존재하는 경우 추가 진입(scale in) 시도
            for pos in self.positions:
                if pos.side == "LONG":
                    additional_size = self.risk_manager.compute_position_size(
                        available_balance=self.account.get_available_balance(),
                        risk_percentage=risk_params.get("risk_per_trade"),
                        entry_price=close_price,
                        stop_loss=stop_loss_price,
                        fee_rate=self.fee_rate,
                        volatility=row.get("volatility", 0)
                    )
                    required_amount = close_price * additional_size * (1 + self.fee_rate)
                    if self.account.get_available_balance() >= required_amount:
                        threshold = dynamic_params.get("scale_in_threshold", 0.02)
                        # 가격이 낮은 종목의 경우 threshold 절반 적용
                        effective_threshold = threshold * (0.5 if close_price < 10 else 1)
                        self.risk_manager.attempt_scale_in_position(
                            position=pos,
                            current_price=close_price,
                            scale_in_threshold=effective_threshold,
                            slippage_rate=self.slippage_rate,
                            stop_loss=stop_loss_price,
                            take_profit=row.get("take_profit_price"),
                            entry_time=current_time,
                            trade_type="scale_in"
                        )
                        self.last_signal_time = current_time
                        return
            # 신규 포지션 진입 계산
            total_size = self.risk_manager.compute_position_size(
                available_balance=self.account.get_available_balance(),
                risk_percentage=risk_params.get("risk_per_trade"),
                entry_price=close_price,
                stop_loss=stop_loss_price,
                fee_rate=self.fee_rate,
                volatility=row.get("volatility", 0)
            )
            required_amount = close_price * total_size * (1 + self.fee_rate)
            if self.account.get_available_balance() >= required_amount:
                # 신규 포지션 생성 (롱)
                new_position = Position(
                    side="LONG",
                    initial_price=close_price,
                    maximum_size=total_size,
                    total_splits=dynamic_params.get("total_splits", 3),
                    allocation_plan=self.risk_manager.allocate_position_splits(
                        total_size=1.0,
                        splits_count=dynamic_params.get("total_splits", 3),
                        allocation_mode=dynamic_params.get("allocation_mode", "equal")
                    )
                )
                try:
                    # 단기 데이터에서 ATR 값을 가져옴 (없으면 0)
                    atr_value = self.df_short.loc[current_time, "atr"]
                except KeyError:
                    atr_value = 0
                # 동적 스톱로스 및 테이크 프로핏 가격 계산
                stop_loss_price_new, take_profit_price = TradeExecutor.calculate_dynamic_stop_and_take(
                    entry_price=close_price,
                    atr=atr_value,
                    risk_params=risk_params
                )
                new_position.add_execution(
                    entry_price=close_price * (1 + self.slippage_rate),
                    size=total_size * new_position.allocation_plan[0],
                    stop_loss=stop_loss_price_new,
                    take_profit=take_profit_price,
                    entry_time=current_time,
                    exit_targets=TradeExecutor.calculate_partial_exit_targets(
                        entry_price=close_price,
                        partial_exit_ratio=dynamic_params.get("partial_exit_ratio", 0.5),
                        partial_profit_ratio=dynamic_params.get("partial_profit_ratio", 0.03),
                        final_profit_ratio=dynamic_params.get("final_profit_ratio", 0.06)
                    ),
                    trade_type="new_entry"
                )
                new_position.executed_splits = 1
                # 신규 포지션을 보유 포지션 및 계좌에 추가
                self.positions.append(new_position)
                self.account.add_position(new_position)
                self.last_signal_time = current_time
        except Exception as e:
            self.logger.error("Error processing bullish entry: " + str(e), exc_info=True)
            raise

    def process_bearish_exit(self, current_time, row):
        """
        약세 청산 신호 처리 함수
        --------------------------
        모든 미체결 롱 포지션에 대해 약세 신호가 발생하면 청산을 진행합니다.
        
        Parameters:
          current_time (pd.Timestamp): 현재 시각
          row (pd.Series): 현재 데이터 행 (가격 정보 포함)
        
        Returns:
          None
        """
        try:
            close_price = row["close"]
            for pos in self.positions:
                for exec_record in pos.executions:
                    if not exec_record.get("closed", False):
                        # 슬리피지 적용하여 청산 가격 산출
                        exit_price = close_price * (1 - self.slippage_rate)
                        size_value = exec_record["size"]
                        if isinstance(size_value, list):
                            size_value = sum(size_value)
                        fee = exit_price * size_value * self.fee_rate
                        pnl = (exit_price - exec_record["entry_price"]) * size_value - fee
                        exec_record["closed"] = True
                        trade_detail = {
                            "entry_time": exec_record["entry_time"],
                            "entry_price": exec_record["entry_price"],
                            "exit_time": current_time,
                            "exit_price": exit_price,
                            "size": size_value,
                            "pnl": pnl,
                            "reason": "exit_regime_change",
                            "trade_type": exec_record.get("trade_type", "unknown"),
                            "position_id": pos.position_id
                        }
                        self.trade_logs.append(trade_detail)
                        self.trades.append(trade_detail)
                        self.account.update_after_trade(trade_detail)
            # 약세 청산 후 마지막 신호 시각 업데이트
            self.last_signal_time = current_time
        except Exception as e:
            self.logger.error("Error processing bearish exit: " + str(e), exc_info=True)
            raise

    def process_sideways_trade(self, current_time, row, risk_params, dynamic_params):
        """
        횡보장(사이드웨이) 거래 처리 함수
        ----------------------------------
        횡보장일 경우 유동성 정보에 따라 조건에 맞으면 강세 진입 또는 약세 청산을 호출합니다.
        
        Parameters:
          current_time (pd.Timestamp): 현재 시각
          row (pd.Series): 현재 데이터 행 (가격 등 포함)
          risk_params (dict): 리스크 관리 관련 파라미터
          dynamic_params (dict): 동적 파라미터 (유동성 정보 등)
        
        Returns:
          None
        """
        try:
            close_price = row["close"]
            # 동적 파라미터에서 유동성 정보 확인 (기본값 'high')
            liquidity = dynamic_params.get('liquidity_info', 'high').lower()
            if liquidity == "high":
                # 고유동성 시장: 최근 20개 행의 최저가 기준 비교
                lower_bound = self.df_short['low'].rolling(window=20, min_periods=1).min().iloc[-1]
                if close_price <= lower_bound:
                    self.process_bullish_entry(current_time, row, risk_params, dynamic_params)
            else:
                # 저유동성 시장: 평균 및 표준편차 기반 비교
                mean_price = self.df_short['close'].rolling(window=20, min_periods=1).mean().iloc[-1]
                std_price = self.df_short['close'].rolling(window=20, min_periods=1).std().iloc[-1]
                if close_price < mean_price - std_price:
                    self.process_bullish_entry(current_time, row, risk_params, dynamic_params)
                elif close_price > mean_price + std_price:
                    self.process_bearish_exit(current_time, row)
        except Exception as e:
            self.logger.error("Error processing sideways trade: " + str(e), exc_info=True)
            raise

    def update_positions(self, current_time, row):
        """
        포지션 업데이트 함수
        ---------------------
        보유 포지션의 최고 가격을 갱신하고, 이에 따라 트레일링 스톱을 조정합니다.
        
        Parameters:
          current_time (pd.Timestamp): 현재 시각
          row (pd.Series): 현재 데이터 행 (종가 포함)
        
        Returns:
          None
        """
        try:
            close_price = row["close"]
            for pos in self.positions:
                for exec_record in pos.executions:
                    if not exec_record.get("closed", False):
                        # 현재 가격과 기존 최고 가격 비교 후 갱신
                        pos.highest_price = max(pos.highest_price, close_price)
                        new_stop = TradeExecutor.adjust_trailing_stop(
                            current_stop=row.get("stop_loss_price", 0),
                            current_price=close_price,
                            highest_price=pos.highest_price,
                            trailing_percentage=self.config_manager.get_defaults().get("trailing_percent", 0.045)
                        )
                        exec_record["stop_loss"] = new_stop
        except Exception as e:
            self.logger.error("Error updating positions: " + str(e), exc_info=True)
            raise

    def finalize_all_positions(self):
        """
        모든 포지션 최종 청산 함수
        ---------------------------
        백테스트 종료 시점에서 남은 미체결 포지션을 모두 청산하고 거래 내역을 업데이트합니다.
        
        Returns:
          None
        """
        try:
            final_time = self.df_short.index[-1]
            final_close = self.df_short.iloc[-1]["close"]
            adjusted_final_close = final_close * (1 - self.final_exit_slippage) if self.final_exit_slippage else final_close
            for pos in self.positions:
                for exec_record in pos.executions:
                    if not exec_record.get("closed", False):
                        exit_price = adjusted_final_close * (1 - self.slippage_rate)
                        size_value = exec_record["size"]
                        if isinstance(size_value, list):
                            size_value = sum(size_value)
                        fee = exit_price * size_value * self.fee_rate
                        pnl = (exit_price - exec_record["entry_price"]) * size_value - fee
                        exec_record["closed"] = True
                        trade_detail = {
                            "entry_time": exec_record["entry_time"],
                            "entry_price": exec_record["entry_price"],
                            "exit_time": final_time,
                            "exit_price": exit_price,
                            "size": size_value,
                            "pnl": pnl,
                            "reason": "final_exit",
                            "trade_type": exec_record.get("trade_type", "unknown"),
                            "position_id": pos.position_id
                        }
                        self.trade_logs.append(trade_detail)
                        self.trades.append(trade_detail)
                        self.account.update_after_trade(trade_detail)
            self.positions = []
        except Exception as e:
            self.logger.error("Error finalizing positions: " + str(e), exc_info=True)
            raise

    def monitor_orders(self, current_time, row):
        """
        주문 모니터링 함수
        ------------------
        포지션별로 미체결 주문이 존재하는 경우, 가격 급변동 등을 모니터링하여 로그를 기록합니다.
        
        Parameters:
          current_time (pd.Timestamp): 현재 시각
          row (pd.Series): 현재 데이터 행 (가격 정보 포함)
        
        Returns:
          None
        """
        try:
            for pos in self.positions:
                for exec_record in pos.executions:
                    if not exec_record.get("closed", False):
                        self.logger.debug(f"Significant price move for position {pos.position_id}.")
        except Exception as e:
            self.logger.error("Error monitoring orders: " + str(e), exc_info=True)
            raise

    def run_backtest(self, dynamic_params=None, walk_forward_days: int = None, holdout_period: tuple = None):
        """
        백테스트 실행 함수
        ------------------
        백테스트 파이프라인 전체를 실행하며, 데이터 준비, 인디케이터 적용, HMM 업데이트, 주문 처리 등을 순차적으로 진행합니다.
        
        Parameters:
          dynamic_params (dict, optional): 동적 파라미터 (없으면 기본값 사용)
          walk_forward_days (int, optional): 워크포워드 기간 (일 단위)
          holdout_period (tuple, optional): (시작, 종료) 기간을 포함하는 holdout 기간
        
        Returns:
          tuple: (trades, trade_logs) - 거래 내역 및 로그 리스트
        
        Raises:
          Exception: 백테스트 실행 중 발생한 예외 전달
        """
        from logging.logging_util import LoggingUtil
        log_util = LoggingUtil(__name__)
        try:
            # 동적 파라미터 설정: 인자가 없으면 ConfigManager에서 기본값 로드
            if dynamic_params is None:
                dynamic_params = self.config_manager.get_dynamic_params()
            else:
                dynamic_params = self.config_manager.validate_params(dynamic_params)
            try:
                # 수익률 및 변동성 계산 (종가의 변화율 및 이동 표준편차)
                self.df_long['returns'] = self.df_long['close'].pct_change().fillna(0)
                self.df_long['volatility'] = self.df_long['returns'].rolling(window=20).std().fillna(0)
            except Exception as e:
                self.logger.error("Error computing returns/volatility: " + str(e), exc_info=True)
                raise

            try:
                # 기술적 인디케이터 적용 (예: SMA, RSI 등)
                self.apply_indicators()
                log_util.log_event("Indicators applied", state_key="indicator_applied")
            except Exception as e:
                self.logger.error("Error during indicator application: " + str(e), exc_info=True)
                raise

            try:
                # HMM 업데이트 모듈 호출: 시장 상태 예측 수행
                from backtesting.steps.hmm_manager import update_hmm
                regime_series = update_hmm(self, dynamic_params)
                log_util.log_event("HMM updated", state_key="hmm_update")
            except Exception as e:
                self.logger.error("Error updating HMM: " + str(e), exc_info=True)
                regime_series = pd.Series(["sideways"] * len(self.df_long), index=self.df_long.index)

            try:
                # 짧은 타임프레임 데이터 업데이트: 인디케이터 및 시장 상태 적용
                self.update_short_dataframe(regime_series, dynamic_params)
                log_util.log_event("Data loaded successfully", state_key="data_load")
            except Exception as e:
                self.logger.error("Error updating short dataframe: " + str(e), exc_info=True)
                raise

            # Holdout 기간이 지정된 경우, 학습용 데이터와 holdout 데이터 분리
            if holdout_period:
                holdout_start, holdout_end = pd.to_datetime(holdout_period[0]), pd.to_datetime(holdout_period[1])
                df_train = self.df_short[self.df_short.index < holdout_start]
                df_holdout = self.df_short[(self.df_short.index >= holdout_start) & (self.df_short.index <= holdout_end)]
            else:
                df_train = self.df_short
                df_holdout = None

            # 워크포워드 기간이 지정된 경우, 초기 시간과 기간 설정
            if walk_forward_days is not None:
                self.window_start = df_train.index[0]
                self.walk_forward_td = pd.Timedelta(days=walk_forward_days)
                self.walk_forward_days = walk_forward_days
            else:
                self.window_start = None
                self.walk_forward_days = None

            # 신호 쿨다운 및 리밸런싱 주기 설정
            signal_cooldown = pd.Timedelta(minutes=dynamic_params.get("signal_cooldown_minutes", 5))
            rebalance_interval = pd.Timedelta(minutes=dynamic_params.get("rebalance_interval_minutes", 60))

            self.df_train = df_train
            try:
                # 주문 처리 관련 모듈 호출: 학습, 추가, holdout, 최종 주문 처리
                from backtesting.steps.order_manager import process_training_orders, process_extra_orders, process_holdout_orders, finalize_orders
                process_training_orders(self, dynamic_params, signal_cooldown, rebalance_interval)
                process_extra_orders(self, dynamic_params)
                process_holdout_orders(self, dynamic_params, df_holdout)
                finalize_orders(self)
            except Exception as e:
                self.logger.error("Error during order processing: " + str(e), exc_info=True)
                raise

            # 최종 성과 계산 (계좌 잔액, 총 수익률 등)
            available_balance = self.account.get_available_balance()
            total_pnl = sum(trade.get("pnl", 0) if not isinstance(trade.get("pnl", 0), list) 
                            else sum(trade.get("pnl", 0)) for trade in self.trades)
            roi = total_pnl / self.account.initial_balance * 100
            log_util.log_event("Backtest complete", state_key="final_performance")
            return self.trades, self.trade_logs

        except Exception as e:
            self.logger.error("Fatal error in run_backtest: " + str(e), exc_info=True)
            raise

    def run_backtest_pipeline(self, dynamic_params=None, walk_forward_days: int = None, holdout_period: tuple = None):
        """
        백테스트 파이프라인 실행 함수
        -----------------------------
        백테스트를 실행한 후, 최종 계좌 잔액 및 포지션 청산 여부를 로깅하고 결과를 반환합니다.
        
        Parameters:
          dynamic_params (dict, optional): 동적 파라미터
          walk_forward_days (int, optional): 워크포워드 기간 (일 단위)
          holdout_period (tuple, optional): holdout 기간 (시작, 종료)
        
        Returns:
          tuple: (trades, trade_logs) - 전체 거래 내역 및 상세 로그 리스트
        """
        trades, trade_logs = self.run_backtest(dynamic_params, walk_forward_days, holdout_period)
        available_balance = self.account.get_available_balance()
        self.logger.debug(f"최종 계좌 잔액: {available_balance:.2f}")
        if self.positions:
            self.logger.error("백테스트 종료 후에도 미체결 포지션이 남아있습니다.", exc_info=True)
        else:
            self.logger.debug("모든 포지션이 정상적으로 종료되었습니다.")
        return trades, trade_logs
