# scripts/run_backtest.py
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
from parameters.config_manager import ConfigManager
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

# 모듈별 로깅 설정 및 데이터 관련 모듈 임포트
from logging.logger_config import setup_logger                   # 로깅 설정 함수
from data.db.db_manager import fetch_ohlcv_records              # OHLCV 데이터베이스 조회 함수
from data.ohlcv.ohlcv_aggregator import aggregate_to_weekly       # 주간 데이터 집계 함수
from signal_calculation.indicators import compute_bollinger_bands          # 볼린저 밴드 계산 함수
import threading                                                # 멀티스레드 동기화를 위한 모듈
import pandas as pd                                             # 데이터 프레임 처리를 위한 pandas
from logging.logging_util import LoggingUtil                       # 추가 로깅 유틸리티

# --- 전역 변수 및 객체 정의 ---
# 모듈 단위 로깅 인스턴스 설정: 로그를 남길 때 모듈명(__name__)을 이용
logger = setup_logger(__name__)
log_util = LoggingUtil(__name__)

# 멀티스레드 환경에서의 데이터 캐싱을 위한 전역 변수
_cache_lock = threading.Lock()    # 캐시 접근시 동기화를 위한 Lock 객체
_data_cache = {}                  # OHLCV 데이터 프레임을 메모리에 캐싱하기 위한 딕셔너리

# --- 내부 캐시 함수 정의 ---
def _get_cached_ohlcv(table_name, start_date, end_date):
    """
    캐시에서 특정 테이블과 날짜 범위에 해당하는 OHLCV 데이터를 조회합니다.
    
    Parameters:
        table_name (str): OHLCV 데이터가 저장된 테이블의 이름.
        start_date (str 또는 datetime): 데이터 조회의 시작 날짜.
        end_date (str 또는 datetime): 데이터 조회의 종료 날짜.
    
    Returns:
        pandas.DataFrame 또는 None: 캐시에 존재하는 경우 해당 DataFrame, 없으면 None.
    """
    # 캐시 키는 테이블 이름과 날짜 범위를 튜플로 결합하여 생성합니다.
    key = (table_name, start_date, end_date)
    # 멀티스레드 환경에서 동시 접근을 방지하기 위해 락을 사용합니다.
    with _cache_lock:
        return _data_cache.get(key)

def _set_cached_ohlcv(table_name, start_date, end_date, df):
    """
    주어진 OHLCV 데이터 프레임을 캐시에 저장합니다.
    
    Parameters:
        table_name (str): 데이터가 속한 테이블 이름.
        start_date (str 또는 datetime): 데이터 조회의 시작 날짜.
        end_date (str 또는 datetime): 데이터 조회의 종료 날짜.
        df (pandas.DataFrame): 저장할 OHLCV 데이터 프레임.
    
    Returns:
        None
    """
    # 캐시 키 생성
    key = (table_name, start_date, end_date)
    # 동기화를 위해 락 사용 후 캐시에 저장
    with _cache_lock:
        _data_cache[key] = df

def _validate_and_prepare_df(df, table_name):
    """
    불러온 OHLCV 데이터 프레임의 유효성을 검사하고, 필요한 전처리(시간 인덱스 변환, 정렬, 중복 제거 등)를 수행합니다.
    
    Parameters:
        df (pandas.DataFrame): 검증 및 전처리할 데이터 프레임.
        table_name (str): 데이터 프레임이 속한 테이블 이름(로그 메시지 용도).
    
    Returns:
        pandas.DataFrame: 전처리가 완료된 데이터 프레임.
    """
    # 데이터 프레임이 비어있는지 확인 후 에러 로그 출력
    if df.empty:
        logger.error(f"DataFrame for {table_name} is empty.", exc_info=True)
        return df

    # 인덱스가 datetime 형식인지 확인하고, 아니라면 변환 시도
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        try:
            df.index = pd.to_datetime(df.index, errors='coerce')
            # 변환 후 NaT (Not a Time)가 포함된 경우 해당 행 제거
            if df.index.isnull().any():
                logger.warning(f"Some index values in {table_name} could not be converted to datetime and will be dropped.")
                df = df[~df.index.isnull()]
        except Exception as e:
            logger.error(f"Error converting index to datetime for {table_name}: {e}", exc_info=True)
            raise

    # 인덱스를 오름차순으로 정렬 (시간 순 정렬)
    df.sort_index(inplace=True)

    # 중복된 인덱스가 존재하면 경고 로그 출력 후 중복 제거
    if df.index.duplicated().any():
        logger.warning(f"Duplicate datetime indices found in {table_name}; removing duplicates.")
        df = df[~df.index.duplicated(keep='first')]

    # 데이터 열이 존재할 경우, 고가(high), 저가(low), 종가(close)를 활용하여 평균 변동폭 계산
    if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
        avg_range = (df['high'] - df['low']).mean()
        avg_close = df['close'].mean()
        # 변동성이 비정상적으로 낮은 경우 경고 로그 출력 (정상 데이터 여부 점검)
        if avg_range < avg_close * 0.001:
            logger.warning(f"Data for {table_name} shows low volatility: avg_range={avg_range:.6f}, avg_close={avg_close:.6f}.")
    return df

def load_data(backtester, short_table_format, long_table_format, short_tf, long_tf, 
              start_date=None, end_date=None, extra_tf=None, use_weekly=False):
    """
    백테스터(backtester) 객체에 필요한 OHLCV 데이터와 인디케이터 데이터를 로드 및 전처리합니다.
    
    주요 기능:
      - 단기(short) 및 장기(long) 데이터 테이블 이름 생성 후 캐시에서 검색
      - 캐시에 없으면 데이터베이스에서 조회하고 캐시에 저장
      - 데이터 프레임의 유효성을 검사하고 전처리 수행
      - 추가 시간 프레임(extra_tf)이 제공되면 볼린저 밴드를 계산하여 추가 데이터 구성
      - use_weekly가 True면 단기 데이터를 주간 데이터로 집계하여 저장
    
    Parameters:
        backtester (object): 백테스트 실행 객체로, 로드된 데이터를 저장할 속성을 포함.
        short_table_format (str): 단기 데이터 테이블 이름 형식 (문자열 포맷).
        long_table_format (str): 장기 데이터 테이블 이름 형식 (문자열 포맷).
        short_tf (str): 단기 데이터의 시간 프레임 (예: '1m', '5m').
        long_tf (str): 장기 데이터의 시간 프레임 (예: '1h', '1d').
        start_date (str 또는 datetime, optional): 데이터 조회 시작 날짜.
        end_date (str 또는 datetime, optional): 데이터 조회 종료 날짜.
        extra_tf (str, optional): 추가 데이터 시간 프레임 (예: '15m'); 기본값은 None.
        use_weekly (bool, optional): 주간 데이터 집계를 사용할지 여부.
    
    Returns:
        None
    """
    try:
        # 심볼(symbol)을 포맷에 맞게 소문자 및 '/' 제거 처리
        symbol_for_table = backtester.symbol.replace('/', '').lower()
        # 단기 및 장기 테이블 이름 생성 (문자열 포맷 사용)
        short_table = short_table_format.format(symbol=symbol_for_table, timeframe=short_tf)
        long_table = long_table_format.format(symbol=symbol_for_table, timeframe=long_tf)
        
        # 단기 데이터 로드: 캐시에서 검색 후 없으면 DB에서 조회
        df_short = _get_cached_ohlcv(short_table, start_date, end_date)
        if df_short is None:
            df_short = fetch_ohlcv_records(short_table, start_date, end_date)
            _set_cached_ohlcv(short_table, start_date, end_date, df_short)
        # 데이터 프레임 유효성 검사 및 전처리 수행
        df_short = _validate_and_prepare_df(df_short, short_table)
        
        # 장기 데이터 로드: 캐시 검색 후 없으면 DB에서 조회
        df_long = _get_cached_ohlcv(long_table, start_date, end_date)
        if df_long is None:
            df_long = fetch_ohlcv_records(long_table, start_date, end_date)
            _set_cached_ohlcv(long_table, start_date, end_date, df_long)
        df_long = _validate_and_prepare_df(df_long, long_table)
        
        # 백테스터 객체에 로드된 데이터를 할당 (후속 전략 로직에서 사용)
        backtester.df_short = df_short
        backtester.df_long = df_long
        
        # 단기 또는 장기 데이터가 비어있으면 에러 로그 출력 후 예외 발생
        if backtester.df_short.empty or backtester.df_long.empty:
            logger.error("데이터 로드 실패: short 또는 long 데이터가 비어있습니다.", exc_info=True)
            raise ValueError("No data loaded")
        
        # 데이터 로드 성공 이벤트 로깅
        log_util.log_event("Data loaded successfully", state_key="data_load")
    except Exception as e:
        logger.error(f"데이터 로드 중 에러 발생: {e}", exc_info=True)
        raise

    # 추가 시간 프레임(extra_tf)이 지정된 경우 추가 데이터 로드 및 볼린저 밴드 계산 수행
    if extra_tf:
        try:
            extra_table = short_table_format.format(symbol=symbol_for_table, timeframe=extra_tf)
            df_extra = _get_cached_ohlcv(extra_table, start_date, end_date)
            if df_extra is None:
                df_extra = fetch_ohlcv_records(extra_table, start_date, end_date)
                _set_cached_ohlcv(extra_table, start_date, end_date, df_extra)
            df_extra = _validate_and_prepare_df(df_extra, extra_table)
            backtester.df_extra = df_extra
            if not backtester.df_extra.empty:
                # 볼린저 밴드를 계산하여 보조 지표 추가 (가격 열은 'close')
                backtester.df_extra = compute_bollinger_bands(
                    backtester.df_extra,
                    price_column='close',
                    period=20,
                    std_multiplier=2.0,
                    fillna=True
                )
                log_util.log_event("Extra data loaded", state_key="extra_load")
        except Exception as e:
            logger.error(f"Extra 데이터 로드 에러: {e}", exc_info=True)
    # 주간 데이터 집계 옵션이 True인 경우, 단기 데이터를 주간 단위로 집계하여 백테스터에 추가
    if use_weekly:
        try:
            backtester.df_weekly = aggregate_to_weekly(backtester.df_short, compute_indicators=True)
            if backtester.df_weekly.empty:
                logger.warning("주간 데이터 집계 결과가 비어있습니다.")
            else:
                log_util.log_event("Weekly data aggregated", state_key="weekly_load")
        except Exception as e:
            logger.error(f"주간 데이터 집계 에러: {e}", exc_info=True)

from logging.logger_config import setup_logger

# 모듈 로깅 인스턴스 설정
logger = setup_logger(__name__)

def update_hmm(backtester, dynamic_params):
    """
    백테스터 객체의 HMM(은닉 마르코프 모델) 상태를 업데이트하고, 업데이트된 regime(시장 체제)의 분포를 로그로 출력합니다.
    
    Parameters:
        backtester (object): HMM 업데이트 메서드를 가진 백테스터 객체.
        dynamic_params (dict): 동적 파라미터(예: 시장 환경, 유동성 정보 등)를 포함하는 딕셔너리.
    
    Returns:
        pandas.Series: 업데이트된 HMM regime 시리즈.
    """
    # 백테스터 내부의 HMM 업데이트 함수 호출 (예: 시장 체제 분류 업데이트)
    regime_series = backtester.update_hmm_regime(dynamic_params)
    try:
        # 각 regime 값의 빈도수를 계산하여 딕셔너리 형태로 변환 후 디버그 로그 출력
        counts = regime_series.value_counts().to_dict()
        logger.debug(f"HMM 업데이트 완료: 총 {len(regime_series)} 샘플, regime 분포: {counts}")
    except Exception:
        logger.error("HMM 업데이트 완료: regime 분포 정보 산출 실패")
    return regime_series

from logging.logger_config import setup_logger
from signal_calculation.indicators import compute_sma, compute_rsi, compute_macd

# 모듈 로깅 인스턴스 설정
logger = setup_logger(__name__)

def apply_indicators(backtester):
    """
    백테스터 객체의 장기 데이터(df_long)에 SMA, RSI, MACD 등 다양한 트레이딩 인디케이터를 적용합니다.
    
    주요 동작:
      - 단순 이동평균(SMA) 계산 후 'sma' 열에 저장
      - 상대 강도 지수(RSI) 계산 후 'rsi' 열에 저장
      - MACD 및 시그널, 차이값 계산 후 'macd_' 접두사로 열 추가
      - 적용된 인디케이터 값들의 최소/최대 범위를 로그에 출력
    
    Parameters:
        backtester (object): 인디케이터를 적용할 데이터 프레임(df_long)을 포함하는 백테스터 객체.
    
    Returns:
        None
    """
    # SMA 계산: 종가('close') 기준, 200 기간, 결측값 채움 옵션 활성화, 결과는 'sma' 열에 저장
    backtester.df_long = compute_sma(backtester.df_long, price_column='close', period=200, fillna=True, output_col='sma')
    # RSI 계산: 종가('close') 기준, 14 기간, 결측값 채움, 결과는 'rsi' 열에 저장
    backtester.df_long = compute_rsi(backtester.df_long, price_column='close', period=14, fillna=True, output_col='rsi')
    # MACD 계산: 종가('close') 기준, 느린 기간=26, 빠른 기간=12, 시그널 기간=9, 결측값 채움, 결과 열은 'macd_' 접두사를 사용
    backtester.df_long = compute_macd(backtester.df_long, price_column='close', slow_period=26, fast_period=12, signal_period=9, fillna=True, prefix='macd_')
    
    # 인디케이터가 적용된 데이터의 값 범위를 계산하여 로그에 출력 (모든 값의 최소 및 최대값)
    sma_min = backtester.df_long['sma'].min()
    sma_max = backtester.df_long['sma'].max()
    rsi_min = backtester.df_long['rsi'].min()
    rsi_max = backtester.df_long['rsi'].max()
    macd_diff_min = backtester.df_long['macd_diff'].min()
    macd_diff_max = backtester.df_long['macd_diff'].max()
    
    logger.debug(
        f"인디케이터 적용 완료: SMA 범위=({sma_min:.2f}, {sma_max:.2f}), "
        f"RSI 범위=({rsi_min:.2f}, {rsi_max:.2f}), MACD diff 범위=({macd_diff_min:.2f}, {macd_diff_max:.2f})"
    )

from logging.logger_config import setup_logger
from logging.logging_util import LoggingUtil  # 동적 상태 변화 로깅 유틸리티

# 모듈 로깅 인스턴스 및 추가 로깅 유틸리티 설정
logger = setup_logger(__name__)
log_util = LoggingUtil(__name__)

def get_signal_with_weekly_override(backtester, row, current_time, dynamic_params):
    """
    주간 데이터(weekly data)가 존재할 경우, 주간 저점/고점 근접 여부에 따라 주문 신호(enter_long 또는 exit_all)를 우선 적용합니다.
    만약 주간 데이터 조건이 충족되지 않으면, ensemble_manager를 이용해 최종 신호를 반환합니다.
    
    Parameters:
        backtester (object): 주문 신호 생성을 위한 백테스터 객체.
        row (pandas.Series): 현재 시점의 데이터 행 (OHLCV 및 기타 지표 포함).
        current_time (datetime): 현재 시점의 시간.
        dynamic_params (dict): 동적 파라미터 (예: 유동성 정보 등).
    
    Returns:
        str: 주문 신호 (예: "enter_long", "exit_all", 또는 ensemble_manager의 반환 값).
    """
    try:
        # 주간 데이터가 존재하며, 비어있지 않은 경우
        if hasattr(backtester, 'df_weekly') and backtester.df_weekly is not None and not backtester.df_weekly.empty:
            # 현재 시간보다 작거나 같은 주간 데이터 중 가장 최근 데이터(주간 바)를 선택
            weekly_bar = backtester.df_weekly.loc[backtester.df_weekly.index <= current_time].iloc[-1]
            # 주간 데이터에 'weekly_low' 및 'weekly_high' 값이 존재하는지 확인
            if "weekly_low" in weekly_bar and "weekly_high" in weekly_bar:
                tolerance = 0.002  # 주간 저점/고점에 대한 허용 오차 비율
                # 현재 종가가 주간 저점에 근접하면 'enter_long' 신호 반환
                if abs(row["close"] - weekly_bar["weekly_low"]) / weekly_bar["weekly_low"] <= tolerance:
                    log_util.log_event("Weekly override: enter_long", state_key="order_signal")
                    return "enter_long"
                # 현재 종가가 주간 고점에 근접하면 'exit_all' 신호 반환
                elif abs(row["close"] - weekly_bar["weekly_high"]) / weekly_bar["weekly_high"] <= tolerance:
                    log_util.log_event("Weekly override: exit_all", state_key="order_signal")
                    return "exit_all"
            else:
                # 주간 데이터에 필요한 키가 없으면 경고 로그 출력
                backtester.logger.warning("Weekly override skipped: weekly_bar missing 'weekly_low' or 'weekly_high' keys.")
        # 주간 override 조건이 충족되지 않으면 ensemble_manager를 통해 최종 신호 계산
        return backtester.ensemble_manager.get_final_signal(
            row.get('market_regime', 'unknown'),
            dynamic_params.get('liquidity_info', 'high'),
            backtester.df_short,
            current_time,
            data_weekly=getattr(backtester, 'df_weekly', None)
        )
    except Exception as e:
        # 오류 발생 시 에러 로그 기록 후 ensemble_manager의 최종 신호 반환
        backtester.logger.error(f"Error in weekly override signal generation: {e}", exc_info=True)
        return backtester.ensemble_manager.get_final_signal(
            row.get('market_regime', 'unknown'),
            dynamic_params.get('liquidity_info', 'high'),
            backtester.df_short,
            current_time,
            data_weekly=getattr(backtester, 'df_weekly', None)
        )

def process_training_orders(backtester, dynamic_params, signal_cooldown, rebalance_interval):
    """
    학습 데이터(df_train)를 순회하며 각 시점에 대해 주문 신호를 생성하고 주문을 실행합니다.
    또한, 주간 종료, walk-forward window, 포지션 업데이트 및 자산 리밸런싱 등을 처리합니다.
    
    Parameters:
        backtester (object): 주문 처리 로직을 포함하는 백테스터 객체.
        dynamic_params (dict): 주문 실행 시 필요한 동적 파라미터들.
        signal_cooldown (timedelta): 신호 간 최소 시간 간격.
        rebalance_interval (timedelta): 리밸런싱 간 최소 시간 간격.
    
    Returns:
        None
    """
    # 학습 데이터의 각 시간별 행을 순회하며 주문 처리 수행
    for current_time, row in backtester.df_train.iterrows():
        try:
            # 주간 종료 처리: 매주 금요일(weekday()==4)이며, 이전에 처리되지 않은 날짜이면 주간 종료 처리 실행
            try:
                if current_time.weekday() == 4 and (
                    backtester.last_weekly_close_date is None or 
                    backtester.last_weekly_close_date != current_time.date()
                ):
                    try:
                        backtester.handle_weekly_end(current_time, row)
                    except Exception as e:
                        logger.error(f"Weekly end handling error {e}", exc_info=True)
                    backtester.last_weekly_close_date = current_time.date()
                    continue  # 주간 종료 후 나머지 주문 로직 생략
            except Exception as e:
                logger.error(f"Error during weekly end check {e}", exc_info=True)
            
            # walk-forward window 처리: 정해진 기간이 경과하면 walk-forward 처리를 실행
            try:
                if backtester.walk_forward_days is not None and (current_time - backtester.window_start) >= backtester.walk_forward_td:
                    try:
                        backtester.handle_walk_forward_window(current_time, row)
                    except Exception as e:
                        logger.error(f"Walk-forward window handling error {e}", exc_info=True)
                    backtester.window_start = current_time
            except Exception as e:
                logger.error(f"Error during walk-forward window check {e}", exc_info=True)
            
            # 신호 쿨다운을 고려하여 일정 시간 간격 이후에만 신호 생성 (즉, 너무 짧은 간격은 무시)
            if backtester.last_signal_time is None or (current_time - backtester.last_signal_time) >= signal_cooldown:
                action = get_signal_with_weekly_override(backtester, row, current_time, dynamic_params)
            else:
                action = "hold"
                
            # 기본 위험 파라미터 설정 (거래당 위험, ATR 곱수, 수익 비율, 현재 변동성)
            base_risk_params = {
                "risk_per_trade": dynamic_params.get("risk_per_trade", 0.01),
                "atr_multiplier": dynamic_params.get("atr_multiplier", 2.0),
                "profit_ratio": dynamic_params.get("profit_ratio", 0.05),
                "current_volatility": row.get("volatility", 0)
            }
            risk_params = base_risk_params
            try:
                # 시장 체제 및 유동성 정보에 따른 위험 파라미터 보정
                risk_params = backtester.risk_manager.compute_risk_parameters_by_regime(
                    base_risk_params,
                    row.get('market_regime', 'unknown'),
                    dynamic_params.get('liquidity_info', 'high')
                )
            except Exception as e:
                logger.error(f"Risk parameter computation error {e}", exc_info=True)
                risk_params = base_risk_params
            try:
                # 주문 실행: 신호(action)에 따라 bullish entry, bearish exit 또는 sideways trade 처리
                if action == "enter_long":
                    backtester.process_bullish_entry(current_time, row, risk_params, dynamic_params)
                    log_util.log_event("Order executed: enter_long", state_key="order_execution")
                elif action == "exit_all":
                    backtester.process_bearish_exit(current_time, row)
                    log_util.log_event("Order executed: exit_all", state_key="order_execution")
                elif row.get('market_regime', 'unknown') == "sideways":
                    backtester.process_sideways_trade(current_time, row, risk_params, dynamic_params)
                    log_util.log_event("Order executed: sideways", state_key="order_execution")
            except Exception as e:
                logger.error(f"Error processing order with action '{action}': {e}", exc_info=True)
            # 마지막 신호 발생 시각 갱신
            backtester.last_signal_time = current_time

            # 포지션 업데이트: 각 시점에서 보유 포지션의 상태 갱신
            try:
                backtester.update_positions(current_time, row)
            except Exception as e:
                logger.error(f"Error updating positions {e}", exc_info=True)

            # 리밸런싱 처리: 정해진 간격이 경과하면 자산 리밸런싱 실행
            try:
                if backtester.last_rebalance_time is None or (current_time - backtester.last_rebalance_time) >= rebalance_interval:
                    try:
                        backtester.asset_manager.rebalance(row.get('market_regime', 'unknown'))
                    except Exception as e:
                        logger.error(f"Error during rebalance {e}", exc_info=True)
                    backtester.last_rebalance_time = current_time
                log_util.log_event("Rebalance executed", state_key="rebalance")
            except Exception as e:
                logger.error(f"Error in rebalance check {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Unexpected error during processing training orders {e}", exc_info=True)
            continue

def process_extra_orders(backtester, dynamic_params):
    """
    추가 데이터(df_extra)가 있을 경우, 각 시점에 대해 주문 신호를 생성하고 주문을 실행합니다.
    단, 시장 체제(realm)를 재조회하여 위험 파라미터를 재계산하고, 주문 모니터링도 수행합니다.
    
    Parameters:
        backtester (object): 주문 처리 로직을 포함하는 백테스터 객체.
        dynamic_params (dict): 주문 실행 시 필요한 동적 파라미터들.
    
    Returns:
        None
    """
    if backtester.df_extra is not None and not backtester.df_extra.empty:
        for current_time, row in backtester.df_extra.iterrows():
            try:
                # 주간 override 신호를 포함한 주문 신호 생성
                hf_signal = get_signal_with_weekly_override(backtester, row, current_time, dynamic_params)
                # 현재 시장 체제 정보를 가져오기 위해 장기 데이터(df_long)에서 최신 값을 조회
                regime = "sideways"
                try:
                    regime = backtester.df_long.loc[backtester.df_long.index <= current_time].iloc[-1].get('market_regime', 'sideways')
                except Exception as e:
                    logger.error(f"Retrieving regime failed {e}", exc_info=True)
                    regime = "sideways"
                # 기본 위험 파라미터 설정
                base_risk_params = {
                    "risk_per_trade": dynamic_params.get("risk_per_trade", 0.01),
                    "atr_multiplier": dynamic_params.get("atr_multiplier", 2.0),
                    "profit_ratio": dynamic_params.get("profit_ratio", 0.05),
                    "current_volatility": row.get("volatility", 0)
                }
                risk_params = base_risk_params
                try:
                    # 위험 파라미터를 시장 체제와 유동성 정보에 따라 조정
                    risk_params = backtester.risk_manager.compute_risk_parameters_by_regime(
                        base_risk_params,
                        regime,
                        dynamic_params.get('liquidity_info', 'high')
                    )
                except Exception as e:
                    logger.error(f"Risk params error (extra data) {e}", exc_info=True)
                    risk_params = base_risk_params
                try:
                    # 주문 실행: 신호에 따라 bullish entry 또는 bearish exit 처리
                    if hf_signal == "enter_long":
                        backtester.process_bullish_entry(current_time, row, risk_params, dynamic_params)
                        log_util.log_event("Extra: Order executed: enter_long", state_key="order_execution")
                    elif hf_signal == "exit_all":
                        backtester.process_bearish_exit(current_time, row)
                        log_util.log_event("Extra: Order executed: exit_all", state_key="order_execution")
                except Exception as e:
                    logger.error(f"Error processing extra order with hf_signal '{hf_signal}': {e}", exc_info=True)
                # 주문 모니터링: 주문 상태 및 포지션 관리
                try:
                    backtester.monitor_orders(current_time, row)
                except Exception as e:
                    logger.error(f"Error monitoring orders {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Unexpected error in process_extra_orders {e}", exc_info=True)
                continue

def process_holdout_orders(backtester, dynamic_params, df_holdout):
    """
    보류(holdout) 데이터(df_holdout)에 대해 각 시점마다 주문 신호를 생성하고 주문을 실행합니다.
    위험 파라미터 재계산, 포지션 업데이트 등 일반 주문 처리 로직과 유사하게 진행합니다.
    
    Parameters:
        backtester (object): 주문 처리 로직을 포함하는 백테스터 객체.
        dynamic_params (dict): 주문 실행 시 필요한 동적 파라미터들.
        df_holdout (pandas.DataFrame): 보류 데이터 (테스트 또는 검증용 데이터).
    
    Returns:
        None
    """
    if df_holdout is not None:
        for current_time, row in df_holdout.iterrows():
            try:
                # 주간 override를 고려한 주문 신호 생성
                action = get_signal_with_weekly_override(backtester, row, current_time, dynamic_params)
                # 기본 위험 파라미터 설정
                base_risk_params = {
                    "risk_per_trade": dynamic_params.get("risk_per_trade", 0.01),
                    "atr_multiplier": dynamic_params.get("atr_multiplier", 2.0),
                    "profit_ratio": dynamic_params.get("profit_ratio", 0.05),
                    "current_volatility": row.get("volatility", 0)
                }
                risk_params = base_risk_params
                try:
                    # 위험 파라미터 보정: 시장 체제 및 유동성 정보에 기반
                    risk_params = backtester.risk_manager.compute_risk_parameters_by_regime(
                        base_risk_params,
                        row.get('market_regime', 'unknown'),
                        dynamic_params.get('liquidity_info', 'high')
                    )
                except Exception as e:
                    logger.error(f"Risk params error (holdout) {e}", exc_info=True)
                    risk_params = base_risk_params
                try:
                    # 주문 실행: 신호에 따라 bullish entry, bearish exit, 또는 sideways trade 처리
                    if action == "enter_long":
                        backtester.process_bullish_entry(current_time, row, risk_params, dynamic_params)
                        log_util.log_event("Holdout: Order executed: enter_long", state_key="order_execution")
                    elif action == "exit_all":
                        backtester.process_bearish_exit(current_time, row)
                        log_util.log_event("Holdout: Order executed: exit_all", state_key="order_execution")
                    elif row.get('market_regime', 'unknown') == "sideways":
                        backtester.process_sideways_trade(current_time, row, risk_params, dynamic_params)
                        log_util.log_event("Holdout: Order executed: sideways", state_key="order_execution")
                except Exception as e:
                    logger.error(f"Error processing holdout order with action '{action}': {e}", exc_info=True)
                try:
                    # 보류 데이터에 대해 포지션 상태 업데이트 실행
                    backtester.update_positions(current_time, row)
                except Exception as e:
                    logger.error(f"Error updating positions in holdout {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Unexpected error in process_holdout_orders {e}", exc_info=True)
                continue

def finalize_orders(backtester):
    """
    백테스터 객체 내에서 모든 포지션을 마감(finalize) 처리합니다.
    
    Parameters:
        backtester (object): 최종 포지션 마감을 실행할 백테스터 객체.
    
    Returns:
        None
    """
    try:
        backtester.finalize_all_positions()
    except Exception as e:
        logger.error(f"Error finalizing orders: {e}", exc_info=True)
        raise
