# strategies/optimizer.py
import optuna
from logs.logger_config import setup_logger  # 로거 설정 함수
from backtesting.backtester import Backtester  # 백테스트 수행 클래스
from config.config_manager import ConfigManager, TradingConfig  # 기본 파라미터 관리 및 검증을 위한 클래스
from data.db.db_manager import get_unique_symbol_list  # 데이터베이스에서 유니크 심볼 목록을 가져오는 함수

# 전역 logger 객체: 모듈 내에서 로깅에 사용
logger = setup_logger(__name__)

class DynamicParameterOptimizer:
    def __init__(self, n_trials=10, assets=None):
        """
        동적 파라미터 최적화 클래스의 생성자.
        
        목적:
          - 최적화 시도 횟수와 대상 자산 목록을 초기화합니다.
        
        Parameters:
            n_trials (int): 최적화 시도 횟수 (기본값: 10).
            assets (list, optional): 최적화에 사용할 자산 목록.
                                    주어지지 않을 경우 데이터베이스에서 유니크 심볼을 가져오거나 기본 목록 사용.
        """
        self.n_trials = n_trials  # 최적화 시도 횟수 저장
        self.study = None         # 최적화 결과를 저장할 optuna study 객체 (초기에는 None)
        self.config_manager = ConfigManager()  # 파라미터 기본값 및 유효성 검증을 위한 객체 생성
        # 자산 목록이 주어지지 않으면 데이터베이스에서 가져오거나, 없으면 기본 자산 사용
        self.assets = assets if assets is not None else (get_unique_symbol_list() or ["BTC/USDT", "ETH/USDT", "XRP/USDT"])

    def objective(self, trial):
        """
        optuna 최적화 objective 함수.
        
        목적:
          - 제안된 파라미터 조합에 대해 백테스트를 수행하고, 성능 점수를 계산합니다.
          - 점수가 낮을수록 해당 파라미터 조합이 더 우수하다는 의미입니다.
        
        Parameters:
            trial (optuna.trial.Trial): 현재 최적화 시도의 trial 객체.
        
        Returns:
            float: 계산된 성능 점수 (낮을수록 좋은 파라미터).
        """
        try:
            # 기본 파라미터 값 불러오기
            base_params = self.config_manager.get_defaults()
            
            # optuna를 사용하여 동적으로 파라미터 값을 추천받음
            suggested_params = {
                "hmm_confidence_threshold": trial.suggest_float("hmm_confidence_threshold", 0.7, 0.95),
                "liquidity_info": trial.suggest_categorical("liquidity_info", ["high", "low"]),
                "atr_multiplier": trial.suggest_float("atr_multiplier", 1.5, 3.0),
                "profit_ratio": trial.suggest_float("profit_ratio", 0.05, 0.15),
                "risk_per_trade": trial.suggest_float("risk_per_trade", 0.005, 0.02),
                "scale_in_threshold": trial.suggest_float("scale_in_threshold", 0.01, 0.03),
                "partial_exit_ratio": trial.suggest_float("partial_exit_ratio", 0.4, 0.6),
                "partial_profit_ratio": trial.suggest_float("partial_profit_ratio", 0.02, 0.04),
                "final_profit_ratio": trial.suggest_float("final_profit_ratio", 0.05, 0.1),
                "weekly_breakout_threshold": trial.suggest_float("weekly_breakout_threshold", 0.005, 0.02),
                "weekly_momentum_threshold": trial.suggest_float("weekly_momentum_threshold", 0.3, 0.7),
                "risk_reward_ratio": trial.suggest_float("risk_reward_ratio", 1.0, 5.0)
            }
            # 기본 파라미터와 추천받은 파라미터를 병합하여 최종 파라미터 집합 생성
            dynamic_params = {**base_params, **suggested_params}
            # 구성 매니저를 통해 파라미터 유효성 검사 수행
            dynamic_params = self.config_manager.validate_params(dynamic_params)

            try:
                # TradingConfig 객체를 생성해 파라미터의 추가 검증을 진행
                _ = TradingConfig(**dynamic_params)
            except Exception as e:
                logger.error("Validation error in dynamic_params: " + str(e), exc_info=True)
                return 1e6  # 유효하지 않은 경우 큰 패널티 값 반환

            # 백테스트 기간 설정 (학습, 테스트, 홀드아웃 단계로 구분)
            splits = [{
                "train_start": "2018-06-01",
                "train_end": "2020-12-31",
                "test_start": "2021-01-01",
                "test_end": "2023-12-31"
            }]
            holdout = {"holdout_start": "2024-01-01", "holdout_end": "2025-02-01"}

            total_score, num_evaluations = 0.0, 0

            # 각 기간 및 자산에 대해 백테스트를 수행하여 점수를 계산
            for split in splits:
                for asset in self.assets:
                    # 심볼 키 생성 (예: "BTC/USDT" -> "btcusdt") => 테이블 이름 생성에 사용
                    symbol_key = asset.replace("/", "").lower()
                    try:
                        # 학습 단계: 지정된 기간 동안 백테스트 실행
                        bt_train = Backtester(symbol=asset, account_size=10000)
                        bt_train.load_data(
                            short_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                            long_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                            short_tf="4h", long_tf="1d",
                            start_date=split["train_start"], end_date=split["train_end"]
                        )
                        trades_train, _ = bt_train.run_backtest(dynamic_params)
                        # 학습 단계 수익률(ROI) 계산
                        roi_train = sum(trade["pnl"] for trade in trades_train) / 10000 * 100

                        # 테스트 단계: 다른 기간에 대해 백테스트 실행
                        bt_test = Backtester(symbol=asset, account_size=10000)
                        bt_test.load_data(
                            short_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                            long_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                            short_tf="4h", long_tf="1d",
                            start_date=split["test_start"], end_date=split["test_end"]
                        )
                        trades_test, _ = bt_test.run_backtest(dynamic_params)
                        roi_test = sum(trade["pnl"] for trade in trades_test) / 10000 * 100

                        # 홀드아웃 단계: 최종 검증을 위한 별도 기간
                        bt_holdout = Backtester(symbol=asset, account_size=10000)
                        bt_holdout.load_data(
                            short_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                            long_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                            short_tf="4h", long_tf="1d",
                            start_date=holdout["holdout_start"], end_date=holdout["holdout_end"]
                        )
                        trades_holdout, _ = bt_holdout.run_backtest(dynamic_params)
                        roi_holdout = sum(trade["pnl"] for trade in trades_holdout) / 10000 * 100

                        # 과적합(overfit) 패널티: 학습과 테스트 단계의 수익률 차이를 절대값으로 계산
                        overfit_penalty = abs(roi_train - roi_test)
                        # 홀드아웃 패널티: 홀드아웃 수익률이 최소 2% 미만이면 부족분에 비례해 패널티 부과
                        holdout_penalty = 0 if roi_holdout >= 2.0 else (2.0 - roi_holdout) * 20
                        # 최종 점수 계산: 테스트 수익률의 음수에 패널티들을 더함 (낮을수록 우수)
                        score = -roi_test + overfit_penalty + holdout_penalty
                        total_score += score
                        num_evaluations += 1
                    except Exception as e:
                        logger.error(f"Backtest error for {asset} in split {split}: {e}", exc_info=True)
                        # 백테스트 중 오류가 발생하면 해당 평가 건너뜀
                        continue

            if num_evaluations == 0:
                return 1e6  # 평가가 하나도 진행되지 않았을 경우 큰 페널티 반환
            avg_score = total_score / num_evaluations
            # 정규화 패널티: 주요 파라미터가 기본값에서 벗어난 정도에 따른 패널티 부과
            reg_penalty = 0.1 * sum(
                (dynamic_params.get(key, base_params.get(key, 1.0)) - base_params.get(key, 1.0)) ** 2
                for key in ["atr_multiplier", "profit_ratio", "risk_per_trade", "scale_in_threshold",
                            "weekly_breakout_threshold", "weekly_momentum_threshold"]
            )
            return avg_score + reg_penalty

        except Exception as e:
            logger.error("Unexpected error in objective: " + str(e), exc_info=True)
            return 1e6

    def optimize(self):
        """
        최적화 실행 함수.
        
        목적:
          - n_trials 횟수 동안 objective 함수를 최적화하여 최적의 동적 파라미터 조합을 도출합니다.
        
        Returns:
            optuna.trial.FrozenTrial: 최적의 파라미터 조합과 해당 점수를 포함하는 trial 객체.
        """
        # TPESampler: 효율적인 파라미터 탐색을 위해 Tree-structured Parzen Estimator 사용 (시드 42로 고정)
        sampler = optuna.samplers.TPESampler(seed=42)
        # 최적화 study 생성 (방향은 최소화)
        self.study = optuna.create_study(direction="minimize", sampler=sampler)
        # n_trials 횟수 만큼 objective 함수를 최적화 수행
        self.study.optimize(self.objective, n_trials=self.n_trials)
        best_trial = self.study.best_trial
        # 최적 결과를 디버그 로그로 기록
        logger.debug(f"Best trial: {best_trial.number}, Value: {best_trial.value:.2f}")
        logger.debug(f"Best parameters: {best_trial.params}")
        return best_trial
