# backtesting/parameter_optimization.py

import os
import json
import hashlib
import optuna
import traceback

from backtesting.backtest_advanced import run_advanced_backtest

# 캐시 폴더 설정 (없으면 생성)
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_cache_key(params: dict, start_date: str, end_date: str) -> str:
    key_data = {"params": params, "start_date": start_date, "end_date": end_date}
    key_str = json.dumps(key_data, sort_keys=True)
    key_hash = hashlib.md5(key_str.encode("utf-8")).hexdigest()
    return key_hash

def load_cached_result(cache_key: str):
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
            return data.get("roi")
        except Exception:
            return None
    return None

def save_cached_result(cache_key: str, roi: float):
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    with open(cache_file, "w") as f:
        json.dump({"roi": roi}, f)

def sample_parameter(trial, name, param_spec):
    if "options" in param_spec:
        return trial.suggest_categorical(name, param_spec["options"])
    elif "type" in param_spec and param_spec["type"] == "boolean":
        return trial.suggest_categorical(name, [True, False])
    elif "range" in param_spec:
        min_val = param_spec["range"]["min"]
        max_val = param_spec["range"]["max"]
        if isinstance(min_val, int) and isinstance(max_val, int):
            return trial.suggest_int(name, min_val, max_val)
        else:
            return trial.suggest_float(name, min_val, max_val)
    else:
        raise ValueError(f"Parameter spec for {name} not recognized.")

def objective(trial: optuna.trial.Trial) -> float:
    # parameter_group.json 파일 읽기
    param_file = os.path.join(os.path.dirname(__file__), "parameter_group.json")
    with open(param_file, "r", encoding="utf-8") as f:
        param_groups = json.load(f)

    candidate_params = {}

    # [신호 생성] - breakout_signal 그룹
    breakout_signal = param_groups["signal_generation"]["breakout_signal"]
    candidate_params["lookback_window"] = sample_parameter(trial, "lookback_window", breakout_signal["lookback_window"])
    candidate_params["volume_factor"] = sample_parameter(trial, "volume_factor", breakout_signal["volume_factor"])
    candidate_params["confirmation_bars"] = sample_parameter(trial, "confirmation_bars", breakout_signal["confirmation_bars"])
    candidate_params["breakout_buffer"] = sample_parameter(trial, "breakout_buffer", breakout_signal["breakout_buffer"])
    
    # [신호 생성] - 리테스트 관련
    retest_settings = breakout_signal["retest_settings"]
    candidate_params["retest_threshold"] = sample_parameter(trial, "retest_threshold", retest_settings["retest_threshold"])
    candidate_params["retest_confirmation_bars"] = sample_parameter(trial, "retest_confirmation_bars", retest_settings["retest_confirmation_bars"])
    
    # [신호 생성] - 옵션 (signal_options 그룹)
    signal_options = breakout_signal["signal_options"]
    candidate_params["use_short_term_indicators"] = sample_parameter(trial, "use_short_term_indicators", signal_options["use_short_term_indicators"])
    candidate_params["short_rsi_threshold"] = sample_parameter(trial, "short_rsi_threshold", signal_options["short_rsi_threshold"])
    candidate_params["short_rsi_period"] = sample_parameter(trial, "short_rsi_period", signal_options["short_rsi_period"])

    # [신호 생성] - trend_filter 그룹
    trend_filter = param_groups["signal_generation"]["trend_filter"]
    candidate_params["sma_period"] = sample_parameter(trial, "sma_period", trend_filter["sma_period"])
    candidate_params["macd_slow_period"] = sample_parameter(trial, "macd_slow_period", trend_filter["macd_slow_period"])
    candidate_params["macd_fast_period"] = sample_parameter(trial, "macd_fast_period", trend_filter["macd_fast_period"])
    candidate_params["macd_signal_period"] = sample_parameter(trial, "macd_signal_period", trend_filter["macd_signal_period"])
    candidate_params["rsi_period"] = sample_parameter(trial, "rsi_period", trend_filter["rsi_period"])
    candidate_params["rsi_threshold"] = sample_parameter(trial, "rsi_threshold", trend_filter["rsi_threshold"])
    candidate_params["bb_period"] = sample_parameter(trial, "bb_period", trend_filter["bb_period"])
    candidate_params["bb_std_multiplier"] = sample_parameter(trial, "bb_std_multiplier", trend_filter["bb_std_multiplier"])

    # [리스크 관리] - atr_stop 그룹
    atr_stop = param_groups["risk_management"]["atr_stop"]
    candidate_params["atr_period"] = sample_parameter(trial, "atr_period", atr_stop["atr_period"])
    candidate_params["atr_multiplier"] = sample_parameter(trial, "atr_multiplier", atr_stop["atr_multiplier"])

    # [리스크 관리] - take_profit 그룹
    take_profit = param_groups["risk_management"]["take_profit"]
    candidate_params["profit_ratio"] = sample_parameter(trial, "profit_ratio", take_profit["profit_ratio"])
    candidate_params["use_trailing_stop"] = sample_parameter(trial, "use_trailing_stop", take_profit["use_trailing_stop"])
    candidate_params["trailing_percent"] = (
        sample_parameter(trial, "trailing_percent", take_profit["trailing_percent"])
        if candidate_params["use_trailing_stop"] else 0.0
    )
    candidate_params["use_partial_take_profit"] = sample_parameter(trial, "use_partial_take_profit", take_profit["use_partial_take_profit"])
    candidate_params["partial_tp_factor"] = sample_parameter(trial, "partial_tp_factor", take_profit["partial_tp_factor"])
    candidate_params["final_tp_factor"] = sample_parameter(trial, "final_tp_factor", take_profit["final_tp_factor"])
    candidate_params["use_trend_exit"] = sample_parameter(trial, "use_trend_exit", take_profit["use_trend_exit"])
    candidate_params["risk_per_trade"] = sample_parameter(trial, "risk_per_trade", take_profit["risk_per_trade"])

    # [포지션 사이징]
    position_sizing = param_groups["position_sizing"]
    candidate_params["total_splits"] = sample_parameter(trial, "total_splits", position_sizing["total_splits"])
    candidate_params["allocation_mode"] = sample_parameter(trial, "allocation_mode", position_sizing["allocation_mode"])
    candidate_params["scale_in_threshold"] = sample_parameter(trial, "scale_in_threshold", position_sizing["scale_in_threshold"])

    # [청산 관리] - exit_management 그룹
    exit_management = param_groups["exit_management"]
    candidate_params["trend_exit_lookback"] = sample_parameter(trial, "trend_exit_lookback", exit_management["trend_exit_lookback"])

    # [기타] - misc (타임프레임은 최적화 대상이 아니므로 예시값 사용)
    candidate_params["short_timeframe"] = param_groups["misc"]["short_timeframe"]["example"]
    candidate_params["long_timeframe"] = param_groups["misc"]["long_timeframe"]["example"]
    candidate_params["trend_exit_price_column"] = "close"

    # 추가 고정 파라미터
    candidate_params["atr_window"] = candidate_params["atr_period"]
    candidate_params["window"] = candidate_params["lookback_window"]

    # 신호 관련 컬럼명 (고정값)
    candidate_params["breakout_flag_col"] = "breakout_signal"
    candidate_params["confirmed_breakout_flag_col"] = "confirmed_breakout"
    candidate_params["retest_signal_col"] = "retest_signal"
    candidate_params["long_entry_col"] = "long_entry"

    fee_rate = 0.001
    slippage_rate = 0.0005
    taker_fee_rate = 0.001
    account_size = 10000.0

    walk_forward_periods = [
        ("2018-01-01", "2019-01-01"),
        ("2019-01-01", "2020-01-01"),
        ("2020-01-01", "2021-01-01")
    ]

    cumulative_roi = 0.0
    period_count = 0

    params_for_cache = candidate_params.copy()

    for start_date, end_date in walk_forward_periods:
        cache_key = get_cache_key(params_for_cache, start_date, end_date)
        cached_roi = load_cached_result(cache_key)
        if cached_roi is not None:
            roi = cached_roi
        else:
            try:
                result = run_advanced_backtest(
                    symbol="BTC/USDT",
                    short_timeframe=candidate_params["short_timeframe"],
                    long_timeframe=candidate_params["long_timeframe"],
                    window=candidate_params["window"],
                    volume_factor=candidate_params["volume_factor"],
                    confirm_bars=candidate_params["confirmation_bars"],
                    breakout_buffer=candidate_params["breakout_buffer"],
                    atr_window=candidate_params["atr_window"],
                    atr_multiplier=candidate_params["atr_multiplier"],
                    profit_ratio=candidate_params["profit_ratio"],
                    account_size=account_size,
                    risk_per_trade=candidate_params["risk_per_trade"],
                    fee_rate=fee_rate,
                    slippage_rate=slippage_rate,
                    taker_fee_rate=taker_fee_rate,
                    total_splits=candidate_params["total_splits"],
                    allocation_mode=candidate_params["allocation_mode"],
                    threshold_percent=candidate_params["scale_in_threshold"],
                    use_trailing_stop=candidate_params["use_trailing_stop"],
                    trailing_percent=candidate_params["trailing_percent"],
                    use_trend_exit=candidate_params["use_trend_exit"],
                    trend_exit_lookback=candidate_params["trend_exit_lookback"],
                    trend_exit_price_column=candidate_params["trend_exit_price_column"],
                    start_date=start_date,
                    end_date=end_date,
                    use_partial_take_profit=candidate_params["use_partial_take_profit"],
                    partial_tp_factor=candidate_params["partial_tp_factor"],
                    final_tp_factor=candidate_params["final_tp_factor"],
                    # Trend Filter 관련 파라미터
                    sma_period=candidate_params["sma_period"],
                    macd_slow_period=candidate_params["macd_slow_period"],
                    macd_fast_period=candidate_params["macd_fast_period"],
                    macd_signal_period=candidate_params["macd_signal_period"],
                    rsi_period=candidate_params["rsi_period"],
                    rsi_threshold=candidate_params["rsi_threshold"],
                    bb_period=candidate_params["bb_period"],
                    bb_std_multiplier=candidate_params["bb_std_multiplier"],
                    # 리테스트 및 신호 관련 파라미터
                    retest_threshold=candidate_params["retest_threshold"],
                    retest_confirmation_bars=candidate_params["retest_confirmation_bars"],
                    breakout_flag_col=candidate_params["breakout_flag_col"],
                    confirmed_breakout_flag_col=candidate_params["confirmed_breakout_flag_col"],
                    retest_signal_col=candidate_params["retest_signal_col"],
                    long_entry_col=candidate_params["long_entry_col"]
                )
            except Exception:
                trial.set_user_attr("error", traceback.format_exc())
                return 1e6

            if result is None:
                roi = -5.0
            else:
                trades_df, _ = result
                if trades_df is None or trades_df.empty:
                    roi = -5.0
                else:
                    total_pnl = trades_df["pnl"].sum()
                    final_balance = account_size + total_pnl
                    roi = (final_balance - account_size) / account_size * 100
            save_cached_result(cache_key, roi)
        cumulative_roi += roi
        period_count += 1

    average_roi = cumulative_roi / period_count if period_count > 0 else -100.0
    return -average_roi  # Optuna는 최소화를 목표로 하므로 ROI 부호를 반전

def optimize_parameters(n_trials: int = 50) -> optuna.study.Study:
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study
