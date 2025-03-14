# backtesting/steps/order_manager.py

from logs.logger_config import setup_logger
from logs.logging_util import LoggingUtil  # 동적 상태 변화 로깅 유틸리티

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
