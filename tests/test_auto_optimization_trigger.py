# tests/test_auto_optimization_trigger.py

import logging

logger = logging.getLogger(__name__)

def auto_optimization_trigger(performance):
    """
    더미 함수: 월간 ROI가 2% 미만인 달이 있으면 True를 반환합니다.
    """
    monthly_roi = performance.get("monthly_roi", {})
    for month, roi in monthly_roi.items():
        logger.debug(f"검사 중 - {month}: ROI = {roi}")
        if roi < 2.0:
            logger.info(f"자동 최적화 트리거 활성화: {month}의 ROI({roi})가 2% 미만입니다.")
            return True
    logger.info("자동 최적화 트리거 비활성화: 모든 월간 ROI가 2% 이상입니다.")
    return False

def test_auto_optimization_trigger():
    performance_trigger = {
        "monthly_roi": {
            "2023-01": 1.5,
            "2023-02": 2.5,
            "2023-03": 1.8,
        }
    }
    performance_no_trigger = {
        "monthly_roi": {
            "2023-01": 2.1,
            "2023-02": 2.5,
            "2023-03": 2.3,
        }
    }
    assert auto_optimization_trigger(performance_trigger) is True
    assert auto_optimization_trigger(performance_no_trigger) is False
