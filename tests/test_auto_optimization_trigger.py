# tests/test_auto_optimization_trigger.py

def auto_optimization_trigger(performance):
    """
    더미 함수: 월간 ROI가 2% 미만인 달이 있으면 True를 반환합니다.
    """
    monthly_roi = performance.get("monthly_roi", {})
    for roi in monthly_roi.values():
        if roi < 2.0:
            return True
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
