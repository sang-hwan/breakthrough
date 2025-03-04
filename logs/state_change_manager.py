# logs/state_change_manager.py
class StateChangeManager:
    """
    여러 모듈에서 공통으로 상태 변화(예: 주문 신호, 데이터 로드 결과, 계좌 상태 등)를 관리하는 클래스입니다.
    각 key 별로 이전 상태를 저장하고, 새로운 값이 기존과 다를 경우에만 변화로 간주하여 True를 반환합니다.
    """

    def __init__(self):
        self._state_dict = {}

    def has_changed(self, key: str, new_value) -> bool:
        """
        key에 해당하는 상태가 이전과 다르면 True를 반환하고, 새로운 값을 저장합니다.
        """
        old_value = self._state_dict.get(key)
        if old_value != new_value:
            self._state_dict[key] = new_value
            return True
        return False

    def get_state(self, key: str):
        """
        현재 저장된 상태 값을 반환합니다.
        """
        return self._state_dict.get(key)

    def reset_state(self, key: str = None):
        """
        상태를 리셋합니다.
        - key가 제공되면 해당 key의 상태만 리셋.
        - key가 없으면 전체 상태를 초기화.
        """
        if key:
            if key in self._state_dict:
                del self._state_dict[key]
        else:
            self._state_dict.clear()
