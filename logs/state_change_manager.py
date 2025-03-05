# logs/state_change_manager.py
class StateChangeManager:
    def __init__(self, numeric_threshold: float = 0.01):
        """
        :param numeric_threshold: 숫자형 상태 값의 상대 변화가 이 값 이상일 때만 변화를 감지 (기본 1%).
        """
        self._state_dict = {}
        self.numeric_threshold = numeric_threshold

    def has_changed(self, key: str, new_value) -> bool:
        """
        key에 해당하는 상태가 이전과 비교하여 의미 있는 변화(숫자형이면 상대 1% 이상, 그 외는 단순 불일치)가 있으면 True를 반환하고, 새로운 값을 저장합니다.
        """
        old_value = self._state_dict.get(key)
        # 처음 상태인 경우
        if old_value is None:
            self._state_dict[key] = new_value
            return True

        # 숫자형 값인 경우: 상대 변화율 비교 (단, old_value가 0이면 절대 변화량으로 판단)
        if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
            if old_value == 0:
                # 0인 경우, 절대 변화량이 임계값보다 크면 변화로 판단
                if abs(new_value) >= self.numeric_threshold:
                    self._state_dict[key] = new_value
                    return True
                else:
                    return False
            else:
                relative_change = abs(new_value - old_value) / abs(old_value)
                if relative_change >= self.numeric_threshold:
                    self._state_dict[key] = new_value
                    return True
                else:
                    return False
        else:
            # 숫자형이 아닌 경우에는 기존 방식 그대로
            if old_value != new_value:
                self._state_dict[key] = new_value
                return True
            else:
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
