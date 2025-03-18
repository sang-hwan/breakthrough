# logs/state_change_manager.py
class StateChangeManager:
    """
    StateChangeManager는 상태 값을 추적하며,
    이전 상태와 비교하여 의미 있는 변화가 있는지 확인합니다.
    숫자형 상태의 경우, 상대 변화율이 일정 임계값 이상일 때만 변화를 감지합니다.
    """
    def __init__(self, numeric_threshold: float = 0.01):
        """
        초기화 메서드
        
        Parameters:
            numeric_threshold (float): 숫자형 상태 값의 상대 변화 임계값 (기본값 0.01, 즉 1%)
        
        주요 동작:
          - 상태 값을 저장할 내부 딕셔너리 초기화
          - 숫자형 상태 변화 감지를 위한 임계값 설정
        """
        self._state_dict = {}  # 상태 값을 저장할 딕셔너리, key: 상태 키, value: 상태 값
        self.numeric_threshold = numeric_threshold

    def has_changed(self, key: str, new_value) -> bool:
        """
        기존 상태와 비교하여 주어진 key의 상태가 의미 있게 변경되었는지 확인합니다.
        - 숫자형 상태의 경우, 상대 변화율(또는 절대값 비교)을 통해 판단.
        - 숫자형이 아닌 경우, 단순 불일치를 확인.
        
        Parameters:
            key (str): 상태를 식별하는 키
            new_value: 새로운 상태 값 (숫자형 또는 기타)
        
        Returns:
            bool: 상태가 변경되었으면 True, 그렇지 않으면 False
        
        주요 동작:
          - 초기 상태라면 저장 후 True 반환
          - 숫자형 값이면, old_value가 0인 경우 절대 변화량 비교, 아니면 상대 변화율 비교
          - 숫자형이 아닌 경우 단순 비교 수행
        """
        old_value = self._state_dict.get(key)
        # 처음 상태인 경우
        if old_value is None:
            self._state_dict[key] = new_value
            return True

        # 숫자형 값인 경우: 상대 변화율 비교 (old_value가 0인 경우 절대 변화량으로 판단)
        if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
            if old_value == 0:
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
            # 숫자형이 아닌 경우 단순 비교
            if old_value != new_value:
                self._state_dict[key] = new_value
                return True
            else:
                return False

    def get_state(self, key: str):
        """
        현재 저장된 상태 값을 반환합니다.
        
        Parameters:
            key (str): 상태를 식별하는 키
        
        Returns:
            해당 key에 대한 상태 값 (존재하지 않으면 None)
        """
        return self._state_dict.get(key)

    def reset_state(self, key: str = None):
        """
        상태 값을 리셋합니다.
        
        Parameters:
            key (str, optional): 특정 상태만 리셋할 경우 해당 키 지정. (지정하지 않으면 전체 상태 초기화)
        
        Returns:
            None
        """
        if key:
            if key in self._state_dict:
                del self._state_dict[key]
        else:
            self._state_dict.clear()
