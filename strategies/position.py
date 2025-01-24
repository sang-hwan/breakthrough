# strategies/position.py

class Position:
    """
    하나의 트레이드(포지션) 정보를 관리.
    - sub_positions: 실제 매수 체결(분할 진입)한 각각을 리스트로 저장
    - initial_entry_price: 첫 진입 시의 기준 가격
    - max_position_size: 이 포지션에서 허용할 최대 보유 수량
    - num_splits: 총 분할 매수 횟수(첫 진입 포함)
    - splits_filled: 현재까지 몇 번(몇 개) 분할을 실행했는지
    """

    def __init__(
        self,
        side="LONG",
        initial_entry_price=None,
        max_position_size=0.0,
        num_splits=1
    ):
        self.side = side  # "LONG" 또는 "SHORT" 가정
        self.sub_positions = []  # 분할 진입 내역 목록

        # 분할매수 계획 관련
        self.initial_entry_price = initial_entry_price  # 최초 진입가
        self.max_position_size = max_position_size      # 이 포지션에서 허용할 최대 수량
        self.num_splits = num_splits                    # 전체 분할 횟수(예: 3)
        self.splits_filled = 0                          # 현재까지 몇 분할이 체결되었나
        
        # split_plan: 예) [0.33, 0.33, 0.34] 처럼, 분할 비중 리스트
        #             size = max_position_size * split_plan[i]로 계산
        if split_plan is None:
            split_plan = []
        self.split_plan = split_plan

    def add_sub_position(self, entry_price, size, stop_loss=None, take_profit=None, entry_time=None):
        """
        분할매수 1회 실행 시, 실제 체결된 sub_position 정보 등록
        """
        self.sub_positions.append({
            'entry_price': entry_price,
            'size': size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': entry_time
        })

    def total_size(self) -> float:
        """
        현재까지 매수된 총 수량
        """
        return sum(sp['size'] for sp in self.sub_positions)

    def average_price(self) -> float:
        """
        현재까지의 가중평균 매수가
        """
        total_cost = 0.0
        total_qty = 0.0
        for sp in self.sub_positions:
            total_cost += sp['entry_price'] * sp['size']
            total_qty += sp['size']
        return (total_cost / total_qty) if total_qty > 0 else 0.0

    def close_sub_position(self, idx: int):
        """
        특정 sub_position 전량 청산(리스트에서 제거)
        """
        if 0 <= idx < len(self.sub_positions):
            self.sub_positions.pop(idx)

    def is_empty(self) -> bool:
        """
        보유 중인 sub_position이 하나도 없으면 True
        """
        return (len(self.sub_positions) == 0)
