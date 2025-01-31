# strategies/position.py
# 포지션(거래 한 건)을 객체로 관리해 주는 클래스.

class Position:
    """
    분할 매수/매도 등을 고려하기 위해, 하나의 '포지션' 안에 여러 sub_position(분할체결)을 저장하는 구조.
    
    - side: "LONG" 혹은 "SHORT" (현재는 LONG만 가정)
    - sub_positions: 실제 매수 체결 내역을 리스트로 보관 (entry_price, size 등)
    - sub_positions[i]['sub_tps']: 여러 익절 가격대와 청산 비중을 담을 수 있음 (분할 익절).
    - initial_entry_price: 분할매수를 시작할 때 기준이 되는 가격
    - max_position_size: 이 포지션에서 최대로 잡을 전체 수량
    - num_splits: 총 분할 매수 횟수
    - splits_filled: 지금까지 몇 분할이 체결되었는지
    - split_plan: 각 분할마다 어느 비중으로 매수할지 비율 리스트 (예: [0.3, 0.3, 0.4])
    """

    def __init__(
        self,
        side="LONG",
        initial_entry_price=None,
        max_position_size=0.0,
        num_splits=1,
        split_plan=None
    ):
        self.side = side
        self.sub_positions = []

        self.initial_entry_price = initial_entry_price
        self.max_position_size = max_position_size
        self.num_splits = num_splits
        self.splits_filled = 0

        if split_plan is None:
            split_plan = []
        self.split_plan = split_plan

    def add_sub_position(
        self,
        entry_price,
        size,
        stop_loss=None,
        take_profit=None,
        entry_time=None,
        sub_tps=None
    ):
        """
        한 번의 매수(분할 체결)를 기록하여 sub_positions에 추가합니다.
        sub_tps: 여러 익절 가격과 청산비중을 담은 리스트 (예: [(50000, 0.5), (55000, 1.0)])
            => 첫 가격도달 시 50% 청산, 두 번째 가격도달 시 나머지 전량 청산
        """
        if sub_tps is None:
            sub_tps = []
        sub_tps_with_flags = []
        for (tp_price, tp_ratio) in sub_tps:
            sub_tps_with_flags.append({
                'price': tp_price,
                'close_ratio': tp_ratio,
                'hit': False
            })

        self.sub_positions.append({
            'entry_price': entry_price,
            'size': size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': entry_time,
            'sub_tps': sub_tps_with_flags
        })
        
    def total_size(self) -> float:
        """
        현재까지 매수된 총 수량 합.
        """
        return sum(sp['size'] for sp in self.sub_positions)

    def average_price(self) -> float:
        """
        가중평균 매수가 계산: (매수가*수량)의 합 / 총수량
        """
        total_cost = 0.0
        total_qty = 0.0
        for sp in self.sub_positions:
            total_cost += sp['entry_price'] * sp['size']
            total_qty += sp['size']
        return (total_cost / total_qty) if total_qty > 0 else 0.0

    def close_sub_position(self, idx: int):
        """
        지정된 인덱스의 sub_position 하나를 전량 청산(목록에서 제거).
        """
        if 0 <= idx < len(self.sub_positions):
            self.sub_positions.pop(idx)

    def is_empty(self) -> bool:
        """
        보유 중인 sub_position이 하나도 없으면 True (즉 포지션이 비어있음)
        """
        return (len(self.sub_positions) == 0)

    def partial_close_sub_position(self, idx: int, close_ratio: float) -> float:
        """
        지정된 sub_position을 close_ratio 비율만큼 청산.
        예) close_ratio=0.5 => 절반만 청산
        반환값: 청산된 수량
        """
        if 0 <= idx < len(self.sub_positions):
            sp = self.sub_positions[idx]
            qty_to_close = sp['size'] * close_ratio
            sp['size'] = sp['size'] - qty_to_close
            return qty_to_close
        return 0.0