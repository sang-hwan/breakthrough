# trading/positions.py
import uuid
import logging

class TradePosition:
    def __init__(self, side="LONG", initial_price: float = None, maximum_size: float = 0.0, total_splits: int = 1, allocation_plan: list = None):
        self.position_id = str(uuid.uuid4())
        self.side = side
        self.executions = []
        self.initial_price = initial_price
        self.maximum_size = maximum_size
        self.total_splits = total_splits
        self.executed_splits = 0
        self.allocation_plan = allocation_plan if allocation_plan is not None else []
        self.highest_price = initial_price if initial_price is not None else 0.0

    def add_execution(self, entry_price: float, size: float, stop_loss: float = None, take_profit: float = None, entry_time = None, exit_targets: list = None, trade_type: str = "unknown", min_order_size: float = 1e-8):
        if size < min_order_size:
            logging.info("Execution size below min_order_size; skipping execution.")
            return
        exit_targets_flagged = []
        if exit_targets is not None:
            for target_price, exit_ratio in exit_targets:
                exit_targets_flagged.append({
                    'price': target_price,
                    'exit_ratio': exit_ratio,
                    'hit': False
                })
        self.executions.append({
            'entry_price': entry_price,
            'size': size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': entry_time,
            'exit_targets': exit_targets_flagged,
            'trade_type': trade_type,
            'highest_price_since_entry': entry_price
        })
        
    def get_total_size(self) -> float:
        return sum(exec_record['size'] for exec_record in self.executions)

    def get_average_entry_price(self) -> float:
        total_cost = sum(exec_record['entry_price'] * exec_record['size'] for exec_record in self.executions)
        total_qty = self.get_total_size()
        return (total_cost / total_qty) if total_qty > 0 else 0.0

    def remove_execution(self, index: int):
        if 0 <= index < len(self.executions):
            self.executions.pop(index)

    def is_empty(self) -> bool:
        return len(self.executions) == 0

    def partial_close_execution(self, index: int, close_ratio: float, min_order_size: float = 1e-8) -> float:
        if 0 <= index < len(self.executions):
            exec_record = self.executions[index]
            qty_to_close = exec_record['size'] * close_ratio
            exec_record['size'] -= qty_to_close
            if 'exit_targets' in exec_record and exec_record['exit_targets']:
                exec_record['exit_targets'] = [t for t in exec_record['exit_targets'] if not t.get('hit', False)]
            if exec_record['size'] < min_order_size:
                exec_record['closed'] = True
            return qty_to_close
        return 0.0
