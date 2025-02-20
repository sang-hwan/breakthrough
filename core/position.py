# core/position.py
import uuid
from logs.logger_config import setup_logger

logger = setup_logger(__name__)

class Position:
    def __init__(self, side: str = "LONG", initial_price: float = None, maximum_size: float = 0.0,
                 total_splits: int = 1, allocation_plan: list = None) -> None:
        if initial_price is None or initial_price <= 0:
            raise ValueError("Initial price must be positive.")
        self.position_id: str = str(uuid.uuid4())
        self.side: str = side.upper()
        self.executions: list = []
        self.initial_price: float = initial_price
        self.maximum_size: float = maximum_size
        self.total_splits: int = total_splits
        self.executed_splits: int = 0
        self.allocation_plan: list = allocation_plan if allocation_plan is not None else []
        if self.side == "SHORT":
            self.lowest_price: float = initial_price
        else:
            self.highest_price: float = initial_price
        logger.debug(f"New position created: ID={self.position_id}, side={self.side}, entry price={self.initial_price}")

    def add_execution(self, entry_price: float, size: float, stop_loss: float = None,
                      take_profit: float = None, entry_time=None, exit_targets: list = None,
                      trade_type: str = "unknown", min_order_size: float = 1e-8) -> None:
        if size < min_order_size:
            logger.warning("Execution size below minimum order size; execution not added.")
            return
        if exit_targets and not isinstance(exit_targets, list):
            logger.error("exit_targets must be a list.", exc_info=True)
            return

        targets = []
        if exit_targets:
            for target_price, exit_ratio in exit_targets:
                targets.append({'price': target_price, 'exit_ratio': exit_ratio, 'hit': False})
        execution = {
            'entry_price': entry_price,
            'size': size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': entry_time,
            'exit_targets': targets,
            'trade_type': trade_type,
            'closed': False
        }
        if self.side == "SHORT":
            execution["lowest_price_since_entry"] = entry_price
        else:
            execution["highest_price_since_entry"] = entry_price

        self.executions.append(execution)
        logger.debug(f"Execution added: entry_price={entry_price}, size={size}, type={trade_type}")

    def update_extremum(self, current_price: float) -> None:
        for record in self.executions:
            if record.get("closed", False):
                continue
            if self.side == "LONG":
                prev = record.get("highest_price_since_entry", record["entry_price"])
                if current_price > prev:
                    record["highest_price_since_entry"] = current_price
                    logger.debug(f"Updated highest price: {prev} -> {current_price} for execution at entry {record['entry_price']}")
            elif self.side == "SHORT":
                prev = record.get("lowest_price_since_entry", record["entry_price"])
                if current_price < prev:
                    record["lowest_price_since_entry"] = current_price
                    logger.debug(f"Updated lowest price: {prev} -> {current_price} for execution at entry {record['entry_price']}")
        logger.debug(f"Extremum values updated with current_price={current_price}")

    def get_total_size(self) -> float:
        return sum(record['size'] for record in self.executions if not record.get("closed", False))

    def get_average_entry_price(self) -> float:
        total_cost = sum(record['entry_price'] * record['size'] for record in self.executions if not record.get("closed", False))
        total_qty = self.get_total_size()
        return total_cost / total_qty if total_qty > 0 else 0.0

    def remove_execution(self, index: int) -> None:
        if 0 <= index < len(self.executions):
            self.executions.pop(index)
            logger.debug(f"Execution removed at index {index}")
        else:
            logger.warning(f"Failed to remove execution: invalid index {index}")

    def is_empty(self) -> bool:
        return all(record.get("closed", False) for record in self.executions)

    def partial_close_execution(self, index: int, close_ratio: float, min_order_size: float = 1e-8) -> float:
        if not (0 < close_ratio <= 1):
            logger.error("close_ratio must be between 0 and 1.", exc_info=True)
            return 0.0
        if 0 <= index < len(self.executions):
            record = self.executions[index]
            qty_to_close = record['size'] * close_ratio
            record['size'] -= qty_to_close
            if record.get('exit_targets'):
                record['exit_targets'] = [t for t in record['exit_targets'] if not t.get('hit', False)]
            if record['size'] < min_order_size:
                record['closed'] = True
                logger.debug(f"Execution at index {index} closed due to size below minimum order size.")
            logger.debug(f"Partial close executed: index={index}, ratio={close_ratio}, closed qty={qty_to_close}")
            return qty_to_close
        logger.warning(f"Partial close failed: invalid index {index}")
        return 0.0
