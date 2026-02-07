"""
Message protocol and broker for hierarchical control

Handles all communication between components with logging and tracking

Author: Guowen Li
Date: 2025-02-06
"""

import json
import logging
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class MessageBroker:
    """
    Central message broker for hierarchical control
    
    Logs all communication between:
    - Buildings → Aggregator (flexibility bands)
    - Aggregator → Buildings (power budgets)
    - Buildings → Coordinator (states, MPC results)
    """
    
    def __init__(self, log_to_file: bool = True, output_dir: str = 'results'):
        self.log_to_file = log_to_file
        self.output_dir = Path(output_dir)
        self.messages = []
        self.message_count = 0
        
        # Create output directory
        if self.log_to_file:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def send_flexibility_band(self, building_id: str, band: Any, recipient: str = 'Aggregator'):
        """Log flexibility band transmission"""
        message = {
            'timestamp': band.timestamp,
            'message_id': self.message_count,
            'type': 'flexibility_band',
            'sender': building_id,
            'recipient': recipient,
            'data': {
                'P_lower_0': band.P_lower_kW[0] if band.P_lower_kW else None,
                'P_upper_0': band.P_upper_kW[0] if band.P_upper_kW else None,
                'horizon_length': len(band.P_lower_kW),
                'computation_time_s': band.computation_time_s,
                'feasible': band.feasible
            }
        }
        self._log_message(message)
        return self.message_count - 1
    
    def send_power_budget(self, budget: Any, recipient: str):
        """Log power budget allocation"""
        message = {
            'timestamp': budget.timestamp,
            'message_id': self.message_count,
            'type': 'power_budget',
            'sender': 'Aggregator',
            'recipient': recipient,
            'data': {
                'P_ref_0': budget.P_ref_kW[0] if budget.P_ref_kW else None,
                'P_limit': budget.P_limit_kW,
                'priority_level': budget.priority_level,
                'horizon_length': len(budget.P_ref_kW)
            }
        }
        self._log_message(message)
        return self.message_count - 1
    
    def send_building_state(self, state: Any, recipient: str = 'Coordinator'):
        """Log building state update"""
        message = {
            'timestamp': state.timestamp,
            'message_id': self.message_count,
            'type': 'building_state',
            'sender': state.building_id,
            'recipient': recipient,
            'data': {
                'power_actual_kW': state.power_actual_kW,
                'status': state.status.value if hasattr(state.status, 'value') else str(state.status),
                'control_mode': state.control_mode.value if hasattr(state.control_mode, 'value') else str(state.control_mode)
            }
        }
        self._log_message(message)
        return self.message_count - 1
    
    def send_mpc_result(self, result: Any, recipient: str = 'Coordinator'):
        """Log MPC solution"""
        message = {
            'timestamp': result.timestamp,
            'message_id': self.message_count,
            'type': 'mpc_result',
            'sender': result.building_id,
            'recipient': recipient,
            'data': {
                'objective_value': result.objective_value,
                'solve_time_s': result.solve_time_s,
                'feasible': result.feasible,
                'max_budget_violation': result.get_max_budget_violation()
            }
        }
        self._log_message(message)
        return self.message_count - 1
    
    def send_allocation_result(self, allocation: Any):
        """Log aggregator allocation result"""
        message = {
            'timestamp': allocation.timestamp,
            'message_id': self.message_count,
            'type': 'allocation_result',
            'sender': 'Aggregator',
            'recipient': 'All_Buildings',
            'data': {
                'total_power_0': allocation.total_power_kW[0] if allocation.total_power_kW else None,
                'feeder_limit_0': allocation.feeder_limit_kW[0] if allocation.feeder_limit_kW else None,
                'objective_value': allocation.objective_value,
                'solve_time_s': allocation.solve_time_s,
                'feasible': allocation.feasible,
                'num_buildings': len(allocation.budgets)
            }
        }
        self._log_message(message)
        return self.message_count - 1
    
    def _log_message(self, message: Dict):
        """Internal: Log message and increment counter"""
        self.messages.append(message)
        self.message_count += 1
        
        logger.debug(f"Message {self.message_count-1}: {message['sender']} → {message['recipient']} "
                    f"({message['type']})")
    
    def save_messages(self, filename: str = None):
        """Save all messages to JSON file"""
        if not self.log_to_file:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'messages_{timestamp}.json'
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.messages, f, indent=2)
        
        logger.info(f"💾 Saved {len(self.messages)} messages to {filepath}")
        
        return filepath
    
    def get_statistics(self) -> Dict:
        """Get message statistics"""
        stats = {
            'total_messages': len(self.messages),
            'by_type': {},
            'by_sender': {}
        }
        
        for msg in self.messages:
            msg_type = msg['type']
            sender = msg['sender']
            
            stats['by_type'][msg_type] = stats['by_type'].get(msg_type, 0) + 1
            stats['by_sender'][sender] = stats['by_sender'].get(sender, 0) + 1
        
        return stats
    
    def clear(self):
        """Clear all messages"""
        self.messages = []
        self.message_count = 0
        logger.info("📨 Message broker cleared")
