"""
Communication interface for hierarchical control

Author: Guowen Li, AI Assistant
Date: 2025-01-07
"""

import json
import logging
from typing import Dict, Any, Optional
from dataclasses import asdict
from .data_models import (
    BuildingState, AggregatorCommand, FeederStatus, 
    BuildingStatus, ControlMode
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageBroker:
    """Central message broker for component communication"""
    
    def __init__(self):
        self.message_queue = {
            'building_a_to_aggregator': [],
            'building_b_to_aggregator': [],
            'aggregator_to_building_a': [],
            'aggregator_to_building_b': [],
            'system_status': []
        }
        
        self.latest_states = {
            'building_a': None,
            'building_b': None,
            'aggregator': None,
            'feeder': None
        }
    
    def send_building_state(self, state: BuildingState):
        """Building sends state update to aggregator"""
        channel = f"{state.building_id.lower()}_to_aggregator"
        if channel in self.message_queue:
            self.message_queue[channel].append(state)
            self.latest_states[state.building_id.lower()] = state
            logger.debug(f"📤 {state.building_id} state sent: P={state.power_actual_kW:.2f}kW, Status={state.status.value}")
    
    def get_building_state(self, building_id: str) -> Optional[BuildingState]:
        """Aggregator retrieves latest building state"""
        return self.latest_states.get(building_id.lower())
    
    def send_aggregator_command(self, command: AggregatorCommand):
        """Aggregator sends command to building"""
        channel = f"aggregator_to_{command.building_id.lower()}"
        if channel in self.message_queue:
            self.message_queue[channel].append(command)
            logger.debug(f"📥 Command to {command.building_id}: P_ref={command.power_reference_kW[0]:.2f}kW, Attack={command.attack_flag}")
    
    def get_aggregator_command(self, building_id: str) -> Optional[AggregatorCommand]:
        """Building retrieves latest command from aggregator"""
        channel = f"aggregator_to_{building_id.lower()}"
        if channel in self.message_queue and self.message_queue[channel]:
            return self.message_queue[channel][-1]  # Latest command
        return None
    
    def update_feeder_status(self, status: FeederStatus):
        """Update feeder status"""
        self.latest_states['feeder'] = status
        self.message_queue['system_status'].append(status)
    
    def get_feeder_status(self) -> Optional[FeederStatus]:
        """Retrieve latest feeder status"""
        return self.latest_states.get('feeder')
    
    def clear_queue(self, channel: str):
        """Clear message queue for a channel"""
        if channel in self.message_queue:
            self.message_queue[channel].clear()
    
    def save_messages(self, filepath: str):
        """Save message history to JSON"""
        serializable_queue = {}
        for channel, messages in self.message_queue.items():
            serializable_queue[channel] = [
                asdict(msg) if hasattr(msg, '__dataclass_fields__') else msg
                for msg in messages
            ]
        
        with open(filepath, 'w') as f:
            json.dump(serializable_queue, f, indent=2, default=str)
        logger.info(f"💾 Message history saved to {filepath}")