"""
Abstract base class for building interfaces

Author: Guowen Li, AI Assistant
Date: 2025-01-07
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../buildingA_wo_TES'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../buildingB_w_TES'))

from ..communication.data_models import BuildingState, AggregatorCommand

class BaseBuilding(ABC):
    """Abstract base class for building controllers"""
    
    def __init__(self, building_id: str, config: Dict):
        self.building_id = building_id
        self.config = config
        self.current_state = None
        self.mpc_controller = None
    
    @abstractmethod
    def initialize(self, initial_conditions: Dict):
        """Initialize building controller and FMU"""
        pass
    
    @abstractmethod
    def step(self, aggregator_command: Optional[AggregatorCommand], dt: float) -> BuildingState:
        """
        Execute one control step
        
        Parameters:
        -----------
        aggregator_command : AggregatorCommand or None
            Commands from aggregator
        dt : float
            Timestep duration (seconds)
        
        Returns:
        --------
        BuildingState with current status
        """
        pass
    
    @abstractmethod
    def get_power_forecast(self, horizon: int) -> List[float]:
        """Get power consumption forecast"""
        pass
    
    @abstractmethod
    def shutdown(self):
        """Clean shutdown"""
        pass