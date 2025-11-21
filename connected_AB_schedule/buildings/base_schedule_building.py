"""
Base class for schedule-based building controllers

Author: Guowen Li, AI Assistant
Date: 2025-01-20
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import logging

from schedule.control_models import DailySchedule, AttackEvent
from schedule.schedule_manager import ScheduleManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseScheduleBuilding(ABC):
    """Abstract base for schedule-based building control"""
    
    def __init__(self, building_id: str, config: Dict, daily_schedule: DailySchedule):
        """
        Initialize building controller
        
        Parameters:
        -----------
        building_id : str
            Building identifier
        config : Dict
            System configuration
        daily_schedule : DailySchedule
            Daily control schedule
        """
        self.building_id = building_id
        self.config = config
        self.daily_schedule = daily_schedule
        
        # Initialize schedule manager
        self.schedule_manager = None  # Set in initialize()
        
        # Current state
        self.current_time = 0
        self.current_state = None
        
        # Active attacks
        self.active_attacks: List[AttackEvent] = []
        
        # Control interval - can be overridden by schedule
        self.default_control_interval = None
        
    @abstractmethod
    def initialize(self, initial_conditions: Dict):
        """Initialize building controller and FMU"""
        pass
    
    @abstractmethod
    def apply_schedule(self, current_time: float) -> Dict[str, float]:
        """
        Get scheduled control actions for current time
        
        Returns:
        --------
        Dict of scheduled variables (may be empty)
        """
        pass
    
    @abstractmethod
    def optimize_unscheduled(self, scheduled_vars: Dict[str, float]) -> Dict[str, float]:
        """
        MPC optimizes variables not in schedule
        
        Parameters:
        -----------
        scheduled_vars : Dict
            Variables with scheduled values (hard constraints)
        
        Returns:
        --------
        Dict of optimized variables
        """
        pass
    
    @abstractmethod
    def apply_attacks(self, control_vars: Dict[str, float]) -> Dict[str, float]:
        """
        Apply active attacks to control variables
        
        Parameters:
        -----------
        control_vars : Dict
            Control variables before attack
        
        Returns:
        --------
        Dict with attacked variables modified
        """
        pass
    
    @abstractmethod
    def step(self, dt: float, active_attacks: List[AttackEvent]) -> Dict:
        """
        Execute one control step
        
        Parameters:
        -----------
        dt : float
            Timestep duration (seconds)
        active_attacks : List[AttackEvent]
            Currently active attacks
        
        Returns:
        --------
        BuildingState dict with current status
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
    
    def get_control_interval(self) -> float:
        """Get effective control interval (schedule overrides default)"""
        if self.schedule_manager:
            schedule_interval = self.schedule_manager.get_control_interval()
            # Use minimum of schedule and default
            if self.default_control_interval:
                return min(schedule_interval, self.default_control_interval)
            return schedule_interval
        return self.default_control_interval
