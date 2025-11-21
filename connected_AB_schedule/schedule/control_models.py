"""
Data models for schedule-based control

Author: Guowen Li, AI Assistant
Date: 2025-01-20
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

class BuildingType(Enum):
    """Building types"""
    BUILDING_A = "Building_A"
    BUILDING_B = "Building_B"

@dataclass
class ControlAction:
    """Single control action at one time interval"""
    time_of_day: str  # "14:00" format
    building_id: str
    scheduled_vars: Dict[str, float]  # {bcp: 1, bahu: 1, ...}
    
    def __post_init__(self):
        """Validate time format"""
        parts = self.time_of_day.split(':')
        if len(parts) != 2:
            raise ValueError(f"Invalid time format: {self.time_of_day}, expected HH:MM")
        hour, minute = int(parts[0]), int(parts[1])
        if not (0 <= hour < 24 and 0 <= minute < 60):
            raise ValueError(f"Invalid time: {self.time_of_day}")

@dataclass
class DailySchedule:
    """Complete daily schedule for one building"""
    building_id: str
    control_interval_seconds: float  # Control interval in seconds
    actions: List[ControlAction] = field(default_factory=list)
    
    def get_action_at_time(self, time_of_day_seconds: float) -> Optional[ControlAction]:
        """Get control action for specific time of day"""
        # Convert time to HH:MM format
        hours = int(time_of_day_seconds // 3600)
        minutes = int((time_of_day_seconds % 3600) // 60)
        time_str = f"{hours:02d}:{minutes:02d}"
        
        # Find exact match or closest previous action
        for action in sorted(self.actions, key=lambda a: self._time_to_seconds(a.time_of_day)):
            if self._time_to_seconds(action.time_of_day) <= time_of_day_seconds:
                closest_action = action
            else:
                break
        
        return closest_action if 'closest_action' in locals() else None
    
    @staticmethod
    def _time_to_seconds(time_str: str) -> float:
        """Convert HH:MM to seconds since midnight"""
        parts = time_str.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60

@dataclass
class BuildingAVariables:
    """Building A decision variables"""
    # All 11 variables for Building A
    bcp: Optional[float] = None      # Chiller on/off (0 or 1)
    bahu: Optional[float] = None     # AHU on/off (0 or 1)
    Tchw: Optional[float] = None     # Chilled water temp setpoint (°C)
    Tcw: Optional[float] = None      # Condenser water temp setpoint (°C)
    Tsa: Optional[float] = None      # Supply air temp setpoint (°C)
    Vcore: Optional[float] = None    # Core zone VAV damper position
    Veast: Optional[float] = None    # East zone VAV damper position
    Vnorth: Optional[float] = None   # North zone VAV damper position
    Vsouth: Optional[float] = None   # South zone VAV damper position
    Vwest: Optional[float] = None    # West zone VAV damper position
    epsilon: Optional[float] = None  # Slack variable
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'BuildingAVariables':
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary (only non-None values)"""
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    def get_scheduled_vars(self) -> Dict[str, float]:
        """Get only scheduled (non-None) variables"""
        return self.to_dict()
    
    def get_unscheduled_vars(self) -> List[str]:
        """Get names of unscheduled (None) variables"""
        return [k for k, v in self.__dict__.items() if v is None]

@dataclass
class BuildingBVariables:
    """Building B decision variables"""
    uMod: Optional[int] = None  # TES mode: -1 (Charge), 0 (Off), 1 (Discharge), 2 (Chiller only)
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'BuildingBVariables':
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary (only non-None values)"""
        return {k: v for k, v in self.__dict__.items() if v is not None}

@dataclass
class AttackEvent:
    """Attack event definition"""
    name: str
    target_building: str  # "Building_A" or "Building_B"
    start_day: float      # Day of year
    start_hour: float     # Hour of day
    duration_hours: float
    affected_variables: List[str]  # Empty list means no attack
    attack_type: str      # "vav_reinitialization", "setpoint_manipulation", etc.
    attack_params: Dict = field(default_factory=dict)
    
    def is_active(self, current_time: float, simulation_start: float) -> bool:
        """Check if attack is active at current time"""
        sim_hours = (current_time - simulation_start) / 3600
        attack_start = (self.start_day * 24 + self.start_hour)
        attack_end = attack_start + self.duration_hours
        
        return attack_start <= sim_hours < attack_end

@dataclass
class SimulationConfig:
    """Complete simulation configuration"""
    scenario_name: str
    start_day: int
    duration_days: int
    building_a_schedule: DailySchedule
    building_b_schedule: DailySchedule
    attack_events: List[AttackEvent] = field(default_factory=list)
    feeder_capacity_kW: float = 50.0
    feeder_safety_margin: float = 0.9
