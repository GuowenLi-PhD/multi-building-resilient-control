"""
Data models for inter-component communication

Author: Guowen Li, AI Assistant
Date: 2025-10-07
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import numpy as np

class BuildingStatus(Enum):
    NORMAL = "normal"
    UNDER_ATTACK = "under_attack"
    RECOVERING = "recovering"
    OFFLINE = "offline"

class ControlMode(Enum):
    NOMINAL = "nominal"
    ADAPTIVE = "adaptive"
    ISOLATION = "isolation"

@dataclass
class BuildingState:
    """Current state of a building"""
    building_id: str
    timestamp: float
    status: BuildingStatus
    control_mode: ControlMode
    
    # Power
    power_actual_kW: float
    power_forecast_kW: List[float]
    
    # Thermal comfort
    zone_temperatures: Dict[str, float]  # {zone_name: temp_C}
    comfort_violations: float  # degree-hours
    
    # Building-specific
    extra_data: Dict = field(default_factory=dict)

@dataclass
class BuildingAState(BuildingState):
    """Building A specific state"""
    def __post_init__(self):
        self.extra_data = {
            'core_zone_airflow': 0.0,  # m3/s
            'adjacent_zones_airflow': {},
            'attack_detected': False,
            'compromised_device': None
        }

@dataclass
class BuildingBState(BuildingState):
    """Building B specific state"""
    def __post_init__(self):
        self.extra_data = {
            'SOC_current': 0.5,
            'SOC_forecast': [],
            'TES_mode': 0,  # -1: charge, 0: off, 1: discharge, 2: chiller
            'flexibility_up_kW': 0.0,
            'flexibility_down_kW': 0.0,
            'TES_available_energy_kWh': 0.0
        }

@dataclass
class AggregatorCommand:
    """Commands from aggregator to buildings"""
    timestamp: float
    building_id: str
    
    # Power allocation
    power_reference_kW: List[float]  # Reference power trajectory
    power_limit_kW: float            # Hard upper limit
    
    # Attack status
    attack_flag: bool
    attack_anticipated: bool
    anticipation_horizon_hours: float
    
    # Building-specific guidance
    guidance: Dict = field(default_factory=dict)

@dataclass
class AggregatorCommandBuildingB(AggregatorCommand):
    """Building B specific commands"""
    def __post_init__(self):
        self.guidance = {
            'SOC_target': 0.5,
            'precharge_recommended': False,
            'discharge_requested': False,
            'priority': 'comfort'  # 'comfort', 'energy', 'support_A'
        }

@dataclass
class FeederStatus:
    """Electrical feeder status"""
    timestamp: float
    total_power_kW: float
    capacity_kW: float
    utilization_percent: float
    voltage_pu: float
    constraint_violated: bool
    margin_kW: float

@dataclass
class AttackEvent:
    """Cyber-attack event information"""
    attack_id: str
    target_building: str
    attack_type: str
    affected_component: str
    start_time: float
    end_time: Optional[float]
    severity: str  # 'low', 'medium', 'high'
    detected: bool
    detection_time: Optional[float]