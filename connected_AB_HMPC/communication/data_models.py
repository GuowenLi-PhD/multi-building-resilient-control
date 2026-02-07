"""
Data models for hierarchical multi-building control with flexibility bands

This module defines all data structures for communication between:
- Buildings and Aggregator (flexibility bands, power budgets)
- Simulation components (states, metrics, events)

Author: Guowen Li
Date: 2025-02-06
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import numpy as np


class BuildingStatus(Enum):
    """Building operational status"""
    NORMAL = "normal"
    UNDER_ATTACK = "under_attack"
    RECOVERING = "recovering"
    OFFLINE = "offline"


class ControlMode(Enum):
    """Control mode for building MPC"""
    NOMINAL = "nominal"
    ADAPTIVE = "adaptive"
    ISOLATION = "isolation"


@dataclass
class FlexibilityBand:
    """
    Flexibility band computed by building via two-pass MPC
    
    Represents the feasible power range a building can operate within
    while satisfying all internal constraints (comfort, equipment, etc.)
    """
    building_id: str
    timestamp: float                # Current time [s]
    time_horizon: List[float]       # Future time points [s]
    P_lower_kW: List[float]         # Min power trajectory from min-MPC
    P_upper_kW: List[float]         # Max power trajectory from max-MPC
    baseline_P_kW: List[float]      # Nominal forecast (optional)
    computation_time_s: float       # Time to compute both passes
    feasible: bool                  # Were both MPC passes feasible?
    
    def validate(self) -> bool:
        """Check consistency and validity of flexibility band"""
        if len(self.P_lower_kW) != len(self.P_upper_kW):
            return False
        if len(self.time_horizon) != len(self.P_lower_kW):
            return False
        # Check P_lower <= P_upper at all time points
        for p_low, p_up in zip(self.P_lower_kW, self.P_upper_kW):
            if p_low > p_up + 1e-3:  # Small tolerance for numerical errors
                return False
        return True
    
    def get_width_kW(self) -> List[float]:
        """Return flexibility width at each time point"""
        return [p_up - p_low for p_low, p_up in zip(self.P_lower_kW, self.P_upper_kW)]


@dataclass
class PowerBudget:
    """
    Power budget allocated by aggregator to a building
    
    This is the reference trajectory P_i,ref that the building should track
    via soft constraint in its local MPC
    """
    building_id: str
    timestamp: float
    time_horizon: List[float]       # Future time points [s]
    P_ref_kW: List[float]           # Allocated power reference trajectory
    P_limit_kW: float               # Hard upper limit (safety)
    priority_level: int             # Priority weight used in allocation
    
    def __post_init__(self):
        """Validate budget"""
        assert len(self.time_horizon) == len(self.P_ref_kW), \
            "Time horizon and power reference must have same length"
        assert all(p >= 0 for p in self.P_ref_kW), \
            "Power references must be non-negative"


@dataclass
class BuildingAllocation:
    """
    Complete allocation result from aggregator for all buildings
    """
    timestamp: float
    budgets: List[PowerBudget]      # One per building
    total_power_kW: List[float]     # Total community power trajectory
    feeder_limit_kW: List[float]    # Feeder constraint trajectory
    objective_value: float          # Log-utility objective value
    solve_time_s: float             # Aggregator solve time
    feasible: bool                  # Did aggregator find solution?
    
    def get_budget(self, building_id: str) -> Optional[PowerBudget]:
        """Get budget for specific building"""
        for budget in self.budgets:
            if budget.building_id == building_id:
                return budget
        return None


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
    comfort_violations_degCh: float      # Cumulative degree-hours
    
    # Building-specific data
    extra_data: Dict = field(default_factory=dict)


@dataclass
class BuildingAState(BuildingState):
    """Building A specific state (no TES)"""
    def __post_init__(self):
        if not self.extra_data:
            self.extra_data = {
                'core_zone_airflow_m3s': 0.0,
                'attack_detected': False,
                'compromised_device': None
            }


@dataclass
class BuildingBState(BuildingState):
    """Building B specific state (with TES)"""
    def __post_init__(self):
        if not self.extra_data:
            self.extra_data = {
                'SOC_current': 0.5,
                'SOC_forecast': [],
                'TES_mode': 0,  # -1: charge, 0: off, 1: discharge
                'flexibility_up_kW': 0.0,
                'flexibility_down_kW': 0.0
            }


@dataclass
class MPCResult:
    """Result from building MPC solve"""
    building_id: str
    timestamp: float
    control_inputs: Dict            # {variable_name: value}
    power_trajectory: List[float]   # Predicted power [kW]
    budget_slack: List[float]       # Budget violation μ [kW]
    objective_value: float
    solve_time_s: float
    feasible: bool
    
    def get_max_budget_violation(self) -> float:
        """Return maximum budget violation"""
        return max(self.budget_slack) if self.budget_slack else 0.0


@dataclass
class TwoPassMPCResult:
    """Result from two-pass MPC (min + max power)"""
    building_id: str
    timestamp: float
    min_power_trajectory: List[float]
    max_power_trajectory: List[float]
    min_power_solve_time_s: float
    max_power_solve_time_s: float
    both_feasible: bool
    
    def to_flexibility_band(self, 
                           time_horizon: List[float],
                           baseline: Optional[List[float]] = None) -> FlexibilityBand:
        """Convert to FlexibilityBand dataclass"""
        return FlexibilityBand(
            building_id=self.building_id,
            timestamp=self.timestamp,
            time_horizon=time_horizon,
            P_lower_kW=self.min_power_trajectory,
            P_upper_kW=self.max_power_trajectory,
            baseline_P_kW=baseline if baseline else self.min_power_trajectory,
            computation_time_s=self.min_power_solve_time_s + self.max_power_solve_time_s,
            feasible=self.both_feasible
        )


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
    detection_time: Optional[float] = None


@dataclass
class SimulationMetrics:
    """Performance metrics for entire simulation"""
    total_energy_cost: float
    total_comfort_violations: float
    feeder_violations_count: int
    average_feeder_utilization: float
    max_feeder_utilization: float
    
    building_metrics: Dict[str, Dict]  # {building_id: metrics}
    
    computation_times: Dict[str, List[float]] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Generate summary string"""
        return f"""
Simulation Metrics Summary:
===========================
Total Energy Cost: ${self.total_energy_cost:.2f}
Total Comfort Violations: {self.total_comfort_violations:.2f} °C·h
Feeder Violations: {self.feeder_violations_count}
Avg Feeder Utilization: {self.average_feeder_utilization:.1f}%
Max Feeder Utilization: {self.max_feeder_utilization:.1f}%
"""
