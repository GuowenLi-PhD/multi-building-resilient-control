"""
Data Models for Hierarchical Multi-Building Control

Defines the communication protocol between:
  - Buildings → Aggregator:  FlexibilityReport  (measurements + flexibility bands + priority)
  - Aggregator → Buildings:  PowerBudget         (allocated power reference trajectory)

Author: Guowen Li, AI Assistant
Date: 2025-02
"""

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, List, Optional
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────
class BuildingStatus(Enum):
    NORMAL       = "normal"
    UNDER_ATTACK = "under_attack"
    DEGRADED     = "degraded"

class ControlMode(Enum):
    NOMINAL  = "nominal"
    ADAPTIVE = "adaptive"       # Building A under cyber-attack
    SUPPORT  = "support"        # Building B providing grid support

class EnergyPriority(IntEnum):
    """Priority weights for aggregator allocation (Eq. 29)"""
    LOW    = 1
    MEDIUM = 2
    HIGH   = 3


# ──────────────────────────────────────────────────────────────────────────────
# Building → Aggregator
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class FlexibilityBand:
    """
    Power flexibility band over the aggregator prediction horizon (Eqs. 27-28).
    
    Computed via two-pass local MPC:
      P_lower[k] = min-power MPC solve at step k  (lower-bound trajectory)
      P_upper[k] = max-power MPC solve at step k  (upper-bound trajectory)
    
    Units: kW.  Length = aggregator prediction horizon.
    """
    P_lower_kW: List[float]     # P̲_i(t+k), k = 0..PH_agg-1
    P_upper_kW: List[float]     # P̄_i(t+k), k = 0..PH_agg-1

    def __post_init__(self):
        assert len(self.P_lower_kW) == len(self.P_upper_kW), \
            "Upper/lower band lengths must match"
        for k, (lo, hi) in enumerate(zip(self.P_lower_kW, self.P_upper_kW)):
            if lo > hi + 1e-3:
                raise ValueError(
                    f"Flexibility band inverted at step {k}: "
                    f"P_lower={lo:.2f} > P_upper={hi:.2f}"
                )

    @property
    def horizon(self) -> int:
        return len(self.P_lower_kW)


@dataclass
class FlexibilityReport:
    """
    Information packet sent from one building to the aggregator at each
    coordination interval.
    """
    building_id: str
    timestamp: float                        # simulation time [s]
    power_actual_kW: float                  # current measured power
    flexibility_band: FlexibilityBand       # predicted flexibility over PH
    priority: EnergyPriority                # energy priority weight ω_i
    status: BuildingStatus = BuildingStatus.NORMAL
    control_mode: ControlMode = ControlMode.NOMINAL
    zone_temperatures: Dict[str, float] = field(default_factory=dict)
    comfort_violation_Kh: float = 0.0
    extra_data: Dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Aggregator → Building
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class PowerBudget:
    """
    Allocated power budget trajectory for a single building.
    
    The local MPC enforces: P_i(t+k) ≤ P_ref_kW[k] + μ_i(t+k)
    with soft-constraint penalty ω_budget · (μ_i / μ̄_i)² in the local objective.
    
    Units: kW.  Length = aggregator prediction horizon.
    """
    building_id: str
    P_ref_kW: List[float]
    timestamp: float

    @property
    def horizon(self) -> int:
        return len(self.P_ref_kW)

    def get_budget_at_step(self, step: int) -> float:
        """Get budget for a specific step, clamped to last value if beyond horizon."""
        if step < len(self.P_ref_kW):
            return self.P_ref_kW[step]
        return self.P_ref_kW[-1]


@dataclass
class AggregatorDecision:
    """Complete aggregator output for all buildings."""
    timestamp: float
    budgets: Dict[str, PowerBudget]
    feeder_limit_kW: float
    total_allocated_kW: List[float]
    solver_status: str = "optimal"
    objective_value: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Simulation logging
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class StepLog:
    """Record of one simulation step for post-processing."""
    timestamp: float
    building_id: str
    power_actual_kW: float
    power_budget_kW: float
    budget_violation_kW: float
    zone_temperatures: Dict[str, float]
    comfort_violation_Kh: float
    control_mode: str
    extra: Dict = field(default_factory=dict)


@dataclass
class AggregatorLog:
    """Record of one aggregator decision for post-processing."""
    timestamp: float
    feeder_limit_kW: float
    total_power_kW: float
    total_allocated_kW: float
    budgets: Dict[str, float]
    flex_bands: Dict[str, tuple]
    priorities: Dict[str, int]
    solver_status: str
    objective_value: float
