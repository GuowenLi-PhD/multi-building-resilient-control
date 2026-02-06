"""
Communication messages for the hierarchical MPC framework.

Defines typed dataclasses for every signal exchanged between the aggregator
and the two building-level controllers (upward and downward).

Author  : Guowen Li
Date    : 2026-02
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
#  Upward messages  (Buildings → Aggregator)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BuildingAStatus:
    """Status report sent from Building A to the aggregator every Δt."""

    timestamp: float                            # current simulation time [s]
    P_A_current_kW: float                       # measured total power [kW]
    Tz: np.ndarray                              # zone temperatures (n_zones,) [°C]
    alpha_attack: float                         # attack-detection confidence [0,1]
    P_A_flex_min: np.ndarray                    # min feasible power (agg_PH,) [kW]
    P_A_flex_max: np.ndarray                    # max feasible power (agg_PH,) [kW]
    u_current: np.ndarray                       # current control vector (n_inputs,)
    comfort_violations: np.ndarray              # per-zone violation [K] (n_zones,)
    mode: str = "NOMINAL"                       # NOMINAL | COMFORT_PRIORITY | RECOVERY


@dataclass
class BuildingBStatus:
    """Status report sent from Building B to the aggregator every Δt."""

    timestamp: float
    P_B_current_kW: float                       # measured total power [kW]
    SOC: float                                  # TES state-of-charge [0,1]
    mode_current: int                           # current TES mode {-1,0,1,2}
    P_flex_up: np.ndarray                       # max power trajectory (agg_PH,) [kW]
    P_flex_down: np.ndarray                     # min power trajectory (agg_PH,) [kW]
    Tz: Optional[np.ndarray] = None             # zone temps if available
    comfort_violations: Optional[np.ndarray] = None


# ═══════════════════════════════════════════════════════════════════════════════
#  Downward commands  (Aggregator → Buildings)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CommandToA:
    """Aggregator command to Building A."""

    P_A_alloc: np.ndarray                       # allocated power budget (PH,) [kW]
    theta_priority: int                         # urgency level {0,1,2,3}
    recovery_horizon: float = 0.0               # estimated recovery time [s]


@dataclass
class CommandToB:
    """Aggregator command to Building B."""

    P_B_alloc: np.ndarray                       # allocated power budget (PH,) [kW]
    m_B_suggested: np.ndarray                   # suggested TES modes (PH,) ∈ {-1,0,1,2}
    delta_P_support: np.ndarray                 # requested load reduction (PH,) [kW]
    support_flag: bool = False                  # whether Building A needs help


# ═══════════════════════════════════════════════════════════════════════════════
#  Aggregator allocation result (internal)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AllocationResult:
    """Internal result container from the aggregator's allocation step."""

    P_A_alloc: np.ndarray
    P_B_alloc: np.ndarray
    m_B_suggested: np.ndarray
    support_flag: bool
    delta_P_support: np.ndarray
    mode: str = "NORMAL"                         # NORMAL | ATTACK_OPTIMIZE
    theta_priority: int = 0
    solve_time_s: float = 0.0
    status: str = "ok"
