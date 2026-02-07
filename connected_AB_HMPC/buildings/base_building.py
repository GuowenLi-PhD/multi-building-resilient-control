"""
Base building class for hierarchical control with flexibility bands

Defines the interface that all buildings must implement to participate
in the hierarchical control framework.

Key methods:
- compute_flexibility_band(): Two-pass MPC to report (P_lower, P_upper)
- solve_mpc_with_budget(): Local MPC with soft budget constraint

Author: Guowen Li
Date: 2025-02-06
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from communication.data_models import (
    FlexibilityBand,
    BuildingState,
    MPCResult,
    BuildingStatus,
    ControlMode
)

logger = logging.getLogger(__name__)


class BaseBuilding(ABC):
    """
    Abstract base class for buildings in hierarchical control
    
    This class defines the standard interface that enables:
    1. Flexibility band computation (two-pass MPC)
    2. Power budget tracking (soft constraint)
    3. Standard state reporting
    
    The interface is designed to be building-agnostic, allowing
    different building types (HVAC, HVAC+TES, HVAC+Battery, etc.)
    to participate in the same coordination framework.
    """
    
    def __init__(self, building_id: str, config: Dict):
        """
        Initialize building
        
        Parameters:
        -----------
        building_id : str
            Unique identifier
        config : Dict
            Configuration dict (from YAML)
        """
        self.building_id = building_id
        self.config = config
        
        # State tracking
        self.current_state: Optional[BuildingState] = None
        self.flexibility_band: Optional[FlexibilityBand] = None
        self.last_mpc_result: Optional[MPCResult] = None
        
        # Timing
        self._last_band_computation_time = 0.0
        self._last_mpc_solve_time = 0.0
        
        # Control mode
        self.control_mode = ControlMode.NOMINAL
        self.status = BuildingStatus.NORMAL
        
        logger.info(f"✓ {self.building_id} base initialized")
    
    # ========================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # ========================================================================
    
    @abstractmethod
    def compute_flexibility_band(self,
                                 current_time: float,
                                 weather_forecast: Dict,
                                 price_forecast: List[float],
                                 horizon_steps: int) -> FlexibilityBand:
        """
        Compute flexibility band via two-pass MPC
        
        This is the core method for reporting building flexibility to aggregator.
        
        Implementation should:
        1. Solve min-power MPC → P_lower trajectory
        2. Solve max-power MPC → P_upper trajectory  
        3. Validate P_lower ≤ P_upper
        4. Return FlexibilityBand
        
        Parameters:
        -----------
        current_time : float
            Current simulation time [s]
        weather_forecast : Dict
            Weather data: {variable: [values over horizon]}
        price_forecast : List[float]
            Electricity price [$/kWh] over horizon
        horizon_steps : int
            Number of steps to predict
        
        Returns:
        --------
        FlexibilityBand
            Flexibility band with P_lower and P_upper trajectories
        """
        pass
    
    @abstractmethod
    def solve_mpc_with_budget(self,
                              current_time: float,
                              power_budget: List[float],
                              weather_forecast: Dict,
                              price_forecast: List[float]) -> MPCResult:
        """
        Solve MPC with soft power budget constraint
        
        Implementation should add to existing MPC:
            Soft constraint: P^k ≤ P_budget^k + μ^k
            Penalty: ω_budget · Σ (μ^k / μ̄)²
        
        Parameters:
        -----------
        current_time : float
            Current time [s]
        power_budget : List[float]
            Allocated power reference from aggregator [kW]
        weather_forecast : Dict
            Weather data
        price_forecast : List[float]
            Electricity price [$/kWh]
        
        Returns:
        --------
        MPCResult
            MPC solution including budget slack μ
        """
        pass
    
    @abstractmethod
    def get_state(self) -> BuildingState:
        """
        Get current building state
        
        Returns:
        --------
        BuildingState
            Current power, temperatures, status, etc.
        """
        pass
    
    @abstractmethod
    def apply_control(self, control_input: Dict) -> bool:
        """
        Apply control inputs to building/FMU
        
        Parameters:
        -----------
        control_input : Dict
            Control variables and values
        
        Returns:
        --------
        bool
            Success flag
        """
        pass
    
    @abstractmethod
    def step(self, dt: float):
        """
        Step building simulation forward in time
        
        Parameters:
        -----------
        dt : float
            Time step [s]
        """
        pass
    
    @abstractmethod
    def shutdown(self):
        """Clean up resources (close FMU, etc.)"""
        pass
    
    # ========================================================================
    # OPTIONAL METHODS - Can be overridden
    # ========================================================================
    
    def get_flexibility_band_cached(self) -> Optional[FlexibilityBand]:
        """Return most recently computed flexibility band"""
        return self.flexibility_band
    
    def get_last_mpc_result(self) -> Optional[MPCResult]:
        """Return most recent MPC result"""
        return self.last_mpc_result
    
    def set_control_mode(self, mode: ControlMode):
        """Update control mode"""
        self.control_mode = mode
        logger.info(f"{self.building_id}: Control mode → {mode.value}")
    
    def set_status(self, status: BuildingStatus):
        """Update building status"""
        self.status = status
        logger.info(f"{self.building_id}: Status → {status.value}")
    
    def get_timing_metrics(self) -> Dict:
        """Get computation time metrics"""
        return {
            'last_band_computation_time': self._last_band_computation_time,
            'last_mpc_solve_time': self._last_mpc_solve_time
        }
