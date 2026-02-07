"""
Log-Utility Aggregator for N-Building Hierarchical Control

Implements the convex log-utility allocation formulation:

Objective: Minimize -Σ_i Σ_k ω_i · log(P_i,ref^k + δ)

Subject to:
    - Feeder limit: Σ_i P_i,ref^k ≤ P_feeder^k  ∀k
    - Flexibility bands: P̲_i^k ≤ P_i,ref^k ≤ P̄_i^k  ∀i,k

Key features:
- Smooth proportional allocation based on priority weights
- Fair tie-breaking for equal-priority buildings
- Convex optimization with guaranteed feasibility
- Fixed 2-signal interface per building (scalable to N buildings)

Author: Guowen Li
Date: 2025-02-06
"""

import casadi as ca
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from communication.data_models import FlexibilityBand, PowerBudget, BuildingAllocation

logger = logging.getLogger(__name__)


@dataclass
class LogUtilityAggregatorConfig:
    """Configuration for log-utility aggregator"""
    PH: int = 20                     # Prediction horizon (steps)
    dt: float = 3600.0               # Timestep (seconds) - 1 hour
    P_feeder_limit_kW: float = 50.0  # Feeder capacity
    safety_margin: float = 0.95      # Use 95% of capacity
    delta: float = 0.001             # Small constant to prevent log(0)
    
    # Default priority weights for buildings
    default_priority_weights: Dict[str, int] = None
    
    # Solver options
    solver_max_iter: int = 200
    solver_tol: float = 1e-4
    solver_print_level: int = 0
    
    def __post_init__(self):
        if self.default_priority_weights is None:
            # Default: Building A (victim, no TES) = 1, Building B (with TES) = 2
            self.default_priority_weights = {
                'Building_A': 1,
                'Building_B': 2
            }


class LogUtilityAggregator:
    """
    N-building aggregator using log-utility allocation
    
    This is the upper-level coordinator in the hierarchical framework.
    It allocates power budgets to buildings to maximize overall utility
    while respecting feeder constraints and building flexibility limits.
    
    Design Philosophy:
    - Aggregator only sees flexibility bands (P_lower, P_upper) from buildings
    - No knowledge of internal building dynamics required
    - Scalable: Adding new building requires zero changes to existing buildings
    - Convex optimization ensures reliable real-time performance
    """
    
    def __init__(self, config: LogUtilityAggregatorConfig):
        self.config = config
        self.PH = config.PH
        self.dt = config.dt
        self.P_feeder_max = config.P_feeder_limit_kW * config.safety_margin
        self.delta = config.delta
        
        # Track registered buildings
        self.buildings: Dict[str, int] = {}  # {building_id: priority_weight}
        
        logger.info("="*80)
        logger.info("🎯 LOG-UTILITY AGGREGATOR INITIALIZED")
        logger.info("="*80)
        logger.info(f"  Prediction Horizon: {self.PH} steps @ {self.dt/3600:.1f} hr")
        logger.info(f"  Feeder Limit: {config.P_feeder_limit_kW:.1f} kW "
                   f"(using {config.safety_margin*100:.0f}% = {self.P_feeder_max:.1f} kW)")
        logger.info(f"  Log-utility δ: {self.delta}")
        logger.info(f"  Default Priorities: {config.default_priority_weights}")
        logger.info("="*80)
    
    def register_building(self, building_id: str, priority_weight: int = 1):
        """
        Register a building in the coordination framework
        
        Parameters:
        -----------
        building_id : str
            Unique identifier for building
        priority_weight : int
            Energy priority weight ω_i (higher = more priority)
        """
        self.buildings[building_id] = priority_weight
        logger.info(f"  ✓ Registered: {building_id} with priority ω={priority_weight}")
    
    def allocate_power(self,
                       flexibility_bands: List[FlexibilityBand],
                       feeder_limit: List[float],
                       current_time: float,
                       custom_priorities: Optional[Dict[str, int]] = None) -> BuildingAllocation:
        """
        Solve power allocation optimization problem
        
        This is the core method that implements the log-utility formulation.
        
        Parameters:
        -----------
        flexibility_bands : List[FlexibilityBand]
            Flexibility bands from each building (from two-pass MPC)
        feeder_limit : List[float]
            Time-varying feeder power limit [kW] over horizon
        current_time : float
            Current simulation time [s]
        custom_priorities : Dict[str, int], optional
            Override default priority weights (useful for attack scenarios)
        
        Returns:
        --------
        BuildingAllocation
            Power budgets for all buildings
        """
        
        import time
        solve_start = time.time()
        
        # === INPUT VALIDATION ===
        N = len(flexibility_bands)
        if N == 0:
            logger.error("❌ No flexibility bands provided!")
            return self._create_infeasible_allocation(current_time, feeder_limit)
        
        # Validate all bands
        for band in flexibility_bands:
            if not band.validate():
                logger.error(f"❌ Invalid flexibility band for {band.building_id}")
                logger.error(f"   P_lower={band.P_lower_kW[:3]}")
                logger.error(f"   P_upper={band.P_upper_kW[:3]}")
                return self._create_infeasible_allocation(current_time, feeder_limit)
        
        # Get effective horizon (minimum across all bands and feeder limit)
        effective_PH = min(
            min(len(band.P_lower_kW) for band in flexibility_bands),
            self.PH,
            len(feeder_limit)
        )
        
        if effective_PH == 0:
            logger.error("❌ Effective prediction horizon is 0!")
            return self._create_infeasible_allocation(current_time, feeder_limit)
        
        # Get priority weights
        priorities = custom_priorities if custom_priorities else self.buildings
        omega = [priorities.get(band.building_id, 1) for band in flexibility_bands]
        
        logger.info("")
        logger.info("="*80)
        logger.info(f"🎯 ALLOCATING POWER @ t={current_time/3600/24:.2f} days")
        logger.info("="*80)
        logger.info(f"  Buildings: {N}")
        logger.info(f"  Prediction Horizon: {effective_PH} steps")
        logger.info(f"  Priorities (ω): {dict(zip([b.building_id for b in flexibility_bands], omega))}")
        logger.info("")
        
        # Log flexibility bands
        for i, band in enumerate(flexibility_bands):
            width = band.get_width_kW()[0] if band.get_width_kW() else 0
            logger.info(f"  {band.building_id}:")
            logger.info(f"    P_lower[0] = {band.P_lower_kW[0]:.2f} kW")
            logger.info(f"    P_upper[0] = {band.P_upper_kW[0]:.2f} kW")
            logger.info(f"    Flexibility = {width:.2f} kW")
        
        logger.info("")
        logger.info(f"  Feeder limit[0] = {feeder_limit[0]:.2f} kW")
        logger.info("")
        
        # === FORMULATE OPTIMIZATION PROBLEM ===
        
        # Decision variables: P_ref for each building over horizon
        # U = [P_1[0], P_2[0], ..., P_N[0],
        #      P_1[1], P_2[1], ..., P_N[1],
        #      ...
        #      P_1[PH-1], ..., P_N[PH-1]]
        U = ca.MX.sym('U', N * effective_PH)
        
        # Objective: -Σ_i Σ_k ω_i · log(P_i,ref^k + δ)
        obj = 0
        for k in range(effective_PH):
            for i in range(N):
                P_ref_ik = U[k * N + i]
                obj -= omega[i] * ca.log(P_ref_ik + self.delta)
        
        # Constraints
        g = []
        lbg = []
        ubg = []
        
        for k in range(effective_PH):
            # 1. Feeder limit: Σ_i P_i,ref^k ≤ P_feeder^k
            P_total_k = sum(U[k * N + i] for i in range(N))
            g.append(P_total_k)
            lbg.append(0.0)
            ubg.append(feeder_limit[k])
            
            # 2. Flexibility bands: P̲_i^k ≤ P_i,ref^k ≤ P̄_i^k
            for i in range(N):
                P_ref_ik = U[k * N + i]
                g.append(P_ref_ik)
                lbg.append(flexibility_bands[i].P_lower_kW[k])
                ubg.append(flexibility_bands[i].P_upper_kW[k])
        
        # Variable bounds (redundant with constraints, but helps solver)
        u_lb = []
        u_ub = []
        for k in range(effective_PH):
            for i in range(N):
                u_lb.append(max(0.0, flexibility_bands[i].P_lower_kW[k]))
                u_ub.append(flexibility_bands[i].P_upper_kW[k])
        
        # Initial guess: Proportional allocation based on priorities
        u_init = self._compute_initial_guess(
            flexibility_bands, feeder_limit, omega, N, effective_PH
        )
        
        # === SOLVE ===
        nlp = {'x': U, 'f': obj, 'g': ca.vertcat(*g)}
        opts = {
            'ipopt.print_level': self.config.solver_print_level,
            'print_time': 0,
            'ipopt.max_iter': self.config.solver_max_iter,
            'ipopt.tol': self.config.solver_tol,
            'ipopt.acceptable_tol': self.config.solver_tol * 10,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.mu_strategy': 'adaptive'
        }
        
        try:
            solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
            res = solver(x0=u_init, lbx=u_lb, ubx=u_ub, lbg=lbg, ubg=ubg)
        except Exception as e:
            logger.error(f"❌ Solver exception: {e}")
            return self._create_infeasible_allocation(current_time, feeder_limit)
        
        solve_time = time.time() - solve_start
        
        # === EXTRACT SOLUTION ===
        success = solver.stats()['success']
        
        if success:
            u_opt = res['x'].full().flatten()
            
            # Create power budgets for each building
            budgets = []
            for i, band in enumerate(flexibility_bands):
                P_ref_trajectory = [u_opt[k * N + i] for k in range(effective_PH)]
                
                budget = PowerBudget(
                    building_id=band.building_id,
                    timestamp=current_time,
                    time_horizon=[current_time + k * self.dt for k in range(effective_PH)],
                    P_ref_kW=P_ref_trajectory,
                    P_limit_kW=max(band.P_upper_kW),  # Safety hard limit
                    priority_level=omega[i]
                )
                budgets.append(budget)
            
            # Compute total power
            total_power = [sum(u_opt[k * N + i] for i in range(N)) 
                          for k in range(effective_PH)]
            
            allocation = BuildingAllocation(
                timestamp=current_time,
                budgets=budgets,
                total_power_kW=total_power,
                feeder_limit_kW=feeder_limit[:effective_PH],
                objective_value=float(res['f']),
                solve_time_s=solve_time,
                feasible=True
            )
            
            # Log results
            logger.info("✅ ALLOCATION SUCCESSFUL")
            logger.info(f"   Solve time: {solve_time:.3f} s")
            logger.info(f"   Objective value: {float(res['f']):.4f}")
            logger.info("")
            logger.info("   Allocations:")
            for i, budget in enumerate(budgets):
                band = flexibility_bands[i]
                logger.info(f"     {budget.building_id}: P_ref[0] = {budget.P_ref_kW[0]:.2f} kW "
                          f"(band: [{band.P_lower_kW[0]:.2f}, {band.P_upper_kW[0]:.2f}])")
            logger.info("")
            logger.info(f"   Total: {total_power[0]:.2f} kW / {feeder_limit[0]:.2f} kW "
                       f"({100*total_power[0]/feeder_limit[0]:.1f}%)")
            logger.info("="*80)
            logger.info("")
            
            return allocation
            
        else:
            logger.error("❌ ALLOCATION FAILED")
            logger.error(f"   Solver status: {solver.stats()['return_status']}")
            logger.error(f"   Solve time: {solve_time:.3f} s")
            logger.error("="*80)
            logger.error("")
            
            return self._create_infeasible_allocation(current_time, feeder_limit)
    
    def _compute_initial_guess(self, 
                              flexibility_bands: List[FlexibilityBand],
                              feeder_limit: List[float],
                              omega: List[int],
                              N: int,
                              PH: int) -> List[float]:
        """Compute smart initial guess for optimization"""
        u_init = []
        total_priority = sum(omega)
        
        for k in range(PH):
            # Available power at this timestep
            available_power = min(
                feeder_limit[k],
                sum(band.P_upper_kW[k] for band in flexibility_bands)
            )
            
            # Allocate proportionally to priorities
            for i in range(N):
                # Proportional share
                proportional = (omega[i] / total_priority) * available_power
                
                # Clamp to flexibility band
                clamped = max(flexibility_bands[i].P_lower_kW[k],
                             min(proportional, flexibility_bands[i].P_upper_kW[k]))
                
                u_init.append(clamped)
        
        return u_init
    
    def _create_infeasible_allocation(self, 
                                      current_time: float,
                                      feeder_limit: List[float]) -> BuildingAllocation:
        """Create infeasible allocation result"""
        return BuildingAllocation(
            timestamp=current_time,
            budgets=[],
            total_power_kW=[],
            feeder_limit_kW=feeder_limit,
            objective_value=np.inf,
            solve_time_s=0.0,
            feasible=False
        )
