"""
Aggregator MPC - Upper-level coordinator for multi-building resilient control

Author: Guowen Li, AI Assistant
Date: 2025-01-07
"""

import casadi as ca
import numpy as np
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AggregatorMPCConfig:
    """Configuration for aggregator MPC"""
    PH: int = 12  # Prediction horizon (steps)
    dt: float = 900.0  # Timestep (seconds)
    
    # Feeder constraints
    P_feeder_limit_kW: float = 50.0
    safety_margin: float = 0.9
    
    # Weights
    w_feeder: float = 100.0
    w_comfort: float = 50.0
    w_balance: float = 10.0
    w_TES: float = 5.0
    
    # Building parameters
    P_A_baseline_kW: float = 8.0
    P_B_baseline_kW: float = 12.0

class AggregatorMPC:
    """Upper-level coordinator MPC"""
    
    def __init__(self, config: AggregatorMPCConfig):
        self.config = config
        self.PH = config.PH
        self.dt = config.dt
        
        # Effective feeder limit
        self.P_feeder_max = config.P_feeder_limit_kW * config.safety_margin
        
        logger.info(f"✓ Aggregator MPC initialized: PH={self.PH}, dt={self.dt/60:.0f}min, P_limit={self.P_feeder_max:.1f}kW")
    
    def optimize(self, 
                 current_time: float,
                 P_A_forecast: List[float],
                 P_B_forecast: List[float],
                 SOC_B_forecast: List[float],
                 attack_flag: bool,
                 attack_anticipated: bool) -> Dict:
        """
        Solve aggregator optimization problem
        
        Parameters:
        -----------
        current_time : float
            Current simulation time
        P_A_forecast : List[float]
            Building A power forecast [kW] over PH
        P_B_forecast : List[float]
            Building B power forecast [kW] over PH
        SOC_B_forecast : List[float]
            Building B SOC forecast over PH
        attack_flag : bool
            Is attack currently active on Building A?
        attack_anticipated : bool
            Is attack anticipated soon?
        
        Returns:
        --------
        Dict with:
            - P_A_ref: Power reference for Building A [kW]
            - P_B_ref: Power reference for Building B [kW]
            - SOC_B_target: Target SOC for Building B
            - priority_A: Control priority for Building A
            - priority_B: Control priority for Building B
        """
        
        ## Decision variables
        # P_A_ref[k], P_B_ref[k] for k=0 to PH-1
        U = ca.MX.sym('U', 2 * self.PH)
        
        ## Objective function
        obj = 0
        
        for k in range(self.PH):
            # Extract decision variables
            P_A_ref_k = U[2*k]
            P_B_ref_k = U[2*k + 1]
            
            # Total power
            P_total_k = P_A_forecast[k] + P_B_forecast[k]
            
            # Feeder tracking penalty
            feeder_penalty = (P_total_k - self.P_feeder_max)**2
            
            # Power balance penalty (minimize deviations from references)
            balance_penalty = (P_A_forecast[k] - P_A_ref_k)**2 + (P_B_forecast[k] - P_B_ref_k)**2
            
            # TES utilization penalty (encourage use during attack)
            if attack_flag or attack_anticipated:
                TES_penalty = (1.0 - SOC_B_forecast[min(k, len(SOC_B_forecast)-1)])**2
            else:
                TES_penalty = 0.0
            
            # Weighted objective
            obj += (self.config.w_feeder * feeder_penalty + 
                   self.config.w_balance * balance_penalty +
                   self.config.w_TES * TES_penalty)
        
        ## Constraints
        g = []
        lbg = []
        ubg = []
        
        for k in range(self.PH):
            P_A_ref_k = U[2*k]
            P_B_ref_k = U[2*k + 1]
            
            # Feeder capacity constraint
            P_total_k = P_A_ref_k + P_B_ref_k
            g.append(P_total_k)
            lbg.append(0.0)
            ubg.append(self.P_feeder_max)
            
            # Individual building constraints
            g.append(P_A_ref_k)
            lbg.append(0.0)
            ubg.append(self.P_feeder_max * 0.6)  # Building A max 60% of feeder
            
            g.append(P_B_ref_k)
            lbg.append(0.0)
            ubg.append(self.P_feeder_max * 0.6)  # Building B max 60% of feeder
        
        ## Variable bounds
        u_lb = [0.0, 0.0] * self.PH
        u_ub = [self.P_feeder_max * 0.6, self.P_feeder_max * 0.6] * self.PH
        
        ## Initial guess
        if attack_flag:
            # Under attack: Allow Building A more power, reduce Building B
            u_init = [self.config.P_A_baseline_kW * 1.3, self.config.P_B_baseline_kW * 0.7] * self.PH
        else:
            # Normal: Balanced allocation
            u_init = [self.config.P_A_baseline_kW, self.config.P_B_baseline_kW] * self.PH
        
        ## Solve
        nlp = {'x': U, 'f': obj, 'g': ca.vertcat(*g)}
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 100,
            'ipopt.tol': 1e-3
        }
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        res = solver(x0=u_init, lbx=u_lb, ubx=u_ub, lbg=lbg, ubg=ubg)
        
        ## Extract solution
        u_opt = res['x'].full().flatten()
        P_A_ref_opt = [u_opt[2*k] for k in range(self.PH)]
        P_B_ref_opt = [u_opt[2*k+1] for k in range(self.PH)]
        
        ## Determine SOC target for Building B
        if attack_anticipated:
            SOC_target = 0.90  # Pre-charge before attack
        elif attack_flag:
            SOC_target = 0.30  # Allow discharge during attack
        else:
            SOC_target = 0.60  # Maintain reserve
        
        ## Determine priorities
        if attack_flag:
            priority_A = "comfort"  # Building A focuses on comfort
            priority_B = "support_A"  # Building B supports Building A
        elif attack_anticipated:
            priority_A = "comfort"
            priority_B = "precharge"  # Building B pre-charges TES
        else:
            priority_A = "balanced"
            priority_B = "balanced"
        
        result = {
            'P_A_ref': P_A_ref_opt,
            'P_B_ref': P_B_ref_opt,
            'SOC_B_target': SOC_target,
            'priority_A': priority_A,
            'priority_B': priority_B,
            'feeder_utilization': (P_A_ref_opt[0] + P_B_ref_opt[0]) / self.P_feeder_max
        }
        
        logger.debug(f"🎯 Aggregator solution: P_A={P_A_ref_opt[0]:.2f}kW, P_B={P_B_ref_opt[0]:.2f}kW, SOC_target={SOC_target:.2f}")
        
        return result