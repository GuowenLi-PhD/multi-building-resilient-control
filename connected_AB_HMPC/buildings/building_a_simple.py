"""
Simplified Building A implementation for demonstration

This is a streamlined version that demonstrates:
- Two-pass MPC (min/max power)
- Soft budget constraints
- Proper interface implementation

For production use, replace with full FMU-based implementation.

Author: Guowen Li
Date: 2025-02-06
"""

import casadi as ca
import numpy as np
from typing import Dict, List
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buildings.base_building import BaseBuilding
from communication.data_models import (
    FlexibilityBand, BuildingState, MPCResult, TwoPassMPCResult,
    BuildingStatus, ControlMode
)

logger = logging.getLogger(__name__)


class BuildingASimple(BaseBuilding):
    """
    Simplified Building A (traditional HVAC, no TES)
    
    Simplified dynamics for demonstration:
    - Thermal model: dT/dt = -a*(T - T_out) + b*Q_hvac
    - Power model: P = c*Q_hvac
    
    MPC Features:
    - Two-pass MPC for flexibility bands
    - Soft budget constraints
    """
    
    def __init__(self, building_id: str, config: Dict):
        super().__init__(building_id, config)
        
        # Building parameters (simplified but realistic)
        self.a = 1.0 / 14400.0  # Thermal conductance (4 hour time constant)
        self.b = 0.0002         # HVAC effectiveness (realistic cooling rate)
        self.c = 0.8            # Power coefficient [kW/kW_cooling]
        
        # State
        self.T_zone = 22.0  # Current zone temperature [°C]
        self.T_out = 30.0   # Outdoor temperature [°C]
        
        # Constraints
        self.T_min = 21.0
        self.T_max = 24.0
        self.Q_min = 0.0
        self.Q_max = 15.0  # Max cooling [kW]
        
        # MPC parameters
        self.dt = config.get('timing', {}).get('building_a_timestep', 900.0)
        self.PH = 4  # Prediction horizon (4 * 15min = 1 hour)
        
        # Weights
        self.w_energy = 1.0
        self.w_comfort = 100.0
        self.w_budget = 10.0
        self.mu_bar = 5.0  # Normalization for budget slack
        
        logger.info(f"✓ {self.building_id} (Simple) initialized")
    
    def compute_flexibility_band(self, current_time, weather_forecast, 
                                 price_forecast, horizon_steps) -> FlexibilityBand:
        """Compute flexibility band via two-pass MPC"""
        
        logger.info(f"  🔄 {self.building_id}: Computing flexibility band...")
        
        import time
        start = time.time()
        
        # Use building's own prediction horizon
        effective_horizon = min(self.PH, horizon_steps)
        
        # Pass 1: Minimize power
        result_min = self._solve_min_power_mpc(weather_forecast, price_forecast, effective_horizon)
        t_min = time.time() - start
        
        if not result_min['feasible']:
            logger.error(f"    ✗ {self.building_id}: Min-power MPC INFEASIBLE")
        
        # Pass 2: Maximize power
        start = time.time()
        result_max = self._solve_max_power_mpc(weather_forecast, price_forecast, effective_horizon)
        t_max = time.time() - start
        
        if not result_max['feasible']:
            logger.error(f"    ✗ {self.building_id}: Max-power MPC INFEASIBLE")
        
        # Ensure P_lower <= P_upper (fix any numerical issues)
        P_lower = result_min['power']
        P_upper = result_max['power']
        
        # If bounds are reversed, swap them
        for k in range(len(P_lower)):
            if P_lower[k] > P_upper[k]:
                P_lower[k], P_upper[k] = P_upper[k], P_lower[k]
        
        # Create flexibility band
        two_pass = TwoPassMPCResult(
            building_id=self.building_id,
            timestamp=current_time,
            min_power_trajectory=P_lower,
            max_power_trajectory=P_upper,
            min_power_solve_time_s=t_min,
            max_power_solve_time_s=t_max,
            both_feasible=result_min['feasible'] and result_max['feasible']
        )
        
        time_horizon = [current_time + k * self.dt for k in range(effective_horizon)]
        band = two_pass.to_flexibility_band(time_horizon)
        
        # Validate
        if not band.validate():
            logger.error(f"    ✗ Invalid flexibility band for {self.building_id}!")
            logger.error(f"      P_lower = {band.P_lower_kW[:3]}")
            logger.error(f"      P_upper = {band.P_upper_kW[:3]}")
            band.feasible = False
        
        # Cache
        self.flexibility_band = band
        self._last_band_computation_time = t_min + t_max
        
        status = "✓" if band.feasible else "✗"
        logger.info(f"    {status} P ∈ [{band.P_lower_kW[0]:.2f}, {band.P_upper_kW[0]:.2f}] kW "
                   f"({t_min+t_max:.3f}s) [feasible={band.feasible}]")
        
        return band
    
    def solve_mpc_with_budget(self, current_time, power_budget, 
                              weather_forecast, price_forecast) -> MPCResult:
        """Solve MPC with soft budget constraint"""
        
        import time
        start = time.time()
        
        PH = min(self.PH, len(power_budget))
        
        # Get forecast
        T_out_forecast = weather_forecast.get('Toa', [self.T_out] * PH)[:PH]
        price = price_forecast[:PH] if price_forecast else [0.1] * PH
        
        # Decision variables: X = [Q[0..PH-1], T[0..PH-1], mu[0..PH-1]]
        X = ca.MX.sym('X', 3 * PH)
        Q_hvac = X[0*PH:1*PH]
        T = X[1*PH:2*PH]
        mu_slack = X[2*PH:3*PH]
        
        # Objective
        obj = 0
        for k in range(PH):
            # Energy cost
            P_k = self.c * Q_hvac[k]
            obj += self.w_energy * price[k] * P_k * (self.dt / 3600.0)
            
            # Comfort penalty
            T_viol = ca.fmax(0, T[k] - self.T_max) + ca.fmax(0, self.T_min - T[k])
            obj += self.w_comfort * T_viol**2
            
            # Budget violation penalty
            obj += self.w_budget * (mu_slack[k] / self.mu_bar)**2
        
        # Constraints
        g = []
        lbg = []
        ubg = []
        
        for k in range(PH):
            # HVAC limits
            g.append(Q_hvac[k])
            lbg.append(self.Q_min)
            ubg.append(self.Q_max)
            
            # Temperature limits (slightly soft for numerical stability)
            g.append(T[k])
            lbg.append(self.T_min - 2.0)
            ubg.append(self.T_max + 2.0)
            
            # Dynamics
            if k == 0:
                T_prev = self.T_zone
            else:
                T_prev = T[k-1]
            
            T_next = T_prev + self.dt * (-self.a * (T_prev - T_out_forecast[k]) - self.b * Q_hvac[k])
            g.append(T[k] - T_next)
            lbg.append(0.0)
            ubg.append(0.0)
            
            # Budget constraint: P <= P_budget + mu
            P_k = self.c * Q_hvac[k]
            g.append(P_k - mu_slack[k])
            lbg.append(-ca.inf)
            ubg.append(power_budget[k])
            
            # Slack >= 0
            g.append(mu_slack[k])
            lbg.append(0.0)
            ubg.append(50.0)
        
        # Solve
        nlp = {'x': X, 'f': obj, 'g': ca.vertcat(*g)}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 200}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        x0 = [5.0]*PH + [22.0]*PH + [0.0]*PH
        res = solver(x0=x0, lbg=lbg, ubg=ubg)
        
        solve_time = time.time() - start
        
        # Extract solution
        u_opt = res['x'].full().flatten()
        Q_opt = u_opt[0*PH:1*PH]
        mu_opt = u_opt[2*PH:3*PH]
        P_opt = [self.c * Q_opt[k] for k in range(PH)]
        
        result = MPCResult(
            building_id=self.building_id,
            timestamp=current_time,
            control_inputs={'Q_hvac': Q_opt[0]},
            power_trajectory=P_opt,
            budget_slack=list(mu_opt),
            objective_value=float(res['f']),
            solve_time_s=solve_time,
            feasible=solver.stats()['success']
        )
        
        self.last_mpc_result = result
        
        return result
    
    def _solve_min_power_mpc(self, weather, price, PH):
        """Minimize power consumption"""
        T_out = weather.get('Toa', [self.T_out] * PH)[:PH]
        
        # Decision variables: X = [Q[0], ..., Q[PH-1], T[0], ..., T[PH-1]]
        X = ca.MX.sym('X', 2 * PH)
        Q = X[:PH]
        T = X[PH:]
        
        # Objective: minimize power
        obj = sum(self.c * Q[k] for k in range(PH))
        
        # Constraints
        g = []
        lbg = []
        ubg = []
        
        for k in range(PH):
            # Control limits
            g.append(Q[k])
            lbg.append(self.Q_min)
            ubg.append(self.Q_max)
            
            # Temperature limits
            g.append(T[k])
            lbg.append(self.T_min)
            ubg.append(self.T_max)
            
            # Dynamics: T[k] = T[k-1] + dt*(...)
            if k == 0:
                T_prev = self.T_zone
            else:
                T_prev = T[k-1]
            
            T_next_predicted = T_prev + self.dt * (-self.a * (T_prev - T_out[k]) - self.b * Q[k])
            g.append(T[k] - T_next_predicted)
            lbg.append(0.0)
            ubg.append(0.0)
        
        nlp = {'x': X, 'f': obj, 'g': ca.vertcat(*g)}
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        x0 = [5.0]*PH + [22.0]*PH  # Initial guess
        res = solver(x0=x0, lbg=lbg, ubg=ubg)
        
        x_opt = res['x'].full().flatten()
        Q_opt = x_opt[:PH]
        
        # Check feasibility - accept both optimal and "acceptable" solutions
        stats = solver.stats()
        is_feasible = stats['success'] or stats['return_status'] in [
            'Solve_Succeeded',
            'Solved_To_Acceptable_Level', 
            'Feasible_Point_Found'
        ]
        
        return {
            'power': [self.c * Q_opt[k] for k in range(PH)],
            'feasible': is_feasible
        }
    
    def _solve_max_power_mpc(self, weather, price, PH):
        """Maximize power consumption"""
        T_out = weather.get('Toa', [self.T_out] * PH)[:PH]
        
        # Decision variables: X = [Q[0], ..., Q[PH-1], T[0], ..., T[PH-1]]
        X = ca.MX.sym('X', 2 * PH)
        Q = X[:PH]
        T = X[PH:]
        
        # Objective: maximize power (minimize negative)
        obj = -sum(self.c * Q[k] for k in range(PH))
        
        # Constraints
        g = []
        lbg = []
        ubg = []
        
        for k in range(PH):
            # Control limits
            g.append(Q[k])
            lbg.append(self.Q_min)
            ubg.append(self.Q_max)
            
            # Temperature limits
            g.append(T[k])
            lbg.append(self.T_min)
            ubg.append(self.T_max)
            
            # Dynamics
            if k == 0:
                T_prev = self.T_zone
            else:
                T_prev = T[k-1]
            
            T_next_predicted = T_prev + self.dt * (-self.a * (T_prev - T_out[k]) - self.b * Q[k])
            g.append(T[k] - T_next_predicted)
            lbg.append(0.0)
            ubg.append(0.0)
        
        nlp = {'x': X, 'f': obj, 'g': ca.vertcat(*g)}
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        x0 = [10.0]*PH + [22.0]*PH  # Initial guess
        res = solver(x0=x0, lbg=lbg, ubg=ubg)
        
        x_opt = res['x'].full().flatten()
        Q_opt = x_opt[:PH]
        
        # Check feasibility - accept both optimal and "acceptable" solutions
        stats = solver.stats()
        is_feasible = stats['success'] or stats['return_status'] in [
            'Solve_Succeeded',
            'Solved_To_Acceptable_Level',
            'Feasible_Point_Found'
        ]
        
        return {
            'power': [self.c * Q_opt[k] for k in range(PH)],
            'feasible': is_feasible
        }
    
    def get_state(self) -> BuildingState:
        return BuildingState(
            building_id=self.building_id,
            timestamp=0.0,
            status=self.status,
            control_mode=self.control_mode,
            power_actual_kW=self.c * 5.0,  # Placeholder
            power_forecast_kW=[],
            zone_temperatures={'core': self.T_zone},
            comfort_violations_degCh=0.0
        )
    
    def apply_control(self, control_input: Dict) -> bool:
        Q = control_input.get('Q_hvac', 5.0)
        # Apply to dynamics (simplified)
        self.T_zone += self.dt * (-self.a * (self.T_zone - self.T_out) - self.b * Q)
        return True
    
    def step(self, dt: float):
        # Simple dynamics step
        pass
    
    def shutdown(self):
        logger.info(f"{self.building_id} shutdown")
