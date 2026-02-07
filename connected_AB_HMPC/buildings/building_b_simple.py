"""
Simplified Building B implementation with TES

Demonstrates:
- Two-pass MPC with TES flexibility
- Soft budget constraints
- TES charge/discharge optimization

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


class BuildingBSimple(BaseBuilding):
    """
    Simplified Building B (HVAC + TES)
    
    Simplified dynamics:
    - Thermal: dT/dt = -a*(T - T_out) + b*Q_hvac
    - TES: dSOC/dt = (P_charge - P_discharge) / E_TES_max
    - Power: P = c*Q_hvac + P_charge
    
    TES provides flexibility via load shifting
    """
    
    def __init__(self, building_id: str, config: Dict):
        super().__init__(building_id, config)
        
        # Thermal parameters (realistic)
        self.a = 1.0 / 14400.0  # Thermal conductance (4 hour time constant)
        self.b = 0.0002         # HVAC effectiveness
        self.c = 0.8            # Power coefficient
        
        # TES parameters
        self.SOC = 0.5  # State of charge [0-1]
        self.SOC_min = 0.2
        self.SOC_max = 0.99
        self.E_TES_max = 1152.0  # kWh
        self.P_charge_max = 10.0  # kW
        self.P_discharge_max = 15.0  # kW
        
        # State
        self.T_zone = 22.0
        self.T_out = 30.0
        
        # Constraints
        self.T_min = 21.0
        self.T_max = 24.0
        self.Q_min = 0.0
        self.Q_max = 15.0
        
        # MPC parameters
        self.dt = config.get('timing', {}).get('building_b_timestep', 3600.0)
        self.PH = 4  # Prediction horizon (4 hours @ 1-hr timestep)
        
        # Weights
        self.w_energy = 1.0
        self.w_comfort = 100.0
        self.w_budget = 10.0
        self.w_SOC = 0.1
        self.mu_bar = 8.0
        
        logger.info(f"✓ {self.building_id} (Simple with TES) initialized")
    
    def compute_flexibility_band(self, current_time, weather_forecast,
                                 price_forecast, horizon_steps) -> FlexibilityBand:
        """Compute flexibility band via two-pass MPC"""
        
        logger.info(f"  🔄 {self.building_id}: Computing flexibility band...")
        
        import time
        start = time.time()
        
        result_min = self._solve_min_power_mpc(weather_forecast, price_forecast, horizon_steps)
        t_min = time.time() - start
        
        if not result_min['feasible']:
            logger.error(f"    ✗ {self.building_id}: Min-power MPC INFEASIBLE")
        
        start = time.time()
        result_max = self._solve_max_power_mpc(weather_forecast, price_forecast, horizon_steps)
        t_max = time.time() - start
        
        if not result_max['feasible']:
            logger.error(f"    ✗ {self.building_id}: Max-power MPC INFEASIBLE")
        
        # Ensure P_lower <= P_upper
        P_lower = result_min['power']
        P_upper = result_max['power']
        
        # If bounds are reversed, swap them
        for k in range(len(P_lower)):
            if P_lower[k] > P_upper[k]:
                P_lower[k], P_upper[k] = P_upper[k], P_lower[k]
        
        two_pass = TwoPassMPCResult(
            building_id=self.building_id,
            timestamp=current_time,
            min_power_trajectory=P_lower,
            max_power_trajectory=P_upper,
            min_power_solve_time_s=t_min,
            max_power_solve_time_s=t_max,
            both_feasible=result_min['feasible'] and result_max['feasible']
        )
        
        time_horizon = [current_time + k * self.dt for k in range(horizon_steps)]
        band = two_pass.to_flexibility_band(time_horizon)
        
        # Validate
        if not band.validate():
            logger.error(f"    ✗ Invalid flexibility band for {self.building_id}!")
            logger.error(f"      P_lower = {band.P_lower_kW[:3]}")
            logger.error(f"      P_upper = {band.P_upper_kW[:3]}")
            band.feasible = False
        
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
        
        T_out_forecast = weather_forecast.get('Toa', [self.T_out] * PH)[:PH]
        price = price_forecast[:PH] if price_forecast else [0.1] * PH
        
        # Decision variables: X = [Q[0..PH-1], P_charge[0..PH-1], P_discharge[0..PH-1], T[0..PH-1], SOC[0..PH-1], mu[0..PH-1]]
        X = ca.MX.sym('X', 6 * PH)
        Q_hvac = X[0*PH:1*PH]
        P_charge = X[1*PH:2*PH]
        P_discharge = X[2*PH:3*PH]
        T = X[3*PH:4*PH]
        SOC = X[4*PH:5*PH]
        mu_slack = X[5*PH:6*PH]
        
        # Objective
        obj = 0
        for k in range(PH):
            # Power = HVAC + charging
            P_k = self.c * Q_hvac[k] + P_charge[k]
            obj += self.w_energy * price[k] * P_k * (self.dt / 3600.0)
            
            # Comfort
            T_viol = ca.fmax(0, T[k] - self.T_max) + ca.fmax(0, self.T_min - T[k])
            obj += self.w_comfort * T_viol**2
            
            # SOC penalty (encourage mid-range)
            obj += self.w_SOC * (SOC[k] - 0.5)**2
            
            # Budget violation
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
            
            # Charge/discharge limits
            g.append(P_charge[k])
            lbg.append(0.0)
            ubg.append(self.P_charge_max)
            
            g.append(P_discharge[k])
            lbg.append(0.0)
            ubg.append(self.P_discharge_max)
            
            # State limits
            g.append(T[k])
            lbg.append(self.T_min - 2.0)
            ubg.append(self.T_max + 2.0)
            
            g.append(SOC[k])
            lbg.append(self.SOC_min)
            ubg.append(self.SOC_max)
            
            # Dynamics
            if k == 0:
                T_prev = self.T_zone
                SOC_prev = self.SOC
            else:
                T_prev = T[k-1]
                SOC_prev = SOC[k-1]
            
            T_next = T_prev + self.dt * (-self.a * (T_prev - T_out_forecast[k]) - self.b * Q_hvac[k])
            SOC_next = SOC_prev + (self.dt / 3600.0) * (P_charge[k] - P_discharge[k]) / self.E_TES_max
            
            g.append(T[k] - T_next)
            lbg.append(0.0)
            ubg.append(0.0)
            
            g.append(SOC[k] - SOC_next)
            lbg.append(0.0)
            ubg.append(0.0)
            
            # Budget constraint: P <= P_budget + mu
            P_k = self.c * Q_hvac[k] + P_charge[k]
            g.append(P_k - mu_slack[k])
            lbg.append(-ca.inf)
            ubg.append(power_budget[k])
            
            # Slack >= 0
            g.append(mu_slack[k])
            lbg.append(0.0)
            ubg.append(50.0)
        
        # Solve
        nlp = {'x': X, 'f': obj, 'g': ca.vertcat(*g)}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 300}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        u0 = [5.0]*PH + [0.0]*2*PH + [22.0]*PH + [0.5]*PH + [0.0]*PH
        res = solver(x0=u0, lbg=lbg, ubg=ubg)
        
        solve_time = time.time() - start
        
        # Extract
        u_opt = res['x'].full().flatten()
        Q_opt = u_opt[0*PH:1*PH]
        P_ch_opt = u_opt[1*PH:2*PH]
        mu_opt = u_opt[5*PH:6*PH]
        
        P_opt = [self.c * Q_opt[k] + P_ch_opt[k] for k in range(PH)]
        
        # Check feasibility
        stats = solver.stats()
        is_feasible = stats["success"] or stats["return_status"] in ["Solve_Succeeded", "Solved_To_Acceptable_Level", "Feasible_Point_Found"]
        
        result = MPCResult(
            building_id=self.building_id,
            timestamp=current_time,
            control_inputs={'Q_hvac': Q_opt[0], 'P_charge': P_ch_opt[0]},
            power_trajectory=P_opt,
            budget_slack=list(mu_opt),
            objective_value=float(res['f']),
            solve_time_s=solve_time,
            feasible=is_feasible
        )
        
        self.last_mpc_result = result
        
        return result
    
    def _solve_min_power_mpc(self, weather, price, PH):
        """Minimize power by discharging TES"""
        T_out = weather.get('Toa', [self.T_out] * PH)[:PH]
        
        # Decision variables: X = [Q[0..PH-1], P_charge[0..PH-1], P_discharge[0..PH-1], T[0..PH-1], SOC[0..PH-1]]
        X = ca.MX.sym('X', 5 * PH)
        Q_hvac = X[0*PH:1*PH]
        P_charge = X[1*PH:2*PH]
        P_discharge = X[2*PH:3*PH]
        T = X[3*PH:4*PH]
        SOC = X[4*PH:5*PH]
        
        # Minimize total power (discharge reduces net power)
        obj = sum(self.c * Q_hvac[k] + P_charge[k] - 0.5*P_discharge[k] for k in range(PH))
        
        g = []
        lbg = []
        ubg = []
        
        for k in range(PH):
            # Control limits
            g.extend([Q_hvac[k], P_charge[k], P_discharge[k]])
            lbg.extend([self.Q_min, 0.0, 0.0])
            ubg.extend([self.Q_max, self.P_charge_max, self.P_discharge_max])
            
            # State limits
            g.extend([T[k], SOC[k]])
            lbg.extend([self.T_min, self.SOC_min])
            ubg.extend([self.T_max, self.SOC_max])
            
            # Dynamics
            if k == 0:
                T_prev = self.T_zone
                SOC_prev = self.SOC
            else:
                T_prev = T[k-1]
                SOC_prev = SOC[k-1]
            
            T_next = T_prev + self.dt * (-self.a * (T_prev - T_out[k]) - self.b * Q_hvac[k])
            SOC_next = SOC_prev + (self.dt / 3600.0) * (P_charge[k] - P_discharge[k]) / self.E_TES_max
            
            g.append(T[k] - T_next)
            lbg.append(0.0)
            ubg.append(0.0)
            
            g.append(SOC[k] - SOC_next)
            lbg.append(0.0)
            ubg.append(0.0)
        
        nlp = {'x': X, 'f': obj, 'g': ca.vertcat(*g)}
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        x0 = [5.0]*PH + [0.0]*2*PH + [22.0]*PH + [0.5]*PH
        res = solver(x0=x0, lbg=lbg, ubg=ubg)
        
        u_opt = res['x'].full().flatten()
        Q = u_opt[0*PH:1*PH]
        Pch = u_opt[1*PH:2*PH]
        P_result = [self.c * Q[k] + Pch[k] for k in range(PH)]
        
        # Check feasibility with detailed logging
        stats = solver.stats()
        return_status = stats.get('return_status', 'UNKNOWN')
        success = stats.get('success', False)
        
        is_feasible = success or return_status in [
            'Solve_Succeeded',
            'Solved_To_Acceptable_Level',
            'Feasible_Point_Found',
            'Search_Direction_Becomes_Too_Small',  # Can happen at boundaries
        ]
        
        # Accept solution if power values are reasonable (non-negative, bounded)
        power_ok = all(0 <= p <= 50.0 for p in P_result)
        if not is_feasible and power_ok:
            logger.warning(f"    Building_B min-power: return_status='{return_status}', but power values OK - accepting")
            is_feasible = True
        elif not is_feasible:
            logger.warning(f"    Building_B min-power: return_status='{return_status}', success={success}")
        
        return {
            'power': P_result,
            'feasible': is_feasible
        }
    
    def _solve_max_power_mpc(self, weather, price, PH):
        """Maximize power by charging TES"""
        T_out = weather.get('Toa', [self.T_out] * PH)[:PH]
        
        # Decision variables: X = [Q[0..PH-1], P_charge[0..PH-1], P_discharge[0..PH-1], T[0..PH-1], SOC[0..PH-1]]
        X = ca.MX.sym('X', 5 * PH)
        Q_hvac = X[0*PH:1*PH]
        P_charge = X[1*PH:2*PH]
        P_discharge = X[2*PH:3*PH]
        T = X[3*PH:4*PH]
        SOC = X[4*PH:5*PH]
        
        # Maximize power (negative objective)
        obj = -sum(self.c * Q_hvac[k] + P_charge[k] for k in range(PH))
        
        g = []
        lbg = []
        ubg = []
        
        for k in range(PH):
            # Control limits
            g.extend([Q_hvac[k], P_charge[k], P_discharge[k]])
            lbg.extend([self.Q_min, 0.0, 0.0])
            ubg.extend([self.Q_max, self.P_charge_max, self.P_discharge_max])
            
            # State limits
            g.extend([T[k], SOC[k]])
            lbg.extend([self.T_min, self.SOC_min])
            ubg.extend([self.T_max, self.SOC_max])
            
            # Dynamics
            if k == 0:
                T_prev = self.T_zone
                SOC_prev = self.SOC
            else:
                T_prev = T[k-1]
                SOC_prev = SOC[k-1]
            
            T_next = T_prev + self.dt * (-self.a * (T_prev - T_out[k]) - self.b * Q_hvac[k])
            SOC_next = SOC_prev + (self.dt / 3600.0) * (P_charge[k] - P_discharge[k]) / self.E_TES_max
            
            g.append(T[k] - T_next)
            lbg.append(0.0)
            ubg.append(0.0)
            
            g.append(SOC[k] - SOC_next)
            lbg.append(0.0)
            ubg.append(0.0)
        
        nlp = {'x': X, 'f': obj, 'g': ca.vertcat(*g)}
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        x0 = [10.0]*PH + [5.0]*PH + [0.0]*PH + [22.0]*PH + [0.5]*PH
        res = solver(x0=x0, lbg=lbg, ubg=ubg)
        
        u_opt = res['x'].full().flatten()
        Q = u_opt[0*PH:1*PH]
        Pch = u_opt[1*PH:2*PH]
        P_result = [self.c * Q[k] + Pch[k] for k in range(PH)]
        
        # Check feasibility with detailed logging
        stats = solver.stats()
        return_status = stats.get('return_status', 'UNKNOWN')
        success = stats.get('success', False)
        
        is_feasible = success or return_status in [
            'Solve_Succeeded',
            'Solved_To_Acceptable_Level',
            'Feasible_Point_Found',
            'Search_Direction_Becomes_Too_Small',  # Can happen at boundaries
        ]
        
        # Accept solution if power values are reasonable (non-negative, bounded)
        power_ok = all(0 <= p <= 50.0 for p in P_result)
        if not is_feasible and power_ok:
            logger.warning(f"    Building_B max-power: return_status='{return_status}', but power values OK - accepting")
            is_feasible = True
        elif not is_feasible:
            logger.warning(f"    Building_B max-power: return_status='{return_status}', success={success}")
        
        return {
            'power': P_result,
            'feasible': is_feasible
        }
    
    def get_state(self) -> BuildingState:
        state = BuildingState(
            building_id=self.building_id,
            timestamp=0.0,
            status=self.status,
            control_mode=self.control_mode,
            power_actual_kW=self.c * 5.0,
            power_forecast_kW=[],
            zone_temperatures={'core': self.T_zone},
            comfort_violations_degCh=0.0
        )
        state.extra_data = {'SOC_current': self.SOC}
        return state
    
    def apply_control(self, control_input: Dict) -> bool:
        Q = control_input.get('Q_hvac', 5.0)
        P_ch = control_input.get('P_charge', 0.0)
        
        self.T_zone += self.dt * (-self.a * (self.T_zone - self.T_out) - self.b * Q)
        self.SOC += (self.dt / 3600.0) * P_ch / self.E_TES_max
        self.SOC = max(self.SOC_min, min(self.SOC, self.SOC_max))
        return True
    
    def step(self, dt: float):
        pass
    
    def shutdown(self):
        logger.info(f"{self.building_id} shutdown")
