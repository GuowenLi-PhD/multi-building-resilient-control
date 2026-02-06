"""
Main Simulation Orchestrator — Closed-Loop Hierarchical Control

Implements the coordination sequence (per aggregator interval):
  1. Both buildings compute flexibility bands (two-pass MPC / DNN forward sim)
  2. Aggregator solves log-utility allocation → sends power budgets
  3. Building B solves MPC with budget → applies control for 1 hour
  4. Building A solves MPC with budget at 15-min sub-intervals
  5. Both buildings simulate forward (FMU or surrogate) → report measurements

Supports:
  - Pure simulation (FMU-based) and mock simulation (for code validation)
  - Configurable attack scenarios (feeder limit + DoS on Building A)
  - Comprehensive logging for post-processing

Author: Guowen Li, AI Assistant
Date: 2025-02
"""

import numpy as np
import os
import sys
import copy
import json
import logging
import time as time_module
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict

from data_models import (
    FlexibilityBand, FlexibilityReport, PowerBudget, AggregatorDecision,
    StepLog, AggregatorLog, BuildingStatus, ControlMode, EnergyPriority
)
from aggregator import AggregatorMPC
from mpc_a_hierarchical import BuildingAHierarchicalMPC
from mpc_b_hierarchical import BuildingBHierarchicalMPC

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-18s | %(levelname)-5s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("orchestrator")


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_CONFIG = {
    # ── Timing ──
    'timing': {
        'simulation_start_time': 0,             # [s] simulation start (e.g., 0 = midnight)
        'simulation_duration': 86400,           # [s] total duration (24 hours)
        'building_a_timestep': 900,             # [s] 15 min
        'building_b_timestep': 3600,            # [s] 1 hour
        'aggregator_timestep': 3600,            # [s] 1 hour (matches Building B)
        'prediction_horizon_building_a': 4,     # steps (4 × 15min = 1 hour)
        'prediction_horizon_building_b': 16,    # steps (16 × 1h = 16 hours)
    },
    # ── Aggregator ──
    'aggregator': {
        'prediction_horizon': 8,                # steps at 1-hour resolution
        'dt': 3600,                             # [s]
        'feeder_limit_kW': 50.0,               # [kW] nominal feeder capacity
        'delta': 0.1,                           # log regularizer
        'solver_print': False,
    },
    # ── Building A ──
    'building_a': {
        'number_zones': 5,
        'priority_nominal': 2,                  # MEDIUM
        'priority_under_attack': 3,             # HIGH
        'budget_config': {
            'w_budget': 50.0,
            'mu_bar_kW': 10.0,
            'mu_max_kW': 20.0,
        },
    },
    # ── Building B ──
    'building_b': {
        'number_zones': 5,
        'priority_nominal': 2,                  # MEDIUM
        'priority_support': 1,                  # LOW (sacrificing for grid)
        'SOC_initial': 0.50,
        'budget_config': {
            'w_budget': 5000.0,
            'mu_bar_kW': 5.0,
        },
    },
    # ── Attack scenario ──
    'attack': {
        'enabled': True,
        'type': 'compound',                     # feeder_limit + DoS
        'feeder_limit_attack_kW': 40.0,         # reduced feeder (80% of 50 kW)
        'dos_start_time': 28800,                # [s] 8:00 AM
        'dos_end_time': 36000,                  # [s] 10:00 AM
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Simulation Orchestrator
# ══════════════════════════════════════════════════════════════════════════════
class HierarchicalSimulation:
    """
    Main closed-loop simulation for hierarchical multi-building control.
    
    Orchestrates the Measure → Allocate → Optimize → Actuate sequence
    at each aggregator interval.
    """

    def __init__(self, config: Optional[Dict] = None, mock_mode: bool = False):
        """
        Parameters
        ----------
        config : dict, optional
            Configuration dictionary. Uses DEFAULT_CONFIG if not provided.
        mock_mode : bool
            If True, uses surrogate models instead of FMU simulation.
            Useful for testing the control logic without Modelica dependencies.
        """
        self.config = config or DEFAULT_CONFIG
        self.mock_mode = mock_mode
        
        # ── Timing ──
        tc = self.config['timing']
        self.t_start    = tc['simulation_start_time']
        self.t_end      = self.t_start + tc['simulation_duration']
        self.dt_A       = tc['building_a_timestep']       # 900 s
        self.dt_B       = tc['building_b_timestep']       # 3600 s
        self.dt_agg     = tc['aggregator_timestep']       # 3600 s
        self.PH_A       = tc['prediction_horizon_building_a']
        self.PH_B       = tc['prediction_horizon_building_b']
        
        # Aggregator config
        self.PH_agg = self.config['aggregator']['prediction_horizon']
        
        # ── Controllers ──
        self.aggregator: Optional[AggregatorMPC] = None
        self.mpc_A_wrapper: Optional[BuildingAHierarchicalMPC] = None
        self.mpc_B_wrapper: Optional[BuildingBHierarchicalMPC] = None
        self.mpc_A = None  # raw mpc_a.mpc_case instance
        self.mpc_B = None  # raw mpc_b.mpc_case instance
        
        # ── State tracking ──
        self.current_time = self.t_start
        self.power_A_kW = 0.0
        self.power_B_kW = 0.0
        self.SOC_B = self.config['building_b']['SOC_initial']
        self.zone_temps_A = {z: 24.0 for z in ['core','east','north','south','west']}
        self.zone_temps_B = {z: 24.0 for z in ['core','east','north','south','west']}
        self.attack_active = False
        self.current_budget_A: Optional[PowerBudget] = None
        self.current_budget_B: Optional[PowerBudget] = None
        
        # ── Logging ──
        self.step_logs: List[StepLog] = []
        self.agg_logs: List[AggregatorLog] = []

    # ──────────────────────────────────────────────────────────────────────
    # Initialization
    # ──────────────────────────────────────────────────────────────────────
    def initialize(
        self,
        mpc_a_instance=None,
        mpc_b_instance=None,
        building_a_fmu=None,
        building_b_fmu=None
    ):
        """
        Initialize all components.
        
        Parameters
        ----------
        mpc_a_instance : mpc_a.mpc_case, optional
            Pre-initialized Building A MPC. If None, uses mock.
        mpc_b_instance : mpc_b.mpc_case, optional  
            Pre-initialized Building B MPC. If None, uses mock.
        building_a_fmu : FMU object, optional
            Building A FMU for simulation. If None, uses surrogate.
        building_b_fmu : FMU object, optional
            Building B FMU for simulation. If None, uses surrogate.
        """
        logger.info("=" * 70)
        logger.info("HIERARCHICAL CONTROL SIMULATION — INITIALIZATION")
        logger.info("=" * 70)
        
        # ── Aggregator ──
        self.aggregator = AggregatorMPC(self.config)
        
        # ── Building A ──
        if mpc_a_instance is not None:
            self.mpc_A = mpc_a_instance
        else:
            logger.info("  Building A: using mock MPC (no FMU)")
        
        self.mpc_A_wrapper = BuildingAHierarchicalMPC(
            mpc=self.mpc_A,
            budget_config=self.config['building_a']['budget_config']
        ) if self.mpc_A is not None else None
        
        # ── Building B ──
        if mpc_b_instance is not None:
            self.mpc_B = mpc_b_instance
        else:
            logger.info("  Building B: using mock MPC (no FMU)")
        
        self.mpc_B_wrapper = BuildingBHierarchicalMPC(
            mpc=self.mpc_B,
            budget_config=self.config['building_b']['budget_config']
        ) if self.mpc_B is not None else None
        
        # ── FMUs ──
        self.fmu_A = building_a_fmu
        self.fmu_B = building_b_fmu
        
        logger.info(f"  Simulation: {self.t_start}s → {self.t_end}s ({(self.t_end-self.t_start)/3600:.0f}h)")
        logger.info(f"  Building A: dt={self.dt_A}s, PH={self.PH_A}")
        logger.info(f"  Building B: dt={self.dt_B}s, PH={self.PH_B}")
        logger.info(f"  Aggregator: dt={self.dt_agg}s, PH={self.PH_agg}")
        logger.info(f"  Feeder limit: {self.config['aggregator']['feeder_limit_kW']} kW")
        
        if self.config['attack']['enabled']:
            atk = self.config['attack']
            logger.info(
                f"  Attack scenario: {atk['type']}, "
                f"feeder→{atk['feeder_limit_attack_kW']}kW, "
                f"DoS [{atk['dos_start_time']/3600:.0f}h, {atk['dos_end_time']/3600:.0f}h]"
            )
        
        logger.info("✓ Initialization complete\n")

    # ──────────────────────────────────────────────────────────────────────
    # Main Simulation Loop
    # ──────────────────────────────────────────────────────────────────────
    def run(self) -> Dict:
        """
        Execute the full closed-loop simulation.
        
        Returns
        -------
        results : dict with simulation logs and summary statistics
        """
        logger.info("=" * 70)
        logger.info("STARTING CLOSED-LOOP HIERARCHICAL SIMULATION")
        logger.info("=" * 70)
        
        t = self.t_start
        wall_start = time_module.time()
        agg_step_count = 0
        
        while t < self.t_end:
            # ── Check attack status ──
            self._update_attack_status(t)
            
            # ══════════════════════════════════════════════════════════════
            # AGGREGATOR COORDINATION (runs every dt_agg = 1 hour)
            # ══════════════════════════════════════════════════════════════
            
            # Step 1: Both buildings compute flexibility bands
            report_A = self._get_building_A_report(t)
            report_B = self._get_building_B_report(t)
            
            reports = {
                'Building_A': report_A,
                'Building_B': report_B,
            }
            
            # Step 2: Aggregator solves allocation
            decision = self.aggregator.allocate(reports, timestamp=t)
            
            self.current_budget_A = decision.budgets.get('Building_A')
            self.current_budget_B = decision.budgets.get('Building_B')
            
            # Log aggregator decision
            self.agg_logs.append(AggregatorLog(
                timestamp=t,
                feeder_limit_kW=decision.feeder_limit_kW,
                total_power_kW=self.power_A_kW + self.power_B_kW,
                total_allocated_kW=decision.total_allocated_kW[0],
                budgets={bid: bgt.P_ref_kW[0] for bid, bgt in decision.budgets.items()},
                flex_bands={
                    'Building_A': (report_A.flexibility_band.P_lower_kW[0],
                                   report_A.flexibility_band.P_upper_kW[0]),
                    'Building_B': (report_B.flexibility_band.P_lower_kW[0],
                                   report_B.flexibility_band.P_upper_kW[0]),
                },
                priorities={bid: int(r.priority) for bid, r in reports.items()},
                solver_status=decision.solver_status,
                objective_value=decision.objective_value,
            ))
            
            # Step 3: Building B solves MPC with budget (1-hour step)
            self._step_building_B(t)
            
            # Step 4: Building A solves MPC at 15-min sub-intervals
            n_sub = max(1, int(self.dt_agg / self.dt_A))
            for sub in range(n_sub):
                t_sub = t + sub * self.dt_A
                if t_sub >= self.t_end:
                    break
                self._step_building_A(t_sub, sub)
            
            # Advance to next aggregator step
            t += self.dt_agg
            agg_step_count += 1
        
        wall_elapsed = time_module.time() - wall_start
        
        # ── Summary ──
        logger.info("\n" + "=" * 70)
        logger.info("SIMULATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"  Aggregator steps: {agg_step_count}")
        logger.info(f"  Wall-clock time: {wall_elapsed:.1f}s")
        logger.info(f"  Building A steps: {sum(1 for l in self.step_logs if l.building_id == 'Building_A')}")
        logger.info(f"  Building B steps: {sum(1 for l in self.step_logs if l.building_id == 'Building_B')}")
        
        return self._compile_results()

    # ──────────────────────────────────────────────────────────────────────
    # Building A: Report + Step
    # ──────────────────────────────────────────────────────────────────────
    def _get_building_A_report(self, t: float) -> FlexibilityReport:
        """Compute Building A's flexibility report for the aggregator."""
        
        if self.mpc_A_wrapper is not None:
            # Real MPC: compute flexibility bands via two-pass optimization
            P_lo, P_hi = self.mpc_A_wrapper.compute_flexibility_bands(
                PH_agg=self.PH_agg, dt_agg=self.dt_agg
            )
        else:
            # Mock: use heuristic bands
            P_lo, P_hi = self._mock_flexibility_A(t)
        
        priority = (EnergyPriority.HIGH if self.attack_active 
                    else EnergyPriority(self.config['building_a']['priority_nominal']))
        
        return FlexibilityReport(
            building_id='Building_A',
            timestamp=t,
            power_actual_kW=self.power_A_kW,
            flexibility_band=FlexibilityBand(P_lower_kW=P_lo, P_upper_kW=P_hi),
            priority=priority,
            status=BuildingStatus.UNDER_ATTACK if self.attack_active else BuildingStatus.NORMAL,
            control_mode=ControlMode.ADAPTIVE if self.attack_active else ControlMode.NOMINAL,
            zone_temperatures=dict(self.zone_temps_A),
        )

    def _step_building_A(self, t: float, sub_step: int):
        """Execute one 15-min step for Building A with budget constraint."""
        
        # Get budget for this sub-step (budget is constant within aggregator interval)
        P_ref_kW = None
        if self.current_budget_A is not None:
            # The aggregator budget is at 1-hour resolution.
            # For Building A's 15-min steps within the hour, use step 0's budget.
            P_ref_kW = self.current_budget_A.P_ref_kW
        
        if self.mpc_A_wrapper is not None:
            # Real MPC with budget constraint
            self.mpc_A.set_time(t)
            res = self.mpc_A_wrapper.optimize_with_budget(P_ref_kW)
            u_opt = np.array(res['x']).flatten()
            
            # Extract first-step power
            self.power_A_kW = float(self.mpc_A.get_power_pred(u_opt[:11])) / 1000.0
            self.power_A_kW = max(self.power_A_kW, 0.0)
            
            # Apply to FMU if available
            if self.fmu_A is not None:
                self._apply_control_A(u_opt, t)
        else:
            # Mock simulation
            self.power_A_kW = self._mock_step_A(t, P_ref_kW)
        
        # Log
        budget_kW = P_ref_kW[0] if P_ref_kW else 0.0
        violation = max(0, self.power_A_kW - budget_kW) if P_ref_kW else 0.0
        
        self.step_logs.append(StepLog(
            timestamp=t,
            building_id='Building_A',
            power_actual_kW=self.power_A_kW,
            power_budget_kW=budget_kW,
            budget_violation_kW=violation,
            zone_temperatures=dict(self.zone_temps_A),
            comfort_violation_Kh=0.0,
            control_mode='adaptive' if self.attack_active else 'nominal',
            extra={'sub_step': sub_step},
        ))

    # ──────────────────────────────────────────────────────────────────────
    # Building B: Report + Step
    # ──────────────────────────────────────────────────────────────────────
    def _get_building_B_report(self, t: float) -> FlexibilityReport:
        """Compute Building B's flexibility report for the aggregator."""
        
        if self.mpc_B_wrapper is not None:
            Toa_forecast = self.mpc_B.predictor['Toa']
            P_lo, P_hi = self.mpc_B_wrapper.compute_flexibility_bands(
                PH_agg=self.PH_agg,
                dt_agg=self.dt_agg,
                SOC_current=self.SOC_B,
                Toa_forecast=Toa_forecast,
            )
        else:
            P_lo, P_hi = self._mock_flexibility_B(t)
        
        priority = (EnergyPriority(self.config['building_b']['priority_support'])
                    if self.attack_active
                    else EnergyPriority(self.config['building_b']['priority_nominal']))
        
        report = FlexibilityReport(
            building_id='Building_B',
            timestamp=t,
            power_actual_kW=self.power_B_kW,
            flexibility_band=FlexibilityBand(P_lower_kW=P_lo, P_upper_kW=P_hi),
            priority=priority,
            status=BuildingStatus.NORMAL,
            control_mode=ControlMode.SUPPORT if self.attack_active else ControlMode.NOMINAL,
            zone_temperatures=dict(self.zone_temps_B),
        )
        report.extra_data['SOC'] = self.SOC_B
        return report

    def _step_building_B(self, t: float):
        """Execute one 1-hour step for Building B with budget constraint."""
        
        P_ref_kW = None
        if self.current_budget_B is not None:
            P_ref_kW = self.current_budget_B.P_ref_kW
        
        if self.mpc_B_wrapper is not None:
            self.mpc_B.set_time(t)
            res, status = self.mpc_B_wrapper.optimize_with_budget(P_ref_kW)
            u_opt = res['x']
            mode = int(round(u_opt[0]))
            
            # Get predicted power and SOC
            power_dnn = self.mpc_B_wrapper._get_power_dnn()
            SOC_dnn = self.mpc_B_wrapper._get_SOC_dnn()
            Toa = self.mpc_B.predictor['Toa'][0]
            
            self.power_B_kW = max(float(power_dnn(ca.vertcat(mode, Toa))), 0.0) / 1000.0
            self.SOC_B = float(SOC_dnn(ca.vertcat(mode, self.SOC_B)))
            self.SOC_B = np.clip(self.SOC_B, 0.0, 1.0)
            
            if self.fmu_B is not None:
                self._apply_control_B(mode, t)
        else:
            self.power_B_kW, self.SOC_B = self._mock_step_B(t, P_ref_kW)
        
        budget_kW = P_ref_kW[0] if P_ref_kW else 0.0
        violation = max(0, self.power_B_kW - budget_kW) if P_ref_kW else 0.0
        
        self.step_logs.append(StepLog(
            timestamp=t,
            building_id='Building_B',
            power_actual_kW=self.power_B_kW,
            power_budget_kW=budget_kW,
            budget_violation_kW=violation,
            zone_temperatures=dict(self.zone_temps_B),
            comfort_violation_Kh=0.0,
            control_mode='support' if self.attack_active else 'nominal',
            extra={'SOC': self.SOC_B, 'TES_mode': 0},
        ))

    # ──────────────────────────────────────────────────────────────────────
    # Attack management
    # ──────────────────────────────────────────────────────────────────────
    def _update_attack_status(self, t: float):
        """Update attack status and feeder limit based on scenario config."""
        atk = self.config['attack']
        if not atk['enabled']:
            return
        
        was_active = self.attack_active
        self.attack_active = (atk['dos_start_time'] <= t < atk['dos_end_time'])
        
        if self.attack_active and not was_active:
            logger.warning(f"\n{'!'*70}")
            logger.warning(f"  ⚠ ATTACK ACTIVATED at t={t/3600:.1f}h")
            logger.warning(f"    DoS on Building A core zone VAV")
            logger.warning(f"    Feeder limit: {self.config['aggregator']['feeder_limit_kW']} → {atk['feeder_limit_attack_kW']} kW")
            logger.warning(f"{'!'*70}\n")
            
            self.aggregator.set_feeder_limit(atk['feeder_limit_attack_kW'])
            
            if self.mpc_A is not None:
                self.mpc_A.dos_attack_core_VAV = True
                self.mpc_A.w = [0., 100., 10.]
                
        elif not self.attack_active and was_active:
            logger.info(f"\n  ✓ Attack ended at t={t/3600:.1f}h — restoring nominal operation\n")
            self.aggregator.set_feeder_limit(self.config['aggregator']['feeder_limit_kW'])
            
            if self.mpc_A is not None:
                self.mpc_A.dos_attack_core_VAV = False
                self.mpc_A.w = [1., 1., 100.]

    # ──────────────────────────────────────────────────────────────────────
    # Mock simulation (for testing without FMU/DNN)
    # ──────────────────────────────────────────────────────────────────────
    def _mock_flexibility_A(self, t: float) -> Tuple[List[float], List[float]]:
        """Heuristic flexibility bands for Building A (no MPC available)."""
        t_hour = int((t % 86400) / 3600)
        is_occ = (7 <= t_hour < 19)
        
        if is_occ:
            if self.attack_active:
                P_lo = [6.0 + np.random.uniform(-0.5, 0.5)] * self.PH_agg
                P_hi = [14.0 + np.random.uniform(-0.5, 0.5)] * self.PH_agg
            else:
                P_lo = [3.0 + np.random.uniform(-0.3, 0.3)] * self.PH_agg
                P_hi = [12.0 + np.random.uniform(-0.3, 0.3)] * self.PH_agg
        else:
            P_lo = [0.1] * self.PH_agg
            P_hi = [0.5] * self.PH_agg
        
        return P_lo, P_hi

    def _mock_flexibility_B(self, t: float) -> Tuple[List[float], List[float]]:
        """Heuristic flexibility bands for Building B (no MPC available)."""
        t_hour = int((t % 86400) / 3600)
        is_occ = (6 <= t_hour < 19)
        
        P_lo, P_hi = [], []
        SOC = self.SOC_B
        
        for k in range(self.PH_agg):
            t_k = int(((t + k * self.dt_agg) % 86400) / 3600)
            if 6 <= t_k < 19:
                # Occupied: modes 1 (discharge TES) or 2 (chiller)
                P_lo.append(3.0 if SOC > 0.25 else 10.0)   # mode 1 vs 2
                P_hi.append(16.0)
                SOC = max(SOC - 0.04, 0.20)
            else:
                # Unoccupied: modes -1 (charge) or 0 (off)
                P_lo.append(0.0)
                P_hi.append(10.0)
                SOC = min(SOC + 0.08, 0.99)
        
        return P_lo, P_hi

    def _mock_step_A(self, t: float, P_ref_kW: Optional[List[float]]) -> float:
        """Mock Building A simulation step."""
        t_hour = int((t % 86400) / 3600)
        is_occ = (7 <= t_hour < 19)
        
        if not is_occ:
            return 0.1
        
        base_power = 8.0
        if self.attack_active:
            base_power = 10.5  # higher due to compensatory control
        
        # Respect budget if available
        if P_ref_kW:
            budget = P_ref_kW[0]
            base_power = min(base_power, budget + 1.0)  # allow small violation
        
        noise = np.random.uniform(-0.3, 0.3)
        return max(base_power + noise, 0.1)

    def _mock_step_B(self, t: float, P_ref_kW: Optional[List[float]]) -> Tuple[float, float]:
        """Mock Building B simulation step. Returns (power_kW, SOC)."""
        t_hour = int((t % 86400) / 3600)
        is_occ = (6 <= t_hour < 19)
        
        if is_occ:
            if self.SOC_B > 0.25:
                # Discharge TES → low power
                power = 5.0
                SOC_new = self.SOC_B - 0.04
            else:
                # Chiller only → high power
                power = 14.0
                SOC_new = self.SOC_B
        else:
            if self.SOC_B < 0.90:
                # Charge TES
                power = 8.0
                SOC_new = self.SOC_B + 0.08
            else:
                # Off
                power = 0.0
                SOC_new = self.SOC_B
        
        # Respect budget
        if P_ref_kW:
            budget = P_ref_kW[0]
            if power > budget + 0.5:
                # Try to shift to lower-power mode
                power = min(power, budget + 0.5)
        
        SOC_new = np.clip(SOC_new, 0.20, 0.99)
        return power + np.random.uniform(-0.2, 0.2), SOC_new

    # ──────────────────────────────────────────────────────────────────────
    # FMU interface (for real simulation)
    # ──────────────────────────────────────────────────────────────────────
    def _apply_control_A(self, u_opt, t):
        """Apply optimal controls to Building A FMU."""
        uMPC = [
            float(u_opt[0]), float(u_opt[1]),
            float(u_opt[2]) + 273.15, float(u_opt[3]) + 273.15,
            float(u_opt[4]) + 273.15,
            float(u_opt[5]), float(u_opt[6]), float(u_opt[7]),
            float(u_opt[8]), float(u_opt[9])
        ]
        ctrl_names = [
            'oveOnChiPla', 'conAHU_supFan_oveOnSupFan',
            'oveTChiWatSupSet', 'oveTConWatSupSet', 'conAHU_oveTSupAir',
            'conVAVCor_damVal_oveVDisSet', 'conVAVEas_damVal_oveVDisSet',
            'conVAVNor_damVal_oveVDisSet', 'conVAVSou_damVal_oveVDisSet',
            'conVAVWes_damVal_oveVDisSet'
        ]
        for name, val in zip(ctrl_names, uMPC):
            self.fmu_A.set(name + '_u', val)
            self.fmu_A.set(name + '_activate', 1)

    def _apply_control_B(self, mode, t):
        """Apply optimal mode to Building B FMU."""
        self.fmu_B.set('uMod', mode)

    # ──────────────────────────────────────────────────────────────────────
    # Results compilation
    # ──────────────────────────────────────────────────────────────────────
    def _compile_results(self) -> Dict:
        """Compile simulation results for analysis."""
        
        # Extract time series
        t_A = [l.timestamp for l in self.step_logs if l.building_id == 'Building_A']
        P_A = [l.power_actual_kW for l in self.step_logs if l.building_id == 'Building_A']
        B_A = [l.power_budget_kW for l in self.step_logs if l.building_id == 'Building_A']
        V_A = [l.budget_violation_kW for l in self.step_logs if l.building_id == 'Building_A']
        
        t_B = [l.timestamp for l in self.step_logs if l.building_id == 'Building_B']
        P_B = [l.power_actual_kW for l in self.step_logs if l.building_id == 'Building_B']
        B_B = [l.power_budget_kW for l in self.step_logs if l.building_id == 'Building_B']
        V_B = [l.budget_violation_kW for l in self.step_logs if l.building_id == 'Building_B']
        
        t_agg = [l.timestamp for l in self.agg_logs]
        P_total = [l.total_power_kW for l in self.agg_logs]
        P_alloc = [l.total_allocated_kW for l in self.agg_logs]
        feeder = [l.feeder_limit_kW for l in self.agg_logs]
        
        # Compute KPIs
        feeder_violations = sum(
            1 for l in self.agg_logs 
            if l.total_power_kW > l.feeder_limit_kW + 0.1
        )
        
        total_budget_violation_A = sum(V_A)
        total_budget_violation_B = sum(V_B)
        
        results = {
            'building_A': {'time': t_A, 'power_kW': P_A, 'budget_kW': B_A, 'violation_kW': V_A},
            'building_B': {'time': t_B, 'power_kW': P_B, 'budget_kW': B_B, 'violation_kW': V_B},
            'aggregator': {'time': t_agg, 'total_power': P_total, 'allocated': P_alloc, 'feeder': feeder},
            'KPIs': {
                'feeder_violations': feeder_violations,
                'total_budget_violation_A_kWh': total_budget_violation_A * self.dt_A / 3600.0,
                'total_budget_violation_B_kWh': total_budget_violation_B * self.dt_B / 3600.0,
                'peak_power_kW': max(P_total) if P_total else 0,
                'total_energy_kWh_A': sum(P_A) * self.dt_A / 3600.0,
                'total_energy_kWh_B': sum(P_B) * self.dt_B / 3600.0,
            },
            'step_logs': self.step_logs,
            'agg_logs': self.agg_logs,
        }
        
        logger.info(f"\n  KPIs:")
        for k, v in results['KPIs'].items():
            logger.info(f"    {k}: {v:.2f}" if isinstance(v, float) else f"    {k}: {v}")
        
        return results


# ══════════════════════════════════════════════════════════════════════════════
# Entry point: Quick demo with mock simulation
# ══════════════════════════════════════════════════════════════════════════════
def run_mock_demo():
    """Run a quick demonstration with mock buildings (no FMU/DNN required)."""
    
    config = copy.deepcopy(DEFAULT_CONFIG)
    
    sim = HierarchicalSimulation(config=config, mock_mode=True)
    sim.initialize()
    results = sim.run()
    
    return results


if __name__ == "__main__":
    results = run_mock_demo()
