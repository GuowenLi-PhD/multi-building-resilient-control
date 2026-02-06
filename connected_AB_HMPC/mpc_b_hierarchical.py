"""
Building B Hierarchical MPC Wrapper

Extends the existing Building B MPC (mpc_b.mpc_case) with:
  1. compute_flexibility_bands() — DNN forward simulation for P̲ and P̄ trajectories
  2. optimize_with_budget()     — DEAP GA with additional budget penalty

Building B characteristics:
  - HVAC + Thermal Energy Storage (TES, ice tank)
  - dt = 3600 s (1 hour), PH = 16–20 hours
  - 1 discrete decision variable per step: uMod ∈ {-1, 0, 1, 2}
    Mode -1: Charge TES (unoccupied only)
    Mode  0: Off         (unoccupied only)
    Mode  1: Discharge TES (occupied only)
    Mode  2: Chiller only  (occupied only)
  - DNN prediction models: power, SOC, zone temperatures
  - DEAP genetic algorithm solver (discrete variables)

Author: Guowen Li, AI Assistant
Date: 2025-02
"""

import casadi as ca
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import copy

logger = logging.getLogger(__name__)


class BuildingBHierarchicalMPC:
    """
    Hierarchical wrapper for Building B's MPC.
    
    Parameters
    ----------
    mpc : mpc_b.mpc_case instance (already initialized with DNN model paths)
    budget_config : dict with keys:
        - 'w_budget'  : float  (weight for budget violation penalty, default 5000.0)
        - 'mu_bar_kW' : float  (normalization for μ, default 5.0 kW)
    """

    def __init__(self, mpc, budget_config: Optional[Dict] = None):
        self.mpc = mpc
        
        cfg = budget_config or {}
        self.w_budget  = float(cfg.get('w_budget', 5000.0))
        self.mu_bar_W  = float(cfg.get('mu_bar_kW', 5.0)) * 1000.0  # kW → W
        
        # Cache DNN CasADi functions (to avoid reloading .h5 files every call)
        self._power_dnn = None
        self._SOC_dnn = None
        self._temp_dnns = {}

    # ══════════════════════════════════════════════════════════════════════
    # DNN Model Loading (cached)
    # ══════════════════════════════════════════════════════════════════════
    def _get_power_dnn(self):
        """Load or return cached power DNN CasADi function."""
        if self._power_dnn is None:
            path = (self.mpc.model_paths['power'] 
                    if hasattr(self.mpc, 'model_paths') 
                    else "system_identification/dnn_power_model.h5")
            self._power_dnn = self.mpc.power_dnn_tensorflow(path)
        return self._power_dnn

    def _get_SOC_dnn(self):
        """Load or return cached SOC DNN CasADi function."""
        if self._SOC_dnn is None:
            path = (self.mpc.model_paths['SOC'] 
                    if hasattr(self.mpc, 'model_paths') 
                    else "system_identification/dnn_SOC_model.h5")
            self._SOC_dnn = self.mpc.SOC_dnn_tensorflow(path)
        return self._SOC_dnn

    def _get_temp_dnn(self, zone: str):
        """Load or return cached temperature DNN CasADi function for a zone."""
        if zone not in self._temp_dnns:
            path = (self.mpc.model_paths[zone] 
                    if hasattr(self.mpc, 'model_paths') 
                    else f"system_identification/dnn_model_{zone}_temperature.h5")
            self._temp_dnns[zone] = self.mpc.temp_dnn_tensorflow(path)
        return self._temp_dnns[zone]

    # ══════════════════════════════════════════════════════════════════════
    # 1. FLEXIBILITY BAND COMPUTATION (DNN Forward Simulation)
    # ══════════════════════════════════════════════════════════════════════
    def compute_flexibility_bands(
        self,
        PH_agg: int,
        dt_agg: float,
        SOC_current: float,
        Toa_forecast: List[float],
        Tz_his: Optional[Dict] = None
    ) -> Tuple[List[float], List[float]]:
        """
        Compute power flexibility bands via greedy forward simulation.
        
        At each step, enumerate all feasible modes {-1, 0} or {1, 2} based
        on occupancy, evaluate power and SOC via DNN, and select the mode
        that achieves min/max power while keeping SOC in [0.20, 0.99].
        
        This is a sequential greedy approach that correctly propagates SOC
        dynamics (resolves the chicken-and-egg dependency from Q1 analysis).
        
        Parameters
        ----------
        PH_agg : int
            Aggregator prediction horizon
        dt_agg : float
            Aggregator timestep [s]
        SOC_current : float
            Current TES state-of-charge [0, 1]
        Toa_forecast : list[float]
            Outdoor temperature forecast [°C], length ≥ PH_agg
        Tz_his : dict, optional
            Historical zone temperature states (used for temp DNN inputs)
            
        Returns
        -------
        (P_lower_kW, P_upper_kW) : tuple of lists, length PH_agg, in kW
        """
        mpc = self.mpc
        power_dnn = self._get_power_dnn()
        SOC_dnn = self._get_SOC_dnn()
        
        SOC_bounds = (0.20, 0.99)
        
        # Time info for occupancy
        current_time = mpc.time
        
        P_lower_kW = []
        P_upper_kW = []
        
        # Two separate SOC trajectories: one for min-power path, one for max-power path
        SOC_min_path = SOC_current
        SOC_max_path = SOC_current
        
        for k in range(PH_agg):
            t_sim = current_time + k * dt_agg
            t_hour = int((t_sim % 86400) / 3600)
            is_occupied = (t_hour >= mpc.occ_start and t_hour < mpc.occ_end)
            
            # Feasible modes
            if is_occupied:
                feasible_modes = [1, 2]     # Discharge TES or Chiller only
            else:
                feasible_modes = [-1, 0]    # Charge TES or Off
            
            Toa_k = Toa_forecast[min(k, len(Toa_forecast) - 1)]
            
            # ── Evaluate power for each mode ──
            mode_power = {}
            mode_soc_after = {}
            
            for mode in feasible_modes:
                # Power prediction
                P_W = float(power_dnn(ca.vertcat(mode, Toa_k)))
                P_W = max(P_W, 0.0)
                mode_power[mode] = P_W
            
            # ── MIN-POWER pass ──
            # For each feasible mode, evaluate SOC transition from min-path SOC
            best_min_mode = None
            best_min_P = float('inf')
            
            for mode in feasible_modes:
                soc_next = float(SOC_dnn(ca.vertcat(mode, SOC_min_path)))
                # Check SOC feasibility
                if soc_next < SOC_bounds[0] - 0.05 or soc_next > SOC_bounds[1] + 0.05:
                    continue  # Skip infeasible SOC transition
                soc_next = np.clip(soc_next, SOC_bounds[0], SOC_bounds[1])
                
                if mode_power[mode] < best_min_P:
                    best_min_P = mode_power[mode]
                    best_min_mode = mode
                    best_min_soc = soc_next
            
            if best_min_mode is None:
                # All modes infeasible → use least-bad option
                best_min_mode = feasible_modes[0]
                best_min_P = mode_power[best_min_mode]
                best_min_soc = float(SOC_dnn(ca.vertcat(best_min_mode, SOC_min_path)))
            
            SOC_min_path = best_min_soc
            P_lower_kW.append(best_min_P / 1000.0)
            
            # ── MAX-POWER pass ──
            best_max_mode = None
            best_max_P = -float('inf')
            
            for mode in feasible_modes:
                soc_next = float(SOC_dnn(ca.vertcat(mode, SOC_max_path)))
                if soc_next < SOC_bounds[0] - 0.05 or soc_next > SOC_bounds[1] + 0.05:
                    continue
                soc_next = np.clip(soc_next, SOC_bounds[0], SOC_bounds[1])
                
                if mode_power[mode] > best_max_P:
                    best_max_P = mode_power[mode]
                    best_max_mode = mode
                    best_max_soc = soc_next
            
            if best_max_mode is None:
                best_max_mode = feasible_modes[-1]
                best_max_P = mode_power[best_max_mode]
                best_max_soc = float(SOC_dnn(ca.vertcat(best_max_mode, SOC_max_path)))
            
            SOC_max_path = best_max_soc
            P_upper_kW.append(best_max_P / 1000.0)
        
        # Validate: ensure lower ≤ upper
        for k in range(PH_agg):
            if P_lower_kW[k] > P_upper_kW[k]:
                mid = 0.5 * (P_lower_kW[k] + P_upper_kW[k])
                P_lower_kW[k] = mid - 0.1
                P_upper_kW[k] = mid + 0.1
        
        logger.info(
            f"  Building B flexibility: [{P_lower_kW[0]:.1f}, {P_upper_kW[0]:.1f}] kW "
            f"(step 0, SOC={SOC_current:.2f})"
        )
        
        return P_lower_kW, P_upper_kW

    # ══════════════════════════════════════════════════════════════════════
    # 2. OPTIMIZE WITH POWER BUDGET CONSTRAINT
    # ══════════════════════════════════════════════════════════════════════
    def optimize_with_budget(
        self,
        P_ref_kW: Optional[List[float]] = None
    ):
        """
        Run Building B MPC with additional power budget penalty in DEAP objective.
        
        Modifies the DEAP evaluate_individual function to add:
          penalty += ω_budget · (max(0, P(k) - P_ref(k)) / μ̄)²
        
        Parameters
        ----------
        P_ref_kW : list of float or None
            Power budget trajectory [kW], length ≥ mpc.PH.
            If None, runs original MPC without budget.
        
        Returns
        -------
        res : dict with 'x' (optimal mode sequence) and 'f' (objective value)
        solver_status : dict with 'return_status'
        """
        mpc = self.mpc
        
        if P_ref_kW is None:
            return mpc.optimize()
        
        # Convert budget to Watts
        P_ref_W = [p * 1000.0 for p in P_ref_kW]
        while len(P_ref_W) < mpc.PH:
            P_ref_W.append(P_ref_W[-1])
        
        # ── Replicate the full MPC optimization with budget penalty ──
        # This follows the same structure as mpc_b.optimize() but adds
        # the budget penalty to the DEAP evaluate_individual function.
        
        occupied_ph = [0] * mpc.PH
        for i in range(mpc.PH):
            t = int(((mpc.time + i * mpc.dt) % 86400) / 3600)
            if t >= mpc.occ_start and t < mpc.occ_end:
                occupied_ph[i] = 1
        
        # Occupancy-dependent bounds
        u_lb_occ = []
        u_ub_occ = []
        for k in range(mpc.PH):
            t = int((mpc.time + k * mpc.dt) % 86400 / 3600)
            if t >= mpc.occ_start and t < mpc.occ_end:
                u_lb_occ.append(1)
                u_ub_occ.append(2)
            else:
                u_lb_occ.append(-1)
                u_ub_occ.append(0)
        
        # Load DNN models
        power_dnn = self._get_power_dnn()
        SOC_dnn = self._get_SOC_dnn()
        temp_dnns = {z: self._get_temp_dnn(z) for z in ['core','east','north','south','west']}
        
        # Get states
        To_pred_ph = mpc.predictor['Toa']
        price_ph = mpc.predictor['price']
        
        Tz_core_his = list(np.array(mpc.states['Tz_core_his_meas'][:]))
        Tz_east_his = list(np.array(mpc.states['Tz_east_his_meas'][:]))
        Tz_north_his = list(np.array(mpc.states['Tz_north_his_meas'][:]))
        Tz_south_his = list(np.array(mpc.states['Tz_south_his_meas'][:]))
        Tz_west_his = list(np.array(mpc.states['Tz_west_his_meas'][:]))
        SOC_his = list(np.array(mpc.states['SOC_his_meas'][:]))
        
        # Autocorrection
        n_his = len(Tz_core_his)
        Tz_core_pred_his = np.array(mpc.states['Tz_core_his_pred'])
        Tz_east_pred_his = np.array(mpc.states['Tz_east_his_pred'])
        Tz_north_pred_his = np.array(mpc.states['Tz_north_his_pred'])
        Tz_south_pred_his = np.array(mpc.states['Tz_south_his_pred'])
        Tz_west_pred_his = np.array(mpc.states['Tz_west_his_pred'])
        
        ae = {}
        for z, m_his, p_his in [
            ('core', Tz_core_his, Tz_core_pred_his),
            ('east', Tz_east_his, Tz_east_pred_his),
            ('north', Tz_north_his, Tz_north_pred_his),
            ('south', Tz_south_his, Tz_south_pred_his),
            ('west', Tz_west_his, Tz_west_pred_his),
        ]:
            ae[z] = sum((m_his[i] - p_his[i]) for i in range(n_his)) / n_his
        
        # ── Define evaluate_individual with budget penalty ──
        w_budget = self.w_budget
        mu_bar_W = self.mu_bar_W
        w = mpc.w  # [energy_cost, temp_violation, SOC_penalty]
        
        def evaluate_individual_with_budget(individual):
            """DEAP fitness function with power budget soft constraint."""
            individual = [max(lbx, min(round(val), ubx))
                          for val, lbx, ubx in zip(individual, u_lb_occ, u_ub_occ)]
            
            # Forward simulate through PH
            SOC_k = [s for s in SOC_his]
            Tz_c_k = [t for t in Tz_core_his]
            
            total_obj = 0.0
            penalties = 0.0
            
            for k in range(mpc.PH):
                mode = individual[k]
                
                # SOC prediction
                SOC_next = float(SOC_dnn(ca.vertcat(mode, SOC_k[0])))
                
                # Power prediction
                P_k = float(power_dnn(ca.vertcat(mode, To_pred_ph[k])))
                P_k = max(P_k, 0.0)
                
                # Temperature prediction
                Tz_core_k = float(temp_dnns['core'](ca.vertcat(mode, Tz_c_k[0])) + ae['core'])
                
                # Energy cost term
                energy_cost = (w[0] * price_ph[k] * P_k * mpc.dt / 3600.0 / 1000.0) ** 2
                
                # Temperature violation term
                t_hour = int(((mpc.time + k * mpc.dt) % 86400) / 3600)
                Tz_upper = mpc.T_upper[t_hour]
                delta_Thigh = max(0, Tz_core_k - Tz_upper)
                temp_penalty = (w[1] * delta_Thigh) ** 2
                
                # SOC penalty
                delta_SOC_low = max(0, 0.2 - SOC_next)
                soc_penalty = 0  # SOC handled via hard constraint below
                
                total_obj += energy_cost + temp_penalty + soc_penalty
                
                # ── NEW: Power budget penalty ──
                budget_violation = max(0.0, P_k - P_ref_W[k])
                total_obj += w_budget * (budget_violation / mu_bar_W) ** 2
                
                # SOC hard constraint via penalty
                if SOC_next < 0.200:
                    penalties += 10000 * (0.200 - SOC_next)
                elif SOC_next > 0.999:
                    penalties += 10000 * (SOC_next - 0.999)
                
                # Update state for next step
                SOC_k.insert(0, SOC_next)
                SOC_k.pop()
                Tz_c_k.insert(0, Tz_core_k)
                Tz_c_k.pop()
            
            return total_obj + penalties,
        
        # ── Run DEAP GA ──
        import random
        from deap import base, creator, tools, algorithms
        
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)
        
        toolbox = base.Toolbox()
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         lambda: random.uniform(-1.4, 2.4), n=mpc.PH)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        
        def discrete_mutation(individual):
            for i in range(len(individual)):
                individual[i] = round(individual[i])
                individual[i] = max(u_lb_occ[i], min(individual[i], u_ub_occ[i]))
            return individual,
        
        toolbox.register("mutate", discrete_mutation)
        toolbox.register("select", tools.selTournament, tournsize=5)
        toolbox.register("evaluate", evaluate_individual_with_budget)
        
        population = toolbox.population(n=200)
        algorithms.eaSimple(population, toolbox, cxpb=0.8, mutpb=0.3, ngen=100, verbose=False)
        
        best = tools.selBest(population, k=1)[0]
        best = [max(lbx, min(round(val), ubx))
                for val, lbx, ubx in zip(best, u_lb_occ, u_ub_occ)]
        best_obj = evaluate_individual_with_budget(best)[0]
        
        # Compute actual budget violation for logging
        SOC_track = SOC_his[0]
        for k in range(mpc.PH):
            P_k = float(power_dnn(ca.vertcat(best[k], To_pred_ph[k])))
            P_k = max(P_k, 0.0)
            violation = max(0, P_k - P_ref_W[k])
            if violation > 10.0:
                logger.info(
                    f"    Step {k}: mode={best[k]}, P={P_k/1000:.2f}kW, "
                    f"budget={P_ref_W[k]/1000:.2f}kW, μ={violation/1000:.2f}kW"
                )
        
        res = {'x': best, 'f': best_obj}
        solver_status = {'return_status': 'OPTIMAL'}
        
        logger.info(f"  Building B optimal modes: {best[:6]}...")
        
        return res, solver_status

    # ══════════════════════════════════════════════════════════════════════
    # Utility
    # ══════════════════════════════════════════════════════════════════════
    def get_predicted_power_kW(self, mode_sequence: List[int], Toa_forecast: List[float]) -> List[float]:
        """Evaluate power trajectory for a given mode sequence using DNN."""
        power_dnn = self._get_power_dnn()
        P_kW = []
        for k in range(len(mode_sequence)):
            Toa = Toa_forecast[min(k, len(Toa_forecast) - 1)]
            P_W = float(power_dnn(ca.vertcat(mode_sequence[k], Toa)))
            P_kW.append(max(P_W, 0.0) / 1000.0)
        return P_kW

    def get_predicted_SOC(self, mode_sequence: List[int], SOC_initial: float) -> List[float]:
        """Evaluate SOC trajectory for a given mode sequence using DNN."""
        SOC_dnn = self._get_SOC_dnn()
        SOC_traj = []
        SOC_k = SOC_initial
        for k in range(len(mode_sequence)):
            SOC_k = float(SOC_dnn(ca.vertcat(mode_sequence[k], SOC_k)))
            SOC_k = np.clip(SOC_k, 0.0, 1.0)
            SOC_traj.append(SOC_k)
        return SOC_traj
