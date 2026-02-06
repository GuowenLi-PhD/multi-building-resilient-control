"""
Building A Hierarchical MPC Wrapper

Extends the existing Building A MPC (mpc_a.mpc_case) with:
  1. compute_flexibility_bands() — two-pass MPC for P̲ and P̄ trajectories
  2. optimize_with_budget()     — original MPC + soft power budget constraint

Building A characteristics:
  - AHU-VAV-Chiller system (no TES)
  - dt = 900 s (15 min), PH = 4 (1-hour look-ahead)
  - 11 decision variables per step: [bcp, bahu, Tchw, Tcw, Tsa, V_core..V_west, ε]
  - IPOPT (NLP) solver via CasADi
  - Polynomial power models (fan + chiller plant)
  - ARX zone temperature models
  - Adaptive MPC mode for DoS attack on core zone VAV

Author: Guowen Li, AI Assistant
Date: 2025-02
"""

import casadi as ca
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
import logging
import copy

logger = logging.getLogger(__name__)


class BuildingAHierarchicalMPC:
    """
    Hierarchical wrapper for Building A's MPC.
    
    Delegates core optimization to the existing mpc_a.mpc_case patterns,
    but adds:
    - Two-pass flexibility band computation (min/max power under comfort constraints)
    - Soft power budget constraint: P(k) ≤ P_ref(k) + μ(k),
      with ω_budget · (μ(k) / μ̄(k))² added to the local objective
    
    Parameters
    ----------
    mpc : mpc_a.mpc_case instance (already initialized)
    budget_config : dict with keys:
        - 'w_budget'  : float  (weight for budget violation penalty, default 50.0)
        - 'mu_bar_kW' : float  (normalization for μ, default 10.0 kW)
        - 'mu_max_kW' : float  (maximum allowed budget violation, default 20.0 kW)
    """

    def __init__(self, mpc, budget_config: Optional[Dict] = None):
        self.mpc = mpc  # existing mpc_a.mpc_case instance
        
        cfg = budget_config or {}
        self.w_budget  = float(cfg.get('w_budget', 50.0))
        self.mu_bar_W  = float(cfg.get('mu_bar_kW', 10.0)) * 1000.0  # kW → W
        self.mu_max_W  = float(cfg.get('mu_max_kW', 20.0)) * 1000.0
        
        # Store latest flexibility bands
        self._P_lower_W: List[float] = []
        self._P_upper_W: List[float] = []

    # ══════════════════════════════════════════════════════════════════════
    # 1. FLEXIBILITY BAND COMPUTATION (Two-Pass MPC)
    # ══════════════════════════════════════════════════════════════════════
    def compute_flexibility_bands(self, PH_agg: int, dt_agg: float) -> Tuple[List[float], List[float]]:
        """
        Compute power flexibility bands at aggregator time resolution.
        
        Runs two optimization passes:
          Pass 1 (min power): Minimize ∑P(k) subject to comfort constraints
          Pass 2 (max power): Maximize ∑P(k) subject to comfort constraints
          
        Both passes use the current states/predictors from the MPC instance.
        
        Parameters
        ----------
        PH_agg : int
            Aggregator prediction horizon (number of steps)
        dt_agg : float
            Aggregator timestep [s] (typically 3600 s)
            
        Returns
        -------
        (P_lower_kW, P_upper_kW) : tuple of lists, length PH_agg, in kW
        """
        mpc = self.mpc
        
        # Number of Building A steps per aggregator step
        steps_per_agg = max(1, int(dt_agg / mpc.dt))
        
        # We need PH_agg * steps_per_agg local steps total, but Building A's PH
        # is typically only 4 (1 hour).  So we compute bands for min(PH_agg, ...)
        # and pad the rest.
        local_PH = mpc.PH  # typically 4 steps at 15 min = 1 hour
        
        # Run the two passes at the local MPC resolution
        P_min_local = self._solve_power_bound_pass('min')   # length = local_PH, in W
        P_max_local = self._solve_power_bound_pass('max')   # length = local_PH, in W
        
        # Resample from local resolution (15 min) to aggregator resolution (1 hour)
        P_lower_kW = []
        P_upper_kW = []
        
        for k_agg in range(PH_agg):
            # Local steps corresponding to this aggregator step
            k_start = k_agg * steps_per_agg
            k_end = min(k_start + steps_per_agg, local_PH)
            
            if k_start < local_PH:
                # Average power over local steps within this aggregator step
                lo_vals = P_min_local[k_start:k_end]
                hi_vals = P_max_local[k_start:k_end]
                P_lower_kW.append(np.mean(lo_vals) / 1000.0 if lo_vals else 0.0)
                P_upper_kW.append(np.mean(hi_vals) / 1000.0 if hi_vals else 0.0)
            else:
                # Beyond local PH: repeat last value
                P_lower_kW.append(P_lower_kW[-1] if P_lower_kW else 0.0)
                P_upper_kW.append(P_upper_kW[-1] if P_upper_kW else 0.0)
        
        # Ensure valid band
        for k in range(PH_agg):
            if P_lower_kW[k] > P_upper_kW[k]:
                mid = 0.5 * (P_lower_kW[k] + P_upper_kW[k])
                P_lower_kW[k] = mid
                P_upper_kW[k] = mid
        
        self._P_lower_W = [p * 1000.0 for p in P_lower_kW]
        self._P_upper_W = [p * 1000.0 for p in P_upper_kW]
        
        logger.info(
            f"  Building A flexibility: [{P_lower_kW[0]:.1f}, {P_upper_kW[0]:.1f}] kW (step 0)"
        )
        
        return P_lower_kW, P_upper_kW

    def _solve_power_bound_pass(self, direction: str) -> List[float]:
        """
        Solve one pass of the flexibility band computation.
        
        Parameters
        ----------
        direction : 'min' or 'max'
        
        Returns
        -------
        P_trajectory : list of float, length = mpc.PH, in Watts
        """
        mpc = self.mpc
        time = mpc.time
        
        # ── Retrieve states/predictors (same as optimize()) ──
        To_pred_ph  = mpc.predictor['Toa']
        RHo_pred_ph = mpc.predictor['RHoa']
        
        Tz_core_his = np.array(mpc.states['Tz_core_his_meas'][:]) - 273.15
        Tz_east_his = np.array(mpc.states['Tz_east_his_meas'][:]) - 273.15
        Tz_north_his = np.array(mpc.states['Tz_north_his_meas'][:]) - 273.15
        Tz_south_his = np.array(mpc.states['Tz_south_his_meas'][:]) - 273.15
        Tz_west_his = np.array(mpc.states['Tz_west_his_meas'][:]) - 273.15
        To_his = np.array(mpc.states['To_his_meas'][:])
        
        Tz_core_pred_his = np.array(mpc.states['Tz_core_his_pred']) - 273.15
        Tz_east_pred_his = np.array(mpc.states['Tz_east_his_pred']) - 273.15
        Tz_north_pred_his = np.array(mpc.states['Tz_north_his_pred']) - 273.15
        Tz_south_pred_his = np.array(mpc.states['Tz_south_his_pred']) - 273.15
        Tz_west_pred_his = np.array(mpc.states['Tz_west_his_pred']) - 273.15
        
        # ── Autocorrection ──
        n_his = len(Tz_core_his)
        ae = {z: 0.0 for z in ['core','east','north','south','west']}
        for k in range(n_his):
            ae['core']  += (Tz_core_his[k] - Tz_core_pred_his[k]) / n_his
            ae['east']  += (Tz_east_his[k] - Tz_east_pred_his[k]) / n_his
            ae['north'] += (Tz_north_his[k] - Tz_north_pred_his[k]) / n_his
            ae['south'] += (Tz_south_his[k] - Tz_south_pred_his[k]) / n_his
            ae['west']  += (Tz_west_his[k] - Tz_west_pred_his[k]) / n_his
        
        # ── Temperature bounds ──
        T_upper = np.array([30.0] * 24)
        T_upper[mpc.occ_start:mpc.occ_end] = 24.0
        T_lower = np.array([18.0] * 24)
        T_lower[mpc.occ_start:mpc.occ_end] = 20.0
        
        # ── CasADi variables (same 11 per step as original) ──
        n_inp = mpc.number_inputs  # 11
        U = ca.MX.sym("U", n_inp * mpc.PH)
        
        # ── Build predicted trajectories ──
        Tz_c_k = list(Tz_core_his)
        Tz_e_k = list(Tz_east_his)
        Tz_n_k = list(Tz_north_his)
        Tz_s_k = list(Tz_south_his)
        Tz_w_k = list(Tz_west_his)
        To_k   = list(To_his)
        
        P_ph  = [None] * mpc.PH
        Tz_core_ph = [None] * mpc.PH
        Tz_east_ph = [None] * mpc.PH
        Tz_north_ph = [None] * mpc.PH
        Tz_south_ph = [None] * mpc.PH
        Tz_west_ph = [None] * mpc.PH
        
        u_prev = mpc.x_opt_0
        
        for k in range(mpc.PH):
            u = U[k * n_inp:(k + 1) * n_inp]
            
            Tz_avg_k = (Tz_c_k[-1] + Tz_e_k[-1] + Tz_n_k[-1] + Tz_s_k[-1] + Tz_w_k[-1]) / 5.0
            
            # Power prediction
            P_ph[k] = u[0] * mpc.ChillerPlantPower(
                mpc.params_chiller, u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9],
                Tz_avg_k, To_pred_ph[k], RHo_pred_ph[k]
            ) + u[1] * mpc.FanPower(mpc.params_fan, u[5], u[6], u[7], u[8], u[9])
            
            # Zone temperature predictions
            Tz_core_ph[k] = mpc.ZoneTemperature(
                mpc.params_core, u[4], Tz_e_k[-1], Tz_n_k[-1], Tz_s_k[-1], Tz_w_k[-1],
                u[5], u[6], u[7], u[8], u[9],
                Tz_c_k[0], Tz_c_k[1], Tz_c_k[2], Tz_c_k[3],
                To_k[0], To_k[1], To_k[2], To_k[3], u[0], u[1], ae['core']
            )
            Tz_east_ph[k] = mpc.ZoneTemperature(
                mpc.params_east, u[4], Tz_c_k[-1], Tz_n_k[-1], Tz_s_k[-1], Tz_w_k[-1],
                u[6], u[5], u[7], u[8], u[9],
                Tz_e_k[0], Tz_e_k[1], Tz_e_k[2], Tz_e_k[3],
                To_k[0], To_k[1], To_k[2], To_k[3], u[0], u[1], ae['east']
            )
            Tz_north_ph[k] = mpc.ZoneTemperature(
                mpc.params_north, u[4], Tz_c_k[-1], Tz_e_k[-1], Tz_s_k[-1], Tz_w_k[-1],
                u[7], u[5], u[6], u[8], u[9],
                Tz_n_k[0], Tz_n_k[1], Tz_n_k[2], Tz_n_k[3],
                To_k[0], To_k[1], To_k[2], To_k[3], u[0], u[1], ae['north']
            )
            Tz_south_ph[k] = mpc.ZoneTemperature(
                mpc.params_south, u[4], Tz_c_k[-1], Tz_e_k[-1], Tz_n_k[-1], Tz_w_k[-1],
                u[8], u[5], u[6], u[7], u[9],
                Tz_s_k[0], Tz_s_k[1], Tz_s_k[2], Tz_s_k[3],
                To_k[0], To_k[1], To_k[2], To_k[3], u[0], u[1], ae['south']
            )
            Tz_west_ph[k] = mpc.ZoneTemperature(
                mpc.params_west, u[4], Tz_c_k[-1], Tz_e_k[-1], Tz_n_k[-1], Tz_s_k[-1],
                u[9], u[5], u[6], u[7], u[8],
                Tz_w_k[0], Tz_w_k[1], Tz_w_k[2], Tz_w_k[3],
                To_k[0], To_k[1], To_k[2], To_k[3], u[0], u[1], ae['west']
            )
            
            # Propagate historical buffers
            Tz_c_k = Tz_c_k[1:] + [Tz_core_ph[k]]
            Tz_e_k = Tz_e_k[1:] + [Tz_east_ph[k]]
            Tz_n_k = Tz_n_k[1:] + [Tz_north_ph[k]]
            Tz_s_k = Tz_s_k[1:] + [Tz_south_ph[k]]
            Tz_w_k = Tz_w_k[1:] + [Tz_west_ph[k]]
        
        # ── Objective: minimize or maximize total power ──
        sign = 1.0 if direction == 'min' else -1.0
        obj_power = sign * ca.sum1(ca.vertcat(*P_ph))
        
        # Add a small comfort regularization to prevent extreme solutions
        comfort_reg = 0.0
        for k in range(mpc.PH):
            eps = U[k * n_inp + 10]  # existing comfort slack
            comfort_reg += eps ** 2
        
        f_total = obj_power + 0.01 * comfort_reg
        
        obj_fn = ca.Function('f_bound', [U], [f_total])
        f = obj_fn(U)
        
        # ── Constraints: same as original MPC ──
        g, lbg, ubg = [], [], []
        u_lb, u_ub = [], []
        
        for k in range(mpc.PH):
            t = int(time + k * mpc.dt)
            t_hour = int((t % 86400) / 3600)
            
            if t_hour >= mpc.occ_start and t_hour < mpc.occ_end:
                if mpc.dos_attack_core_VAV:
                    u_lb += [1, 1, 5, 15.6, 11.8, 0.00, 0.05, 0.05, 0.05, 0.04, 0.00]
                    u_ub += [1, 1, 10, 29.4, 18, 0.01, 0.90, 0.95, 0.95, 0.70, 0.10]
                else:
                    u_lb += [1, 1, 5, 15.6, 11.8, 0.23, 0.05, 0.05, 0.05, 0.04, 0.00]
                    u_ub += [1, 1, 10, 29.4, 18, 2.80, 0.90, 0.95, 0.95, 0.70, 0.10]
            else:
                u_lb += [0, 0, 5, 20, 11.8, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
                u_ub += [0, 0, 10, 20, 18, 0.01, 0.01, 0.01, 0.01, 0.01, 0.10]
            
            # Temperature constraints (same as original)
            eps = U[n_inp * k + 10]
            Tchw = U[n_inp * k + 2]
            Tcw  = U[n_inp * k + 3]
            Tsa  = U[n_inp * k + 4]
            
            g += [
                Tz_core_ph[k] + eps, Tz_core_ph[k] - eps,
                Tz_east_ph[k] + eps, Tz_east_ph[k] - eps,
                Tz_north_ph[k] + eps, Tz_north_ph[k] - eps,
                Tz_south_ph[k] + eps, Tz_south_ph[k] - eps,
                Tz_west_ph[k] + eps, Tz_west_ph[k] - eps,
            ]
            g += [(Tsa - 11.8) / 6.2 * 5 + 5 - Tchw]
            
            Td = To_pred_ph[k]
            RH = RHo_pred_ph[k] * 100
            Twet = (Td * math.atan(0.151977 * (RH + 8.313659)**0.5) +
                    math.atan(Td + RH) - math.atan(RH - 1.676331) +
                    0.00391838 * RH**1.5 * math.atan(0.023101 * RH) - 4.686035)
            g += [Tcw - Twet]
            
            lbg += [T_lower[t_hour], 0.] * 5
            lbg += [-0.1]
            lbg += [1.5]
            ubg += [ca.inf, T_upper[t_hour]] * 5
            ubg += [1.]
            ubg += [3.]
        
        # ── Solve ──
        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 200,
            "print_time": False,
        }
        solver = ca.nlpsol("flex_bound", "ipopt", {"x": U, "f": f, "g": ca.vertcat(*g)}, opts)
        
        u_ini = mpc.u_start if hasattr(mpc, 'u_start') else [0.5 * (l + h) for l, h in zip(u_lb, u_ub)]
        
        try:
            res = solver(x0=u_ini, lbx=u_lb, ubx=u_ub, lbg=lbg, ubg=ubg)
            x_opt = np.array(res['x']).flatten()
        except Exception as e:
            logger.warning(f"Flexibility {direction}-pass failed: {e}. Using heuristic bounds.")
            return self._heuristic_power_bounds(direction)
        
        # Extract power trajectory using the power models with optimal controls
        P_traj_W = []
        Tz_c_eval = list(Tz_core_his)
        Tz_e_eval = list(Tz_east_his)
        Tz_n_eval = list(Tz_north_his)
        Tz_s_eval = list(Tz_south_his)
        Tz_w_eval = list(Tz_west_his)
        
        for k in range(mpc.PH):
            u_k = x_opt[k * n_inp:(k + 1) * n_inp]
            Tz_avg = (Tz_c_eval[-1] + Tz_e_eval[-1] + Tz_n_eval[-1] + Tz_s_eval[-1] + Tz_w_eval[-1]) / 5.0
            
            P_k = (u_k[0] * float(mpc.ChillerPlantPower(
                       mpc.params_chiller, u_k[2], u_k[3], u_k[4],
                       u_k[5], u_k[6], u_k[7], u_k[8], u_k[9],
                       Tz_avg, To_pred_ph[k], RHo_pred_ph[k])) +
                   u_k[1] * float(mpc.FanPower(
                       mpc.params_fan, u_k[5], u_k[6], u_k[7], u_k[8], u_k[9])))
            P_traj_W.append(max(P_k, 0.0))
            
            # Propagate temperature predictions for next step
            Tz_c_eval = Tz_c_eval[1:] + [float(mpc.ZoneTemperature(
                mpc.params_core, u_k[4], Tz_e_eval[-1], Tz_n_eval[-1], Tz_s_eval[-1], Tz_w_eval[-1],
                u_k[5], u_k[6], u_k[7], u_k[8], u_k[9],
                Tz_c_eval[0], Tz_c_eval[1], Tz_c_eval[2], Tz_c_eval[3],
                To_his[0], To_his[1], To_his[2], To_his[3], u_k[0], u_k[1], ae['core']))]
            Tz_e_eval = Tz_e_eval[1:] + [Tz_e_eval[-1]]  # simplified propagation
            Tz_n_eval = Tz_n_eval[1:] + [Tz_n_eval[-1]]
            Tz_s_eval = Tz_s_eval[1:] + [Tz_s_eval[-1]]
            Tz_w_eval = Tz_w_eval[1:] + [Tz_w_eval[-1]]
        
        return P_traj_W

    def _heuristic_power_bounds(self, direction: str) -> List[float]:
        """Fallback heuristic bounds if optimization fails."""
        mpc = self.mpc
        if direction == 'min':
            # Minimum ventilation power only
            P_min_W = float(mpc.FanPower(mpc.params_fan, 0.23, 0.05, 0.05, 0.05, 0.04))
            return [max(P_min_W, 500.0)] * mpc.PH
        else:
            # Maximum power estimate at full capacity
            P_max_W = float(mpc.FanPower(mpc.params_fan, 2.80, 0.90, 0.95, 0.95, 0.70))
            P_max_W += 15000.0  # rough chiller estimate
            return [P_max_W] * mpc.PH

    # ══════════════════════════════════════════════════════════════════════
    # 2. OPTIMIZE WITH POWER BUDGET CONSTRAINT
    # ══════════════════════════════════════════════════════════════════════
    def optimize_with_budget(self, P_ref_kW: Optional[List[float]] = None):
        """
        Run MPC optimization with an additional soft power budget constraint.
        
        If P_ref_kW is None, runs the original MPC without budget constraint.
        
        Adds to the original formulation:
          - Slack variable μ(k) ≥ 0 for each step k
          - Constraint: P(k) ≤ P_ref(k) + μ(k)  [in Watts]
          - Penalty:    ω_budget · (μ(k) / μ̄)² added to objective
        
        Parameters
        ----------
        P_ref_kW : list of float or None
            Power budget trajectory [kW]. Length ≥ mpc.PH.
            If None, runs without budget constraint.
        
        Returns
        -------
        res : CasADi result dict (same format as original mpc.optimize())
        """
        mpc = self.mpc
        
        if P_ref_kW is None:
            # No budget → run original MPC
            return mpc.optimize()
        
        # Convert to Watts and align to local PH
        P_ref_W = [p * 1000.0 for p in P_ref_kW[:mpc.PH]]
        while len(P_ref_W) < mpc.PH:
            P_ref_W.append(P_ref_W[-1])
        
        time = mpc.time
        
        # ── Get states/predictors (same as original optimize) ──
        To_pred_ph  = mpc.predictor['Toa']
        RHo_pred_ph = mpc.predictor['RHoa']
        price_ph    = mpc.predictor['price']
        
        Tz_core_his = np.array(mpc.states['Tz_core_his_meas'][:]) - 273.15
        Tz_east_his = np.array(mpc.states['Tz_east_his_meas'][:]) - 273.15
        Tz_north_his = np.array(mpc.states['Tz_north_his_meas'][:]) - 273.15
        Tz_south_his = np.array(mpc.states['Tz_south_his_meas'][:]) - 273.15
        Tz_west_his = np.array(mpc.states['Tz_west_his_meas'][:]) - 273.15
        To_his = np.array(mpc.states['To_his_meas'][:])
        
        Tz_core_pred_his = np.array(mpc.states['Tz_core_his_pred']) - 273.15
        Tz_east_pred_his = np.array(mpc.states['Tz_east_his_pred']) - 273.15
        Tz_north_pred_his = np.array(mpc.states['Tz_north_his_pred']) - 273.15
        Tz_south_pred_his = np.array(mpc.states['Tz_south_his_pred']) - 273.15
        Tz_west_pred_his = np.array(mpc.states['Tz_west_his_pred']) - 273.15
        
        # Autocorrection
        n_his = len(Tz_core_his)
        ae = {z: 0.0 for z in ['core','east','north','south','west']}
        for k_h in range(n_his):
            ae['core']  += (Tz_core_his[k_h] - Tz_core_pred_his[k_h]) / n_his
            ae['east']  += (Tz_east_his[k_h] - Tz_east_pred_his[k_h]) / n_his
            ae['north'] += (Tz_north_his[k_h] - Tz_north_pred_his[k_h]) / n_his
            ae['south'] += (Tz_south_his[k_h] - Tz_south_pred_his[k_h]) / n_his
            ae['west']  += (Tz_west_his[k_h] - Tz_west_pred_his[k_h]) / n_his
        mpc._autoerror = ae
        
        u_prev = mpc.x_opt_0
        
        # ── Decision variables: original U (11*PH) + budget slack MU (PH) ──
        n_inp = mpc.number_inputs  # 11
        U  = ca.MX.sym("U", n_inp * mpc.PH)
        MU = ca.MX.sym("MU", mpc.PH)          # budget violation slack [W]
        X_all = ca.vertcat(U, MU)
        
        # Temperature bounds
        T_upper = np.array([30.0] * 24)
        T_upper[mpc.occ_start:mpc.occ_end] = 24.0
        T_lower = np.array([18.0] * 24)
        T_lower[mpc.occ_start:mpc.occ_end] = 20.0
        
        # ── Build predicted trajectories ──
        Tz_c_k = list(Tz_core_his)
        Tz_e_k = list(Tz_east_his)
        Tz_n_k = list(Tz_north_his)
        Tz_s_k = list(Tz_south_his)
        Tz_w_k = list(Tz_west_his)
        To_k = list(To_his)
        
        P_ph = [None] * mpc.PH
        Tz_core_ph = [None] * mpc.PH
        Tz_east_ph = [None] * mpc.PH
        Tz_north_ph = [None] * mpc.PH
        Tz_south_ph = [None] * mpc.PH
        Tz_west_ph = [None] * mpc.PH
        
        fval = []
        
        for k in range(mpc.PH):
            u = U[k * n_inp:(k + 1) * n_inp]
            mu_k = MU[k]
            
            Tz_avg_k = (Tz_c_k[-1] + Tz_e_k[-1] + Tz_n_k[-1] + Tz_s_k[-1] + Tz_w_k[-1]) / 5.0
            
            # Power prediction
            P_ph[k] = u[0] * mpc.ChillerPlantPower(
                mpc.params_chiller, u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9],
                Tz_avg_k, To_pred_ph[k], RHo_pred_ph[k]
            ) + u[1] * mpc.FanPower(mpc.params_fan, u[5], u[6], u[7], u[8], u[9])
            
            # Zone temperatures (same structure as original)
            Tz_core_ph[k] = mpc.ZoneTemperature(
                mpc.params_core, u[4], Tz_e_k[-1], Tz_n_k[-1], Tz_s_k[-1], Tz_w_k[-1],
                u[5], u[6], u[7], u[8], u[9],
                Tz_c_k[0], Tz_c_k[1], Tz_c_k[2], Tz_c_k[3],
                To_k[0], To_k[1], To_k[2], To_k[3], u[0], u[1], ae['core']
            )
            Tz_east_ph[k] = mpc.ZoneTemperature(
                mpc.params_east, u[4], Tz_c_k[-1], Tz_n_k[-1], Tz_s_k[-1], Tz_w_k[-1],
                u[6], u[5], u[7], u[8], u[9],
                Tz_e_k[0], Tz_e_k[1], Tz_e_k[2], Tz_e_k[3],
                To_k[0], To_k[1], To_k[2], To_k[3], u[0], u[1], ae['east']
            )
            Tz_north_ph[k] = mpc.ZoneTemperature(
                mpc.params_north, u[4], Tz_c_k[-1], Tz_e_k[-1], Tz_s_k[-1], Tz_w_k[-1],
                u[7], u[5], u[6], u[8], u[9],
                Tz_n_k[0], Tz_n_k[1], Tz_n_k[2], Tz_n_k[3],
                To_k[0], To_k[1], To_k[2], To_k[3], u[0], u[1], ae['north']
            )
            Tz_south_ph[k] = mpc.ZoneTemperature(
                mpc.params_south, u[4], Tz_c_k[-1], Tz_e_k[-1], Tz_n_k[-1], Tz_w_k[-1],
                u[8], u[5], u[6], u[7], u[9],
                Tz_s_k[0], Tz_s_k[1], Tz_s_k[2], Tz_s_k[3],
                To_k[0], To_k[1], To_k[2], To_k[3], u[0], u[1], ae['south']
            )
            Tz_west_ph[k] = mpc.ZoneTemperature(
                mpc.params_west, u[4], Tz_c_k[-1], Tz_e_k[-1], Tz_n_k[-1], Tz_s_k[-1],
                u[9], u[5], u[6], u[7], u[8],
                Tz_w_k[0], Tz_w_k[1], Tz_w_k[2], Tz_w_k[3],
                To_k[0], To_k[1], To_k[2], To_k[3], u[0], u[1], ae['west']
            )
            
            # Propagate
            Tz_c_k = Tz_c_k[1:] + [Tz_core_ph[k]]
            Tz_e_k = Tz_e_k[1:] + [Tz_east_ph[k]]
            Tz_n_k = Tz_n_k[1:] + [Tz_north_ph[k]]
            Tz_s_k = Tz_s_k[1:] + [Tz_south_ph[k]]
            Tz_w_k = Tz_w_k[1:] + [Tz_west_ph[k]]
            
            # ── Original objective terms ──
            normalizer = [1.0 / (mpc.u_ub[i] - mpc.u_lb[i]) for i in range(n_inp)]
            du_k = u - u_prev
            u_prev = u
            du_norm = [normalizer[i] * du_k[i] for i in range(2, n_inp - 1)]
            du_nom2 = ca.sumsqr(ca.vertcat(*du_norm)) / len(du_norm)
            
            fo = (mpc.w[0] * price_ph[k] * P_ph[k] * mpc.dt / 3600.0 / 1000.0 +
                  mpc.w[1] * u[-1]**2 +
                  mpc.w[2] * du_nom2)
            
            # ── NEW: Budget violation penalty ──
            # ω_budget · (μ / μ̄)²
            fo += self.w_budget * (mu_k / self.mu_bar_W) ** 2
            
            fval.append(fo)
        
        fval_sum = ca.sum1(ca.vertcat(*fval))
        obj_fn = ca.Function('fval', [X_all], [fval_sum])
        f = obj_fn(X_all)
        
        # ── Constraints ──
        g, lbg, ubg = [], [], []
        u_lb_all, u_ub_all = [], []
        
        for k in range(mpc.PH):
            t = int(time + k * mpc.dt)
            t_hour = int((t % 86400) / 3600)
            
            if t_hour >= mpc.occ_start and t_hour < mpc.occ_end:
                if mpc.dos_attack_core_VAV:
                    u_lb_all += [1, 1, 5, 15.6, 11.8, 0.00, 0.05, 0.05, 0.05, 0.04, 0.00]
                    u_ub_all += [1, 1, 10, 29.4, 18, 0.01, 0.90, 0.95, 0.95, 0.70, 0.10]
                else:
                    u_lb_all += [1, 1, 5, 15.6, 11.8, 0.23, 0.05, 0.05, 0.05, 0.04, 0.00]
                    u_ub_all += [1, 1, 10, 29.4, 18, 2.80, 0.90, 0.95, 0.95, 0.70, 0.10]
            else:
                u_lb_all += [0, 0, 5, 20, 11.8, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
                u_ub_all += [0, 0, 10, 20, 18, 0.01, 0.01, 0.01, 0.01, 0.01, 0.10]
            
            # Temperature constraints
            eps = U[n_inp * k + 10]
            Tchw = U[n_inp * k + 2]
            Tcw  = U[n_inp * k + 3]
            Tsa  = U[n_inp * k + 4]
            
            g += [
                Tz_core_ph[k] + eps, Tz_core_ph[k] - eps,
                Tz_east_ph[k] + eps, Tz_east_ph[k] - eps,
                Tz_north_ph[k] + eps, Tz_north_ph[k] - eps,
                Tz_south_ph[k] + eps, Tz_south_ph[k] - eps,
                Tz_west_ph[k] + eps, Tz_west_ph[k] - eps,
            ]
            g += [(Tsa - 11.8) / 6.2 * 5 + 5 - Tchw]
            
            Td = To_pred_ph[k]
            RH = RHo_pred_ph[k] * 100
            Twet = (Td * math.atan(0.151977 * (RH + 8.313659)**0.5) +
                    math.atan(Td + RH) - math.atan(RH - 1.676331) +
                    0.00391838 * RH**1.5 * math.atan(0.023101 * RH) - 4.686035)
            g += [Tcw - Twet]
            
            lbg += [T_lower[t_hour], 0.] * 5
            lbg += [-0.1]
            lbg += [1.5]
            ubg += [ca.inf, T_upper[t_hour]] * 5
            ubg += [1.]
            ubg += [3.]
            
            # ── NEW: Budget constraint  P(k) - P_ref(k) ≤ μ(k) ──
            # Rearranged: P(k) - μ(k) ≤ P_ref(k)
            g.append(P_ph[k] - MU[k])
            lbg.append(-ca.inf)
            ubg.append(P_ref_W[k])
        
        # Bounds for μ: [0, mu_max]
        mu_lb = [0.0] * mpc.PH
        mu_ub = [self.mu_max_W] * mpc.PH
        
        # Combined bounds
        lbx = u_lb_all + mu_lb
        ubx = u_ub_all + mu_ub
        
        # Initial guess
        u_ini = list(mpc.u_start) + [0.0] * mpc.PH
        
        # ── Solve ──
        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 200,
            "print_time": False,
        }
        solver = ca.nlpsol(
            "mpc_a_budget", "ipopt",
            {"x": X_all, "f": f, "g": ca.vertcat(*g)}, opts
        )
        
        res_full = solver(x0=u_ini, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        
        # Extract original format result for compatibility
        x_all_opt = np.array(res_full['x']).flatten()
        x_opt_U  = x_all_opt[:n_inp * mpc.PH]
        x_opt_MU = x_all_opt[n_inp * mpc.PH:]
        
        # Log budget tracking
        for k in range(mpc.PH):
            if x_opt_MU[k] > 1.0:  # >1W violation
                logger.info(
                    f"    Step {k}: budget violation μ={x_opt_MU[k]/1000:.2f}kW"
                )
        
        # Package result compatible with original MPC interface
        res = {'x': ca.DM(x_opt_U), 'f': res_full['f']}
        
        # Store budget violation for logging
        res['budget_violation_W'] = x_opt_MU
        
        return res
