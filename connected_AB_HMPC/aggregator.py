"""
Aggregator MPC — Upper-Level Power Budget Allocation (Eqs. 25–29)

Solves the log-utility convex program at each coordination interval:

    min  -∑_i ∑_k  ω_i · log(P_i,ref(t+k) + δ)

    s.t.  ∑_i P_i,ref(t+k)  ≤  P_feeder(t+k)         ∀k   (feeder limit)
          P̲_i(t+k) ≤ P_i,ref(t+k) ≤ P̄_i(t+k)        ∀i,k (flexibility bands)

Designed for N buildings.  Convex program → guaranteed global optimum.

Author: Guowen Li, AI Assistant
Date: 2025-02
"""

import casadi as ca
import numpy as np
from typing import Dict, List, Optional
import logging

from data_models import (
    FlexibilityReport, PowerBudget, AggregatorDecision, EnergyPriority
)

logger = logging.getLogger(__name__)


class AggregatorMPC:
    """
    Upper-level coordinator for multi-building power budget allocation.
    
    Parameters
    ----------
    config : dict
        Must contain 'aggregator' sub-dict with:
        - prediction_horizon : int
        - dt                 : float (seconds)
        - feeder_limit_kW    : float or list[float]
        - delta              : float (log regularizer, default 0.1 kW)
        - solver_print       : bool (default False)
    """

    def __init__(self, config: Dict):
        agg_cfg = config.get('aggregator', config)
        
        self.PH    = int(agg_cfg['prediction_horizon'])
        self.dt    = float(agg_cfg['dt'])
        self.delta = float(agg_cfg.get('delta', 0.1))
        self.print_solver = bool(agg_cfg.get('solver_print', False))
        
        # Feeder limit: scalar or per-step trajectory [kW]
        fl = agg_cfg['feeder_limit_kW']
        if isinstance(fl, (int, float)):
            self.feeder_limit = [float(fl)] * self.PH
        else:
            self.feeder_limit = [float(x) for x in fl]
        
        self._building_ids: List[str] = []
        self._last_decision: Optional[AggregatorDecision] = None
        
        logger.info(
            f"✓ Aggregator initialized: PH={self.PH}, dt={self.dt}s, "
            f"feeder={self.feeder_limit[0]:.1f}kW"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════
    def allocate(
        self,
        reports: Dict[str, FlexibilityReport],
        timestamp: float
    ) -> AggregatorDecision:
        """
        Solve the aggregator allocation problem.
        
        Parameters
        ----------
        reports : dict  {building_id: FlexibilityReport}
        timestamp : float  (simulation time in seconds)
            
        Returns
        -------
        AggregatorDecision with PowerBudget for each building.
        """
        self._building_ids = sorted(reports.keys())
        N = len(self._building_ids)
        assert N > 0, "No building reports received"
        
        logger.info(f"\n{'='*70}")
        logger.info(f"AGGREGATOR ALLOCATION at t={timestamp:.0f}s ({timestamp/3600:.1f}h)")
        logger.info(f"{'='*70}")
        
        # ── Extract data ──
        priorities, P_lower, P_upper = {}, {}, {}
        
        for bid in self._building_ids:
            rpt = reports[bid]
            priorities[bid] = int(rpt.priority)
            band = rpt.flexibility_band
            P_lower[bid] = self._align_horizon(band.P_lower_kW)
            P_upper[bid] = self._align_horizon(band.P_upper_kW)
            
            logger.info(
                f"  {bid}: P_now={rpt.power_actual_kW:.2f}kW, "
                f"ω={priorities[bid]}, "
                f"band=[{P_lower[bid][0]:.1f}, {P_upper[bid][0]:.1f}]kW"
            )
        
        # ── Feasibility check ──
        for k in range(self.PH):
            sum_lb = sum(P_lower[bid][k] for bid in self._building_ids)
            if sum_lb > self.feeder_limit[k] + 1e-3:
                logger.warning(
                    f"  ⚠ Step {k}: ∑P̲={sum_lb:.1f}kW > feeder={self.feeder_limit[k]:.1f}kW"
                )
        
        # ── Solve ──
        budgets_raw, status, obj = self._solve(priorities, P_lower, P_upper)
        
        # ── Package results ──
        budgets = {}
        total_alloc = [0.0] * self.PH
        for bid in self._building_ids:
            budgets[bid] = PowerBudget(
                building_id=bid,
                P_ref_kW=budgets_raw[bid],
                timestamp=timestamp
            )
            for k in range(self.PH):
                total_alloc[k] += budgets_raw[bid][k]
        
        decision = AggregatorDecision(
            timestamp=timestamp,
            budgets=budgets,
            feeder_limit_kW=self.feeder_limit[0],
            total_allocated_kW=total_alloc,
            solver_status=status,
            objective_value=obj
        )
        
        logger.info(f"  ── Allocation (step 0) ──")
        for bid in self._building_ids:
            logger.info(f"    {bid}: P_ref = {budgets_raw[bid][0]:.2f} kW")
        logger.info(
            f"    Total: {total_alloc[0]:.2f} / {self.feeder_limit[0]:.1f} kW | "
            f"status={status}"
        )
        
        self._last_decision = decision
        return decision

    # ══════════════════════════════════════════════════════════════════════
    # CasADi Solver
    # ══════════════════════════════════════════════════════════════════════
    def _solve(self, priorities, P_lower, P_upper):
        """
        Solve the log-utility allocation via IPOPT.
        
        Variable layout:  P_ref[i * PH + k]  for building i, step k.
        """
        bids = self._building_ids
        N = len(bids)
        n_vars = N * self.PH
        
        P = ca.MX.sym("P_ref", n_vars)
        
        # ── Objective: -∑_i ∑_k ω_i · log(P_i + δ) ──
        obj = 0.0
        for i, bid in enumerate(bids):
            w_i = float(priorities[bid])
            for k in range(self.PH):
                obj -= w_i * ca.log(P[i * self.PH + k] + self.delta)
        
        # ── Constraints: feeder limit per step ──
        g, lbg, ubg = [], [], []
        for k in range(self.PH):
            g.append(sum(P[i * self.PH + k] for i in range(N)))
            lbg.append(-ca.inf)
            ubg.append(self.feeder_limit[k])
        
        # ── Variable bounds: flexibility bands ──
        lbx, ubx, x0 = [], [], []
        for i, bid in enumerate(bids):
            for k in range(self.PH):
                lb = max(P_lower[bid][k], 0.0)
                ub = P_upper[bid][k]
                lbx.append(lb)
                ubx.append(ub)
                x0.append(0.5 * (lb + ub))
        
        # ── Solve ──
        nlp = {"x": P, "f": obj, "g": ca.vertcat(*g) if g else ca.MX(0)}
        opts = {
            "ipopt.print_level": 5 if self.print_solver else 0,
            "ipopt.max_iter": 500,
            "ipopt.tol": 1e-6,
            "print_time": self.print_solver,
        }
        solver = ca.nlpsol("aggregator", "ipopt", nlp, opts)
        
        try:
            res = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            x_opt = np.array(res['x']).flatten()
            obj_val = float(res['f'])
            status = "optimal"
        except Exception as e:
            logger.error(f"Aggregator solve failed: {e}. Using proportional fallback.")
            x_opt = np.array(x0)
            obj_val = float('nan')
            status = "fallback"
        
        # ── Unpack ──
        result = {}
        for i, bid in enumerate(bids):
            result[bid] = [float(x_opt[i * self.PH + k]) for k in range(self.PH)]
        
        return result, status, obj_val

    # ══════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════
    def _align_horizon(self, traj: List[float]) -> List[float]:
        """Pad or truncate trajectory to match aggregator PH."""
        if len(traj) >= self.PH:
            return traj[:self.PH]
        pad = traj[-1] if traj else 0.0
        return traj + [pad] * (self.PH - len(traj))

    def set_feeder_limit(self, limit_kW, step: Optional[int] = None):
        """Update feeder limit (scalar for all steps, or at a specific step)."""
        if step is None:
            val = float(limit_kW) if isinstance(limit_kW, (int, float)) else limit_kW
            self.feeder_limit = [float(val)] * self.PH if isinstance(val, float) else [float(x) for x in val]
        else:
            self.feeder_limit[step] = float(limit_kW)

    def get_proportional_fallback(
        self, reports: Dict[str, FlexibilityReport]
    ) -> Dict[str, List[float]]:
        """Proportional allocation fallback if solver fails."""
        bids = sorted(reports.keys())
        actuals = {b: max(reports[b].power_actual_kW, 0.1) for b in bids}
        total = sum(actuals.values())
        return {
            b: [actuals[b] / total * self.feeder_limit[k] for k in range(self.PH)]
            for b in bids
        }
