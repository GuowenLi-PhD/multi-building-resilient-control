"""
Aggregator MPC — Upper-level coordinator for the two-building community.

Modes
─────
NORMAL           Rule-based power allocation using EMA baselines.
ATTACK_OPTIMIZE  Log-utility convex optimisation for proportionally fair
                 budget allocation subject to feeder-capacity hard constraint.

The aggregator NEVER touches individual building decision variables.
It operates only on power budgets and priority signals, preserving
true hierarchical separation (Dissertation §5).

Coordination per step:
    Buildings report  →  Aggregator allocates  →  Buildings execute

Author  : Guowen Li
Date    : 2026-02
"""

from __future__ import annotations

import time as _time

import numpy as np

from communication.messages import (
    AllocationResult,
    BuildingAStatus,
    BuildingBStatus,
)
from utils.helpers import get_logger, pad_to, seconds_to_hour, soc_target

logger = get_logger("aggregator")


class AggregatorMPC:
    """
    Hybrid rule-based / log-utility aggregator.

    Parameters
    ----------
    cfg : dict   Full system configuration (loaded YAML).
    """

    def __init__(self, cfg: dict):
        agg = cfg["aggregator"]
        fdr = cfg["feeder"]

        # Feeder
        self.P_feeder_cap = fdr["capacity_kW"]
        self.P_feeder_max = fdr["capacity_kW"] * fdr["safety_margin"]
        self.P_margin = fdr.get("power_margin_kW", 2.0)

        # Timing
        self.dt = cfg["timing"]["dt_aggregator"]
        self.PH = cfg["timing"]["prediction_horizon_agg"]

        # Attack hysteresis
        self.alpha_enter = agg["alpha_threshold_enter"]
        self.alpha_exit = agg["alpha_threshold_exit"]
        self.hyst_steps = agg["hysteresis_steps"]

        # Normal-mode buffer
        self.gamma_buf = agg["gamma_buffer"]

        # Attack multipliers β(θ)
        self.beta = {int(k): v for k, v in agg["attack_multipliers"].items()}

        # Optimisation weights
        self.w_feeder = agg["w_feeder"]
        self.w_comfort = agg["w_comfort"]
        self.w_balance = agg["w_balance"]
        self.w_tes = agg["w_tes"]

        # Building configs
        self.cfg_a = cfg["building_a"]
        self.cfg_b = cfg["building_b"]

        # ── runtime state ───────────────────────────────────────────────
        self.mode: str = "NORMAL"
        self.theta: int = 0
        self._hyst_cnt: int = 0

        # EMA baselines (initialised to reasonable defaults)
        self._P_A_base = np.full(self.PH, 8.0)
        self._P_B_base = np.full(self.PH, 12.0)

        logger.info(
            "Aggregator init  PH=%d  dt=%.0fs  P_feeder_max=%.1f kW",
            self.PH, self.dt, self.P_feeder_max,
        )

    # ═══════════════════════════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════════════════════════

    def coordinate(
        self,
        sim_time: float,
        msg_a: BuildingAStatus,
        msg_b: BuildingBStatus,
    ) -> AllocationResult:
        """Run one aggregator step: classify → mode-switch → allocate."""
        t0 = _time.perf_counter()

        self.theta = self._classify(msg_a.alpha_attack)
        self._update_mode(msg_a.alpha_attack)
        self._update_baselines(msg_a, msg_b)

        if self.mode == "NORMAL":
            res = self._rule_based(sim_time, msg_a, msg_b)
        else:
            res = self._log_utility(sim_time, msg_a, msg_b)

        res.solve_time_s = _time.perf_counter() - t0
        res.mode = self.mode
        res.theta_priority = self.theta

        logger.info(
            "t=%5.0fs  %-17s  θ=%d  P_A=%.1f  P_B=%.1f  Σ=%.1f/%.0f  "
            "dt=%.3fs",
            sim_time, self.mode, self.theta,
            res.P_A_alloc[0], res.P_B_alloc[0],
            res.P_A_alloc[0] + res.P_B_alloc[0], self.P_feeder_max,
            res.solve_time_s,
        )
        return res

    # ═══════════════════════════════════════════════════════════════════
    #  Priority & mode logic
    # ═══════════════════════════════════════════════════════════════════

    def _classify(self, alpha: float) -> int:
        """Map attack confidence α → priority level θ ∈ {0,1,2,3}."""
        if alpha < 0.3:
            return 0
        if alpha < 0.6:
            return 1
        if alpha < 0.85:
            return 2
        return 3

    def _update_mode(self, alpha: float) -> None:
        """State machine: NORMAL ↔ ATTACK_OPTIMIZE with hysteresis."""
        if self.mode == "NORMAL":
            if alpha >= self.alpha_enter:
                self.mode = "ATTACK_OPTIMIZE"
                self._hyst_cnt = 0
                logger.warning(
                    "→ ATTACK_OPTIMIZE  (α=%.2f θ=%d)", alpha, self.theta
                )
        else:
            if alpha < self.alpha_exit and self.theta == 0:
                self._hyst_cnt += 1
                if self._hyst_cnt >= self.hyst_steps:
                    self.mode = "NORMAL"
                    self._hyst_cnt = 0
                    logger.info("→ NORMAL  (hysteresis cleared)")
            else:
                self._hyst_cnt = 0

    def _update_baselines(
        self, msg_a: BuildingAStatus, msg_b: BuildingBStatus
    ) -> None:
        """Exponential moving average of actual power consumption."""
        α = 0.3  # EMA smoothing factor
        self._P_A_base = (1 - α) * self._P_A_base + α * msg_a.P_A_current_kW
        self._P_B_base = (1 - α) * self._P_B_base + α * msg_b.P_B_current_kW

    # ═══════════════════════════════════════════════════════════════════
    #  NORMAL: rule-based allocation
    # ═══════════════════════════════════════════════════════════════════

    def _rule_based(
        self, sim_time: float, msg_a: BuildingAStatus, msg_b: BuildingBStatus,
    ) -> AllocationResult:
        """
        Simple rule-based power allocation during normal operation.

        A gets baseline + γ buffer;  B gets remainder up to feeder limit.
        TES mode chosen by SOC deviation from time-of-day target.
        """
        PH = self.PH
        hour = seconds_to_hour(sim_time)

        # A gets baseline + buffer
        P_A = self._P_A_base * (1.0 + self.gamma_buf)
        # B gets remainder (with safety margin)
        P_B = np.maximum(0.0, self.P_feeder_max - self.P_margin - P_A)

        # TES mode based on SOC tracking
        soc_tgt = soc_target(hour, self.cfg_b)
        gap = soc_tgt - msg_b.SOC
        if gap > 0.10 and (hour < 8 or hour >= 22):
            m_B = np.full(PH, -1, dtype=int)   # charge
        elif gap < -0.10:
            m_B = np.full(PH, 1, dtype=int)    # discharge
        else:
            m_B = np.full(PH, 2, dtype=int)    # chiller-only

        return AllocationResult(
            P_A_alloc=P_A.copy(),
            P_B_alloc=P_B.copy(),
            m_B_suggested=m_B,
            support_flag=False,
            delta_P_support=np.zeros(PH),
        )

    # ═══════════════════════════════════════════════════════════════════
    #  ATTACK: log-utility convex optimisation
    # ═══════════════════════════════════════════════════════════════════

    def _log_utility(
        self, sim_time: float, msg_a: BuildingAStatus, msg_b: BuildingBStatus,
    ) -> AllocationResult:
        """
        Proportionally fair power allocation during attack.

        max  w_a · ln(P_A) + w_b · ln(P_B)
        s.t. P_A + P_B  ≤  budget
             P_A_min ≤ P_A ≤ P_A_max   (from flexibility bands)
             P_B_min ≤ P_B ≤ P_B_max

        Closed-form solution:  P_i* = w_i / Σw · budget, then clip.
        """
        PH = self.PH
        hour = seconds_to_hour(sim_time)
        beta = self.beta.get(self.theta, 1.0)
        budget = self.P_feeder_max - self.P_margin

        # Flexibility bands from building messages (pad to PH)
        P_A_lo = pad_to(msg_a.P_A_flex_min, PH, 1.0)
        P_A_hi = pad_to(msg_a.P_A_flex_max, PH, budget)
        P_B_lo = pad_to(msg_b.P_flex_down, PH, 1.0)
        P_B_hi = pad_to(msg_b.P_flex_up, PH, budget)

        # Widen A's upper bound during attack (β > 1)
        P_A_hi = np.minimum(budget, P_A_hi * beta)

        # Weights: favour A more at higher θ
        w_a = 1.0 + 0.5 * self.theta
        soc_tgt = soc_target(hour, self.cfg_b)
        w_b = 1.0 + 2.0 * max(0.0, soc_tgt - msg_b.SOC)
        total_w = w_a + w_b

        P_A_alloc = np.zeros(PH)
        P_B_alloc = np.zeros(PH)
        m_B = np.zeros(PH, dtype=int)

        for k in range(PH):
            # Enforce sane bounds
            pa_lo = float(np.clip(P_A_lo[k], 0.01, budget))
            pa_hi = float(np.clip(P_A_hi[k], pa_lo, budget))
            pb_lo = float(np.clip(P_B_lo[k], 0.01, budget - pa_lo))
            pb_hi = float(np.clip(P_B_hi[k], pb_lo, budget))

            # Analytical log-utility solution
            pa = np.clip(w_a / total_w * budget, pa_lo, pa_hi)
            pb = np.clip(budget - pa, pb_lo, pb_hi)
            # Ensure feeder hard constraint after double clip
            pa = min(pa, budget - pb)

            P_A_alloc[k] = pa
            P_B_alloc[k] = pb

            # TES mode: discharge to support A during severe attack
            if self.theta >= 2 and msg_b.SOC > self.cfg_b["soc_min"] + 0.05:
                m_B[k] = 1   # discharge
            elif msg_b.SOC < self.cfg_b["soc_min"] + 0.10:
                m_B[k] = -1  # charge to prevent depletion
            else:
                m_B[k] = 2   # chiller-only

        delta_P = np.maximum(0.0, self._P_B_base - P_B_alloc)

        return AllocationResult(
            P_A_alloc=P_A_alloc,
            P_B_alloc=P_B_alloc,
            m_B_suggested=m_B,
            support_flag=True,
            delta_P_support=delta_P,
        )
