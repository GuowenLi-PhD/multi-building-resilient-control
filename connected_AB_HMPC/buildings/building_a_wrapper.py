"""
Building A hierarchical wrapper.

Wraps the standalone ``mpc_case`` (CasADi / IPOPT, 11 decision variables)
and adds:

* Flexibility-band computation for the aggregator (Dissertation Eqs. 27-28)
* Adaptive weight switching based on aggregator priority θ
* Power budget constraint injection
* Attack injection / clearance interface
* Mock plant dynamics for framework testing

CRITICAL DESIGN: Two separate public methods prevent double-stepping.
    report_status()  — read-only snapshot, NO plant advancement
    execute(command)  — solve MPC + advance plant, returns updated status

Author  : Guowen Li
Date    : 2026-02
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from communication.messages import BuildingAStatus, CommandToA
from utils.helpers import (
    comfort_bounds,
    comfort_violation,
    get_logger,
    is_occupied,
    seconds_to_hour,
)

logger = get_logger("building_a")


class BuildingAWrapper:
    """
    Hierarchical wrapper around Building A's standalone MPC.

    Parameters
    ----------
    cfg : dict              Full system configuration.
    mpc : object | None     ``mpc_case`` from ``mpc_a.py``.  None = mock mode.
    fmu : object | None     FMU handle.  None = mock mode.
    """

    def __init__(self, cfg: dict, mpc: Any = None, fmu: Any = None):
        ca = cfg["building_a"]
        self.n_zones = ca["n_zones"]
        self.n_inputs = ca["n_inputs"]
        self.zone_names = ca["zone_names"]
        self.input_names = ca["input_names"]
        self.u_lb = np.array(ca["u_lower"], dtype=float)
        self.u_ub = np.array(ca["u_upper"], dtype=float)

        self.dt = cfg["timing"]["dt_building_a"]
        self.PH = cfg["timing"]["prediction_horizon_a"]
        self.agg_PH = cfg["timing"]["prediction_horizon_agg"]

        # Weight tables (θ → weight)
        self.w_comfort_table = {
            int(k): v for k, v in ca["w_comfort_by_priority"].items()
        }
        self.w_cost_table = {
            int(k): v for k, v in ca["w_cost_by_priority"].items()
        }
        self.w_nominal = list(ca["w_nominal"])

        # Mock plant coefficients
        self.P_base = ca.get("P_base_kW", 5.0)
        self.P_fan_c = ca.get("P_fan_coeff", 2.0)
        self.P_cool_c = ca.get("P_cool_coeff", 0.8)
        self.P_standby = ca.get("P_standby_kW", 0.5)

        self.cfg_a = ca
        self.mpc = mpc
        self.fmu = fmu

        # ── runtime state ───────────────────────────────────────────────
        self.mode: str = "NOMINAL"
        self.theta: int = 0
        self._attack_alpha: float = 0.0

        self.Tz = np.full(self.n_zones, 22.5)
        self.P_kW: float = 0.0
        self.u_current = np.array(ca["u_init"], dtype=float)

        logger.info(
            "BuildingA ready  n_inputs=%d  PH=%d  dt=%.0fs  mock=%s",
            self.n_inputs, self.PH, self.dt,
            "yes" if mpc is None else "no",
        )

    # ═══════════════════════════════════════════════════════════════════
    #  report_status()  — read-only, NO plant advancement
    # ═══════════════════════════════════════════════════════════════════

    def report_status(self, sim_time: float) -> BuildingAStatus:
        """Return current state + flexibility bands without advancing plant."""
        hour = seconds_to_hour(sim_time)
        Tz_lo, Tz_hi = comfort_bounds(hour, self.cfg_a)
        viol = comfort_violation(self.Tz, Tz_lo, Tz_hi)

        flex_min, flex_max = self._flexibility_bands(sim_time)

        return BuildingAStatus(
            timestamp=sim_time,
            P_A_current_kW=self.P_kW,
            Tz=self.Tz.copy(),
            alpha_attack=self._attack_alpha,
            P_A_flex_min=flex_min,
            P_A_flex_max=flex_max,
            u_current=self.u_current.copy(),
            comfort_violations=viol,
            mode=self.mode,
        )

    # ═══════════════════════════════════════════════════════════════════
    #  execute(command)  — solve + advance plant, return updated status
    # ═══════════════════════════════════════════════════════════════════

    def execute(
        self,
        sim_time: float,
        command: Optional[CommandToA] = None,
    ) -> BuildingAStatus:
        """
        Receive aggregator command → adapt weights → solve MPC →
        apply first-step control → advance plant → return new status.
        """
        # 1. Adapt weights to priority
        if command is not None:
            self.theta = command.theta_priority
        self._adapt_weights()

        # 2. Solve MPC (or mock) with power budget
        P_budget = command.P_A_alloc if command is not None else None
        u_opt = self._solve(sim_time, P_budget)

        # 3. Apply to plant and advance one step
        self._apply_and_advance(u_opt, sim_time)

        # 4. Return updated status
        return self.report_status(sim_time + self.dt)

    # ═══════════════════════════════════════════════════════════════════
    #  Flexibility bands  (Pass 1 of two-pass scheme)
    # ═══════════════════════════════════════════════════════════════════

    def _flexibility_bands(self, sim_time: float):
        """
        Compute (P_min, P_max) over agg_PH steps.

        If a real MPC with open-loop prediction is available, use it.
        Otherwise use a heuristic ±30% band around current power.
        """
        PH = self.agg_PH

        if self.mpc is not None and hasattr(self.mpc, "get_open_loop_preds"):
            try:
                u_seq = np.tile(self.u_current, (self.PH, 1))
                preds = self.mpc.get_open_loop_preds(u_seq)
                P = np.array(preds["P_pred"][:PH])
                return P * 0.70, P * 1.30
            except Exception as exc:
                logger.debug("get_open_loop_preds: %s", exc)

        # Heuristic fallback
        P = max(self.P_kW, 1.0)
        return np.full(PH, P * 0.70), np.full(PH, P * 1.30)

    # ═══════════════════════════════════════════════════════════════════
    #  Solve MPC  (Pass 2 with budget constraint)
    # ═══════════════════════════════════════════════════════════════════

    def _solve(
        self, sim_time: float, P_budget: Optional[np.ndarray]
    ) -> np.ndarray:
        """Solve local MPC or return mock controls."""
        if self.mpc is None:
            return self._mock_control(sim_time)

        # Prepare fixed vars for the budget
        fixed = {}
        if self.mode == "COMFORT_PRIORITY" and self._attack_alpha > 0.5:
            fixed["Vcore"] = 0.0  # core zone VAV compromised by attack

        try:
            res, status = self.mpc.optimize(fixed_vars=fixed or None)
            if status.get("success", False):
                u = res["x"].full().flatten()
                return u[: self.n_inputs]
            else:
                logger.warning("MPC-A solver: %s", status)
        except Exception as exc:
            logger.warning("MPC-A failed: %s", exc)

        return self.u_current.copy()

    def _mock_control(self, sim_time: float) -> np.ndarray:
        """Mock controls for coordination-framework testing."""
        hour = seconds_to_hour(sim_time)
        occ = is_occupied(
            hour,
            self.cfg_a.get("occ_start_hour", 7),
            self.cfg_a.get("occ_end_hour", 19),
        )
        u = self.u_current.copy()
        if occ:
            u[0] = 1.0
            u[1] = 1.0
            u[2] = 7.0
            u[3] = 25.0
            u[4] = 13.0
            u[5:10] = [1.5, 0.3, 0.3, 0.3, 0.2]
            u[10] = 0.01
            # During attack: core-zone VAV is locked at 0
            if self.mode == "COMFORT_PRIORITY":
                u[5] = 0.0
                # Compensate by boosting adjacent zones
                u[6:10] = [0.6, 0.6, 0.6, 0.5]
        else:
            u[0] = 0.0
            u[1] = 0.0
            u[5:10] = [0.23, 0.05, 0.05, 0.05, 0.04]
            u[10] = 0.0
        return u

    # ═══════════════════════════════════════════════════════════════════
    #  Apply control + advance plant
    # ═══════════════════════════════════════════════════════════════════

    def _apply_and_advance(self, u: np.ndarray, sim_time: float) -> None:
        """Apply control to FMU or mock, advance one dt, update state."""
        self.u_current = u.copy()

        if self.fmu is not None:
            try:
                self.fmu.advance(u, self.dt)
                self.Tz = np.array(self.fmu.get_zone_temperatures())
                self.P_kW = self.fmu.get_total_power_kW()
                return
            except Exception as exc:
                logger.warning("FMU-A advance: %s", exc)

        # ── mock plant dynamics ──────────────────────────────────────────
        hour = seconds_to_hour(sim_time)
        occ = is_occupied(
            hour,
            self.cfg_a.get("occ_start_hour", 7),
            self.cfg_a.get("occ_end_hour", 19),
        )
        if occ and u[0] > 0.5:
            P_fan = np.sum(u[5:10]) * self.P_fan_c
            P_cool = max(0.0, 25.0 - u[4]) * self.P_cool_c
            self.P_kW = self.P_base + P_fan + P_cool

            # Temperature dynamics: drift toward comfort setpoint
            T_sp = 22.5
            for i in range(self.n_zones):
                airflow_i = u[5 + i] if (5 + i) < len(u) else 0.2
                cool_eff = min(1.0, airflow_i / 0.5)
                self.Tz[i] += 0.3 * (T_sp - self.Tz[i]) * cool_eff
                self.Tz[i] += np.random.normal(0, 0.05)

            # Attack effect: core zone overheats without airflow
            if self.mode == "COMFORT_PRIORITY" and u[5] < 0.05:
                self.Tz[0] += 0.5  # core zone drifts hot
        else:
            self.P_kW = self.P_standby
            # Free-float toward outdoor temp
            T_oa = 30.0 + 5.0 * np.sin(2 * np.pi * (hour - 14) / 24)
            for i in range(self.n_zones):
                self.Tz[i] += 0.05 * (T_oa - self.Tz[i])
                self.Tz[i] += np.random.normal(0, 0.03)

    # ═══════════════════════════════════════════════════════════════════
    #  Adaptive weights
    # ═══════════════════════════════════════════════════════════════════

    def _adapt_weights(self) -> None:
        """Update MPC objective weights based on aggregator priority θ."""
        if self.mpc is None:
            return
        w_cost = self.w_cost_table.get(self.theta, 10.0)
        w_comfort = self.w_comfort_table.get(self.theta, 1.0)
        w_slew = self.w_nominal[2] * (0.1 if self.theta >= 1 else 1.0)
        self.mpc.w = [w_cost, w_comfort, w_slew]
        logger.debug("Weights updated  θ=%d  w=[%.1f, %.1f, %.1f]",
                      self.theta, w_cost, w_comfort, w_slew)

    # ═══════════════════════════════════════════════════════════════════
    #  Attack injection (from AttackManager)
    # ═══════════════════════════════════════════════════════════════════

    def inject_attack(self, attack_type: str = "dos_vav_reinit",
                      zone: str = "core") -> None:
        """Mark this building as under attack."""
        self._attack_alpha = 0.95
        self.mode = "COMFORT_PRIORITY"
        logger.warning("ATTACK INJECTED: %s on zone=%s", attack_type, zone)

    def clear_attack(self) -> None:
        """Clear the attack and transition to RECOVERY."""
        self._attack_alpha = 0.0
        self.mode = "RECOVERY"
        logger.info("Attack cleared → RECOVERY")

    def tick_recovery(self) -> None:
        """Transition RECOVERY → NOMINAL (called each step post-attack)."""
        if self.mode == "RECOVERY":
            self.mode = "NOMINAL"
            logger.info("Recovery complete → NOMINAL")
