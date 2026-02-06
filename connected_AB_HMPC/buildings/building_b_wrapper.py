"""
Building B hierarchical wrapper.

Wraps the standalone ``mpc_case`` (DEAP / GA, 1 discrete decision variable)
and adds:

* Flexibility-band computation for the aggregator
* TES support coordination with aggregator support requests
* Mode-dependent power mapping for mock plant
* Mock plant dynamics for framework testing

The building B MPC optimises a single discrete variable:
    uMod ∈ {-1: charge TES, 0: off, 1: discharge TES, 2: chiller-only}

Author  : Guowen Li
Date    : 2026-02
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from communication.messages import BuildingBStatus, CommandToB
from utils.helpers import (
    comfort_bounds,
    comfort_violation,
    get_logger,
    is_occupied,
    seconds_to_hour,
    soc_target,
)

logger = get_logger("building_b")


class BuildingBWrapper:
    """
    Hierarchical wrapper around Building B's standalone MPC.

    Parameters
    ----------
    cfg : dict              Full system configuration.
    mpc : object | None     ``mpc_case`` from ``mpc_b.py``.  None = mock mode.
    fmu : object | None     FMU handle.  None = mock mode.
    """

    def __init__(self, cfg: dict, mpc: Any = None, fmu: Any = None):
        cb = cfg["building_b"]
        self.modes = cb["modes"]
        self.mode_names = cb["mode_names"]

        self.dt = cfg["timing"]["dt_building_b"]
        self.PH = cfg["timing"]["prediction_horizon_b"]
        self.agg_PH = cfg["timing"]["prediction_horizon_agg"]
        self.dt_agg = cfg["timing"]["dt_aggregator"]

        # TES parameters
        self.tes_cap = cb["tes_capacity_kWh"]
        self.soc_min = cb["soc_min"]
        self.soc_max = cb["soc_max"]
        self.charge_kW = cb["charge_rate_kW"]
        self.discharge_kW = cb["discharge_rate_kW"]
        self.chiller_kW = cb["chiller_power_kW"]
        self.standby_kW = cb["standby_power_kW"]
        self.eta_rt = cb["roundtrip_efficiency"]
        self.n_zones = cb.get("n_zones", 5)

        # Weights
        self.w_cost = cb["w_cost"]
        self.w_support = cb["w_support"]
        self.w_soc = cb["w_soc"]

        self.cfg_b = cb
        self.cfg = cfg
        self.mpc = mpc
        self.fmu = fmu

        # ── runtime state ───────────────────────────────────────────────
        self.SOC: float = cb["soc_initial"]
        self.P_kW: float = 0.0
        self.uMod: int = 0                      # current TES mode
        self.Tz = np.full(self.n_zones, 22.5)
        self._support_flag: bool = False
        self._delta_P: np.ndarray = np.zeros(self.agg_PH)

        logger.info(
            "BuildingB ready  PH=%d  dt=%.0fs  SOC_init=%.2f  mock=%s",
            self.PH, self.dt, self.SOC,
            "yes" if mpc is None else "no",
        )

    # ═══════════════════════════════════════════════════════════════════
    #  report_status()  — read-only, NO plant advancement
    # ═══════════════════════════════════════════════════════════════════

    def report_status(self, sim_time: float) -> BuildingBStatus:
        """Return current state + flexibility bands without advancing plant."""
        flex_down, flex_up = self._flexibility_bands(sim_time)

        hour = seconds_to_hour(sim_time)
        Tz_lo, Tz_hi = comfort_bounds(hour, self.cfg_b)
        viol = comfort_violation(self.Tz, Tz_lo, Tz_hi)

        return BuildingBStatus(
            timestamp=sim_time,
            P_B_current_kW=self.P_kW,
            SOC=self.SOC,
            mode_current=self.uMod,
            P_flex_up=flex_up,
            P_flex_down=flex_down,
            Tz=self.Tz.copy(),
            comfort_violations=viol,
        )

    # ═══════════════════════════════════════════════════════════════════
    #  execute(command)  — solve + advance plant, return updated status
    # ═══════════════════════════════════════════════════════════════════

    def execute(
        self,
        sim_time: float,
        command: Optional[CommandToB] = None,
    ) -> BuildingBStatus:
        """
        Receive aggregator command → adjust targets → solve MPC →
        apply TES mode → advance plant → return new status.
        """
        # 1. Store aggregator support request
        if command is not None:
            self._support_flag = command.support_flag
            self._delta_P = command.delta_P_support.copy()

        # 2. Solve MPC (or mock) for optimal TES mode sequence
        m_suggested = command.m_B_suggested if command is not None else None
        u_opt = self._solve(sim_time, m_suggested)

        # 3. Apply first-step mode and advance plant
        self._apply_and_advance(u_opt, sim_time)

        # 4. Return updated status
        return self.report_status(sim_time + self.dt)

    # ═══════════════════════════════════════════════════════════════════
    #  Flexibility bands
    # ═══════════════════════════════════════════════════════════════════

    def _flexibility_bands(self, sim_time: float):
        """
        Compute (P_min, P_max) over agg_PH steps.

        P_min corresponds to the lowest-power mode feasible at this SOC,
        P_max corresponds to the highest-power mode feasible.
        """
        PH = self.agg_PH

        if self.mpc is not None and hasattr(self.mpc, "get_open_loop_preds"):
            try:
                # Evaluate all four modes and take min/max power
                P_modes = []
                for mode in self.modes:
                    u_seq = np.full(self.PH, mode)
                    preds = self.mpc.get_open_loop_preds(u_seq)
                    P_modes.append(preds["P_pred"][:PH])
                P_all = np.array(P_modes)  # (4, PH)
                return np.min(P_all, axis=0), np.max(P_all, axis=0)
            except Exception as exc:
                logger.debug("get_open_loop_preds: %s", exc)

        # Heuristic fallback based on TES parameters
        P_lo = np.full(PH, self.discharge_kW)    # min = discharge pump only
        P_hi = np.full(PH, self.charge_kW)        # max = charge (chiller on)
        # If SOC is too low, can't discharge → min is standby
        if self.SOC < self.soc_min + 0.10:
            P_lo = np.full(PH, self.standby_kW)
        # If SOC is too high, can't charge → max is chiller-only
        if self.SOC > self.soc_max - 0.10:
            P_hi = np.full(PH, self.chiller_kW)
        return P_lo, P_hi

    # ═══════════════════════════════════════════════════════════════════
    #  Solve MPC
    # ═══════════════════════════════════════════════════════════════════

    def _solve(
        self, sim_time: float, m_suggested: Optional[np.ndarray]
    ) -> int:
        """Solve local MPC or return heuristic mode."""
        if self.mpc is None:
            return self._mock_mode_select(sim_time, m_suggested)

        try:
            # If aggregator provided a suggested mode, use as fixed_vars
            fixed = {}
            if m_suggested is not None:
                fixed["uMod"] = int(m_suggested[0])

            res, status = self.mpc.optimize(fixed_vars=fixed or None)
            if status.get("return_status") in ("OPTIMAL", "optimal", None):
                u_opt = res["x"]
                if hasattr(u_opt, "full"):
                    u_opt = u_opt.full().flatten()
                return int(u_opt[0])
            else:
                logger.warning("MPC-B solver: %s", status)
        except Exception as exc:
            logger.warning("MPC-B failed: %s", exc)

        # Fallback to suggested mode
        if m_suggested is not None:
            return int(m_suggested[0])
        return self.uMod

    def _mock_mode_select(
        self, sim_time: float, m_suggested: Optional[np.ndarray]
    ) -> int:
        """Heuristic TES mode for mock testing."""
        hour = seconds_to_hour(sim_time)
        occ = is_occupied(
            hour,
            self.cfg_b.get("occ_start_hour", 7),
            self.cfg_b.get("occ_end_hour", 19),
        )

        # If aggregator suggested a mode, prefer it
        if m_suggested is not None:
            return int(m_suggested[0])

        # SOC-based heuristic
        soc_tgt = soc_target(hour, self.cfg_b)
        gap = soc_tgt - self.SOC

        if not occ:
            # Unoccupied: charge if below target, off otherwise
            if gap > 0.10:
                return -1   # charge
            return 0        # off
        else:
            # Occupied: discharge if above target, chiller-only otherwise
            if self.SOC > self.soc_min + 0.15 and gap < -0.05:
                return 1    # discharge
            return 2        # chiller-only

    # ═══════════════════════════════════════════════════════════════════
    #  Apply mode + advance plant
    # ═══════════════════════════════════════════════════════════════════

    def _apply_and_advance(self, mode: int, sim_time: float) -> None:
        """Apply TES mode to FMU or mock, advance one dt, update state."""
        self.uMod = mode

        if self.fmu is not None:
            try:
                self.fmu.set("uMod", mode)
                self.fmu.advance(self.dt)
                self.P_kW = self.fmu.get_total_power_kW()
                self.SOC = self.fmu.get_SOC()
                self.Tz = np.array(self.fmu.get_zone_temperatures())
                return
            except Exception as exc:
                logger.warning("FMU-B advance: %s", exc)

        # ── mock plant dynamics ──────────────────────────────────────────
        hour = seconds_to_hour(sim_time)
        dt_h = self.dt / 3600.0  # timestep in hours

        if mode == -1:  # charge
            self.P_kW = self.charge_kW
            delta_soc = (self.charge_kW * self.eta_rt * dt_h) / self.tes_cap
            self.SOC = min(self.soc_max, self.SOC + delta_soc)
        elif mode == 0:  # off
            self.P_kW = self.standby_kW
            # Small SOC decay (thermal losses)
            self.SOC = max(self.soc_min, self.SOC - 0.002 * dt_h)
        elif mode == 1:  # discharge
            self.P_kW = self.discharge_kW
            delta_soc = (self.chiller_kW * dt_h) / (self.tes_cap * self.eta_rt)
            self.SOC = max(self.soc_min, self.SOC - delta_soc)
        else:  # mode == 2, chiller-only
            self.P_kW = self.chiller_kW

        # Temperature dynamics (simplified)
        occ = is_occupied(
            hour,
            self.cfg_b.get("occ_start_hour", 7),
            self.cfg_b.get("occ_end_hour", 19),
        )
        T_sp = 22.5
        T_oa = 30.0 + 5.0 * np.sin(2 * np.pi * (hour - 14) / 24)
        for i in range(self.n_zones):
            if occ and mode in (1, 2):
                # Active cooling
                self.Tz[i] += 0.2 * (T_sp - self.Tz[i])
            else:
                # Free-float
                self.Tz[i] += 0.05 * (T_oa - self.Tz[i])
            self.Tz[i] += np.random.normal(0, 0.03)

    # ═══════════════════════════════════════════════════════════════════
    #  Attack injection interface (same pattern as Building A)
    # ═══════════════════════════════════════════════════════════════════

    def inject_attack(self, attack_type: str = "dos_vav_reinit",
                      zone: str = "core") -> None:
        """Building B attack injection (not primary target, but interface exists)."""
        logger.warning("ATTACK on Building B: %s (zone=%s)", attack_type, zone)

    def clear_attack(self) -> None:
        """Clear attack on Building B."""
        logger.info("Building B attack cleared")
