"""
Hierarchical MPC Simulation Orchestrator.

Implements the **Measure → Allocate → Optimize → Actuate** coordination
loop for a two-building community with a shared electrical feeder.

Multi-rate scheduling:
    Building A : every 15 min  (dt_building_a)
    Building B : every  1 hr   (dt_building_b)
    Aggregator : every 15 min  (dt_aggregator)

Usage (mock mode):
    python run_hmpc.py
    python run_hmpc.py --config config/system_config.yaml --duration 2

Author  : Guowen Li
Date    : 2026-02
"""

from __future__ import annotations

import argparse
import os
import sys
import time as wall_time
from pathlib import Path

import numpy as np

# ── Ensure project root is on sys.path ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aggregator.aggregator_mpc import AggregatorMPC
from aggregator.attack_manager import AttackManager
from buildings.building_a_wrapper import BuildingAWrapper
from buildings.building_b_wrapper import BuildingBWrapper
from communication.messages import CommandToA, CommandToB
from utils.helpers import (
    comfort_bounds,
    get_logger,
    load_config,
    seconds_to_hour,
    unmet_degree_hours,
)

logger = get_logger("orchestrator")


# ═══════════════════════════════════════════════════════════════════════════════
#  Data recorder
# ═══════════════════════════════════════════════════════════════════════════════

class MetricsRecorder:
    """Collects simulation data at every aggregator timestep."""

    def __init__(self):
        self.records = []

    def log(self, sim_time, agg_result, status_a, status_b):
        self.records.append({
            "time_s": sim_time,
            "hour": seconds_to_hour(sim_time),
            # Aggregator
            "agg_mode": agg_result.mode,
            "theta": agg_result.theta_priority,
            "P_A_alloc": float(agg_result.P_A_alloc[0]),
            "P_B_alloc": float(agg_result.P_B_alloc[0]),
            "P_total_alloc": float(agg_result.P_A_alloc[0] + agg_result.P_B_alloc[0]),
            "support_flag": agg_result.support_flag,
            # Building A
            "P_A_kW": status_a.P_A_current_kW,
            "alpha_attack": status_a.alpha_attack,
            "Tz_A_core": float(status_a.Tz[0]),
            "Tz_A_east": float(status_a.Tz[1]),
            "Tz_A_north": float(status_a.Tz[2]),
            "Tz_A_south": float(status_a.Tz[3]),
            "Tz_A_west": float(status_a.Tz[4]),
            "comfort_viol_A": float(np.sum(status_a.comfort_violations)),
            "mode_A": status_a.mode,
            # Building B
            "P_B_kW": status_b.P_B_current_kW,
            "SOC_B": status_b.SOC,
            "uMod_B": status_b.mode_current,
            "Tz_B_core": float(status_b.Tz[0]) if status_b.Tz is not None else np.nan,
            "comfort_viol_B": float(np.sum(status_b.comfort_violations))
                if status_b.comfort_violations is not None else 0.0,
            # Community
            "P_total_kW": status_a.P_A_current_kW + status_b.P_B_current_kW,
        })

    def to_csv(self, path: str):
        """Export records to CSV."""
        import csv
        if not self.records:
            return
        keys = self.records[0].keys()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.records)
        logger.info("Metrics saved → %s  (%d rows)", path, len(self.records))


# ═══════════════════════════════════════════════════════════════════════════════
#  Main simulation loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_simulation(cfg: dict) -> MetricsRecorder:
    """
    Execute the hierarchical MPC simulation.

    Parameters
    ----------
    cfg : dict   Loaded system configuration.

    Returns
    -------
    MetricsRecorder with all time-series data.
    """
    t_wall_start = wall_time.perf_counter()

    # ── timing ──────────────────────────────────────────────────────────
    start_day = cfg["timing"]["start_day"]
    duration_s = cfg["timing"]["duration_days"] * 86400
    dt_agg = cfg["timing"]["dt_aggregator"]
    dt_a = cfg["timing"]["dt_building_a"]
    dt_b = cfg["timing"]["dt_building_b"]

    t_start = 0.0
    t_end = duration_s
    n_steps = int(duration_s / dt_agg)

    logger.info("=" * 72)
    logger.info("  HIERARCHICAL MPC SIMULATION")
    logger.info("  Start day: %d   Duration: %d day(s)   Steps: %d",
                start_day, cfg["timing"]["duration_days"], n_steps)
    logger.info("  dt_agg=%.0fs  dt_A=%.0fs  dt_B=%.0fs",
                dt_agg, dt_a, dt_b)
    logger.info("=" * 72)

    # ── create components ───────────────────────────────────────────────
    aggregator = AggregatorMPC(cfg)
    building_a = BuildingAWrapper(cfg, mpc=None, fmu=None)   # mock mode
    building_b = BuildingBWrapper(cfg, mpc=None, fmu=None)   # mock mode
    attack_mgr = AttackManager(cfg.get("attacks", []), start_day)
    recorder = MetricsRecorder()

    # ── simulation loop ─────────────────────────────────────────────────
    feeder_max = cfg["feeder"]["capacity_kW"] * cfg["feeder"]["safety_margin"]
    n_feeder_violations = 0
    b_last_executed = -dt_b  # ensure Building B executes at t=0

    for step in range(n_steps):
        t = t_start + step * dt_agg

        # ① ATTACK MANAGER — inject / clear attacks
        attack_mgr.update(t, building_a, building_b)

        # Check recovery transition
        if building_a.mode == "RECOVERY":
            building_a.tick_recovery()

        # ② MEASURE — buildings report current state
        status_a = building_a.report_status(t)
        status_b = building_b.report_status(t)

        # ③ ALLOCATE — aggregator computes power budgets
        alloc = aggregator.coordinate(t, status_a, status_b)

        # ④ BUILD COMMANDS
        cmd_a = CommandToA(
            P_A_alloc=alloc.P_A_alloc,
            theta_priority=alloc.theta_priority,
        )
        cmd_b = CommandToB(
            P_B_alloc=alloc.P_B_alloc,
            m_B_suggested=alloc.m_B_suggested,
            delta_P_support=alloc.delta_P_support,
            support_flag=alloc.support_flag,
        )

        # ⑤ EXECUTE — buildings solve MPC and advance plant
        # Building A: every dt_agg step (aligned with aggregator)
        status_a = building_a.execute(t, cmd_a)

        # Building B: only every dt_b seconds (multi-rate)
        if (t - b_last_executed) >= dt_b - 1e-6:
            status_b = building_b.execute(t, cmd_b)
            b_last_executed = t

        # ⑥ RECORD
        recorder.log(t, alloc, status_a, status_b)

        # ⑦ CHECK feeder constraint
        P_total = status_a.P_A_current_kW + status_b.P_B_current_kW
        if P_total > feeder_max:
            n_feeder_violations += 1
            logger.warning(
                "FEEDER VIOLATION at t=%.0fs: %.1f kW > %.1f kW",
                t, P_total, feeder_max,
            )

    # ── summary ─────────────────────────────────────────────────────────
    elapsed = wall_time.perf_counter() - t_wall_start
    logger.info("=" * 72)
    logger.info("  SIMULATION COMPLETE")
    logger.info("  Wall time:  %.2f s", elapsed)
    logger.info("  Steps:      %d", n_steps)
    logger.info("  Feeder violations: %d / %d (%.1f%%)",
                n_feeder_violations, n_steps,
                100.0 * n_feeder_violations / max(1, n_steps))

    # Compute UDH for both buildings
    if recorder.records:
        Tz_A = np.array([[r[f"Tz_A_{z}"] for z in ["core","east","north","south","west"]]
                          for r in recorder.records])
        Tz_B = np.array([[r.get("Tz_B_core", 22.5)] for r in recorder.records])
        lo, hi = cfg["building_a"]["Tz_min_occ"], cfg["building_a"]["Tz_max_occ"]
        udh_a = unmet_degree_hours(Tz_A, lo, hi, dt_agg)
        udh_b = unmet_degree_hours(Tz_B, lo, hi, dt_agg)
        logger.info("  UDH(A): %.2f °C·h   UDH(B): %.2f °C·h", udh_a, udh_b)

    logger.info("=" * 72)
    return recorder


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical MPC simulation for two-building community",
    )
    parser.add_argument(
        "--config", "-c",
        default="config/system_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--duration", "-d", type=int, default=None,
        help="Override simulation duration (days)",
    )
    parser.add_argument(
        "--start-day", type=int, default=None,
        help="Override start day",
    )
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help="Override results output directory",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Apply overrides
    if args.duration is not None:
        cfg["timing"]["duration_days"] = args.duration
    if args.start_day is not None:
        cfg["timing"]["start_day"] = args.start_day
    if args.output_dir is not None:
        cfg["output"]["results_dir"] = args.output_dir

    # Ensure results directory exists
    results_dir = cfg["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    # Run simulation
    recorder = run_simulation(cfg)

    # Save metrics
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_dir, f"metrics_{ts}.csv")
    recorder.to_csv(csv_path)

    return recorder


if __name__ == "__main__":
    main()
