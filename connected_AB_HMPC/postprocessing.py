"""
Post-processing and visualization for hierarchical MPC simulation results.

Reads the CSV metrics file produced by run_hmpc.py and generates
publication-quality figures for:

  1. Community power consumption + feeder limit
  2. Building A zone temperatures
  3. Building B SOC trajectory
  4. Aggregator mode and priority timeline
  5. Comfort violations
  6. Summary metrics table

Usage:
    python postprocessing.py results/metrics_YYYYMMDD_HHMMSS.csv
    python postprocessing.py                 # auto-finds latest CSV

Author  : Guowen Li
Date    : 2026-02
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

# ── Ensure project root is on sys.path ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import get_logger, load_config

logger = get_logger("postprocess")


def load_metrics(csv_path: str) -> dict:
    """Load CSV into a dict of numpy arrays (one key per column)."""
    import csv
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")

    data = {}
    for key in rows[0]:
        vals = []
        for r in rows:
            v = r[key]
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                vals.append(v)
        if all(isinstance(x, float) for x in vals):
            data[key] = np.array(vals)
        else:
            data[key] = vals
    return data


def plot_results(data: dict, cfg: dict, save_dir: str = "results"):
    """Generate all result figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.error("matplotlib not available — skipping plots")
        return

    os.makedirs(save_dir, exist_ok=True)
    hours = data["hour"]
    feeder_max = cfg["feeder"]["capacity_kW"] * cfg["feeder"]["safety_margin"]

    # Attack shading helper
    attacks = cfg.get("attacks", [])
    day_offset = cfg["timing"]["start_day"] * 86400

    def shade_attacks(ax):
        for atk in attacks:
            t0 = (atk["start_time_s"] - day_offset) / 3600.0
            t1 = t0 + atk["duration_s"] / 3600.0
            ax.axvspan(t0, t1, alpha=0.15, color="red", label="Attack")

    # ── Figure 1: Community Power ────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    ax = axes[0]
    ax.plot(hours, data["P_A_kW"], "b-", lw=1.5, label="Building A")
    ax.plot(hours, data["P_B_kW"], "g-", lw=1.5, label="Building B")
    ax.plot(hours, data["P_total_kW"], "k--", lw=2, label="Total")
    ax.axhline(feeder_max, color="r", ls="--", lw=1.5, label=f"Feeder limit ({feeder_max:.0f} kW)")
    shade_attacks(ax)
    ax.set_ylabel("Power [kW]")
    ax.set_title("Community Power Consumption")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(hours, data["P_A_alloc"], "b:", lw=1.5, label="P_A allocated")
    ax.plot(hours, data["P_B_alloc"], "g:", lw=1.5, label="P_B allocated")
    ax.plot(hours, data["P_total_alloc"], "k:", lw=1.5, label="Total allocated")
    ax.axhline(feeder_max, color="r", ls="--", lw=1.5)
    shade_attacks(ax)
    ax.set_ylabel("Power [kW]")
    ax.set_title("Aggregator Power Allocation")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(hours, data["theta"], "m-", lw=2, label="θ (priority)")
    ax.plot(hours, data["alpha_attack"], "r-", lw=1.5, label="α (attack)")
    shade_attacks(ax)
    ax.set_ylabel("Level")
    ax.set_xlabel("Hour of day")
    ax.set_title("Attack Detection & Priority")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 3.5)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig1_power_allocation.png"), dpi=200)
    plt.close(fig)
    logger.info("Saved fig1_power_allocation.png")

    # ── Figure 2: Zone Temperatures ──────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    zone_colors = {"core": "r", "east": "b", "north": "g", "south": "orange", "west": "purple"}
    ax = axes[0]
    for z, c in zone_colors.items():
        key = f"Tz_A_{z}"
        if key in data:
            ax.plot(hours, data[key], color=c, lw=1.2, label=z.capitalize())
    # Comfort bounds
    lo_occ, hi_occ = cfg["building_a"]["Tz_min_occ"], cfg["building_a"]["Tz_max_occ"]
    ax.axhline(lo_occ, color="gray", ls="--", lw=1, alpha=0.7)
    ax.axhline(hi_occ, color="gray", ls="--", lw=1, alpha=0.7, label="Comfort bounds")
    shade_attacks(ax)
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("Building A — Zone Temperatures")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if "Tz_B_core" in data:
        ax.plot(hours, data["Tz_B_core"], "r-", lw=1.5, label="Core zone")
    lo_occ_b, hi_occ_b = cfg["building_b"]["Tz_min_occ"], cfg["building_b"]["Tz_max_occ"]
    ax.axhline(lo_occ_b, color="gray", ls="--", lw=1, alpha=0.7)
    ax.axhline(hi_occ_b, color="gray", ls="--", lw=1, alpha=0.7, label="Comfort bounds")
    shade_attacks(ax)
    ax.set_ylabel("Temperature [°C]")
    ax.set_xlabel("Hour of day")
    ax.set_title("Building B — Zone Temperatures")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig2_temperatures.png"), dpi=200)
    plt.close(fig)
    logger.info("Saved fig2_temperatures.png")

    # ── Figure 3: Building B SOC + TES Mode ──────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ax = axes[0]
    ax.plot(hours, data["SOC_B"], "b-", lw=2)
    ax.axhline(cfg["building_b"]["soc_min"], color="r", ls="--", lw=1, label="SOC min")
    ax.axhline(cfg["building_b"]["soc_max"], color="r", ls="--", lw=1, label="SOC max")
    shade_attacks(ax)
    ax.set_ylabel("SOC [-]")
    ax.set_title("Building B — TES State of Charge")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    mode_data = data["uMod_B"]
    ax.step(hours, mode_data, "k-", lw=1.5, where="post")
    ax.set_yticks([-1, 0, 1, 2])
    ax.set_yticklabels(["Charge", "Off", "Discharge", "Chiller"])
    shade_attacks(ax)
    ax.set_ylabel("TES Mode")
    ax.set_xlabel("Hour of day")
    ax.set_title("Building B — TES Operating Mode")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig3_tes_soc.png"), dpi=200)
    plt.close(fig)
    logger.info("Saved fig3_tes_soc.png")

    # ── Figure 4: Comfort Violations ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(hours, data["comfort_viol_A"], width=0.2, color="blue", alpha=0.6, label="Building A")
    ax.bar(hours + 0.2, data["comfort_viol_B"], width=0.2, color="green", alpha=0.6, label="Building B")
    shade_attacks(ax)
    ax.set_ylabel("Comfort violation [K]")
    ax.set_xlabel("Hour of day")
    ax.set_title("Community Comfort Violations")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig4_comfort_violations.png"), dpi=200)
    plt.close(fig)
    logger.info("Saved fig4_comfort_violations.png")


def print_summary(data: dict, cfg: dict):
    """Print summary metrics to console."""
    feeder_max = cfg["feeder"]["capacity_kW"] * cfg["feeder"]["safety_margin"]
    P_total = data["P_total_kW"]

    n_violations = int(np.sum(P_total > feeder_max))
    n_steps = len(P_total)
    peak_power = float(np.max(P_total))
    avg_power = float(np.mean(P_total))
    max_comfort_a = float(np.max(data["comfort_viol_A"]))
    avg_soc = float(np.mean(data["SOC_B"]))
    total_cost_a = float(np.sum(data["P_A_kW"])) * cfg["timing"]["dt_aggregator"] / 3600 / 1000
    total_cost_b = float(np.sum(data["P_B_kW"])) * cfg["timing"]["dt_aggregator"] / 3600 / 1000

    print("\n" + "=" * 60)
    print("  SIMULATION SUMMARY")
    print("=" * 60)
    print(f"  Total steps           : {n_steps}")
    print(f"  Feeder limit          : {feeder_max:.1f} kW")
    print(f"  Peak community power  : {peak_power:.1f} kW")
    print(f"  Avg community power   : {avg_power:.1f} kW")
    print(f"  Feeder violations     : {n_violations} / {n_steps} "
          f"({100*n_violations/max(1,n_steps):.1f}%)")
    print(f"  Max comfort viol (A)  : {max_comfort_a:.2f} K")
    print(f"  Avg SOC (B)           : {avg_soc:.3f}")
    print(f"  Energy A              : {total_cost_a:.2f} MWh")
    print(f"  Energy B              : {total_cost_b:.2f} MWh")
    print("=" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Post-process HMPC results")
    parser.add_argument("csv", nargs="?", default=None, help="Path to metrics CSV")
    parser.add_argument("--config", default="config/system_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Find CSV
    csv_path = args.csv
    if csv_path is None:
        results_dir = cfg["output"]["results_dir"]
        csvs = sorted(glob.glob(os.path.join(results_dir, "metrics_*.csv")))
        if not csvs:
            logger.error("No metrics CSV found in %s", results_dir)
            sys.exit(1)
        csv_path = csvs[-1]
        logger.info("Auto-selected: %s", csv_path)

    data = load_metrics(csv_path)
    print_summary(data, cfg)
    plot_results(data, cfg, save_dir=cfg["output"]["results_dir"])


if __name__ == "__main__":
    main()
