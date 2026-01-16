# Auto-generated post-processing script: Baseline vs Proposed hierarchical control
# Run: python hierarchical_postprocess.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

baseline_path = Path(r"/mnt/data/scenario1_metrics_20260102_090125.csv")
proposed_path = Path(r"/mnt/data/scenario2_metrics_20260102_090125.csv")

df_base = pd.read_csv(baseline_path)
df_prop = pd.read_csv(proposed_path)

# Strip any whitespace in headers
df_base.columns = [c.strip() for c in df_base.columns]
df_prop.columns = [c.strip() for c in df_prop.columns]

# Sort by time
df_base = df_base.sort_values("sim_hours").reset_index(drop=True)
df_prop = df_prop.sort_values("sim_hours").reset_index(drop=True)

def infer_dt(df):
    diffs = np.diff(df["sim_hours"].values)
    diffs = diffs[diffs > 0]
    return float(np.median(diffs)) if len(diffs) else 0.25

dt = float(np.median([infer_dt(df_base), infer_dt(df_prop)]))
print('Inferred timestep (hours):', dt)

required_cols = ['sim_hours', 'P_total_kW', 'feeder_capacity_kW', 'P_A_kW', 'P_B_kW', 'SOC_B', 'T_core_A', 'T_east_A', 'T_north_A', 'T_south_A', 'T_west_A', 'T_core_B', 'T_east_B', 'T_north_B', 'T_south_B', 'T_west_B']
missing_base = [c for c in required_cols if c not in df_base.columns]
missing_prop = [c for c in required_cols if c not in df_prop.columns]
if missing_base or missing_prop:
    raise ValueError(
        f"Missing required columns.\nBaseline missing: {missing_base}\nProposed missing: {missing_prop}"
    )




ZONE_COLS_A = ["T_core_A", "T_east_A", "T_north_A", "T_south_A", "T_west_A"]
ZONE_COLS_B = ["T_core_B", "T_east_B", "T_north_B", "T_south_B", "T_west_B"]

def occupied_mask(sim_hours, occ_start=7.0, occ_end=19.0):
    h = np.asarray(sim_hours)
    return (h >= occ_start) & (h < occ_end)

def feeder_metrics(df, dt_hours, limit_frac=0.8):
    cap = df["feeder_capacity_kW"].to_numpy(dtype=float)
    p = df["P_total_kW"].to_numpy(dtype=float)
    limit = limit_frac * cap
    exceed = np.maximum(p - limit, 0.0)
    viol = exceed > 0
    return {
        "feeder_capacity_mean_kW": float(np.mean(cap)),
        "feeder_limit_mean_kW": float(np.mean(limit)),
        "viol_hours": float(np.sum(viol) * dt_hours),
        "viol_percent_time": float(np.mean(viol) * 100.0),
        "max_exceed_kW": float(np.max(exceed)) if np.any(viol) else 0.0,
        "exceed_energy_kWh": float(np.sum(exceed) * dt_hours),
        "compliance_percent_time": float((1.0 - np.mean(viol)) * 100.0),
    }

def variability_metrics(df, dt_hours, col="P_total_kW"):
    p = df[col].to_numpy(dtype=float)
    dp = np.diff(p) / dt_hours  # kW per hour
    return {
        f"{col}_mean_kW": float(np.mean(p)),
        f"{col}_std_kW": float(np.std(p, ddof=0)),
        f"{col}_cv_percent": float(np.std(p, ddof=0) / (np.mean(p) + 1e-9) * 100.0),
        f"{col}_peak_kW": float(np.max(p)),
        f"{col}_mean_abs_ramp_kW_per_h": float(np.mean(np.abs(dp))) if len(dp) else 0.0,
        f"{col}_max_abs_ramp_kW_per_h": float(np.max(np.abs(dp))) if len(dp) else 0.0,
    }

def comfort_metrics(df, dt_hours, zone_cols, setpoint_C=24.0, occ_start=7.0, occ_end=19.0):
    occ = occupied_mask(df["sim_hours"], occ_start, occ_end)
    out = {"occupied_hours": float(np.sum(occ) * dt_hours)}
    uhd = {}
    ocd = {}
    for z in zone_cols:
        T = df[z].to_numpy(dtype=float)
        Tocc = T[occ]
        uhd[z] = float(np.sum(np.maximum(Tocc - setpoint_C, 0.0)) * dt_hours)
        ocd[z] = float(np.sum(np.maximum(setpoint_C - Tocc, 0.0)) * dt_hours)
        out[f"{z}_UHD_degC_h"] = uhd[z]
        out[f"{z}_OCD_degC_h"] = ocd[z]
        out[f"{z}_max_occ_C"] = float(np.max(Tocc)) if len(Tocc) else np.nan
    out["unmet_degree_hours_sum"] = float(np.sum(list(uhd.values())))
    out["unmet_degree_hours_worst_zone"] = float(np.max(list(uhd.values()))) if uhd else np.nan
    out["overcool_degree_hours_sum"] = float(np.sum(list(ocd.values())))
    out["overcool_degree_hours_worst_zone"] = float(np.max(list(ocd.values()))) if ocd else np.nan
    return out

def tes_metrics(df, dt_hours, soc_col="SOC_B", soc_min=0.20, soc_max=0.99, d_soc_eps=1e-4):
    soc = df[soc_col].to_numpy(dtype=float)
    dsoc = np.diff(soc) / dt_hours
    charge = dsoc > d_soc_eps
    discharge = dsoc < -d_soc_eps
    viol = (soc < soc_min) | (soc > soc_max)

    sign = np.sign(dsoc)
    active = np.abs(dsoc) > d_soc_eps
    sign_active = sign[active]
    switches = int(np.sum(sign_active[1:] * sign_active[:-1] < 0)) if len(sign_active) > 1 else 0

    return {
        "soc_mean": float(np.mean(soc)),
        "soc_min": float(np.min(soc)),
        "soc_max": float(np.max(soc)),
        "soc_swing": float(np.max(soc) - np.min(soc)),
        "soc_viol_hours": float(np.sum(viol) * dt_hours),
        "charge_hours": float(np.sum(charge) * dt_hours),
        "discharge_hours": float(np.sum(discharge) * dt_hours),
        "soc_switch_count": switches,
    }

def add_unoccupied_shading(ax, occ_start=7.0, occ_end=19.0, day_end=24.0):
    ax.axvspan(0, occ_start, color="grey", alpha=0.15, linewidth=0)
    ax.axvspan(occ_end, day_end, color="grey", alpha=0.15, linewidth=0)

def plot_zone_temps(df, building="A", scenario_name="", setpoint_C=24.0, occ_start=7.0, occ_end=19.0):
    cols = ZONE_COLS_A if building.upper() == "A" else ZONE_COLS_B
    x = df["sim_hours"].to_numpy(dtype=float)
    fig = plt.figure(figsize=(10, 4.5))
    ax = fig.gca()

    for c in cols:
        ax.plot(x, df[c].to_numpy(dtype=float), label=c.replace("_", " "))
    ax.plot(x, np.full_like(x, setpoint_C), linestyle="--", label="Setpoint (24°C)")

    add_unoccupied_shading(ax, occ_start, occ_end, 24.0)
    ax.set_xlim(0, 24)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Zone temperature (°C)")
    ax.set_title(f"Building {building.upper()} zone temperatures — {scenario_name}")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    return fig

def plot_powers(df, scenario_name="", limit_frac=0.8):
    x = df["sim_hours"].to_numpy(dtype=float)
    cap = df["feeder_capacity_kW"].to_numpy(dtype=float)
    limit = limit_frac * cap

    fig = plt.figure(figsize=(10, 4.5))
    ax = fig.gca()
    ax.plot(x, df["P_A_kW"].to_numpy(dtype=float), label="P_A_kW")
    ax.plot(x, df["P_B_kW"].to_numpy(dtype=float), label="P_B_kW")
    ax.plot(x, df["P_total_kW"].to_numpy(dtype=float), label="P_total_kW")
    ax.plot(x, cap, linestyle=":", label="Feeder capacity (kW)")
    ax.plot(x, limit, linestyle="--", label=f"{int(limit_frac*100)}% feeder limit (kW)")

    add_unoccupied_shading(ax, 7.0, 19.0, 24.0)
    ax.set_xlim(0, 24)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Power (kW)")
    ax.set_title(f"Building and feeder power — {scenario_name}")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    return fig

def plot_soc(df, scenario_name="", soc_col="SOC_B"):
    x = df["sim_hours"].to_numpy(dtype=float)
    fig = plt.figure(figsize=(10, 4.0))
    ax = fig.gca()
    ax.plot(x, df[soc_col].to_numpy(dtype=float), label=soc_col)

    add_unoccupied_shading(ax, 7.0, 19.0, 24.0)
    ax.set_xlim(0, 24)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("TES SOC (–)")
    ax.set_title(f"Building B TES SOC — {scenario_name}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def compute_all_metrics(df, dt_hours, scenario_name):
    m = {"scenario": scenario_name}
    m.update(feeder_metrics(df, dt_hours, limit_frac=0.8))
    m.update({f"agg_{k}": v for k, v in variability_metrics(df, dt_hours, col="P_total_kW").items()})
    cmA = comfort_metrics(df, dt_hours, ZONE_COLS_A, setpoint_C=24.0)
    cmB = comfort_metrics(df, dt_hours, ZONE_COLS_B, setpoint_C=24.0)
    m.update({f"A_{k}": v for k, v in cmA.items()})
    m.update({f"B_{k}": v for k, v in cmB.items()})
    m.update({f"TES_{k}": v for k, v in tes_metrics(df, dt_hours, soc_col="SOC_B").items()})
    return m

metrics_base = compute_all_metrics(df_base, dt, "Baseline (non-coordinated)")
metrics_prop = compute_all_metrics(df_prop, dt, "Proposed (hierarchical)")

metrics_df = pd.DataFrame([metrics_base, metrics_prop])
metrics_df


# Key comparison table (Proposed vs Baseline)
key_compare = [
    ("viol_hours", "Feeder limit violation hours (lower better)"),
    ("exceed_energy_kWh", "Feeder exceedance energy (kWh) (lower better)"),
    ("agg_P_total_kW_std_kW", "Aggregate feeder power std (kW) (lower better)"),
    ("agg_P_total_kW_mean_abs_ramp_kW_per_h", "Mean abs ramp rate (kW/h) (lower better)"),
    ("A_unmet_degree_hours_sum", "Building A unmet degree-hours sum (lower better)"),
    ("B_unmet_degree_hours_sum", "Building B unmet degree-hours sum (lower better)"),
    ("TES_soc_swing", "TES SOC swing (context-dependent)"),
]

base_row = metrics_df.loc[metrics_df["scenario"].str.contains("Baseline")].iloc[0]
prop_row = metrics_df.loc[metrics_df["scenario"].str.contains("Proposed")].iloc[0]

rows = []
for k, desc in key_compare:
    b = float(base_row[k])
    p = float(prop_row[k])
    pct = (p - b) / (b + 1e-9) * 100.0
    rows.append({"metric_key": k, "description": desc, "baseline": b, "proposed": p, "pct_change_%": pct})

compare_df = pd.DataFrame(rows)
compare_df


# Optional: export summary tables and figures
out_dir = Path("hierarchical_postprocess_outputs")
out_dir.mkdir(exist_ok=True)

metrics_df.to_csv(out_dir / "full_metrics_summary.csv", index=False)
compare_df.to_csv(out_dir / "key_metrics_comparison.csv", index=False)

# Save figures as PNG
for name, fig in {
    "A_zone_temps_baseline": plot_zone_temps(df_base, "A", "Baseline"),
    "A_zone_temps_proposed": plot_zone_temps(df_prop, "A", "Proposed"),
    "B_zone_temps_baseline": plot_zone_temps(df_base, "B", "Baseline"),
    "B_zone_temps_proposed": plot_zone_temps(df_prop, "B", "Proposed"),
    "power_baseline": plot_powers(df_base, "Baseline"),
    "power_proposed": plot_powers(df_prop, "Proposed"),
    "soc_baseline": plot_soc(df_base, "Baseline"),
    "soc_proposed": plot_soc(df_prop, "Proposed"),
}.items():
    fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

print("Saved outputs to:", out_dir.resolve())

