"""
Shared utility functions for the hierarchical MPC framework.

Provides:  logging factory, YAML loader, time/comfort/pricing/SOC helpers.

Author  : Guowen Li
Date    : 2026-02
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import yaml


# ── Logging ─────────────────────────────────────────────────────────────────

_LOG_FMT = "[%(asctime)s] %(name)-26s %(levelname)-7s  %(message)s"


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a named logger with a consistent format (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(_LOG_FMT, datefmt="%H:%M:%S"))
        logger.addHandler(h)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def load_config(path: str) -> dict:
    """Load YAML configuration file and return as dict."""
    with open(path) as f:
        return yaml.safe_load(f)


# ── Time helpers ────────────────────────────────────────────────────────────

def seconds_to_hour(ts: float) -> float:
    """Simulation time [s] → hour-of-day [0, 24)."""
    return (ts % 86400) / 3600.0


def is_occupied(hour: float, start: int = 7, end: int = 19) -> bool:
    """Return True if the given hour falls within occupied period."""
    return start <= hour < end


# ── Pricing ─────────────────────────────────────────────────────────────────

def get_price(hour: float, cfg: dict) -> float:
    """Return $/kWh for the given hour using TOU schedule in *cfg*."""
    h = int(hour) % 24
    if h in cfg.get("on_peak_hours", []):
        return cfg["on_peak_rate"]
    if h in cfg.get("mid_peak_hours", []):
        return cfg["mid_peak_rate"]
    return cfg["off_peak_rate"]


def price_forecast(start_hour: float, n: int, dt: float, cfg: dict) -> np.ndarray:
    """Return an (n,) vector of electricity prices over the horizon."""
    return np.array([
        get_price(start_hour + k * dt / 3600.0, cfg) for k in range(n)
    ])


# ── Comfort ─────────────────────────────────────────────────────────────────

def comfort_bounds(hour: float, cfg: dict):
    """Return (T_min, T_max) based on occupancy schedule in *cfg*."""
    occ = is_occupied(
        hour, cfg.get("occ_start_hour", 7), cfg.get("occ_end_hour", 19)
    )
    if occ:
        return cfg["Tz_min_occ"], cfg["Tz_max_occ"]
    return cfg["Tz_min_unocc"], cfg["Tz_max_unocc"]


def comfort_violation(Tz: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Per-zone thermal comfort violation [K] (positive = violated)."""
    return np.maximum(0.0, lo - Tz) + np.maximum(0.0, Tz - hi)


# ── SOC target ──────────────────────────────────────────────────────────────

def soc_target(hour: float, cfg: dict) -> float:
    """Return time-of-day SOC target for Building B's TES."""
    tgt = cfg.get("soc_target_by_hour", {})
    if hour < 6 or hour >= 23:
        return tgt.get("off_peak", 0.70)
    if 8 <= hour < 18:
        return tgt.get("daytime", 0.50)
    return tgt.get("evening", 0.40)


# ── Array padding ───────────────────────────────────────────────────────────

def pad_to(arr, n: int, fill: float = 0.0) -> np.ndarray:
    """Ensure *arr* has exactly length *n*, padding with *fill* if shorter."""
    a = np.asarray(arr, dtype=float).ravel()
    if a.size >= n:
        return a[:n].copy()
    return np.concatenate([a, np.full(n - a.size, fill)])


# ── Metric helpers ──────────────────────────────────────────────────────────

def unmet_degree_hours(Tz_history: np.ndarray, lo: float, hi: float,
                        dt_s: float) -> float:
    """
    Cumulative unmet degree-hours [°C·h].

    Parameters
    ----------
    Tz_history : (T, n_zones) array of zone temperatures over time.
    lo, hi     : comfort bounds.
    dt_s       : timestep in seconds.
    """
    viol = np.maximum(0.0, lo - Tz_history) + np.maximum(0.0, Tz_history - hi)
    return float(np.sum(viol) * dt_s / 3600.0)
