"""
Attack scenario manager.

Reads ``attacks`` from YAML config and calls inject/clear on the
appropriate building wrapper at the correct simulation time.

Author  : Guowen Li
Date    : 2026-02
"""

from __future__ import annotations

from typing import List

from utils.helpers import get_logger

logger = get_logger("attack_mgr")


class AttackManager:
    """
    Manages scheduled cyber-attack injection and clearance.

    The attack schedule is read from the ``attacks`` key of the YAML config.
    Start times in the config are *absolute* seconds; this class converts
    them to simulation-relative offsets using ``start_day``.
    """

    def __init__(self, attack_list: List[dict], start_day: int):
        self.attacks: List[dict] = []
        day_offset = start_day * 86400

        for atk in attack_list:
            t0 = atk["start_time_s"] - day_offset
            self.attacks.append({
                "name": atk["name"],
                "target": atk["target"],
                "type": atk.get("type", "dos_vav_reinit"),
                "zone": atk.get("affected_zone", "core"),
                "start": t0,
                "end": t0 + atk["duration_s"],
                "active": False,
            })

        logger.info("Loaded %d attack scenario(s)", len(self.attacks))
        for a in self.attacks:
            logger.info(
                "  '%s' on %s  [%.0f → %.0f s]",
                a["name"], a["target"], a["start"], a["end"],
            )

    # ─────────────────────────────────────────────────────────────────

    def update(self, sim_time: float, building_a, building_b) -> None:
        """Inject or clear each attack based on current *sim_time*."""
        for atk in self.attacks:
            tgt = building_a if atk["target"] == "building_a" else building_b

            if atk["start"] <= sim_time < atk["end"]:
                if not atk["active"]:
                    atk["active"] = True
                    tgt.inject_attack(atk["type"], atk["zone"])
                    logger.warning(
                        "▶ ATTACK START: '%s' on %s at t=%.0fs",
                        atk["name"], atk["target"], sim_time,
                    )
            else:
                if atk["active"]:
                    atk["active"] = False
                    tgt.clear_attack()
                    logger.info(
                        "■ ATTACK END: '%s' at t=%.0fs", atk["name"], sim_time
                    )

    def any_active(self) -> bool:
        """Return True if any attack is currently active."""
        return any(a["active"] for a in self.attacks)
