"""
Visualization for Hierarchical Control Simulation Results

Generates publication-quality plots for:
  1. Power profiles with feeder limit and budget allocations
  2. Building B TES state-of-charge
  3. Budget violation analysis
  4. Aggregator allocation over time

Author: Guowen Li, AI Assistant
Date: 2025-02
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Dict
import os


def plot_results(results: Dict, save_path: str = "results"):
    """Generate all plots from simulation results."""
    os.makedirs(save_path, exist_ok=True)
    
    # Extract data
    t_A = np.array(results['building_A']['time']) / 3600.0  # hours
    P_A = np.array(results['building_A']['power_kW'])
    B_A = np.array(results['building_A']['budget_kW'])
    
    t_B = np.array(results['building_B']['time']) / 3600.0
    P_B = np.array(results['building_B']['power_kW'])
    B_B = np.array(results['building_B']['budget_kW'])
    
    t_agg = np.array(results['aggregator']['time']) / 3600.0
    P_total = np.array(results['aggregator']['total_power'])
    P_alloc = np.array(results['aggregator']['allocated'])
    feeder = np.array(results['aggregator']['feeder'])
    
    # SOC from Building B logs
    SOC = [l.extra.get('SOC', 0.5) for l in results['step_logs'] if l.building_id == 'Building_B']
    t_SOC = t_B[:len(SOC)]
    
    # ── Figure 1: Community Power Profile ──
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                              gridspec_kw={'height_ratios': [2, 2, 1]})
    
    # Subplot 1: Building A
    ax1 = axes[0]
    ax1.step(t_A, P_A, 'b-', linewidth=1.2, where='post', label='Building A power')
    ax1.step(t_A, B_A, 'b--', linewidth=0.8, where='post', alpha=0.7, label='Budget (allocated)')
    ax1.set_ylabel('Power [kW]', fontsize=11)
    ax1.set_title('Building A (AHU-VAV-Chiller, no TES)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=-0.5)
    
    # Attack shading
    ax1.axvspan(8, 10, alpha=0.15, color='red', label='Attack period')
    ax1.annotate('DoS Attack\n+ Feeder Limit', xy=(9, ax1.get_ylim()[1]*0.8),
                fontsize=8, ha='center', color='red', fontweight='bold')
    
    # Subplot 2: Building B
    ax2 = axes[1]
    ax2.step(t_B, P_B, 'g-', linewidth=1.2, where='post', label='Building B power')
    ax2.step(t_B, B_B, 'g--', linewidth=0.8, where='post', alpha=0.7, label='Budget (allocated)')
    ax2.set_ylabel('Power [kW]', fontsize=11)
    ax2.set_title('Building B (HVAC + TES)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=-0.5)
    ax2.axvspan(8, 10, alpha=0.15, color='red')
    
    # Subplot 3: Community total vs feeder
    ax3 = axes[2]
    P_total_at_A = []
    for t_val in t_A:
        # Find closest aggregator step
        idx = np.searchsorted(t_agg, t_val, side='right') - 1
        idx = max(0, min(idx, len(P_total) - 1))
        P_total_at_A.append(P_total[idx])
    
    ax3.step(t_agg, P_total, 'k-', linewidth=1.5, where='post', label='Total community power')
    ax3.step(t_agg, feeder, 'r-', linewidth=2.0, where='post', label='Feeder limit')
    ax3.step(t_agg, P_alloc, 'k--', linewidth=0.8, where='post', alpha=0.5, label='Total allocated')
    ax3.fill_between(t_agg, P_total, feeder, where=np.array(P_total) > np.array(feeder),
                     alpha=0.3, color='red', step='post', label='Violation')
    ax3.set_ylabel('Power [kW]', fontsize=11)
    ax3.set_xlabel('Time [hours]', fontsize=11)
    ax3.set_title('Community Total vs Feeder Limit', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=-0.5)
    ax3.axvspan(8, 10, alpha=0.15, color='red')
    
    for ax in axes:
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 2))
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_path, 'power_profiles.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # ── Figure 2: Aggregator Allocation Detail ──
    fig2, (ax_alloc, ax_soc) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    
    # Stacked area for allocation
    budgets_A = [l.budgets.get('Building_A', 0) for l in results['agg_logs']]
    budgets_B = [l.budgets.get('Building_B', 0) for l in results['agg_logs']]
    
    ax_alloc.fill_between(t_agg, 0, budgets_A, alpha=0.5, color='#2196F3', step='post', label='Building A budget')
    ax_alloc.fill_between(t_agg, budgets_A, np.array(budgets_A) + np.array(budgets_B),
                          alpha=0.5, color='#4CAF50', step='post', label='Building B budget')
    ax_alloc.step(t_agg, feeder, 'r-', linewidth=2.0, where='post', label='Feeder limit')
    ax_alloc.set_ylabel('Power budget [kW]', fontsize=11)
    ax_alloc.set_title('Aggregator Power Allocation (Stacked)', fontsize=12, fontweight='bold')
    ax_alloc.legend(loc='upper right', fontsize=9)
    ax_alloc.grid(True, alpha=0.3)
    ax_alloc.axvspan(8, 10, alpha=0.15, color='red')
    
    # SOC trajectory
    if SOC:
        ax_soc.step(t_SOC, SOC, 'purple', linewidth=1.5, where='post')
        ax_soc.axhline(y=0.20, color='r', linestyle='--', linewidth=0.8, label='SOC bounds')
        ax_soc.axhline(y=0.99, color='r', linestyle='--', linewidth=0.8)
        ax_soc.fill_between(t_SOC, 0.20, 0.99, alpha=0.05, color='green')
        ax_soc.set_ylabel('TES SOC [-]', fontsize=11)
        ax_soc.set_xlabel('Time [hours]', fontsize=11)
        ax_soc.set_title('Building B — TES State of Charge', fontsize=12, fontweight='bold')
        ax_soc.set_ylim(0, 1.05)
        ax_soc.legend(loc='upper right', fontsize=9)
        ax_soc.grid(True, alpha=0.3)
        ax_soc.axvspan(8, 10, alpha=0.15, color='red')
    
    for ax in [ax_alloc, ax_soc]:
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 2))
    
    plt.tight_layout()
    fig2.savefig(os.path.join(save_path, 'allocation_detail.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # ── Figure 3: Priority weights ──
    fig3, ax_pri = plt.subplots(1, 1, figsize=(14, 3))
    
    pri_A = [l.priorities.get('Building_A', 2) for l in results['agg_logs']]
    pri_B = [l.priorities.get('Building_B', 2) for l in results['agg_logs']]
    
    ax_pri.step(t_agg, pri_A, 'b-', linewidth=1.5, where='post', label='Building A priority (ω)')
    ax_pri.step(t_agg, pri_B, 'g-', linewidth=1.5, where='post', label='Building B priority (ω)')
    ax_pri.set_ylabel('Priority weight ω', fontsize=11)
    ax_pri.set_xlabel('Time [hours]', fontsize=11)
    ax_pri.set_title('Energy Priority Weights', fontsize=12, fontweight='bold')
    ax_pri.set_yticks([1, 2, 3])
    ax_pri.set_yticklabels(['LOW (1)', 'MEDIUM (2)', 'HIGH (3)'])
    ax_pri.legend(loc='upper right', fontsize=9)
    ax_pri.grid(True, alpha=0.3)
    ax_pri.set_xlim(0, 24)
    ax_pri.axvspan(8, 10, alpha=0.15, color='red')
    
    plt.tight_layout()
    fig3.savefig(os.path.join(save_path, 'priority_weights.png'), dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    print(f"\n✓ Plots saved to {save_path}/")
    print(f"  - power_profiles.png")
    print(f"  - allocation_detail.png")
    print(f"  - priority_weights.png")


if __name__ == "__main__":
    # Run mock demo and plot
    from main_simulation import run_mock_demo
    results = run_mock_demo()
    plot_results(results, save_path="/home/claude/hierarchical_control/results")
