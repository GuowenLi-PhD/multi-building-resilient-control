"""
Script to analyze and compare Nominal MPC vs Adaptive MPC performance

Usage:
    python analyze_mpc_performance.py
"""

import pandas as pd
from metrics_analyzer import MPCPerformanceAnalyzer

# TOU pricing schedule
price_tou = [0.0640, 0.0640, 0.0640, 0.0640,  # 00:00-03:59
             0.0640, 0.0640, 0.0640, 0.0640,  # 04:00-07:59
             0.1391, 0.1391, 0.1391, 0.1391,  # 08:00-11:59
             0.3548, 0.3548, 0.3548, 0.3548,  # 12:00-15:59 (HIGH PRICE)
             0.3548, 0.3548, 0.1391, 0.1391,  # 16:00-19:59
             0.1391, 0.1391, 0.1391, 0.0640]  # 20:00-23:59

# Load simulation results
results_nominal = pd.read_csv('mpc_results/results_opt_PH4.csv', index_col=0)
results_attack = pd.read_csv('mpc_results/results_DoS_attack_core_VAV_opt_PH4.csv', index_col=0)

# Create analyzer instance
analyzer = MPCPerformanceAnalyzer(
    results_nominal=results_nominal,
    results_attack=results_attack,
    price_tou=price_tou,
    dt=900  # 15-minute timestep
)

# Generate comprehensive report
report = analyzer.generate_comprehensive_report(
    save_path='mpc_results/comprehensive_analysis_buildingA.json'
)