"""
Analysis and Visualization Script

Analyzes simulation results and generates plots

Usage:
    python analyze_results.py results/metrics_YYYYMMDD_HHMMSS.csv

Author: Guowen Li
Date: 2025-02-06
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def analyze_results(metrics_file: str):
    """Analyze simulation results"""
    
    print("="*80)
    print("📊 ANALYZING SIMULATION RESULTS")
    print("="*80)
    print(f"File: {metrics_file}")
    print("")
    
    # Load data
    df = pd.DataFrame(pd.read_csv(metrics_file))
    
    print("📈 PERFORMANCE METRICS")
    print("="*80)
    print(f"Total timesteps: {len(df)}")
    print(f"Simulation duration: {df['timestamp'].max() / 86400:.2f} days")
    print("")
    
    print("⏱️  Computational Performance:")
    print(f"  Avg cycle time: {df['cycle_time'].mean():.3f}s")
    print(f"  Max cycle time: {df['cycle_time'].max():.3f}s")
    print(f"  Avg flexibility time: {df['flex_time'].mean():.3f}s")
    print(f"  Avg aggregator time: {df['agg_time'].mean():.3f}s")
    print(f"  Avg MPC time: {df['mpc_time'].mean():.3f}s")
    print("")
    
    print("⚡ Power & Feeder:")
    print(f"  Avg total power: {df['total_power'].mean():.2f} kW")
    print(f"  Max total power: {df['total_power'].max():.2f} kW")
    print(f"  Avg utilization: {df['utilization'].mean():.1f}%")
    print(f"  Max utilization: {df['utilization'].max():.1f}%")
    print(f"  Feeder violations: {(df['utilization'] > 100).sum()}")
    print("")
    
    # Generate plots
    print("📊 Generating plots...")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Power vs Feeder Limit
    ax = axes[0]
    time_days = df['timestamp'] / 86400
    ax.plot(time_days, df['total_power'], 'b-', label='Total Power', linewidth=2)
    ax.plot(time_days, df['feeder_limit'], 'r--', label='Feeder Limit', linewidth=2)
    ax.fill_between(time_days, 0, df['feeder_limit'], alpha=0.2, color='red')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Power (kW)')
    ax.set_title('Total Power vs Feeder Limit')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Feeder Utilization
    ax = axes[1]
    ax.plot(time_days, df['utilization'], 'g-', linewidth=2)
    ax.axhline(y=100, color='r', linestyle='--', label='100% Limit')
    ax.fill_between(time_days, 0, df['utilization'], alpha=0.3, color='green')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Utilization (%)')
    ax.set_title('Feeder Utilization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Computational Times
    ax = axes[2]
    ax.plot(time_days, df['cycle_time'], 'b-', label='Total Cycle', alpha=0.7)
    ax.plot(time_days, df['flex_time'], 'r-', label='Flexibility', alpha=0.7)
    ax.plot(time_days, df['agg_time'], 'g-', label='Aggregator', alpha=0.7)
    ax.plot(time_days, df['mpc_time'], 'm-', label='MPC', alpha=0.7)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Computational Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_file = metrics_file.replace('.csv', '_plots.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved plots to {output_file}")
    
    plt.show()
    
    print("")
    print("="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Analyze simulation results')
    parser.add_argument('metrics_file', help='Path to metrics CSV file')
    
    args = parser.parse_args()
    
    try:
        analyze_results(args.metrics_file)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
