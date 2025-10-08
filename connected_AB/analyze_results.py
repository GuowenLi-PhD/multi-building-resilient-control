"""
Post-processing and visualization of hierarchical control results

Author: Guowen Li, AI Assistant
Date: 2025-01-07
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import argparse
import os
import glob
import json

def find_latest_results():
    """Find the most recent results files"""
    results_dir = 'results'
    
    # Find latest metrics file
    metrics_files = glob.glob(os.path.join(results_dir, 'metrics_*.csv'))
    if not metrics_files:
        raise FileNotFoundError("No metrics files found in results directory")
    
    latest_metrics = max(metrics_files, key=os.path.getctime)
    
    # Find corresponding message file
    timestamp = latest_metrics.split('_')[-1].replace('.csv', '')
    messages_file = os.path.join(results_dir, f'messages_{timestamp}.json')
    
    return latest_metrics, messages_file

def plot_comprehensive_results(metrics_file: str, output_dir: str = 'results/plots'):
    """Generate comprehensive visualization of results"""
    
    print(f"📊 Analyzing results from: {metrics_file}")
    
    # Load data
    df = pd.read_csv(metrics_file)
    
    # Convert timestamp to hours
    df['time_hours'] = (df['timestamp'] - df['timestamp'].min()) / 3600
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # ========== PLOT 1: System Overview ==========
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(5, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Power consumption
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df['time_hours'], df['P_A_kW'], 'b-', linewidth=2, label='Building A')
    ax1.plot(df['time_hours'], df['P_B_kW'], 'g-', linewidth=2, label='Building B')
    ax1.plot(df['time_hours'], df['P_total_kW'], 'r-', linewidth=2, label='Total')
    ax1.axhline(y=50, color='k', linestyle='--', linewidth=1, label='Feeder Limit')
    
    # Shade attack periods
    attack_periods = df[df['attack_active'] == True]
    if len(attack_periods) > 0:
        for idx in attack_periods.index:
            if idx == 0 or not df.loc[idx-1, 'attack_active']:
                ax1.axvspan(df.loc[idx, 'time_hours'], 
                           df.loc[idx, 'time_hours'] + 0.25,
                           alpha=0.2, color='red')
    
    ax1.set_ylabel('Power (kW)', fontsize=12, fontweight='bold')
    ax1.set_title('Multi-Building Power Consumption', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Feeder utilization
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(df['time_hours'], 0, df['feeder_utilization_pct'], 
                     alpha=0.3, color='orange')
    ax2.plot(df['time_hours'], df['feeder_utilization_pct'], 'orange', linewidth=2)
    ax2.axhline(y=90, color='red', linestyle='--', linewidth=1, label='Safety Limit (90%)')
    ax2.axhline(y=100, color='darkred', linestyle='-', linewidth=2, label='Hard Limit (100%)')
    ax2.set_ylabel('Utilization (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Feeder Utilization', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 110])
    
    # Building A zone temperatures
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(df['time_hours'], df['T_core_A'], label='Core', linewidth=1.5)
    ax3.plot(df['time_hours'], df['T_east_A'], label='East', linewidth=1, alpha=0.7)
    ax3.plot(df['time_hours'], df['T_north_A'], label='North', linewidth=1, alpha=0.7)
    ax3.plot(df['time_hours'], df['T_south_A'], label='South', linewidth=1, alpha=0.7)
    ax3.plot(df['time_hours'], df['T_west_A'], label='West', linewidth=1, alpha=0.7)
    ax3.axhline(y=26, color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax3.axhline(y=22, color='b', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_ylabel('Temperature (°C)', fontsize=10, fontweight='bold')
    ax3.set_title('Building A Zone Temperatures', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Building B SOC
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(df['time_hours'], df['SOC_B'], 'b-', linewidth=2, label='Actual SOC')
    ax4.plot(df['time_hours'], df['SOC_B_target'], 'r--', linewidth=2, label='Target SOC')
    ax4.axhline(y=0.9, color='g', linestyle='--', linewidth=1, alpha=0.5, label='Pre-charge Target')
    ax4.axhline(y=0.2, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Min SOC')
    ax4.fill_between(df['time_hours'], 0.2, 0.9, alpha=0.1, color='green')
    ax4.set_ylabel('State of Charge', fontsize=10, fontweight='bold')
    ax4.set_title('Building B TES State of Charge', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    # Building A control mode
    ax5 = fig.add_subplot(gs[3, 0])
    mode_mapping = {'nominal': 0, 'adaptive': 1, 'isolation': 2}
    df['mode_A_numeric'] = df['mode_A'].map(mode_mapping)
    ax5.step(df['time_hours'], df['mode_A_numeric'], 'b-', linewidth=2, where='post')
    ax5.set_ylabel('Control Mode', fontsize=10, fontweight='bold')
    ax5.set_yticks([0, 1, 2])
    ax5.set_yticklabels(['Nominal', 'Adaptive', 'Isolation'])
    ax5.set_title('Building A Control Mode', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Building B TES mode
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.step(df['time_hours'], df['TES_mode_B'], 'g-', linewidth=2, where='post')
    ax6.set_ylabel('TES Mode', fontsize=10, fontweight='bold')
    ax6.set_yticks([-1, 0, 1, 2])
    ax6.set_yticklabels(['Charge', 'Off', 'Discharge', 'Chiller'])
    ax6.set_title('Building B TES Operating Mode', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Comfort violations
    ax7 = fig.add_subplot(gs[4, :])
    ax7.bar(df['time_hours'], df['comfort_violation_A'], width=0.2, 
           alpha=0.6, color='blue', label='Building A')
    ax7.bar(df['time_hours'] + 0.2, df['comfort_violation_B'], width=0.2,
           alpha=0.6, color='green', label='Building B')
    ax7.set_ylabel('Violation (°C·h)', fontsize=10, fontweight='bold')
    ax7.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax7.set_title('Thermal Comfort Violations', fontsize=12, fontweight='bold')
    ax7.legend(loc='upper right', fontsize=10)
    ax7.grid(True, alpha=0.3)
    
    # Add attack indicator
    attack_patch = mpatches.Patch(color='red', alpha=0.2, label='Cyber-Attack Period')
    fig.legend(handles=[attack_patch], loc='upper right', fontsize=10)
    
    plt.suptitle('Hierarchical Multi-Building Resilient Control - System Performance',
                fontsize=16, fontweight='bold', y=0.995)
    
    output_file = os.path.join(output_dir, 'system_overview.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()
    
    # ========== PLOT 2: Attack Response Comparison ==========
    if df['attack_active'].sum() > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Before, during, after attack analysis
        normal_data = df[df['attack_active'] == False]
        attack_data = df[df['attack_active'] == True]
        
        # Power comparison
        ax = axes[0, 0]
        categories = ['Normal', 'Under Attack']
        power_a = [normal_data['P_A_kW'].mean(), attack_data['P_A_kW'].mean()]
        power_b = [normal_data['P_B_kW'].mean(), attack_data['P_B_kW'].mean()]
        
        x = np.arange(len(categories))
        width = 0.35
        ax.bar(x - width/2, power_a, width, label='Building A', color='blue', alpha=0.7)
        ax.bar(x + width/2, power_b, width, label='Building B', color='green', alpha=0.7)
        ax.set_ylabel('Average Power (kW)', fontweight='bold')
        ax.set_title('Power Consumption Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Comfort violations comparison
        ax = axes[0, 1]
        comfort_a = [normal_data['comfort_violation_A'].sum(), attack_data['comfort_violation_A'].sum()]
        comfort_b = [normal_data['comfort_violation_B'].sum(), attack_data['comfort_violation_B'].sum()]
        
        ax.bar(x - width/2, comfort_a, width, label='Building A', color='blue', alpha=0.7)
        ax.bar(x + width/2, comfort_b, width, label='Building B', color='green', alpha=0.7)
        ax.set_ylabel('Total Violations (°C·h)', fontweight='bold')
        ax.set_title('Comfort Violation Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # SOC utilization
        ax = axes[1, 0]
        ax.hist([normal_data['SOC_B'], attack_data['SOC_B']], 
               bins=20, label=['Normal', 'Under Attack'],
               alpha=0.7, color=['green', 'red'])
        ax.set_xlabel('State of Charge', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Building B SOC Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Feeder utilization
        ax = axes[1, 1]
        ax.boxplot([normal_data['feeder_utilization_pct'], 
                   attack_data['feeder_utilization_pct']],
                  labels=['Normal', 'Under Attack'])
        ax.axhline(y=90, color='red', linestyle='--', linewidth=1, label='Safety Limit')
        ax.set_ylabel('Feeder Utilization (%)', fontweight='bold')
        ax.set_title('Feeder Utilization Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Attack Response Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, 'attack_response_analysis.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    # ========== PRINT SUMMARY STATISTICS ==========
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    print("\n📊 OVERALL METRICS:")
    print(f"  Total energy consumed: {(df['P_total_kW'].sum() * 0.25):.2f} kWh")
    print(f"  Average power: {df['P_total_kW'].mean():.2f} kW")
    print(f"  Peak power: {df['P_total_kW'].max():.2f} kW")
    print(f"  Average feeder utilization: {df['feeder_utilization_pct'].mean():.1f}%")
    print(f"  Peak feeder utilization: {df['feeder_utilization_pct'].max():.1f}%")
    print(f"  Feeder violations: {df['feeder_violated'].sum()} timesteps")
    
    print("\n🏢 BUILDING A (Victim):")
    print(f"  Total energy: {(df['P_A_kW'].sum() * 0.25):.2f} kWh")
    print(f"  Average power: {df['P_A_kW'].mean():.2f} kW")
    print(f"  Total comfort violations: {df['comfort_violation_A'].sum():.2f} °C·h")
    
    if df['attack_active'].sum() > 0:
        normal_power_a = df[~df['attack_active']]['P_A_kW'].mean()
        attack_power_a = df[df['attack_active']]['P_A_kW'].mean()
        power_increase = (attack_power_a - normal_power_a) / normal_power_a * 100
        print(f"  Power increase during attack: {power_increase:.1f}%")
        
        normal_comfort_a = df[~df['attack_active']]['comfort_violation_A'].sum()
        attack_comfort_a = df[df['attack_active']]['comfort_violation_A'].sum()
        if normal_comfort_a > 0:
            comfort_ratio = attack_comfort_a / normal_comfort_a
            print(f"  Comfort degradation ratio: {comfort_ratio:.2f}x")
    
    print("\n🏢 BUILDING B (Support with TES):")
    print(f"  Total energy: {(df['P_B_kW'].sum() * 0.25):.2f} kWh")
    print(f"  Average power: {df['P_B_kW'].mean():.2f} kW")
    print(f"  Average SOC: {df['SOC_B'].mean():.2f}")
    print(f"  SOC range: [{df['SOC_B'].min():.2f}, {df['SOC_B'].max():.2f}]")
    print(f"  Total comfort violations: {df['comfort_violation_B'].sum():.2f} °C·h")
    
    if df['attack_active'].sum() > 0:
        soc_change = df[~df['attack_active']]['SOC_B'].mean() - df[df['attack_active']]['SOC_B'].mean()
        print(f"  TES utilization (ΔSOC): {soc_change:.2f}")
    
    print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description='Analyze hierarchical control results')
    parser.add_argument('--metrics', type=str, default=None,
                       help='Path to metrics CSV file (default: latest in results/)')
    parser.add_argument('--output', type=str, default='results/plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Find metrics file
    if args.metrics is None:
        metrics_file, _ = find_latest_results()
        print(f"📂 Using latest results: {metrics_file}")
    else:
        metrics_file = args.metrics
    
    # Generate plots and analysis
    plot_comprehensive_results(metrics_file, args.output)
    
    print("\n✅ Analysis complete!")
    print(f"📊 Plots saved to: {args.output}")

if __name__ == "__main__":
    main()