"""
Visualizer - Generate comparison plots

Author: Guowen Li, AI Assistant
Date: 2025-01-20
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Visualizer:
    """Generate visualization plots for scenario comparison"""
    
    def __init__(self):
        # Set plot style
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = {
            'scenario1': '#1f77b4',  # Blue
            'scenario2': '#ff7f0e',  # Orange
            'feeder_limit': '#d62728',  # Red
            'comfort_upper': '#d62728',  # Red
            'comfort_lower': '#2ca02c',  # Green
            'attack': '#ff0000'  # Red
        }
        
        logger.info("📊 Visualizer initialized")
    
    def plot_all(self, 
                 df1: pd.DataFrame, 
                 df2: pd.DataFrame,
                 metrics1: Dict,
                 metrics2: Dict,
                 comparison: Dict,
                 output_dir: str = 'results'):
        """
        Generate all comparison plots
        
        Parameters:
        -----------
        df1, df2 : pd.DataFrame
            Simulation data for both scenarios
        metrics1, metrics2 : Dict
            Summary metrics
        comparison : Dict
            Comparison metrics
        output_dir : str
            Directory to save plots
        """
        
        logger.info(f"📊 Generating comparison plots...")
        
        # 1. Power consumption time series
        self._plot_power_comparison(df1, df2, metrics1, metrics2, output_dir)
        
        # 2. Feeder utilization
        self._plot_feeder_utilization(df1, df2, metrics1, metrics2, output_dir)
        
        # 3. Zone temperatures (Building A)
        self._plot_zone_temperatures(df1, df2, 'A', output_dir)
        
        # 4. Zone temperatures (Building B)
        self._plot_zone_temperatures(df1, df2, 'B', output_dir)
        
        # 5. Comfort violations cumulative
        self._plot_comfort_violations(df1, df2, output_dir)
        
        # 6. TES SOC (Building B)
        self._plot_tes_soc(df1, df2, output_dir)
        
        # 7. Summary bar charts
        self._plot_summary_comparison(metrics1, metrics2, comparison, output_dir)
        
        logger.info(f"✅ All plots saved to {output_dir}/")
    
    def _plot_power_comparison(self, df1, df2, metrics1, metrics2, output_dir):
        """Plot power consumption comparison"""
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # Building A
        axes[0].plot(df1['sim_hours'], df1['P_A_kW'], 
                     label=metrics1['scenario_name'], color=self.colors['scenario1'], linewidth=1.5)
        axes[0].plot(df2['sim_hours'], df2['P_A_kW'], 
                     label=metrics2['scenario_name'], color=self.colors['scenario2'], linewidth=1.5)
        axes[0].fill_between(df1['sim_hours'], 0, df1['P_A_kW'], 
                             where=df1['attack_active'], alpha=0.2, color=self.colors['attack'], label='Attack')
        axes[0].set_ylabel('Building A Power (kW)', fontsize=12)
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Power Consumption Comparison', fontsize=14, fontweight='bold')
        
        # Building B
        axes[1].plot(df1['sim_hours'], df1['P_B_kW'], 
                     label=metrics1['scenario_name'], color=self.colors['scenario1'], linewidth=1.5)
        axes[1].plot(df2['sim_hours'], df2['P_B_kW'], 
                     label=metrics2['scenario_name'], color=self.colors['scenario2'], linewidth=1.5)
        axes[1].set_ylabel('Building B Power (kW)', fontsize=12)
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        # Total with feeder capacity
        feeder_cap = metrics1.get('feeder_capacity_kW', 50.0)
        axes[2].plot(df1['sim_hours'], df1['P_total_kW'], 
                     label=metrics1['scenario_name'], color=self.colors['scenario1'], linewidth=1.5)
        axes[2].plot(df2['sim_hours'], df2['P_total_kW'], 
                     label=metrics2['scenario_name'], color=self.colors['scenario2'], linewidth=1.5)
        axes[2].axhline(feeder_cap, color=self.colors['feeder_limit'], linestyle='--', 
                       linewidth=2, label=f'Feeder Limit ({feeder_cap:.0f} kW)')
        axes[2].set_ylabel('Total Power (kW)', fontsize=12)
        axes[2].set_xlabel('Simulation Time (hours)', fontsize=12)
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/power_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("  ✓ Power comparison plot saved")
    
    def _plot_feeder_utilization(self, df1, df2, metrics1, metrics2, output_dir):
        """Plot feeder utilization"""
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(df1['sim_hours'], df1['feeder_utilization_pct'], 
                label=metrics1['scenario_name'], color=self.colors['scenario1'], linewidth=1.5)
        ax.plot(df2['sim_hours'], df2['feeder_utilization_pct'], 
                label=metrics2['scenario_name'], color=self.colors['scenario2'], linewidth=1.5)
        ax.axhline(100, color=self.colors['feeder_limit'], linestyle='--', linewidth=2, label='100% Capacity')
        
        # Highlight violations
        violations1 = df1[df1['feeder_violated']]
        violations2 = df2[df2['feeder_violated']]
        if len(violations1) > 0:
            ax.scatter(violations1['sim_hours'], violations1['feeder_utilization_pct'], 
                      color='red', marker='x', s=50, label=f'{metrics1["scenario_name"]} Violations')
        if len(violations2) > 0:
            ax.scatter(violations2['sim_hours'], violations2['feeder_utilization_pct'], 
                      color='darkred', marker='o', s=30, label=f'{metrics2["scenario_name"]} Violations')
        
        ax.set_xlabel('Simulation Time (hours)', fontsize=12)
        ax.set_ylabel('Feeder Utilization (%)', fontsize=12)
        ax.set_title('Feeder Utilization Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feeder_utilization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("  ✓ Feeder utilization plot saved")
    
    def _plot_zone_temperatures(self, df1, df2, building: str, output_dir):
        """Plot zone temperatures for a building"""
        
        fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
        zones = ['core', 'east', 'north', 'south', 'west']
        
        for i, zone in enumerate(zones):
            col_name = f'T_{zone}_{building}'
            axes[i].plot(df1['sim_hours'], df1[col_name], 
                        label=df1['scenario_name'].iloc[0] if 'scenario_name' in df1.columns else 'Scenario 1',
                        color=self.colors['scenario1'], linewidth=1.5)
            axes[i].plot(df2['sim_hours'], df2[col_name], 
                        label=df2['scenario_name'].iloc[0] if 'scenario_name' in df2.columns else 'Scenario 2',
                        color=self.colors['scenario2'], linewidth=1.5)
            
            # Comfort bounds
            axes[i].axhline(25, color=self.colors['comfort_upper'], linestyle='--', linewidth=1, alpha=0.5)
            axes[i].axhline(20, color=self.colors['comfort_lower'], linestyle='--', linewidth=1, alpha=0.5)
            
            axes[i].set_ylabel(f'{zone.capitalize()}\nTemp (°C)', fontsize=10)
            axes[i].legend(loc='upper right', fontsize=9)
            axes[i].grid(True, alpha=0.3)
        
        axes[0].set_title(f'Building {building} Zone Temperatures', fontsize=14, fontweight='bold')
        axes[-1].set_xlabel('Simulation Time (hours)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/zone_temperatures_building_{building}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Building {building} zone temperatures plot saved")
    
    def _plot_comfort_violations(self, df1, df2, output_dir):
        """Plot cumulative comfort violations"""
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Building A
        comfort_A_1 = df1['comfort_violation_A_degCh'].cumsum()
        comfort_A_2 = df2['comfort_violation_A_degCh'].cumsum()
        
        axes[0].plot(df1['sim_hours'], comfort_A_1, 
                    label=df1['scenario_name'].iloc[0] if 'scenario_name' in df1.columns else 'Scenario 1',
                    color=self.colors['scenario1'], linewidth=2)
        axes[0].plot(df2['sim_hours'], comfort_A_2, 
                    label=df2['scenario_name'].iloc[0] if 'scenario_name' in df2.columns else 'Scenario 2',
                    color=self.colors['scenario2'], linewidth=2)
        axes[0].set_ylabel('Cumulative Comfort\nViolations (°C·h)', fontsize=12)
        axes[0].set_title('Building A Comfort Violations', fontsize=13, fontweight='bold')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # Building B
        comfort_B_1 = df1['comfort_violation_B_degCh'].cumsum()
        comfort_B_2 = df2['comfort_violation_B_degCh'].cumsum()
        
        axes[1].plot(df1['sim_hours'], comfort_B_1, 
                    label=df1['scenario_name'].iloc[0] if 'scenario_name' in df1.columns else 'Scenario 1',
                    color=self.colors['scenario1'], linewidth=2)
        axes[1].plot(df2['sim_hours'], comfort_B_2, 
                    label=df2['scenario_name'].iloc[0] if 'scenario_name' in df2.columns else 'Scenario 2',
                    color=self.colors['scenario2'], linewidth=2)
        axes[1].set_ylabel('Cumulative Comfort\nViolations (°C·h)', fontsize=12)
        axes[1].set_xlabel('Simulation Time (hours)', fontsize=12)
        axes[1].set_title('Building B Comfort Violations', fontsize=13, fontweight='bold')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/comfort_violations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("  ✓ Comfort violations plot saved")
    
    def _plot_tes_soc(self, df1, df2, output_dir):
        """Plot TES SOC for Building B"""
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(df1['sim_hours'], df1['SOC_B'], 
                label=df1['scenario_name'].iloc[0] if 'scenario_name' in df1.columns else 'Scenario 1',
                color=self.colors['scenario1'], linewidth=2)
        ax.plot(df2['sim_hours'], df2['SOC_B'], 
                label=df2['scenario_name'].iloc[0] if 'scenario_name' in df2.columns else 'Scenario 2',
                color=self.colors['scenario2'], linewidth=2)
        
        ax.set_xlabel('Simulation Time (hours)', fontsize=12)
        ax.set_ylabel('TES State of Charge (SOC)', fontsize=12)
        ax.set_title('Building B Thermal Energy Storage (TES) SOC', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/tes_soc.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("  ✓ TES SOC plot saved")
    
    def _plot_summary_comparison(self, metrics1, metrics2, comparison, output_dir):
        """Plot summary comparison bar charts"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Energy consumption
        categories = ['Building A', 'Building B', 'Total']
        s1_energy = [metrics1['energy_A_kWh'], metrics1['energy_B_kWh'], metrics1['energy_total_kWh']]
        s2_energy = [metrics2['energy_A_kWh'], metrics2['energy_B_kWh'], metrics2['energy_total_kWh']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, s1_energy, width, label=metrics1['scenario_name'], color=self.colors['scenario1'])
        axes[0, 0].bar(x + width/2, s2_energy, width, label=metrics2['scenario_name'], color=self.colors['scenario2'])
        axes[0, 0].set_ylabel('Energy Consumption (kWh)', fontsize=11)
        axes[0, 0].set_title('Energy Consumption', fontsize=12, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(categories)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Comfort violations
        categories = ['Building A', 'Building B', 'Total']
        s1_comfort = [metrics1['comfort_violation_A_total_degCh'], 
                     metrics1['comfort_violation_B_total_degCh'],
                     metrics1['comfort_violation_total_degCh']]
        s2_comfort = [metrics2['comfort_violation_A_total_degCh'],
                     metrics2['comfort_violation_B_total_degCh'],
                     metrics2['comfort_violation_total_degCh']]
        
        axes[0, 1].bar(x - width/2, s1_comfort, width, label=metrics1['scenario_name'], color=self.colors['scenario1'])
        axes[0, 1].bar(x + width/2, s2_comfort, width, label=metrics2['scenario_name'], color=self.colors['scenario2'])
        axes[0, 1].set_ylabel('Comfort Violations (°C·h)', fontsize=11)
        axes[0, 1].set_title('Thermal Comfort Violations', fontsize=12, fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(categories)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Feeder metrics
        categories = ['Avg Util %', 'Peak Util %', 'Violations']
        s1_feeder = [metrics1['feeder_utilization_avg_pct'], 
                    metrics1['feeder_utilization_peak_pct'],
                    metrics1['feeder_violations_count']]
        s2_feeder = [metrics2['feeder_utilization_avg_pct'],
                    metrics2['feeder_utilization_peak_pct'],
                    metrics2['feeder_violations_count']]
        
        axes[1, 0].bar(x - width/2, s1_feeder, width, label=metrics1['scenario_name'], color=self.colors['scenario1'])
        axes[1, 0].bar(x + width/2, s2_feeder, width, label=metrics2['scenario_name'], color=self.colors['scenario2'])
        axes[1, 0].set_ylabel('Value', fontsize=11)
        axes[1, 0].set_title('Feeder Performance', fontsize=12, fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(categories, rotation=15)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Peak power
        categories = ['Building A', 'Building B', 'Total']
        s1_peak = [metrics1['power_A_peak_kW'], metrics1['power_B_peak_kW'], metrics1['power_total_peak_kW']]
        s2_peak = [metrics2['power_A_peak_kW'], metrics2['power_B_peak_kW'], metrics2['power_total_peak_kW']]
        
        axes[1, 1].bar(x - width/2, s1_peak, width, label=metrics1['scenario_name'], color=self.colors['scenario1'])
        axes[1, 1].bar(x + width/2, s2_peak, width, label=metrics2['scenario_name'], color=self.colors['scenario2'])
        axes[1, 1].set_ylabel('Peak Power (kW)', fontsize=11)
        axes[1, 1].set_title('Peak Power Demand', fontsize=12, fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(categories)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/summary_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("  ✓ Summary comparison plot saved")
