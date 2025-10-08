"""
Metrics Collector - Performance evaluation and analysis

Author: Guowen Li, AI Assistant
Date: 2025-01-07
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

import sys
import os
# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    print(f"🔧 Current directory added to path: {current_dir}")
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    print(f"🔧 Parent directory added to path: {parent_dir}")

from communication.data_models import BuildingState, FeederStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and analyzes performance metrics"""
    
    def __init__(self):
        self.data = []
    
    def record_step(self,
                   timestamp: float,
                   building_a_state: BuildingState,
                   building_b_state: BuildingState,
                   feeder_status: FeederStatus,
                   aggregator_result: Dict,
                   attack_active: bool):
        """Record metrics for one timestep"""
        
        record = {
            'timestamp': timestamp,
            
            # Building A
            'P_A_kW': building_a_state.power_actual_kW,
            'status_A': building_a_state.status.value,
            'mode_A': building_a_state.control_mode.value,
            'comfort_violation_A': building_a_state.comfort_violations,
            'T_core_A': building_a_state.zone_temperatures.get('core', np.nan),
            'T_east_A': building_a_state.zone_temperatures.get('east', np.nan),
            'T_north_A': building_a_state.zone_temperatures.get('north', np.nan),
            'T_south_A': building_a_state.zone_temperatures.get('south', np.nan),
            'T_west_A': building_a_state.zone_temperatures.get('west', np.nan),
            
            # Building B
            'P_B_kW': building_b_state.power_actual_kW,
            'SOC_B': building_b_state.extra_data.get('SOC_current', np.nan),
            'TES_mode_B': building_b_state.extra_data.get('TES_mode', 0),
            'comfort_violation_B': building_b_state.comfort_violations,
            'T_core_B': building_b_state.zone_temperatures.get('core', np.nan),
            
            # Feeder
            'P_total_kW': feeder_status.total_power_kW,
            'feeder_utilization_pct': feeder_status.utilization_percent,
            'feeder_margin_kW': feeder_status.margin_kW,
            'feeder_violated': feeder_status.constraint_violated,
            
            # Aggregator
            'P_A_ref_kW': aggregator_result['P_A_ref'][0],
            'P_B_ref_kW': aggregator_result['P_B_ref'][0],
            'SOC_B_target': aggregator_result['SOC_B_target'],
            
            # Attack status
            'attack_active': attack_active
        }
        
        self.data.append(record)
    
    def save_to_csv(self, filepath: str):
        """Save metrics to CSV"""
        df = pd.DataFrame(self.data)
        df.to_csv(filepath, index=False)
        logger.info(f"✓ Saved {len(df)} records to {filepath}")
    
    def generate_summary_report(self, filepath: str):
        """Generate summary statistics"""
        
        df = pd.DataFrame(self.data)
        
        with open(filepath, 'w') as f:
            f.write("="*80 + "\n")
            f.write("HIERARCHICAL MULTI-BUILDING CONTROL - PERFORMANCE SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            # Simulation info
            f.write("SIMULATION INFO\n")
            f.write("-"*80 + "\n")
            f.write(f"Duration: {(df['timestamp'].max() - df['timestamp'].min())/3600:.1f} hours\n")
            f.write(f"Timesteps: {len(df)}\n")
            f.write(f"Attack periods: {df['attack_active'].sum()} timesteps\n\n")
            
            # Building A metrics
            f.write("BUILDING A (Victim)\n")
            f.write("-"*80 + "\n")
            f.write(f"Total energy: {df['P_A_kW'].sum() * 0.25:.2f} kWh\n")  # 15-min timesteps
            f.write(f"Average power: {df['P_A_kW'].mean():.2f} kW\n")
            f.write(f"Peak power: {df['P_A_kW'].max():.2f} kW\n")
            f.write(f"Total comfort violations: {df['comfort_violation_A'].sum():.2f} °C·h\n")
            f.write(f"Attack periods power: {df[df['attack_active']]['P_A_kW'].mean():.2f} kW\n")
            f.write(f"Normal periods power: {df[~df['attack_active']]['P_A_kW'].mean():.2f} kW\n\n")
            
            # Building B metrics
            f.write("BUILDING B (Support with TES)\n")
            f.write("-"*80 + "\n")
            f.write(f"Total energy: {df['P_B_kW'].sum() * 0.25:.2f} kWh\n")
            f.write(f"Average power: {df['P_B_kW'].mean():.2f} kW\n")
            f.write(f"Peak power: {df['P_B_kW'].max():.2f} kW\n")
            f.write(f"Average SOC: {df['SOC_B'].mean():.2f}\n")
            f.write(f"Min SOC: {df['SOC_B'].min():.2f}\n")
            f.write(f"Total comfort violations: {df['comfort_violation_B'].sum():.2f} °C·h\n\n")
            
            # Feeder metrics
            f.write("FEEDER STATUS\n")
            f.write("-"*80 + "\n")
            f.write(f"Average utilization: {df['feeder_utilization_pct'].mean():.1f}%\n")
            f.write(f"Peak utilization: {df['feeder_utilization_pct'].max():.1f}%\n")
            f.write(f"Violations: {df['feeder_violated'].sum()} timesteps\n")
            f.write(f"Average margin: {df['feeder_margin_kW'].mean():.2f} kW\n\n")
            
            # Resilience metrics
            f.write("RESILIENCE METRICS\n")
            f.write("-"*80 + "\n")
            if df['attack_active'].sum() > 0:
                comfort_reduction_A = (df[df['attack_active']]['comfort_violation_A'].sum() / 
                                      df[~df['attack_active']]['comfort_violation_A'].sum() - 1) * 100
                f.write(f"Building A comfort degradation during attack: {comfort_reduction_A:.1f}%\n")
                
                power_increase_A = (df[df['attack_active']]['P_A_kW'].mean() / 
                                   df[~df['attack_active']]['P_A_kW'].mean() - 1) * 100
                f.write(f"Building A power increase during attack: {power_increase_A:.1f}%\n")
                
                SOC_utilization = df[~df['attack_active']]['SOC_B'].mean() - df[df['attack_active']]['SOC_B'].mean()
                f.write(f"Building B TES utilization (ΔSOC): {SOC_utilization:.2f}\n")
            
            f.write("\n" + "="*80 + "\n")
        
        logger.info(f"✓ Summary report generated: {filepath}")