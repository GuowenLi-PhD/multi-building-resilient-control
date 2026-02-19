"""
Metrics Collector - Record and save simulation metrics

Author: Guowen Li, AI Assistant
Date: 2025-01-20
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collect simulation metrics for analysis"""
    
    def __init__(self, scenario_name: str):
        self.scenario_name = scenario_name
        self.data = []
        
        logger.info(f"📊 Metrics collector initialized for '{scenario_name}'")
    
    def record_step(self,
                   timestamp: float,
                   building_a_state: Dict,
                   building_b_state: Dict,
                   feeder_total_power: float,
                   feeder_capacity: float,
                   attack_active: bool,
                   attack_name: str = ""):
        """Record metrics for one timestep"""
        
        record = {
            # Time
            'timestamp': timestamp,
            'sim_hours': (timestamp - self.data[0]['timestamp']) / 3600 if len(self.data) > 0 else 0,
            
            # Building A
            'P_A_kW': building_a_state['power_kW'],
            'comfort_violation_A_degCh': building_a_state['comfort_violation_degCh'],
            'T_core_A': building_a_state['zone_temps']['core'],
            'T_east_A': building_a_state['zone_temps']['east'],
            'T_north_A': building_a_state['zone_temps']['north'],
            'T_south_A': building_a_state['zone_temps']['south'],
            'T_west_A': building_a_state['zone_temps']['west'],
            'V_core_A': building_a_state['zone_airflows']['core'],
            'V_east_A': building_a_state['zone_airflows']['east'],
            'V_north_A': building_a_state['zone_airflows']['north'],
            'V_south_A': building_a_state['zone_airflows']['south'],
            'V_west_A': building_a_state['zone_airflows']['west'],
            'scheduled_vars_A': ','.join(building_a_state.get('scheduled_vars', [])),
            'under_attack_A': building_a_state.get('under_attack', False),
            
            # Building A control actions
            'bcp_A': building_a_state.get('controls_applied', {}).get('bcp', np.nan),
            'bahu_A': building_a_state.get('controls_applied', {}).get('bahu', np.nan),
            'Tsa_A': building_a_state.get('controls_applied', {}).get('Tsa', np.nan),
            'Vcore_A': building_a_state.get('controls_applied', {}).get('Vcore', np.nan),
            
            # Building B
            'P_B_kW': building_b_state['power_kW'],
            'SOC_B': building_b_state.get('SOC', np.nan),
            'TES_mode_B': building_b_state.get('TES_mode', 0),
            'comfort_violation_B_degCh': building_b_state['comfort_violation_degCh'],
            'T_core_B': building_b_state['zone_temps']['core'],
            'T_east_B': building_b_state['zone_temps']['east'],
            'T_north_B': building_b_state['zone_temps']['north'],
            'T_south_B': building_b_state['zone_temps']['south'],
            'T_west_B': building_b_state['zone_temps']['west'],
            'V_core_B': building_b_state['zone_airflows']['core'],
            'V_east_B': building_b_state['zone_airflows']['east'],
            'V_north_B': building_b_state['zone_airflows']['north'],
            'V_south_B': building_b_state['zone_airflows']['south'],
            'V_west_B': building_b_state['zone_airflows']['west'],
            'scheduled_vars_B': ','.join(building_b_state.get('scheduled_vars', [])),
            'under_attack_B': building_b_state.get('under_attack', False),
            
            # Feeder
            'P_total_kW': feeder_total_power,
            'feeder_capacity_kW': feeder_capacity,
            'feeder_utilization_pct': (feeder_total_power / feeder_capacity * 100),
            'feeder_margin_kW': feeder_capacity - feeder_total_power,
            'feeder_violated': feeder_total_power > feeder_capacity,
            
            # Attack status
            'attack_active': attack_active,
            'attack_name': attack_name
        }
        
        self.data.append(record)
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get collected data as DataFrame"""
        return pd.DataFrame(self.data)
    
    def save_to_csv(self, filepath: str):
        """Save metrics to CSV"""
        df = self.get_dataframe()
        df.to_csv(filepath, index=False)
        logger.info(f"✅ Saved {len(df)} records to {filepath}")
    
    def calculate_summary_metrics(self) -> Dict:
        """Calculate summary performance metrics"""
        
        df = self.get_dataframe()
        
        if len(df) == 0:
            logger.warning("⚠️ No data to calculate metrics")
            return {}
        
        # Time step duration (assume uniform)
        dt_hours = df['sim_hours'].diff().mean() if len(df) > 1 else 0.25
        
        # Building A metrics
        metrics = {
            'scenario_name': self.scenario_name,
            'simulation_duration_hours': df['sim_hours'].max(),
            
            # Energy consumption
            'energy_A_kWh': df['P_A_kW'].sum() * dt_hours,
            'energy_B_kWh': df['P_B_kW'].sum() * dt_hours,
            'energy_total_kWh': (df['P_A_kW'] + df['P_B_kW']).sum() * dt_hours,
            
            # Power statistics
            'power_A_avg_kW': df['P_A_kW'].mean(),
            'power_A_peak_kW': df['P_A_kW'].max(),
            'power_B_avg_kW': df['P_B_kW'].mean(),
            'power_B_peak_kW': df['P_B_kW'].max(),
            'power_total_avg_kW': df['P_total_kW'].mean(),
            'power_total_peak_kW': df['P_total_kW'].max(),
            
            # Comfort violations
            'comfort_violation_A_total_degCh': df['comfort_violation_A_degCh'].sum(),
            'comfort_violation_B_total_degCh': df['comfort_violation_B_degCh'].sum(),
            'comfort_violation_total_degCh': (df['comfort_violation_A_degCh'] + df['comfort_violation_B_degCh']).sum(),
            
            # Feeder metrics
            'feeder_utilization_avg_pct': df['feeder_utilization_pct'].mean(),
            'feeder_utilization_peak_pct': df['feeder_utilization_pct'].max(),
            'feeder_violations_count': int(df['feeder_violated'].sum()),
            'feeder_violations_pct': df['feeder_violated'].mean() * 100,
            'feeder_margin_avg_kW': df['feeder_margin_kW'].mean(),
            'feeder_margin_min_kW': df['feeder_margin_kW'].min(),
            'feeder_stability_std_kW': df['P_total_kW'].std(),
            
            # TES metrics (Building B)
            'SOC_avg': df['SOC_B'].mean(),
            'SOC_min': df['SOC_B'].min(),
            'SOC_max': df['SOC_B'].max(),
            
            # Attack periods
            'attack_timesteps': int(df['attack_active'].sum()),
            'attack_duration_hours': df[df['attack_active']]['sim_hours'].count() * dt_hours if df['attack_active'].any() else 0
        }
        
        # Attack-specific metrics
        if df['attack_active'].any():
            attack_df = df[df['attack_active']]
            normal_df = df[~df['attack_active']]
            
            if len(normal_df) > 0:
                metrics['power_A_attack_avg_kW'] = attack_df['P_A_kW'].mean()
                metrics['power_A_normal_avg_kW'] = normal_df['P_A_kW'].mean()
                metrics['power_A_increase_during_attack_pct'] = (
                    (metrics['power_A_attack_avg_kW'] / metrics['power_A_normal_avg_kW'] - 1) * 100
                    if metrics['power_A_normal_avg_kW'] > 0 else 0
                )
                
                metrics['comfort_violation_A_attack_degCh'] = attack_df['comfort_violation_A_degCh'].sum()
                metrics['comfort_violation_A_normal_degCh'] = normal_df['comfort_violation_A_degCh'].sum()
        
        return metrics
