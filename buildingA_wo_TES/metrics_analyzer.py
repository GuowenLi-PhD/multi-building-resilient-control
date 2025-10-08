"""
Enhanced Metrics Analyzer for Building A MPC Performance Evaluation

Author: Guowen Li
Email: guowenli@tamu.edu
Date: 2025-10-07
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime

class MPCPerformanceAnalyzer:
    """Comprehensive performance analysis for Nominal vs Adaptive MPC"""
    
    def __init__(self, results_nominal, results_attack, price_tou, dt=900):
        """
        Parameters:
        -----------
        results_nominal : pandas.DataFrame
            Simulation results under nominal MPC (no attack)
        results_attack : pandas.DataFrame
            Simulation results under AMPC (with DoS attack)
        price_tou : list
            24-hour TOU pricing schedule [$/kWh]
        dt : float
            Timestep in seconds (default: 900s = 15min)
        """
        self.results_nominal = results_nominal
        self.results_attack = results_attack
        self.price_tou = price_tou
        self.dt = dt
        self.nsteps_per_hour = int(3600 / dt)
        
        # Temperature bounds
        self.T_upper_occ = 26.0  # °C during occupied
        self.T_lower_occ = 22.0  # °C during occupied
        self.T_upper_unocc = 30.0  # °C during unoccupied
        self.T_lower_unocc = 18.0  # °C during unoccupied
        self.occ_start = 7  # 7 AM
        self.occ_end = 19   # 7 PM
        
    def get_total_power(self, results):
        """Calculate total power consumption [W]"""
        power_columns = ['mod.eleChi.y', 'mod.eleCHWP.y', 'mod.eleCT.y', 
                        'mod.eleCWP.y', 'mod.eleSupFan.y']
        return results[power_columns].sum(axis=1)
    
    def get_zone_temps(self, results):
        """Extract all zone temperatures [K]"""
        zone_columns = ['mod.flo.temAirPer5.T',  # Core
                       'mod.flo.temAirEas.T',   # East
                       'mod.flo.temAirNor.T',   # North
                       'mod.flo.temAirSou.T',   # South
                       'mod.flo.temAirWes.T']   # West
        return results[zone_columns]
    
    def calculate_energy_metrics(self):
        """Calculate comprehensive energy metrics"""
        
        # Total power [W]
        P_nominal = self.get_total_power(self.results_nominal)
        P_attack = self.get_total_power(self.results_attack)
        
        # Total energy [kWh]
        E_nominal = P_nominal.sum() / self.nsteps_per_hour / 1000
        E_attack = P_attack.sum() / self.nsteps_per_hour / 1000
        
        # Energy cost [$]
        cost_nominal = 0
        cost_attack = 0
        
        for i in range(len(P_nominal)):
            hour_idx = (i % (self.nsteps_per_hour * 24)) // self.nsteps_per_hour
            cost_nominal += P_nominal.iloc[i] / 1000 / self.nsteps_per_hour * self.price_tou[hour_idx]
            cost_attack += P_attack.iloc[i] / 1000 / self.nsteps_per_hour * self.price_tou[hour_idx]
        
        # Peak demand [kW]
        peak_nominal = P_nominal.max() / 1000
        peak_attack = P_attack.max() / 1000
        
        metrics = {
            'energy_consumption': {
                'nominal_kWh': float(E_nominal),
                'attack_kWh': float(E_attack),
                'delta_kWh': float(E_attack - E_nominal),
                'delta_percent': float((E_attack - E_nominal) / E_nominal * 100)
            },
            'energy_cost': {
                'nominal_$': float(cost_nominal),
                'attack_$': float(cost_attack),
                'delta_$': float(cost_attack - cost_nominal),
                'delta_percent': float((cost_attack - cost_nominal) / cost_nominal * 100)
            },
            'peak_demand': {
                'nominal_kW': float(peak_nominal),
                'attack_kW': float(peak_attack),
                'delta_kW': float(peak_attack - peak_nominal),
                'delta_percent': float((peak_attack - peak_nominal) / peak_nominal * 100)
            }
        }
        
        return metrics
    
    def calculate_thermal_comfort_metrics(self):
        """Calculate thermal comfort violations"""
        
        # Get zone temperatures [K] and convert to [°C]
        T_zones_nominal = self.get_zone_temps(self.results_nominal) - 273.15
        T_zones_attack = self.get_zone_temps(self.results_attack) - 273.15
        
        # Calculate violations for each timestep
        violations_nominal = []
        violations_attack = []
        
        for i in range(len(T_zones_nominal)):
            hour_idx = (i % (self.nsteps_per_hour * 24)) // self.nsteps_per_hour
            
            # Determine temperature bounds based on occupancy
            if self.occ_start <= hour_idx < self.occ_end:
                T_upper = self.T_upper_occ
                T_lower = self.T_lower_occ
            else:
                T_upper = self.T_upper_unocc
                T_lower = self.T_lower_unocc
            
            # Calculate violations for each zone
            violation_nom = 0
            violation_att = 0
            
            for col in T_zones_nominal.columns:
                # Nominal
                T_nom = T_zones_nominal[col].iloc[i]
                violation_nom += max(0, T_nom - T_upper) + max(0, T_lower - T_nom)
                
                # Attack
                T_att = T_zones_attack[col].iloc[i]
                violation_att += max(0, T_att - T_upper) + max(0, T_lower - T_att)
            
            violations_nominal.append(violation_nom)
            violations_attack.append(violation_att)
        
        violations_nominal = np.array(violations_nominal)
        violations_attack = np.array(violations_attack)
        
        # Unmet degree-hours [°C·h]
        udh_nominal = violations_nominal.sum() / self.nsteps_per_hour
        udh_attack = violations_attack.sum() / self.nsteps_per_hour
        
        # Maximum instantaneous violation [°C]
        max_violation_nominal = violations_nominal.max()
        max_violation_attack = violations_attack.max()
        
        # Percentage of time with violations
        violation_time_nominal = (violations_nominal > 0.1).sum() / len(violations_nominal) * 100
        violation_time_attack = (violations_attack > 0.1).sum() / len(violations_attack) * 100
        
        metrics = {
            'unmet_degree_hours': {
                'nominal_degC_h': float(udh_nominal),
                'attack_degC_h': float(udh_attack),
                'delta_degC_h': float(udh_attack - udh_nominal),
                'reduction_percent': float((udh_nominal - udh_attack) / udh_nominal * 100) if udh_nominal > 0 else 0
            },
            'max_violation': {
                'nominal_degC': float(max_violation_nominal),
                'attack_degC': float(max_violation_attack),
                'delta_degC': float(max_violation_attack - max_violation_nominal)
            },
            'violation_time_percent': {
                'nominal': float(violation_time_nominal),
                'attack': float(violation_time_attack),
                'delta': float(violation_time_attack - violation_time_nominal)
            }
        }
        
        return metrics
    
    def calculate_control_performance_metrics(self):
        """Calculate control action metrics (smoothness, effort)"""
        
        # Control setpoints
        controls_nominal = {
            'T_chw': self.results_nominal['oveTChiWatSupSet_u'] - 273.15,
            'T_cw': self.results_nominal['oveTConWatSupSet_u'] - 273.15,
            'T_sa': self.results_nominal['conAHU_oveTSupAir_u'] - 273.15,
            'V_core': self.results_nominal['conVAVCor_damVal_oveVDisSet_u'],
            'V_east': self.results_nominal['conVAVEas_damVal_oveVDisSet_u']
        }
        
        controls_attack = {
            'T_chw': self.results_attack['oveTChiWatSupSet_u'] - 273.15,
            'T_cw': self.results_attack['oveTConWatSupSet_u'] - 273.15,
            'T_sa': self.results_attack['conAHU_oveTSupAir_u'] - 273.15,
            'V_core': self.results_attack['conVAVCor_damVal_oveVDisSet_u'],
            'V_east': self.results_attack['conVAVEas_damVal_oveVDisSet_u']
        }
        
        # Calculate total variation (TV) - measure of control smoothness
        def calculate_tv(signal):
            return np.abs(np.diff(signal)).sum()
        
        tv_nominal = {key: calculate_tv(val.values) for key, val in controls_nominal.items()}
        tv_attack = {key: calculate_tv(val.values) for key, val in controls_attack.items()}
        
        metrics = {
            'control_smoothness_TV': {
                'nominal': tv_nominal,
                'attack': tv_attack,
                'delta_percent': {key: (tv_attack[key] - tv_nominal[key]) / tv_nominal[key] * 100 
                                 for key in tv_nominal.keys()}
            }
        }
        
        return metrics
    
    def calculate_resilience_score(self):
        """Calculate overall resilience score"""
        
        energy_metrics = self.calculate_energy_metrics()
        comfort_metrics = self.calculate_thermal_comfort_metrics()
        
        # Resilience score components
        # 1. Comfort maintenance (0-100): Higher is better
        comfort_score = max(0, 100 * (1 - comfort_metrics['unmet_degree_hours']['attack_degC_h'] / 
                                      (comfort_metrics['unmet_degree_hours']['nominal_degC_h'] + 1e-6)))
        
        # 2. Energy penalty (0-100): Lower penalty is better
        energy_penalty = energy_metrics['energy_consumption']['delta_percent']
        energy_score = max(0, 100 - energy_penalty)
        
        # 3. Operational continuity (0-100): System remains operational
        continuity_score = 100  # Assuming system continues to operate
        
        # Overall resilience score (weighted average)
        w_comfort = 0.5
        w_energy = 0.3
        w_continuity = 0.2
        
        resilience_score = (w_comfort * comfort_score + 
                           w_energy * energy_score + 
                           w_continuity * continuity_score)
        
        metrics = {
            'resilience_score': float(resilience_score),
            'components': {
                'comfort_maintenance': float(comfort_score),
                'energy_efficiency': float(energy_score),
                'operational_continuity': float(continuity_score)
            },
            'weights': {
                'comfort': w_comfort,
                'energy': w_energy,
                'continuity': w_continuity
            }
        }
        
        return metrics
    
    def generate_comprehensive_report(self, save_path='mpc_results/comprehensive_analysis.json'):
        """Generate comprehensive performance report"""
        
        report = {
            'metadata': {
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'timestep_seconds': self.dt,
                'simulation_duration_hours': len(self.results_nominal) / self.nsteps_per_hour,
                'attack_scenario': 'DoS Attack on Core Zone VAV Box'
            },
            'energy_metrics': self.calculate_energy_metrics(),
            'thermal_comfort_metrics': self.calculate_thermal_comfort_metrics(),
            'control_performance_metrics': self.calculate_control_performance_metrics(),
            'resilience_metrics': self.calculate_resilience_score()
        }
        
        # Save to JSON
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        self.print_summary(report)
        
        return report
    
    def print_summary(self, report):
        """Print formatted summary to console"""
        
        print("\n" + "="*100)
        print("BUILDING A - MPC PERFORMANCE ANALYSIS SUMMARY")
        print("="*100)
        print(f"Attack Scenario: {report['metadata']['attack_scenario']}")
        print(f"Simulation Duration: {report['metadata']['simulation_duration_hours']:.1f} hours")
        print("="*100)
        
        # Energy metrics
        print("\n📊 ENERGY METRICS")
        print("-" * 100)
        energy = report['energy_metrics']
        print(f"  Total Energy Consumption:")
        print(f"    Nominal MPC:  {energy['energy_consumption']['nominal_kWh']:.2f} kWh")
        print(f"    AMPC (Attack): {energy['energy_consumption']['attack_kWh']:.2f} kWh")
        print(f"    → Increase:    {energy['energy_consumption']['delta_kWh']:.2f} kWh ({energy['energy_consumption']['delta_percent']:.1f}%)")
        
        print(f"\n  Energy Cost:")
        print(f"    Nominal MPC:  ${energy['energy_cost']['nominal_$']:.2f}")
        print(f"    AMPC (Attack): ${energy['energy_cost']['attack_$']:.2f}")
        print(f"    → Increase:    ${energy['energy_cost']['delta_$']:.2f} ({energy['energy_cost']['delta_percent']:.1f}%)")
        
        print(f"\n  Peak Demand:")
        print(f"    Nominal MPC:  {energy['peak_demand']['nominal_kW']:.2f} kW")
        print(f"    AMPC (Attack): {energy['peak_demand']['attack_kW']:.2f} kW")
        print(f"    → Increase:    {energy['peak_demand']['delta_kW']:.2f} kW ({energy['peak_demand']['delta_percent']:.1f}%)")
        
        # Thermal comfort metrics
        print("\n🌡️  THERMAL COMFORT METRICS")
        print("-" * 100)
        comfort = report['thermal_comfort_metrics']
        print(f"  Unmet Degree-Hours:")
        print(f"    Nominal MPC:  {comfort['unmet_degree_hours']['nominal_degC_h']:.2f} °C·h")
        print(f"    AMPC (Attack): {comfort['unmet_degree_hours']['attack_degC_h']:.2f} °C·h")
        print(f"    → Reduction:   {comfort['unmet_degree_hours']['delta_degC_h']:.2f} °C·h ({comfort['unmet_degree_hours']['reduction_percent']:.1f}%)")
        
        print(f"\n  Maximum Violation:")
        print(f"    Nominal MPC:  {comfort['max_violation']['nominal_degC']:.2f} °C")
        print(f"    AMPC (Attack): {comfort['max_violation']['attack_degC']:.2f} °C")
        
        print(f"\n  Time with Violations:")
        print(f"    Nominal MPC:  {comfort['violation_time_percent']['nominal']:.1f}%")
        print(f"    AMPC (Attack): {comfort['violation_time_percent']['attack']:.1f}%")
        
        # Resilience score
        print("\n🛡️  RESILIENCE METRICS")
        print("-" * 100)
        resilience = report['resilience_metrics']
        print(f"  Overall Resilience Score: {resilience['resilience_score']:.1f}/100")
        print(f"    ├─ Comfort Maintenance:       {resilience['components']['comfort_maintenance']:.1f}/100")
        print(f"    ├─ Energy Efficiency:         {resilience['components']['energy_efficiency']:.1f}/100")
        print(f"    └─ Operational Continuity:    {resilience['components']['operational_continuity']:.1f}/100")
        
        print("\n" + "="*100)
        print("✓ Analysis Complete - Full report saved to 'mpc_results/comprehensive_analysis.json'")
        print("="*100 + "\n")