"""
Scenario Comparator - Compare two simulation scenarios

Author: Guowen Li, AI Assistant
Date: 2025-01-20
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScenarioComparator:
    """Compare two simulation scenarios"""
    
    def __init__(self, scenario1_name: str = "Scenario 1", scenario2_name: str = "Scenario 2"):
        self.scenario1_name = scenario1_name
        self.scenario2_name = scenario2_name
        
        logger.info(f"📊 Scenario comparator initialized: '{scenario1_name}' vs '{scenario2_name}'")
    
    def compare(self, metrics1: Dict, metrics2: Dict) -> Dict:
        """
        Compare two scenarios
        
        Parameters:
        -----------
        metrics1 : Dict
            Summary metrics from scenario 1
        metrics2 : Dict
            Summary metrics from scenario 2
        
        Returns:
        --------
        Dict with comparison metrics
        """
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🔍 Comparing: '{metrics1['scenario_name']}' vs '{metrics2['scenario_name']}'")
        logger.info(f"{'='*80}\n")
        
        comparison = {
            'scenario1_name': metrics1['scenario_name'],
            'scenario2_name': metrics2['scenario_name'],
        }
        
        # Energy comparison
        energy_diff_kWh = metrics2['energy_total_kWh'] - metrics1['energy_total_kWh']
        energy_diff_pct = (energy_diff_kWh / metrics1['energy_total_kWh'] * 100) if metrics1['energy_total_kWh'] > 0 else 0
        
        comparison['energy_diff_kWh'] = energy_diff_kWh
        comparison['energy_diff_pct'] = energy_diff_pct
        comparison['energy_savings_kWh'] = -energy_diff_kWh  # Negative diff = savings
        comparison['energy_savings_pct'] = -energy_diff_pct
        
        # Power comparison
        comparison['power_avg_diff_kW'] = metrics2['power_total_avg_kW'] - metrics1['power_total_avg_kW']
        comparison['power_peak_diff_kW'] = metrics2['power_total_peak_kW'] - metrics1['power_total_peak_kW']
        
        # Comfort comparison
        comfort_diff = metrics2['comfort_violation_total_degCh'] - metrics1['comfort_violation_total_degCh']
        comfort_diff_pct = (comfort_diff / metrics1['comfort_violation_total_degCh'] * 100) if metrics1['comfort_violation_total_degCh'] > 0 else 0
        
        comparison['comfort_diff_degCh'] = comfort_diff
        comparison['comfort_diff_pct'] = comfort_diff_pct
        comparison['comfort_improvement_degCh'] = -comfort_diff  # Negative diff = improvement
        comparison['comfort_improvement_pct'] = -comfort_diff_pct
        
        # Feeder comparison
        comparison['feeder_util_avg_diff_pct'] = metrics2['feeder_utilization_avg_pct'] - metrics1['feeder_utilization_avg_pct']
        comparison['feeder_util_peak_diff_pct'] = metrics2['feeder_utilization_peak_pct'] - metrics1['feeder_utilization_peak_pct']
        comparison['feeder_violations_diff'] = metrics2['feeder_violations_count'] - metrics1['feeder_violations_count']
        comparison['feeder_stability_diff_kW'] = metrics2['feeder_stability_std_kW'] - metrics1['feeder_stability_std_kW']
        comparison['feeder_stability_improvement_pct'] = (
            (metrics1['feeder_stability_std_kW'] - metrics2['feeder_stability_std_kW']) / 
            metrics1['feeder_stability_std_kW'] * 100
            if metrics1['feeder_stability_std_kW'] > 0 else 0
        )
        
        # Cost estimate (simplified)
        # Assume average price of $0.15/kWh
        avg_price = 0.15
        cost1 = metrics1['energy_total_kWh'] * avg_price
        cost2 = metrics2['energy_total_kWh'] * avg_price
        comparison['cost_savings_usd'] = cost1 - cost2
        
        # TES utilization comparison (Building B)
        comparison['SOC_avg_diff'] = metrics2['SOC_avg'] - metrics1['SOC_avg']
        
        # Attack resilience comparison
        if 'power_A_increase_during_attack_pct' in metrics1 and 'power_A_increase_during_attack_pct' in metrics2:
            comparison['attack_power_increase_diff_pct'] = (
                metrics2['power_A_increase_during_attack_pct'] - 
                metrics1['power_A_increase_during_attack_pct']
            )
        
        # Print comparison summary
        self._print_comparison_summary(comparison, metrics1, metrics2)
        
        return comparison
    
    def _print_comparison_summary(self, comparison: Dict, metrics1: Dict, metrics2: Dict):
        """Print comparison summary to logger"""
        
        logger.info("=" * 80)
        logger.info("COMPARISON SUMMARY")
        logger.info("=" * 80)
        
        logger.info(f"\n📊 ENERGY CONSUMPTION:")
        logger.info(f"  Scenario 1: {metrics1['energy_total_kWh']:.2f} kWh")
        logger.info(f"  Scenario 2: {metrics2['energy_total_kWh']:.2f} kWh")
        logger.info(f"  Difference: {comparison['energy_diff_kWh']:+.2f} kWh ({comparison['energy_diff_pct']:+.1f}%)")
        if comparison['energy_savings_kWh'] > 0:
            logger.info(f"  ✅ Scenario 2 saves {comparison['energy_savings_kWh']:.2f} kWh ({comparison['energy_savings_pct']:.1f}%)")
        else:
            logger.info(f"  ⚠️  Scenario 2 uses {-comparison['energy_savings_kWh']:.2f} kWh more ({-comparison['energy_savings_pct']:.1f}%)")
        
        logger.info(f"\n🌡️  THERMAL COMFORT:")
        logger.info(f"  Scenario 1: {metrics1['comfort_violation_total_degCh']:.2f} °C·h")
        logger.info(f"  Scenario 2: {metrics2['comfort_violation_total_degCh']:.2f} °C·h")
        logger.info(f"  Difference: {comparison['comfort_diff_degCh']:+.2f} °C·h ({comparison['comfort_diff_pct']:+.1f}%)")
        if comparison['comfort_improvement_degCh'] > 0:
            logger.info(f"  ✅ Scenario 2 improves comfort by {comparison['comfort_improvement_degCh']:.2f} °C·h ({comparison['comfort_improvement_pct']:.1f}%)")
        else:
            logger.info(f"  ⚠️  Scenario 2 degrades comfort by {-comparison['comfort_improvement_degCh']:.2f} °C·h ({-comparison['comfort_improvement_pct']:.1f}%)")
        
        logger.info(f"\n⚡ FEEDER UTILIZATION:")
        logger.info(f"  Average utilization:")
        logger.info(f"    Scenario 1: {metrics1['feeder_utilization_avg_pct']:.1f}%")
        logger.info(f"    Scenario 2: {metrics2['feeder_utilization_avg_pct']:.1f}%")
        logger.info(f"    Difference: {comparison['feeder_util_avg_diff_pct']:+.1f}%")
        
        logger.info(f"  Peak utilization:")
        logger.info(f"    Scenario 1: {metrics1['feeder_utilization_peak_pct']:.1f}%")
        logger.info(f"    Scenario 2: {metrics2['feeder_utilization_peak_pct']:.1f}%")
        logger.info(f"    Difference: {comparison['feeder_util_peak_diff_pct']:+.1f}%")
        
        logger.info(f"  Violations:")
        logger.info(f"    Scenario 1: {metrics1['feeder_violations_count']} timesteps ({metrics1['feeder_violations_pct']:.1f}%)")
        logger.info(f"    Scenario 2: {metrics2['feeder_violations_count']} timesteps ({metrics2['feeder_violations_pct']:.1f}%)")
        logger.info(f"    Difference: {comparison['feeder_violations_diff']:+d} timesteps")
        
        logger.info(f"  Stability (std dev):")
        logger.info(f"    Scenario 1: {metrics1['feeder_stability_std_kW']:.2f} kW")
        logger.info(f"    Scenario 2: {metrics2['feeder_stability_std_kW']:.2f} kW")
        logger.info(f"    Difference: {comparison['feeder_stability_diff_kW']:+.2f} kW")
        if comparison['feeder_stability_improvement_pct'] > 0:
            logger.info(f"    ✅ Scenario 2 improves stability by {comparison['feeder_stability_improvement_pct']:.1f}%")
        
        logger.info(f"\n💰 COST ESTIMATE:")
        logger.info(f"  Scenario 1: ${metrics1['energy_total_kWh'] * 0.15:.2f}")
        logger.info(f"  Scenario 2: ${metrics2['energy_total_kWh'] * 0.15:.2f}")
        logger.info(f"  Savings: ${comparison['cost_savings_usd']:+.2f}")
        
        logger.info("\n" + "=" * 80)
    
    def save_comparison_report(self, comparison: Dict, filepath: str):
        """Save comparison report to file"""
        
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SCENARIO COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Scenario 1: {comparison['scenario1_name']}\n")
            f.write(f"Scenario 2: {comparison['scenario2_name']}\n\n")
            
            f.write("ENERGY CONSUMPTION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Energy difference: {comparison['energy_diff_kWh']:+.2f} kWh ({comparison['energy_diff_pct']:+.1f}%)\n")
            f.write(f"Energy savings: {comparison['energy_savings_kWh']:.2f} kWh\n\n")
            
            f.write("THERMAL COMFORT\n")
            f.write("-" * 80 + "\n")
            f.write(f"Comfort difference: {comparison['comfort_diff_degCh']:+.2f} °C·h ({comparison['comfort_diff_pct']:+.1f}%)\n")
            f.write(f"Comfort improvement: {comparison['comfort_improvement_degCh']:.2f} °C·h\n\n")
            
            f.write("FEEDER METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Average utilization difference: {comparison['feeder_util_avg_diff_pct']:+.1f}%\n")
            f.write(f"Peak utilization difference: {comparison['feeder_util_peak_diff_pct']:+.1f}%\n")
            f.write(f"Violations difference: {comparison['feeder_violations_diff']:+d} timesteps\n")
            f.write(f"Stability improvement: {comparison['feeder_stability_improvement_pct']:+.1f}%\n\n")
            
            f.write("COST\n")
            f.write("-" * 80 + "\n")
            f.write(f"Cost savings: ${comparison['cost_savings_usd']:+.2f}\n\n")
            
            f.write("=" * 80 + "\n")
        
        logger.info(f"✅ Comparison report saved to {filepath}")
