"""
Main Execution Script for Schedule-Based Multi-Building Control

This script runs two scenarios and compares their performance:
- Scenario 1: Stand-alone MPC (no coordination)
- Scenario 2: User-coordinated control with schedules

Author: Guowen Li, AI Assistant
Date: 2025-01-20
"""

import sys
import os
import argparse
import logging
from datetime import datetime

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from schedule.schedule_parser import ScheduleParser
from simulation.scenario_runner import ScenarioRunner
from analysis.scenario_comparator import ScenarioComparator
from analysis.visualizer import Visualizer
from utils.data_loader import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(
        description='Schedule-Based Multi-Building Control Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python run_schedule_simulation.py
  
  # Custom scenarios and duration
  python run_schedule_simulation.py --scenario1 config/my_scenario1.yaml --duration 3
  
  # Custom start day
  python run_schedule_simulation.py --start-day 180 --duration 2
        """
    )
    
    parser.add_argument('--config', type=str, 
                       default='config/system_config.yaml',
                       help='Path to system configuration file')
    
    parser.add_argument('--scenario1', type=str,
                       default='config/schedule_scenario1.yaml',
                       help='Path to Scenario 1 schedule file')
    
    parser.add_argument('--scenario2', type=str,
                       default='config/schedule_scenario2.yaml',
                       help='Path to Scenario 2 schedule file')
    
    parser.add_argument('--attacks', type=str,
                       default='config/attack_scenarios.yaml',
                       help='Path to attack scenarios file')
    
    parser.add_argument('--weather', type=str,
                       default='../buildingA_wo_TES/weather_data/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw',
                       help='Path to weather file (EPW format)')
    
    parser.add_argument('--start-day', type=int, default=212,
                       help='Simulation start day (day of year, 1-365)')
    
    parser.add_argument('--duration', type=int, default=2,
                       help='Simulation duration (days)')
    
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*80)
    print("🚀 SCHEDULE-BASED MULTI-BUILDING CONTROL SIMULATION")
    print("="*80)
    print(f"Configuration: {args.config}")
    print(f"Scenario 1: {args.scenario1}")
    print(f"Scenario 2: {args.scenario2}")
    print(f"Attacks: {args.attacks}")
    print(f"Weather file: {args.weather}")
    print(f"Start day: {args.start_day}")
    print(f"Duration: {args.duration} days")
    print(f"Output directory: {args.output}")
    print("="*80 + "\n")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load system configuration
    try:
        system_config = ScheduleParser.parse_system_config(args.config)
    except FileNotFoundError:
        logger.error(f"❌ Configuration file not found: {args.config}")
        return 1
    
    # Load weather data
    logger.info("\n" + "="*80)
    logger.info("📡 LOADING DATA")
    logger.info("="*80)
    
    try:
        dt_min = min(
            system_config['building_a']['control_interval_minutes'],
            system_config['building_b']['control_interval_minutes']
        ) * 60
        
        weather_data = DataLoader.load_weather_data(args.weather, dt=dt_min)
        price_data = DataLoader.load_price_data(dt=dt_min, n_days=args.duration + 1)
    
    except FileNotFoundError:
        logger.error(f"❌ Weather file not found: {args.weather}")
        return 1
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ========================================================================
    # RUN SCENARIO 1
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("🎬 SCENARIO 1: Stand-alone MPC")
    logger.info("="*80)
    
    try:
        sim_config_1 = ScheduleParser.create_simulation_config(
            system_config_path=args.config,
            schedule_path=args.scenario1,
            attack_path=args.attacks,
            start_day=args.start_day,
            duration_days=args.duration
        )
        
        runner_1 = ScenarioRunner(sim_config_1, system_config)
        results_1 = runner_1.run(weather_data, price_data)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_1['metrics_collector'].save_to_csv(
            f"{args.output}/scenario1_metrics_{timestamp}.csv"
        )
        
        logger.info(f"✅ Scenario 1 complete")
    
    except Exception as e:
        logger.error(f"❌ Scenario 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ========================================================================
    # RUN SCENARIO 2
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("🎬 SCENARIO 2: User Coordinated Control")
    logger.info("="*80)
    
    try:
        sim_config_2 = ScheduleParser.create_simulation_config(
            system_config_path=args.config,
            schedule_path=args.scenario2,
            attack_path=args.attacks,
            start_day=args.start_day,
            duration_days=args.duration
        )
        
        runner_2 = ScenarioRunner(sim_config_2, system_config)
        results_2 = runner_2.run(weather_data, price_data)
        
        # Save results
        results_2['metrics_collector'].save_to_csv(
            f"{args.output}/scenario2_metrics_{timestamp}.csv"
        )
        
        logger.info(f"✅ Scenario 2 complete")
    
    except Exception as e:
        logger.error(f"❌ Scenario 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ========================================================================
    # COMPARISON AND VISUALIZATION
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("📊 COMPARISON AND VISUALIZATION")
    logger.info("="*80)
    
    try:
        # Compare scenarios
        comparator = ScenarioComparator(
            scenario1_name=results_1['summary_metrics']['scenario_name'],
            scenario2_name=results_2['summary_metrics']['scenario_name']
        )
        
        comparison = comparator.compare(
            results_1['summary_metrics'],
            results_2['summary_metrics']
        )
        
        # Save comparison report
        comparator.save_comparison_report(
            comparison,
            f"{args.output}/comparison_report_{timestamp}.txt"
        )
        
        # Generate visualizations
        visualizer = Visualizer()
        visualizer.plot_all(
            df1=results_1['dataframe'],
            df2=results_2['dataframe'],
            metrics1=results_1['summary_metrics'],
            metrics2=results_2['summary_metrics'],
            comparison=comparison,
            output_dir=args.output
        )
        
        logger.info(f"✅ Comparison and visualization complete")
    
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("✅ SIMULATION COMPLETE!")
    logger.info("="*80)
    logger.info(f"\n📁 Results saved to: {args.output}/")
    logger.info(f"   • Scenario 1 metrics: scenario1_metrics_{timestamp}.csv")
    logger.info(f"   • Scenario 2 metrics: scenario2_metrics_{timestamp}.csv")
    logger.info(f"   • Comparison report: comparison_report_{timestamp}.txt")
    logger.info(f"   • Visualization plots: *.png")
    
    logger.info(f"\n📊 Key Findings:")
    logger.info(f"   • Energy savings: {comparison['energy_savings_kWh']:.2f} kWh ({comparison['energy_savings_pct']:.1f}%)")
    logger.info(f"   • Comfort improvement: {comparison['comfort_improvement_degCh']:.2f} °C·h ({comparison['comfort_improvement_pct']:.1f}%)")
    logger.info(f"   • Cost savings: ${comparison['cost_savings_usd']:.2f}")
    logger.info(f"   • Feeder stability improvement: {comparison['feeder_stability_improvement_pct']:.1f}%")
    
    if comparison['feeder_violations_diff'] < 0:
        logger.info(f"   • Feeder violations reduced by {-comparison['feeder_violations_diff']} timesteps ✅")
    elif comparison['feeder_violations_diff'] > 0:
        logger.warning(f"   • Feeder violations increased by {comparison['feeder_violations_diff']} timesteps ⚠️")
    else:
        logger.info(f"   • No change in feeder violations")
    
    logger.info("\n" + "="*80 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
