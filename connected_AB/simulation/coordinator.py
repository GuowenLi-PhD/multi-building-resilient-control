"""
Main Coordinator - Orchestrates hierarchical multi-building control

Author: Guowen Li, AI Assistant
Date: 2025-01-07
"""

import yaml
import numpy as np
import pandas as pd
from typing import Dict, List
import logging
from datetime import datetime
import os

import sys
# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    print(f"🔧 Current directory added to path: {current_dir}")
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    print(f"🔧 Parent directory added to path: {parent_dir}")

from buildings.building_a_interface import BuildingAInterface
from buildings.building_b_interface import BuildingBInterface
from aggregator.aggregator_mpc import AggregatorMPC, AggregatorMPCConfig
from aggregator.attack_anticipator import AttackAnticipator, AttackPrediction
from communication.message_protocol import MessageBroker
from communication.data_models import (
    AggregatorCommand, AggregatorCommandBuildingB, FeederStatus,
    BuildingStatus, ControlMode
)
from simulation.metrics_collector import MetricsCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HierarchicalCoordinator:
    """Main coordinator for multi-building resilient control"""
    
    def __init__(self, config_path: str = 'config/system_config.yaml'):
        """Initialize coordinator"""
        
        logger.info("="*80)
        logger.info("🚀 HIERARCHICAL MULTI-BUILDING RESILIENT CONTROL FRAMEWORK")
        logger.info("="*80)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.building_a = BuildingAInterface(self.config)
        self.building_b = BuildingBInterface(self.config)
        
        # Aggregator MPC
        agg_config = AggregatorMPCConfig(
            PH=self.config['timing']['prediction_horizon_aggregator'],
            dt=self.config['timing']['aggregator_timestep'],
            P_feeder_limit_kW=self.config['feeder']['capacity_kW'],
            safety_margin=self.config['feeder']['safety_margin'],
            w_feeder=self.config['weights']['aggregator']['feeder_tracking'],
            w_comfort=self.config['weights']['aggregator']['comfort_priority'],
            w_balance=self.config['weights']['aggregator']['power_balance'],
            w_TES=self.config['weights']['aggregator']['TES_utilization'],
            P_A_baseline_kW=self.config['building_a']['baseline_power_kW'],
            P_B_baseline_kW=self.config['building_b']['baseline_power_kW']
        )
        self.aggregator = AggregatorMPC(agg_config)
        
        # Attack anticipator
        self.attack_anticipator = AttackAnticipator(
            method=self.config['detection']['method'],
            config_path='config/attack_scenarios.yaml'
        )
        
        # Communication
        self.message_broker = MessageBroker()
        
        # Metrics collector
        self.metrics_collector = MetricsCollector()
        
        # Simulation parameters
        self.current_time = 0
        self.simulation_start_time = 0
        self.dt_aggregator = self.config['timing']['aggregator_timestep']
        
        logger.info("✓ Coordinator initialized successfully")
    
    def run_simulation(self, 
                       start_time: float,
                       end_time: float,
                       weather_data: Dict,
                       price_data: List[float]):
        """
        Run hierarchical control simulation
        
        Parameters:
        -----------
        start_time : float
            Simulation start time (seconds from epoch)
        end_time : float
            Simulation end time (seconds from epoch)
        weather_data : Dict
            Weather forecasts (Toa, RHoa, GHI)
        price_data : List[float]
            Electricity price forecast
        """
        
        logger.info(f"📅 Simulation period: {(end_time - start_time)/86400:.1f} days")
        logger.info(f"⏱️  Aggregator timestep: {self.dt_aggregator/60:.0f} min")
        
        # Initialize buildings
        initial_conditions = {
            'simulation_start_time': start_time,
            'Toa_history': weather_data['Toa'][:4],
            'GHI_history': weather_data.get('GHI', [0]*4)[:4],
            'Toa_forecast': weather_data['Toa'][:20],
            'RHoa_forecast': weather_data['RHoa'][:20],
            'GHI_forecast': weather_data.get('GHI', [0]*20)[:20],
            'price_forecast': price_data[:20],
            'SOC_ini': 0.5
        }
        
        self.building_a.initialize(initial_conditions)
        self.building_b.initialize(initial_conditions)
        
        self.current_time = start_time
        self.simulation_start_time = start_time
        
        # Main simulation loop
        step_count = 0
        while self.current_time < end_time:
            
            logger.info("")
            logger.info("="*80)
            logger.info(f"⏰ Simulation Time: {(self.current_time - start_time)/3600:.2f} hours")
            logger.info("="*80)
            
            # 1. Attack anticipation and detection
            attack_prediction = self.attack_anticipator.predict_attack(
                current_time=self.current_time,
                simulation_start=self.simulation_start_time,
                anticipation_hours=self.config['detection']['anticipation_hours']
            )
            
            attack_active, attack_info = self.attack_anticipator.is_attack_active(
                current_time=self.current_time,
                simulation_start=self.simulation_start_time
            )
            
            if attack_prediction.anticipated:
                logger.warning(f"🔮 Attack anticipated in {attack_prediction.time_to_attack_hours:.1f}h: {attack_prediction.target_building}")
            
            if attack_active:
                logger.error(f"⚠️⚠️⚠️ CYBER-ATTACK ACTIVE: {attack_info['name']} ⚠️⚠️⚠️")
            
            # 2. Get current building states (from previous step or initialization)
            state_a = self.building_a.current_state
            state_b = self.building_b.current_state
            
            # 3. Run Aggregator MPC
            if state_a and state_b:
                # Get forecasts from buildings
                P_A_forecast = self.building_a.get_power_forecast(self.aggregator.PH)
                P_B_forecast = self.building_b.get_power_forecast(self.aggregator.PH)
                SOC_B_forecast = self.building_b.get_SOC_forecast(self.aggregator.PH)
                
                # Solve aggregator optimization
                agg_result = self.aggregator.optimize(
                    current_time=self.current_time,
                    P_A_forecast=P_A_forecast,
                    P_B_forecast=P_B_forecast,
                    SOC_B_forecast=SOC_B_forecast,
                    attack_flag=attack_active,
                    attack_anticipated=attack_prediction.anticipated
                )
                
                logger.info(f"🎯 Aggregator allocation: P_A={agg_result['P_A_ref'][0]:.2f}kW, P_B={agg_result['P_B_ref'][0]:.2f}kW")
                logger.info(f"   Feeder utilization: {agg_result['feeder_utilization']*100:.1f}%")
            
            else:
                # First step: use default allocation
                agg_result = {
                    'P_A_ref': [self.config['building_a']['baseline_power_kW']] * self.aggregator.PH,
                    'P_B_ref': [self.config['building_b']['baseline_power_kW']] * self.aggregator.PH,
                    'SOC_B_target': 0.5,
                    'priority_A': 'balanced',
                    'priority_B': 'balanced',
                    'feeder_utilization': 0.4
                }
            
            # 4. Create aggregator commands
            cmd_a = AggregatorCommand(
                timestamp=self.current_time,
                building_id='Building_A',
                power_reference_kW=agg_result['P_A_ref'],
                power_limit_kW=self.config['feeder']['capacity_kW'] * 0.6,
                attack_flag=attack_active,
                attack_anticipated=attack_prediction.anticipated,
                anticipation_horizon_hours=attack_prediction.time_to_attack_hours if attack_prediction.anticipated else 0.0
            )
            
            cmd_b = AggregatorCommandBuildingB(
                timestamp=self.current_time,
                building_id='Building_B',
                power_reference_kW=agg_result['P_B_ref'],
                power_limit_kW=self.config['feeder']['capacity_kW'] * 0.6,
                attack_flag=attack_active,
                attack_anticipated=attack_prediction.anticipated,
                anticipation_horizon_hours=attack_prediction.time_to_attack_hours if attack_prediction.anticipated else 0.0
            )
            cmd_b.guidance['SOC_target'] = agg_result['SOC_B_target']
            cmd_b.guidance['precharge_recommended'] = attack_prediction.anticipated
            cmd_b.guidance['discharge_requested'] = attack_active
            cmd_b.guidance['priority'] = agg_result['priority_B']
            
            # Send commands via message broker
            self.message_broker.send_aggregator_command(cmd_a)
            self.message_broker.send_aggregator_command(cmd_b)
            
            # 5. Execute building control steps
            logger.info("🏢 Building A control step...")
            state_a = self.building_a.step(cmd_a, self.dt_aggregator)
            self.message_broker.send_building_state(state_a)
            
            logger.info("🏢 Building B control step...")
            state_b = self.building_b.step(cmd_b, self.dt_aggregator)
            self.message_broker.send_building_state(state_b)
            
            # 6. Update feeder status
            total_power = state_a.power_actual_kW + state_b.power_actual_kW
            feeder_status = FeederStatus(
                timestamp=self.current_time,
                total_power_kW=total_power,
                capacity_kW=self.config['feeder']['capacity_kW'],
                utilization_percent=total_power / self.config['feeder']['capacity_kW'] * 100,
                voltage_pu=1.0,  # Simplified
                constraint_violated=total_power > self.config['feeder']['capacity_kW'],
                margin_kW=self.config['feeder']['capacity_kW'] - total_power
            )
            self.message_broker.update_feeder_status(feeder_status)
            
            # 7. Collect metrics
            self.metrics_collector.record_step(
                timestamp=self.current_time,
                building_a_state=state_a,
                building_b_state=state_b,
                feeder_status=feeder_status,
                aggregator_result=agg_result,
                attack_active=attack_active
            )
            
            # Log step summary
            logger.info(f"📊 Step {step_count} Summary:")
            logger.info(f"   Building A: P={state_a.power_actual_kW:.2f}kW, Status={state_a.status.value}, Mode={state_a.control_mode.value}")
            logger.info(f"   Building B: P={state_b.power_actual_kW:.2f}kW, SOC={state_b.extra_data['SOC_current']:.2f}, TES_mode={state_b.extra_data['TES_mode']}")
            logger.info(f"   Feeder: Total={total_power:.2f}kW ({feeder_status.utilization_percent:.1f}%), Margin={feeder_status.margin_kW:.2f}kW")
            
            if feeder_status.constraint_violated:
                logger.error("🚨 FEEDER CAPACITY VIOLATED! 🚨")
            
            # 8. Advance time
            self.current_time += self.dt_aggregator
            step_count += 1
        
        logger.info("")
        logger.info("="*80)
        logger.info("✅ SIMULATION COMPLETE")
        logger.info("="*80)
        
        # Save results
        self._save_results()
    
    def _save_results(self):
        """Save simulation results"""
        
        # Create results directory
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save metrics
        metrics_file = os.path.join(results_dir, f'metrics_{timestamp}.csv')
        self.metrics_collector.save_to_csv(metrics_file)
        logger.info(f"💾 Metrics saved to {metrics_file}")
        
        # Save message history
        messages_file = os.path.join(results_dir, f'messages_{timestamp}.json')
        self.message_broker.save_messages(messages_file)
        logger.info(f"💾 Messages saved to {messages_file}")
        
        # Generate summary report
        summary_file = os.path.join(results_dir, f'summary_{timestamp}.txt')
        self.metrics_collector.generate_summary_report(summary_file)
        logger.info(f"📄 Summary report saved to {summary_file}")