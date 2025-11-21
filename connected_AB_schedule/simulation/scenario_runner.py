"""
Scenario Runner - Execute single scenario simulation

Author: Guowen Li, AI Assistant
Date: 2025-01-20
"""

import sys
import os
import logging
from typing import Dict, List
import numpy as np

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

from buildings.building_a_scheduler import BuildingAScheduler
from buildings.building_b_scheduler import BuildingBScheduler
from simulation.metrics_collector import MetricsCollector
from schedule.control_models import SimulationConfig, AttackEvent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ScenarioRunner:
    """Run single scenario simulation"""
    
    def __init__(self, sim_config: SimulationConfig, system_config: Dict):
        """
        Initialize scenario runner
        
        Parameters:
        -----------
        sim_config : SimulationConfig
            Complete simulation configuration
        system_config : Dict
            System parameters
        """
        self.sim_config = sim_config
        self.system_config = system_config
        
        # Initialize buildings
        self.building_a = BuildingAScheduler(system_config, sim_config.building_a_schedule)
        self.building_b = BuildingBScheduler(system_config, sim_config.building_b_schedule)
        
        # Metrics collector
        self.metrics = MetricsCollector(sim_config.scenario_name)
        
        # Simulation state
        self.current_time = 0
        self.simulation_start_time = 0
        
        logger.info("="*80)
        logger.info(f"🚀 SCENARIO: {sim_config.scenario_name}")
        logger.info("="*80)
    
    def run(self, weather_data: Dict, price_data: List[float]) -> Dict:
        """
        Run scenario simulation
        
        Parameters:
        -----------
        weather_data : Dict
            Weather forecasts (Toa, RHoa, GHI)
        price_data : List[float]
            Electricity price forecast
        
        Returns:
        --------
        Dict with simulation results and metrics
        """
        
        # Calculate simulation times
        t_start = self.sim_config.start_day * 24 * 3600
        t_end = t_start + self.sim_config.duration_days * 24 * 3600
        
        logger.info(f"📅 Simulation period: Day {self.sim_config.start_day} for {self.sim_config.duration_days} days")
        logger.info(f"⏱️  Duration: {(t_end - t_start)/3600:.1f} hours")
        
        # Determine control timestep (minimum of both buildings)
        dt_a = self.building_a.get_control_interval()
        dt_b = self.building_b.get_control_interval()
        dt_sim = min(dt_a, dt_b)
        
        logger.info(f"⚙️  Control intervals: Building A={dt_a/60:.0f}min, Building B={dt_b/60:.0f}min")
        logger.info(f"⚙️  Simulation timestep: {dt_sim/60:.0f}min")
        
        # Initialize buildings
        initial_conditions = {
            'simulation_start_time': t_start,
            'Toa_history': weather_data['Toa'][:10],
            'GHI_history': weather_data.get('GHI', [0]*10)[:10],
            'Toa_forecast': weather_data['Toa'][:max(self.building_a.PH, self.building_b.PH)],
            'RHoa_forecast': weather_data['RHoa'][:max(self.building_a.PH, self.building_b.PH)],
            'GHI_forecast': weather_data.get('GHI', [0]*20)[:max(self.building_a.PH, self.building_b.PH)],
            'price_forecast': price_data[:max(self.building_a.PH, self.building_b.PH)],
            'SOC_ini': self.system_config['building_b']['tes']['SOC_initial']
        }
        
        self.building_a.initialize(initial_conditions)
        self.building_b.initialize(initial_conditions)
        
        self.current_time = t_start
        self.simulation_start_time = t_start
        
        # Control step counters
        step_a_counter = 0
        step_b_counter = 0
        next_step_a = t_start
        next_step_b = t_start
        
        # Main simulation loop
        step_count = 0
        logger.info("\n" + "="*80)
        logger.info("▶️  Starting simulation loop...")
        logger.info("="*80 + "\n")
        
        while self.current_time < t_end:
            
            # Check for active attacks
            active_attacks = self._get_active_attacks(self.current_time)
            
            # Determine if buildings should take control steps
            step_a = (self.current_time >= next_step_a)
            step_b = (self.current_time >= next_step_b)
            
            # Building A control step
            if step_a:
                state_a = self.building_a.step(dt_a, active_attacks)
                step_a_counter += 1
                next_step_a += dt_a
            else:
                state_a = self.building_a.current_state
            
            # Building B control step
            if step_b:
                state_b = self.building_b.step(dt_b, active_attacks)
                step_b_counter += 1
                next_step_b += dt_b
            else:
                state_b = self.building_b.current_state
            
            # Calculate feeder metrics
            if state_a and state_b:
                feeder_total = state_a['power_kW'] + state_b['power_kW']
                feeder_capacity = self.sim_config.feeder_capacity_kW
                
                # Record metrics
                attack_active = len(active_attacks) > 0
                attack_name = active_attacks[0].name if attack_active else ""
                
                self.metrics.record_step(
                    timestamp=self.current_time,
                    building_a_state=state_a,
                    building_b_state=state_b,
                    feeder_total_power=feeder_total,
                    feeder_capacity=feeder_capacity,
                    attack_active=attack_active,
                    attack_name=attack_name
                )
                
                # Log progress
                if step_count % 20 == 0:  # Log every 20 steps
                    sim_hours = (self.current_time - t_start) / 3600
                    logger.info(f"⏰ t={sim_hours:.1f}h: P_A={state_a['power_kW']:.2f}kW, "
                               f"P_B={state_b['power_kW']:.2f}kW, "
                               f"Total={feeder_total:.2f}kW ({feeder_total/feeder_capacity*100:.1f}%)")
                    
                    if attack_active:
                        logger.warning(f"   ⚠️  ATTACK ACTIVE: {attack_name}")
                    
                    if feeder_total > feeder_capacity:
                        logger.error(f"   🚨 FEEDER VIOLATION! Exceed by {feeder_total-feeder_capacity:.2f}kW")
                        logger.error(f"   💡 Suggestion: Reduce scheduled power during peak hours")
            
            # Advance simulation time
            self.current_time += dt_sim
            step_count += 1
        
        logger.info("\n" + "="*80)
        logger.info("✅ SIMULATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Total steps: {step_count}")
        logger.info(f"Building A control steps: {step_a_counter}")
        logger.info(f"Building B control steps: {step_b_counter}")
        
        # Calculate summary metrics
        summary_metrics = self.metrics.calculate_summary_metrics()
        
        # Shutdown buildings
        self.building_a.shutdown()
        self.building_b.shutdown()
        
        return {
            'metrics_collector': self.metrics,
            'summary_metrics': summary_metrics,
            'dataframe': self.metrics.get_dataframe()
        }
    
    def _get_active_attacks(self, current_time: float) -> List[AttackEvent]:
        """Get currently active attacks"""
        active = []
        for attack in self.sim_config.attack_events:
            if attack.is_active(current_time, self.simulation_start_time):
                active.append(attack)
        return active
