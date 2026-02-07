"""
Hierarchical Coordinator - Main Orchestration

Implements proper execution sequence:
1. Measure → Buildings report states
2. Flexibility → Buildings compute bands (two-pass MPC)  
3. Allocate → Aggregator solves allocation
4. Optimize → Buildings solve MPC with budgets
5. Actuate → Apply control inputs

Author: Guowen Li
Date: 2025-02-06
"""

import logging
import time
from typing import Dict, List, Optional
import sys
import os
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buildings import BuildingASimple, BuildingBSimple
from aggregator import LogUtilityAggregator, LogUtilityAggregatorConfig, AttackAnticipator
from communication import MessageBroker

logger = logging.getLogger(__name__)


class HierarchicalCoordinator:
    """Main coordinator for hierarchical control"""
    
    def __init__(self, config_path: str):
        logger.info("="*80)
        logger.info("🚀 HIERARCHICAL COORDINATOR")
        logger.info("="*80)
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize buildings
        logger.info("🏢 Buildings...")
        self.building_a = BuildingASimple('Building_A', self.config)
        self.building_b = BuildingBSimple('Building_B', self.config)
        self.buildings = [self.building_a, self.building_b]
        
        # Initialize aggregator
        logger.info("🎯 Aggregator...")
        agg_config = LogUtilityAggregatorConfig(
            PH=self.config['timing']['prediction_horizon_aggregator'],
            dt=self.config['timing']['aggregator_timestep'],
            P_feeder_limit_kW=self.config['feeder']['capacity_kW'],
            safety_margin=self.config['feeder']['safety_margin'],
            default_priority_weights=self.config.get('priorities', {})
        )
        self.aggregator = LogUtilityAggregator(agg_config)
        
        for bldg_id, priority in self.config.get('priorities', {}).items():
            self.aggregator.register_building(bldg_id, priority)
        
        self.message_broker = MessageBroker()
        self.dt_aggregator = self.config['timing']['aggregator_timestep']
        self.metrics_history = []
        
        logger.info("✅ READY")
        logger.info("="*80)
    
    def execute_control_cycle(self, current_time, weather_forecast, 
                              price_forecast, feeder_limit) -> Dict:
        """Execute one control cycle"""
        
        cycle_start = time.time()
        
        logger.info(f"🔄 CYCLE @ t={current_time/86400:.2f} days")
        
        # STEP 1: Measure
        building_states = {}
        for bldg in self.buildings:
            state = bldg.get_state()
            building_states[bldg.building_id] = state
        
        # STEP 2: Flexibility
        flex_start = time.time()
        flexibility_bands = []
        for bldg in self.buildings:
            band = bldg.compute_flexibility_band(
                current_time, weather_forecast, price_forecast,
                self.aggregator.PH
            )
            flexibility_bands.append(band)
        flex_time = time.time() - flex_start
        
        if not all(b.feasible for b in flexibility_bands):
            return {'success': False, 'reason': 'infeasible_bands'}
        
        # STEP 3: Allocate
        agg_start = time.time()
        allocation = self.aggregator.allocate_power(
            flexibility_bands, feeder_limit, current_time
        )
        agg_time = time.time() - agg_start
        
        if not allocation.feasible:
            return {'success': False, 'reason': 'infeasible_allocation'}
        
        # STEP 4: Optimize
        mpc_start = time.time()
        control_inputs = {}
        for i, bldg in enumerate(self.buildings):
            budget = allocation.budgets[i]
            result = bldg.solve_mpc_with_budget(
                current_time, budget.P_ref_kW, weather_forecast, price_forecast
            )
            if not result.feasible:
                return {'success': False, 'reason': f'{bldg.building_id}_mpc'}
            control_inputs[bldg.building_id] = result.control_inputs
        mpc_time = time.time() - mpc_start
        
        # STEP 5: Actuate
        for bldg in self.buildings:
            bldg.apply_control(control_inputs[bldg.building_id])
        
        cycle_time = time.time() - cycle_start
        
        metrics = {
            'timestamp': current_time,
            'cycle_time': cycle_time,
            'flex_time': flex_time,
            'agg_time': agg_time,
            'mpc_time': mpc_time,
            'total_power': allocation.total_power_kW[0],
            'feeder_limit': feeder_limit[0],
            'utilization': 100 * allocation.total_power_kW[0] / feeder_limit[0]
        }
        self.metrics_history.append(metrics)
        
        logger.info(f"✅ Complete: {cycle_time:.3f}s, Power: {allocation.total_power_kW[0]:.1f}kW")
        
        return {'success': True, 'allocation': allocation, 'metrics': metrics}
    
    def run_simulation(self, start_time, end_time, weather_data, 
                       price_data, feeder_limit_schedule=None):
        """Run full simulation"""
        
        logger.info("🚀 SIMULATION START")
        
        if feeder_limit_schedule is None:
            n = int((end_time - start_time) / self.dt_aggregator) + 100
            feeder_limit_schedule = [self.config['feeder']['capacity_kW']] * n
        
        t = start_time
        while t < end_time:
            idx = int((t - start_time) / self.dt_aggregator)
            horizon = self.aggregator.PH
            
            weather_forecast = {'Toa': weather_data['Toa'][idx:idx+horizon]}
            price_forecast = price_data[idx:idx+horizon]
            feeder_limit = feeder_limit_schedule[idx:idx+horizon]
            
            result = self.execute_control_cycle(
                t, weather_forecast, price_forecast, feeder_limit
            )
            
            if not result['success']:
                logger.error(f"❌ Failed: {result.get('reason')}")
                break
            
            t += self.dt_aggregator
        
        logger.info("✅ SIMULATION COMPLETE")
        self._save_results()
    
    def _save_results(self):
        """Save results"""
        import pandas as pd
        from datetime import datetime
        
        if not self.metrics_history:
            logger.warning("⚠️  No metrics to save (simulation failed early)")
            return
        
        df = pd.DataFrame(self.metrics_history)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'results/metrics_{timestamp}.csv'
        df.to_csv(filename, index=False)
        logger.info(f"💾 Saved: {filename}")
        
        logger.info(f"📊 Avg cycle: {df['cycle_time'].mean():.3f}s")
        logger.info(f"📊 Avg utilization: {df['utilization'].mean():.1f}%")
    
    def shutdown(self):
        for bldg in self.buildings:
            bldg.shutdown()
