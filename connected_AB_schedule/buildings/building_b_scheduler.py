"""
Building B Scheduler - TES schedule-based control with MPC hybrid

Author: Guowen Li, AI Assistant
Date: 2025-01-20
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)
sys.path.append(os.path.join(parent_dir, '../buildingB_w_TES'))

from pyfmi import load_fmu
from mpc_b import mpc_case
from buildings.base_schedule_building import BaseScheduleBuilding
from schedule.control_models import DailySchedule, AttackEvent
from schedule.schedule_manager import ScheduleManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BuildingBScheduler(BaseScheduleBuilding):
    """Building B with TES schedule-based control"""
    
    def __init__(self, config: Dict, daily_schedule: DailySchedule):
        super().__init__("Building_B", config, daily_schedule)
        
        # MPC parameters
        self.PH = config['building_b']['prediction_horizon_steps']
        self.default_control_interval = config['building_b']['control_interval_minutes'] * 60
        
        # TES parameters
        self.mIce_max = config['building_b']['tes']['mIce_max_kg']
        self.SOC_current = config['building_b']['tes']['SOC_initial']
        self.SOC_target = config['building_b']['tes']['SOC_target']
        
        # FMU and MPC
        self.fmu = None
        self.mpc = None
        
        # Historical states
        self.states = None
        self.predictor = None
        
        # Model paths
        building_b_base = os.path.join(parent_dir, '../buildingB_w_TES')
        self.model_paths = {
            'core': os.path.join(building_b_base, "system_identification/dnn_model_core_temperature.h5"),
            'east': os.path.join(building_b_base, "system_identification/dnn_model_east_temperature.h5"),
            'north': os.path.join(building_b_base, "system_identification/dnn_model_north_temperature.h5"),
            'south': os.path.join(building_b_base, "system_identification/dnn_model_south_temperature.h5"),
            'west': os.path.join(building_b_base, "system_identification/dnn_model_west_temperature.h5"),
            'SOC': os.path.join(building_b_base, "system_identification/dnn_SOC_model.h5"),
            'power': os.path.join(building_b_base, "system_identification/dnn_power_model.h5")
        }
        
        logger.info(f"🏢 Building B Scheduler initialized: PH={self.PH}, dt={self.default_control_interval/60:.0f}min")
    
    def initialize(self, initial_conditions: Dict):
        """Initialize Building B FMU and MPC"""
        
        logger.info("🏗️ Initializing Building B...")
        
        # Set Dymola license
        if "DYMOLA_RUNTIME_LICENSE" not in os.environ:
            os.environ["DYMOLA_RUNTIME_LICENSE"] = "c:/programdata/dassaultsystemes/dymola/dymola.lic"
        
        # Load FMU
        fmu_path = os.path.join(
            parent_dir,
            '../buildingB_w_TES/modelica_model/VirtualTestbed_NISTChillerTestbed_DemandFlexibilityInvestigation_FakeSystem_SystemForMPC_01bInput_0modeSignal.fmu'
        )
        self.fmu = load_fmu(fmu_path, log_level=3)
        
        # Set TES parameters
        self.SOC_current = initial_conditions.get('SOC_ini', 0.5)
        self.fmu.set('mIce_max', self.mIce_max)
        self.fmu.set('mIce_start', self.SOC_current * self.mIce_max)
        
        # FMU options
        self.fmu_options = self.fmu.simulate_options()
        self.fmu_options['ncp'] = 100
        self.fmu_options['initialize'] = True
        
        # Initialize states
        t_start = initial_conditions['simulation_start_time']
        Toa_his = initial_conditions.get('Toa_history', [25.0]*4)
        GHI_his = initial_conditions.get('GHI_history', [0.0]*4)
        
        self.states = {
            'Tz_core_his_meas': [24.0] * 4,
            'Tz_east_his_meas': [24.0] * 4,
            'Tz_north_his_meas': [24.0] * 4,
            'Tz_south_his_meas': [24.0] * 4,
            'Tz_west_his_meas': [24.0] * 4,
            'Tz_ave_his_meas': [24.0] * 4,
            'To_his_meas': Toa_his,
            'GHI_his_meas': GHI_his,
            'SOC_his_meas': [self.SOC_current] * 4,
            'P_his_meas': [0.0] * 4,
            'Tz_core_his_pred': [24.0] * 4,
            'Tz_east_his_pred': [24.0] * 4,
            'Tz_north_his_pred': [24.0] * 4,
            'Tz_south_his_pred': [24.0] * 4,
            'Tz_west_his_pred': [24.0] * 4
        }
        
        # Initialize predictor
        self.predictor = {
            'Toa': initial_conditions.get('Toa_forecast', [25.0] * self.PH),
            'RHoa': initial_conditions.get('RHoa_forecast', [0.5] * self.PH),
            'GHI': initial_conditions.get('GHI_forecast', [0.0] * self.PH),
            'price': initial_conditions.get('price_forecast', [0.1] * self.PH)
        }
        
        # Initialize MPC
        self.mpc = mpc_case(
            PH=self.PH,
            CH=1,
            time=t_start,
            dt=self.get_control_interval(),
            measurement={},
            states=self.states,
            predictor=self.predictor
        )
        
        self.current_time = t_start
        
        # Store model paths
        self.mpc.model_paths = self.model_paths
        
        # Initialize schedule manager
        self.schedule_manager = ScheduleManager(self.daily_schedule, t_start)
        
        logger.info(f"✅ Building B initialized: SOC={self.SOC_current:.2f}, TES={self.mIce_max:.0f}kg")
        logger.info(f"   {self.schedule_manager.get_schedule_summary()}")
    
    def apply_schedule(self, current_time: float) -> Dict[str, float]:
        """Get scheduled TES mode for current time"""
        scheduled_vars = self.schedule_manager.get_control_action(current_time)
        return scheduled_vars if scheduled_vars else {}
    
    def optimize_unscheduled(self, scheduled_vars: Dict[str, float]) -> Dict[str, float]:
        """MPC optimizes TES mode if not scheduled"""
        
        # Update MPC
        self.mpc.set_time(self.current_time)
        self.mpc.set_states(self.states)
        self.mpc.set_predictor(self.predictor)
        
        # If uMod is scheduled, use it; otherwise optimize
        if 'uMod' in scheduled_vars:
            uMod = int(scheduled_vars['uMod'])
            logger.debug(f"📅 Building B: Using scheduled uMod={uMod}")
        else:
            # Run MPC optimization
            try:
                res, solver_status = self.mpc.optimize()
                
                if solver_status['return_status'] == 'INFEASIBLE':
                    logger.warning(f"⚠️ Building B MPC infeasible, using fallback")
                    uMod = 0  # Fallback: off
                else:
                    u_opt = res['x']
                    uMod = int(u_opt[0])
                    logger.debug(f"🎯 Building B: MPC optimized uMod={uMod}")
            
            except Exception as e:
                logger.error(f"❌ Building B MPC failed: {e}")
                uMod = 0
        
        return {'uMod': uMod}
    
    def apply_attacks(self, control_vars: Dict[str, float]) -> Dict[str, float]:
        """Apply active attacks (Building B typically not attacked in this scenario)"""
        if not self.active_attacks:
            return control_vars
        
        attacked_vars = control_vars.copy()
        
        for attack in self.active_attacks:
            if attack.target_building != "Building_B":
                continue
            
            if 'uMod' in attack.affected_variables:
                # Example attack: force TES off
                attacked_vars['uMod'] = attack.attack_params.get('forced_mode', 0)
                logger.warning(f"⚠️ ATTACK: Building B uMod forced to {attacked_vars['uMod']}")
        
        return attacked_vars
    
    def step(self, dt: float, active_attacks: List[AttackEvent]) -> Dict:
        """Execute one control step"""
        
        self.active_attacks = active_attacks
        
        # 1. Get scheduled controls
        scheduled_vars = self.apply_schedule(self.current_time)
        
        # 2. MPC optimizes if not scheduled
        all_controls = self.optimize_unscheduled(scheduled_vars)
        
        # 3. Apply attacks
        final_controls = self.apply_attacks(all_controls)
        
        uMod = int(final_controls['uMod'])
        
        # 4. Apply to FMU
        self.fmu.set('uMod', uMod)
        
        # 5. Simulate FMU
        ts = self.current_time
        te = ts + dt
        fmu_result = self.fmu.simulate(start_time=ts, final_time=te, options=self.fmu_options)
        
        # 6. Extract measurement
        measurement = self._extract_measurement(fmu_result)
        
        # 7. Update SOC
        self.SOC_current = measurement['iceTan.SOC'].values[0]
        
        # 8. Get predictions
        Tz_pred = {
            'core': float(self.mpc.get_core_temp_pred([uMod])),
            'east': float(self.mpc.get_east_temp_pred([uMod])),
            'north': float(self.mpc.get_north_temp_pred([uMod])),
            'south': float(self.mpc.get_south_temp_pred([uMod])),
            'west': float(self.mpc.get_west_temp_pred([uMod]))
        }
        
        # 9. Update states
        self.states = self._update_states(self.states, measurement, Tz_pred)
        
        # 10. Update time
        self.current_time = te
        self.fmu_options['initialize'] = False
        
        # 11. Calculate metrics
        power_actual = max(measurement['chi.P'].values[0], 0) + \
                      max(measurement['priPum.P'].values[0], 0) + \
                      max(measurement['secPum.P'].values[0], 0) + \
                      max(measurement['fanSup.P'].values[0], 0)
        
        zone_temps = {
            'core': measurement['conVAVCor.TZon'].values[0] - 273.15,
            'east': measurement['conVAVEas.TZon'].values[0] - 273.15,
            'north': measurement['conVAVNor.TZon'].values[0] - 273.15,
            'south': measurement['conVAVSou.TZon'].values[0] - 273.15,
            'west': measurement['conVAVWes.TZon'].values[0] - 273.15
        }
        
        comfort_violation = self._calculate_comfort_violation(zone_temps, dt)
        
        # 12. Return state
        state = {
            'timestamp': self.current_time,
            'building_id': 'Building_B',
            'power_kW': power_actual / 1000.0,
            'zone_temps': zone_temps,
            'comfort_violation_degCh': comfort_violation,
            'SOC': self.SOC_current,
            'TES_mode': uMod,
            'controls_applied': final_controls,
            'scheduled_vars': list(scheduled_vars.keys()) if scheduled_vars else [],
            'under_attack': len(self.active_attacks) > 0
        }
        
        self.current_state = state
        return state
    
    def _extract_measurement(self, fmu_result) -> pd.DataFrame:
        """Extract measurement from FMU"""
        measurement_names = [
            'time', 'TOut.y', 'weaBus.HGloHor', 'uMod', 'iceTan.SOC',
            'conVAVCor.TZon', 'conVAVEas.TZon', 'conVAVNor.TZon',
            'conVAVSou.TZon', 'conVAVWes.TZon', 'ave.y',
            'chi.P', 'priPum.P', 'secPum.P', 'fanSup.P'
        ]
        
        dic = {name: fmu_result[name][-1] for name in measurement_names}
        return pd.DataFrame(dic, index=[dic['time']])
    
    def _update_states(self, states, measurement, Tz_pred):
        """Update MPC states"""
        def FILO(a_list, x):
            a_list.insert(0, x)
            a_list.pop()
            return a_list
        
        # Extract measurements
        Tz_core = measurement['conVAVCor.TZon'].values[0] - 273.15
        Tz_east = measurement['conVAVEas.TZon'].values[0] - 273.15
        Tz_north = measurement['conVAVNor.TZon'].values[0] - 273.15
        Tz_south = measurement['conVAVSou.TZon'].values[0] - 273.15
        Tz_west = measurement['conVAVWes.TZon'].values[0] - 273.15
        Tz_ave = measurement['ave.y'].values[0] - 273.15
        Toa = measurement['TOut.y'].values[0] - 273.15
        GHI = measurement['weaBus.HGloHor'].values[0]
        SOC = measurement['iceTan.SOC'].values[0]
        P = max(measurement['chi.P'].values[0], 0) + \
            max(measurement['priPum.P'].values[0], 0) + \
            max(measurement['secPum.P'].values[0], 0) + \
            max(measurement['fanSup.P'].values[0], 0)
        
        # Update states
        states['Tz_core_his_meas'] = FILO(states['Tz_core_his_meas'], Tz_core)
        states['Tz_east_his_meas'] = FILO(states['Tz_east_his_meas'], Tz_east)
        states['Tz_north_his_meas'] = FILO(states['Tz_north_his_meas'], Tz_north)
        states['Tz_south_his_meas'] = FILO(states['Tz_south_his_meas'], Tz_south)
        states['Tz_west_his_meas'] = FILO(states['Tz_west_his_meas'], Tz_west)
        states['Tz_ave_his_meas'] = FILO(states['Tz_ave_his_meas'], Tz_ave)
        states['To_his_meas'] = FILO(states['To_his_meas'], Toa)
        states['GHI_his_meas'] = FILO(states['GHI_his_meas'], GHI)
        states['SOC_his_meas'] = FILO(states['SOC_his_meas'], SOC)
        states['P_his_meas'] = FILO(states['P_his_meas'], P)
        states['Tz_core_his_pred'] = FILO(states['Tz_core_his_pred'], Tz_pred['core'])
        states['Tz_east_his_pred'] = FILO(states['Tz_east_his_pred'], Tz_pred['east'])
        states['Tz_north_his_pred'] = FILO(states['Tz_north_his_pred'], Tz_pred['north'])
        states['Tz_south_his_pred'] = FILO(states['Tz_south_his_pred'], Tz_pred['south'])
        states['Tz_west_his_pred'] = FILO(states['Tz_west_his_pred'], Tz_pred['west'])
        
        return states
    
    def _calculate_comfort_violation(self, zone_temps: Dict[str, float], dt: float) -> float:
        """Calculate comfort violation in degree-hours"""
        T_upper = self.config['building_b']['comfort']['T_upper']
        T_lower = self.config['building_b']['comfort']['T_lower']
        violation = 0.0
        
        for temp in zone_temps.values():
            violation += max(0, temp - T_upper) + max(0, T_lower - temp)
        
        return violation * dt / 3600.0
    
    def get_power_forecast(self, horizon: int) -> List[float]:
        """Get power forecast (simplified)"""
        baseline = self.config['building_b']['baseline_power_kW']
        return [baseline] * horizon
    
    def shutdown(self):
        """Clean shutdown"""
        logger.info("🛑 Shutting down Building B")
