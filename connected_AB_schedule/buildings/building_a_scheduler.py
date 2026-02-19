"""
Building A Scheduler - Schedule-based control with MPC hybrid

Author: Guowen Li, AI Assistant
Date: 2025-01-20
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)
sys.path.append(os.path.join(parent_dir, '../buildingA_wo_TES'))

from pyfmi import load_fmu
from mpc_a import mpc_case
from buildings.base_schedule_building import BaseScheduleBuilding
from schedule.control_models import DailySchedule, AttackEvent, BuildingAVariables
from schedule.schedule_manager import ScheduleManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BuildingAScheduler(BaseScheduleBuilding):
    """Building A with schedule-based control"""
    
    def __init__(self, config: Dict, daily_schedule: DailySchedule):
        super().__init__("Building_A", config, daily_schedule)
        
        # MPC parameters
        self.PH = config['building_a']['prediction_horizon_steps']
        self.default_control_interval = config['building_a']['control_interval_minutes'] * 60
        
        # FMU and MPC
        self.fmu = None
        self.mpc = None
        
        # Historical states
        self.states = None
        self.predictor = None
        
        logger.info(f"🏢 Building A Scheduler initialized: PH={self.PH}, dt={self.default_control_interval/60:.0f}min")
    
    def initialize(self, initial_conditions: Dict):
        """Initialize Building A FMU and MPC"""
        
        logger.info("🏗️ Initializing Building A...")
        
        # Set Dymola license
        if "DYMOLA_RUNTIME_LICENSE" not in os.environ:
            os.environ["DYMOLA_RUNTIME_LICENSE"] = "c:/programdata/dassaultsystemes/dymola/dymola.lic"
        
        # Load FMU
        fmu_path = os.path.join(
            parent_dir,
            '../buildingA_wo_TES/modelica_model/wrapped_fixed_modified_ecoRet09_02162023.fmu'
        )
        self.fmu = load_fmu(fmu_path, log_level=3)
        
        # FMU options
        self.fmu_options = self.fmu.simulate_options()
        self.fmu_options['ncp'] = 100
        self.fmu_options['initialize'] = True
            
        # Initialize states
        t_start = initial_conditions['simulation_start_time']
        Toa_his = initial_conditions.get('Toa_history', [25.0]*4)
        
        self.states = {
            'Tz_core_his_meas': [273.15 + 24] * 4,
            'Tz_east_his_meas': [273.15 + 24] * 4,
            'Tz_north_his_meas': [273.15 + 24] * 4,
            'Tz_south_his_meas': [273.15 + 24] * 4,
            'Tz_west_his_meas': [273.15 + 24] * 4,
            'To_his_meas': Toa_his,
            'P_his_meas': [0.0],
            'Tz_core_his_pred': [273.15 + 24] * 4,
            'Tz_east_his_pred': [273.15 + 24] * 4,
            'Tz_north_his_pred': [273.15 + 24] * 4,
            'Tz_south_his_pred': [273.15 + 24] * 4,
            'Tz_west_his_pred': [273.15 + 24] * 4
        }
        
        # Initialize predictor
        self.predictor = {
            'Toa': initial_conditions.get('Toa_forecast', [25.0] * self.PH),
            'RHoa': initial_conditions.get('RHoa_forecast', [0.5] * self.PH),
            'price': initial_conditions.get('price_forecast', [0.1] * self.PH)
        }
        
        
        # Load MPC models
        import json
        model_path = os.path.join(os.path.dirname(__file__), '../../buildingA_wo_TES/system_identification')
        
        mpc_models = {
            'fan': json.load(open(os.path.join(model_path, 'fan.json'))),
            'fan_Tset22': json.load(open(os.path.join(model_path, 'fan_Tset22.json'))),
            'fan_Tset26': json.load(open(os.path.join(model_path, 'fan_Tset26.json'))),
            'chiller_plant': json.load(open(os.path.join(model_path, 'chiller_plant.json'))),
            'chiller_plant_Tset22': json.load(open(os.path.join(model_path, 'chiller_plant_Tset22.json'))),
            'chiller_plant_Tset26': json.load(open(os.path.join(model_path, 'chiller_plant_Tset26.json'))),
            'core': json.load(open(os.path.join(model_path, 'zone_results/TZone_fiveV_Core.json'))),
            'east': json.load(open(os.path.join(model_path, 'zone_results/TZone_fiveV_East.json'))),
            'north': json.load(open(os.path.join(model_path, 'zone_results/TZone_fiveV_North.json'))),
            'south': json.load(open(os.path.join(model_path, 'zone_results/TZone_fiveV_South.json'))),
            'west': json.load(open(os.path.join(model_path, 'zone_results/TZone_fiveV_West.json')))
        }
        
        # Initialize MPC
        self.mpc = mpc_case(
            PH=self.PH,
            CH=1,
            time=t_start,
            dt=self.get_control_interval(),
            measurement={},
            states=self.states,
            predictor=self.predictor,
            mpc_models=mpc_models,
        )
        
        self.current_time = t_start
        
        # Initialize schedule manager
        self.schedule_manager = ScheduleManager(self.daily_schedule, t_start)
        
        logger.info(f"✅ Building A initialized")
        logger.info(f"   {self.schedule_manager.get_schedule_summary()}")
    
    def apply_schedule(self, current_time: float) -> Dict[str, float]:
        """Get scheduled control actions for current time"""
        scheduled_vars = self.schedule_manager.get_control_action(current_time)
        return scheduled_vars if scheduled_vars else {}
    
    def optimize_unscheduled(self, scheduled_vars: Dict[str, float]) -> Dict[str, float]:
        """
        MPC optimizes variables not in schedule
        
        For scheduled variables, they are set as hard constraints in MPC
        """
        # Update MPC time and states
        self.mpc.set_time(self.current_time)
        self.mpc.set_states(self.states)
        self.mpc.set_predictor(self.predictor)
        
        # Run MPC with scheduled variables as constraints
        try:
            res, solver_status = self.mpc.optimize(fixed_vars=scheduled_vars)
            
            if solver_status['return_status'] != 'Solve_Succeeded':
                logger.warning(f"⚠️ Building A MPC solver status: {solver_status['return_status']}")
                logger.warning(f"   Scheduled constraints: {scheduled_vars}")
                
                # Fallback: use scheduled vars only, default values for others
                optimized_vars = self._get_fallback_controls(scheduled_vars)
            else:
                # Extract optimized variables
                u_opt = res['x']
                optimized_vars = {
                    'bcp': int(u_opt[0]),
                    'bahu': int(u_opt[1]),
                    'Tchw': float(u_opt[2]),
                    'Tcw': float(u_opt[3]),
                    'Tsa': float(u_opt[4]),
                    'Vcore': float(u_opt[5]),
                    'Veast': float(u_opt[6]),
                    'Vnorth': float(u_opt[7]),
                    'Vsouth': float(u_opt[8]),
                    'Vwest': float(u_opt[9]),
                    'epsilon': float(u_opt[10])
                }
                
                # Override with scheduled values
                optimized_vars.update(scheduled_vars)
        
        except Exception as e:
            logger.error(f"❌ Building A MPC optimization failed: {e}")
            optimized_vars = self._get_fallback_controls(scheduled_vars)
        
        return optimized_vars
    
    def _get_fallback_controls(self, scheduled_vars: Dict[str, float]) -> Dict[str, float]:
        """Fallback control when MPC fails"""
        defaults = {
            'bcp': 1,
            'bahu': 1,
            'Tchw': 7.0,
            'Tcw': 25.0,
            'Tsa': 13.0,
            'Vcore': 0.5,
            'Veast': 0.5,
            'Vnorth': 0.5,
            'Vsouth': 0.5,
            'Vwest': 0.5,
            'epsilon': 0.0
        }
        defaults.update(scheduled_vars)
        return defaults
    
    def apply_attacks(self, control_vars: Dict[str, float]) -> Dict[str, float]:
        """Apply active attacks to control variables"""
        if not self.active_attacks:
            return control_vars
        
        attacked_vars = control_vars.copy()
        
        for attack in self.active_attacks:
            if attack.target_building != "Building_A":
                continue
            
            for var_name in attack.affected_variables:
                if var_name in attacked_vars:
                    if attack.attack_type == "vav_reinitialization":
                        attacked_vars[var_name] = attack.attack_params.get('reinitialization_value', 0.0)
                        logger.warning(f"⚠️ ATTACK: {var_name} forced to {attacked_vars[var_name]}")
                    
                    elif attack.attack_type == "setpoint_manipulation":
                        attacked_vars[var_name] = attack.attack_params.get('manipulated_value', attacked_vars[var_name])
                        logger.warning(f"⚠️ ATTACK: {var_name} manipulated to {attacked_vars[var_name]}")
        
        return attacked_vars
    
    def step(self, dt: float, active_attacks: List[AttackEvent]) -> Dict:
        """Execute one control step"""
        
        self.active_attacks = active_attacks
        
        # 1. Get scheduled controls
        scheduled_vars = self.apply_schedule(self.current_time)
        
        # 2. MPC optimizes unscheduled variables
        all_controls = self.optimize_unscheduled(scheduled_vars)
        
        # 3. Apply attacks (attacks override both scheduled and optimized)
        final_controls = self.apply_attacks(all_controls)
        
        # 4. Apply to FMU
        # self.fmu.set('bcp', final_controls['bcp'])
        # self.fmu.set('bahu', final_controls['bahu'])
        # self.fmu.set('chiConWatTemp', final_controls['Tchw'])
        # self.fmu.set('conWatTemp', final_controls['Tcw'])
        # self.fmu.set('ahuSATemp', final_controls['Tsa'])
        # self.fmu.set('coreVAV', final_controls['Vcore'])
        # self.fmu.set('eastVAV', final_controls['Veast'])
        # self.fmu.set('northVAV', final_controls['Vnorth'])
        # self.fmu.set('southVAV', final_controls['Vsouth'])
        # self.fmu.set('westVAV', final_controls['Vwest'])
        
        # Extract control actions [b_cp, b_ahu, T_chw, T_cw, T_sa, V_core, V_east, V_north, V_south, V_west]
        uMPC = [
            float(final_controls['bcp']),  # b_cp
            float(final_controls['bahu']),  # b_ahu
            float(final_controls['Tchw']) + 273.15,  # T_chw (C to K)
            float(final_controls['Tcw']) + 273.15,  # T_cw
            float(final_controls['Tsa']) + 273.15,  # T_sa
            float(final_controls['Vcore']),  # V_core
            float(final_controls['Veast']),  # V_east
            float(final_controls['Vnorth']),  # V_north
            float(final_controls['Vsouth']),  # V_south
            float(final_controls['Vwest'])   # V_west
        ]
        # ============ Apply controls to FMU ============
        # Based on run_mpc_v2.py structure: separate _u and _activate variables
        # Define control variable names (without _u suffix)
        control_var_names = [
            'oveOnChiPla',
            'conAHU_supFan_oveOnSupFan',
            'oveTChiWatSupSet',
            'oveTConWatSupSet',
            'conAHU_oveTSupAir',
            'conVAVCor_damVal_oveVDisSet',
            'conVAVEas_damVal_oveVDisSet',
            'conVAVNor_damVal_oveVDisSet',
            'conVAVSou_damVal_oveVDisSet',
            'conVAVWes_damVal_oveVDisSet'
        ]

        # Set control values (_u variables)
        for var_name, val in zip(control_var_names, uMPC):
            self.fmu.set(var_name + '_u', val)

        # Activate all overrides (_activate variables)
        for var_name in control_var_names:
            self.fmu.set(var_name + '_activate', 1)        
        
        
        # 5. Simulate FMU
        ts = self.current_time
        te = ts + dt
        fmu_result = self.fmu.simulate(start_time=ts, final_time=te, options=self.fmu_options)
        
        # 6. Extract measurement
        measurement = self._extract_measurement(fmu_result)
        
        # 7. Update states
        # Update states for next iteration
        u_opt = uMPC
        Tz_pred = {
            'core': float(self.mpc.get_core_temp_pred(u_opt[:10], self.mpc._autoerror['core'])),
            'east': float(self.mpc.get_east_temp_pred(u_opt[:10], self.mpc._autoerror['east'])),
            'north': float(self.mpc.get_north_temp_pred(u_opt[:10], self.mpc._autoerror['north'])),
            'south': float(self.mpc.get_south_temp_pred(u_opt[:10], self.mpc._autoerror['south'])),
            'west': float(self.mpc.get_west_temp_pred(u_opt[:10], self.mpc._autoerror['west']))
        }
        
        self.states = self._update_states(self.states, measurement, Tz_pred)
        
        # Update time
        self.current_time = te
        self.fmu_options['initialize'] = False
        
        # Calculate metrics
        power_actual = abs(measurement['mod.eleChi.y'].values[0]) + \
                      abs(measurement['mod.eleCHWP.y'].values[0]) + \
                      abs(measurement['mod.eleCT.y'].values[0]) + \
                      abs(measurement['mod.eleCWP.y'].values[0]) + \
                      abs(measurement['mod.eleSupFan.y'].values[0])
        
        zone_temps = {
            'core': measurement['mod.flo.temAirPer5.T'].values[0] - 273.15,
            'east': measurement['mod.flo.temAirEas.T'].values[0] - 273.15,
            'north': measurement['mod.flo.temAirNor.T'].values[0] - 273.15,
            'south': measurement['mod.flo.temAirSou.T'].values[0] - 273.15,
            'west': measurement['mod.flo.temAirWes.T'].values[0] - 273.15
        }
        
        zone_airflows = {
            'core': measurement['mod.conVAVCor.VDis_flow'].values[0],
            'east': measurement['mod.conVAVEas.VDis_flow'].values[0],
            'north': measurement['mod.conVAVNor.VDis_flow'].values[0],
            'south': measurement['mod.conVAVSou.VDis_flow'].values[0],
            'west': measurement['mod.conVAVWes.VDis_flow'].values[0]
        }
                
        comfort_violation = self._calculate_comfort_violation(zone_temps, dt)
        
        # 10. Return state
        state = {
            'timestamp': self.current_time,
            'building_id': 'Building_A',
            'power_kW': power_actual / 1000.0,
            'zone_temps': zone_temps,
            'zone_airflows': zone_airflows,
            'comfort_violation_degCh': comfort_violation,
            'controls_applied': final_controls,
            'scheduled_vars': list(scheduled_vars.keys()) if scheduled_vars else [],
            'under_attack': len(self.active_attacks) > 0
        }
        
        self.current_state = state
        return state
    
    def _extract_measurement(self, fmu_result) -> pd.DataFrame:
        """Extract measurement from FMU"""
        measurement_names = [            
            'time', 'mod.TCHWSup.T', 'mod.TCWSup.T', 'mod.TSup.T',
            'mod.conVAVSou.VDis_flow', 'mod.conVAVEas.VDis_flow',
            'mod.conVAVNor.VDis_flow', 'mod.conVAVWes.VDis_flow',
            'mod.conVAVCor.VDis_flow', 'mod.flo.temAirSou.T',
            'mod.flo.temAirEas.T', 'mod.flo.temAirNor.T',
            'mod.flo.temAirWes.T', 'mod.flo.temAirPer5.T',
            'mod.eleChi.y', 'mod.eleCHWP.y', 'mod.eleCWP.y',
            'mod.eleCT.y', 'mod.eleSupFan.y', 'mod.TOut.y'
        ]
        
        dic = {name: fmu_result[name][-1] for name in measurement_names}
        return pd.DataFrame(dic, index=[dic['time']])
    
    def _update_states(self, states, measurement, Tz_pred):
        """Update MPC states using FILO (First In Last Out)"""
        
        def FILO(a_list, x):
            a_list.insert(0, x)
            a_list.pop()
            return a_list
        
        # Extract measurements (temperatures in Kelvin, convert to Celsius)
        # Use .values[0] to access the first (and only) row by position
        Tz_core = measurement['mod.flo.temAirPer5.T'].values[0] - 273.15
        Tz_east = measurement['mod.flo.temAirEas.T'].values[0] - 273.15
        Tz_north = measurement['mod.flo.temAirNor.T'].values[0] - 273.15
        Tz_south = measurement['mod.flo.temAirSou.T'].values[0] - 273.15
        Tz_west = measurement['mod.flo.temAirWes.T'].values[0] - 273.15
        Toa = measurement['mod.TOut.y'].values[0] - 273.15
        P = abs(measurement['mod.eleChi.y'].values[0]) + \
            abs(measurement['mod.eleCHWP.y'].values[0]) + \
            abs(measurement['mod.eleCT.y'].values[0]) + \
            abs(measurement['mod.eleCWP.y'].values[0]) + \
            abs(measurement['mod.eleSupFan.y'].values[0])
        
        # Update states
        states['Tz_core_his_meas'] = FILO(states['Tz_core_his_meas'], Tz_core)
        states['Tz_east_his_meas'] = FILO(states['Tz_east_his_meas'], Tz_east)
        states['Tz_north_his_meas'] = FILO(states['Tz_north_his_meas'], Tz_north)
        states['Tz_south_his_meas'] = FILO(states['Tz_south_his_meas'], Tz_south)
        states['Tz_west_his_meas'] = FILO(states['Tz_west_his_meas'], Tz_west)
        states['To_his_meas'] = FILO(states['To_his_meas'], Toa)
        states['P_his_meas'] = FILO(states['P_his_meas'], P)
        states['Tz_core_his_pred'] = FILO(states['Tz_core_his_pred'], Tz_pred['core'])
        states['Tz_east_his_pred'] = FILO(states['Tz_east_his_pred'], Tz_pred['east'])
        states['Tz_north_his_pred'] = FILO(states['Tz_north_his_pred'], Tz_pred['north'])
        states['Tz_south_his_pred'] = FILO(states['Tz_south_his_pred'], Tz_pred['south'])
        states['Tz_west_his_pred'] = FILO(states['Tz_west_his_pred'], Tz_pred['west'])
        
        return states
    
    def _calculate_comfort_violation(self, zone_temps: Dict[str, float], dt: float) -> float:
        """Calculate comfort violation in degree-hours"""
        T_upper = self.config['building_a']['comfort']['T_upper']
        T_lower = self.config['building_a']['comfort']['T_lower']
        violation = 0.0
        
        for temp in zone_temps.values():
            violation += max(0, temp - T_upper) + max(0, T_lower - temp)
        
        return violation * dt / 3600.0
    
    def get_power_forecast(self, horizon: int) -> List[float]:
        """Get power forecast (simplified)"""
        baseline = self.config['building_a']['baseline_power_kW']
        return [baseline] * horizon
    
    def shutdown(self):
        """Clean shutdown"""
        logger.info("🛑 Shutting down Building A")
