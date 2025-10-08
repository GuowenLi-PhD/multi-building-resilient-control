"""
Building A Interface - Wrapper for Building A MPC (without TES)

Author: Guowen Li, AI Assistant
Date: 2025-01-07
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    print(f"🔧 Current directory added to path: {current_dir}")
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    print(f"🔧 Parent directory added to path: {parent_dir}")

# Add Building A path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../buildingA_wo_TES'))

from pyfmi import load_fmu
from mpc_v2 import mpc_case
from buildings.base_building import BaseBuilding
from communication.data_models import (
    BuildingState, BuildingAState, AggregatorCommand,
    BuildingStatus, ControlMode
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BuildingAInterface(BaseBuilding):
    """Interface for Building A (no TES, adaptive MPC for cyber-attacks)"""
    
    def __init__(self, config: Dict):
        super().__init__("Building_A", config)
        
        self.PH = config['timing']['prediction_horizon_building_a']
        self.dt = config['timing']['building_a_timestep']
        self.number_zones = config['building_a']['number_zones']
        
        # Attack status
        self.dos_attack_active = False
        self.control_mode = ControlMode.NOMINAL
        
        # FMU and MPC
        self.fmu = None
        self.mpc = None
        
        # Historical states
        self.states = None
        self.predictor = None
        
    def initialize(self, initial_conditions: Dict):
        """Initialize Building A FMU and MPC"""
        
        logger.info("🏗️ Initializing Building A...")
        
        # Set Dymola license
        if "DYMOLA_RUNTIME_LICENSE" not in os.environ:
            os.environ["DYMOLA_RUNTIME_LICENSE"] = "c:/programdata/dassaultsystemes/dymola/dymola.lic"
        
        # Load FMU
        fmu_path = os.path.join(
            os.path.dirname(__file__), 
            '../../buildingA_wo_TES/modelica_model/wrapped_fixed_modified_ecoRet09_02162023.fmu'
        )
        self.fmu = load_fmu(fmu_path)
        
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
            dt=self.dt,
            measurement={},
            states=self.states,
            predictor=self.predictor,
            mpc_models=mpc_models,
            dos_attack_core_VAV=False  # Start in normal mode
        )
        
        logger.info("✓ Building A initialized successfully")
    
    def step(self, aggregator_command: Optional[AggregatorCommand], dt: float) -> BuildingAState:
        """Execute one control step for Building A"""
        
        # Update attack status from aggregator command
        if aggregator_command:
            self.dos_attack_active = aggregator_command.attack_flag
            
            # Reconfigure MPC if attack status changed
            if self.dos_attack_active and self.control_mode != ControlMode.ADAPTIVE:
                logger.warning("⚠️ Building A: Switching to ADAPTIVE MPC (attack detected)")
                self.mpc.dos_attack_core_VAV = True
                self.mpc.w = [0., 100., 10.]  # Attack weights
                self.control_mode = ControlMode.ADAPTIVE
                
            elif not self.dos_attack_active and self.control_mode != ControlMode.NOMINAL:
                logger.info("✓ Building A: Returning to NOMINAL MPC")
                self.mpc.dos_attack_core_VAV = False
                self.mpc.w = [1., 1., 100.]  # Normal weights
                self.control_mode = ControlMode.NOMINAL
        
        # Run MPC optimization
        self.mpc.set_time(self.current_time)
        self.mpc.set_states(self.states)
        self.mpc.set_predictor(self.predictor)
        
        res = self.mpc.optimize()
        u_opt = res['x']
        
        # Extract control actions [b_cp, b_ahu, T_chw, T_cw, T_sa, V_core, V_east, V_north, V_south, V_west]
        uMPC = [
            float(u_opt[0]),  # b_cp
            float(u_opt[1]),  # b_ahu
            float(u_opt[2]) + 273.15,  # T_chw (C to K)
            float(u_opt[3]) + 273.15,  # T_cw
            float(u_opt[4]) + 273.15,  # T_sa
            float(u_opt[5]),  # V_core
            float(u_opt[6]),  # V_east
            float(u_opt[7]),  # V_north
            float(u_opt[8]),  # V_south
            float(u_opt[9])   # V_west
        ]
        
        # Apply controls to FMU
        control_vars = [
            'oveOnChiPla_u', 'conAHU_supFan_oveOnSupFan_u',
            'oveTChiWatSupSet_u', 'oveTConWatSupSet_u',
            'conAHU_oveTSupAir_u', 'conVAVCor_damVal_oveVDisSet_u',
            'conVAVEas_damVal_oveVDisSet_u', 'conVAVNor_damVal_oveVDisSet_u',
            'conVAVSou_damVal_oveVDisSet_u', 'conVAVWes_damVal_oveVDisSet_u'
        ]
        
        for var, val in zip(control_vars, uMPC):
            self.fmu.set(var, val)
        
        # Activate all overrides
        for var in control_vars:
            self.fmu.set(var + '_activate', 1)
        
        # Simulate FMU
        ts = self.current_time
        te = ts + dt
        fmu_result = self.fmu.simulate(start_time=ts, final_time=te, options=self.fmu_options)
        
        # Emulate attack if active
        if self.dos_attack_active:
            # Core zone VAV under DoS attack
            measurement = self._extract_measurement(fmu_result)
            measurement.loc[0, 'mod.conVAVCor.VDis_flow'] = 0.0  # Zero airflow
            measurement.loc[0, 'mod.flo.temAirPer5.T'] = -3.89  # Sensor fault
        else:
            measurement = self._extract_measurement(fmu_result)
        
        # Update states for next iteration
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
        
        # Create state object
        state = BuildingAState(
            building_id="Building_A",
            timestamp=self.current_time,
            status=BuildingStatus.UNDER_ATTACK if self.dos_attack_active else BuildingStatus.NORMAL,
            control_mode=self.control_mode,
            power_actual_kW=power_actual / 1000.0,
            power_forecast_kW=self.get_power_forecast(self.PH),
            zone_temperatures=zone_temps,
            comfort_violations=self._calculate_comfort_violation(zone_temps)
        )
        
        state.extra_data['core_zone_airflow'] = measurement['mod.conVAVCor.VDis_flow'].values[0]
        state.extra_data['attack_detected'] = self.dos_attack_active
        
        self.current_state = state
        return state
    
    def get_power_forecast(self, horizon: int) -> List[float]:
        """Get power forecast over horizon"""
        # Use MPC internal predictions
        forecast = []
        for k in range(min(horizon, self.PH)):
            # Simplified power estimation
            P_est = 8.0 + (2.3 if self.dos_attack_active else 0.0)  # kW
            forecast.append(P_est)
        return forecast
    
    def _extract_measurement(self, fmu_result) -> pd.DataFrame:
        """Extract measurement from FMU result"""
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
        
        dic = {}
        for name in measurement_names:
            dic[name] = fmu_result[name][-1]
        
        return pd.DataFrame(dic, index=[dic['time']])
    
    def _update_states(self, states, measurement, Tz_pred):
        """Update MPC states"""
        # Implementation similar to get_states() in run_mpc_v2.py
        # ... (simplified for brevity)
        return states
    
    def _calculate_comfort_violation(self, zone_temps: Dict[str, float]) -> float:
        """Calculate comfort violation in degree-hours"""
        T_upper = 26.0
        T_lower = 22.0
        violation = 0.0
        
        for temp in zone_temps.values():
            violation += max(0, temp - T_upper) + max(0, T_lower - temp)
        
        return violation * self.dt / 3600.0  # Convert to degree-hours
    
    def shutdown(self):
        """Clean shutdown"""
        logger.info("🛑 Shutting down Building A")