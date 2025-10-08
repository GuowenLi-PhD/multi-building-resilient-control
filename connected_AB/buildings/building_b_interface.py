"""
Building B Interface - Wrapper for Building B MPC (with TES)

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

# Add Building B path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../buildingB_w_TES'))

from pyfmi import load_fmu
from mpc_dnn import mpc_case
from buildings.base_building import BaseBuilding
from communication.data_models import (
    BuildingState, BuildingBState, AggregatorCommand,
    BuildingStatus, ControlMode
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BuildingBInterface(BaseBuilding):
    """Interface for Building B (with TES, ANN-based MPC)"""
    
    def __init__(self, config: Dict):
        super().__init__("Building_B", config)
        
        self.PH = config['timing']['prediction_horizon_building_b']
        self.dt = config['timing']['building_b_timestep']
        self.number_zones = 5
        
        # TES parameters
        self.mIce_max = 3105.0 * 5.0  # kg
        self.SOC_current = 0.5
        self.SOC_target = 0.5
        
        # Control mode
        self.control_mode = ControlMode.NOMINAL
        
        # FMU and MPC
        self.fmu = None
        self.mpc = None
        
        # Historical states
        self.states = None
        self.predictor = None
        
    def initialize(self, initial_conditions: Dict):
        """Initialize Building B FMU and MPC"""
        
        logger.info("🏗️ Initializing Building B...")
        
        # Set Dymola license
        if "DYMOLA_RUNTIME_LICENSE" not in os.environ:
            os.environ["DYMOLA_RUNTIME_LICENSE"] = "c:/programdata/dassaultsystemes/dymola/dymola.lic"
        
        # Load FMU
        fmu_path = os.path.join(
            os.path.dirname(__file__),
            '../../buildingB_w_TES/modelica_model/VirtualTestbed_NISTChillerTestbed_DemandFlexibilityInvestigation_FakeSystem_SystemForMPC_01bInput_0modeSignal.fmu'
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
            dt=self.dt,
            measurement={},
            states=self.states,
            predictor=self.predictor
        )
        
        self.current_time = t_start
        
        logger.info(f"✓ Building B initialized: SOC={self.SOC_current:.2f}, TES={self.mIce_max:.0f}kg")
    
    def step(self, aggregator_command: Optional[AggregatorCommand], dt: float) -> BuildingBState:
        """Execute one control step for Building B"""
        
        # Update SOC target from aggregator
        if aggregator_command and 'SOC_target' in aggregator_command.guidance:
            self.SOC_target = aggregator_command.guidance['SOC_target']
        
        # Modify MPC weights based on priority
        if aggregator_command and 'priority' in aggregator_command.guidance:
            priority = aggregator_command.guidance['priority']
            
            if priority == 'precharge':
                # Encourage charging
                self.mpc.w = [0.5, 1000., 2000.]  # Reduce energy cost weight, increase SOC penalty
                logger.info("🔋 Building B: Pre-charging mode activated")
                
            elif priority == 'support_A':
                # Encourage TES discharge to reduce power
                self.mpc.w = [0.1, 1000., 500.]  # Minimal energy cost, allow SOC decrease
                logger.info("🤝 Building B: Supporting Building A (discharging TES)")
                
            else:
                # Normal balanced operation
                self.mpc.w = [1., 1000., 1000.]
        
        # Run MPC optimization
        self.mpc.set_time(self.current_time)
        self.mpc.set_states(self.states)
        self.mpc.set_predictor(self.predictor)
        
        res, solver_status = self.mpc.optimize()
        
        if solver_status['return_status'] == 'INFEASIBLE':
            uMPC = [1]  # Fallback: discharge TES
            logger.warning("⚠️ Building B MPC infeasible, using fallback control")
        else:
            u_opt = res['x']
            uMPC = [int(u_opt[0])]  # Mode: -1, 0, 1, or 2
        
        # Apply control to FMU
        self.fmu.set('uMod', uMPC[0])
        
        # Simulate FMU
        ts = self.current_time
        te = ts + dt
        fmu_result = self.fmu.simulate(start_time=ts, final_time=te, options=self.fmu_options)
        
        # Extract measurement
        measurement = self._extract_measurement(fmu_result)
        
        # Update SOC
        self.SOC_current = measurement['iceTan.SOC'].values[0]
        
        # Get predictions
        Tz_pred = {
            'core': float(self.mpc.get_core_temp_pred(uMPC)),
            'east': float(self.mpc.get_east_temp_pred(uMPC)),
            'north': float(self.mpc.get_north_temp_pred(uMPC)),
            'south': float(self.mpc.get_south_temp_pred(uMPC)),
            'west': float(self.mpc.get_west_temp_pred(uMPC))
        }
        
        # Update states
        self.states = self._update_states(self.states, measurement, Tz_pred)
        
        # Update time
        self.current_time = te
        self.fmu_options['initialize'] = False
        
        # Calculate power
        power_actual = max(measurement['chi.P'].values[0], 0) + \
                      max(measurement['priPum.P'].values[0], 0) + \
                      max(measurement['secPum.P'].values[0], 0) + \
                      max(measurement['fanSup.P'].values[0], 0)
        
        # Zone temperatures
        zone_temps = {
            'core': measurement['conVAVCor.TZon'].values[0] - 273.15,
            'east': measurement['conVAVEas.TZon'].values[0] - 273.15,
            'north': measurement['conVAVNor.TZon'].values[0] - 273.15,
            'south': measurement['conVAVSou.TZon'].values[0] - 273.15,
            'west': measurement['conVAVWes.TZon'].values[0] - 273.15
        }
        
        # Create state object
        state = BuildingBState(
            building_id="Building_B",
            timestamp=self.current_time,
            status=BuildingStatus.NORMAL,
            control_mode=self.control_mode,
            power_actual_kW=power_actual / 1000.0,
            power_forecast_kW=self.get_power_forecast(self.PH),
            zone_temperatures=zone_temps,
            comfort_violations=self._calculate_comfort_violation(zone_temps)
        )
        
        # Update extra data
        state.extra_data['SOC_current'] = self.SOC_current
        state.extra_data['SOC_forecast'] = self.get_SOC_forecast(self.PH)
        state.extra_data['TES_mode'] = uMPC[0]
        state.extra_data['flexibility_up_kW'] = self._calculate_flexibility_up()
        state.extra_data['flexibility_down_kW'] = self._calculate_flexibility_down()
        state.extra_data['TES_available_energy_kWh'] = self.SOC_current * 1152.0  # Total TES capacity
        
        self.current_state = state
        return state
    
    def get_power_forecast(self, horizon: int) -> List[float]:
        """Get power forecast"""
        forecast = []
        for k in range(min(horizon, self.PH)):
            # Estimate based on current mode
            P_est = 12.0  # Baseline kW
            forecast.append(P_est)
        return forecast
    
    def get_SOC_forecast(self, horizon: int) -> List[float]:
        """Get SOC forecast"""
        # Simplified linear projection
        return [self.SOC_current] * horizon
    
    def _calculate_flexibility_up(self) -> float:
        """Calculate upward flexibility (kW)"""
        # Mode 2 (chiller) vs Mode 1 (TES discharge)
        return 5.0  # Simplified
    
    def _calculate_flexibility_down(self) -> float:
        """Calculate downward flexibility (kW)"""
        # Charging capacity
        if self.SOC_current < 0.9:
            return 10.0
        return 0.0
    
    def _extract_measurement(self, fmu_result) -> pd.DataFrame:
        """Extract measurement from FMU"""
        measurement_names = [
            'time', 'TOut.y', 'weaBus.HGloHor', 'uMod', 'iceTan.SOC',
            'conVAVCor.TZon', 'conVAVEas.TZon', 'conVAVNor.TZon',
            'conVAVSou.TZon', 'conVAVWes.TZon', 'ave.y',
            'chi.P', 'priPum.P', 'secPum.P', 'fanSup.P'
        ]
        
        dic = {}
        for name in measurement_names:
                    dic[name] = fmu_result[name][-1]
                
        return pd.DataFrame(dic, index=[dic['time']])
    
    def _update_states(self, states, measurement, Tz_pred):
        """Update MPC states using FILO (First In Last Out)"""
        
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
    
    def _calculate_comfort_violation(self, zone_temps: Dict[str, float]) -> float:
        """Calculate comfort violation in degree-hours"""
        T_upper = 25.0
        T_lower = 20.0
        violation = 0.0
        
        for temp in zone_temps.values():
            violation += max(0, temp - T_upper) + max(0, T_lower - temp)
        
        return violation * self.dt / 3600.0
    
    def shutdown(self):
        """Clean shutdown"""
        logger.info("🛑 Shutting down Building B")