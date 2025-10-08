"""
This script is used to test the baseline and adaptive MPC.

Author: Guowen Li, Yangyang Fu
Email: guowenli@tamu.edu, yangyang.fu@tamu.edu 
Revisions:
    2023: Implement adaptive MPC for Device Reinitialization Attack on Core zone's VAV box
"""

# load testbed
from pyfmi import load_fmu
# import others
import numpy as np
import pandas as pd
import matplotlib
# for ipython plot
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import os
import json

from sympy import maximum
# load MPC
from mpc import mpc_case

### ======
####  Define FMU model path 
### ======================================
# MODELICAPATH=os.getenv('MODELICAPATH')
# MODELICAPATH=MODELICAPATH.split(';')[0]
# LIBRARYPATH=os.path.join(MODELICAPATH,'modelica-template','dymola')
# print(MODELICAPATH,LIBRARYPATH)

## Put the terminal prints into .txt file
# class Logger(object):
#     def __init__(self, filename="Default.log"):
#         self.terminal = sys.stdout
#         self.log = open(filename, "a")

#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)

#     def flush(self):
#         pass

# path = os.path.abspath(os.path.dirname(__file__))
# type = sys.getfilesystemencoding()
# sys.stdout = Logger('CYDRES_MPC_Terminal_Log.txt')
# print(path)

def get_measurement(fmu_result, names):
    if 'time' not in names:
        names.append('time')

    dic = {}
    for name in names:
        dic[name] = fmu_result[name][-1]

    # return a pandas data frame
    return pd.DataFrame(dic, index=[dic['time']])

def interpolate_dataframe(df, new_index):
    """Interpolate a dataframe along its index based on a new index
    """
    df_out = pd.DataFrame(index=new_index)
    df_out.index.name = df.index.name

    for col_name, col in df.items():
        df_out[col_name] = np.interp(new_index, df.index, col)
    return df_out

def LIFO(a_list,x):
    """Last in First out: 
    x: scalor
    """
    a_list.reverse()
    a_list.pop()
    a_list.reverse()
    a_list.append(x)

    return a_list

def get_states(states, measurement, Tz_pred):
    """Update current states using measurement data
    """
    # read list
    Tz_core_his = states['Tz_core_his_meas']
    Tz_east_his = states['Tz_east_his_meas']
    Tz_north_his = states['Tz_north_his_meas']
    Tz_south_his = states['Tz_south_his_meas']
    Tz_west_his = states['Tz_west_his_meas']
    Toa_his = states['To_his_meas']
    P_his = states['P_his_meas']
    Tz_core_his_pred = states['Tz_core_his_pred']
    Tz_east_his_pred = states['Tz_east_his_pred']
    Tz_north_his_pred = states['Tz_north_his_pred']
    Tz_south_his_pred = states['Tz_south_his_pred']
    Tz_west_his_pred = states['Tz_west_his_pred']

    # read scalor
    Tz_core = measurement['mod.flo.temAirPer5.T'].values[0]  # K
    Tz_east = measurement['mod.flo.temAirEas.T'].values[0]  # K
    Tz_north = measurement['mod.flo.temAirNor.T'].values[0]  # K
    Tz_south = measurement['mod.flo.temAirSou.T'].values[0]  # K
    Tz_west = measurement['mod.flo.temAirWes.T'].values[0]  # K
    Toa = measurement['mod.TOut.y'].values[0] - 273.15  # K to C
    P = abs(measurement['mod.eleChi.y'].values[0])+abs(measurement['mod.eleCHWP.y'].values[0])+abs(
        measurement['mod.eleCT.y'].values[0])+abs(measurement['mod.eleCWP.y'].values[0])+abs(measurement['mod.eleSupFan.y'].values[0])
    Tz_core_pred = Tz_pred['core']
    Tz_east_pred = Tz_pred['east']
    Tz_north_pred = Tz_pred['north']    
    Tz_south_pred = Tz_pred['south']
    Tz_west_pred = Tz_pred['west']    

    # new dic
    states['Tz_core_his_meas'] = LIFO(Tz_core_his,Tz_core)
    states['Tz_east_his_meas'] = LIFO(Tz_east_his,Tz_east)
    states['Tz_north_his_meas'] = LIFO(Tz_north_his,Tz_north)
    states['Tz_south_his_meas'] = LIFO(Tz_south_his,Tz_south)
    states['Tz_west_his_meas'] = LIFO(Tz_west_his,Tz_west)
    states['To_his_meas'] = LIFO(Toa_his,Toa)
    states['P_his_meas'] = LIFO(P_his, P)
    states['Tz_core_his_pred'] = LIFO(Tz_core_his_pred, Tz_core_pred)
    states['Tz_east_his_pred'] = LIFO(Tz_east_his_pred, Tz_east_pred)
    states['Tz_north_his_pred'] = LIFO(Tz_north_his_pred, Tz_north_pred)
    states['Tz_south_his_pred'] = LIFO(Tz_south_his_pred, Tz_south_pred)
    states['Tz_west_his_pred'] = LIFO(Tz_west_his_pred, Tz_west_pred)
    print("\n states_att:",states)
    
    return states

def get_states_baseline(states, measurement, Tz_pred, Tub=273.15+26, Tlb=273.15+22, Tub_pre=273.15+30, Tlb_pre=273.15+18):
    """Update current baseline states using measurement data
    """
    # read list
    Toa_his = states['To_his_meas']
    P_his = states['P_his_meas']
    L = len(states['Tz_core_his_meas'])
    Tz_core_his,Tz_east_his,Tz_north_his,Tz_south_his,Tz_west_his = [273.15+24]*L,[273.15+24]*L,[273.15+24]*L,[273.15+24]*L,[273.15+24]*L
    Tz_core_his_pred,Tz_east_his_pred,Tz_north_his_pred,Tz_south_his_pred,Tz_west_his_pred = [273.15+24]*L,[273.15+24]*L,[273.15+24]*L,[273.15+24]*L,[273.15+24]*L
    for i in range(L):
        Tz_core_his[i] = np.minimum(np.maximum(states['Tz_core_his_meas'][i],Tlb),Tub)
        Tz_east_his[i] = np.minimum(np.maximum(states['Tz_east_his_meas'][i],Tlb),Tub)
        Tz_north_his[i] = np.minimum(np.maximum(states['Tz_north_his_meas'][i],Tlb),Tub)
        Tz_south_his[i] = np.minimum(np.maximum(states['Tz_south_his_meas'][i],Tlb),Tub)
        Tz_west_his[i] = np.minimum(np.maximum(states['Tz_west_his_meas'][i],Tlb),Tub)
        Tz_core_his_pred[i] = np.minimum(np.maximum(states['Tz_core_his_pred'][i],Tlb_pre),Tub_pre)
        Tz_east_his_pred[i] = np.minimum(np.maximum(states['Tz_east_his_pred'][i],Tlb_pre),Tub_pre)
        Tz_north_his_pred[i] = np.minimum(np.maximum(states['Tz_north_his_pred'][i],Tlb_pre),Tub_pre)
        Tz_south_his_pred[i] = np.minimum(np.maximum(states['Tz_south_his_pred'][i],Tlb_pre),Tub_pre)
        Tz_west_his_pred[i] = np.minimum(np.maximum(states['Tz_west_his_pred'][i],Tlb_pre),Tub_pre)

    # read scalor
    Tz_core = np.minimum(np.maximum(measurement['mod.flo.temAirPer5.T'].values[0],Tlb),Tub)  # K
    Tz_east = np.minimum(np.maximum(measurement['mod.flo.temAirEas.T'].values[0],Tlb),Tub)  # K
    Tz_north = np.minimum(np.maximum(measurement['mod.flo.temAirNor.T'].values[0],Tlb),Tub)  # K
    Tz_south = np.minimum(np.maximum(measurement['mod.flo.temAirSou.T'].values[0],Tlb),Tub)  # K
    Tz_west = np.minimum(np.maximum(measurement['mod.flo.temAirWes.T'].values[0],Tlb),Tub)  # K
    Toa = measurement['mod.TOut.y'].values[0] - 273.15  # K to C
    P = abs(measurement['mod.eleChi.y'].values[0])+abs(measurement['mod.eleCHWP.y'].values[0])+abs(
        measurement['mod.eleCT.y'].values[0])+abs(measurement['mod.eleCWP.y'].values[0])+abs(measurement['mod.eleSupFan.y'].values[0])
    Tz_core_pred = np.minimum(np.maximum(Tz_pred['core'],Tlb_pre),Tub_pre)
    Tz_east_pred = np.minimum(np.maximum(Tz_pred['east'],Tlb_pre),Tub_pre)
    Tz_north_pred = np.minimum(np.maximum(Tz_pred['north'],Tlb_pre),Tub_pre) 
    Tz_south_pred = np.minimum(np.maximum(Tz_pred['south'],Tlb_pre),Tub_pre)
    Tz_west_pred = np.minimum(np.maximum(Tz_pred['west'],Tlb_pre),Tub_pre) 
    
    states_bas = {'Tz_core_his_meas': LIFO(Tz_core_his,Tz_core),  # Unit: Kelvin
                'Tz_east_his_meas': LIFO(Tz_east_his,Tz_east),
                'Tz_north_his_meas': LIFO(Tz_north_his,Tz_north),
                'Tz_south_his_meas': LIFO(Tz_south_his,Tz_south),
                'Tz_west_his_meas': LIFO(Tz_west_his,Tz_west),
                'To_his_meas': LIFO(Toa_his,Toa),  # Unit: Celsius
                'P_his_meas': LIFO(P_his, P),
                'Tz_core_his_pred': LIFO(Tz_core_his_pred, Tz_core_pred),  # Unit: K
                'Tz_east_his_pred': LIFO(Tz_east_his_pred, Tz_east_pred),
                'Tz_north_his_pred': LIFO(Tz_north_his_pred, Tz_north_pred),
                'Tz_south_his_pred': LIFO(Tz_south_his_pred, Tz_south_pred),
                'Tz_west_his_pred': LIFO(Tz_west_his_pred, Tz_west_pred)
                }
    print("\n states_bas:",states_bas)

    return states_bas

def get_price(time, dt, PH):
    # unite - $/kwh
    price_tou = [0.0640, 0.0640, 0.0640, 0.0640,
                 0.0640, 0.0640, 0.0640, 0.0640,
                 0.1391, 0.1391, 0.1391, 0.1391,
                 0.3548, 0.3548, 0.3548, 0.3548,
                 0.3548, 0.3548, 0.1391, 0.1391,
                 0.1391, 0.1391, 0.1391, 0.0640]
    #- assume hourly TOU pricing
    t_ph = np.arange(time, time+dt*PH, dt)
    price_ph = [price_tou[int(t % 86400 / 3600)] for t in t_ph]

    return price_ph

def read_temperature(weather_file, dt):
    """Read temperature and solar radiance from epw file. 
        This module serves as an ideal weather predictor.
    :return: a data frame at an interval of defined time_step
    """
    from pvlib.iotools import read_epw

    dat = read_epw(weather_file)

    tem_sol_h = dat[0][['temp_air']]  # celsius
    index_h = np.arange(3600, 3600.*(len(tem_sol_h)+1), 3600.)
    tem_sol_h.index = index_h

    # interpolate temperature into simulation steps
    index_step = np.arange(3600, 3600.*(len(tem_sol_h)+1), dt)

    return interpolate_dataframe(tem_sol_h, index_step)

def get_Toa(time, dt, PH, Toa_year):
    index_ph = np.arange(time, time+dt*PH, dt)
    Toa = Toa_year.loc[index_ph, :]

    return list(Toa.values.flatten())

def read_RH(weather_file, dt):  # Relative Humidity
    """Read Relative Humidity from epw file. 
        This module serves as an ideal weather predictor.
    :return: a data frame at an interval of defined time_step
    """
    from pvlib.iotools import read_epw

    dat = read_epw(weather_file)

    RH_h = dat[0][['relative_humidity']]*0.01  # convert to 100%
    index_h = np.arange(3600, 3600.*(len(RH_h)+1), 3600.)
    RH_h.index = index_h

    # interpolate relative humidity into simulation steps
    index_step = np.arange(3600, 3600.*(len(RH_h)+1), dt)

    return interpolate_dataframe(RH_h, index_step)

def get_RHoa(time, dt, PH, RHoa_year):
    index_ph = np.arange(time, time+dt*PH, dt)
    RHoa = RHoa_year.loc[index_ph, :]

    return list(RHoa.values.flatten())

def module3_impact_analysis(ts, case, u_opt_ph, w1=0.25, w2=0.25, w3=0.25, w4=0.25):
    '''Mode selection through  the Imapct Analysis (Module 3 in CYDRES)
       KPI based on future-2-hour data
       w1, w2, w3, w4 weight factors for 
       Energy Efficiency Cost, Peak Power Cost, Total Discomfort Cost, Demand Flexibility Cost
    :return: a float number of impact score
    '''
    ## temperature bounds (degree hour)
    T_upper = 26 if (ts%86400)/3600 >= 7 and (ts%86400)/3600 < 19 else 30
    T_lower = 22 if (ts%86400)/3600 >= 7 and (ts%86400)/3600 < 19 else 18
    P_nominal = 20000. # nominal total power consumption, need to be confirmed
    metric_Power, kpi_efficiency, kpi_Temp, kpi_DF = 0., 0., 0., 0.
    kpi_power = np.array([0]*PH)

    for i in range(PH):
        Tz_core_mpc = float(case.get_core_temp_pred(u_opt_ph[i*11:i*11+10], case._autoerror['core']))
        Tz_east_mpc = float(case.get_east_temp_pred(u_opt_ph[i*11:i*11+10], case._autoerror['east']))
        Tz_north_mpc = float(case.get_north_temp_pred(u_opt_ph[i*11:i*11+10], case._autoerror['north']))
        Tz_south_mpc = float(case.get_south_temp_pred(u_opt_ph[i*11:i*11+10], case._autoerror['south']))
        Tz_west_mpc = float(case.get_west_temp_pred(u_opt_ph[i*11:i*11+10], case._autoerror['west']))
        P_mpc = float(case.get_power_pred(u_opt_ph[i*11:i*11+10]))
        print("\n MPC Predicted Temperature: [Tz_core_mpc,Tz_east_mpc,Tz_north_mpc,Tz_south_mpc,Tz_west_mpc]\n",\
                Tz_core_mpc,Tz_east_mpc,Tz_north_mpc,Tz_south_mpc,Tz_west_mpc)

        ## KPI Calculator
        # KPI 1: Energy Efficiency Cost
        metric_Power += np.maximum(0.,P_mpc-P_nominal)
        kpi_efficiency += P_mpc/(P_nominal*PH)

        # KPI 2: Peak Power Cost
        kpi_power[i] = P_mpc

        # KPI 3: Total Discomfort Cost 
        T_tot = 2. # tolerant of zone temperature violation                
        kpi_Temp += np.maximum(T_lower-Tz_core_mpc,np.maximum(0.,Tz_core_mpc-T_upper))/(T_tot*PH)
        kpi_Temp += np.maximum(T_lower-Tz_east_mpc,np.maximum(0.,Tz_east_mpc-T_upper))/(T_tot*PH)
        kpi_Temp += np.maximum(T_lower-Tz_north_mpc,np.maximum(0.,Tz_north_mpc-T_upper))/(T_tot*PH)
        kpi_Temp += np.maximum(T_lower-Tz_south_mpc,np.maximum(0.,Tz_south_mpc-T_upper))/(T_tot*PH)
        kpi_Temp += np.maximum(T_lower-Tz_west_mpc,np.maximum(0.,Tz_west_mpc-T_upper))/(T_tot*PH)

        # KPI 4: Demand Flexibility (DF) Cost, need to model a DF predictor
        P_ref = P_mpc # reference power trajectory
        P_upward = float(case.upward_DF(u_opt_ph[i*11:i*11+10])) # upward flexibility
        P_downward = float(case.downward_DF(u_opt_ph[i*11:i*11+10])) # downward flexibility
        print("\n MPC Predicted Power: [P_mpc, P_upward, P_downward]\n",\
                P_mpc, P_upward, P_downward)
        kpi_DF += (1 - (abs(P_ref-P_upward) + abs(P_downward-P_ref))/P_nominal)/PH

    kpi_peak_power = kpi_power.max()/P_nominal
    print("\n kpi_efficiency, kpi_peak_power, kpi_Temp, kpi_DF: ", kpi_efficiency, kpi_peak_power, kpi_Temp, kpi_DF)
    # Weighted total cost
    kpi_tot = w1*kpi_efficiency + w2*kpi_peak_power + w3*kpi_Temp + w4*kpi_DF
    
    return kpi_tot

### Load FMU
## ============================================================

# set attack type
dos_attack_core_VAV = True # True: DoS attack on core zone VAV box; False: No attack

# Set the environment variable for the Dymola license
if "DYMOLA_RUNTIME_LICENSE" not in os.environ:
    os.environ["DYMOLA_RUNTIME_LICENSE"] = "c:/programdata/dassaultsystemes/dymola/dymola.lic"
    
# load Modelica model - VAV system virtual teatbed
#hvac = load_fmu(os.path.join(LIBRARYPATH,"wrapped_fixed_modified_si2.fmu"))
hvac = load_fmu("modelica_model/wrapped_fixed_modified_ecoRet09_02162023.fmu")
# fmu settings
options = hvac.simulate_options()
options['ncp'] = 100 # ???
options['initialize'] = True
measurement_names = ['time',
                     'mod.TCHWSup.T',
                     'mod.TCWSup.T',
                     'mod.TSup.T',
                     'mod.conVAVSou.VDis_flow',
                     'mod.conVAVEas.VDis_flow',
                     'mod.conVAVNor.VDis_flow',
                     'mod.conVAVWes.VDis_flow',
                     'mod.conVAVCor.VDis_flow',
                     'mod.fanSup.VMachine_flow',
                     'mod.flo.temAirSou.T',
                     'mod.flo.temAirEas.T',
                     'mod.flo.temAirNor.T',
                     'mod.flo.temAirWes.T',
                     'mod.flo.temAirPer5.T',
                     'mod.eleCoiVAV.y',
                     'mod.eleSupFan.y',
                     'mod.eleChi.y',
                     'mod.eleCHWP.y',
                     'mod.eleCWP.y',
                     'mod.eleCT.y',
                     'mod.eleHWP.y',
                     'mod.TOut.y',
                     'oveTCooOn_p',
                     'oveTChiWatSupSet_u',
                     'oveTConWatSupSet_u',
                     'conAHU_oveTSupAir_u',
                     'conVAVSou_damVal_oveVDisSet_u',
                     'conVAVEas_damVal_oveVDisSet_u',
                     'conVAVNor_damVal_oveVDisSet_u',
                     'conVAVWes_damVal_oveVDisSet_u',
                     'conVAVCor_damVal_oveVDisSet_u',
                     'mod.conAHU.supFan.oveOnSupFan.y',
                     'mod.oveOnChiPla.y',
                     'mod.oveRelHumOutAir.y'
                     ]
#options['filter'] = measurement_names
# 5-minute

### 2. Experiment setup
## define simulation period
t_start = 212*24*3600  # simulation start time: 207*24*3600 issue day: 218 # three-month training: 7,8,9, start from Day 181 #August 1st is Day 212
t_period = 1*24*3600. #2*24*3600.
t_end = t_start + t_period  # simulation end time
te_warm = t_start + 4*3600.  # warm up time
dt = 15*60.  # MPC timestep
PH = 4 #1
CH = 1
number_inputs = 11  # 10 control varaibles + 1 slack variable

## predictors
predictor = {}
# energy prices
predictor['price'] = get_price(t_start, dt, PH)
# outdoor air temperature
weather_file = 'weather_data/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw'
Toa_year = read_temperature(weather_file, dt)
predictor['Toa'] = get_Toa(t_start+dt, dt, PH, Toa_year)
RHoa_year = read_RH(weather_file, dt)
predictor['RHoa'] = get_RHoa(t_start+dt, dt, PH, RHoa_year)

## initialize states
# historical Toa measurements
Toa_his_meas_ini = get_Toa(t_start-3*dt, dt, 4, Toa_year)
# states
states_ini = {'Tz_core_his_meas': [273.15+24]*4,  # Unit: Kelvin
              'Tz_east_his_meas': [273.15+24]*4,
              'Tz_north_his_meas': [273.15+24]*4,
              'Tz_south_his_meas': [273.15+24]*4,
              'Tz_west_his_meas': [273.15+24]*4,
              'To_his_meas': Toa_his_meas_ini,  # Unit: Celsius
              'P_his_meas': [0]*1,
              'Tz_core_his_pred': [273.15+24]*4,  # Unit: K
              'Tz_east_his_pred': [273.15+24]*4,
              'Tz_north_his_pred': [273.15+24]*4,
              'Tz_south_his_pred': [273.15+24]*4,
              'Tz_west_his_pred': [273.15+24]*4}  # initial states used for MPC models

# load MPC model coefficients
#MPCMODELPATH = os.path.join(MODELICAPATH, 'testcases', '3-system-identification')
MPCMODELPATH = "system_identification"
mpc_models = {}
mpc_models['fan'] = json.load(open(os.path.join(MPCMODELPATH, 'fan.json')))
mpc_models['fan_Tset22'] = json.load(open(os.path.join(MPCMODELPATH, 'fan_Tset22.json')))
mpc_models['fan_Tset26'] = json.load(open(os.path.join(MPCMODELPATH, 'fan_Tset26.json')))
mpc_models['chiller_plant'] = json.load(open(os.path.join(MPCMODELPATH, 'chiller_plant.json')))
mpc_models['chiller_plant_Tset22'] = json.load(open(os.path.join(MPCMODELPATH, 'chiller_plant_Tset22.json')))
mpc_models['chiller_plant_Tset26'] = json.load(open(os.path.join(MPCMODELPATH, 'chiller_plant_Tset26.json')))
mpc_models['core'] = json.load(open(os.path.join(MPCMODELPATH, 'TZone_Core.json')))
mpc_models['east'] = json.load(open(os.path.join(MPCMODELPATH, 'TZone_East.json')))
mpc_models['west'] = json.load(open(os.path.join(MPCMODELPATH, 'TZone_West.json')))
mpc_models['south'] = json.load(open(os.path.join(MPCMODELPATH, 'TZone_South.json')))
mpc_models['north'] = json.load(open(os.path.join(MPCMODELPATH, 'TZone_North.json')))

## initialize mpc case
measurement_ini = {}
measurement = measurement_ini

case = mpc_case(PH=PH,
                CH=CH,
                time=t_start,
                dt=dt,
                measurement=measurement_ini,
                states=states_ini,
                predictor=predictor,
                mpc_models = mpc_models,
                dos_attack_core_VAV = dos_attack_core_VAV,
                )

inputs = ['oveOnChiPla_u', 'oveOnChiPla_activate', 'conAHU_supFan_oveOnSupFan_u', 'conAHU_supFan_oveOnSupFan_activate',
               'oveTChiWatSupSet_u', 'oveTChiWatSupSet_activate', 'oveTConWatSupSet_u', 'oveTConWatSupSet_activate',
               'conAHU_oveTSupAir_u', 'conAHU_oveTSupAir_activate', 'conVAVCor_damVal_oveVDisSet_u', 'conVAVCor_damVal_oveVDisSet_activate',
               'conVAVEas_damVal_oveVDisSet_u', 'conVAVEas_damVal_oveVDisSet_activate', 'conVAVNor_damVal_oveVDisSet_u', 'conVAVNor_damVal_oveVDisSet_activate',
               'conVAVSou_damVal_oveVDisSet_u', 'conVAVSou_damVal_oveVDisSet_activate', 'conVAVWes_damVal_oveVDisSet_u', 'conVAVWes_damVal_oveVDisSet_activate']

#input_control=[bcp, bahu, Tchw, Tcw, Tsa, Vcore, Veast, Vnorth, Vsouth, Vwest, \epsilon] 
input_control = []
input_control_activate = []
for name in inputs:
    if name.endswith('_activate'):
        input_control_activate.append(name)
    else:
        input_control.append(name)

## initialize start time
ts = t_start
print("t_start:", t_start)
print("t_end:", t_end)

## initialize inputs
uMPC_ini = [0, 0, 273.15+10, 273.15+20, 273.15+20, 0, 0, 0, 0, 0, 0.1]
uMPC = uMPC_ini
states = states_ini
states_baseline = states_ini

## initialize outputs
t_opt = []
u_opt = []
Tz_core_pred_opt = []
Tz_east_pred_opt = []
Tz_north_pred_opt = []
Tz_south_pred_opt = []
Tz_west_pred_opt = []
P_pred_opt = []
warmup = True

## main loop
while ts < t_end:
    te = ts+dt*CH
    print("\n============================================================================================")
    print("ts:", ts)
    print("te:", te)
    t_opt.append(ts)
    print('Simulation time Clock (hour):', (ts % 86400)/3600, '~', ((te) % 86400)/3600)

    # generate control action from MPC
    if not warmup:  # activate mpc after warmup
        print("Prediction Horizon: ", PH, "timestep = ", PH/4, "hour")
        # update mpc case
        case.set_time(ts)
        case.set_measurement(measurement)
        case.set_states(states)
        print("\nstate 1")
        print(case.states)
        case.set_predictor(predictor)
        case.set_u_prev(u_opt_ch)

        # call optimizer
        res = case.optimize()
        u_opt_ph = res['x']
        
        # [bcp, bahu, Tchw, Tcw, Tsa, Vcore, Veast, Vnorth, Vsouth, Vwest, \epsilon]
        # uMPC = [res['x'][0], res['x'][1], res['x'][2]+273.15, res['x'][3]+273.15, res['x']
        #         [4]+273.15, res['x'][5], res['x'][6], res['x'][7], res['x'][8], res['x'][9]]
        uMPC = [res['x'][0], res['x'][1], res['x'][2]+273.15, res['x'][3]+273.15, res['x']
                [4]+273.15, res['x'][5], res['x'][6], res['x'][7], res['x'][8], 0.] # Attack one zone
        uMPC = [float(u) for u in uMPC] # convert DM object to float

        # get the control action for the control horizon
        # keep the same unit as mpc optimizer
        u_opt_ch = u_opt_ph[0:case.number_inputs]

        # update predictions after MPC predictor is called otherwise use measurement
        Tz_core_pred = float(case.get_core_temp_pred(u_opt_ph[:10], case._autoerror['core'])) + 273.15
        Tz_east_pred = float(case.get_east_temp_pred(u_opt_ph[:10], case._autoerror['east'])) + 273.15
        Tz_north_pred = float(case.get_north_temp_pred(u_opt_ph[:10], case._autoerror['north'])) + 273.15
        Tz_south_pred = float(case.get_south_temp_pred(u_opt_ph[:10], case._autoerror['south'])) + 273.15 
        Tz_west_pred = float(case.get_west_temp_pred(u_opt_ph[:10], case._autoerror['west'])) + 273.15
        P_pred = float(case.get_power_pred(u_opt_ph[:10]))

        print("------Start the Imapct Analysis (Module 3 in CYDRES)------")
        
        kpi_tot_att = module3_impact_analysis(ts, case, u_opt_ph)
        
        case_baseline = mpc_case(PH=PH,
                                CH=CH,
                                time=ts,
                                dt=dt,
                                measurement=measurement,
                                states=states_baseline,
                                predictor=predictor,
                                mpc_models = mpc_models,
                                )
        #case_baseline.set_states(states_baseline)
        kpi_tot_bas = module3_impact_analysis(ts, case_baseline, u_opt_ph)
        
        print(" KPI attack:",kpi_tot_att,"KPI baseline:",kpi_tot_bas)
        
        impact_score = (kpi_tot_att - kpi_tot_bas) / kpi_tot_bas
        if impact_score >= 0.5: # isolation threshold
            print("\nAlert! the impact score exceeds the safety threshold, current value is",impact_score,"\n")
        else:
            print("\nThe impact score is normal, current value is",impact_score,"\n")
        print("------------------Module 3 - Finished---------------------")
        ## Mode Selection (to be implemented)

        
        # update start points for optimizer using previous optimum value
        case.set_u_start(u_opt_ph)

    # advance building simulation by one step
    for activate in input_control_activate:
        hvac.set(activate,1)
    for i, control in zip(range(len(input_control)),input_control):
        hvac.set(control,uMPC[i])
    res = hvac.simulate(start_time=ts,
                        final_time=te,
                        options=options)
    # update clock
    ts = te

    # get measurement
    measurement = get_measurement(res, measurement_names)
    print("\nmeasurement normal:", measurement)
    if dos_attack_core_VAV:
        print("\n***Under DoS attack on core zone VAV box***")
        # emulate & mimic cyber-attacks on core zone VAV box
        measurement.loc[0,'mod.conVAVCor.VDis_flow'] = 0
        measurement.loc[0,'mod.flo.temAirPer5.T'] = -3.89 
        print("\nmeasurement after attack:", measurement)

    # update MPC model states
    # if not warmup then measurement else from mpc
    if warmup:
        Tz_core_pred = measurement['mod.flo.temAirPer5.T'].values[0]
        Tz_east_pred = measurement['mod.flo.temAirEas.T'].values[0]
        Tz_north_pred = measurement['mod.flo.temAirNor.T'].values[0]
        Tz_south_pred = measurement['mod.flo.temAirSou.T'].values[0]
        Tz_west_pred = measurement['mod.flo.temAirWes.T'].values[0]
        P_pred = abs(measurement['mod.eleChi.y'].values[0])+abs(measurement['mod.eleCHWP.y'].values[0])+abs(
            measurement['mod.eleCT.y'].values[0])+abs(measurement['mod.eleCWP.y'].values[0])+abs(measurement['mod.eleSupFan.y'].values[0])
        u_opt_ch = [0, 0, 10, 20, 20, 0, 0, 0, 0, 0, 0]# need has the same unit as mpc optimizer
    
    Tz_pred = {'core': Tz_core_pred,
        'east':Tz_east_pred,
        'north':Tz_north_pred,
        'south':Tz_south_pred,
        'west':Tz_west_pred}   
    states = get_states(states, measurement, Tz_pred)
    states_baseline = get_states_baseline(states_baseline, measurement, Tz_pred)
    print("\nstate 4")
    print(states)

    # online MPC model calibration if applied - NOT IMPLEMENTED
    # update parameter_zones and parameters_power - NOT IMPLEMENTED

    # update predictor
    predictor['price'] = get_price(ts, dt, PH)
    predictor['Toa'] = get_Toa(ts+dt, dt, PH, Toa_year)
    predictor['RHoa'] = get_RHoa(ts+dt, dt, PH, RHoa_year)

    # update fmu settings
    options['initialize'] = False

    # update warmup flag for next step
    warmup = ts<te_warm

    # save optimal results of the control horizon for future simulation
    u_opt.append(uMPC)
    Tz_core_pred_opt.append(Tz_core_pred)
    Tz_east_pred_opt.append(Tz_east_pred)
    Tz_north_pred_opt.append(Tz_north_pred)
    Tz_south_pred_opt.append(Tz_south_pred)
    Tz_west_pred_opt.append(Tz_west_pred)
    P_pred_opt.append(P_pred)

### 3. save results  
final = {'u_opt': u_opt,
         'Tz_core_pred_opt': Tz_core_pred_opt,
         'Tz_east_pred_opt': Tz_east_pred_opt,
         'Tz_north_pred_opt': Tz_north_pred_opt,
         'Tz_south_pred_opt': Tz_south_pred_opt,
         'Tz_west_pred_opt': Tz_west_pred_opt,
         'P_pred_opt': P_pred_opt,
         't_opt': t_opt}

if not os.path.exists('mpc_results'):
    os.makedirs('mpc_results')

if dos_attack_core_VAV:
    final['dos_attack_core_VAV'] = True
    with open('mpc_results/mpc_results_DoS_attack_core_VAV.json', 'w') as outfile:
        json.dump(final, outfile)
else: 
    with open('mpc_results/mpc_results.json', 'w') as outfile:
        json.dump(final, outfile)
