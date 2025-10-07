"""
This script is used to run the formulated MPC with the virtual building testbed.

Author: Guowen Li, Yangyang Fu
Email: guowenli@tamu.edu, yangyang.fu@tamu.edu 
Revisions:
    05/09/2024: CasADi optimization framework with diffential neural network models
"""

# load testbed
from pyfmi import load_fmu
import numpy as np
import pandas as pd
import sys
import os
import json
# load MPC
from mpc_dnn import mpc_case

def run_mpc(PH=12, mIce_max=3105.*5, SOC_ini=0.5):
    """
    Run MPC optimization problem with adjustable parameters.

    Parameters:
    - PH: Prediction horizon (default: 12)
    - mIce_max: Maximum mass of ice (default: 15 kg)
    - SOC_ini: Initial state of charge (default: 0.5)
    """
    # Use the provided PH, mIce_max, and SOC_ini values in your MPC calculations
    print(f"Running MPC with PH={PH}, mIce_max={mIce_max}, SOC_ini={SOC_ini}")
    
    def get_measurement(fmu_result, names):
        if 'time' not in names:
            names.append('time')

        dic = {}
        #dic['time'] = fmu_result.values(names[0])[0]
        for name in names:
            dic[name] = fmu_result[name][-1]
            #dic[name] = fmu_result[name].mean() # need to consider whether use a mean value or last value of a time-series data for one control horizon

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

    # def LIFO(a_list,x):
    #     """Last in First out: 
    #     x: scalor
    #     """
    #     #a_list.reverse()
    #     a_list.pop(0)
    #     #a_list.reverse()
    #     a_list.append(x)

    #     return a_list

    def FILO(a_list,x):
        """First in Last out: 
        x: scalor
        """
        a_list.insert(0,x)
        a_list.pop()

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
        Tz_ave_his = states['Tz_ave_his_meas']
        Toa_his = states['To_his_meas']
        GHI_his = states['GHI_his_meas']
        SOC_his = states['SOC_his_meas']
        P_his = states['P_his_meas']
        Tz_core_his_pred = states['Tz_core_his_pred']
        Tz_east_his_pred = states['Tz_east_his_pred']
        Tz_north_his_pred = states['Tz_north_his_pred']
        Tz_south_his_pred = states['Tz_south_his_pred']
        Tz_west_his_pred = states['Tz_west_his_pred']

        # read scalor
        Tz_core = measurement['conVAVCor.TZon'].values[0] - 273.15 # K to C
        Tz_east = measurement['conVAVEas.TZon'].values[0] - 273.15 # K to C
        Tz_north = measurement['conVAVNor.TZon'].values[0] - 273.15 # K to C
        Tz_south = measurement['conVAVSou.TZon'].values[0] - 273.15 # K to C
        Tz_west = measurement['conVAVWes.TZon'].values[0] - 273.15 # K to C
        Tz_ave = measurement['ave.y'].values[0] - 273.15 # K to C
        Toa = measurement['TOut.y'].values[0] - 273.15  # K to C
        GHI = measurement['weaBus.HGloHor'].values[0] # [W/m2]
        SOC = measurement['iceTan.SOC'].values[0]
        P = max(measurement['chi.P'].values[0],0)+max(measurement['priPum.P'].values[0],0)+\
            max(measurement['secPum.P'].values[0],0)+max(measurement['fanSup.P'].values[0],0)
        Tz_core_pred = Tz_pred['core']
        Tz_east_pred = Tz_pred['east']
        Tz_north_pred = Tz_pred['north']    
        Tz_south_pred = Tz_pred['south']
        Tz_west_pred = Tz_pred['west']    

        # new dic
        states['Tz_core_his_meas'] = FILO(Tz_core_his,Tz_core)
        states['Tz_east_his_meas'] = FILO(Tz_east_his,Tz_east)
        states['Tz_north_his_meas'] = FILO(Tz_north_his,Tz_north)
        states['Tz_south_his_meas'] = FILO(Tz_south_his,Tz_south)
        states['Tz_west_his_meas'] = FILO(Tz_west_his,Tz_west)
        states['Tz_ave_his_meas'] = FILO(Tz_ave_his,Tz_ave)
        states['To_his_meas'] = FILO(Toa_his,Toa)
        states['GHI_his_meas'] = FILO(GHI_his,GHI)
        states['SOC_his_meas'] = FILO(SOC_his,SOC)
        states['P_his_meas'] = FILO(P_his, P)
        states['Tz_core_his_pred'] = FILO(Tz_core_his_pred, Tz_core_pred)
        states['Tz_east_his_pred'] = FILO(Tz_east_his_pred, Tz_east_pred)
        states['Tz_north_his_pred'] = FILO(Tz_north_his_pred, Tz_north_pred)
        states['Tz_south_his_pred'] = FILO(Tz_south_his_pred, Tz_south_pred)
        states['Tz_west_his_pred'] = FILO(Tz_west_his_pred, Tz_west_pred)
        print("\nstates_updated:")
        # Iterate and print every element in the dictionary states 
        for key, values in states.items():
            print(f"{key}: {values}")
        
        return states

    def get_price(time, dt, PH):
        # unite - $/kwh
        price_tou = [0.0640, 0.0640, 0.0640, 0.0640,
                    0.0640, 0.0640, 0.0640, 0.0640,
                    0.1391, 0.1391, 0.1391, 0.1391,
                    0.3548, 0.3548, 0.3548, 0.3548,
                    0.3548, 0.3548, 0.1391, 0.1391,
                    0.1391, 0.1391, 0.1391, 0.0640]
        # assume hourly TOU pricing
        t_ph = np.arange(time, time+dt*PH, dt)
        price_ph = [price_tou[int(t % 86400 / 3600)] for t in t_ph]

        return price_ph

    # def get_price(time, dt, PH):
    #     """Chicago Time-of-Day Pricing Program
    #     Ref.: https://www.citizensutilityboard.org/time-use-pricing-plans/
    #     Time-of-Day Rates (June through September):
    #     Off-Peak (10 p.m. ~ 6 a.m.): 2.447 cents per kWh
    #     Peak (6 a.m. ~ 2 p.m. & 7 p.m. ~ 10 p.m.): 3.970 cents per kWh
    #     Super Peak (2 p.m. ~ 7 p.m.): 6.117 cents per kWh
    #     """
    #     # unite - $/kwh
    #     price_tou = [0.02447, 0.02447, 0.02447, 0.02447,
    #                  0.02447, 0.02447, 0.03970, 0.03970,
    #                  0.03970, 0.03970, 0.03970, 0.03970,
    #                  0.03970, 0.03970, 0.06117, 0.06117,
    #                  0.06117, 0.06117, 0.06117, 0.03970,
    #                  0.03970, 0.03970, 0.02447, 0.02447]
    #     # assume hourly TOU pricing
    #     t_ph = np.arange(time, time+dt*PH, dt)
    #     price_ph = [price_tou[int(t % 86400 / 3600)] for t in t_ph]

    #     return price_ph

    def read_temperature(weather_file, dt):
        """Read temperature and solar radiance from epw file. 
            This module serves as an ideal weather predictor.
        :return: a data frame at an interval of defined time_step
        """
        from pvlib.iotools import read_epw

        dat = read_epw(weather_file)
        # variable name list can be found in: https://pvlib-python.readthedocs.io/en/v0.7.1/generated/pvlib.iotools.read_epw.html
        # note: outdoor air wet bulb is not in the list
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

        RH_h = dat[0][['relative_humidity']]*0.01  # convert from 100% to [0,1]
        index_h = np.arange(3600, 3600.*(len(RH_h)+1), 3600.)
        RH_h.index = index_h

        # interpolate relative humidity into simulation steps
        index_step = np.arange(3600, 3600.*(len(RH_h)+1), dt)

        return interpolate_dataframe(RH_h, index_step)

    def get_RHoa(time, dt, PH, RHoa_year):
        index_ph = np.arange(time, time+dt*PH, dt)
        RHoa = RHoa_year.loc[index_ph, :]

        return list(RHoa.values.flatten())

    def read_GHI(weather_file, dt):  # Global horizontal solar irradiation [W/m2]
        """Read global horizontal solar irradiation from epw file. 
            This module serves as an ideal weather predictor.
        :return: a data frame at an interval of defined time_step
        """
        from pvlib.iotools import read_epw

        dat = read_epw(weather_file)
        #print("\nneed to check GHI units and values!!!")
        GHI_h = dat[0][['ghi']] # Direct and diffuse horizontal radiation recv’d during 60 minutes prior to timestamp, Wh/m^2 (it seems to be consistence with W/m^2)
        index_h = np.arange(3600, 3600.*(len(GHI_h)+1), 3600.)
        GHI_h.index = index_h

        # interpolate into simulation steps
        index_step = np.arange(3600, 3600.*(len(GHI_h)+1), dt)

        return interpolate_dataframe(GHI_h, index_step)

    def get_GHI(time, dt, PH, GHI_year):
        index_ph = np.arange(time, time+dt*PH, dt)
        GHI = GHI_year.loc[index_ph, :]

        return list(GHI.values.flatten())

    ### Load FMU
    ## ============================================================

    # Set the environment variable for the Dymola license
    if "DYMOLA_RUNTIME_LICENSE" not in os.environ:
        os.environ["DYMOLA_RUNTIME_LICENSE"] = "c:/programdata/dassaultsystemes/dymola/dymola.lic"

    # load Modelica model - VAV system virtual teatbed
    #hvac = load_fmu("VirtualTestbed_NISTChillerTestbed_DemandFlexibilityInvestigation_FakeSystem_SystemForMPC_1bInput_modeSignal.fmu", log_level=3)
    hvac = load_fmu("modelica_model/VirtualTestbed_NISTChillerTestbed_DemandFlexibilityInvestigation_FakeSystem_SystemForMPC_01bInput_0modeSignal.fmu", log_level=3)

    # Set the initial parameter value
    # 3105 L/tank from NIST testbed, Calmac part number 1082A
    # https://www.trane.com/commercial/north-america/us/en/products-systems/energy-storage-solutions/calmac-model-a-tank.html
    mIce_max = mIce_max
    hvac.set('mIce_max', mIce_max) # Nominal mass of ice in the tank, original 3105*5 kg, 
    SOC_ini = SOC_ini
    hvac.set('mIce_start', SOC_ini*mIce_max) # Start value of ice mass in the tank, original SOC = 0.5

    # # Initialize the model
    # hvac.initialize()

    # fmu settings
    options = hvac.simulate_options()
    options['ncp'] = 100 # number of output points, need to be integer
    options['initialize'] = True
    measurement_names = ['time',
                        'TOut.y',
                        'weaBus.TWetBul',
                        'weaBus.relHum',
                        'weaBus.HGloHor',
                        'uMod',
                        'uModActual.y',
                        'occSch.occupied',
                        'modCon.y',
                        'iceTan.SOC',
                        'chi.TSet',
                        'iceTan.TOutSet',
                        'TCHWSup.T',
                        'TCHWRetCoi.T',
                        'priPum.m_flow',
                        'secPum.m_flow',
                        'senRelPre.p_rel',
                        'conAHU.TSupSet',
                        'conAHU.TSup',
                        'senSupFlo.V_flow',
                        'conVAVCor.damVal.VDisSet_flow',
                        'VSupCor_flow.V_flow',
                        'conVAVSou.damVal.VDisSet_flow',
                        'VSupSou_flow.V_flow',
                        'conVAVEas.damVal.VDisSet_flow',
                        'VSupEas_flow.V_flow',
                        'conVAVNor.damVal.VDisSet_flow',
                        'VSupNor_flow.V_flow',
                        'conVAVWes.damVal.VDisSet_flow',
                        'VSupWes_flow.V_flow',
                        'conAHU.TZonCooSet',
                        'conAHU.TZonHeaSet',
                        'conVAVCor.TZon',
                        'conVAVSou.TZon',
                        'conVAVEas.TZon',
                        'conVAVNor.TZon',
                        'conVAVWes.TZon',
                        'ave.y',
                        'chi.P',
                        'priPum.P',
                        'secPum.P',
                        'fanSup.P'
                        ]
    #options['filter'] = measurement_names

    ### 2. Experiment setup
    ## define simulation period
    t_start = 212*24*3600+31*24*3600  # July 1st is Day181, August 1st is Day212, September 1st is Day 243, original test on Day 221
    t_period = 2*24*3600.
    t_end = t_start + t_period  # simulation end time
    te_warm = t_start + 1*3600.  # warm up time
    dt = 4*15*60.  # MPC timestep
    PH = PH
    CH = 1
    number_inputs = 1  # 1 mode input signal

    ## predictors
    predictor = {}
    # energy prices
    predictor['price'] = get_price(t_start, dt, PH)
    # outdoor air conditions
    weather_file = 'weather_data/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw'
    Toa_year = read_temperature(weather_file, dt)
    predictor['Toa'] = get_Toa(t_start+dt, dt, PH, Toa_year)
    RHoa_year = read_RH(weather_file, dt)
    predictor['RHoa'] = get_RHoa(t_start+dt, dt, PH, RHoa_year)
    GHI_year = read_GHI(weather_file, dt)
    predictor['GHI'] = get_GHI(t_start+dt, dt, PH, GHI_year)

    ## initialize states
    # historical Toa measurements
    Toa_his_meas_ini = get_Toa(t_start-3*dt, dt, 4, Toa_year)
    GHI_his_meas_ini = get_GHI(t_start-3*dt, dt, 4, GHI_year)
    #print("\nGHI_his_meas_ini:",GHI_his_meas_ini)

    # states
    states_ini = {'Tz_core_his_meas': [24.]*4,  # Unit: Celsius
                'Tz_east_his_meas': [24.]*4,
                'Tz_north_his_meas': [24.]*4,
                'Tz_south_his_meas': [24.]*4,
                'Tz_west_his_meas': [24.]*4,
                'Tz_ave_his_meas': [24.]*4,
                'To_his_meas': Toa_his_meas_ini,  # Unit: Celsius
                'GHI_his_meas': GHI_his_meas_ini,
                'SOC_his_meas':[SOC_ini]*4,
                'P_his_meas': [0.]*4,
                'Tz_core_his_pred': [24.]*4,  # Unit: Celsius
                'Tz_east_his_pred': [24.]*4,
                'Tz_north_his_pred': [24.]*4,
                'Tz_south_his_pred': [24.]*4,
                'Tz_west_his_pred': [24.]*4}  # initial states used for MPC models

    ## initialize mpc case
    measurement_ini = {}
    measurement = measurement_ini

    case = mpc_case(PH=PH,
                    CH=CH,
                    time=t_start,
                    dt=dt,
                    measurement=measurement_ini,
                    states=states_ini,
                    predictor=predictor)

    # input variables for fmu
    inputs = ['uMod'] # -1: Charge TES; 0: off; 1: Discharge TES; 2: Discharge chiller

    ## initialize start time
    ts = t_start
    print("t_start:", t_start)
    print("t_end:", t_end)

    ## initialize inputs
    uMPC_ini = [0] # # -1: Charge TES; 0: off; 1: Discharge TES; 2: Discharge chiller
    uMPC = uMPC_ini
    states = states_ini

    ## initialize outputs
    results = pd.DataFrame()
    t_opt = []
    u_opt = []
    Tz_core_pred_opt = []
    Tz_east_pred_opt = []
    Tz_north_pred_opt = []
    Tz_south_pred_opt = []
    Tz_west_pred_opt = []
    P_pred_opt = []
    SOC_pred_opt = []
    warmup = True

    ## main loop
    while ts < t_end:
        te = ts+dt*CH
        print("\n============================================================================================")
        t_opt.append(ts)
        print('Simulation time Clock (hour):', (ts % 86400)/3600, '~', ((te) % 86400)/3600)

        # generate control action from MPC
        if not warmup:  # activate mpc after warmup
            print("Prediction Horizon (hour): ", PH*dt/3600)
            # update mpc case
            case.set_time(ts)
            case.set_measurement(measurement)
            case.set_states(states)
            #print(case.states)
            case.set_predictor(predictor)
            case.set_u_prev(u_opt_ch)

            # call optimizer
            res, solver_status = case.optimize()
            #print("\nsolver_status['return_status']:",solver_status['return_status'])
            if solver_status['return_status'] == 'INFEASIBLE':
                u_opt_ph = case.u_start
            else:
                u_opt_ph = res['x']
            
            # [uMod, \epsilon]
            #uMPC = [res['x'][0]]
            uMPC = [u_opt_ph[0]]
            #uMPC = [float(u) for u in uMPC] # convert DM object to float
            uMPC = [int(u) for u in uMPC] # convert DM object to int since input mode is integer

            # get the control action for the control horizon
            # keep the same unit as mpc optimizer
            u_opt_ch = u_opt_ph[0:1]
            print("\nControl input:", u_opt_ch)

            # update predictions after MPC predictor is called, otherwise use measurement
            P_pred = float(case.get_power_pred(u_opt_ch))
            SOC_pred = float(case.get_SOC_pred(u_opt_ch))
            Tz_core_pred = float(case.get_core_temp_pred(u_opt_ch))
            Tz_east_pred = float(case.get_east_temp_pred(u_opt_ch))
            Tz_north_pred = float(case.get_north_temp_pred(u_opt_ch))
            Tz_south_pred = float(case.get_south_temp_pred(u_opt_ch))
            Tz_west_pred = float(case.get_west_temp_pred(u_opt_ch)) # , case._autoerror['west']
            # print("\nP_pred:",P_pred)
            # print("P_pred_modified:",P_pred*abs(u_opt_ch[0]))
            # print("SOC_pred:",SOC_pred)
            # print("Tz_core_pred:",Tz_core_pred)
            # print("Tz_east_pred:",Tz_east_pred)
            # print("Tz_north_pred:",Tz_north_pred)
            # print("Tz_south_pred:",Tz_south_pred)
            # print("Tz_west_pred:",Tz_west_pred)
            # # get all open-loop predicted values
            P_pred_ph, SOC_pred_ph, Tz_core_pred_ph, Tz_east_pred_ph, Tz_north_pred_ph, Tz_south_pred_ph, Tz_west_pred_ph = case.get_open_loop_preds(res['x'])
            print("\nP_pred_ph:",P_pred_ph)
            print("SOC_pred_ph:",SOC_pred_ph)
            print("Tz_core_pred_ph:",Tz_core_pred_ph)
            print("Tz_east_pred_ph:",Tz_east_pred_ph)
            print("Tz_north_pred_ph:",Tz_north_pred_ph)
            print("Tz_south_pred_ph:",Tz_south_pred_ph)
            print("Tz_west_pred_ph:",Tz_west_pred_ph,"\n")
            
            # update start points for optimizer using previous optimum value
            case.set_u_start(u_opt_ph)

        # advance building simulation by one step
        for i, var in zip(range(len(inputs)),inputs):
            hvac.set(var,uMPC[i])
        res = hvac.simulate(start_time=ts,
                            final_time=te,
                            options=options)
        # update clock
        ts = te

        # get measurement
        measurement = get_measurement(res, measurement_names)
        #print("\nmeasurement:", measurement)

        # update MPC model states
        # if not warmup then measurement else from mpc
        if warmup:
            Tz_core_pred = measurement['conVAVCor.TZon'].values[0]-273.15 # deg C
            Tz_east_pred = measurement['conVAVEas.TZon'].values[0]-273.15
            Tz_north_pred = measurement['conVAVNor.TZon'].values[0]-273.15
            Tz_south_pred = measurement['conVAVSou.TZon'].values[0]-273.15
            Tz_west_pred = measurement['conVAVWes.TZon'].values[0]-273.15
            P_pred = max(measurement['chi.P'].values[0],0)+max(measurement['priPum.P'].values[0],0)+\
                    max(measurement['secPum.P'].values[0],0)+max(measurement['fanSup.P'].values[0],0)
            SOC_pred = measurement['iceTan.SOC'].values[0]
            u_opt_ch = uMPC_ini # need has the same unit as mpc optimizer
        
        Tz_pred = {'core': Tz_core_pred,
            'east':Tz_east_pred,
            'north':Tz_north_pred,
            'south':Tz_south_pred,
            'west':Tz_west_pred}   
        states = get_states(states, measurement, Tz_pred)
        #print("\nstates after one control horizon simulation:")
        #print(states)

        # online MPC model calibration if applied - NOT IMPLEMENTED
        # update parameter_zones and parameters_power - NOT IMPLEMENTED

        # update predictor
        predictor['price'] = get_price(ts, dt, PH)
        predictor['Toa'] = get_Toa(ts+dt, dt, PH, Toa_year)
        predictor['RHoa'] = get_RHoa(ts+dt, dt, PH, RHoa_year)
        predictor['GHI'] = get_GHI(ts+dt, dt, PH, GHI_year)
        #print("\npredictor['GHI']:",predictor['GHI'])

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
        SOC_pred_opt.append(SOC_pred)
        
        # save measurements into DataFrame and as a CSV file at the end
        outputs_dict = {}
        output_names = measurement_names
        for output_name in output_names:
            outputs_dict[output_name] = res[output_name]
        outputs_dict['Tz_core_pred'] = Tz_core_pred + 273.15
        outputs_dict['Tz_east_pred'] = Tz_east_pred + 273.15
        outputs_dict['Tz_north_pred'] = Tz_north_pred + 273.15
        outputs_dict['Tz_south_pred'] = Tz_south_pred + 273.15
        outputs_dict['Tz_west_pred'] = Tz_west_pred + 273.15
        outputs_dict['Ptot_meas'] = max(measurement['chi.P'].values[0],0)+max(measurement['priPum.P'].values[0],0)+\
                                    max(measurement['secPum.P'].values[0],0)+max(measurement['fanSup.P'].values[0],0)
        outputs_dict['P_pred'] = P_pred
        outputs_dict['SOC_pred'] = SOC_pred
        outputs_dict['Elec_rate'] = get_price(ts-dt, dt, 1)[0] # [$/kWh]
        outputs = pd.DataFrame(outputs_dict, index=res['time'])
        results = pd.concat([results, outputs], axis=0)

    ### 3. save results  
    final = {'u_opt': u_opt,
            'Tz_core_pred_opt': Tz_core_pred_opt,
            'Tz_east_pred_opt': Tz_east_pred_opt,
            'Tz_north_pred_opt': Tz_north_pred_opt,
            'Tz_south_pred_opt': Tz_south_pred_opt,
            'Tz_west_pred_opt': Tz_west_pred_opt,
            'P_pred_opt': P_pred_opt,
            'SOC_pred_opt': SOC_pred_opt,
            't_opt': t_opt}
    # # save optimal control action and predicted values as JSON file
    # with open('./results/mpc_results_PH'+str(PH)+'.json', 'w') as outfile:
    #     json.dump(final, outfile)

    # save measurements and prediction values as CSV file
    results.to_csv('./mpc_results/results_measurements_PH'+str(PH)+'.csv')
    # end of MPC experiment
    print("\nMPC simulation finished!")
    
    return

# If you have any main method to run, make sure it is not triggered when imported
if __name__ == "__main__":
    run_mpc(PH=24, mIce_max=3105.*5, SOC_ini=0.5)
