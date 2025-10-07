"""
this script is to plot the measurements and mpc optimized output/predictions from the optimization results

Author:Guowen Li
Email: guowenli@tamu.edu
Revisions:
    10/19/2024
"""

import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import json

def plot_results(PH=12):
    """
    Plot and evaluate MPC results based on the provided PH value.

    Parameters:
    - PH: Prediction horizon used in the MPC (default: 12)
    """
    print(f"Plotting results for PH={PH}")
    # prediction horizon (hour)
    PH = PH

    # show plot in interactive window
    show_plot = False

    # Load CSV file as DataFrame, use 'time' as the index
    df_mpc_results = pd.read_csv('./mpc_results/results_measurements_PH'+str(PH)+'.csv', index_col='time')

    # Convert time from seconds to hours and use relative time (starting at 0)
    df_mpc_results['time-hr'] = (df_mpc_results.index - df_mpc_results.index[0]) / 3600

    # Helper function to shade occupied periods
    def shade_occupied_periods(ax, df, start_time_col, end_time_col):
        for idx, row in df.iterrows():
            if row['occSch.occupied'] == 1:
                ax.axvspan(row[start_time_col], row[end_time_col], alpha=0.1, color='grey')

    # Plot 1: Chiller and TES details (6x1 subplots)
    plt.figure(figsize=(12, 14))

    # 1st subplot: Electricity rate
    plt.subplot(611)
    plt.plot(df_mpc_results['time-hr'], df_mpc_results['Elec_rate'], 'k-', label='Time-of-Use')
    plt.ylabel('$/kWh')
    plt.title("Peak price = 0.3548 $/kWh; Middle price = 0.1391 $/kWh; Valley price = 0.0640 $/kWh")
    shade_occupied_periods(plt.gca(), df_mpc_results, 'time-hr', 'time-hr')
    plt.legend(loc='upper left')

    # 2nd subplot: Chiller and TES ON/OFF
    plt.subplot(612)
    plt.plot(df_mpc_results['time-hr'], df_mpc_results['uModActual.y'], 'b-', label='Actual mode')
    plt.plot(df_mpc_results['time-hr'], df_mpc_results['uMod'], color='k', linestyle='dashdot', label='Input mode') # marker = '*', 
    plt.title("Operation mode = -1 (Charge TES); 0 (off); 1 (Discharge TES); 2 (Discharge chiller)")
    plt.ylabel('Mode')
    plt.ylim(-1.1,2.1)
    shade_occupied_periods(plt.gca(), df_mpc_results, 'time-hr', 'time-hr')
    plt.legend(loc='upper left')

    # 3rd subplot: State of Charge (SOC)
    plt.subplot(613)
    plt.plot(df_mpc_results['time-hr'], df_mpc_results['iceTan.SOC'], 'k-', label='Measured SOC')
    plt.plot(df_mpc_results['time-hr'], df_mpc_results['SOC_pred'], 'k:', label='Predicted SOC')
    plt.ylabel('SOC')
    plt.ylim(0,1)
    shade_occupied_periods(plt.gca(), df_mpc_results, 'time-hr', 'time-hr')
    plt.legend(loc='upper left')

    # 4th subplot: Power
    plt.subplot(614)
    plt.plot(df_mpc_results['time-hr'], df_mpc_results[['chi.P', 'priPum.P', 'secPum.P', 'fanSup.P']].sum(axis=1), 'k-', label="Measured power") # note df_mpc_results['Ptot_meas'] is hourly data
    plt.plot(df_mpc_results['time-hr'], df_mpc_results['P_pred'], 'k:', label='Predicted power')
    plt.ylabel('Total Power [W]')
    plt.ylim(0,25000)
    shade_occupied_periods(plt.gca(), df_mpc_results, 'time-hr', 'time-hr')
    plt.legend(loc='upper left')

    # 5th subplot: Outdoor Air Temperature
    plt.subplot(615)
    plt.plot(df_mpc_results['time-hr'], df_mpc_results['TOut.y'] - 273.15, 'k-', label='OA dry-bulb temperature')
    plt.plot(df_mpc_results['time-hr'], df_mpc_results['weaBus.TWetBul'] - 273.15, color='grey', linestyle='-', label='OA wet-bulb temperature')  # Corrected line
    plt.ylabel('Temperature (°C)')
    shade_occupied_periods(plt.gca(), df_mpc_results, 'time-hr', 'time-hr')
    plt.legend(loc='upper left')

    # 6th subplot: Solar irradiance
    plt.subplot(616)
    plt.plot(df_mpc_results['time-hr'], df_mpc_results['weaBus.HGloHor'], 'k-', label='Global horizontal solar irradiance')
    plt.ylabel('Solar irradiance (W/m²)')
    plt.xlabel('Time (hour)')
    shade_occupied_periods(plt.gca(), df_mpc_results, 'time-hr', 'time-hr')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('./mpc_results/Chiller_TES_PH'+str(PH)+'.png')
    if show_plot:
        plt.show()

    # Plot 2: Zonal temperature and airflow rate (5x2 subplots)
    plt.figure(figsize=(12, 10))
    zones = ['Core', 'East', 'North', 'South', 'West']
    for i, zone in enumerate(zones):
        # Left column: Zonal temperature
        plt.subplot(5, 2, 2*i+1)
        plt.plot(df_mpc_results['time-hr'], df_mpc_results[f'conAHU.TZonCooSet']-273.15, 'k--', lw=0.5) #, label='Bounds'
        plt.plot(df_mpc_results['time-hr'], df_mpc_results[f'conAHU.TZonHeaSet']-273.15, 'k--', lw=0.5, label='Bounds') 
        plt.plot(df_mpc_results['time-hr'], df_mpc_results[f'conVAV{zone[0:3]}.TZon']-273.15, 'b-', label='Measured')
        plt.plot(df_mpc_results['time-hr'], df_mpc_results[f'Tz_{zone.lower()}_pred']-273.15, 'b:', label='Predicted')
        plt.ylabel(f'{zone} Zone Temp (°C)')
        shade_occupied_periods(plt.gca(), df_mpc_results, 'time-hr', 'time-hr')
        plt.legend(loc='upper left')

        # Right column: Airflow rate
        plt.subplot(5, 2, 2*i+2)
        plt.plot(df_mpc_results['time-hr'], df_mpc_results[f'VSup{zone[0:3]}_flow.V_flow'], 'b-', label='Measured')
        plt.plot(df_mpc_results['time-hr'], df_mpc_results[f'conVAV{zone[0:3]}.damVal.VDisSet_flow'], color='grey', linestyle='-', label='Setpoint')  # Corrected line
        plt.ylabel(f'{zone} Airflow Rate (m³/s)')
        plt.legend(loc='upper left')
    plt.xlabel('Time (hour)')
    plt.tight_layout()
    plt.savefig('./mpc_results/zone_temp_flowrate_PH'+str(PH)+'.png')
    if show_plot:
        plt.show()




    # define Time-of-Use electricity rate - $/kwh
    price_tou = [0.0640, 0.0640, 0.0640, 0.0640,
                0.0640, 0.0640, 0.0640, 0.0640,
                0.1391, 0.1391, 0.1391, 0.1391,
                0.3548, 0.3548, 0.3548, 0.3548,
                0.3548, 0.3548, 0.1391, 0.1391,
                0.1391, 0.1391, 0.1391, 0.0640]
    def get_metrics(Ptot,TZone,price_tou,nsteps_h=4):
        """
        TZone: ixk - k is the number of zones
        """
        n= len(Ptot)
        energy_cost = []
        temp_violation = []
        high_price_load = []  # Initialize high-price period load
        low_price_load = []   # Initialize low-price period load

        for i in range(n):
            # assume 1 step is 15 minutes and data starts from hour 0
            hindex = (i%(nsteps_h*24))//nsteps_h

            # energy cost
            power=Ptot[i]
            price = price_tou[hindex]
            energy_cost.append(power/1000./nsteps_h*price)
            
            # Calculate high and low price period loads
            if price == 0.3548 or price == 0.1391:
                high_price_load.append(power / 1000. / nsteps_h)  # Convert to kWh
                low_price_load.append(0)  # Zero load for low-price periods in high price load metric
            elif price == 0.0640:
                low_price_load.append(power / 1000. / nsteps_h)  # Convert to kWh
                high_price_load.append(0)  # Zero load for high-price periods in low price load metric

            # maximum temperature violation
            number_zone = 5

            T_upper = np.array([30.0 for j in range(24)])
            T_upper[6:19] = 24.0 # 25.0
            T_lower = np.array([18.0 for j in range(24)])
            T_lower[6:19] = 22.0

            overshoot = []
            undershoot = []
            violation = []
            for k in range(number_zone):
                overshoot.append(np.array([float((TZone[i,k] -273.15) - T_upper[hindex]), 0.0]).max())
                undershoot.append(np.array([float(T_lower[hindex] - (TZone[i,k]-273.15)), 0.0]).max())
                violation.append(overshoot[k]+undershoot[k])
            temp_violation.append(violation)
            
        print(np.array(energy_cost).shape)
        print(np.array(temp_violation).shape)
        print(np.array(high_price_load).shape)
        print(np.array(low_price_load).shape)
        
        # Convert metrics to arrays and concatenate for final output
        metrics = np.concatenate(
            (np.array(energy_cost).reshape(-1, 1),
             np.array(temp_violation),
             np.array(high_price_load).reshape(-1, 1),
             np.array(low_price_load).reshape(-1, 1)),
            axis=1
        )
        
        return metrics

    # Group by the interval of timestep (e.g., 15 minutes) of the dataset and pick the first row of each timestep 
    df_mpc_results['time-sec'] = df_mpc_results.index - df_mpc_results.index[0]
    df_mpc_results['time-15min'] = (df_mpc_results['time-sec'] // 900) * 900
    df_15min = df_mpc_results.groupby('time-15min').first() # pick the first row of each timestep 

    PTot = df_15min[['chi.P', 'priPum.P', 'secPum.P', 'fanSup.P']].sum(axis=1)
    TZones = df_15min[['conVAVCor.TZon',
                            'conVAVEas.TZon',
                            'conVAVNor.TZon',
                            'conVAVSou.TZon',
                            'conVAVWes.TZon']].values

    nsteps_hour = 4 # number of data time steps in 1-hour
    metrics = get_metrics(PTot.values, TZones, price_tou)
    metrics = pd.DataFrame(metrics, columns=[['energy_cost', 'temp_violation1',
                        'temp_violation2', 'temp_violation3', 'temp_violation4', 'temp_violation5',
                        'high_price_period_load', 'low_price_period_load']])
    # save for future comparison
    mpc_results_metrics = {'energy': float(PTot.sum()/nsteps_hour/1000), # kWh
                            'energy_cost': float(metrics[["energy_cost"]].sum()),
                            'max_temp_violation': float(metrics[['temp_violation1',
                                                            'temp_violation2', 
                                                            'temp_violation3', 
                                                            'temp_violation4', 
                                                            'temp_violation5']].max().max()), # degC-hour
                            'total_temp_violation': float(metrics[['temp_violation1',
                                                                'temp_violation2',
                                                                'temp_violation3',
                                                                'temp_violation4',
                                                                'temp_violation5']].sum().sum()/nsteps_hour),
                            'core_Zone_temp_violation': float(metrics[['temp_violation1']].sum()/nsteps_hour),
                            'east_Zone_temp_violation': float(metrics[['temp_violation2']].sum()/nsteps_hour),
                            'north_Zone_temp_violation': float(metrics[['temp_violation3']].sum()/nsteps_hour),
                            'south_Zone_temp_violation': float(metrics[['temp_violation4']].sum()/nsteps_hour),
                            'west_Zone_temp_violation': float(metrics[['temp_violation5']].sum()/nsteps_hour), 
                            'high_price_period_load': float(metrics['high_price_period_load'].sum()), # kWh
                            'low_price_period_load': float(metrics['low_price_period_load'].sum()) # kWh
                            }
    

    with open('./mpc_results/mpc_performance_metrics_PH'+str(PH)+'.json', 'w') as outfile:
        json.dump(mpc_results_metrics, outfile)
    print("\nPost-procesing finished!")
    
    return

# If there's a main method, ensure it won't trigger when importing
if __name__ == "__main__":
    plot_results(PH=24)