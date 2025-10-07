import os
import pandas as pd
from compensator.algorithms import ModularCompensator
rules = { 
        'mod.conVAVCor.VDis_flow' : (0,  2.8), 
        'mod.conVAVEas.VDis_flow' : (0,  0.9),
        'mod.conVAVNor.VDis_flow': (0, 0.95),
        'mod.conVAVSou.VDis_flow': (0, 0.95),
        'mod.conVAVWes.VDis_flow': (0, 0.70),
        'mod.fanSup.VMachine_flow': (0, 4.72), 
        'mod.flo.temAirPer5.T': (291.15, 303.15),
        'mod.flo.temAirEas.T': (291.15, 303.15),
        'mod.flo.temAirNor.T': (291.15, 303.15),
        'mod.flo.temAirSou.T': (291.15, 303.15),
        'mod.flo.temAirWes.T': (291.15, 303.15),
        'mod.TSup.T': (284.15, 303.15),
        'mod.TOut.y': (279.85, 308.26),
        'mod.oveRelHumOutAir.y': (0, 1),
        'mod.eleSupFan.y': (-10, 7000),
        'mod.eleChi.y': (-10, 30000),
        'mod.eleCHWP.y': (-10, 1500),
        'mod.eleCWP.y': (-10, 1500),
        'mod.eleCT.y': (-10, 4300),
        'mod.eleTot.y': (-10, 42000)
}

def load_train_data(data_fname, var_fname):
    # sample_filename = os.path.join(os.path.dirname(__file__), data_fname)  
    Z = pd.read_csv(data_fname, index_col=0)  # historical data
    Z.reset_index(inplace=True)
    Z = Z.iloc[:5856, :].copy()  #only keep July and August (15min data)
    # var_filename = os.path.join(os.path.dirname(__file__), var_fname) 
    with open(var_fname) as f:
        cols = f.read()
    cols = cols.split('\n')
    cols1 = cols[:len(cols)//2] # simulation naming 
    cols2 = cols[len(cols)//2:] # HIL naming
    Z = Z[cols1].copy()
    Z.columns = cols2
    Z['mod.fanSup.VMachine_flow'] = Z['mod.fanSup.VMachine_flow'].abs()
    Z['mod.eleSupFan.y'] = Z['mod.eleSupFan.y'].abs()
    Z['mod.eleChi.y'] = Z['mod.eleChi.y'].abs()
    Z['mod.eleCHWP.y'] = Z['mod.eleCHWP.y'].abs()
    Z['mod.eleCWP.y'] = Z['mod.eleCWP.y'].abs()
    Z['mod.eleCT.y'] = Z['mod.eleCT.y'].abs()
    Z['mod.eleTot.y'] = Z['mod.eleTot.y'].abs()
    return Z
def run_compensator(targets, measurement, Z, comp):
    # afatt_mea = pd.concat([afatt_mea, pd.DataFrame([measurement])]) #record the after attack measurement
    org_measurement = measurement.copy()
    measurement = {x:org_measurement[x] for x in list(org_measurement.keys())}
    mea_df = pd.DataFrame([measurement])
    # convert all columns from  non-numeric values (object) to numeric values (float64)
    mea_df = mea_df.astype(float)
    # mea_df.drop('total_power', axis=1, inplace = True)
    # new_targets = mea_df.columns[mea_df.isna().any()].tolist() # a list of attacked data
    new_targets = mea_df.columns[(mea_df==-123456).any()].tolist() # a list of attacked data #empty data is set to zero (confirm it)
    for var in list(rules.keys()):
        if ((mea_df[var]<rules[var][0]) | (mea_df[var]>rules[var][1])).any():
            new_targets.append(var)         
    
    print("\ntargets:",targets)
    print("new_targets:",new_targets)
    new_targets = []
    if (sorted(targets) != sorted(new_targets)) and (len(new_targets) != 0): # if new targets detected
        targets = targets + new_targets
        comp = ModularCompensator(targets=targets, sensors=Z.columns, n_models=10)
        comp.fit(Z) # train for the new_target
        target_hat = comp.predict(mea_df) #predict attacked value
        for var in targets:
            measurement[var] = target_hat[var].values[0]     

    #elif targets == new_targets: # if new targets equals previous targets, no train, only predict
    elif comp != None:
        target_hat = comp.predict(mea_df)
        for var in targets:
            measurement[var] = target_hat[var].values[0]
    else:
        comp = ModularCompensator(targets=targets, sensors=Z.columns, n_models=10)
        comp.fit(Z) # train for the new_target
        target_hat = comp.predict(mea_df) #predict attacked value
        for var in targets:
            measurement[var] = target_hat[var].values[0]    
    
    for x in list(org_measurement.keys()): 
        org_measurement[x][-1] = measurement[x]
    # comp_mea = pd.concat([comp_mea, pd.DataFrame([measurement])])
    return org_measurement, targets, comp
            
if __name__ == "__main__":
    data_fname = 'compensator\\dataset_mpc_compensator.csv'
    var_fname = 'compensator\\mpc_inputs.txt'
    Z = load_train_data(data_fname, var_fname)
    targets = None
    measurement = { 
        'V_core' : [1,1,1,1], 
        'V_east' : [1,1,1,1], 
        'V_north': [0.5, 0.5,0.5,0.5],
        'V_south': [0.5, 0.5,0.5,0.5],
        'V_west': [0.5, 0.5,0.5,0.5],
        'V_total': [3,3,3,3], 
        'T_core': [300,300,300,300], 
        'T_east': [300,300,300,400], 
        'T_north': [300,300,300,300], 
        'T_south': [300,300,300,300], 
        'T_west': [300,300,300,300], 
        'T_outdoor': [300,300,300,300], 
        'RH_outdoor': [1,1,1,1],
        'P_fan': [300,300,300,300], 
        'P_chiller': [1,1,1,1],
        'P_chwPump': [1,1,1,1],
        'P_cwPump': [1,1,1,1],
        'P_cooTower': [1,1,1,1],
        'P_totalPower': [4,4,4,4],
}

    measurement,targets =run_compensator(targets, measurement, Z)