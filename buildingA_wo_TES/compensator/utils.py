#!/usr/bin/python3
"""
-----------------------------------------------------------------------
U.S. Export Classification:  ECCN EAR99.
-----------------------------------------------------------------------
This information is subject to the export control laws of the United
States, specifically including the Export Administration Regulations
(EAR), 15 C.F.R. Part 730 et. seq. Transfer, retransfer, or disclosure
of this data by any means to a non-U.S. person (individual or company),
whether in the United States or abroad, without any required export
license or other approval from the U.S Government is prohibited.
-----------------------------------------------------------------------
-----------------------------------------------------------------------
                        - RTX PROPRIETARY -
-----------------------------------------------------------------------
This material contains proprietary information of Raytheon Technologies
Corporation.  Any copying, distribution, or dissemination of the
contents of this material is strictly prohibited and may be unlawful
without the express written permission of RTX.
If you have obtained this material in error, please notify RTRC Counsel
at (860) 610-7000 immediately.
_______________________________________________________________________
Created: 1/11/2022
Author(s):
            Fragkiskos Koufogiannis (RTRC)
 
"""

import matplotlib.pyplot as plt
# from algorithms import JITCompensator, ModularCompensator
import pandas as pd 



def load_dataset(dataset):
    # TODO: Fix relative paths
    assert dataset in {'210721', '210804', '211012'}

    if dataset == '210721':
        Z = pd.read_csv('../data/210721/Simulation_Data_to_RTRC_07212021.csv', index_col=0)
        controls = ['CHW_Tset', 'CW_Tset', 'TSupSet', 'Tset_core', 'Tset_South', 'Tset_East', 'Tset_North', 'Tset_West',
                    'VDis_Cor', 'VDis_Sou', 'VDis_Eas', 'VDis_Nor', 'VDis_Wes', 'TDis_Cor', 'TDis_Sou', 'TDis_Eas', 'TDis_Nor',
                    'TDis_Wes']
        disturbances = ['T_oa', 'RH_oa']
        sensors = ['T_core', 'T_South', 'T_East', 'T_North', 'T_West', 'CHW_dP', 'ducStaPre']
        mpc_sensors = []
    elif dataset == '210804':
        Z = pd.read_csv('../data/210804/Simulation_Data_to_RTRC_08042021.csv', index_col=0)
        controls = ['CHW_Tset', 'CW_Tset', 'TSupSet', 'Tset_core', 'Tset_South', 'Tset_East', 'Tset_North', 'Tset_West',
                    'VDis_Cor', 'VDis_Sou', 'VDis_Eas', 'VDis_Nor', 'VDis_Wes', 'TDis_Cor', 'TDis_Sou', 'TDis_Eas', 'TDis_Nor',
                    'TDis_Wes']
        disturbances = []
        sensors = ['CHW_dP', 'ducStaPre', 'TCHWRet', 'TCHWSup', 'TCWRet',
                'TCWSup', 'TMix', 'TRet', 'TSup', 'TSupCor', 'TSupEas', 'TSupNor', 'TSupSou', 'TSupWes', 'TZonCor',
                'TZonEas', 'TZonNor', 'TZonSou', 'TZonWes', 'VRet', 'VSup', 'VSupCor', 'VSupEas', 'VSupNor',
                'VSupSou', 'VSupWes', 'Voa', 'm_CHW', 'm_CW', 'Toa', 'RH_oa']
        mpc_sensors = ['TZonEas', 'TZonNor', 'TZonSou', 'TZonWes', 'Toa', 'VSupEas', 'VSupNor', 'VSupSou', 'VSupWes']
    elif dataset == '211012':
        Z = pd.read_csv('../data/211012/AnnSim_Tus_TMY3.csv')

        # Mapping new names to old ones based on Notes_for_OneYearData.txt
        feature_mapper = {
            'chiWSE.TSet': 'CHW_Tset',
            'TCWSupSet.y': 'CW_Tset',
            'senRelPre.p_rel': 'CHW_dP',
            'conAHU.TSupSet': 'TSupSet',
            'conAHU.ducStaPre': 'ducStaPre',
            'conVAVCor.TZonCooSet': 'TZonCorSet',
            'conVAVSou.TZonCooSet': 'TZonSouSet',
            'conVAVEas.TZonCooSet': 'TZonEasSet',
            'conVAVNor.TZonCooSet': 'TZonNorSet',
            'conVAVWes.TZonCooSet': 'TZonWesSet',
            'conVAVCor.TZon': 'TZonCor',
            'conVAVSou.TZon': 'TZonSou',
            'conVAVEas.TZon': 'TZonEas',
            'conVAVNor.TZon': 'TZonNor',
            'conVAVWes.TZon': 'TZonWes',
            'TOut.y': 'Toa',
            'amb.weaBus.relHum': 'RH',
            'conVAVCor.sysReq.VDisSet_flow': 'VDis_Cor',
            'conVAVSou.sysReq.VDisSet_flow': 'VDis_Sou',
            'conVAVEas.sysReq.VDisSet_flow': 'VDis_Eas',
            'conVAVNor.sysReq.VDisSet_flow': 'VDis_Nor',
            'conVAVWes.sysReq.VDisSet_flow': 'VDis_Wes',
            'conVAVCor.sysReq.TDisHeaSet': 'TDis_Cor',
            'conVAVSou.sysReq.TDisHeaSet': 'TDis_Sou',
            'conVAVEas.sysReq.TDisHeaSet': 'TDis_Eas',
            'conVAVNor.sysReq.TDisHeaSet': 'TDis_Nor',
            'conVAVWes.sysReq.TDisHeaSet': 'TDis_Wes',
            'TCHWSup.T': 'TCHWSup',
            'TCHWRet.T': 'TCHWRet',
            'TCWSup.T': 'TCWSup',
            'TCWRet.T': 'TCWRet',
            'chiWSE.m2_flow': 'm_CHW',
            'pumCW.m_flow': 'm_CW',
            'TRet.T': 'TRet',
            'senRetFlo.V_flow': 'VRet_flow',
            'VOut1.V_flow': 'Voa',
            'TMix.T': 'TMix',
            'TSup.T': 'TSup',
            'senSupFlo.V_flow': 'VSup',
            'TSupCor.T': 'TSupCor',
            'TSupSou.T': 'TSupSou',
            'TSupEas.T': 'TSupEas',
            'TSupNor.T': 'TSupNor',
            'TSupWes.T': 'TSupWes',
            'VSupCor_flow.V_flow': 'VSupCor',
            'VSupSou_flow.V_flow': 'VSupSou',
            'VSupEas_flow.V_flow': 'VSupEas',
            'VSupNor_flow.V_flow': 'VSupNor',
            'VSupWes_flow.V_flow': 'VSupWes',
        }

        # This dataset is in a different format. Casting it to the old format.
        Z = Z.transpose()
        Z.columns = Z.iloc[0]
        Z = Z.drop(labels=Z.index[0], axis='index')
        Z.index = Z.index.astype(np.int64)
        Z = Z.astype(np.float64)
        Z = Z.rename(columns=feature_mapper)
        
        controls = ['CHW_Tset', 'CW_Tset', 'TSupSet', 'Tset_core', 'Tset_South', 'Tset_East', 'Tset_North', 'Tset_West',
                    'VDis_Cor', 'VDis_Sou', 'VDis_Eas', 'VDis_Nor', 'VDis_Wes', 'TDis_Cor', 'TDis_Sou', 'TDis_Eas', 'TDis_Nor',
                    'TDis_Wes']
        disturbances = []
        sensors = ['CHW_dP', 'ducStaPre', 'TCHWRet', 'TCHWSup', 'TCWRet',
                'TCWSup', 'TMix', 'TRet', 'TSup', 'TSupCor', 'TSupEas', 'TSupNor', 'TSupSou', 'TSupWes', 'TZonCor',
                'TZonEas', 'TZonNor', 'TZonSou', 'TZonWes', 'VRet_flow', 'VSup', 'VSupCor', 'VSupEas', 'VSupNor',
                'VSupSou', 'VSupWes', 'Voa', 'm_CHW', 'm_CW', 'Toa', 'RH']
        mpc_sensors = ['TZonEas', 'TZonNor', 'TZonSou', 'TZonWes', 'Toa', 'VSupEas', 'VSupNor', 'VSupSou', 'VSupWes']
    else:
        raise ValueError('Unknown dataset')


def plot_JITCompensator(attack):
    # Unit test of JITCompensator
    t = int(train_test_ratio / (train_test_ratio + 1) * Z.shape[0])
    Z_train = Z.iloc[:t]
    Z_test = Z.iloc[t:]

    comp = JITCompensator(target='TZonEas')
    comp.fit(Z_train)

    TZonEas_hat = comp.predict(Z_test[[s for s in Z.columns if s not in attack]])

    plt.figure(figsize=(12, 6))
    plt.plot(Z_test['TZonEas'])
    plt.plot(TZonEas_hat)
    plt.legend(['Ground truth', f"Reconstructed (RMSE={(Z_test['TZonEas'] - TZonEas_hat).std():.3f})"])
    plt.title('Target: TZonEas \n Attack: ' + ', '.join(attack))
    plt.show()


def plot_ModularCompensator(train_test_ratio, Z, target):
    # Unit test of ModularCompensator
    t = int(train_test_ratio / (train_test_ratio + 1) * Z.shape[0])
    Z_train = Z.iloc[:t]
    Z_test = Z.iloc[t:]

    comp = ModularCompensator(targets=target, sensors=Z.columns, n_models=10)
    comp.fit(Z_train)

    Target_hat = comp.predict(Z_test)

    plt.figure(figsize=(12, 6))
    plt.plot(Z_test[target])
    plt.plot(Target_hat)
    plt.legend(['Ground truth', f"Reconstructed (RMSE={(Z_test[target] - Target_hat).std().values[0]:.3f})"])
    plt.title(f'Target: {target} ')
    plt.show()

def eval_hil(comp_data, beatt_data, afatt_data, target):
    comp_data = pd.read_csv(comp_data)
    beatt_data = pd.read_csv(beatt_data)
    afatt_data = pd.read_csv(afatt_data)
    for t in target:
        plt.figure(figsize=(12, 6))
        plt.plot(beatt_data[t])
        plt.plot(comp_data[t])    
        plt.plot(afatt_data[t])     
        plt.legend(['Ground truth', f"Reconstructed (RMSE={(beatt_data[t] - comp_data[t]).std():.3f})", 'Without Compensator'])
        plt.title(f'Target: {t} ')
        plt.show()

if __name__ == "__main__":
    # import os
    # train_test_ratio = 2
    # sample_filename = os.path.join(os.path.dirname(__file__), 'dataset_mpc_02162023.csv')  # Data is located relatively to this file
    # Z = pd.read_csv(sample_filename, index_col=0)  # Some data to test functionality
    # Z.reset_index(inplace=True)
    # with open('mpc_inputs.txt') as f:
    #     cols = f.read()
    # cols = cols.split('\n')
    # Z = Z[cols].copy()
    
    target = ['mod.conVAVCor.VDis_flow']

    import os
    comp_data = os.path.join(os.path.dirname(__file__), 'compensator_results.csv')
    beatt_data = os.path.join(os.path.dirname(__file__), 'before_attack_measurement.csv')
    afatt_data = os.path.join(os.path.dirname(__file__), 'after_attack_measurement.csv')
    eval_hil(comp_data, beatt_data, afatt_data, target)
