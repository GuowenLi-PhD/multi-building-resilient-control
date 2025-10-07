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

import os
import logging
import sys
import pytest
import pandas as pd


if not logging.getLogger().hasHandlers():
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger().setLevel(logging.INFO)

from algorithms import ModularCompensator


def test_single_output():
    sample_filename = os.path.join(os.path.dirname(__file__), 'dataset_mpc_02162023.csv')  # Data is located relatively to this file
    Z = pd.read_csv(sample_filename, index_col=0)  # Some data to test functionality
    Z.reset_index(inplace=True)
    # Z.drop(index=[0,1], inplace=True)
    with open('mpc_inputs.txt') as f:
        cols = f.read()
    cols = cols.split('\n')
    Z = Z[cols].copy()
    T = Z.shape[0]
    
    # Base training
    comp = ModularCompensator(targets='var12', sensors=Z.columns, n_models=10)
    comp.fit(Z.iloc[:T // 2])
    t = T // 2

    # Single predictions
    for _ in range(100):
        d = Z.iloc[t]
        y_hat = comp.predict(d)
        t += 1

    # Batch predictions
    for _ in range(10):
        d = Z.iloc[t: t + 8]
        y_hat = comp.predict(d)
        assert y_hat.shape[0] == d.shape[0]
        t += 8

    # TODO: Add more tests that explore what happens when `d` has more/less columns, None values, etc.


def test_multi_output():
    sample_filename = os.path.join(os.path.dirname(__file__), 'dataset_mpc_02162023.csv')  # Data is located relatively to this file
    Z = pd.read_csv(sample_filename, index_col=0)  # Some data to test functionality
    Z.reset_index(inplace=True)
    # Z.drop(index=[0,1], inplace=True)
    with open('mpc_inputs.txt') as f:
        cols = f.read()
    cols = cols.split('\n')
    Z = Z[cols].copy()
    T = Z.shape[0]
    
    # Base training
    comp = ModularCompensator(targets=['var2', 'var12'], sensors=Z.columns, n_models=10)
    comp.fit(Z.iloc[:T // 2])

    t = T // 2

    # Single predictions
    for _ in range(100):
        d = Z.iloc[t]
        y_hat = comp.predict(d)
        t += 1


def test_retraining():
    return

    sample_filename = os.path.join(os.path.dirname(__file__), 'dataset_mpc.csv')  # Data is located relatively to this file
    Z = pd.read_csv(sample_filename, index_col=0)  # Some data to test functionality
    T = Z.shape[0]
    
    # Base training
    comp = ModularCompensator(targets=['TZonEas', 'TZonWes'], n_models=10)
    comp.fit(Z.iloc[:T // 2])

    # Generate an arbitratily attacked dataset
    Z2 = Z.copy()
    N, m = Z2.shape
    for s in mpc_sensors:
        for _ in range(2):  # Each sensor is attacked a couple of times throughout the year
            t1 = np.random.randint(N)  # Start time of the attack
            t2 = t1 + np.random.poisson(lam=1000)  # End time of the attack
            t2 = min(t2, N)
            Z2[s].iloc[t1:t2] = None

    # TODO: Check the latency, accuracy, triggering, completion of retrainings.


if __name__ == "__main__":
    # test_single_output()
    test_multi_output()
