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
os.chdir(os.path.dirname(os.path.dirname(__file__)))
import pytest
import pandas as pd
from data_plane import DataStore


def test_insert():
    sample_filename = os.path.join(os.path.dirname(__file__), 'data/Simulation_Data_to_RTRC_08042021.csv')  # Data is located relatively to this file
    Z = pd.read_csv(sample_filename, index_col=0)  # Some data to test functionality
    ds = DataStore(columns=Z.columns, n_max=10)  # Init the database

    # Test single-point insertions
    for i in range(15):
        ds.insert(Z.iloc[i])
        assert ds.D.shape[0] <= 10

    # Test small bulk insertions
    for i in range(15, 100, 8):
        ds.insert(Z.iloc[i: i + 8])
        assert ds.D.shape[0] <= 10

    # Test retrieval
    D = ds.select()
    assert isinstance(D, pd.DataFrame)

    # Test large bulk insertions
    for i in range(100, 200, 12):
        ds.insert(Z.iloc[i: i + 8])
        assert ds.D.shape[0] <= 10
        
    # Test bounded retrieval
    D = ds.select(5)
    assert D.shape[0] == 5

    # Test again single-point insertions
    for i in range(200, 210):
        ds.insert(Z.iloc[i])
        assert ds.D.shape[0] <= 10
    

def test_save_load():
    sample_filename = os.path.join(os.path.dirname(__file__), 'data/Simulation_Data_to_RTRC_08042021.csv')  # Data is located relatively to this file
    Z = pd.read_csv(sample_filename, index_col=0)  # Some data to test functionality
    ds = DataStore(columns=Z.columns, n_max=10)  # Init the database

    # Load some data and save the database
    ds.insert(Z.iloc[:50])
    ds.save('~deleteme.p')

    # Restore the database
    ds2 = DataStore()
    ds2.load('~deleteme.p')

    assert (ds2.D == ds.D).all().all()
    os.remove('~deleteme.p')


def test_robust_insert():
    sample_filename = os.path.join(os.path.dirname(__file__), 'data/Simulation_Data_to_RTRC_08042021.csv')  # Data is located relatively to this file
    Z = pd.read_csv(sample_filename, index_col=0)  # Some data to test functionality
    ds = DataStore(columns=Z.columns[:3], n_max=10)  # Init the database
    
    # Check that we don't add garbage columns in bulk inserts ...
    d = Z.iloc[:8]
    ds.insert(d)
    assert len(ds.columns) == 3

    # ... and in single inserts
    d = Z.iloc[9]
    ds.insert(d)
    assert len(ds.columns) == 3

if __name__ == "__main__":
    test_insert()
    test_save_load()




