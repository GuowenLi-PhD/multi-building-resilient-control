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

import pandas as pd
import pickle


class DataStore:
    """
    A simple in-memory database to be used by the compensator.

    :param n_max (int): Maximum number of data points to keep in memory
    """
    def __init__(self, columns=[], n_max=None):
        self.n_max = n_max
        self.D = pd.DataFrame(columns=columns)

    @property
    def columns(self):
        return self.D.columns.to_list()

    def insert(self, d):
        """
        Append a new data point (as a Series) or several data points (as a DataFrame).
        """
        assert isinstance(d, (pd.Series, pd.DataFrame))
        d = d[self.columns]  # Keep only relevent columns
        if isinstance(d, pd.Series):
            self.D = pd.concat([self.D, d.to_frame().T], ignore_index=True)
        else:
            self.D = pd.concat([self.D, d], ignore_index=True)
        self.truncate()

    def select(self, n=float('inf')):
        """
        Return a DataFrame containing up to `n` data points.
        """
        self.D.sort_index(inplace=True)
        n = min(n, self.D.shape[0])
        return self.D.iloc[-n:, :]

    def truncate(self):
        """
        Drop older data once we have accumulated enough of them. For performance, we opt for dropping several data, less frequently.
        """
        if self.D.shape[0] > self.n_max:
            self.D.sort_index(inplace=True)
            self.D.drop(self.D.index[:-int(.9 * self.n_max)], inplace=True)

    def reset(self):
        """
        Reset the database, i.e. remove all stored data. Equivalent to mySQL's TRUNCATE
        """
        self.D = pd.DataFrame(columns=self.D.columns)

    def save(self, file):
        """
        Store the database in a file for future use
        """
        S = {'data': self.D, 'n_max': self.n_max}
        # NOTE: Pickle may have scaling, security, performance, etc issues
        with open(file, 'wb') as f:
            pickle.dump(S, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file):
        """
        Restore a database from a file
        """
        with open(file, 'rb') as f:
            S = pickle.load(f)
            self.n_max = S['n_max']
            self.D = S['data']
