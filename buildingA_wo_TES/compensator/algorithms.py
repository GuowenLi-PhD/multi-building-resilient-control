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

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import RobustScaler
# from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
# from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
# from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from functools import partial
import logging
import threading
import time, os, pickle

from compensator.data_plane import DataStore


class JITCompensator:
    """
    A baseline compensator that trains a model in runtime.
    """
    def __init__(self, target, regressor_fn=None):
        self.inputs = []
        self.target = target
        self.regressor_fn = regressor_fn or MLPRegressor  # The regressor to use for each model
        self.Z_train = None  # Placeholder for the training dataset
        self.X_scaler = None
        self.y_scaler = None
        self.model = None

    def fit(self, Z):
        self.Z_train = Z  # Cache the whole training dataset. Training happens in run-time
 
    def predict(self, X):
        # assert self.target not in X.columns, 'Testing data contains target as a feature'
        if self.target in X.columns:
            return X[self.target]

        # We need to retrain if the existing model has the wrong inputs
        if self.inputs != X.columns.to_list():
            self.inputs = X.columns.to_list()
            self.X_scaler = RobustScaler(quantile_range=(2, 98)).fit(self.Z_train[X.columns])
            self.y_scaler = RobustScaler(quantile_range=(2, 98)).fit(self.Z_train[[self.target]])
            
            self.model = self.regressor_fn()
            self.model.fit(
                X=self.X_scaler.transform(self.Z_train[X.columns]), 
                y=self.y_scaler.transform(self.Z_train[[self.target]]).ravel()
            )

        # Do predictions
        y_hat = self.model.predict(self.X_scaler.transform(X))
        y_hat = self.y_scaler.inverse_transform(y_hat.reshape(-1, 1))
        y_hat = pd.Series(data=y_hat[:, 0], index=X.index, name=self.target)
        return y_hat
        

class ModularCompensator:
    """
    """
    def __init__(self, targets, sensors, regressor_fn=None, n_models=10, train_valid_ratio=5, pc_inputs_max=.25, maxsize=10000):
        # The sensors that need reconstruction
        if isinstance(targets, (list, set)):
            self.targets = list(targets)
        else:
            self.targets = [targets]
        self.regressor_fn = regressor_fn or MLPRegressor  # The regressor to use for each model
        self.n_models = n_models  # Number of models in total
        self.train_valid_ratio = train_valid_ratio
        self.pc_inputs_max = pc_inputs_max  # The max number of inputs for each model as a percentage of the number of features available under no attack
        self.models = None  # Placeholder for trained models. self.models[s] is a list of base regressors for sensor s
        self.sigma = None  # Covariance matrix of residual errors of all the models
        self.prior_value = None  # In case no model is usable, we return the average value of the target
        self.datastore = DataStore(columns=sensors, n_max=maxsize)  # The in-memory database
        self.flags = {  # Flags for inter-thread communication
            'train_done': True,  # Training thread sets this to True once it's done training. Reset to False by main tread
            'safe_to_write': False, # Main thread sets this to True to signal to relinquish write rights to the training thread 
            }
        self.n_infer_max = 10  # Number of inferences before we need to retrain
        self.n_infer = 0  # Counter of inferences since last retraining
        self.train_thread = None  # Handle to the training thread
        # Targets must be included in the sensors
        for t in self.targets:
            assert t in self.sensors

    @property
    def sensors(self):
        return self.datastore.columns

    def fit(self, Z=None):
        logging.info(f'Started blocking training')
        # If no data is provided, then, use the internal database
        if Z is None:
            Z = self.datastore.select()
        else:
            self.datastore.reset()
            self.datastore.insert(Z)

        self.models, self.sigma, self.prior_value = self.fit_all(Z)
        logging.info(f'Completed blocking training')

    def threaded_fit(self):
        Z = self.datastore.select()  # Fetch the data
        models, sigma, prior_value = self.fit_all(Z)  #  Train

        # Careful here with the thread synchronization
        self.flags['train_done'] = True
        logging.info(f'Completed threaded training. New model will be installed in the next inference')
        while not self.flags['safe_to_write']:
            time.sleep(.5)

        # Install new model and exit ASAP
        self.models = models
        self.sigma = sigma
        self.prior_value = prior_value
        logging.info(f'Installed retrained models')

    def fit_all(self, Z):
        """
        Train a single-output compensator for each target sensor
        """
        logging.info(f'Training for all targets using {Z.shape[0]} points')
        # Train-validation split
        t = int(self.train_valid_ratio / (self.train_valid_ratio + 1) * Z.shape[0])
        Z_train = Z.iloc[:t]
        Z_valid = Z.iloc[t:]

        # Placeholders for results. The object's attributes may be currently used and we don't want to overwrite them yet
        models, sigma, prior_value = {}, {}, {}

        for target in self.targets:
            models[target], sigma[target], prior_value[target] = self.fit_one(target, Z_train, Z_valid)

        return models, sigma, prior_value

    def fit_one(self, target, Z_train, Z_valid):
        """
        Train an ensemble of base regressors using `Z` as training/validation dataset and reconstruct `target`.
        """
        logging.info(f'Training for {target} using {Z_train.shape[0]}/{Z_valid.shape[0]} for training/validation')
        residuals = []

        # Placeholders for results. The object's attributes may be currently used and we don't want to overwrite them yet
        models = []
        sigma = np.zeros(shape=(self.n_models, self.n_models))
        prior_value =  None

        # Train ensemble
        while len(models) < self.n_models:
            logging.info(f'Training base regressor no. {len(models) + 1} for target {target}')
            # Sample the number of input features. If we don't condition on the number, practically all models will have ~n_sensors/2 inputs.
            n_inputs = np.random.randint(1, int(self.pc_inputs_max * Z_train.shape[1]))  # Models with several inputs are prone to be invalidated by most attacks
            inputs = np.random.choice(self.sensors, size=n_inputs, replace=False)

            # Check for empty inputs/outputs
            if (target not in inputs) and len(inputs) > 0:
                # Prep data ---drop Nones in training and validation sets
                Zt = Z_train[list(inputs) + [target]]
                Zt = Zt.dropna()  # Drop rows with None and return a new DataFrame
                if len(Zt) == 0:  # We risk ending up with empty training / validation sets
                    break

                Zv = Z_valid[list(inputs) + [target]]
                Zv = Zv.dropna()  # Drop rows with None and return a new DataFrame
                if len(Zv) == 0:  # We risk ending up with empty training / validation sets
                    break

                # Train a base model
                model = self.regressor_fn()
                model.fit(X=Zt[inputs], y=Zt[target])

                # Estimate accuracy (typically done using the score fn in sk-learn); needed for fusing
                e = Zv[target].to_numpy() - model.predict(X=Zv[inputs])
                e = pd.Series(data=e, index=Zv.index)  # Keep track of the indices; validation points need to be paired up later to estimate the correlation matrix of the ensemble.
                
                residuals.append(e)
                models.append((inputs, model))

        # Compute the covariance matrix
        # There are multiple options here, e.g. use the best model, take a weighted average, etc
        # Ideally, the covariance matrix should be well-conditioned. The pseudo-inverse is used to avoid numerical instabilities.
        # Here, we assume that the regressors provide estimates corrupted wit white additive noise, i.e regressors are unbiased / residuals are zero-mean (they are not!) and errors are normally distributed.
        # As a sanity check or as a future work, we can test the assumption on the distribution
        logging.info(f'Validating ensemble for target {target}')
        for i in range(self.n_models):
            for j in range(self.n_models):
                r = pd.concat([residuals[i], residuals[j]], axis=1, join='inner')  # Each model may have slightly different validation points due to missing values. Perform a join operation here.
                if len(r) > 0:
                    sigma[i, j] = np.dot(r.iloc[:, 0], r.iloc[:, 1]) / r.shape[0]
                else:
                    sigma[i, j] = 0  # We risk that two regressors share no validation points. In that unlucky case, we assume that they do not share training points either and, thus, are assumed uncorrelated

        # Compute the average value to handle the corner case of no usable models during prediction
        prior_value = Z_train.mean()  # Nones are skipped by default

        return models, sigma, prior_value[target]

    def predict(self, X, allow_retrains=True):
        # Store the data point for future training
        if self.datastore:
            self.datastore.insert(X)

        # Sanitize input: treat everything in batch mode
        if isinstance(X, pd.Series):
            X = X.to_frame().transpose()

        # Decide wether to retrain
        if allow_retrains:
            self.n_infer += X.shape[0]

            # Fire a training thread if it's time and no other training thread is running
            if self.flags['train_done'] and self.n_infer > self.n_infer_max:
                self.n_infer = 0
                self.flags['train_done'] = False
                self.train_thread = threading.Thread(target=self.threaded_fit)
                self.train_thread.start()

            # If a training thread is running and it signaled it's done, allow it to exit
            if self.train_thread and self.flags['train_done']:
                self.flags['safe_to_write'] = True
                self.train_thread.join()
                self.flags['safe_to_write'] = False
                self.train_thread = None

        # Placeholder for the output
        ans = pd.DataFrame(index=X.index, columns=self.targets)

        for target in self.targets:
            usable_models = []
            for i, m in enumerate(self.models[target]):
                # Check whether model is usable
                valid = True
                for inp in m[0]:
                    if inp in X.columns[X.isna().any()].tolist():
                        valid = False
                        break
                # If usable, keep its index
                if valid:
                    usable_models.append(i)
            k = len(usable_models)

            # If no usable model is found, return the average value
            if k == 0:
                y_hat = [self.prior_value[target]] * len(X.index)
                logging.warning(f"No pretrained usable model was found for this attack. Available sensors are {' '.join(X.columns)}. Using oblivious sensors.")
            else:
                y_hats = [self.models[target][i][1].predict(X[self.models[target][i][0]]) for i in usable_models]
                sub_sigma = np.zeros(shape=(k, k))
                for i in range(k):
                    for j in range(k):
                        sub_sigma[i, j] = self.sigma[target][usable_models[i], usable_models[j]]

                w = np.linalg.pinv(sub_sigma) @ np.ones((k, 1))  # Using the pseudo-inverse for stability
                w /= w.sum()  # Pray for it to be non-zero!
                y_hat = sum([w_ * y_ for w_, y_ in zip(w, y_hats)])

            ans[target] = y_hat

        return ans

    def save(self, folder):
        """
        Save the compensator to a file for persistency. We avoid pickling the whole object to allow for better control during code upgrades.
        """
        S = {
            'target': self.target,
            'regressor_fn': self.regressor_fn,
            'n_models': self.n_models,
            'train_valid_ratio': self.train_valid_ratio,
            'pc_inputs_max': self.pc_inputs_max,
            'models': self.models,
            'sigma': self.sigma,
            'prior_value': self.prior_value,
            'version': 1,  # In case we change the data format, leave an easy way to tell them apart
            }

        os.makedirs(folder, exist_ok=True)
        file = os.path.join(folder, 'compensator.p')
        with open(file, 'wb') as f:
            pickle.dump(S, f)

        file = os.path.join(folder, 'datastore.p')
        self.datastore.save(file)

    def load(self, folder):
        """
        Restore a compensator from a file.
        """
        file = os.path.join(folder, 'compensator.p')
        with open(file, 'rb') as f:
            S = pickle.load(f)

        self.target = S['target']
        self.regressor_fn = S['regressor_fn']
        self.n_models = S['n_models']
        self.train_valid_ratio = S['train_valid_ratio']
        self.pc_inputs_max = S['pc_inputs_max']
        self.models = S['models']
        self.sigma = S['sigma']
        self.prior_value = S['prior_value']
        assert S['version'] == 1

        file = os.path.join(folder, 'datastore.p')
        self.datastore = DataStore()
        self.datastore.load(file)
