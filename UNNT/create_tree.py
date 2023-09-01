from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import argparse
from xgboost_nci60_fda_preprocess import load_and_preprocess
from xgboost import XGBRegressor

# create class xgboost to train an xgboost model. constructor takes in train test splits
# and the model parameters
class Tree:
    #constructor
    def __init__(self, args, default_data=True, n_estimators=500, max_depth=10, eta=0.1, subsample=0.5, colsample_bytree=0.8, tree_method='hist', n_jobs=-1):
        
        self.args = args
        self.default_data = default_data
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.eta = eta
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.tree_method = tree_method
        
        self.xgb_model = XGBRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, eta=self.eta, subsample=self.subsample, colsample_bytree=self.colsample_bytree, tree_method=self.tree_method, n_jobs=-1)
        
        self.start_time = None
        self.end_time = None
        self.runtime = None
        self.predictions = None
        self.rmse = None
        self.r2_error = None

        self.load_data()

    def load_data(self):
        if self.default_data:
            self.x_train, self.x_test, self.y_train, self.y_test = load_and_preprocess()

            print("Loaded NCI60 datasets.........")
        else:
            ## code to load custom data
            self.x_train, self.x_test, self.y_train, self.y_test = load_and_preprocess_custom_data(args)

            print("Loaded custom dataset.........")
            pass

    def train(self):
        print("Training XGBoost Model........")
        self.start_time = time.time()
        self.xgb_model.fit(self.x_train, self.y_train)
        self.end_time = time.time()
        self.runtime = self.end_time - self.start_time
        print("Training Time: ", self.runtime, " seconds")

    def evaluate(self):
        self.predictions = self.xgb_model.predict(self.x_test)
        self.rmse = np.sqrt(MSE(self.y_test, self.predictions))
        print("RMSE : % f" %(self.rmse))
        self.r2_error = r2_score(self.y_test, self.predictions)
        print("R2 : % f" %(self.r2_error))