from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import argparse
from . import xgboost_preprocess
from xgboost import XGBRegressor

class Tree:
    #constructor
    def __init__(self, args, default_data=True):
        
        self.args = args
        self.default_data = default_data
        
        if args.gpu:
            self.xgb_model = XGBRegressor(n_estimators=self.args.n_estimators,
                                        max_depth=self.args.max_depth,
                                        eta=self.args.eta,
                                        subsample=self.args.subsample,
                                        colsample_bytree=self.args.colsample_bytree,
                                        tree_method='gpu_hist')
        else:
            self.xgb_model = XGBRegressor(n_estimators=self.args.n_estimators,
                                        max_depth=self.args.max_depth, 
                                        eta=self.args.eta, 
                                        subsample=self.args.subsample, 
                                        colsample_bytree=self.args.colsample_bytree, 
                                        tree_method='hist', n_jobs=-1)
        
        self.start_time = None
        self.end_time = None
        self.runtime = None
        self.predictions = None
        self.rmse = None
        self.r2_error = None

        if args.data_path:
            self.default_data = False

        self.load_data()

    def load_data(self):
        if self.default_data:
            self.x_train, self.x_test, self.y_train, self.y_test = xgboost_preprocess.load_and_preprocess_default_data()

            print("Loaded NCI60 datasets.........")
        else:
            ## code to load custom data
            self.x_train, self.x_test, self.y_train, self.y_test = xgboost_preprocess.load_and_preprocess_custom_data(self.args)

            print("Loaded custom dataset.........")

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