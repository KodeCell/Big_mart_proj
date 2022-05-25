# -*- coding: utf-8 -*-
"""
Created on Wed May 18 18:11:52 2022

@author: Anshul
"""
import pandas as pd

df = pd.read_csv('data_cleaned.csv')
train_target = pd.read_csv('train_target.csv')
# splitting the original train and test set from the dataframe
test = df.loc[8523:]
test.reset_index(drop=True, inplace=True)
train = df.loc[:8522]
train.reset_index(drop=True, inplace=True)
train = pd.concat([train, train_target], axis=1)  # merging the target column to the train set

import preprocessing
X,Y = preprocessing.preprocess(train) # getting the scaled data
# train test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 2)

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
dt_reg = DecisionTreeRegressor(random_state = 0)
neigh = KNeighborsRegressor(n_neighbors=2)
lin_reg = LinearRegression()
rid_reg = Ridge()
svm_reg = SVR()
rf_reg = RandomForestRegressor(max_depth = 3)
las_reg = Lasso(alpha = 0.004) # found the optimal value in the jupyter notebook

model_list = [lin_reg,las_reg,rid_reg,svm_reg,dt_reg,neigh]
import model_test
model_performance = model_test.train(model_list,X_train,X_test,Y_train,Y_test)

