

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 01:27:41 2021

@author: sakethpachika
"""

### Code for preprocessing 

import pandas as pd

import numpy as np

from numpy import nan

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix, accuracy_score,roc_auc_score

#import lightgbm as lgb

import imblearn

from imblearn.over_sampling import SMOTE

#basic tools
import os
import numpy as np
import pandas as pd
import warnings

from numpy.ma import MaskedArray
import sklearn.utils.fixes

sklearn.utils.fixes.MaskedArray = MaskedArray

import skopt

#tuning hyperparameters
from bayes_opt import BayesianOptimization
from skopt  import BayesSearchCV

#graph, plots
import matplotlib.pyplot as plt
import seaborn as sns

#building models
import lightgbm as lgb
#import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import time
#import syspip3

#metrics
from sklearn.metrics import roc_auc_score, roc_curve
#import shap
warnings.simplefilter(action='ignore', category=FutureWarning)

filename = '2021_Competition_Training.csv'
# n = sum(1 for line in open(filename)) - 1 #number of records in file (excludes header)
# s = 200000 #desired sample size
# skip = sorted(random.sample(range(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
df = pd.read_csv(filename,low_memory=False) #,skiprows=skip)

import random
for col in df.columns:
    if df.dtypes[col]==np.object:
        df[col] = df[col].fillna(-1)
    else:
        df[col] = df[col].fillna(-999)


filteredcolumns = df.dtypes[df.dtypes==np.object]
listofColumnNames = list(filteredcolumns.index)

filtered_numeric = df.dtypes[df.dtypes==np.int64]
listofint = list(filtered_numeric.index)

filtered_float = df.dtypes[df.dtypes==np.float64]
listoffloat = list(filtered_float.index)

listofnumeric = listofint+listoffloat

sc = StandardScaler()
df[listofnumeric[1:]] = sc.fit_transform(df[listofnumeric[1:]])

for col in listofColumnNames[1:]:
    df[col] = df[col].astype('category')

## coding the category columns

for col in listofColumnNames[1:]:
    df[col+'cat'] = df[col].cat.codes

## dropping the actual columns

for col in listofColumnNames[1:]:
    df = df.drop(col,axis=1)

## Below steps only for training data

df['target'] = df['covid_vaccinationcat']

df = df.drop(['covid_vaccinationcat'],axis=1)

new_df = df.drop(['Unnamed: 0','ID','target'],axis=1)

X = df.iloc[:, 2:-1].values

y = df.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


opt_params = {'learning_rate': 0.05,'objective': 'binary', 'metric': 'auc', 'sub_feature': 0.8, 'num_leaves': 100, 'min_data': 400, 'max_depth': 15, 'is_unbalance': True,'boost_from_average': False}



model = lgb.LGBMClassifier(**opt_params)


model.fit(X_train,y_train)

y_pred = model.predict(X_test)

y_pred_prob_train = model.predict_proba(X_test)


print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))




cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
print("initial auc:",roc_auc_score(y_test,y_pred))
print("auc of unvaccinated people:",roc_auc_score(y_test,y_pred_prob_train[:,0]))
print("auc of vaccinated people:",roc_auc_score(y_test,y_pred_prob_train[:,1]))


holdout_df = pd.read_csv('2021_Competition_Holdout.csv',low_memory=False)

for col in holdout_df.columns:
    if holdout_df.dtypes[col]==np.object:
        holdout_df[col] = holdout_df[col].fillna('Unknown')
    else:
        holdout_df[col] = holdout_df[col].fillna(-999)


filteredcolumns = holdout_df.dtypes[holdout_df.dtypes==np.object]
listofColumnNames = list(filteredcolumns.index)


filtered_numeric_holdout = holdout_df.dtypes[holdout_df.dtypes==np.int64]
listofint_holdout = list(filtered_numeric_holdout.index)

filtered_float_holdout = holdout_df.dtypes[holdout_df.dtypes==np.float64]
listoffloat_holdout = list(filtered_float_holdout.index)

listofnumeric = listofint_holdout+listoffloat_holdout

sc = StandardScaler()
holdout_df[listofnumeric[1:]] = sc.fit_transform(holdout_df[listofnumeric[1:]])

for col in listofColumnNames[1:]:
    holdout_df[col] = holdout_df[col].astype('category')

## coding the category columns

for col in listofColumnNames[1:]:
    holdout_df[col+'cat'] = holdout_df[col].cat.codes
    
## dropping the actual columns

for col in listofColumnNames[1:]:
    holdout_df = holdout_df.drop(col,axis=1)
    
X_holdout = holdout_df.iloc[:, 2:].values

#X_holdout = sc.fit_transform(X_holdout)


y_pred_prob = model.predict_proba(X_holdout)

y_pred_new_2 = model.predict(X_holdout)

new_df_holdout = pd.DataFrame(y_pred_prob[:,0].reshape(len(y_pred_prob),1))

new_df_holdout['Score'] = pd.DataFrame(y_pred_prob[:,0].reshape(len(y_pred_prob),1))

final_df = [holdout_df['ID'],new_df_holdout['Score']]

headers = ["ID","Score"]

final_df_2 = pd.concat(final_df,axis=1,keys=headers)

final_df_2["Rank"]=final_df_2["Score"].rank(ascending=False).astype(int)

total = final_df_2.sort_values('Rank')

total.to_csv('lightgbm_classifier_v4_new.csv')


feature_impt = pd.DataFrame(sorted(zip(model.feature_importances_,new_df.columns)),columns=['Value','Feature'])

feature_impt.to_csv('lightgbm_classifier_feature_impt.csv')
