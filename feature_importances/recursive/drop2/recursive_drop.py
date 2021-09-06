#Runs each model with given parameters and saves feature importances to csv

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pickle

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydot

import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from preprocessing import (
    generate_types,
    get_ct_feature_names,
    preprocess_dataframe,
    encode_scale,
)
import time
from datetime import date
import sys

names = [
    "DecisionTree",
    "RandomForest",
    "XGBoost",
    "AdaBoost",
]

classifiers = [
    DecisionTreeClassifier(
        random_state=0,
        max_depth=10,
        min_samples_leaf=2,
        min_samples_split=2,
        splitter="best",
    ),
    RandomForestClassifier(max_depth=3, n_estimators=10, n_jobs=-1),
    xgb.XGBClassifier(
        n_estimators=100,
        colsample_bytree=0.8317,
        learning_rate=0.3,
        max_depth=11,
        min_child_weight=3.0,
        subsample=0.9099,
        gamma=0.292,
        reg_lambda=0.447,
    ),
    AdaBoostClassifier(),
]

clf_map = {
        "DecisionTree": classifiers[0],
        "RandomForest": classifiers[1],
        "XGBoost": classifiers[2],
        "AdaBoost": classifiers[3]
}

def preprocess(train, test):

    train = train.drop("TimeDateStamp", axis=1) #Drop timestamp
    test = test.drop("TimeDateStamp", axis=1)

    for col in train.columns:
        if "Name" in col:
            train = train.drop(col, axis=1) #Drops the string 'NameXX' columns
            test = test.drop(col, axis=1)

    features = list(train.columns) #Get feature names

    #Convert the rest of the data to float
    train = train.apply(pd.to_numeric)
    test = test.apply(pd.to_numeric)

    return (train, test, features)

################################################

train_file = '../train.csv'
test_file = '../test.csv'

df_train = pd.read_csv(train_file, dtype=generate_types(train_file), engine="python")
df_train.set_index(["SampleName"], inplace=True)

df_test = pd.read_csv(test_file, dtype=generate_types(test_file), engine="python")
df_test.set_index(["SampleName"], inplace=True)

y_train = df_train["IsMalware"]
X_train = df_train.drop("IsMalware", axis=1)

y_test = df_test["IsMalware"]
X_test = df_test.drop("IsMalware", axis=1)

(X_train, X_test, feature_list) = preprocess(X_train, X_test)

for name in names:
    try:
        open('{0}_results.csv'.format(name))
        print('Results found for {0}. Skipping to next model...'.format(name))
        continue

    except FileNotFoundError:
        pass
    
    print('\nTraining {0} model...\n'.format(name))
    
    feature_list = list(X_train.columns)

    #Number of droppable features. Divide by 2 and floor, then *2
    #This is because we drop 2 least important features each time
    #If total features not divisible by 2, we leave the remainder at the end
    #This is the reason for floor before re-multiply
    
    len_droppable = 2*np.floor(len(feature_list)/2)

    #Column added for each 2 features dropped
    df_results = np.zeros((6, int(len_droppable/2)))

    #For each feature name dropped each iter
    dropped_features = []

    index = 0

    while len(dropped_features) != len_droppable:
        X_train_dropped = X_train.drop(dropped_features, axis=1) #Drop least important features so far
        X_test_dropped = X_test.drop(dropped_features, axis=1)

        clf = clf_map[name] #Fetch currently observed model (first for loop)

        start_time = time.time()
        clf.fit(X_train_dropped, y_train)

        y_pred = clf.predict(X_test_dropped)

        print('\n{0}: Training successful with {1} dropped features.'.format(index+1, len(dropped_features)))
        print('Training on {0} features...'.format(X_train_dropped.shape[1]))

        #Grab all our statistics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label='1.0')
        rec = recall_score(y_test, y_pred, pos_label='1.0')
        f1 = f1_score(y_test, y_pred, pos_label='1.0')
        end_time = time.time() - start_time

        print('Accuracy:', acc, '\nPrecision:', prec, '\nRecall:', rec, '\nf1:', f1, '\nTime:', end_time)

        df_results[0,index] = len(dropped_features)
        df_results[1,index] = acc
        df_results[2,index] = prec
        df_results[3,index] = rec
        df_results[4,index] = f1
        df_results[5,index] = end_time

        #Find 2 least important features
        imp_results = []

        importances = clf.feature_importances_
        for ind, score in enumerate(importances):
            imp_results.append((feature_list[ind], score))

        imp_results.sort(key=lambda x: x[1])
        dropped_features += [i[0] for i in imp_results[:2]] #Add names of 2 least important features to drop list
        
        #Have to remove dropped features from the feature list to index properly above in imp_results
        for d in dropped_features:
            if d in feature_list:
                feature_list.remove(d)

        index += 1

    #Write model results to CSV
    pd.DataFrame(df_results).to_csv('{0}_results.csv'.format(name), header=None, index=None)
