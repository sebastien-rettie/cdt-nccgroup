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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from plotting import (
    plot_learning_curve,
    plot_validation_curve,
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
    "NearestNeighbors",
    "SGDSVM",
    "DecisionTree",
    "RandomForest",
    "XGBoost",
    "NeuralNet",
    "AdaBoost",
]

classifiers = [
    KNeighborsClassifier(
        n_neighbors=3, 
        p=2,
        n_jobs=-1
    ),
    SGDClassifier(
        alpha=0.00116,
        l1_ratio=0.545077154805471,
        max_iter=6,
    ),
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
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
]

clf_map = {
        "NearestNeighbors": classifiers[0],
        "SGDSVM": classifiers[1],
        "DecisionTree": classifiers[2],
        "RandomForest": classifiers[3],
        "XGBoost": classifiers[4],
        "NeuralNet": classifiers[5],
        "AdaBoost": classifiers[6]
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

train_file = 'train.csv'
test_file = 'test.csv'

df_train = pd.read_csv(train_file, dtype=generate_types(train_file), engine="python")
df_train.set_index(["SampleName"], inplace=True)

df_test = pd.read_csv(test_file, dtype=generate_types(test_file), engine="python")
df_test.set_index(["SampleName"], inplace=True)

y_train = df_train["IsMalware"]
X_train = df_train.drop("IsMalware", axis=1)

y_test = df_test["IsMalware"]
X_test = df_test.drop("IsMalware", axis=1)

(X_train, X_test, feature_list) = preprocess(X_train, X_test)

feature_list.append('all')

for name in names:
    print('\nTraining {0} model...\n'.format(name))

    #Check if previous data has already been collected
    try:
        unpack = np.loadtxt('{0}_features.csv'.format(name), unpack=True, skiprows=1, delimiter=',')
        
        print('Previous file found!')

        #Trying to combine str/float but just ends up outputting string array
        df_results = np.array([feature_list, unpack[:,0].astype(float), unpack[:,1].astype(float), unpack[:,2].astype(float), unpack[:,3].astype(float), unpack[:,4].astype(float)])
        
        print(df_results)
        print(np.shape(df_results))

        index = 0
        while index < len(df_results[1,:]):
            if float(df_results[1,:][index]) == 0.:
                break

            index += 1
        
        if index == len(df_results[1,:]):
            print('Seems the {0} data is complete. Moving to the next model...\n'.format(name))
            continue
        else:
            print('Starting from index {0}...\n'.format(index))

    except (FileNotFoundError, OSError) as e:
        print('No previous file found. Starting at index 0...')

        #RESULTS - feature name, accuracy, precision, recall, f1, training time
        df_results = np.array([feature_list, [0]*len(feature_list),[0]*len(feature_list),[0]*len(feature_list),[0]*len(feature_list),[0]*len(feature_list)])
        index = 0

    #ROWS: Feature name, score, training/predicting time
    #len(feature_list)+1 as control will be last row (all features)

    while index < len(feature_list):
        
        feature = feature_list[index]

        if feature != 'all':
            print('\nINDEX {0}:  DROPPING FEATURE: {1}'.format(index,feature))

            #Drop currently observed feature from full set (both train/test)
            X_train_dropped = X_train.drop(feature, axis=1)
            X_test_dropped = X_test.drop(feature, axis=1)

        else:
            print('\nINDEX {0}: NO FEATURES DROPPED (CONTROL)'.format(index))
            X_train_dropped = X_train
            X_test_dropped = X_test
        
        clf = clf_map[name] #Fetch currently observed model (first for loop)
        
        start_time = time.time()
        clf.fit(X_train_dropped, y_train)
            
        y_pred = clf.predict(X_test_dropped)

        print('Model fit...')

        #Grab all our statistics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label='1.0')
        rec = recall_score(y_test, y_pred, pos_label='1.0')
        f1 = f1_score(y_test, y_pred, pos_label='1.0')

        end_time = time.time() - start_time
        print('Time taken: {0:0.2f}s'.format(end_time))

        df_results[1,index] = acc
        df_results[2,index] = prec
        df_results[3,index] = rec
        df_results[4,index] = f1
        df_results[5,index] = end_time
        
        try:
            print(df_results[:,:index+1])
        except:
            ('AN ATTEMPT WAS MADE TO PRINT THE RESULTS UP TO THIS POINT, BUT SOMETHING FAILED. THIS IS LIKELY AN ARRAY-SETTING ERROR. EXITING...')
            sys.exit()

        #Write current results to CSV
        pd.DataFrame(df_results).to_csv('{0}_features.csv'.format(name), header=None, index=None)

        index += 1


