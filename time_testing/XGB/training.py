import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pickle

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydot

import xgboost as xgb

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
import sys

###########################################

def balance_dataset(data, malware_fraction):
    """
    Samples the dataset such that the malware/benignware balance is built according to the given fraction

    - data: Dataset to balance
    - malware_fraction: Fraction of Malware desired in the dataset balance

    Returns 'data', balanced dataset
    """
    
    benign_fraction = 1-malware_fraction
    counts = data["IsMalware"].value_counts()
    balance = counts[1]/(counts[0]+counts[1])
    mal_amount = counts[1]
    ben_amount = counts[0]
    mal = data[data["IsMalware"] == '1.0']
    ben = data[data["IsMalware"] == '0.0']

    if balance < malware_fraction:
        print('Sampling benignware set...')
        number = benign_fraction/malware_fraction
        ben = ben.sample(n=round(number*mal_amount))

    elif balance > malware_fraction:
        print('Sampling malware set...')
        number = malware_fraction/benign_fraction
        mal = mal.sample(n=round(number*ben_amount))

    data = pd.concat([ben, mal])

    return data

###########################################

start_year = 2000
year_range = np.arange(2010, 2019, 1) #training on 2000-10, 2000-11, ... 2000-18 etc.

clf = xgb.XGBClassifier(
        n_estimators=100,
        colsample_bytree=0.8317,
        learning_rate=0.3,
        max_depth=11,
        min_child_weight=3.0,
        subsample=0.9099,
        gamma=0.292,
        reg_lambda=0.447,
)

fractions = [0.25, 0.5, 0.75, 'N/A']
fraction_index = 0
df_results = np.zeros((len(fractions)*5, len(year_range)))

for malware_frac in fractions:

    if malware_frac != 'N/A':
        print('{0}% MALWARE\n'.format(100*malware_frac))
    else:
        print('UNBALANCED DATASET\n')

    full_size = 50456 
    index = 0

    for year in year_range:
        print('Training on years {0}-{1}...'.format(start_year, year))
        print('===========================================\n')
   
        df_train = 0
        df_test = 0 #ensures clearing of previous datasets

        year_frames = [] #add csvs to list here

        for i in np.arange(start_year,year+1,1): #Read in from 2000 to current year
            train_file = "../csvs/{0}.csv".format(i)
            df = pd.read_csv(
                train_file,
                dtype=generate_types(train_file),
                engine="python",
            )
            year_frames.append(df)

        df_train = pd.concat(year_frames) #Build training set from all read-in years
        df_train.set_index(["SampleName"], inplace=True)
    
        if malware_frac != 'N/A':
            df_train = balance_dataset(df_train, malware_frac)
            print(df_train["IsMalware"].value_counts())

        print('Dataset shape: {0}'.format(np.shape(df_train)))
        print('This is {0:0.2f}% of the total data.\n'.format(100*np.shape(df_train)[0]/full_size))

        # Reading in test data - THIS IS MIXED WITH ALL DIFFERENT YEARS
        test_file = "../2019-21_test.csv"
        df_test = pd.read_csv(
            test_file,
            dtype=generate_types(test_file),
            engine="python"
        )
        df_test.set_index(["SampleName"], inplace=True)

        y_train = df_train["IsMalware"]
        X_train = df_train.drop("IsMalware", axis=1)

        y_test = df_test["IsMalware"]
        X_test = df_test.drop("IsMalware", axis=1)

        # Import prefit encoder
        with open("encoder.pickle", "rb") as f:
            column_trans = pickle.load(f, encoding="bytes")
            encoder = column_trans

        X_train = encoder.transform(X_train)
        X_test = encoder.transform(X_test)

        # Selector
        with open("selector.pickle", "rb") as f:
            selector = pickle.load(f, encoding="bytes")

        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)

        # Scaler
        with open("scale.pickle", "rb") as f:
            scale_transform = pickle.load(f, encoding="bytes")

        X_train = scale_transform.transform(X_train)
        X_test = scale_transform.transform(X_test)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label='1.0')
        rec = recall_score(y_test, y_pred, pos_label='1.0')
        f1 = f1_score(y_test, y_pred, pos_label='1.0')

        print('\n{0} - {1} Accuracy: {2}'.format(start_year,year,acc))
        print('{0} - {1} Precision: {2}'.format(start_year,year,prec))
        print('{0} - {1} Recall: {2}'.format(start_year,year,rec))
        print('{0} - {1} f1: {2}\n'.format(start_year,year,f1))

        df_results[5*fraction_index,index] = year
        df_results[(5*fraction_index)+1,index] = acc
        df_results[(5*fraction_index)+2,index] = prec
        df_results[(5*fraction_index)+3,index] = rec
        df_results[(5*fraction_index)+4,index] = f1

        pd.DataFrame(df_results).to_csv('results.csv',header=None,index=None)

        index += 1
    fraction_index += 1
