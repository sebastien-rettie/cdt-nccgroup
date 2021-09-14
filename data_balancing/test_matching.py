import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

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
    mal = data[data["IsMalware"] == '1.0']
    ben = data[data["IsMalware"] == '0.0']
    mal_amount = mal.shape[0]
    ben_amount = ben.shape[0]
    balance = mal_amount/(mal_amount+ben_amount)

    print('Current dataset balance: {0:0.0f}% malware'.format(100*balance))

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

full_train_size = 50456
test_balances = [0.25, 0.5, 0.75]
train_balances = np.arange(0.1,1,0.05)

fraction_index = 0
df_results = np.zeros((len(test_balances)*5, len(train_balances)))
#ROWS: training balance, acc, prec, rec, f1

for test_fraction in test_balances:
    print("=====================================================\n")
    print("Balancing the test set to {0:0.0f}% malware...\n".format(100*test_fraction))
    print("=====================================================")

    index = 0
    for train_fraction in train_balances:
        print("\n>>> Train set balance: {0:0.0f}% malware\n".format(100*train_fraction))

        train_file = "train.csv"
        df_train = pd.read_csv(
            train_file,
            dtype=generate_types(train_file),
            engine="python",
        )
        df_train.set_index(["SampleName"], inplace=True)
    
        print("Training set counts:")
        df_train = balance_dataset(df_train, train_fraction)
        print(df_train["IsMalware"].value_counts())

        print('This is {0:0.2f}% of the total data.\n'.format(100*np.shape(df_train)[0]/full_train_size))
        if 100*np.shape(df_train)[0]/full_train_size < 10:
            print('Insufficient datasize, looping to next...') #Skip training and testing if less than 10% of data after balancing
            index += 1
            continue

        test_file = "test.csv"
        df_test = pd.read_csv(
            test_file,
            dtype=generate_types(test_file),
            engine="python"
        )
        df_test.set_index(["SampleName"], inplace=True)
        
        print("Testing set counts:")
        df_test = balance_dataset(df_test, test_fraction)
        print(df_test["IsMalware"].value_counts())

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

        print('\nAccuracy: {0}'.format(acc))
        print('Precision: {0}'.format(prec))
        print('Recall: {0}'.format(rec))
        print('f1: {0}\n'.format(f1))

        df_results[5*fraction_index,index] = train_fraction
        df_results[(5*fraction_index)+1,index] = acc
        df_results[(5*fraction_index)+2,index] = prec
        df_results[(5*fraction_index)+3,index] = rec
        df_results[(5*fraction_index)+4,index] = f1

        pd.DataFrame(df_results).to_csv('balance_results.csv',header=None,index=None)

        index += 1

    fraction_index += 1
