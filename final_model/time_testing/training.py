import warnings

from pandas.core.frame import DataFrame
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

# from plotting import (
#     plot_learning_curve,
#     plot_validation_curve,
# )
from preprocessing import (
    generate_types,
    get_ct_feature_names,
    preprocess_dataframe,
    encode_scale,
)

import time
import sys

###########################################

def generate_types(datafile):
    col_names = pd.read_csv(datafile, nrows=0).columns
    dtypes = {col: "str" for col in col_names}
    string_columns = [
        "SampleName",
        "Name0",
        "Name1",
        "Name10",
        "Name11",
        "Name12",
        "Name13",
        "Name14",
        "Name15",
        "Name16",
        "Name17",
        "Name18",
        "Name19",
        "Name2",
        "Name20",
        "Name21",
        "Name22",
        "Name23",
        "Name24",
        "Name25",
        "Name26",
        "Name27",
        "Name28",
        "Name29",
        "Name30",
        "Name3",
        "Name4",
        "Name5",
        "Name6",
        "Name7",
        "Name8",
        "Name9",
        "TimeDateStamp",
        "e_res",
        "e_res2",
    ]
    for column in string_columns:
        dtypes.update({column: "object"})
    return dtypes

def build_benign(year_range):
    """
    Unpacks and processes the necessary CSVs to build a benignware dataset within the given year range

    - year_range: 2 item list (or tuple), start year and end year +1 (i.e. ends of a range, last not included) for sampling

    Returns 'data', only benignware, all files found between years given (first inclusive)
    """

    year_frames = []
        
    for i in range(year_range[0],year_range[1],1):
        train_file = 'csvs/{0}.csv'.format(i)
        df = pd.read_csv(
                train_file,
                dtype=generate_types(train_file),
                engine="python",
        )
        df = df[df["IsMalware"] == '0.0'] #Grab only benign headers
        year_frames.append(df)

    return pd.concat(year_frames) #Return concatenated for all years

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
    print('Pre-sample; Malware:', mal_amount, 'Benignware:', ben_amount)
    mal = data[data["IsMalware"] == '1.0']
    ben = data[data["IsMalware"] == '0.0']

    if balance < malware_fraction:
        print('Sampling benignware set...')
        number = round(mal_amount*(benign_fraction/malware_fraction))
        print('BENIGN:', number)
        ben = proportional_sample([2000,2019], ben, number)

    elif balance > malware_fraction:
        print('Sampling malware set...')
        number = malware_fraction/benign_fraction
        print('MALWARE:', round(number*ben_amount))
        mal = mal.sample(n=round(number*ben_amount))

    data = pd.concat([ben, mal])

    return data

def proportional_sample(year_range, data, n):
    """
    Identifies the dataset distribution and samples to size n while maintaining distribution

    - year_range: 2 item list (or tuple), start year and end year +1 (i.e. ends of a range, last not included) for sampling
    - data: dataset to sample
    - n: size of dataset. 
    
    WARNING: n WILL NOT BE EXACT. input n will output between n and n+len(year_range), due to ceiling functions
    Returns 'data', downsampled dataset
    """

    if n > len(data):
        print('\nYou have provided a size n which is bigger than the dataset size you\'ve provided. Cannot downsample. Returning data...')
        return data

    print('\nDownsampling according to distribution...\n')

    original_datasize = 0 #Size of 'data' input
    year_sizes = [] #Length of each year's data

    for year in range(year_range[0], year_range[1], 1):
        frame = data.loc[data["TimeDateStamp"].str.contains(str(year)+' UTC')] #Get all rows from the data with this year
        original_datasize += len(frame)
        year_sizes.append(len(frame))

    datasizes = {} #Build dict with year/datasize key/pairs for range
    current_year = year_range[0] #For iterating through

    for size in year_sizes:
        datasizes[current_year] = np.ceil((size/original_datasize)*n) #Sample size according to year weight
        current_year += 1

    print('Distribution built:')
    print(datasizes)

    year_frames = []#Datasets to be concatenated

    for year in range(year_range[0], year_range[1], 1):
        frame = data.loc[data["TimeDateStamp"].str.contains(str(year))] #Get all rows from the data with this year
        frame = frame.sample(n=int(datasizes[year])) #And downsample
        year_frames.append(frame)

    print('\nData downsampled.\n')

    return pd.concat(year_frames)

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

df_results = np.zeros((5, len(year_range)))

benign_data = build_benign([start_year,2019])
full_size = 50456 
index = 0

for year in year_range:
    print('Training on years {0}-{1}...'.format(start_year, year))
    print('===========================================\n')
   
    df_train = 0
    df_test = 0 #ensures clearing of previous datasets

    year_frames = [] #add csvs to list here

    for i in np.arange(start_year,year+1,1): #Read in from 2000 to current year
        train_file = 'csvs/{0}.csv'.format(i)
        df = pd.read_csv(
                train_file,
                dtype=generate_types(train_file),
                engine="python",
        )
        df = df[df["IsMalware"] == '1.0'] #Grab only malware, we have benignware built above
        year_frames.append(df)

    b = benign_data #As a JS writer i get very nervous about pointer variables so copy it just for safety
    year_frames.append(b)
    df_train = pd.concat(year_frames) #Build training set from all read-in years
    df_train.set_index(["SampleName"], inplace=True)
    
    df_train = balance_dataset(df_train, 0.5)
    print(df_train["IsMalware"].value_counts())

    print('Dataset shape: {0}'.format(np.shape(df_train)))
    print('This is {0:0.2f}% of the total data.\n'.format(100*np.shape(df_train)[0]/full_size))

    # Reading in test data - THIS IS MIXED WITH ALL DIFFERENT YEARS
    test_file = "2019-21_test.csv"
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
    with open("../encoder.pickle", "rb") as f:
        column_trans = pickle.load(f, encoding="bytes")
        encoder = column_trans

    X_train = encoder.transform(X_train)
    X_test = encoder.transform(X_test)

    # Selector
    with open("../selector.pickle", "rb") as f:
        selector = pickle.load(f, encoding="bytes")

    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)

    # Scaler
    with open("../scale.pickle", "rb") as f:
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

    df_results[0,index] = year
    df_results[1,index] = acc
    df_results[2,index] = prec
    df_results[3,index] = rec
    df_results[4,index] = f1

    pd.DataFrame(df_results).to_csv('results.csv',header=None,index=None)

    index += 1
