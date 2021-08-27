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
    auc,
    classification_report,
    plot_confusion_matrix,
    roc_curve,
    plot_roc_curve,
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

year_range = np.arange(1999, 2019, 1) #years 1999 to 2018 for training

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
    KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
    SGDClassifier(
        alpha=8.59067138335719e-05,
        l1_ratio=0.545077154805471,
        max_iter=6,
    ),
    DecisionTreeClassifier(
        random_state=0,
        max_depth=5,
        min_samples_leaf=1,
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

for name in names:
    print('Training {0} model...'.format(name))

    df_results = np.zeros((3, len(year_range)))
    df_results[0,:] = year_range
   
    train_sizes = [] #Append length of training datasets
    year_scores = [] #Classifier score on each year

    for year in year_range:
        print('TRAINING SET: {0}\n'.format(year))

        df_train = 0
        df_test = 0 #ensures clearing of previous datasets

        # Reading in year's training data
        train_file = "../csvs/{0}.csv".format(year)
        df_train = pd.read_csv(
            train_file,
            dtype=generate_types(train_file),
            engine="python",
        )
        df_train.set_index(["SampleName"], inplace=True)

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

        print('\nShape of training dataset ({0}): {1}\n'.format(year,np.shape(X_train)))
        train_sizes.append(np.shape(X_train)[0])

        clf = clf_map[name]
        clf.fit(X_train, y_train)

        print("Score:", clf.score(X_test, y_test))

        year_scores.append(clf.score(X_test, y_test))

    df_results[1,:] = year_scores
    df_results[2,:] = train_sizes

    pd.DataFrame(df_results).to_csv('model_results/single/{0}_singleyear_results.csv'.format(name),header=None,index=None)
