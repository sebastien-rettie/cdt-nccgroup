import pickle

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydot

from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    plot_confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

year_range = np.arange(1970, 2019, 1) #years 1970 to 2018 for training
train_sizes = [] #Append length of training datasets
year_scores = [] #Classifier score on each year

df_results = np.zeros((3, len(year_range)))
df_results[0,:] = year_range

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

    clf = Pipeline([("classifier", KNeighborsClassifier(n_neighbors=3, p=2))]).fit(X_train, y_train)

    print("Score:", clf.score(X_test, y_test))

    year_scores.append(clf.score(X_test, y_test))

    y_predicted = clf.predict(X_test)

    y_test_array = np.asarray(y_test)
    misclassified = y_test_array != y_predicted

    print("Misclassified samples:", y_test[misclassified].index, "\n\n")

df_results[1,:] = year_scores
df_results[2,:] = train_sizes

pd.DataFrame(df_results).to_csv('singleyear_results.csv',header=None,index=None)
