import csv
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.kernel_approximation import Nystroem

from sklearn.metrics import (
    auc,
    classification_report,
    plot_confusion_matrix,
    roc_curve,
    plot_roc_curve,
    confusion_matrix
)

from preprocessing import (
    generate_types,
)

test_file = "test.csv"
df_test = pd.read_csv(test_file, dtype=generate_types(test_file), engine="python")
df_test.set_index(["SampleName"], inplace=True)

y_test = df_test["IsMalware"]
X_test = df_test.drop("IsMalware", axis=1)


with open("encoder.pickle", "rb") as f:
    column_trans = pickle.load(f, encoding="bytes")

X_test = column_trans.transform(X_test)

with open("selector.pickle", "rb") as f:
    selector = pickle.load(f, encoding="bytes")

X_test = selector.transform(X_test)

with open("scale.pickle", "rb") as f:
    scale_transform = pickle.load(f, encoding="bytes")

X_test = scale_transform.transform(X_test)

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
        n_jobs=-1,
        p=2
    ),
    SGDClassifier(
        loss='modified_huber',
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
    RandomForestClassifier(
        max_depth=18,
        max_features=4,
        min_samples_split=5,
        n_estimators=16,
        n_jobs=-1
    ),
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
    MLPClassifier(
        alpha=1,
        max_iter=1000
    ),
    AdaBoostClassifier(),
]

map = {
    "NearestNeighbors": classifiers[0],
    "SGDSVM": classifiers[1],
    "DecisionTree": classifiers[2],
    "RandomForest": classifiers[3],
    "XGBoost": classifiers[4],
    "NeuralNet": classifiers[5],
    "AdaBoost": classifiers[6],
}

fraction_range = np.arange(0.001,0.011,0.001)
fraction_range = np.append(fraction_range, np.arange(0.02,1.02,0.02))

plt.figure()
plt.xlabel('Fraction of statistics used for training')
plt.ylabel('Percentage (%)')
plt.title('Model accuracies using different fractions of data for training')
plt.grid(True)

for name in names:
    print('Conducting training statistics analysis on {0}...'.format(name))
    train_file = "train.csv"
    df_train = pd.read_csv(
        train_file,
        dtype=generate_types(train_file),
        engine="python",
    )
    df_train.set_index(["SampleName"], inplace=True)

    print(fraction_range)

    scores = [] #Test scores for each fraction

    for fraction in fraction_range:
        print('FRACTION OF TRAINING DATASET: {0}\n'.format(fraction))

        df_train_frac = df_train.sample(frac=fraction)

        y_train = df_train_frac["IsMalware"]
        X_train = df_train_frac.drop("IsMalware", axis=1)

        print(np.shape(X_train), '\n')

        with open("encoder.pickle", "rb") as f:
            column_trans = pickle.load(f, encoding="bytes")
            encoder = column_trans

        X_train = encoder.transform(X_train)

        with open("selector.pickle", "rb") as f:
            selector = pickle.load(f, encoding="bytes")

        X_train = selector.transform(X_train)

        with open("scale.pickle", "rb") as f:
            scale_transform = pickle.load(f, encoding="bytes")

        X_train = scale_transform.transform(X_train)

        if name == 'SGDSVM':
            clf = 0 #Resets the variable to build new model each time

            clf = SGDClassifier(
                loss='modified_huber',
                alpha=0.00116,
                max_iter=2000,
                tol=1e-3
            )

            feature_map_nystroem = Nystroem(
                gamma = 5.623470180576352,
                random_state = 1,
                n_components = 250
            )

            feature_map_nystroem.fit(X_train)
        else:
            clf = 0 #Resets the variable to build new model each time
            clf = map[name]

        clf.fit(X_train, y_train)

        print("\n\nClassifier score on the test data:", clf.score(X_test, y_test))

        scores.append(clf.score(X_test, y_test)*100)

    plt.plot(fraction_range,scores,label='{0} % Test dataset accuracy'.format(name))

plt.legend(loc='best') 
plt.savefig('graphs/trainingfractions_accuracy.png')
