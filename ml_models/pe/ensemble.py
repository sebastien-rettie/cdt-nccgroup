import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


from preprocessing import (
    generate_types,
)

train_file = "train.csv"
df_train = pd.read_csv(
    train_file,
    dtype=generate_types(train_file),
    engine="python",
)
df_train.set_index(["SampleName"], inplace=True)

test_file = "test.csv"
df_test = pd.read_csv(test_file, dtype=generate_types(test_file), engine="python")
df_test.set_index(["SampleName"], inplace=True)


y_train = df_train["IsMalware"]
X_train = df_train.drop("IsMalware", axis=1)

y_test = df_test["IsMalware"]
X_test = df_test.drop("IsMalware", axis=1)


# import prefit encoder
with open("encoder.pickle", "rb") as f:
    column_trans = pickle.load(f, encoding="bytes")
encoder = column_trans
X_train = encoder.transform(X_train)
X_test = encoder.transform(X_test)

with open("selector.pickle", "rb") as f:
    selector = pickle.load(f, encoding="bytes")

X_train = selector.transform(X_train)
X_test = selector.transform(X_test)

scale_transform = StandardScaler(with_mean=False)
scale_transform.fit(X_train)

X_train = scale_transform.transform(X_train)
X_test = scale_transform.transform(X_test)

estimators = [
    ("NearestNeighbors", KNeighborsClassifier(n_neighbors=3, n_jobs=-1)),
    (
        "SGDSVM",
        SGDClassifier(
            alpha=8.59067138335719e-05,
            l1_ratio=0.545077154805471,
            max_iter=6,
        ),
    ),
    (
        "DecisionTree",
        DecisionTreeClassifier(
            random_state=0,
            max_depth=5,
            min_samples_leaf=1,
            min_samples_split=2,
            splitter="best",
        ),
    ),
    (
        "XGBoost",
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
    ),
    (
        "RandomForest",
        RandomForestClassifier(max_depth=5, n_estimators=18, max_features=3, n_jobs=-1),
    ),
    (
        "NeuralNet",
        MLPClassifier(
            alpha=0.012,
            hidden_layer_sizes=21,
            activation="relu",
            solver="adam",
            max_iter=1000,
        ),
    ),
    ("AdaBoost", AdaBoostClassifier()),
]

SC = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

SCNN = StackingClassifier(
    estimators=estimators, final_estimator=MLPClassifier(alpha=1, max_iter=1000)
)


hardVC = VotingClassifier(estimators=estimators, voting="hard")

softVC = VotingClassifier(estimators=estimators, voting="soft")

ensemble_names = [
    "StackingLogRegression",
    "StackingNeuralNet",
    "HardVoting",
    "SoftVoting",
]
ensembles = [SC, SCNN, hardVC, softVC]
results = []

for name, model in zip(ensemble_names, ensembles):
    print("Fitting classifier: ", name)

    clf = model.fit(X_train, y_train)
    print("Fit finished")
    try:
        y_predicted = clf.predict(X_test)
        y_test_array = np.asarray(y_test)
        misclassified = y_test_array != y_predicted
        misclassified_name = str(name) + "_misclassified.csv"
        pd.DataFrame(y_test[misclassified].index).to_csv(misclassified_name)
        print("Misclassified samples found")
        report = classification_report(
            y_test, y_predicted, target_names=["Benign", "Malware"]
        )
    except:
        print("clf.predict unavailable")
        continue
    np.set_printoptions(precision=4)
    print("Classification accuracy on training set:", clf.score(X_train, y_train))
    print("Classification accuracy on test set:", clf.score(X_test, y_test))

    results.append(
        [
            name,
            clf.score(X_train, y_train),
            clf.score(X_test, y_test),
        ]
    )
    # Save trained model
    picklefile = str(name) + "_new" + ".pickle"
    with open(picklefile, "wb") as f:
        pickle.dump(clf, f)

    try:
        y_confidence = clf.predict_proba(X_test)[:, 1]
    except:
        print("clf.predict_proba unavailable")
        continue
    y_confidence_benign = clf.predict_proba(X_test)[:, 0]
    df_results = pd.DataFrame(
        data={
            "IsMalware(1=T)": y_predicted,
            "Malware_confidence": y_confidence,
            "Benign_confidence": y_confidence_benign,
        },
        index=df_test.index,
    )
    result_name = str(name) + "confidence_results.csv"
    df_results.to_csv(result_name)


pd.DataFrame(results).to_csv("ensemble_accuracy.csv")
