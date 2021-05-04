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

from sklearn.metrics import (
    auc,
    classification_report,
    plot_confusion_matrix,
    roc_curve,
    plot_roc_curve,
)

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


with open("encoder.pickle", "rb") as f:
    column_trans = pickle.load(f, encoding="bytes")

X_train = column_trans.transform(X_train)
X_test = column_trans.transform(X_test)

with open("selector.pickle", "rb") as f:
    selector = pickle.load(f, encoding="bytes")

X_train = selector.transform(X_train)
X_test = selector.transform(X_test)

with open("scale.pickle", "rb") as f:
    scale_transform = pickle.load(f, encoding="bytes")

X_train = scale_transform.transform(X_train)
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


fig, ax = plt.subplots()

results = []
model_displays = {}
scoring = "accuracy"


for name, model in zip(names, classifiers):

    print("Predicting with", name)

    picklefile = str(name) + ".pickle"

    clf = model.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    print(clf.score(X_train, y_train), clf.score(X_test, y_test))
    try:
        y_score = clf.predict_proba(X_test)[:, 1]
    except:
        print("clf.predict_proba unavailable")
        continue
    print("Roc plot\n")
    fpr, tpr, auc_thresholds = roc_curve(y_test, y_score)
    print("Area under ROC curve:", auc(fpr, tpr))  # AUC of ROC
    plot_roc_curve(fpr, tpr, name)

    disp = plot_confusion_matrix(
        clf,
        X_test,
        y_test,
        display_labels=["Benign", "Malware"],
        cmap=plt.cm.get_cmap("hot"),
    )
    disp.ax_.set_title(name + "Confusion Matrix")
    print(disp.confusion_matrix)
    plt.savefig(name + "confusion_matrix.png")

    np.set_printoptions(precision=4)

    print("Saving model")
    with open(picklefile, "wb") as f:
        pickle.dump(clf, f)

    print("Plotting roc")
    model_displays[name] = plot_roc_curve(clf, X_test, y_test, ax=ax, name=name)

    report = classification_report(
        y_test, y_predicted, target_names=["Benign", "Malware"]
    )
    results.append(
        [name, clf.score(X_train, y_train), clf.score(X_test, y_test), report]
    )

    print("Classification report:\n", report)


plt.figure(1)
plt.plot([0, 1], [0, 1], "k--")
for model in results:
    plt.plot(model[3], model[4], label=model[0])
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve")
plt.legend(loc="best")
plt.savefig("graphs/all_models_roc_curve.png")
plt.show()

_ = ax.set_title("ROC curve")
ax.set_xlim(0, 0.2)
ax.set_ylim(0.8, 1)
plt.savefig("graphs/multi_model_roc_curve.png")

accuracy_results = dict(zip(names, results))
print(accuracy_results)

with open("scaling_multi_model_results.csv", "w") as file:
    w = csv.writer(file)
    w.writerows(accuracy_results.items())
