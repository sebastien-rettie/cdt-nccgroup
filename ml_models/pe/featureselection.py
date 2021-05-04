# Compares performance of different feature selection methods
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    chi2,
    mutual_info_classif,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from preprocessing import generate_types, get_ct_feature_names

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

X_train = X_train.astype("int64")
X_test = X_test.astype("int64")

GiniFeatures = False
################# Retrieve Gini feature names
if GiniFeatures:

    clf = DecisionTreeClassifier(
        random_state=0,
        max_depth=8,
        min_samples_leaf=2,
        min_samples_split=3,
        splitter="best",
    ).fit(X_train, y_train)

    importances = clf.feature_importances_
    feature_names = get_ct_feature_names(column_trans)

    feature_importances = dict(zip(feature_names, importances))
    for key in sorted(feature_importances, key=feature_importances.get, reverse=True):
        if feature_importances[key] > 0.021:
            print(key, feature_importances[key])
        else:
            feature_importances.pop(key)
    plt.bar(*zip(*feature_importances.items()), align="center")
    plt.title("DT Feature Importance Gini")
    plt.show()
    plt.savefig("DT_gini_importantfeatures.png")

##############################################################
MethodCompare = False

if MethodCompare:
    dt = DecisionTreeClassifier(
        random_state=0,
        max_depth=5,
        min_samples_leaf=1,
        min_samples_split=2,
        splitter="best",
    ).fit(X_train, y_train)

    print("DT fitted")

    pipeline = Pipeline(
        [
            # the reduce_dim stage is populated by the param_grid
            ("reduce_dim", "passthrough"),
            (
                "classify",
                DecisionTreeClassifier(
                    random_state=0,
                    max_depth=5,
                    min_samples_leaf=1,
                    min_samples_split=2,
                    splitter="best",
                ),
            ),
        ]
    )

    N_FEATURES_OPTIONS = [10, 100, 1000]
    MD_OPTIONS = [3, 5]

    param_grid = [
        {
            "reduce_dim": [TruncatedSVD(n_iter=10), NMF(init="nndsvd", max_iter=500)],
            "reduce_dim__n_components": N_FEATURES_OPTIONS,
            "classify__max_depth": MD_OPTIONS,
        },
        {
            "reduce_dim": [SelectFromModel(dt, threshold=-np.inf)],
            "reduce_dim__max_features": N_FEATURES_OPTIONS,
            "classify__max_depth": MD_OPTIONS,
        },
        {
            "reduce_dim": [SelectKBest(chi2), SelectKBest(mutual_info_classif)],
            "reduce_dim__k": N_FEATURES_OPTIONS,
            "classify__max_depth": MD_OPTIONS,
        },
    ]

    reducer_labels = [
        "Truncated SVD",
        "NMF",
        "Gini",
        "KBest(chi2)",
        "KBest(mutual info)",
    ]

    grid = GridSearchCV(pipeline, n_jobs=1, param_grid=param_grid)
    grid.fit(X_train, y_train)

    mean_scores = np.array(grid.cv_results_["mean_test_score"])
    print(mean_scores)
    # scores are in the order of param_grid iteration, which is alphabetical
    mean_scores = mean_scores.reshape(len(MD_OPTIONS), -1, len(N_FEATURES_OPTIONS))
    mean_scores = mean_scores.max(axis=0)
    bar_offsets = np.arange(len(N_FEATURES_OPTIONS)) * (len(reducer_labels) + 1) + 0.5

    plt.figure()
    COLORS = "bgrcmyk"
    for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
        print(label)
        print(reducer_scores)
        plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

    plt.title("Comparing feature reduction techniques")
    plt.xlabel("Reduced number of features")
    plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
    plt.ylabel("Classification accuracy")
    plt.ylim((0.85, 1))
    plt.legend(loc="lower left")

    plt.show()
    plt.savefig("feature_selection_method.png")

# Indicates that mutual info is best method
# Then find optimal number of features to use
######################################################
FeatureNumber = True

if FeatureNumber:
    pipe = Pipeline(
        [
            ("anova", SelectKBest(mutual_info_classif)),
            (
                "classify",
                DecisionTreeClassifier(
                    random_state=0,
                    splitter="best",
                    min_samples_split=2,
                    min_samples_leaf=2,
                    max_depth=5,
                ),
            ),
        ]
    )

    # #############################################################################
    # Plot the cross-validation score as a function of feature number
    score_means = list()
    score_stds = list()

    N_COMPONENTS = (
        50,
        100,
        150,
        200,
        250,
        300,
        350,
        400,
    )

    for component in N_COMPONENTS:
        pipe.set_params(anova__k=component)
        this_scores = cross_val_score(pipe, X_train, y_train)
        score_means.append(this_scores.mean())
        score_stds.append(this_scores.std())

    plt.errorbar(N_COMPONENTS, score_means, np.array(score_stds))
    plt.title("Performance of DT/mutualinfo varying the number of features selected")
    plt.ylim((0.5, 1))
    plt.xlabel("Feature percentile")
    plt.ylabel("Accuracy Score")
    plt.axis("tight")
    plt.show()

    plt.savefig("feature_selection_num.png")
