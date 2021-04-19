# Compares performance of different feature selection methods

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import (
    SelectPercentile,
    SelectKBest,
    chi2,
    mutual_info_classif,
)

from sklearn.feature_selection._base import SelectorMixin
from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from pe_preprocessing import (
    generate_types,
    concat_files,
    preprocess_dataframe,
    encode_scale,
)

input_file_2 = "../dataset/benign/benign-exe.csv"
input_file_3 = "../dataset/malware/00355_malware.csv"

list_files = [input_file_2, input_file_3]

np.set_printoptions(precision=2)

# EITHER concatenate disparate data files
"""df = concat_files(list_files)
"""
# OR if reading from pre-concatenated data
input_file = "all_data.csv"
df = pd.read_csv(input_file, dtype=generate_types(input_file))

df = preprocess_dataframe(df)

y = df["IsMalware"]
X = df.drop("IsMalware", axis=1)

# EITHER fit and save encoder
"""encoder = encode_scale().fit(X)
with open("encoder.pickle", "wb") as f:
    pickle.dump(encoder, f)"""

# OR import prefit encoder
with open("encoder.pickle", "rb") as f:
    encoder = pickle.load(f, encoding="bytes")

X = encoder.transform(X)

pipeline = Pipeline(
    [
        # the reduce_dim stage is populated by the param_grid
        ("reduce_dim", "passthrough"),
        (
            "classify",
            DecisionTreeClassifier(
                random_state=0,
                splitter="best",
                min_samples_split=2,
                min_samples_leaf=2,
            ),
        ),
    ]
)


N_FEATURES_OPTIONS = [4, 5]
MD_OPTIONS = [3, 5]
# , NMF()
param_grid = [
    {
        "reduce_dim": [PCA(iterated_power=7)],
        "reduce_dim__n_components": N_FEATURES_OPTIONS,
        "classify__max_depth": MD_OPTIONS,
    },
    {
        "reduce_dim": [SelectKBest(chi2), SelectKBest(mutual_info_classif)],
        "reduce_dim__k": N_FEATURES_OPTIONS,
        "classify__max_depth": MD_OPTIONS,
    },
]
reducer_labels = ["PCA", "KBest(chi2)", "KBest(mutual info"]
estimator = DecisionTreeClassifier(
    random_state=0,
    splitter="best",
    min_samples_split=2,
    min_samples_leaf=2,
)
print(estimator.get_params().keys())

grid = GridSearchCV(pipeline, n_jobs=1, param_grid=param_grid)
grid.fit(X, y)

mean_scores = np.array(grid.cv_results_["mean_test_score"])
# scores are in the order of param_grid iteration, which is alphabetical
mean_scores = mean_scores.reshape(len(MD_OPTIONS), -1, len(N_FEATURES_OPTIONS))
# select score for best C
mean_scores = mean_scores.max(axis=0)
bar_offsets = np.arange(len(N_FEATURES_OPTIONS)) * (len(reducer_labels) + 1) + 0.5

plt.figure()
COLORS = "bgrcmyk"
for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
    plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

plt.title("Comparing feature reduction techniques")
plt.xlabel("Reduced number of features")
plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
plt.ylabel("Classification accuracy")
plt.ylim((0, 1))
plt.legend(loc="upper left")

plt.show()
plt.savefig("feature_selection_method.png")

# Indicates that chi-squared is best method
# Then find optimal number of features to use


clf = Pipeline(
    [
        # the reduce_dim stage is populated by the param_grid
        ("anova", SelectPercentile(chi2)),
        (
            "classify",
            DecisionTreeClassifier(
                random_state=0,
                splitter="best",
                min_samples_split=2,
                min_samples_leaf=2,
            ),
        ),
    ]
)


# #############################################################################
# Plot the cross-validation score as a function of feature number
score_means = list()
score_stds = list()

percentiles = (1, 3, 6)

for percentile in percentiles:
    clf.set_params(anova__percentile=percentile)
    this_scores = cross_val_score(clf, X, y)
    score_means.append(this_scores.mean())
    score_stds.append(this_scores.std())


plt.errorbar(percentiles, score_means, np.array(score_stds))
plt.title("Performance of the DT-Anova varying the percentile of features selected")
plt.xticks(np.linspace(0, 100, 11, endpoint=True))
plt.xlabel("Percentile")
plt.ylabel("Accuracy Score")
plt.axis("tight")
plt.show()

plt.savefig("feature_selection_num.png")
