import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import graphviz
import pydot
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    plot_confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text

from plotting import (
    adjusted_classes,
    plot_calibration_curve,
    plot_learning_curve,
    plot_precision_recall_vs_threshold,
    plot_validation_curve,
    precision_recall_threshold,
)
from preprocessing import (
    concat_files,
    encode_scale,
    generate_types,
    get_ct_feature_names,
    preprocess_dataframe,
)

# from joblib import parallel_backend
# todo: implement parallelism


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")


np.set_printoptions(precision=2)

# EITHER concatenate disparate data files
"""
df = concat_files(list_files)
"""
# OR if reading from pre-concatenated data
input_file = "all_data.csv"
df = pd.read_csv(input_file, dtype=generate_types(input_file))

df = preprocess_dataframe(df)

y = df["IsMalware"]
X = df.drop("IsMalware", axis=1)

# EITHER fit and save encoder
column_trans = encode_scale().fit(X)
with open("encoder.pickle", "wb") as f:
    pickle.dump(column_trans, f)

# OR import prefit encoder
"""with open("encoder.pickle", "rb") as f:
    column_trans = pickle.load(f, encoding="bytes")"""

X = column_trans.transform(X)


# Note it is bad practice to have split after transform- fix
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)


skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=1)

# Switches to enable tuning/plotting
previously_tuned = True
plot_validate_params = False
performance_report = True


# Hyperparameter tuning, use randomised over grid search for speed
if previously_tuned == False:
    tuned_parameters = [
        {
            "splitter": ["best", "random"],
            "max_depth": [2, 5, 10, 20, 40],
            "min_samples_split": [0.1, 0.2, 0.3, 4, 5, 10, 15],
            "min_samples_leaf": [0.2, 2, 10],
        },
    ]

    scores = ["precision", "recall"]
    n_iter_search = 40

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)

        clf = RandomizedSearchCV(
            DecisionTreeClassifier(random_state=0),
            tuned_parameters,
            n_iter=n_iter_search,
            scoring="%s_macro" % score,
        )
        # note GridSearch automatically uses StratKFold x-validation
        clf.fit(X_train, y_train)
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_["mean_test_score"]
        stds = clf.cv_results_["std_test_score"]
        for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
elif plot_validate_params:
    # Skip tuning if already have optimal hyperparameters, plot validation curves to verify
    # Optimal params = {'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 2, 'splitter': 'best'}

    # Max depth validation
    max_depth_range = list(range(1, 4))

    max_depth_name = "max_depth"

    clf = DecisionTreeClassifier(
        random_state=0, splitter="random", min_samples_split=2, min_samples_leaf=2
    )

    plot_validation_curve(
        max_depth_name, max_depth_range, clf, X_train, y_train, "Max depth validation"
    )

    # Min samples split validation
    samples_split_range = list(range(2, 20))
    for x in range(1, 10):
        samples_split_range.append(x * 0.1)
    samples_split_range.sort()

    samples_split_name = "min_samples_split"

    clf = DecisionTreeClassifier(
        random_state=0, splitter="best", min_samples_leaf=2, max_depth=5
    )

    plot_validation_curve(
        samples_split_name,
        samples_split_range,
        clf,
        X_train,
        y_train,
        "Min samp validation",
    )

elif performance_report:
    clf = DecisionTreeClassifier(
        random_state=0,
        max_depth=10,
        min_samples_leaf=2,
        min_samples_split=5,
        splitter="best",
    ).fit(X_train, y_train)

    y_predicted = clf.predict(X_test)

    y_score = clf.predict_proba(X_test)[:, 1]
    print("Area under ROC curve score:")
    print(roc_auc_score(y_test, y_score))

    print(
        "Accuracy of Decision Tree classifier on training set: {:.3f}".format(
            clf.score(X_train, y_train)
        )
    )
    print(
        "Accuracy of Decision Tree classifier on test set: {:.3f}".format(
            clf.score(X_test, y_test)
        )
    )

    report = classification_report(
        y_test, y_predicted, target_names=["Benign", "Malware"]
    )

    print("Classification report:\n", report)

    print("Feature Importances")

    importances = clf.feature_importances_
    feature_names = get_ct_feature_names(column_trans)

    feature_importances = dict(zip(feature_names, importances))
    for key in sorted(feature_importances, key=feature_importances.get, reverse=True):
        if feature_importances[key] > 0.01:
            print(key, feature_importances[key])

    print("Roc plot\n")

    tn, fp, fn, tp = confusion_matrix(y_test, y_predicted, normalize="true").ravel()
    print("fp", fp)
    print("tp", tp)
    plt.figure()
    lw = 2
    plt.scatter(
        fp,
        tp,
        color="darkorange",
        lw=lw,
        label="decision tree classifier",
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--", label="random guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positives")
    plt.ylabel("True Positives")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")

    plt.savefig("roc_plot.png")
    plt.show()

    # Create tree diagram
    tree_structure = export_text(clf, feature_names=get_ct_feature_names(column_trans))
    dot_data = export_graphviz(
        clf,
        out_file="decision_tree.dot",
        feature_names=get_ct_feature_names(column_trans),
        filled=True,
        rounded=True,
        special_characters=True,
    )
    (graph,) = pydot.graph_from_dot_file("decision_tree.dot")
    graph.write_png("decision_tree.png")

    # Print confusion matrix
    disp = plot_confusion_matrix(
        clf,
        X_test,
        y_test,
        display_labels=["Benign", "Malware"],
        cmap=plt.cm.get_cmap("Spectral"),
    )
    disp.ax_.set_title("Confusion Matrix")
    print(disp.confusion_matrix)
    plt.savefig("DT_confusion_matrix.png")

    print("Learning curve")
    plot_learning_curve(clf, X, y)
