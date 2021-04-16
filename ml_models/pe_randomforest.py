import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydot
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    make_scorer,
    plot_confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)

from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text

from pe_plotting import (
    plot_calibration_curve,
    plot_learning_curve,
    precision_recall_threshold,
    plot_precision_recall_vs_threshold,
    plot_roc_curve,
    plot_validation_curve,
)
from pe_preprocessing import (
    concat_files,
    encode_scale,
    generate_types,
    get_ct_feature_names,
    get_feature_out,
    preprocess_dataframe,
)

input_file_2 = "../dataset/benign/benign-exe.csv"
input_file_3 = "../dataset/malware/00355_malware.csv"

list_files = [input_file_2, input_file_3]


def grid_search_wrapper(refit_score="precision_score"):
    """
    fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics
    """
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    grid_search = GridSearchCV(
        clf,
        param_grid,
        scoring=scorers,
        refit=refit_score,
        cv=skf,
        return_train_score=True,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    # make the predictions
    y_pred = grid_search.predict(X_test)

    print("Best params for {}".format(refit_score))
    print(grid_search.best_params_)

    # confusion matrix on the test data.
    print(
        pd.DataFrame(
            confusion_matrix(y_test, y_pred),
            columns=["pred_neg", "pred_pos"],
            index=["neg", "pos"],
        )
    )
    return grid_search


"""def adjusted_classes(y_score, t):
   
    return [1 if y >= t else 0 for y in y_score]


def precision_recall_threshold(p, r, thresholds, t=0.5):
   

    # generate new class predictions based on the adjusted_classes
    # function above and view the resulting confusion matrix.
    y_pred_adj = adjusted_classes(y_score, t)
    print(
        pd.DataFrame(
            confusion_matrix(y_test, y_pred_adj),
            columns=["pred_neg", "pred_pos"],
            index=["neg", "pos"],
        )
    )

    # plot the curve
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall curve ^ = current threshold")
    plt.step(r, p, color="b", alpha=0.2, where="post")
    plt.fill_between(r, p, step="post", alpha=0.2, color="b")
    plt.ylim([0.5, 1.01])
    plt.xlim([0.5, 1.01])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig("precision_recall_threshold.png")

    # plot the current threshold on the line
    close_default_clf = np.argmin(np.abs(thresholds - t))
    plt.plot(r[close_default_clf], p[close_default_clf], "^", c="k", markersize=15)
"""

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
"""column_trans = encode_scale().fit(X)
with open("encoder.pickle", "wb") as f:
    pickle.dump(column_trans, f)"""

# OR import prefit encoder
with open("encoder.pickle", "rb") as f:
    column_trans = pickle.load(f, encoding="bytes")

X = column_trans.transform(X)


# Note it is bad practice to have split after transform- fix
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)

# Switches to enable hyperparamttuning/plotting
previously_tuned = True
plot_validate_params = False
performance_report = True


clf = RandomForestClassifier(
    max_depth=5,
    max_features=10,
    min_samples_split=5,
    n_estimators=5,
    random_state=0,
)


# Hyperparameter tuning, uses grid search optimise for recall
# Skip tuning if already have optimal hyperparameters
if previously_tuned == False:
    clf = RandomForestClassifier(n_jobs=-1)
    param_grid = {
        "min_samples_split": [3, 5, 10],
        "n_estimators": [100, 300],
        "max_depth": [3, 5, 15, 25],
        "max_features": [3, 5, 10, 20],
    }

    scorers = {
        "precision_score": make_scorer(precision_score),
        "recall_score": make_scorer(recall_score),
        "accuracy_score": make_scorer(accuracy_score),
    }
    grid_search_clf = grid_search_wrapper(refit_score="recall_score")

    results = pd.DataFrame(grid_search_clf.cv_results_)
    results = results.sort_values(by="mean_test_precision_score", ascending=False)
    print(
        results[
            [
                "mean_test_precision_score",
                "mean_test_recall_score",
                "mean_test_accuracy_score",
                "param_max_depth",
                "param_max_features",
                "param_min_samples_split",
                "param_n_estimators",
            ]
        ]
        .round(3)
        .head()
    )


elif plot_validate_params:
    # Plot validation curves to verify hyperparameters
    # Optimal params = {'max_depth': 25, 'max_features': 20, 'min_samples_split': 5, 'n_estimators': 300}

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
    clf = RandomForestClassifier(
        max_depth=5,
        max_features=10,
        min_samples_split=5,
        n_estimators=5,
        random_state=0,
    ).fit(X_train, y_train)

    # with optimised parameters
    # Optimal params = {'max_depth': 25, 'max_features': 20, 'min_samples_split': 5, 'n_estimators': 300}

    y_predicted = clf.predict(X_test)

    y_score = clf.predict_proba(X_test)[:, 1]
    print("Area under ROC curve score:")
    print(roc_auc_score(y_test, y_score))

    print(
        "Accuracy of Random Forest classifier on training set: {:.2f}".format(
            clf.score(X_train, y_train)
        )
    )
    print(
        "Accuracy of Random Forest classifier on test set: {:.2f}".format(
            clf.score(X_test, y_test)
        )
    )
    report = classification_report(
        y_test, y_predicted, target_names=["Benign", "Malware"]
    )
    np.set_printoptions(precision=4)

    print("Feature Importances", clf.feature_importances_)
    importances = clf.feature_importances_
    feature_names = get_ct_feature_names(column_trans)

    feature_importances = dict(zip(feature_names, importances))
    for key in sorted(feature_importances, key=feature_importances.get, reverse=True):
        if feature_importances[key] > 0.01:
            print(key, feature_importances[key])

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
    plt.savefig("confusion_matrix_RF.png")

    # Confusion matrix as percentage
    cm = confusion_matrix(y_test, y_predicted)
    cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        cmn, annot=True, fmt=".4f", xticklabels=["FP", "TP"], yticklabels=["TN", "FN"]
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show(block=False)
    plt.savefig("pc_confusion_matrix_RF.png")

    print("Classification report:\n", report)

    print("Precision recall plot\n")
    p, r, thresholds = precision_recall_curve(y_test, y_score)
    # Adjust this down to remove false negatives
    precision_recall_threshold(p, r, thresholds, y_score, y_test, 0.30)
    plot_precision_recall_vs_threshold(p, r, thresholds)

    """To use new threshold:
    predicted_proba = clf.predict_proba(X_test)
    predicted = (predicted_proba [:,1] >= threshold).astype('int')

    accuracy = accuracy_score(y_test, predicted)
    """

    print("Roc plot\n")
    fpr, tpr, auc_thresholds = roc_curve(y_test, y_score)
    print("Area under ROC curve:"auc(fpr, tpr))  # AUC of ROC
    plot_roc_curve(fpr, tpr, "recall_optimized")

    # Grab a single tree diagram
    print("Tree diagram\n")
    tree = clf.estimators_[2]

    tree_structure = export_text(tree, feature_names=get_ct_feature_names(column_trans))
    dot_data = export_graphviz(
        tree,
        out_file="rf_tree.dot",
        feature_names=get_ct_feature_names(column_trans),
        filled=True,
        rounded=True,
        special_characters=True,
    )
    (graph,) = pydot.graph_from_dot_file("rf_tree.dot")
    graph.write_png("rf_tree.png")

    print("Calibration curve\n")
    plot_calibration_curve(clf, X_test, y_test)

    print("Learning curve")
    plot_learning_curve(clf, X, y)
