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

from sklearn.tree import export_graphviz, export_text

from plotting import (
    adjusted_classes,
    plot_calibration_curve,
    plot_learning_curve,
    precision_recall_threshold,
    plot_precision_recall_vs_threshold,
    plot_roc_curve,
    plot_validation_curve,
)
from preprocessing import (
    concat_files,
    encode_scale,
    generate_types,
    get_ct_feature_names,
    preprocess_dataframe,
    grid_search_wrapper,
)

input_file_2 = "../../dataset/benign/benign-exe.csv"
input_file_3 = "../../dataset/malware/00355_malware.csv"

list_files = [input_file_2, input_file_3]

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

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)

# EITHER fit and save encoder
encoder = encode_scale().fit(X_train)
with open("encoder.pickle", "wb") as f:
    pickle.dump(encoder, f)

# OR import prefit encoder
"""with open("encoder.pickle", "rb") as f:
    encode_scale = pickle.load(f, encoding="bytes")"""

X_train = encoder.transform(X_train)
X_test = encoder.transform(X_test)

# Switches to enable hyperparameter tuning/plotting
previously_tuned = False
plot_validate_params = False
performance_report = False


clf = RandomForestClassifier(
    max_depth=5,
    max_features=5,
    min_samples_split=5,
    n_estimators=3,
    random_state=0,
)


# Hyperparameter tuning, uses grid search optimise for recall
# Skip tuning if already have optimal hyperparameters, want highest recall without compromising accuracy
if previously_tuned == False:
    clf = RandomForestClassifier(n_jobs=-1)
    param_grid = {
        "min_samples_split": [3, 5],
        "n_estimators": [3, 5],
        "max_depth": [3, 5],
        "max_features": [3, 5],
    }

    scorers = {
        "precision_score": make_scorer(precision_score),
        "recall_score": make_scorer(recall_score),
        "accuracy_score": make_scorer(accuracy_score),
    }
    grid_search_clf = grid_search_wrapper(
        clf,
        param_grid,
        scorers,
        X_train,
        X_test,
        y_train,
        y_test,
        refit_score="accuracy_score",
    )

    results = pd.DataFrame(grid_search_clf.cv_results_)
    results = results.sort_values(by="mean_test_accuracy_score", ascending=False)
    print(
        results[
            [
                "mean_test_accuracy_score",
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

    clf = RandomForestClassifier(
        max_features=10,
        min_samples_split=5,
        n_estimators=5,
        random_state=0,
    )

    plot_validation_curve(
        max_depth_name,
        max_depth_range,
        clf,
        X_train,
        y_train,
        "Max depth RF validation",
    )

    # Min samples split validation
    samples_split_range = list(range(2, 20))
    for x in range(1, 10):
        samples_split_range.append(x * 0.1)
    samples_split_range.sort()

    samples_split_name = "min_samples_split"

    clf = RandomForestClassifier(
        max_depth=5,
        max_features=10,
        n_estimators=5,
        random_state=0,
    )

    plot_validation_curve(
        samples_split_name,
        samples_split_range,
        clf,
        X_train,
        y_train,
        "Min sample split RF validation",
    )


elif performance_report:
    clf = RandomForestClassifier(
        max_depth=3,
        max_features=5,
        min_samples_split=5,
        n_estimators=3,
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

    print("Feature Importances")

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

    """
    To use new threshold:
    predicted_proba = clf.predict_proba(X_test)
    predicted = (predicted_proba [:,1] >= threshold).astype('int')

    accuracy = accuracy_score(y_test, predicted)
    """

    print("Roc plot\n")
    fpr, tpr, auc_thresholds = roc_curve(y_test, y_score)
    print("Area under ROC curve:", auc(fpr, tpr))  # AUC of ROC
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
