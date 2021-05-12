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

from sklearn.tree import export_graphviz, export_text

from plotting import (
    plot_calibration_curve,
    plot_learning_curve,
    precision_recall_threshold,
    plot_precision_recall_vs_threshold,
    plot_roc_curve,
    plot_validation_curve,
)
from preprocessing import (
    generate_types,
    get_ct_feature_names,
    grid_search_wrapper,
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

with open("scale.pickle", "rb") as f:
    scale_transform = pickle.load(f, encoding="bytes")

X_train = scale_transform.transform(X_train)
X_test = scale_transform.transform(X_test)

print("finished feature selection")

# Switches to enable hyperparameter tuning/plotting
previously_tuned = False
plot_validate_params = True
performance_report = True


clf = RandomForestClassifier(
    max_depth=5,
    max_features=5,
    min_samples_split=5,
    n_estimators=3,
    random_state=0,
)


# Hyperparameter tuning, uses grid search optimise for recall
# Skip tuning if already have optimal hyperparameters, want highest recall without compromising accuracy
if not previously_tuned:
    clf = RandomForestClassifier(n_jobs=-1)
    param_grid = {
        "min_samples_split": [0.1, 0.2, 0.3, 4, 5, 10, 15],
        "n_estimators": [3, 5, 10, 20, 40],
        "max_depth": [2, 5, 10, 20],
        "max_features": [3, 5, 10, 20],
        "min_samples_leaf": [0.2, 2, 10],
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
    max_depth_range = list(range(1, 40))

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

    # max_features validation
    max_features_range = list(range(2, 20))

    max_features_name = "max_features"

    clf = RandomForestClassifier(
        max_depth=5,
        min_samples_split=5,
        n_estimators=5,
        random_state=0,
    )

    plot_validation_curve(
        max_features_name,
        max_features_range,
        clf,
        X_train,
        y_train,
        "Max features RF validation",
    )

    # min_leaf validation
    samples_leaf_range = list(range(2, 20))
    for x in range(1, 5):
        samples_leaf_range.append(x * 0.1)
    samples_leaf_range.sort()

    samples_leaf_name = "min_samples_leaf"

    clf = RandomForestClassifier(
        max_depth=5,
        min_samples_split=5,
        n_estimators=5,
        random_state=0,
    )

    plot_validation_curve(
        samples_leaf_name,
        samples_leaf_range,
        clf,
        X_train,
        y_train,
        "Min leaf RF validation",
    )

    # n_estimators validation
    n_estimators_range = list(range(3, 300))

    n_estimators_name = "n_estimators"

    clf = RandomForestClassifier(
        max_depth=5,
        min_samples_split=5,
        max_features=10,
        random_state=0,
    )

    plot_validation_curve(
        n_estimators_name,
        n_estimators_range,
        clf,
        X_train,
        y_train,
        "n estimators RF validation",
    )


elif performance_report:
    clf = RandomForestClassifier(
        max_depth=5,
        max_features=10,
        min_samples_split=7,
        n_estimators=5,
        random_state=0,
    ).fit(X_train, y_train)

    y_predicted = clf.predict(X_test)

    y_test_array = np.asarray(y_test)
    misclassified = y_test_array != y_predicted

    print("Misclassified samples:", y_test[misclassified].index)

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
    # Adjust this threshold down to remove false negatives
    precision_recall_threshold(p, r, thresholds, y_score, y_test, 0.30)
    plot_precision_recall_vs_threshold(p, r, thresholds)

    # To use new threshold:
    threshold = 0.30
    predicted_proba = clf.predict_proba(X_test)
    predicted = (predicted_proba[:, 1] >= threshold).astype("int")

    accuracy = accuracy_score(y_test, predicted)

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
    plot_learning_curve(clf, X_test, y_test)
    print("score", clf.score(X_test, y_test))
