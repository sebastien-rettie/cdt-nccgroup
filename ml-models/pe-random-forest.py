import numpy as np
import pandas as pd
import graphviz
import pydot
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import validation_curve
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.tree import export_text
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    make_scorer,
    recall_score,
    accuracy_score,
    precision_score,
    precision_recall_curve,
    confusion_matrix,
    plot_confusion_matrix,
    classification_report,
)
from sklearn.feature_selection._base import SelectorMixin
from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


"""
input_file = "benign/benign_dll1.csv"
input_file_2 = "benign/benign_dll2.csv"
input_file_3 = "malware/00355_malware.csv"
input_file_4 = "malware/00372_malware.csv"
input_file_5 = "malware/00373_malware.csv"
"""
# Some black magic to retrieve feature names from the encoder
def get_feature_out(estimator, feature_in):
    if hasattr(estimator, "get_feature_names"):
        if isinstance(estimator, _VectorizerMixin):
            # handling all vectorizers
            return [f"vec_{f}" for f in estimator.get_feature_names()]
        else:
            return estimator.get_feature_names(feature_in)
    elif isinstance(estimator, SelectorMixin):
        return np.array(feature_in)[estimator.get_support()]
    else:
        return feature_in


def get_ct_feature_names(ct):
    # handles all estimators, pipelines inside ColumnTransfomer
    # doesn't work when remainder =='passthrough'
    # which requires the input column names.
    output_features = []

    for name, estimator, features in ct.transformers_:
        if name != "remainder":
            if isinstance(estimator, Pipeline):
                current_features = features
                for step in estimator:
                    current_features = get_feature_out(step, current_features)
                features_out = current_features
            else:
                features_out = get_feature_out(estimator, features)
            output_features.extend(features_out)
        elif estimator == "passthrough":
            output_features.extend(ct._feature_names_in[features])

    return output_features


def generate_types(datafile):
    col_names = pd.read_csv(sum_input_file, nrows=0).columns
    dtypes = {col: "float64" for col in col_names}
    string_columns = [
        "Name0",
        "Name1",
        "Name10",
        "Name11",
        "Name12",
        "Name13",
        "Name14",
        "Name15",
        "Name16",
        "Name17",
        "Name18",
        "Name19",
        "Name2",
        "Name20",
        "Name21",
        "Name22",
        "Name23",
        "Name24",
        "Name3",
        "Name4",
        "Name5",
        "Name6",
        "Name7",
        "Name8",
        "Name9",
        "TimeDateStamp",
        "e_res",
        "e_res2",
    ]
    for column in string_columns:
        dtypes.update({column: "object"})
    # print(dtypes)
    return dtypes


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
        "\nConfusion matrix of Random Forest optimized for {} on the test data:".format(
            refit_score
        )
    )
    print(
        pd.DataFrame(
            confusion_matrix(y_test, y_pred),
            columns=["pred_neg", "pred_pos"],
            index=["neg", "pos"],
        )
    )
    return grid_search


def adjusted_classes(y_score, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_score]


def precision_recall_threshold(p, r, thresholds, t=0.5):
    """
    plots the precision recall curve and shows the current value for each
    by identifying the classifier's threshold (t).
    """

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


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc="best")
    plt.savefig("precision_recall_curve.png")


def plot_roc_curve(fpr, tpr, label=None):
    """
    The ROC curve, modified from 
    Hands-On Machine learning with Scikit-Learn and TensorFlow; p.91
    """
    plt.figure(figsize=(8, 8))
    plt.title("ROC Curve")
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], "k--")
    plt.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0, 1, 0.05), rotation=90)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc="best")
    plt.savefig("roc_curve.png")


np.set_printoptions(precision=2)
"""
df1 = pd.read_csv(input_file, dtype=generate_types(input_file))
df2 = pd.read_csv(input_file_2, dtype=generate_types(input_file_2))
df3 = pd.read_csv(input_file_3, dtype=generate_types(input_file_3))
df4 = pd.read_csv(input_file_4, dtype=generate_types(input_file_4))
df5 = pd.read_csv(input_file_5, dtype=generate_types(input_file_5))

all_data = pd.concat([df1, df2, df3, df4, df5], axis=0)

all_data.to_csv("all_data.csv", index=False)
"""
sum_input_file = "all_data.csv"
all_data = pd.read_csv(sum_input_file, dtype=generate_types(sum_input_file))

df = all_data.apply(lambda x: x.fillna(0) if x.dtype.kind in "biufc" else x.fillna("0"))

df = df.drop(columns=["TimeDateStamp"])
# Makes 1h encoding too big/costly

y = df["IsMalware"]
x = df.drop("IsMalware", axis=1)

# Feature scaling
scale_transform = MaxAbsScaler()

# One hot encoding, transforms categorical to ML friendly variables
onehot_transform = OneHotEncoder(handle_unknown="ignore")


# Some issues with using 1H encoding on tree classifers but no other way around this in sklearn
# Hash vectorise instead? or do by frequency? need to test

column_trans = ColumnTransformer(
    transformers=[
        ("Numerical", scale_transform, selector(dtype_include="number")),
        ("Categorical", onehot_transform, selector(dtype_include="object")),
    ],
    #  remainder="passthrough",
)

pipeline = Pipeline(steps=[("preprocess", column_trans)])

# Preprocessing and vectorise
x = pipeline.fit_transform(x)
# It is bad practice to have split after transform- fix
print(x.shape)
print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=4)
# default train/test split is 0.75:0.25

# skf = StratifiedKFold(n_splits=6,)

# Switches to enable tuning/plotting
previously_tuned = True
plot_validate_params = False
performance_report = True
plot_learning_curve = False


# Hyperparameter tuning, use randomised over grid search for speed
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


elif plot_validate_params == True:
    # Skip tuning if already have optimal hyperparameters, plot validation curves to verify
    # Optimal params = {'max_depth': 25, 'max_features': 20, 'min_samples_split': 5, 'n_estimators': 300}

    # Max depth validation
    max_depth_range = list(range(1, 40))

    train_scores, test_scores = validation_curve(
        DecisionTreeClassifier(
            random_state=0, splitter="random", min_samples_split=2, min_samples_leaf=2
        ),
        X_train,
        y_train,
        param_name="max_depth",
        param_range=max_depth_range,
        scoring="accuracy",
        cv=skf,
        n_jobs=1,
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots()
    plt.title("Decision Tree Max Depth Validation Curve")
    plt.xlabel("Tree Depth")
    plt.ylabel("Score")

    lw = 2
    plt.semilogx(
        max_depth_range,
        train_scores_mean,
        label="Training score",
        color="darkorange",
        lw=lw,
    )
    plt.fill_between(
        max_depth_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2,
        color="darkorange",
        lw=lw,
    )
    plt.semilogx(
        max_depth_range,
        test_scores_mean,
        label="Cross-validation score",
        color="navy",
        lw=lw,
    )
    plt.fill_between(
        max_depth_range,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.2,
        color="navy",
        lw=lw,
    )
    plt.legend(loc="best")

    plt.savefig("tree_depth_validation.png")
    """
    """
    # Min samples split validation
    samples_split_range = list(range(2, 20))
    for x in range(1, 10):
        samples_split_range.append(x * 0.1)
    samples_split_range.sort()

    train_scores, test_scores = validation_curve(
        DecisionTreeClassifier(
            random_state=0, splitter="best", min_samples_leaf=2, max_depth=5
        ),
        X_train,
        y_train,
        param_name="min_samples_split",
        param_range=samples_split_range,
        scoring="accuracy",
        cv=skf,
        n_jobs=1,
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots()
    plt.title("Decision Tree Min Samples Split Validation Curve")
    plt.xlabel("Minimum Samples Split")
    plt.ylabel("Score")

    lw = 2
    plt.semilogx(
        samples_split_range,
        train_scores_mean,
        label="Training score",
        color="darkorange",
        lw=lw,
    )
    plt.fill_between(
        samples_split_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2,
        color="darkorange",
        lw=lw,
    )
    plt.semilogx(
        samples_split_range,
        test_scores_mean,
        label="Cross-validation score",
        color="navy",
        lw=lw,
    )
    plt.fill_between(
        samples_split_range,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.2,
        color="navy",
        lw=lw,
    )
    plt.legend(loc="best")

    plt.savefig("samples_split_tree_validation.png")

    # Min samples leaf validation
    samples_leaf_range = list(range(2, 40))
    for x in range(1, 10):
        samples_leaf_range.append(x * 0.1)
    samples_leaf_range.sort()

    train_scores, test_scores = validation_curve(
        DecisionTreeClassifier(
            random_state=0, splitter="best", min_samples_split=2, max_depth=5
        ),
        X_train,
        y_train,
        param_name="min_samples_split",
        param_range=samples_leaf_range,
        scoring="accuracy",
        cv=skf,
        n_jobs=1,
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots()
    plt.title("Decision Tree Min Samples Leaf Validation Curve")
    plt.xlabel("Minimum Samples Leaf")
    plt.ylabel("Score")

    lw = 2
    plt.semilogx(
        samples_leaf_range,
        train_scores_mean,
        label="Training score",
        color="darkorange",
        lw=lw,
    )
    plt.fill_between(
        samples_leaf_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2,
        color="darkorange",
        lw=lw,
    )
    plt.semilogx(
        samples_leaf_range,
        test_scores_mean,
        label="Cross-validation score",
        color="navy",
        lw=lw,
    )
    plt.fill_between(
        samples_leaf_range,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.2,
        color="navy",
        lw=lw,
    )
    plt.legend(loc="best")

    plt.savefig("samples_leaf_tree_validation.png")

elif performance_report == True:
    clf = RandomForestClassifier(
        max_depth=10,
        max_features=20,
        min_samples_split=5,
        n_estimators=20,
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

    # confusion matrix as percentage
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

    # plt.show()
    # plt.savefig("confusion_matrix_RF.png")

    print("Classification report:\n", report)

    print("Precision recall plot\n")
    p, r, thresholds = precision_recall_curve(y_test, y_score)
    # Adjust this down to remove false negatives
    precision_recall_threshold(p, r, thresholds, 0.30)
    plot_precision_recall_vs_threshold(p, r, thresholds)

    """To use new threshold:
    predicted_proba = clf.predict_proba(X_test)
    predicted = (predicted_proba [:,1] >= threshold).astype('int')

    accuracy = accuracy_score(y_test, predicted)
    """

    print("Roc plot\n")

    fpr, tpr, auc_thresholds = roc_curve(y_test, y_score)
    print(auc(fpr, tpr))  # AUC of ROC
    plot_roc_curve(fpr, tpr, "recall_optimized")

    # Grab a single tree diagram
    tree = clf.estimators_[5]

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

    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    prob_pos = clf.predict_proba(X_test)[:, 1]
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, prob_pos, n_bins=10
    )

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="Random Forest")

    ax2.hist(
        prob_pos, range=(0, 1), bins=10, label="Random Forest", histtype="step", lw=2
    )

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title("Calibration plots  (reliability curve)")

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.savefig("calibration_RF.png")

elif plot_learning_curve == True:
    fig, ax = plt.subplots()

    clf = DecisionTreeClassifier(
        random_state=0,
        max_depth=5,
        min_samples_leaf=2,
        min_samples_split=5,
        splitter="best",
    )

    train_sizes, train_scores, test_scores = learning_curve(
        clf, X, y, train_sizes=np.linspace(0.1, 1, 10), scoring="accuracy", cv=skf,
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    plt.legend(loc="best")

    plt.xlabel("Train size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve Decision Tree")

    plt.show()
    plt.savefig("rf_learning_curve.png")

