#  Utilities for plotting and visualisation

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.model_selection import validation_curve
from sklearn.metrics import confusion_matrix


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
    plt.savefig("graphs/precision_recall_curve.png")


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
    plt.savefig("graphs/" + str(label) + "roc_curve.png")


def plot_learning_curve(clf, X, y):

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

    train_sizes, train_scores, test_scores = learning_curve(
        clf,
        X,
        y,
        train_sizes=np.linspace(0.1, 1, 10),
        scoring="accuracy",
        cv=skf,
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots()

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
    plt.title("Learning Curve")

    plt.savefig("graphs/learning_curve.png")


def plot_calibration_curve(clf, X_test, y_test):
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
    plt.savefig("graphs/calibration.png")


def plot_validation_curve(
    parameter_name, parameter_range, clf, X_train, y_train, title="Validation Curve"
):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

    train_scores, test_scores = validation_curve(
        clf,
        X_train,
        y_train,
        param_name=parameter_name,
        param_range=parameter_range,
        scoring="accuracy",
        cv=skf,
        n_jobs=1,
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots()
    plt.title(title)
    plt.xlabel(str(parameter_name))
    plt.ylabel("Score")

    lw = 2
    plt.semilogx(
        parameter_range,
        train_scores_mean,
        label="Training score",
        color="darkorange",
        lw=lw,
    )
    plt.fill_between(
        parameter_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2,
        color="darkorange",
        lw=lw,
    )
    plt.semilogx(
        parameter_range,
        test_scores_mean,
        label="Cross-validation score",
        color="navy",
        lw=lw,
    )
    plt.fill_between(
        parameter_range,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.2,
        color="navy",
        lw=lw,
    )
    plt.legend(loc="best")
    plot_file_name = "graphs/" + str(parameter_name) + "_validation.png"
    plt.savefig(plot_file_name)


def adjusted_classes(y_score, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_score]


def precision_recall_threshold(p, r, thresholds, y_score, y_test, t=0.5):
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
    plt.savefig("graphs/precision_recall_threshold.png")

    # plot the current threshold on the line
    close_default_clf = np.argmin(np.abs(thresholds - t))
    plt.plot(r[close_default_clf], p[close_default_clf], "^", c="k", markersize=15)
