#  For large datasets consider use LinearSVC or SGDClassifier, possibly after a Nystroem transformer.
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import (
    RandomizedSearchCV,
)


from sklearn.metrics import (
    auc,
    classification_report,
    plot_confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from sklearn.kernel_approximation import Nystroem


from preprocessing import (
    generate_types,
)

from plotting import (
    precision_recall_threshold,
    plot_precision_recall_vs_threshold,
    plot_roc_curve,
)


clf = SGDClassifier(max_iter=2000, tol=1e-3)


feature_map_nystroem = Nystroem(gamma=0.2, random_state=1, n_components=250)

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
##############
print("finished feature selection")

# Nystroem transform
feature_map_nystroem.fit(X_train)

report = False
if report:
    clf.fit(X_train, y_train)
    print("score", clf.score(X_test, y_test))

    y_predicted = clf.predict(X_test)

    y_test_array = np.asarray(y_test)
    misclassified = y_test_array != y_predicted

    print("Misclassified samples:", y_test[misclassified].index)

    y_score = clf.decision_function(X_test)
    print("Area under ROC curve score:")
    print(roc_auc_score(y_test, y_score))

    print(
        "Accuracy of SVM classifier on training set: {:.2f}".format(
            clf.score(X_train, y_train)
        )
    )
    print(
        "Accuracy of SVM classifier on test set: {:.2f}".format(
            clf.score(X_test, y_test)
        )
    )
    report = classification_report(
        y_test, y_predicted, target_names=["Benign", "Malware"]
    )
    np.set_printoptions(precision=4)
    print("Classification report:\n", report)

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
    plt.savefig("confusion_matrix_SVM.png")

    print("Roc plot\n")
    p, r, thresholds = precision_recall_curve(y_test, y_score)
    # Adjust this down to remove false negatives
    precision_recall_threshold(p, r, thresholds, y_score, y_test, 0.30)
    plot_precision_recall_vs_threshold(p, r, thresholds)
    fpr, tpr, auc_thresholds = roc_curve(y_test, y_score)
    print("Area under ROC curve:", auc(fpr, tpr))  # AUC of ROC
    plot_roc_curve(fpr, tpr, "accuracy_optimized")

tuning = True
if tuning:

    tuned_parameters = [
        {
            "loss": ["hinge", "log", "modified_huber", "perceptron"],
            "alpha": [0.1, 1, 10, 100, 1000],
        }
    ]

    scores = ["recall"]

    n_iter_search = 50

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)

        clf = RandomizedSearchCV(
            SGDClassifier(),
            tuned_parameters,
            n_iter=n_iter_search,
            scoring="%s_macro" % score,
            cv=7,
        )
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
