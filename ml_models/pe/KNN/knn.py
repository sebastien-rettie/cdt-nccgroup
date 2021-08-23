# k nearest neighbors
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from datetime import date

from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    make_scorer,
    plot_confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


from preprocessing import (
    generate_types,
    grid_search_wrapper,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
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

# Switches to enable hyperparameter tuning/plotting
previously_tuned = True
performance_report = True
alt_tuning = False


clf = Pipeline([("classifier", KNeighborsClassifier(n_neighbors=3, p=2))])


if performance_report:
    start_time = time.time()

    clf.fit(X_train, y_train)
    print("score")
    print(clf.score(X_test, y_test))

    y_predicted = clf.predict(X_test)

    y_test_array = np.asarray(y_test)
    misclassified = y_test_array != y_predicted

    print("Misclassified samples:", y_test[misclassified].index)

    y_score = clf.predict_proba(X_test)[:, 1]
    print("Area under ROC curve score:")
    print(roc_auc_score(y_test, y_score))

    print(
        "Accuracy of k nearest neighbors classifier on training set: {:.2f}".format(
            clf.score(X_train, y_train)
        )
    )
    print(
        "Accuracy of k nearest neighbors classifier on test set: {:.2f}".format(
            clf.score(X_test, y_test)
        )
    )
    report = classification_report(
        y_test, y_predicted, target_names=["Benign", "Malware"]
    )
    np.set_printoptions(precision=4)
    print("Classification report:\n", report)

    disp = plot_confusion_matrix(
        clf,
        X_test,
        y_test,
        display_labels=["Benign", "Malware"],
        cmap=plt.cm.get_cmap("hot"),
    )
    disp.ax_.set_title("Confusion Matrix")
    print(disp.confusion_matrix)
    plt.savefig("graphs/confusion_matrix_kNN.png")

    end_time = time.time() - start_time

    try:
        reader = open('performance_report.txt', 'r')
        prev_text = reader.read()
        reader.close()
    except FileNotFoundError:
        open('performance_report.txt', 'x')

    done = open('performance_report.txt', 'w')

    try:
        done.write(prev_text)
    except:
        pass

    done.write('\n==========================================')
    done.write('\nDATE: {0}'.format(date.today()))

    done.write('\nA performance report has been conducted on knn.py! The time taken was {0:0.3f}s.'.format(end_time))
    done.write('\n\nSelected Parameters:\n')

    param_data=[('n_neighbours', 3),('p', 2)]

    try:
        for i in param_data:
            done.write('\t{0}: {1}\n'.format(i[0], i[1]))
    except:
        done.write('No parameter data found.')

    done.write('\nThe performance report was as follows:\n')

    try:
        done.write('{0}'.format(report))
    except:
        done.write('No report was found.')

    done.write("\nFinally, the KNN score on the test set: {0}.".format(clf.score(X_test, y_test)))
    done.close()

# Hyperparameter tuning, uses grid search to optimise for recall
# Skip tuning if already have optimal hyperparameters
if not previously_tuned:
    clf = KNeighborsClassifier(n_jobs=-1)
    param_grid = {
        "n_neighbors": [3, 4, 5, 6, 7],
        "p": [2],
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
    results = results.sort_values(by="mean_test_recall_score", ascending=False)
    print(
        results[
            [
                "mean_test_accuracy_score",
                "mean_test_precision_score",
                "mean_test_recall_score",
                "mean_test_accuracy_score",
                "param_n_neighbors",
            ]
        ]
        .round(3)
        .head()
    )
if alt_tuning:
    print("begin tuning")
    tuned_parameters = {
        "n_neighbors": [3, 4, 5],
        "p": [1, 2],
    }
    scores = ["recall"]

    n_iter_search = 50
    print("# Tuning hyper-parameters for recall")

    clf = RandomizedSearchCV(
        KNeighborsClassifier(n_jobs=-1),
        tuned_parameters,
        n_iter=n_iter_search,
        scoring="recall_macro",
        cv=7,
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
