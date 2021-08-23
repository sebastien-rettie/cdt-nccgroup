import pickle

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydot

from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    plot_confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
)

from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.preprocessing import StandardScaler

from plotting import (
    plot_learning_curve,
    plot_validation_curve,
)
from preprocessing import (
    generate_types,
    get_ct_feature_names,
    preprocess_dataframe,
    encode_scale,
)
import time
from datetime import date
import smtplib
from email.mime.text import MIMEText

start_time = time.time()

# Reading in preprocessed data
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

# EITHER fit and save encoder
prefit_encoder = True
prefit_selector = True
prefit_scaler = True

if not prefit_encoder:
    print('Encoder not found, building one...')
    encoder = encode_scale().fit(X_train)
    with open("encoder.pickle", "wb") as f:
        pickle.dump(encoder, f)
else:
    # OR import prefit encoder
    with open("encoder.pickle", "rb") as f:
        column_trans = pickle.load(f, encoding="bytes")
        encoder = column_trans

X_train = encoder.transform(X_train)
X_test = encoder.transform(X_test)

if not prefit_selector:
    print('Selector not found, building one...')
    selector = SelectKBest(mutual_info_classif, k=300)
    selector.fit(X_train, y_train)
    with open("selector.pickle", "wb") as f:
        pickle.dump(selector, f)
else:
    with open("selector.pickle", "rb") as f:
        selector = pickle.load(f, encoding="bytes")

X_train = selector.transform(X_train)
X_test = selector.transform(X_test)

if not prefit_scaler:
    print('Scaler not found, building one...')
    scale_transform = StandardScaler(with_mean=False)
    scale_transform.fit(X_train)
    with open("scale.pickle", "wb") as f:
        pickle.dump(scale_transform, f)
else:
    with open("scale.pickle", "rb") as f:
        scale_transform = pickle.load(f, encoding="bytes")

X_train = scale_transform.transform(X_train)
X_test = scale_transform.transform(X_test)

# Switches to enable tuning/plotting
previously_tuned = True
plot_validate_params = False
performance_report = True

# Hyperparameter tuning, use randomised over grid search for speed
if not previously_tuned:
    param_data = []
    tuned_parameters = [
        {
            "splitter": ["best"],
            "max_depth": [5],
            "min_samples_split": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 2],
            "min_samples_leaf": [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 5, 10, 20],
        },
    ]

    scores = ["f1", "recall"]

    n_iter_search = 50

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)

        clf = RandomizedSearchCV(
            DecisionTreeClassifier(random_state=0),
            tuned_parameters,
            n_iter=n_iter_search,
            scoring="%s_macro" % score,
            cv=7,
        )
        # note GridSearch automatically uses StratKFold x-validation
        clf.fit(X_train, y_train)
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)

        param_data.append([score, clf.best_params_])

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

        #Save report as variable for txt report
        report = classification_report(y_true, y_pred)

        print()

        end_time = time.time() - start_time

        try:
            reader = open('hyperopt_tuning.txt', 'r')
            prev_text = reader.read()
            reader.close()
        except FileNotFoundError:
            open('hyperopt_tuning.txt', 'x')

        done = open('hyperopt_tuning.txt', 'w')

        try:
            done.write(prev_text)
        except:
            pass

        done.write('\n==========================================')
        done.write('\nDATE: {0}'.format(date.today()))

        done.write('Hyperparameter optimisation has been carried out in decisiontree.py! The time taken was {0:0.3f}s.'.format(end_time))
        done.write('\n\nHyperparameter data:\n')

        try:
            for i in param_data:
                done.write('\t{0} params: {1}\n'.format(i[0], i[1]))
        except:
            done.write('No parameter data found.')

        done.close()

elif plot_validate_params:
    # Plot validation curves to verify hyperparameters for overfit
    print('Plotting validation curves for your parameters...')

    # Max depth validation
    max_depth_range = list(range(1, 40))

    max_depth_name = "max_depth"

    clf = DecisionTreeClassifier(
        random_state=0, splitter="best", min_samples_split=2, min_samples_leaf=2
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

    # Min samples leaf validation
    samples_leaf_range = list(range(2, 20))
    for x in range(1, 5):
        samples_leaf_range.append(x * 0.1)
    samples_leaf_range.sort()

    samples_leaf_name = "min_samples_leaf"

    clf = DecisionTreeClassifier(
        random_state=0, splitter="best", min_samples_split=2, max_depth=5
    )

    plot_validation_curve(
        samples_leaf_name,
        samples_leaf_range,
        clf,
        X_train,
        y_train,
        "Min leaf validation",
    )

elif performance_report:
    print("Writing performance report...")

    clf = DecisionTreeClassifier(
        random_state=0,
        max_depth=10,
        min_samples_leaf=2,
        min_samples_split=2,
        splitter="best",
    ).fit(X_train, y_train)

    print("score", clf.score(X_test, y_test))
    y_predicted = clf.predict(X_test)

    y_test_array = np.asarray(y_test)
    misclassified = y_test_array != y_predicted

    print("Misclassified samples:", y_test[misclassified].index)

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
    test_score = clf.score(X_test, y_test)

    report = classification_report(
        y_test, y_predicted, target_names=["Benign", "Malware"]
    )

    print("Classification report:\n", report)

    print("Feature Importances")
    # Possibly broken by the addition of feature selection

    importances = clf.feature_importances_
    feature_names = get_ct_feature_names(encoder)

    feature_importances = dict(zip(feature_names, importances))
    for key in sorted(feature_importances, key=feature_importances.get, reverse=True):
        if feature_importances[key] > 0.01:
            print(key, feature_importances[key])

    print("Printing ROC plot\n")

    tn, fp, fn, tp = confusion_matrix(y_test, y_predicted, normalize="true").ravel()

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

    plt.savefig("graphs/roc_plot.png")
    plt.show()

    # Create tree diagram
    """tree_structure = export_text(clf, feature_names=get_ct_feature_names(encoder))
    dot_data = export_graphviz(
        clf,
        out_file="decision_tree.dot",
        feature_names=get_ct_feature_names(encoder),
        filled=True,
        rounded=True,
        special_characters=True,
    )
    (graph,) = pydot.graph_from_dot_file("decision_tree.dot")
    graph.write_png("graphs/decision_tree.png")"""

    # Print confusion matrix
    print("Confusion Matrix")
    disp = plot_confusion_matrix(
        clf,
        X_test,
        y_test,
        display_labels=["Benign", "Malware"],
        cmap=plt.cm.get_cmap("hot"),
    )
    disp.ax_.set_title("Confusion Matrix")
    print(disp.confusion_matrix)
    plt.savefig("graphs/DT_confusion_matrix.png")

    print("Learning curve")
    plot_learning_curve(clf, X_train, y_train)

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

    done.write('\nA performance report has been conducted on decisiontree.py! The time taken was {0:0.3f}s.'.format(end_time))
    done.write('\n\nSelected Parameters:\n')

    param_data=[('max_depth', 10),('min_samples_leaf', 2),('min_samples_split', 2)]

    try:
        for i in param_data:
            done.write('\t{0}: {1}\n'.format(i[0], i[1]))
    except:
        done.write('No parameter data found.')

    done.write('\nThe performance report was as follows:\n')

    try:
        done.write(report)
    except:
        done.write('No classification report found.')

    done.write('Finally, the Decision Tree score on the test dataset: {0}'.format(test_score))

    done.close()
