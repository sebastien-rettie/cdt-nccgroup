#  For large datasets consider use LinearSVC or SGDClassifier, possibly after a Nystroem transformer.
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from datetime import date
import time

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

#Previously found model
#clf = SGDClassifier(max_iter=2000, tol=1e-3)

#Tuned using SVM_tuning.py
params_tuned = {
        'C':3.4553579122449825, #C parameter is used when optimising SVM - used to calculate alpha below
        'gamma':5.623470180576352,
        'n_components':250,
        'loss':'modified_huber'
}

print('You are running an SGD model (with Nystroem transform to approximate SVM). You have parameters:\n')
for key in params_tuned:
    print('\t{0}: {1}'.format(key, params_tuned[key]))

#https://stats.stackexchange.com/questions/216095/how-does-alpha-relate-to-c-in-scikit-learns-sgdclassifier
alpha = 1/(params_tuned['C'] * params_tuned['n_components'])
alpha = round(alpha,5)
params_tuned['alpha'] = alpha

print('\nAlpha parameter calculated as:\n\n\t',alpha)

holding = True
while holding:
    ok = str(input('\nAre you satisfied with this alpha value? (y/n): '))
    
    if ok == 'y':
        holding = False
    elif ok == 'n':
        print('\nTake a look at your parameters and decide how you would like to change alpha.')
        sys.exit()
    else:
        continue

#previous CLF
#clf = SGDClassifier(max_iter=2000, tol=1e-3)

clf = SGDClassifier(loss=params_tuned['loss'], alpha=alpha, max_iter=2000, tol=1e-3)

#FOR EWAN (LEARNING): This approximates SVM kernel to use in linear learning processes (like SGD) on large datasets
#This is done by subsampling with n_components
feature_map_nystroem = Nystroem(
        gamma = params_tuned['gamma'], 
        #gamma=0.2,    #original gamma before change
        random_state = 1, 
        n_components = params_tuned['n_components']
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

print("Finished feature selection.")
##############

# Nystroem transform
feature_map_nystroem.fit(X_train)

print('Nystroem transfom applied to the train dataset.')
###############

#Switches - you can run both but 'tuning' does exactly what it says, 'report' is to run the model AFTER tuning (typically)
tuning = False
report = False
fraction_statistics = True

if report:
    start_time = time.time()

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

    #Current issues with ROC plotting?
    """print("Roc plot\n")
    p, r, thresholds = precision_recall_curve(y_test, y_score)
    # Adjust this down to remove false negatives
    precision_recall_threshold(p, r, thresholds, y_score, y_test, 0.30)
    plot_precision_recall_vs_threshold(p, r, thresholds)
    fpr, tpr, auc_thresholds = roc_curve(y_test, y_score)
    print("Area under ROC curve:", auc(fpr, tpr))  # AUC of ROC
    plot_roc_curve(fpr, tpr, "accuracy_optimized")
    """
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

    done.write('\nA performance report has been conducted on SVM.py!')

    done.write('\n\nSelected parameters:')
    for key in params_tuned:
        done.write('\n\t{0}: {1}'.format(key,params_tuned[key]))
    done.write('\n\tresulting alpha: {0}'.format(alpha))

    done.write('\n\nThe performance report was as follows:\n')
    done.write(report)

    done.write('\n\nFinally, the SGD (SVM approximated using Nystroem) score on the test set: {0}'.format(clf.score(X_test, y_test)))

    done.write('\n\nExecution time: {0:0.3f}s'.format(end_time))
    done.close()

elif tuning:

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

elif fraction_statistics:
    print('Conducting training statistics analysis...')
    train_file = "train.csv"
    df_train = pd.read_csv(
        train_file,
        dtype=generate_types(train_file),
        engine="python",
    )
    df_train.set_index(["SampleName"], inplace=True)

    fraction_range = np.arange(0.001,0.011,0.001)
    fraction_range = np.append(fraction_range, np.arange(0.02,1.02,0.02))

    print(fraction_range)

    misclassified_vals = [] #Append no. misclassified samples for each fraction
    scores = [] #Test scores for each fraction

    for fraction in fraction_range:
        print('FRACTION OF TRAINING DATASET: {0}\n'.format(fraction))

        df_train_frac = df_train.sample(frac=fraction)

        y_train = df_train_frac["IsMalware"]
        X_train = df_train_frac.drop("IsMalware", axis=1)

        print(np.shape(X_train), '\n')
        
        with open("encoder.pickle", "rb") as f:
            column_trans = pickle.load(f, encoding="bytes")
            encoder = column_trans

        X_train = encoder.transform(X_train)

        with open("selector.pickle", "rb") as f:
            selector = pickle.load(f, encoding="bytes")

        X_train = selector.transform(X_train)

        with open("scale.pickle", "rb") as f:
            scale_transform = pickle.load(f, encoding="bytes")

        X_train = scale_transform.transform(X_train)

        clf = SGDClassifier(
            loss=params_tuned['loss'],
            alpha=params_tuned['alpha'],
            max_iter=2000,
            tol=1e-3
        )

        feature_map_nystroem = Nystroem(
            gamma = params_tuned['gamma'],
            random_state = 1,
            n_components = params_tuned['n_components']
        )

        feature_map_nystroem.fit(X_train)

        clf.fit(X_train, y_train)

        print("\n\nClassifier score on the test data:", clf.score(X_test, y_test))
        y_predicted = clf.predict(X_test)

        y_test_array = np.asarray(y_test)
        misclassified = y_test_array != y_predicted

        print('Misclassified: {0:0.2f}% ({1})'.format(100*len(y_test[misclassified])/len(y_test),len(y_test[misclassified])))

        misclassified_vals.append(100*len(y_test[misclassified])/len(y_test))
        scores.append(clf.score(X_test, y_test)*100)

    plt.figure()
    plt.plot(fraction_range,misclassified_vals,'r-',label='% misclassified samples')
    plt.plot(fraction_range,scores,'g-',label='% Test dataset accuracy')
    plt.xlabel('Fraction of statistics used for training')
    plt.ylabel('Percentage (%)')
    plt.title('Nystroem SGD (SVM)  accuracy using different fractions of data for training')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('graphs/trainingfractions_misclassified_included.png')

    plt.figure(2)
    plt.plot(fraction_range,scores,'-',label='% Test dataset accuracy')
    plt.xlabel('Fraction of statistics used for training')
    plt.ylabel('Percentage (%)')
    plt.title('Nystroem SGD (SVM)  accuracy using different fractions of data for training')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('graphs/trainingfractions_accuracy.png')
