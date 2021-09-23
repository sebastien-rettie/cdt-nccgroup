##############################################################
#-----------------------PLEASE READ--------------------------#

#This model has been optimised for RECALL.
#Seeing previous findings, recall favours high fractions of
#malware in the training set.
#As such it is recommended to keep your train set at least
#imbalanced in the favour of malware.
#Seeing other previous findings, downsampling the training
#set to ~20% its size is sufficient for high performance.
#Finally, this model is reduced down to ~30 input features
#while still retaining high performance.
#You can run the external scripts labelled 'recursive_drop.py'
#followed by 'importance_parser.py' to output the top features 
#into a txt file for the desired statistic.

#If you wish to optimise the model for precision, or accuracy,
#enable the switch 'user_inputs' after looking at the other data.
#That way it will prompt you for training data fractions.

##############################################################

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import pickle

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydot

import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    plot_confusion_matrix,
    roc_auc_score,
    plot_roc_curve,
    roc_curve
)

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
import sys
from datetime import date
import seaborn as sns

###########################################

def balance_and_downsample(data, dataset_fraction, malware_balance):
    """
    Samples the dataset such that the malware/benignware balance is built according to the given balance, but also downsamples the data to a fraction of its original size according to the given fraction

    - data: Dataset to balance
    - dataset_fraction: Fraction of the dataset size to return, e.g. 0.1 returns a dataset 10% the size of the original
    - malware_balance: Fraction of Malware desired in the dataset balance

    Returns 'data', balanced dataset
    """
    
    full_train_size = data.shape[0]
    desired_amount = dataset_fraction*full_train_size
    benign_balance = 1-malware_balance
    mal = data[data["IsMalware"] == '1.0']
    ben = data[data["IsMalware"] == '0.0']

    df_benign = ben.sample(n=round(benign_balance*desired_amount))
    df_malware = mal.sample(n=round(malware_balance*desired_amount))

    data = pd.concat([df_benign, df_malware])

    return data


def sort_importances(data, importances):
    """
    Runs through top importances as given in external txt file, slices the data such that only those are input

    - data: Dataset to drop least important features from
    - importances: List of top importances to keep

    Returns 'data', dataset with only top features included
    """

    for col in list(data.columns):
        if col not in importances:
            data = data.drop(col, axis=1)

    return data

def encode_types(dataframe):
    """
    Turns all column dtypes to float in a dataframe

    - dataframe: Dataset to encode to float

    Returns 'dataframe', encoded data
    """

    for col in list(dataframe.columns):
        dataframe[col] = pd.to_numeric(dataframe[col])

    return dataframe

###########################################

train_fraction = 0.2
malware_fraction = 0.85

#FOR USER DEFINED DATASET SHAPING
user_inputs = False

if user_inputs:
    train_fraction = input('\n\nPlease select a fraction of the dataset you would like to use for training. For example, putting in 0.2 will use 20% of the available statistics: ')

    holding = True
    while holding:
        try:
            train_fraction = float(train_fraction)
            holding = False
        except:
            train_fraction = input('You didn\'t put in a number. Please try again: ')

    while train_fraction > 1:
        train_fraction = float(input('Please choose a number lower than 1 (it must be a fraction): '))
        if train_fraction < 1:
            break


    malware_fraction = input('\n\nPlease insert a balance fraction for the dataset. This will be the fraction of the training set that comprises malware. For example, 0.25 will generate a set that is 25% malware, 75% benignware: ')

    holding = True
    while holding:
        try:
            malware_fraction = float(malware_fraction)
            holding = False
        except:
            malware_fraction = input('You didn\'t put in a number. Please try again: ')

    while malware_fraction > 1:
        malware_fraction = float(input('Please choose a number lower than 1 (it must be a fraction): '))
        if malware_fraction < 1:
            break


clf = xgb.XGBClassifier(
        n_estimators=100,
        colsample_bytree=0.8317,
        learning_rate=0.3,
        max_depth=11,
        min_child_weight=3.0,
        subsample=0.9099,
        gamma=0.292,
        reg_lambda=0.447,
        verbosity=0      #NO WARNINGS!!!!
)

full_train_size = 50456 #Size of full train set without downsampling/balancing

train_file = "train.csv"
df_train = pd.read_csv(
    train_file,
    dtype=generate_types(train_file),
    engine="python",
)

df_train.set_index(["SampleName"], inplace=True)

test_file = "test.csv"
df_test = pd.read_csv(
    test_file,
    dtype=generate_types(test_file),
    engine="python"
)

df_test.set_index(["SampleName"], inplace=True)

#Copy the dataframes for the 'old' XGBoost which is not trimmed
df_train_old = df_train
df_test_old = df_test



#Load in important features and select those from each dataset
unpack = np.loadtxt('XGBoost_features.txt', delimiter=",", unpack=True, dtype="str")
important_features = [unpack[0][1:-1]]

for i in unpack[1:]:
    important_features.append(i[2:-1]) #Remove spaces and speech marks from each entry

important_features.append('IsMalware') #Don't let it remove the malware classifier column!


print("\nThe top {0} features you have chosen to include are:\n".format(len(important_features)-1))
print(important_features)

#Sort through importances to retain only top features
df_train = sort_importances(df_train, important_features)
df_test = sort_importances(df_test, important_features)



#Downsample and balance the training dataset
df_train = balance_and_downsample(df_train, train_fraction, malware_fraction)

print("\nTraining set counts:")
print(df_train["IsMalware"].value_counts())

print('\nDataset shape (to check - IsMalware column not yet dropped):')
print(df_train.shape)

print('This is {0:0.2f}% of the total data.\n'.format(100*np.shape(df_train)[0]/full_train_size))


#Print test dataset malware balance
print("Testing set counts:")
print(df_test["IsMalware"].value_counts())

y_train = df_train["IsMalware"]
X_train = df_train.drop("IsMalware", axis=1)

y_test = df_test["IsMalware"]
X_test = df_test.drop("IsMalware", axis=1)

#Encode the train/test inputs to float vals
X_train = encode_types(X_train)
X_test = encode_types(X_test)

#Sort and pickle the old dataframes
y_train_old = df_train_old["IsMalware"]
X_train_old = df_train_old.drop("IsMalware", axis=1)

y_test_old = df_test_old["IsMalware"]
X_test_old = df_test_old.drop("IsMalware", axis=1)

# OR import prefit encoder
with open("encoder.pickle", "rb") as f:
    column_trans = pickle.load(f, encoding="bytes")
    encoder = column_trans

X_train_old = encoder.transform(X_train_old)
X_test_old = encoder.transform(X_test_old)

with open("selector.pickle", "rb") as f:
    selector = pickle.load(f, encoding="bytes")

X_train_old = selector.transform(X_train_old)
X_test_old = selector.transform(X_test_old)

with open("scale.pickle", "rb") as f:
    scale_transform = pickle.load(f, encoding="bytes")

X_train_old = scale_transform.transform(X_train_old)
X_test_old = scale_transform.transform(X_test_old)




#Train the models!
models = ["Original XGBoost", "Simplified XGBoost"]

datasets = {
        "Original XGBoost": [X_train_old, X_test_old, y_train_old, y_test_old],
        "Simplified XGBoost": [X_train, X_test, y_train, y_test]
}

displays = {}

fig, ax = plt.subplots()

for m in models:
    start_time = time.time()

    clf.fit(datasets[m][0], datasets[m][2])
    y_predicted = clf.predict(datasets[m][1])

    print("score", clf.score(datasets[m][1], datasets[m][3]))

    y_test_array = np.asarray(datasets[m][3])
    misclassified = y_test_array != y_predicted

    print("Misclassified samples:", datasets[m][3][misclassified].index)

    y_score = clf.predict_proba(datasets[m][1])[:, 1]

    print("Area under ROC curve score:")
    print(roc_auc_score(datasets[m][3], y_score))



    print("Accuracy of classifier on training set: {:.3f}".format(clf.score(datasets[m][0], datasets[m][2])))
    print("Accuracy of classifier on test set: {:.3f}".format(clf.score(datasets[m][1], datasets[m][3])))

    test_score = clf.score(datasets[m][1], datasets[m][3])

    report = classification_report(datasets[m][3], y_predicted, target_names=["Benign", "Malware"])

    end_time = time.time() - start_time

    print("Classification report:\n", report)


    print("Printing ROC plot\n")
    displays[m] = plot_roc_curve(clf, datasets[m][1], datasets[m][3], ax=ax, name=m)
    tn, fp, fn, tp = confusion_matrix(datasets[m][3], y_predicted, normalize="true").ravel()

    """lw = 2
    plt.scatter(
        fp,
        tp,
        color="darkorange",
        lw=lw,
        label=m,
)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--", label="random guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positives")
    plt.ylabel("True Positives")
    plt.title("Final XGBoost ROC Curve")
    plt.legend(loc="lower right")

    plt.savefig("graphs/roc_plot.png")
    plt.show()"""

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Original vs. Simplified XGBoost\nROC curve comparison")
    plt.legend(loc="best")
    plt.show()

    _ = ax.set_title("Original vs. Simplified XGBoost\nROC curve comparison")
    ax.set_xlim(0, 0.2)
    ax.set_ylim(0.8, 1)
    
plt.savefig("graphs/roc_curve_comparison.png")

# Print confusion matrix
print("Confusion Matrix")
disp = plot_confusion_matrix(
    clf,
    X_test,
    y_test,
    display_labels=["Benign", "Malware"],
    cmap=plt.cm.get_cmap("hot"),
)
disp.ax_.set_title("Final XGBoost Confusion Matrix")
print(disp.confusion_matrix)

plt.figure()
plt.title('Final Simplified XGBoost Confusion Matrix, normalized')
plt.xlabel('Predicted label')
plt.ylabel('True label')
group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
group_counts = ["{0:0.1f}%".format(value) for value in 100*disp.confusion_matrix.flatten()/np.sum(disp.confusion_matrix)]
group_numbers = ["{0:0.0f}".format(value) for value in disp.confusion_matrix.flatten()] 
labels = [f"{v1}\n{v2}\nTotal files: {v3}" for v1, v2, v3 in zip(group_names,group_counts,group_numbers)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(disp.confusion_matrix, annot=labels, fmt='', cmap='hot')

plt.savefig("graphs/confusion_matrix.png")

print("Learning curve")
plot_learning_curve(clf, X_train, y_train)

"""
try:
    reader = open('performance_report.txt', 'r')
    prev_text = reader.read()
    reader.close()
except FileNotFoundError:
    open('performance_report.txt','x')

done = open('performance_report.txt', 'w')

try:
    done.write(prev_text)
except:
    pass

done.write('\n\n===============================================\n\n')
done.write('{0}\n'.format(date.today()))

done.write('A performance report has been conducted using the XGBoost model. The time taken to train and test it was {0:0.3f}s.'.format(end_time))

done.write('\n\nSelected Parameters:\n')

param_data=[
        ('n_estimators',100),
        ('colsample_bytree',0.8317),
        ('learning_rate',0.3),
        ('max_depth',11),
        ('min_child_weight',3.0),
        ('subsample',0.9099),
        ('gamma',0.292),
        ('reg_lambda',0.447)
]

try:
    for i in param_data:
        done.write('\t{0}: {1}\n'.format(i[0], i[1]))
except:
    done.write('No parameter data found.')

done.write('\nUSER INPUT PARAMS:\n')
done.write('\tMalware fraction in data balance: {0}\n'.format(malware_fraction))
done.write('\tFraction of training data used (downample fraction): {0}\n'.format(train_fraction))

done.write('\nThe performance report was as follows:\n')

try:
    done.write(report)
except:
    done.write('No classification report found.')

done.write('\n\nFinally, the XGBoost score on the test dataset: {0}'.format(test_score))

done.close()"""
