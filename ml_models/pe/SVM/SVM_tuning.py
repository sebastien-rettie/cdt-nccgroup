import pickle
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier

import time
from datetime import date
import sys

from preprocessing import generate_types

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

#Set sample fraction here
sample_frac = 0.1

#Sample fraction of the dataset to break down training
df_test = df_test.sample(frac=sample_frac)
df_train = df_train.sample(frac=sample_frac)

y_train = df_train["IsMalware"]
X_train = df_train.drop("IsMalware", axis=1)

y_test = df_test["IsMalware"]
X_test = df_test.drop("IsMalware", axis=1)

print('Data before "pickling":')
print('\n', X_train)
print('\n', X_test)

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

print('Train shape\n\n',np.shape(X_train))
print()
print('Test shape\n\n',np.shape(X_test))

holding = True
while holding == True:
    hold = str(input('\nAre you satisfied with these datasets?(y/n): '))
    if hold == 'n':
        sys.exit()
    elif hold == 'y':
        holding = False
    else:
        continue
##############


def hyperopt_train_test(params):
    clf = SVC(**params)
    return cross_val_score(clf, X_train, y_train).mean()

ranges = {
    "C": [0.1,20],
    "gamma": [0.1,20]
 }


space4svm = {
    "C": hp.uniform("C", ranges['C'][0], ranges['C'][1]),
    #"kernel": hp.choice("kernel", ["linear", "sigmoid", "poly", "rbf"]),
    "kernel": "linear",
    "gamma": hp.uniform("gamma", ranges['gamma'][0], ranges['gamma'][1]),
}


def f(params):
    acc = hyperopt_train_test(params)
    return {"loss": -acc, "status": STATUS_OK}


start_time = time.time()

trials = Trials()
best = fmin(f, space4svm, algo=tpe.suggest, max_evals=10, trials=trials, timeout=600)
print("best:")
print(best)

best_loss = trials.best_trial['result']['loss']

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

done.write('\nTraining/testing was conducted on {0}% of the dataset.'.format(100*sample_frac))

done.write('\n\nBest hyperparameters found:')
for key in best:
    done.write('\n\t{0}: {1}'.format(key,best[key]))

done.write('\n\nThese correspond to an accuracy (negative loss) of:')
done.write(str(-1*best_loss))

done.write('\n\nParameter ranges:')

for key in ranges:
    done.write('\n\t{0}: {1}'.format(key,ranges[key]))

done.write('\n\nExecution time: {0:0.3f}s'.format(end_time))
done.close()
