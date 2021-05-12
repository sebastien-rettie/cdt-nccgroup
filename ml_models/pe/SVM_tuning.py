import pickle
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

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


def hyperopt_train_test(params):
    clf = SVC(**params)
    return cross_val_score(clf, X_train, y_train).mean()


space4svm = {
    "C": hp.uniform("C", 0, 20),
    "kernel": hp.choice("kernel", ["linear", "sigmoid", "poly", "rbf"]),
    "gamma": hp.uniform("gamma", 0, 20),
}


def f(params):
    acc = hyperopt_train_test(params)
    return {"loss": -acc, "status": STATUS_OK}


trials = Trials()
best = fmin(f, space4svm, algo=tpe.suggest, max_evals=100, trials=trials)
print("best:")
print(best)
