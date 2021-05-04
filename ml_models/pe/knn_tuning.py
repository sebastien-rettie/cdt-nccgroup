import pickle
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

# Utilities for preprocessing
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


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

X_train = X_train.astype("int64")
X_test = X_test.astype("int64")


############################################


def hyperopt_train_test(params):
    clf = KNeighborsClassifier(**params)
    return cross_val_score(clf, X_train, y_train).mean()


space4knn = {"n_neighbors": hp.quniform("n_neighbors", 2, 10, 1)}


def f(params):
    acc = hyperopt_train_test(params)
    return {"loss": -acc, "status": STATUS_OK}


trials = Trials()
best = fmin(fn=f, space=space4knn, algo=tpe.suggest, max_evals=5, trials=trials)
print("best:")
print(best)
