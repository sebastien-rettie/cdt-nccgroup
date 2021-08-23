import pickle
from sklearn.metrics import accuracy_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import xgboost as xgb

# Utilities for preprocessing
import pandas as pd
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
############################################


def objective(space):
    # Instantiate the classifier
    clf = xgb.XGBClassifier(
        n_estimators=1000,
        colsample_bytree=space["colsample_bytree"],
        learning_rate=0.3,
        max_depth=int(space["max_depth"]),
        min_child_weight=space["min_child_weight"],
        subsample=space["subsample"],
        gamma=space["gamma"],
        reg_lambda=space["reg_lambda"],
    )

    eval_set = [(X_train, y_train), (X_test, y_test)]

    # Fit the classsifier
    clf.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        eval_metric="rmse",
        early_stopping_rounds=10,
        verbose=True,
    )
    # Predict on Cross Validation data
    pred = clf.predict(X_test)

    # Calculate our Metric - accuracy
    accuracy = accuracy_score(y_test, pred > 0.5)
    # return needs to be in this below format. We use negative of accuracy since we want to maximize it.
    return {"loss": -accuracy, "status": STATUS_OK}


space = {
    "max_depth": hp.quniform("x_max_depth", 4, 16, 1),
    "min_child_weight": hp.quniform("x_min_child", 1, 10, 1),
    "subsample": hp.uniform("x_subsample", 0.7, 1),
    "gamma": hp.uniform("x_gamma", 0.1, 0.5),
    "colsample_bytree": hp.uniform("x_colsample_bytree", 0.7, 1),
    "reg_lambda": hp.uniform("x_reg_lambda", 0, 1),
}

trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials)
print(best)
