import numpy as np
import pandas as pd
import csv

from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import model_selection

input_file = "benign/benign-dll.csv"
input_file_2 = "benign/benign-exe.csv"
input_file_3 = "malware/00355_malware.csv"
input_file_4 = "malware/00372_malware.csv"


def generate_types(datafile):
    col_names = pd.read_csv(input_file, nrows=0).columns
    dtypes = {col: "float64" for col in col_names}
    string_columns = [
        "Name0",
        "Name1",
        "Name10",
        "Name11",
        "Name12",
        "Name13",
        "Name14",
        "Name15",
        "Name16",
        "Name17",
        "Name18",
        "Name19",
        "Name2",
        "Name20",
        "Name21",
        "Name22",
        "Name23",
        "Name24",
        "Name3",
        "Name4",
        "Name5",
        "Name6",
        "Name7",
        "Name8",
        "Name9",
        "TimeDateStamp",
        "e_res",
        "e_res2",
    ]
    for column in string_columns:
        dtypes.update({column: "object"})
    # print(dtypes)
    return dtypes


df1 = pd.read_csv(input_file, dtype=generate_types(input_file))
df2 = pd.read_csv(input_file_2, dtype=generate_types(input_file_2))
df3 = pd.read_csv(input_file_3, dtype=generate_types(input_file_3))
df4 = pd.read_csv(input_file_4, dtype=generate_types(input_file_4))

all_data = pd.concat([df1, df2, df3, df4], axis=0)
df = all_data.apply(lambda x: x.fillna(0) if x.dtype.kind in "biufc" else x.fillna("0"))
df = df.drop(columns=["TimeDateStamp"])

y = df["IsMalware"]
x = df.drop("IsMalware", axis=1)

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=100),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
]


# Feature scaling
scale_transform = MaxAbsScaler()

# One hot encoding, transforms categorical to ML friendly variables
onehot_transform = OneHotEncoder(handle_unknown="ignore")

column_trans = ColumnTransformer(
    transformers=[
        ("Numerical", scale_transform, selector(dtype_include="number")),
        ("Categorical", onehot_transform, selector(dtype_include="object")),
    ],
    remainder="passthrough",
)

preprocess = Pipeline(steps=[("preprocess", column_trans),])

# Preprocessing
x = preprocess.fit_transform(x)
# It is bad practice to have split after transform- fix

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=4
)
#

results = []
scoring = "accuracy"

for name, model in zip(names, classifiers):

    clf = model.fit(X_train, y_train)

    print(
        "Accuracy of",
        name,
        "classifier on training set: {:.2f}".format(clf.score(X_train, y_train)),
    )
    print(
        "Accuracy of",
        name,
        " classifier on test set: {:.2f}".format(clf.score(X_test, y_test)),
    )
    results.append(clf.score(X_test, y_test))

    names.append(name)


accuracy_results = dict(zip(names, results))
print(accuracy_results)

with open("multi_model_results.csv", "w") as file:
    w = csv.writer(file)
    w.writerows(accuracy_results.items())
