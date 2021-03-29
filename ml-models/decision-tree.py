import numpy as np
import pandas as pd

from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector

from sklearn.tree import DecisionTreeClassifier

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


# print(df1.shape)
# print(df1.dtypes)

# print(df2.shape)
# print(df2.dtypes)

all_data = pd.concat([df1, df2, df3, df4], axis=0)
# all_data.select_dtypes(include="object").fillna("0", inplace=True)
# all_data.select_dtypes(include="number").fillna(0, inplace=True)

df = all_data.apply(lambda x: x.fillna(0) if x.dtype.kind in "biufc" else x.fillna("0"))
# all_data.fillna(0, inplace=True)
print(all_data.shape)
# print(all_data["IsMalware"])


# strings = df1.select_dtypes(include="object")
# print(strings.columns)
# result = pd.concat([df1, df2], axis=1)
# print(result.shape)


# orig = pd.read_csv(input_file)
# df = orig.copy()


df = df.drop(columns=["TimeDateStamp"])
# Makes 1h encoding too big/costly

y = df["IsMalware"]
x = df.drop("IsMalware", axis=1)

# print(y)
print(y.shape)
print(x.shape)

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

preprocess = Pipeline(steps=[("preprocess", column_trans)])

# Preprocessing
x = preprocess.fit_transform(x)
# It is bad practice to have split after transform- fix

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=4)

clf = DecisionTreeClassifier().fit(X_train, y_train)

print(
    "Accuracy of Decision Tree classifier on training set: {:.2f}".format(
        clf.score(X_train, y_train)
    )
)
print(
    "Accuracy of Decision Tree classifier on test set: {:.2f}".format(
        clf.score(X_test, y_test)
    )
)

with open("decision_tree_model_results.txt", "w") as file:
    results = [
        "Accuracy of Decision Tree classifier on training set:",
        str(clf.score(X_train, y_train)),
        "\n",
        "Accuracy of Decision Tree classifier on test set:",
        str(clf.score(X_test, y_test)),
    ]
    file.writelines(results)
