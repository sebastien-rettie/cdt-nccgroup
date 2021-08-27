# Utilities for preprocessing
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline

from sklearn.feature_selection._base import SelectorMixin
from sklearn.feature_extraction.text import _VectorizerMixin

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)

from sklearn.metrics import confusion_matrix

# Function to read certain columns as objects not floats.
# Reduces overhead of pandas trying to decide for itself.
def generate_types(datafile):
    col_names = pd.read_csv(datafile, nrows=0).columns
    dtypes = {col: "string" for col in col_names}
    string_columns = [
        "SampleName",
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
        "Name25",
        "Name26",
        "Name27",
        "Name28",
        "Name29",
        "Name30",
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
    return dtypes


def concat_files(inputfile_list, output_file):
    dataframe_list = []
    # grabs all csvs in list and concats
    for inputfile in inputfile_list:
        dataframe = pd.read_csv(
            inputfile, engine="python", dtype=generate_types(inputfile)
        )
        dataframe_list.append(dataframe)
    all_data = pd.concat(dataframe_list, axis=0)
    # writes to all data csv for reuse
    all_data.to_csv(output_file, index=False)
    return all_data


def preprocess_dataframe(input_dataframe):
    """
    input_dataframe = input_dataframe.apply(
        lambda x: x.fillna(0) if x.dtype.kind in "biufc" else x.fillna("null")
    )
    """
    input_dataframe = input_dataframe.dropna(axis=1)
    
    try:
        input_dataframe = input_dataframe.drop(columns=["e_res"])
    except KeyError:
        print('column e_res was not found in {0}...'.format(input_dataframe))
        pass

    try:
        input_dataframe = input_dataframe.drop(columns=["e_res2"])
    except KeyError:
        print('column e_res2 was not found in {0}...'.format(input_dataframe))
        pass

    return input_dataframe


def encode_scale():
    # Feature scaling
    # scale_transform = MaxAbsScaler()

    # One hot encoding, transforms categorical to ML friendly variables
    onehot_transform = OneHotEncoder(handle_unknown="ignore")

    column_trans = ColumnTransformer(
        transformers=[
            ("Categorical", onehot_transform, selector(dtype_include="object")),
            #       ("Numerical", scale_transform, selector(dtype_include="number")),
        ],
        remainder="passthrough",
    )
    return column_trans


# Some black magic to retrieve feature names from the encoder
def get_feature_out(estimator, feature_in):
    if hasattr(estimator, "get_feature_names"):
        if isinstance(estimator, _VectorizerMixin):
            # handling all vectorizers
            return [f"vec_{f}" for f in estimator.get_feature_names()]
        else:
            return estimator.get_feature_names(feature_in)
    elif isinstance(estimator, SelectorMixin):
        return np.array(feature_in)[estimator.get_support()]
    else:
        return feature_in


def get_ct_feature_names(ct):
    # handles all estimators, pipelines inside ColumnTransfomer
    # doesn't work when remainder =='passthrough'
    # which requires the input column names.
    output_features = []

    for name, estimator, features in ct.transformers_:
        if name != "remainder":
            if isinstance(estimator, Pipeline):
                current_features = features
                for step in estimator:
                    current_features = get_feature_out(step, current_features)
                features_out = current_features
            else:
                features_out = get_feature_out(estimator, features)
            output_features.extend(features_out)
        elif estimator == "passthrough":
            output_features.extend(ct._feature_names_in[features])

    return output_features


def grid_search_wrapper(
    clf,
    param_grid,
    scorers,
    X_train,
    X_test,
    y_train,
    y_test,
    refit_score="accuracy_score",
):
    """
    fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics
    """
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    grid_search = GridSearchCV(
        clf,
        param_grid,
        scoring=scorers,
        refit=refit_score,
        cv=skf,
        return_train_score=True,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    # make the predictions
    y_pred = grid_search.predict(X_test)

    print("Best params for {}".format(refit_score))
    print(grid_search.best_params_)

    # confusion matrix on the test data.
    print(
        pd.DataFrame(
            confusion_matrix(y_test, y_pred),
            columns=["pred_neg", "pred_pos"],
            index=["neg", "pos"],
        )
    )
    return grid_search

