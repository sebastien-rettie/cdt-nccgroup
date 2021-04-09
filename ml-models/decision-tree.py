import numpy as np
import pandas as pd
import graphviz
import pydot
import matplotlib.pyplot as plt

from sklearn.model_selection import ShuffleSplit

from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import validation_curve
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.metrics import classification_report
from sklearn.tree import export_text
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.feature_selection._base import SelectorMixin
from sklearn.feature_extraction.text import _VectorizerMixin

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import learning_curve

# from joblib import parallel_backend
# todo: implement parallelism


input_file = "benign/benign-dll.csv"
input_file_2 = "benign/benign-exe.csv"
input_file_3 = "malware/00355_malware.csv"
input_file_4 = "malware/00372_malware.csv"
input_file_5 = "malware/00373_malware.csv"

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


"""
def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, "o-")
    axes[2].fill_between(
        fit_times_mean,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

"""
# Function to read certain columns as objects not floats.
# Reduces overhead of pandas trying to decide for itself.
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
    return dtypes


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")


np.set_printoptions(precision=2)

df1 = pd.read_csv(input_file, dtype=generate_types(input_file))
df2 = pd.read_csv(input_file_2, dtype=generate_types(input_file_2))
df3 = pd.read_csv(input_file_3, dtype=generate_types(input_file_3))
df4 = pd.read_csv(input_file_4, dtype=generate_types(input_file_4))
df5 = pd.read_csv(input_file_5, dtype=generate_types(input_file_4))

all_data = pd.concat([df1, df2, df3, df4, df5], axis=0)

# Handle empty cells
df = all_data.apply(lambda x: x.fillna(0) if x.dtype.kind in "biufc" else x.fillna("0"))

# Dropping date value as it makes 1h encoding too costly
df = df.drop(columns=["TimeDateStamp"])

y = df["IsMalware"]
x = df.drop("IsMalware", axis=1)

# Feature scaling
scale_transform = MaxAbsScaler()

# One hot encoding, transforms categorical data to ML friendly variables
onehot_transform = OneHotEncoder(handle_unknown="ignore")

# todo: test alternate methods of encoding e.g. hash vectorising. 1h encoding not ideal for decision trees

column_trans = ColumnTransformer(
    transformers=[
        ("Numerical", scale_transform, selector(dtype_include="number")),
        ("Categorical", onehot_transform, selector(dtype_include="object")),
    ],
    # remainder="passthrough",
    # remainder option improves stability but must be disabled to extract feature names after encoding
)

pipeline = Pipeline(steps=[("preprocess", column_trans)])

# Preprocessing and vectorise
X = pipeline.fit_transform(x)

# todo: fix as is bad practice to have split after preprocessing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=4)
# default train/test split is 0.75:0.25

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# Switches to enable tuning/plotting
previously_tuned = True
plot_validate_params = False
performance_report = False
plot_learning_curve = True

# Hyperparameter tuning, use randomised over grid search for speed
if previously_tuned == False:
    tuned_parameters = [
        {
            "splitter": ["best", "random"],
            "max_depth": [2, 5, 10, 20, 40],
            "min_samples_split": [0.1, 0.2, 0.3, 4, 5, 10, 15],
            "min_samples_leaf": [0.2, 2, 10],
        },
    ]

    scores = ["precision", "recall"]
    n_iter_search = 40

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)

        clf = RandomizedSearchCV(
            DecisionTreeClassifier(random_state=0),
            tuned_parameters,
            n_iter=n_iter_search,
            scoring="%s_macro" % score,
        )
        # note GridSearch automatically uses StratKFold x-validation
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
elif plot_validate_params == True:
    # Skip tuning if already have optimal hyperparameters, plot validation curves to verify
    # Optimal params = {'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 2, 'splitter': 'best'}

    # Max depth validation
    max_depth_range = list(range(1, 40))

    train_scores, test_scores = validation_curve(
        DecisionTreeClassifier(
            random_state=0, splitter="random", min_samples_split=2, min_samples_leaf=2
        ),
        X_train,
        y_train,
        param_name="max_depth",
        param_range=max_depth_range,
        scoring="accuracy",
        cv=skf,
        n_jobs=1,
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots()
    plt.title("Decision Tree Max Depth Validation Curve")
    plt.xlabel("Tree Depth")
    plt.ylabel("Score")

    lw = 2
    plt.semilogx(
        max_depth_range,
        train_scores_mean,
        label="Training score",
        color="darkorange",
        lw=lw,
    )
    plt.fill_between(
        max_depth_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2,
        color="darkorange",
        lw=lw,
    )
    plt.semilogx(
        max_depth_range,
        test_scores_mean,
        label="Cross-validation score",
        color="navy",
        lw=lw,
    )
    plt.fill_between(
        max_depth_range,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.2,
        color="navy",
        lw=lw,
    )
    plt.legend(loc="best")

    plt.savefig("tree_depth_validation.png")
    """
    """
    # Min samples split validation
    samples_split_range = list(range(2, 20))
    for x in range(1, 10):
        samples_split_range.append(x * 0.1)
    samples_split_range.sort()

    train_scores, test_scores = validation_curve(
        DecisionTreeClassifier(
            random_state=0, splitter="best", min_samples_leaf=2, max_depth=5
        ),
        X_train,
        y_train,
        param_name="min_samples_split",
        param_range=samples_split_range,
        scoring="accuracy",
        cv=skf,
        n_jobs=1,
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots()
    plt.title("Decision Tree Min Samples Split Validation Curve")
    plt.xlabel("Minimum Samples Split")
    plt.ylabel("Score")

    lw = 2
    plt.semilogx(
        samples_split_range,
        train_scores_mean,
        label="Training score",
        color="darkorange",
        lw=lw,
    )
    plt.fill_between(
        samples_split_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2,
        color="darkorange",
        lw=lw,
    )
    plt.semilogx(
        samples_split_range,
        test_scores_mean,
        label="Cross-validation score",
        color="navy",
        lw=lw,
    )
    plt.fill_between(
        samples_split_range,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.2,
        color="navy",
        lw=lw,
    )
    plt.legend(loc="best")

    plt.savefig("samples_split_tree_validation.png")

    # Min samples leaf validation
    samples_leaf_range = list(range(2, 40))
    for x in range(1, 10):
        samples_leaf_range.append(x * 0.1)
    samples_leaf_range.sort()

    train_scores, test_scores = validation_curve(
        DecisionTreeClassifier(
            random_state=0, splitter="best", min_samples_split=2, max_depth=5
        ),
        X_train,
        y_train,
        param_name="min_samples_split",
        param_range=samples_leaf_range,
        scoring="accuracy",
        cv=skf,
        n_jobs=1,
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots()
    plt.title("Decision Tree Min Samples Leaf Validation Curve")
    plt.xlabel("Minimum Samples Leaf")
    plt.ylabel("Score")

    lw = 2
    plt.semilogx(
        samples_leaf_range,
        train_scores_mean,
        label="Training score",
        color="darkorange",
        lw=lw,
    )
    plt.fill_between(
        samples_leaf_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2,
        color="darkorange",
        lw=lw,
    )
    plt.semilogx(
        samples_leaf_range,
        test_scores_mean,
        label="Cross-validation score",
        color="navy",
        lw=lw,
    )
    plt.fill_between(
        samples_leaf_range,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.2,
        color="navy",
        lw=lw,
    )
    plt.legend(loc="best")

    plt.savefig("samples_leaf_tree_validation.png")

elif performance_report == True:
    clf = DecisionTreeClassifier(
        random_state=0,
        max_depth=10,
        min_samples_leaf=2,
        min_samples_split=5,
        splitter="best",
    ).fit(X_train, y_train)

    y_predicted = clf.predict(X_test)

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

    report = classification_report(
        y_test, y_predicted, target_names=["Benign", "Malware"]
    )

    print("Classification report:\n", report)

    # Create tree diagram
    tree_structure = export_text(clf, feature_names=get_ct_feature_names(column_trans))
    dot_data = export_graphviz(
        clf,
        out_file="decision_tree.dot",
        feature_names=get_ct_feature_names(column_trans),
        filled=True,
        rounded=True,
        special_characters=True,
    )
    (graph,) = pydot.graph_from_dot_file("decision_tree.dot")
    graph.write_png("decision_tree.png")

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

    plt.show()
    plt.savefig("confusion_matrix.png")
elif plot_learning_curve == True:
    fig, ax = plt.subplots()

    clf = DecisionTreeClassifier(
        random_state=0,
        max_depth=5,
        min_samples_leaf=2,
        min_samples_split=5,
        splitter="best",
    )

    train_sizes, train_scores, test_scores = learning_curve(
        clf,
        X,
        y,
        train_sizes=np.linspace(0.1, 1, 10),
        scoring="accuracy",
        cv=skf,
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    plt.legend(loc="best")

    plt.xlabel("Train size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve Decision Tree")

    plt.show()
    plt.savefig("learning_curve.png")
