# %%
import pickle
from typing import TYPE_CHECKING

import hvplot.polars  # noqa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
)
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.pipeline import Pipeline

if TYPE_CHECKING:
    from typing import Any

    from sklearn.pipeline import Pipeline


def plot_grid_search(grid_search, n_best: int | None = None, output_file=None):
    df_results = pl.DataFrame(
        {
            "params": list(
                map(lambda x: str(tuple(x.values())), grid_search.cv_results_["params"])
            ),
            "mean_test_score": grid_search.cv_results_["mean_test_score"],
            "std_test_score": grid_search.cv_results_["std_test_score"],
        }
    ).sort("mean_test_score", descending=True)

    if n_best is None:
        df_plot = df_results
    else:
        df_plot = df_results.head(n_best)

    plot = df_plot.hvplot.errorbars(
        x="params",
        y="mean_test_score",
        yerr1="std_test_score",
        rot=90,
        height=600,
        grid=True,
    ) * df_plot.hvplot.scatter(x="params", y="mean_test_score")

    if output_file:
        hvplot.save(plot, output_file)
    return plot


def _plot_feature_importance(clf: "Pipeline", n_features: int = 30, output_file=None):
    df_importance = pipeline_mdi_importance(clf).head(n_features)

    plot = df_importance.hvplot.scatter(
        x="feature",
        y="importance",
        rot=90,
        height=600,
        grid=True,
    ) * df_importance.hvplot.errorbars(x="feature", y="importance", yerr1="sem")

    if output_file:
        hvplot.save(plot, output_file)
    return plot


def plot_feature_importance(
    df_importance,
    feature_column="feature",
    score_column="importance",
    err_column: str | None = "std",
    output_file=None,
):
    for c in [feature_column, score_column, err_column]:
        if c not in df_importance and c is not None:
            raise ValueError(f"Column not found: {c!r}")

    scores = df_importance.hvplot.scatter(
        x=feature_column,
        y=score_column,
        rot=90,
        height=600,
        grid=True,
    )

    if err_column is not None:
        errors = df_importance.hvplot.errorbars(
            x=feature_column, y=score_column, yerr1=err_column
        )
        plot = scores * errors
    else:
        plot = scores

    if output_file:
        hvplot.save(plot, output_file)
    return plot


def pipeline_mdi_importance(clf: "Pipeline") -> pl.DataFrame:
    from scipy.stats import sem

    clf_sklearn = clf[-1]
    importance = clf_sklearn.feature_importances_
    names = clf[:-1].get_feature_names_out()
    individual_importances = [
        tree.feature_importances_ for tree in clf_sklearn.estimators_
    ]

    importance_std = np.std(individual_importances, axis=0)
    importance_sem = sem(individual_importances, axis=0)

    importance_qs = {}
    for q in [0.0, 0.1, 0.25, 0.5, 0.75, 0.90, 1.0]:
        importance_qs[f"q{q}"] = np.quantile(individual_importances, q, axis=0)

    return pl.DataFrame(
        {
            "feature": names,
            "importance": importance,
            "std": importance_std,
            "sem": importance_sem,
            **importance_qs,
        }
    ).sort("importance", descending=True)


def pipeline_permutation_importance(
    clf: "Pipeline", X_test, y_test, n_repeats=10, n_jobs=-1, random_state=42
) -> pl.DataFrame:
    result = permutation_importance(
        clf,
        X_test,
        y_test,
        n_repeats=n_repeats,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    names = clf[:-1].get_feature_names_out()
    return pl.DataFrame(
        {
            "feature": names,
            "importance": result.importances_mean,
            "std": result.importances_std,
        },
    ).sort("importance", descending=True)


# def plot_confusion_matrix_and_roc_curve(
#     clf,
#     X_test,
#     y_test,
#     confusion_kwargs: "dict[str, Any] | None" = None,
#     roc_kwargs: "dict[str, Any] | None" = None,
# ):
#     if confusion_kwargs is None:
#         confusion_kwargs = {}
#     if roc_kwargs is None:
#         roc_kwargs = {}

#     print(classification_report(y_test, clf.predict(X_test), digits=3))

#     fig, ax = plt.subplots(figsize=(3, 2))
#     c_kwargs = {"normalize": None, **confusion_kwargs}
#     disp_confusion = ConfusionMatrixDisplay.from_estimator(
#         clf, X_test, y_test, ax=ax, **c_kwargs
#     )

#     if clf[-1].n_classes_ > 2:
#         print(f"Cannot show RocCurveDisplay for n_classes: {clf[-1].n_classes_}")
#         return disp_confusion, None

#     fig2, ax2 = plt.subplots(figsize=(3, 3))
#     disp_roc = RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax2, **roc_kwargs)

#     return disp_confusion, disp_roc


def plot_roc_curve(
    clf,
    X_test,
    y_test,
    output_file=None,
    **kwargs,
):
    if clf[-1].n_classes_ > 2:
        print(f"Cannot show RocCurveDisplay for n_classes: {clf[-1].n_classes_}")
        return

    fig, ax = plt.subplots(figsize=(3, 3))
    disp_roc = RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax, **kwargs)
    if output_file is not None:
        fig.savefig(fname=output_file)
    return disp_roc


def plot_confusion_matrix(
    clf, X_test, y_test, figsize=(3, 2), output_file=None, **kwargs
):
    fig, ax = plt.subplots(figsize=figsize)
    c_kwargs = {"normalize": None, **kwargs}
    disp_confusion = ConfusionMatrixDisplay.from_estimator(
        clf, X_test, y_test, ax=ax, **c_kwargs
    )
    if output_file is not None:
        fig.savefig(fname=output_file, bbox_inches="tight")
    return disp_confusion


def get_pipeline(feature_selector):
    from sklccp.ccp_transformers import PolarsSelectorNew

    pip = Pipeline(
        [
            (
                "selector",
                PolarsSelectorNew(feature_selector).set_output(transform="pandas"),
            ),
            ("imputer", "passthrough"),
            ("scaler", "passthrough"),
            ("reducer", "passthrough"),
            (
                "classifier",
                RandomForestClassifier(
                    class_weight="balanced", max_features="sqrt", oob_score=True
                ),
            ),  # Random forest can handle NaN's
        ]
    )
    print(pip.get_params())
    return pip


def get_parameter_grid():
    from feature_engine.selection import DropCorrelatedFeatures
    from sklearn.decomposition import PCA
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import StandardScaler

    return [
        # PCA dimensionality reduction
        {
            "imputer": [KNNImputer()],
            "scaler": [StandardScaler()],
            "reducer": [PCA()],
            "reducer__n_components": [10, 20, 50, 100, 200, 400],
            # "classifier__class_weight": ["balanced"],
            # "classifier__n_estimators": [50, 100, 150],
        },
        # Drop correlated features
        {
            "reducer": [DropCorrelatedFeatures()],
            "reducer__method": ["pearson", "spearman"],
            "reducer__threshold": [0.5, 0.7, 0.8, 0.9, 0.95, 0.99],
        },
        # Train on full set
        {
            "reducer": ["passthrough"],
            # "classifier__n_estimators": [50, 100, 150],
        },
    ]


def save_pipeline(pip, fn):
    with open(fn, "wb") as f:
        pickle.dump(pip, f)


def load_pipeline(fn):
    with open(fn, "rb") as f:
        return pickle.load(f)


def to_polars(arr: "pd.Series | pd.DataFrame") -> "pl.Series | pl.DataFrame":
    if isinstance(arr, pd.Series):
        return pl.Series(arr)
    elif isinstance(arr, pd.DataFrame):
        return pl.DataFrame(arr)
    else:
        raise ValueError("Only pd.Series or pd.DataFrame allowed as input!")


def train_test_split(
    *arrays,
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle=True,
    stratify=None,
):
    if isinstance(arrays[0], pl.DataFrame):
        pd_arrays = [arr.to_pandas(use_pyarrow_extension_array=True) for arr in arrays]
        out = sklearn_train_test_split(
            *pd_arrays,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )
        return tuple(to_polars(arr) for arr in out)
    else:
        return sklearn_train_test_split(
            *arrays,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )
