# %%
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars.selectors as cs
from polars.type_aliases import SelectorType

from zfish.features.polars_utils import drop_null_columns, id_, log
from zfish.preprocessing.outlier_ranges import (
    INTENSITY_OUTLIERS,
    SEGMENTATION_OUTLIERS,
    Outlier,
    clip_outliers,
    drop_outliers,
)

if TYPE_CHECKING:
    from typing import Literal, Sequence

    import polars as pl


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutlierParams:
    strategy: 'Literal["skip", "drop", "clip"]' = "drop"
    outliers: "Sequence[Outlier]" = tuple(SEGMENTATION_OUTLIERS + INTENSITY_OUTLIERS)
    verbose: bool = True
    plot_results: bool = False


@dataclass(frozen=True)
class NullHandlingParams:
    strategy: 'Literal["skip", "drop_feature_rows", "drop_cols", "drop_all"]' = (
        "drop_all"
    )
    verbose: bool = True
    features: SelectorType = cs.all()


def handle_nulls(
    df,
    strategy: str = "drop_cols",
    verbose: bool = True,
    features: SelectorType = cs.all(),
) -> 'pl.DataFrame':
    if verbose:
        shape_logger = log
    else:
        shape_logger = id_

    features_exp = cs.expand_selector(df, features)

    shape_logger(df, message="before:")
    if strategy == "skip":
        return df.fill_nan(None).pipe(shape_logger, message="after:")
    elif strategy == "drop_cols":
        return (
            df.fill_nan(None)
            .pipe(drop_null_columns, strategy="any", columns=features)
            .pipe(shape_logger, message="after:")
        )
    elif strategy == "drop_rows":
        return df.fill_nan(None).drop_nulls().pipe(shape_logger, message="after:")
    elif strategy == "drop_feature_rows":
        return (
            df.fill_nan(None)
            .drop_nulls(subset=features_exp)
            .pipe(shape_logger, message="after:")
        )
    elif strategy == "drop_all":
        return (
            df.fill_nan(None)
            .pipe(drop_null_columns, strategy="any", columns=features)
            .drop_nulls()
            .pipe(shape_logger, message="after:")
        )
    else:
        raise RuntimeError(f"Unknown null strategy {strategy}")


def handle_outliers(
    df,
    strategy="drop",
    outliers: 'Sequence[Outlier]' = tuple(),
    verbose: bool = True,
    plot_results: bool = False,
) -> 'pl.DataFrame':
    print(outliers)
    if strategy == "skip":
        return df
    elif strategy == "drop":
        return df.pipe(
            drop_outliers,
            outliers=outliers,
            verbose=verbose,
            plot_results=plot_results,
        )
    elif strategy == "clip":
        return df.pipe(
            clip_outliers,
            outliers=outliers,
            verbose=verbose,
            plot_results=plot_results,
        )
    else:
        raise RuntimeError(f"Unknown outlier strategy {strategy}")


# %%
